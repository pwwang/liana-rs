#!/usr/bin/env python3
"""Keyed parity diff between an oracle CSV (expected) and a Rust CSV (actual).

Rows are joined on ``--key`` columns; every other column is compared as either
numeric (all values parse as floats) or object (string equality). Numeric cells
count as exact, within ``--rtol`` of the expected value, or differing; cells
that are NaN on exactly one side are counted separately as NaN-pattern
mismatches. Exit status is the gate: 0 iff there are no one-sided rows, no
differing cells, and no NaN-pattern mismatches; 1 otherwise; 2 for usage or
schema errors. ``--taxonomy-out`` writes one row per mismatching row x column
with the delta (actual - expected) for triage.

Stdlib only, so it runs under any interpreter. Usage:

    python scripts/parity_diff.py --expected testdata/expected/x.csv --actual out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

DEFAULT_KEY = "source,target,ligand_complex,receptor_complex"

MISMATCH_KINDS = ("differs", "nan_pattern")


class ParityError(Exception):
    """Usage / schema error — exit 2, not a gate failure."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expected", required=True, help="oracle CSV (reference values)")
    parser.add_argument("--actual", required=True, help="Rust CSV to check")
    parser.add_argument("--key", default=DEFAULT_KEY, help=f"comma-separated key columns (default: {DEFAULT_KEY})")
    parser.add_argument("--rtol", type=float, default=1e-12, help="numeric tolerance, relative to expected (default: 1e-12)")
    parser.add_argument("--taxonomy-out", help="write a per-mismatch CSV here")
    return parser.parse_args(argv)


def load_table(path: str, key: list[str], side: str) -> tuple[list[str], list[dict], dict[tuple, dict]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in key if c not in columns]
    if missing:
        raise ParityError(f"{side} {path}: key column(s) absent from header: {', '.join(missing)}")
    index: dict[tuple, dict] = {}
    for row in rows:
        row_key = tuple(row[c] for c in key)
        if row_key in index:
            raise ParityError(f"{side} {path}: duplicate key {row_key}")
        index[row_key] = row
    return columns, rows, index


def _is_missing(raw: str | None) -> bool:
    """CSV cells that mean NaN: absent, empty, or a literal nan."""
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() == "nan"


def _as_float(raw: str | None) -> float | None:
    return None if _is_missing(raw) else float(raw)


def _display(raw: str | None) -> str:
    return "nan" if _is_missing(raw) else raw  # type: ignore[return-value]


def numeric_columns(columns: list[str], *tables: list[dict]) -> dict[str, bool]:
    """A column is numeric iff every non-missing value parses as a float (and one does)."""
    verdict: dict[str, bool] = {}
    for col in columns:
        seen_value = False
        is_numeric = True
        for rows in tables:
            for row in rows:
                raw = row.get(col)
                if _is_missing(raw):
                    continue
                seen_value = True
                try:
                    float(raw)  # type: ignore[arg-type]
                except ValueError:
                    is_numeric = False
                    break
            if not is_numeric:
                break
        verdict[col] = is_numeric and seen_value
    return verdict


def classify(expected_raw, actual_raw, is_numeric: bool, rtol: float) -> str:
    """One of: exact, within_tol, differs, nan_pattern."""
    e_missing, a_missing = _is_missing(expected_raw), _is_missing(actual_raw)
    if e_missing or a_missing:
        return "exact" if e_missing and a_missing else "nan_pattern"
    if not is_numeric:
        return "exact" if expected_raw == actual_raw else "differs"
    e, a = float(expected_raw), float(actual_raw)
    if e == a:
        return "exact"
    if math.isfinite(e) and math.isfinite(a) and abs(a - e) <= rtol * abs(e):
        return "within_tol"
    return "differs"


def diff(expected: str, actual: str, key: list[str], rtol: float) -> tuple[list[str], dict, dict[tuple, dict], dict]:
    exp_columns, exp_rows, exp_index = load_table(expected, key, "expected")
    act_columns, act_rows, act_index = load_table(actual, key, "actual")
    if exp_columns != act_columns:
        raise ParityError(f"header mismatch:\n  expected: {exp_columns}\n  actual:   {act_columns}")

    index = {
        "only_in_expected": sorted(k for k in exp_index if k not in act_index),
        "only_in_actual": sorted(k for k in act_index if k not in exp_index),
        "matched": sorted(k for k in exp_index if k in act_index),
    }

    numeric = numeric_columns(exp_columns, exp_rows, act_rows)
    stats: dict[str, dict[str, int]] = {}
    taxonomy: list[dict] = []
    for col in exp_columns:
        counts = {"exact": 0, "within_tol": 0, "differs": 0, "nan_pattern": 0}
        for row_key in index["matched"]:
            exp_raw = exp_index[row_key].get(col)
            act_raw = act_index[row_key].get(col)
            kind = classify(exp_raw, act_raw, numeric[col], rtol)
            counts[kind] += 1
            if kind in MISMATCH_KINDS:
                taxonomy.append(
                    {
                        **dict(zip(key, row_key)),
                        "column": col,
                        "kind": kind,
                        "expected": _display(exp_raw),
                        "actual": _display(act_raw),
                        "delta": _delta(exp_raw, act_raw, kind, numeric[col]),
                    }
                )
        stats[col] = counts

    for kind in ("only_in_expected", "only_in_actual"):
        for row_key in index[kind]:
            taxonomy.append(
                {**dict(zip(key, row_key)), "column": "", "kind": kind, "expected": "", "actual": "", "delta": ""}
            )
    return exp_columns, index, stats, {"numeric": numeric, "taxonomy": taxonomy}


def _delta(expected_raw, actual_raw, kind: str, is_numeric: bool) -> str:
    if kind != "differs" or not is_numeric:
        return ""
    return repr(float(actual_raw) - float(expected_raw))


def gate_failures(result: dict) -> dict[str, int]:
    stats = result["stats"]
    return {
        "only-in-expected": len(result["index"]["only_in_expected"]),
        "only-in-actual": len(result["index"]["only_in_actual"]),
        "differing cells": sum(c["differs"] for c in stats.values()),
        "nan-pattern mismatches": sum(c["nan_pattern"] for c in stats.values()),
    }


def report(expected: str, actual: str, key: list[str], rtol: float, result: dict) -> None:
    print(f"parity diff: expected={expected}")
    print(f"             actual  ={actual}")
    print(f"key: {', '.join(key)}   rtol: {rtol:g}")
    matched = len(result["index"]["matched"])
    print(
        f"rows: {matched + len(result['index']['only_in_expected'])} expected, "
        f"{matched + len(result['index']['only_in_actual'])} actual, {matched} matched"
    )
    print(f"only-in-expected: {len(result['index']['only_in_expected'])}")
    print(f"only-in-actual: {len(result['index']['only_in_actual'])}")
    for col in result["columns"]:
        counts = result["stats"][col]
        kind = "numeric" if result["numeric"][col] else "object"
        print(
            f"  {col}: {kind} exact={counts['exact']} within-tol={counts['within_tol']} "
            f"differs={counts['differs']} nan-pattern={counts['nan_pattern']}"
        )
    failures = gate_failures(result)
    verdict = "PASS" if not any(failures.values()) else "FAIL"
    detail = ", ".join(f"{n} {label}" for label, n in failures.items())
    print(f"GATE: {verdict} ({detail})")


def write_taxonomy(path: str, key: list[str], taxonomy: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*key, "column", "kind", "expected", "actual", "delta"])
        writer.writeheader()
        writer.writerows(taxonomy)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    key = [c.strip() for c in args.key.split(",") if c.strip()]
    if not key:
        print("parity: ERROR: --key must name at least one column", file=sys.stderr)
        return 2
    try:
        columns, index, stats, extra = diff(args.expected, args.actual, key, args.rtol)
    except ParityError as exc:
        print(f"parity: ERROR: {exc}", file=sys.stderr)
        return 2
    result = {"columns": columns, "index": index, "stats": stats, **extra}
    report(args.expected, args.actual, key, args.rtol, result)
    if args.taxonomy_out:
        write_taxonomy(args.taxonomy_out, key, result["taxonomy"])
        print(f"taxonomy: {args.taxonomy_out} ({len(result['taxonomy'])} row(s))")
    return 1 if any(gate_failures(result).values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
