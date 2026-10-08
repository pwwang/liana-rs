#!/usr/bin/env python3
"""Freeze every number the manuscript cites into `paper/numbers.json`.

The manuscript's text carries no hand-typed measurement: each number is read
out of `bench/results.json` here and paired with the JSON path it came from, so
a reader can trace any figure in the text back to the run that produced it.

    python3 paper/make_numbers.py [bench/results.json] [paper/numbers.json]

Fail-loud contract: a label or field this script needs but the manifest does
not carry aborts the run (exit 2) — a number is never substituted, and nothing
is written.

The one section not taken from `bench/results.json` is `parity`: its PASS counts
and fixture tallies are read out of the three gate scripts, which this script
runs read-only (both cargo gates write only under `target/`) so the frozen
counts are the ones this run produced.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench/results.json"
OUT = REPO / "paper/numbers.json"
GATE_LOGS = REPO / "target/w8gates"

# The manuscript's own name for the manifest, used in every `source` path.
SRC = "bench/results.json"

# MB = 1024 kB — the convention bench/collect_results.py uses for its own
# `rss_mb`; GB = 1000 MB on top of it, which is the convention the manuscript's
# text shares (23,746.9 MB = 23.7 GB, 11,041.6 MB = 11.0 GB, 1,901.8 MB =
# 1.9 GB). Both are restated in the output's `unit_conventions`.
KB_PER_MB = 1024
KB_PER_GB = 1024 * 1000

DATASETS = (10000, 50000, 100000)
N_PERMS = (100, 1000)
ARMS = ("rust", "release", "patched")


class Missing(Exception):
    """A number the manuscript cites is not in the manifest (or has no value)."""


class Manifest:
    """`bench/results.json` addressed by label, with exact JSON paths."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.doc = json.loads(path.read_text())
        self.by_label = {}
        for index, row in enumerate(self.require("measurements", list)):
            self.by_label[row["label"]] = index

    def require(self, key: str, kind):
        node = self.doc
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise Missing(f"{SRC}: no `{key}`")
            node = node[part]
        if not isinstance(node, kind):
            raise Missing(f"{SRC}#{key}: expected {kind.__name__}, found {type(node).__name__}")
        return node

    def row(self, label: str) -> tuple[int, dict]:
        if label not in self.by_label:
            raise Missing(f"{SRC}: no measurement labelled `{label}`")
        index = self.by_label[label]
        return index, self.doc["measurements"][index]

    def num(self, label: str, field: str, unit: str, *, nullable: bool = False) -> dict:
        """One measured number: value + unit + the JSON path it came from.

        With `nullable`, an explicit JSON null is recorded as a not-applicable
        field (the manifest's own distinction) instead of a missing number.
        """
        index, row = self.row(label)
        value = row.get(field)
        if value is None and not nullable:
            raise Missing(f"{SRC}#measurements[{index}]: `{label}` has no `{field}`")
        entry = {"value": value, "unit": unit,
                 "source": f"{SRC}#measurements[{index}].{field}", "label": label}
        if value is None:
            entry["note"] = "null in the manifest — the field does not apply"
        return entry

    def at(self, key: str, unit: str) -> dict:
        """One number from a named (non-measurement) path."""
        node = self.doc
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise Missing(f"{SRC}: no `{key}`")
            node = node[part]
        if node is None:
            raise Missing(f"{SRC}#{key}: has no value")
        return {"value": node, "unit": unit, "source": f"{SRC}#{key}"}


def _paths(node: dict) -> list[str]:
    """Every manifest path behind `node` — one for a measured number, all the
    leaves for a derived one (so a ratio of a ratio still traces to the run)."""
    if "source" in node:
        return [node["source"]]
    out = []
    for path in node.get("derived_from", []):
        out.extend(_paths({"source": path}) if isinstance(path, str) else _paths(path))
    return out


def derived(value, unit: str, formula: str, *sources: dict) -> dict:
    """A number computed from frozen ones, with the paths of its inputs."""
    for source in sources:
        if "value" not in source:
            raise Missing(f"derived({formula}): an input carries no value")
    return {"value": value, "unit": unit, "formula": formula,
            "derived_from": [p for source in sources for p in _paths(source)]}


def ratio(a: dict, b: dict, formula: str) -> dict:
    return derived(a["value"] / b["value"], "×", formula, a, b)


def mb(kb: dict) -> dict:
    return derived(kb["value"] / KB_PER_MB, "MB", "kB / 1024", kb)


def gb(kb: dict) -> dict:
    return derived(kb["value"] / KB_PER_GB, "GB", "kB / 1024 / 1000", kb)


def gib(kb: dict) -> dict:
    return derived(kb["value"] / KB_PER_MB / 1024, "GiB", "kB / 1024 / 1024", kb)


def ms(seconds: dict) -> dict:
    return derived(seconds["value"] * 1000, "ms", "s × 1000", seconds)


def pct(a: dict, b: dict, formula: str) -> dict:
    return derived((a["value"] / b["value"]) * 100, "%", formula, a, b)


# --------------------------------------------------------------------------
# sections

def matrix(m: Manifest) -> dict:
    """T1 — `rank_aggregate` end to end, three arms, 10k/50k/100k × p100/p1000."""
    out = {}
    rust_walls = []
    for n_obs in DATASETS:
        for n_perms in N_PERMS:
            key = f"{n_obs // 1000}k/p{n_perms}"
            arms = {}
            for arm in ARMS:
                label = f"t1_ra_{n_obs // 1000}k_p{n_perms}_{arm}"
                wall = m.num(label, "wall_s", "s")
                rss = m.num(label, "rss_kb", "kB")
                arms[arm] = {
                    "wall_s": wall,
                    "elapsed_s": m.num(label, "elapsed_s", "s"),
                    "rss_kb": rss,
                    "rss_mb": mb(rss),
                    "rss_gb": gb(rss),
                    "rows": m.num(label, "rows", "rows"),
                    "threads": m.num(label, "threads", "threads"),
                    # null for the engine, which has no Numba kernels (the
                    # manuscript's "4 jobs, NumPy kernels at 32 threads").
                    "numba_threads": m.num(label, "numba_threads", "threads",
                                           nullable=True),
                }
            rust = arms["rust"]
            for arm in ("release", "patched"):
                arms[f"{arm}_over_rust"] = {
                    "wall_ratio": ratio(arms[arm]["wall_s"], rust["wall_s"],
                                        f"{arm}.wall_s / rust.wall_s"),
                    "rss_ratio": ratio(arms[arm]["rss_kb"], rust["rss_kb"],
                                       f"{arm}.rss_kb / rust.rss_kb"),
                    "rss_less_than": derived(
                        rust["rss_kb"]["value"] / arms[arm]["rss_kb"]["value"], "×",
                        f"rust.rss_kb / {arm}.rss_kb", rust["rss_kb"], arms[arm]["rss_kb"]),
                }
            # The table's own column: peak RSS vs the release arm (its 1.0× row
            # included), so "5.8× less" / "26.4× less" are frozen, not re-derived.
            release = arms["release"]
            for arm in ARMS:
                arms[f"{arm}_vs_release"] = {
                    "rss_ratio": ratio(arms[arm]["rss_kb"], release["rss_kb"],
                                       f"{arm}.rss_kb / release.rss_kb"),
                    "rss_less_than": derived(
                        release["rss_kb"]["value"] / arms[arm]["rss_kb"]["value"], "×",
                        f"release.rss_kb / {arm}.rss_kb",
                        release["rss_kb"], arms[arm]["rss_kb"]),
                }
            out[key] = {"n_obs": n_obs, "n_perms": n_perms, "n_lrs": 2000, "arms": arms}
            rust_walls.append(rust["wall_s"])
    out["rust_wall_range"] = {"min": min(rust_walls, key=lambda w: w["value"]),
                              "max": max(rust_walls, key=lambda w: w["value"])}
    return out


def wall_speedups(grid: dict) -> dict:
    """The manuscript's two speed ranges: wall vs the engine at p1000, per size."""
    out = {}
    for arm in ("release", "patched"):
        ratios = {f"{n // 1000}k": grid[f"{n // 1000}k/p1000"]["arms"]
                  [f"{arm}_over_rust"]["wall_ratio"] for n in DATASETS}
        out[arm] = {
            **ratios,
            "range": {"min": min(ratios.values(), key=lambda r: r["value"]),
                      "max": max(ratios.values(), key=lambda r: r["value"])},
        }
    return out


def cli(m: Manifest) -> dict:
    """T1's cross-check through the shipped CLI binary."""
    rss = m.num("t1_ra_50k_p1000_cli", "rss_kb", "kB")
    return {
        "elapsed_s": m.num("t1_ra_50k_p1000_cli", "elapsed_s", "s"),
        "rss_kb": rss, "rss_mb": mb(rss),
        "rows": m.num("t1_ra_50k_p1000_cli", "rows", "rows"),
        "n_obs": m.num("t1_ra_50k_p1000_cli", "n_obs", "cells"),
        "n_perms": m.num("t1_ra_50k_p1000_cli", "n_perms", "perms"),
        "threads": m.num("t1_ra_50k_p1000_cli", "threads", "threads"),
    }


def law_grid(m: Manifest) -> dict:
    """T2 — the grid the law is read off: peak RSS vs n_perms × n_lrs at 50k."""
    out = {}
    flat = []
    for n_perms in (10, 100, 1000):
        for n_lrs in (200, 1000, 2000):
            key = f"p{n_perms}/lrs{n_lrs}"
            arms = {}
            for arm in ARMS:
                label = f"t2_ra_50k_p{n_perms}_lrs{n_lrs}_{arm}"
                rss = m.num(label, "rss_kb", "kB")
                arms[arm] = {"rss_kb": rss, "rss_mb": mb(rss), "rss_gb": gb(rss),
                             "wall_s": m.num(label, "wall_s", "s"),
                             "rows": m.num(label, "rows", "rows")}
                if arm == "rust":
                    flat.append((n_perms, n_lrs, rss))
            out[key] = {"n_perms": n_perms, "n_lrs": n_lrs, "n_obs": 50000, "arms": arms}

    # the manuscript's flatness claim: per LR column, the spread over a tenfold
    # change in n_perms; and the whole grid's span
    per_column = {}
    for n_lrs in (200, 1000, 2000):
        col = [rss for _, lrs, rss in flat if lrs == n_lrs]
        lo = min(col, key=lambda r: r["value"])
        hi = max(col, key=lambda r: r["value"])
        spread_kb = derived(hi["value"] - lo["value"], "kB", "max - min", hi, lo)
        per_column[f"lrs{n_lrs}"] = {
            "min_rss_kb": lo, "max_rss_kb": hi, "spread_kb": spread_kb,
            "spread_pct": pct(spread_kb, lo, "(max - min) / min × 100"),
        }
    all_kb = [rss for _, _, rss in flat]
    out["rust_flatness"] = {
        "per_column": per_column,
        "grid_min_rss_kb": min(all_kb, key=lambda r: r["value"]),
        "grid_max_rss_kb": max(all_kb, key=lambda r: r["value"]),
        "grid_min_rss_mb": mb(min(all_kb, key=lambda r: r["value"])),
        "grid_max_rss_mb": mb(max(all_kb, key=lambda r: r["value"])),
    }
    out["anchors_10k_release"] = [
        {"label": a["label"], "n_perms": a["n_perms"], "n_lrs": a["n_lrs"],
         "measured_kb": {"value": a["measured_kb"], "unit": "kB",
                         "source": f"{SRC}#law.anchors[{i}].measured_kb"},
         "law_pred_kb": {"value": a["law_pred_kb"], "unit": "kB",
                         "source": f"{SRC}#law.anchors[{i}].law_pred_kb"},
         "residual_kb": {"value": a["residual_kb"], "unit": "kB",
                         "source": f"{SRC}#law.anchors[{i}].residual_kb"}}
        for i, a in enumerate(m.require("law.anchors", list))
    ]
    return out


def law(m: Manifest) -> dict:
    """The recorded law and the 50k re-fit, plus the predictions the text quotes."""
    slope = m.at("law.fit_50k.recorded_slope", "KB/(perm × LR)")
    intercept_mb = m.at("law.fit_50k.recorded_intercept_mb", "MB")
    refit_slope = m.at("law.fit_50k.slope_kb_per_perm_lr", "KB/(perm × LR)")
    refit_intercept_mb = m.at("law.fit_50k.intercept_mb", "MB")

    def predict(n_perms: int, n_lrs: int) -> dict:
        kb = intercept_mb["value"] * KB_PER_MB + slope["value"] * n_perms * n_lrs
        return derived(kb, "kB", "intercept_mb × 1024 + slope × n_perms × n_lrs",
                       intercept_mb, slope)

    defaults = predict(1000, 2000)
    consensus = predict(1000, 4620)
    return {
        "statement": {"value": m.require("law.statement", str), "unit": "text",
                      "source": f"{SRC}#law.statement"},
        "source_note": {"value": m.require("law.source", str), "unit": "text",
                        "source": f"{SRC}#law.source"},
        "recorded_slope": slope,
        "recorded_intercept_mb": intercept_mb,
        "refit_50k": {
            "slope_kb_per_perm_lr": refit_slope,
            "intercept_kb": m.at("law.fit_50k.intercept_kb", "kB"),
            "intercept_mb": refit_intercept_mb,
            "n_points": m.at("law.fit_50k.n_points", "points"),
        },
        "predict_defaults_1000x2000": {
            "kb": defaults, "mb": derived(defaults["value"] / KB_PER_MB, "MB",
                                          "kB / 1024", defaults),
            "gb": derived(defaults["value"] / KB_PER_GB, "GB",
                          "kB / 1024 / 1000", defaults)},
        "predict_consensus_1000x4620": {
            "kb": consensus, "mb": derived(consensus["value"] / KB_PER_MB, "MB",
                                           "kB / 1024", consensus),
            "gb": derived(consensus["value"] / KB_PER_GB, "GB",
                          "kB / 1024 / 1000", consensus),
            "gib": gib(consensus)},
    }


def threads(m: Manifest) -> dict:
    """T3 — thread scaling of the engine, with the Python baselines that exist."""
    out = {"rust": {}}
    for n_threads in (1, 4, 8, 32):
        label = f"t3_ra_50k_p1000_t{n_threads}_rust"
        wall = m.num(label, "wall_s", "s")
        rss = m.num(label, "rss_kb", "kB")
        out["rust"][str(n_threads)] = {
            "wall_s": wall, "rss_kb": rss, "rss_mb": mb(rss),
            "n_obs": m.num(label, "n_obs", "cells"),
            "n_perms": m.num(label, "n_perms", "perms"),
            "n_lrs": m.num(label, "n_lrs", "LR pairs"),
        }
    base = out["rust"]["1"]["wall_s"]
    for n_threads in (1, 4, 8, 32):
        wall = out["rust"][str(n_threads)]["wall_s"]
        out["rust"][str(n_threads)]["speedup_vs_1"] = ratio(
            base, wall, "t1.wall_s / tN.wall_s")
    return out


def python_threads(m: Manifest) -> dict:
    """The Python arms' thread points — the only ones the manifest carries.

    j1/j16 are the recorded 50k × p1000 references; j4 is the suite's own row
    for the same configuration. No j8 or j32 run exists for either Python arm.
    """
    out = {}
    for i, ref in enumerate(m.require("recorded_references", list)):
        desc = ref.get("description", "")
        if "thread-scaling reference" not in (ref.get("note") or ""):
            continue
        j = re.search(r"j(\d+)$", desc)
        if not j:
            raise Missing(f"{SRC}#recorded_references: `{desc}` names no j<n_jobs>")
        out[f"release_j{j.group(1)}"] = {
            "wall_s": {"value": ref["wall_s"], "unit": "s",
                       "source": f"{SRC}#recorded_references[{i}].wall_s"},
            "rss_kb": {"value": ref["rss_kb"], "unit": "kB",
                       "source": f"{SRC}#recorded_references[{i}].rss_kb"},
            "log": ref["log"],
        }
    for arm in ("release", "patched"):
        label = f"t1_ra_50k_p1000_{arm}"
        out[f"{arm}_j4"] = {
            "wall_s": m.num(label, "wall_s", "s"),
            "rss_kb": m.num(label, "rss_kb", "kB"),
            "source_note": "suite row, same configuration (50k x p1000, n_jobs=4)",
        }
    return out


def startup(m: Manifest) -> dict:
    """T4 — startup and first run."""
    hr = m.num("t4_lianars_version_hr", "wall_s", "s")
    rss = m.num("t4_lianars_version_r1", "rss_kb", "kB")
    imports = []
    for rep in range(1, 6):
        label = f"t4_python_import_r{rep}"
        imports.append(m.num(label, "elapsed_s", "s"))
    best = min(imports, key=lambda r: r["value"])
    median = sorted(imports, key=lambda r: r["value"])[len(imports) // 2]
    first = "t4_firstrun_connectome_1k"
    return {
        "version_wall_s": hr,
        "version_wall_ms": ms(hr),
        "version_rss_kb": rss,
        "version_rss_mb": mb(rss),
        "first_run": {
            "elapsed_s": m.num(first, "elapsed_s", "s"),
            "rss_kb": m.num(first, "rss_kb", "kB"),
            "rss_mb": mb(m.num(first, "rss_kb", "kB")),
            "rows": m.num(first, "rows", "rows"),
            "n_obs": m.num(first, "n_obs", "cells"),
            "n_lrs": m.num(first, "n_lrs", "LR pairs"),
        },
        "python_import": {
            "reps": imports,
            "n_reps": {"value": len(imports), "unit": "runs",
                       "source": f"{SRC}#measurements: t4_python_import_r1..r5"},
            "best_s": best, "median_s": median,
            "rss_kb": m.num("t4_python_import_r1", "rss_kb", "kB"),
            "rss_mb": mb(m.num("t4_python_import_r1", "rss_kb", "kB")),
        },
        "recorded_import_baseline": _startup_log(m),
    }


def _startup_log(m: Manifest) -> dict:
    refs = m.require("recorded_references", list)
    for i, ref in enumerate(refs):
        if ref.get("json_runs"):
            for run in ref["json_runs"]:
                if run["label"] == "import_liana":
                    return {"best_s": {"value": run["best_s"], "unit": "s",
                                       "source": f"{SRC}#recorded_references[{i}].json_runs"},
                            "log": ref["log"]}
    raise Missing(f"{SRC}#recorded_references: no `import_liana` startup run")


def t5(m: Manifest) -> dict:
    """T5 — the 4,620-LR resource (the consensus resource's size)."""
    out = {"n_lrs": 4620, "n_obs": 50000, "n_perms": 1000, "arms": {}}
    for arm in ARMS:
        label = f"t5_ra_50k_p1000_lrs4620_{arm}"
        rss = m.num(label, "rss_kb", "kB")
        out["arms"][arm] = {
            "wall_s": m.num(label, "wall_s", "s"),
            "elapsed_s": m.num(label, "elapsed_s", "s"),
            "rss_kb": rss, "rss_mb": mb(rss), "rss_gb": gb(rss),
            "rows": m.num(label, "rows", "rows"),
        }
    out["release_over_rust_rss"] = ratio(out["arms"]["release"]["rss_kb"],
                                         out["arms"]["rust"]["rss_kb"],
                                         "release.rss_kb / rust.rss_kb")
    out["release_over_rust_wall"] = ratio(out["arms"]["release"]["wall_s"],
                                          out["arms"]["rust"]["wall_s"],
                                          "release.wall_s / rust.wall_s")
    return out


def per_method(m: Manifest) -> dict:
    """T6 — CellPhoneDB and CellChat alone, 50k × 2,000 LRs × 1,000 perms."""
    out = {"n_obs": 50000, "n_lrs": 2000, "n_perms": 1000, "threads": 4,
           "methods": {}}
    for method in ("cellphonedb", "cellchat"):
        arms = {}
        for arm in ARMS:
            label = f"t6_{method}_50k_p1000_{arm}"
            rss = m.num(label, "rss_kb", "kB")
            arms[arm] = {
                "wall_s": m.num(label, "wall_s", "s"),
                "elapsed_s": m.num(label, "elapsed_s", "s"),
                "rss_kb": rss, "rss_mb": mb(rss), "rss_gb": gb(rss),
                "rows": m.num(label, "rows", "rows"),
            }
        arms["release_over_rust_rss"] = ratio(
            arms["release"]["rss_kb"], arms["rust"]["rss_kb"],
            "release.rss_kb / rust.rss_kb")
        arms["release_over_rust_wall"] = ratio(
            arms["release"]["wall_s"], arms["rust"]["wall_s"],
            "release.wall_s / rust.wall_s")
        out["methods"][method] = arms
    return out


def _gate_tally(text: str, log: str) -> dict:
    """The fixture tallies a gate log carries: runs, rows, compared columns.

    Columns are summed per permutation count over the methods of that count
    (`numeric` and `object` compared the same way — exact, rtol 0). Every field
    is read from the log; a log this cannot parse aborts rather than tallying
    what is there.
    """
    runs = text.split("parity diff:")[1:]
    if not runs:
        raise Missing(f"{log}: no `parity diff:` runs to tally")
    counts: dict[str, dict] = {}
    rows_expected = rows_matched = differing = nan_pattern = 0
    for run in runs:
        name = re.search(r"expected=\S*__p(\d+)\.csv", run)
        if not name:
            raise Missing(f"{log}: a parity run names no `__p<N>.csv` expected file")
        rows = re.search(r"^rows: (\d+) expected, (\d+) actual, (\d+) matched$", run, re.M)
        if not rows:
            raise Missing(f"{log}: a `p{name.group(1)}` run has no `rows:` line")
        rows_expected += int(rows.group(1))
        rows_matched += int(rows.group(3))
        columns = counts.setdefault(name.group(1), {"methods": 0, "numeric": 0, "object": 0})
        columns["methods"] += 1
        for kind in re.findall(r"^  \S+: (numeric|object) exact=", run, re.M):
            columns[kind] += 1
        differing += sum(int(n) for n in re.findall(r"differs=(\d+)", run))
        nan_pattern += sum(int(n) for n in re.findall(r"nan-pattern=(\d+)", run))

    def counted(value: int, unit: str) -> dict:
        return {"value": value, "unit": unit, "source": f"{log}"}

    out = {"runs": counted(len(runs), "runs"),
           "rows_expected": counted(rows_expected, "rows"),
           "rows_matched": counted(rows_matched, "rows"),
           "differing_cells": counted(differing, "cells"),
           "nan_pattern_mismatches": counted(nan_pattern, "cells"),
           "columns": {}}
    for count, columns in sorted(counts.items()):
        out["columns"][f"p{count}"] = {
            "methods": counted(columns["methods"], "methods"),
            "numeric": counted(columns["numeric"], "columns"),
            "object": counted(columns["object"], "columns"),
            "total": counted(columns["numeric"] + columns["object"], "columns"),
        }
    return out


def parity() -> dict:
    """The three gates' PASS counts and fixture tallies, from a read-only run.

    One PASS per gate run: 18 for each cargo gate (9 methods × p100/p1000), 19
    for the Python module (18 runs + the no-pandas fallback). A gate that does
    not finish, or does not report its full count, is recorded as it stands and
    fails the script — never rounded up.
    """
    GATE_LOGS.mkdir(parents=True, exist_ok=True)
    gates = [
        ("pipe", ["bash", "scripts/check_pipe_parity.sh"], 18,
         "crates/liana-core/tests/{pipe,cellchat}_parity.rs + parity_diff.py, rtol 0"),
        ("cli", ["bash", "scripts/check_cli_parity.sh"], 18,
         "`liana-rs run` on the fixture + toy resource, parity_diff.py, rtol 0"),
        ("py", ["target/py-venv/bin/python", "scripts/check_py_parity.py"], 19,
         "the PyO3 module: 18 runs + the no-pandas fallback"),
    ]
    out = {}
    failures = []
    for name, cmd, expected, what in gates:
        log = GATE_LOGS / f"{name}.log"
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        log.write_text(proc.stdout + proc.stderr)
        text = proc.stdout
        if name == "py":
            found = re.search(r"py gate: (\d+)/(\d+) PASS", text)
            passed = int(found.group(1)) if found else None
            total = int(found.group(2)) if found else expected
        else:
            passed = len(re.findall(r"GATE: PASS", text))
            total = expected
        out[name] = {
            "script": cmd[-1], "what": what,
            "command": " ".join(cmd), "exit_code": proc.returncode,
            "passed": passed, "total": total, "expected": expected,
            "log": str(log.relative_to(REPO)),
            "pass": passed == expected and proc.returncode == 0,
            "tally": _gate_tally(text, str(log.relative_to(REPO))),
        }
        if not out[name]["pass"]:
            failures.append(f"{name}: {passed}/{total} PASS, exit {proc.returncode}")
    out["all_pass"] = not failures
    if failures:
        out["failures"] = failures
    return out


def checks(m: Manifest) -> dict:
    """The suite's own agreements, as recorded in the manifest."""
    rows = m.require("checks", list)
    failed = [c for c in rows if not c.get("pass")]
    return {
        "n_checks": {"value": len(rows), "unit": "checks", "source": f"{SRC}#checks"},
        "n_passed": {"value": len(rows) - len(failed), "unit": "checks",
                     "source": f"{SRC}#checks"},
        "all_pass": not failed,
        "row_count": [{"config": c["config"], "arms": c["arms"],
                       "source": f"{SRC}#checks[{i}]"}
                      for i, c in enumerate(rows) if "row count" in c["check"]],
        "rss_flat_in_n_perms": [
            {"check": c["check"], "rss_kb_by_n_perms": c["rss_kb_by_n_perms"],
             "spread_kb": c["spread_kb"], "source": f"{SRC}#checks[{i}]"}
            for i, c in enumerate(rows) if "flat across n_perms" in c["check"]],
    }


def versions(m: Manifest) -> dict:
    """The toolchain and box the numbers were measured on."""
    out = {}
    for key in ("rust.cargo", "rust.rustc", "git_head", "git_dirty", "python", "liana",
                "liana_commit", "numpy", "pandas", "numba", "scipy", "anndata", "scanpy",
                "cpu", "cores", "mem_total_kb", "kernel"):
        unit = {"cores": "threads", "mem_total_kb": "kB", "git_dirty": "bool"}.get(key, "version")
        out[key] = m.at(f"versions.{key}", unit)
    mem = out["mem_total_kb"]
    out["mem_total_gib"] = gib(mem)
    return out


def box(m: Manifest) -> dict:
    out = {key: m.at(f"box.{key}", {"cores": "threads", "mem_total_kb": "kB"}.get(key, "text"))
           for key in ("cpu", "cores", "mem_total_kb", "kernel")}
    mem = out["mem_total_kb"]
    out["mem_total_gib"] = gib(mem)
    return out


def recorded_references(m: Manifest) -> list:
    """The baseline rows re-read from the p0a logs (read-only) by the suite."""
    out = []
    for i, ref in enumerate(m.require("recorded_references", list)):
        if ref.get("missing"):
            raise Missing(f"{SRC}#recorded_references[{i}]: {ref['log']} is missing")
        entry = {"log": ref["log"], "description": ref["description"],
                 "note": ref.get("note", "")}
        if ref.get("wall_s") is not None:
            entry["wall_s"] = {"value": ref["wall_s"], "unit": "s",
                               "source": f"{SRC}#recorded_references[{i}].wall_s"}
        if ref.get("rss_kb") is not None:
            rss = {"value": ref["rss_kb"], "unit": "kB",
                   "source": f"{SRC}#recorded_references[{i}].rss_kb"}
            entry["rss_kb"] = rss
            entry["rss_mb"] = mb(rss)
        out.append(entry)
    return out


def patched_wall_spread(m: Manifest) -> dict:
    """The `patched` arm's wall across every python-side 50k × p1000 j4 run.

    The manuscript quotes this as a range (its wall is the box-state-sensitive
    cell); the peak RSS is stable across the same runs.
    """
    walls, rss = [], []
    for i, ref in enumerate(m.require("recorded_references", list)):
        if "patched" in ref["description"] and "50k" in ref["description"] \
                and "p1000" in ref["description"] and "j4" in ref["description"]:
            walls.append({"value": ref["wall_s"], "unit": "s",
                          "source": f"{SRC}#recorded_references[{i}].wall_s"})
            rss.append({"value": ref["rss_kb"], "unit": "kB",
                        "source": f"{SRC}#recorded_references[{i}].rss_kb"})
    suite = m.num("t1_ra_50k_p1000_patched", "wall_s", "s")
    suite_rss = m.num("t1_ra_50k_p1000_patched", "rss_kb", "kB")
    walls.append(suite)
    rss.append(suite_rss)
    if not walls:
        raise Missing(f"{SRC}#recorded_references: no patched 50k x p1000 j4 runs")
    return {
        "wall_s_min": min(walls, key=lambda w: w["value"]),
        "wall_s_max": max(walls, key=lambda w: w["value"]),
        "rss_kb_min": min(rss, key=lambda r: r["value"]),
        "rss_kb_max": max(rss, key=lambda r: r["value"]),
        "runs": walls,
    }


def build(results: pathlib.Path) -> dict:
    m = Manifest(results)
    grid = law_grid(m)
    matrix_rows = matrix(m)
    doc = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "unit_conventions": {
            "MB": "1024 kB (bench/collect_results.py's `rss_mb`)",
            "GB": "1000 MB, on top of that MB — the convention the manuscript's "
                  "text uses (23,746.9 MB = 23.7 GB)",
            "GiB": "1024 MB (binary, as bench/RESULTS.md's GiB figures)",
            "wall_s": "in-process wall (see bench/RESULTS.md `Wall bases`)",
            "elapsed_s": "/usr/bin/time -v whole-process wall",
            "rss_kb": "peak RSS, `time -v` Maximum resident set size",
        },
        "source": {
            "results_json": str(results.relative_to(REPO)),
            "results_generated_at": m.at("generated_at", "timestamp"),
            "suite_script": m.at("suite.script", "path"),
            "suite_seed": m.at("suite.seed", "seed"),
            "suite_arms": m.at("suite.arms", "map"),
            "recorded_references_note": {
                "value": "walls/rss re-read from /home/pwwang/p0a/logs by bench/collect_results.py",
                "unit": "text", "source": f"{SRC}#recorded_references"},
        },
        "box": box(m),
        "versions": versions(m),
        "wall_rss_matrix": matrix_rows,
        "wall_speedups": wall_speedups(matrix_rows),
        "cli_cross_check": cli(m),
        "law": law(m),
        "law_grid": grid,
        "thread_scaling": {"rust": threads(m)["rust"], "python_references": python_threads(m)},
        "startup": startup(m),
        "t5_consensus_4620": t5(m),
        "per_method": per_method(m),
        "parity": parity(),
        "checks": checks(m),
        "patched_wall_spread": patched_wall_spread(m),
        "recorded_references": recorded_references(m),
    }
    if not doc["parity"]["all_pass"]:
        print(f"GATE FAILURES: {doc['parity']['failures']}", file=sys.stderr)
    if not doc["checks"]["all_pass"]:
        print("SUITE CHECK FAILURES in bench/results.json", file=sys.stderr)
    return doc


def main(argv: list[str]) -> int:
    results = pathlib.Path(argv[1]) if len(argv) > 1 else RESULTS
    out = pathlib.Path(argv[2]) if len(argv) > 2 else OUT
    try:
        doc = build(results)
    except Missing as exc:
        print(f"make_numbers: MISSING NUMBER — {exc}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    n = sum(1 for _ in _leaves(doc))
    print(f"{out}: {n} frozen numbers from {results}")
    print(f"parity: " + ", ".join(
        f"{k} {v['passed']}/{v['total']}" for k, v in doc["parity"].items()
        if isinstance(v, dict) and "passed" in v))
    return 0 if (doc["parity"]["all_pass"] and doc["checks"]["all_pass"]) else 1


def _leaves(node, key=""):
    if isinstance(node, dict):
        if "value" in node and "unit" in node:
            yield key, node
            return
        for k, v in node.items():
            yield from _leaves(v, f"{key}.{k}" if key else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{key}[{i}]")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
