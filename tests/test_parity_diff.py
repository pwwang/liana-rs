"""TDD tests for scripts/parity_diff.py (W1-B T6).

Run:  python -m pytest tests/test_parity_diff.py -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import parity_diff  # noqa: E402

HEADER = [
    "ligand",
    "ligand_complex",
    "ligand_means",
    "ligand_props",
    "receptor",
    "receptor_complex",
    "receptor_means",
    "receptor_props",
    "source",
    "target",
    "lr_means",
    "cellphone_pvals",
]
ROW_A = ["protF", "protF", "0.5", "1.0", "protE", "protE", "0.25", "0.5", "A", "A", "0.35", "0.15"]
ROW_B = ["protE", "protE", "0.25", "0.5", "protF", "protF", "0.5", "1.0", "A", "B", "0.35", "0.17"]


def write_csv(path: Path, rows: list[list[str]], header: list[str] = HEADER) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def run(tmp_path: Path, expected_rows, actual_rows, *extra: str):
    """Write both tables, run the tool, return (exit_code, taxonomy_path, stdout)."""
    expected = write_csv(tmp_path / "expected.csv", expected_rows)
    actual = write_csv(tmp_path / "actual.csv", actual_rows)
    taxonomy = tmp_path / "taxonomy.csv"
    code = parity_diff.main(
        ["--expected", str(expected), "--actual", str(actual), "--taxonomy-out", str(taxonomy), *extra]
    )
    return code, taxonomy


def read_taxonomy(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_identical_tables_pass(tmp_path, capsys):
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], [ROW_A, ROW_B])
    out = capsys.readouterr().out
    assert code == 0
    assert "only-in-expected: 0" in out
    assert "only-in-actual: 0" in out
    assert "GATE: PASS" in out
    assert read_taxonomy(taxonomy) == []


def test_perturbed_numeric_value_fails_with_taxonomy(tmp_path, capsys):
    perturbed = [ROW_A[:10] + ["0.36", ROW_A[11]], ROW_B]
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], perturbed)
    out = capsys.readouterr().out
    assert code == 1
    assert "lr_means: numeric exact=1 within-tol=0 differs=1 nan-pattern=0" in out
    assert "GATE: FAIL" in out
    rows = read_taxonomy(taxonomy)
    assert len(rows) == 1
    assert rows[0]["column"] == "lr_means"
    assert rows[0]["kind"] == "differs"
    assert rows[0]["source"] == "A" and rows[0]["target"] == "A"
    assert float(rows[0]["expected"]) == 0.35 and float(rows[0]["actual"]) == 0.36
    assert float(rows[0]["delta"]) == pytest.approx(0.01)


def test_row_only_in_expected_fails(tmp_path, capsys):
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], [ROW_A])
    out = capsys.readouterr().out
    assert code == 1
    assert "only-in-expected: 1" in out
    assert "only-in-actual: 0" in out
    rows = read_taxonomy(taxonomy)
    assert [(r["kind"], r["source"], r["target"]) for r in rows] == [("only_in_expected", "A", "B")]


def test_row_only_in_actual_fails(tmp_path, capsys):
    code, _ = run(tmp_path, [ROW_A], [ROW_A, ROW_B])
    out = capsys.readouterr().out
    assert code == 1
    assert "only-in-actual: 1" in out


def test_nan_pattern_mismatch_counted_separately(tmp_path, capsys):
    nan_row = ROW_A[:11] + [""]
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], [nan_row, ROW_B])
    out = capsys.readouterr().out
    assert code == 1
    assert "cellphone_pvals: numeric exact=1 within-tol=0 differs=0 nan-pattern=1" in out
    rows = read_taxonomy(taxonomy)
    assert len(rows) == 1
    assert rows[0]["column"] == "cellphone_pvals"
    assert rows[0]["kind"] == "nan_pattern"
    assert rows[0]["actual"] == "nan"


def test_within_tolerance_passes(tmp_path, capsys):
    close = [ROW_A[:10] + ["0.350000000000175", ROW_A[11]], ROW_B]
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], close)
    out = capsys.readouterr().out
    assert code == 0
    assert "lr_means: numeric exact=1 within-tol=1 differs=0 nan-pattern=0" in out
    assert "GATE: PASS" in out
    assert read_taxonomy(taxonomy) == []


def test_custom_key_flags_object_differs(tmp_path, capsys):
    changed = [ROW_A[:1] + ["protB", *ROW_A[2:]], ROW_B]
    code, taxonomy = run(tmp_path, [ROW_A, ROW_B], changed, "--key", "source,target")
    assert code == 1
    rows = read_taxonomy(taxonomy)
    assert [(r["column"], r["kind"]) for r in rows] == [("ligand_complex", "differs")]


def test_duplicate_key_is_an_error(tmp_path, capsys):
    code, _ = run(tmp_path, [ROW_A, ROW_B], [ROW_A, ROW_A])
    err = capsys.readouterr().err
    assert code == 2
    assert "duplicate key" in err
