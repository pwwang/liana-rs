#!/usr/bin/env bash
# G2 — real-data differential: the pinned oracle vs `liana-rs` on kang_2018
# (IFN-beta-stimulated PBMCs, 24 673 cells x 15 706 genes, 8 cell types),
# log-normalised, consensus resource, n_perms=1000, seed 1337, liana's
# defaults (expr_prop=0.05, min_cells=5), oracle n_jobs=1, Rust threads=0.
#
# Per method it writes the oracle CSV to testdata/real/expected/ (the
# committed evidence), runs the Rust binary, gates the two against each other
# with `parity_diff.py --rtol 0`, then measures: top-k LR overlap per
# cell-type pair, Spearman on the primary score, significance-mask agreement
# (value <= 0.05 on the method's p-value / rank column) and a taxonomy of any
# disagreement. Stats land in target/real_g2/summary.json.
#
# Needs the oracle venv (liana 2.0.0) and a release binary:
#   cargo build --release -p liana-rs
# The dataset is not committed — see testdata/real/README.md for the fetch.
#
# Usage: scripts/real_g2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
LIANA_ORACLE_PYTHON="${LIANA_ORACLE_PYTHON:-/home/pwwang/p0a/venv/bin/python}"
LIANA_RS_BIN="${LIANA_RS_BIN:-$REPO_ROOT/target/release/liana-rs}"
G2_N_PERMS="${G2_N_PERMS:-1000}"
export LIANA_ORACLE_PYTHON LIANA_RS_BIN G2_N_PERMS

exec "$LIANA_ORACLE_PYTHON" - <<'PY'
import csv
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

PINNED_VERSION = "2.0.0"

DATASET = "testdata/real/kang_lognorm.h5ad"
LABEL_KEY = "cell_abbr"
RESOURCE = "consensus"
N_PERMS = int(os.environ["G2_N_PERMS"])
SEED = 1337
TOP_K = 10
ALPHA = 0.05

# method -> (primary score column, higher is better, mask column or None)
METHODS = {
    "cellphonedb": ("lr_means", True, "cellphone_pvals"),
    "geometric_mean": ("lr_gmeans", True, "gmean_pvals"),
    "cellchat": ("lr_probs", True, "cellchat_pvals"),
    "connectome": ("scaled_weight", True, None),
    "logfc": ("lr_logfc", True, None),
    "natmi": ("spec_weight", True, None),
    "scseqcomm": ("inter_score", True, None),
    "singlecellsignalr": ("lrscore", True, None),
    "rank_aggregate": ("magnitude_rank", False, "magnitude_rank"),
}
KEY_COLS = ("source", "target", "ligand_complex", "receptor_complex")

repo = pathlib.Path(os.environ["REPO_ROOT"])
dataset = repo / DATASET
rust_bin = os.environ["LIANA_RS_BIN"]
expected_dir = repo / "testdata" / "real" / "expected"
out_dir = repo / "target" / "real_g2"

import anndata as ad
import liana as li
import numpy as np
from scipy.stats import spearmanr


def die(msg):
    print(f"real_g2: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


if li.__version__ != PINNED_VERSION:
    die(f"liana {li.__version__} installed, expected {PINNED_VERSION}")
if not dataset.is_file():
    die(f"dataset not found: {dataset} (see testdata/real/README.md)")
if not os.access(rust_bin, os.X_OK):
    die(f"Rust binary not executable: {rust_bin} (cargo build --release -p liana-rs)")

expected_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
adata = ad.read_h5ad(dataset)
dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
print(f"real_g2: {DATASET} sha256={dataset_sha[:16]}… {adata.n_obs} x {adata.n_vars}")


def load(path):
    """The CSV's rows and its `KEY_COLS`-indexed rows (duplicate keys fatal)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    index = {}
    for row in rows:
        key = tuple(row[c] for c in KEY_COLS)
        if key in index:
            die(f"{path}: duplicate key {key}")
        index[key] = row
    return rows, index


def missing(raw):
    """parity_diff.py's `_is_missing`: absent, empty or a literal nan."""
    return raw is None or raw.strip() == "" or raw.strip().lower() == "nan"


def same_cell(expected, actual):
    """parity_diff.py's rule: missing matches missing, numeric by value, the
    rest by string."""
    if missing(expected) or missing(actual):
        return missing(expected) and missing(actual)
    try:
        return float(expected) == float(actual)
    except ValueError:
        return expected == actual


def measure(method, expected_path, actual_path):
    expected_rows, expected = load(expected_path)
    actual_rows, actual = load(actual_path)
    score, higher_better, mask_col = METHODS[method]
    both = sorted(set(expected) & set(actual))
    direction = -1.0 if higher_better else 1.0

    # top-k LR pairs per cell-type pair, ordered by (score, ligand, receptor)
    # so ties break deterministically on both sides.
    pairs = sorted({(key[0], key[1]) for key in both})

    def top_k(index, pair):
        rows = [index[key] for key in both if (key[0], key[1]) == pair]
        rows.sort(key=lambda row: (direction * float(row[score]), row["ligand_complex"], row["receptor_complex"]))
        return [tuple(row[c] for c in KEY_COLS) for row in rows[:TOP_K]]

    overlapping = sum(top_k(expected, pair) == top_k(actual, pair) for pair in pairs)

    expected_scores = [float(expected[key][score]) for key in both]
    actual_scores = [float(actual[key][score]) for key in both]
    spearman = spearmanr(expected_scores, actual_scores).statistic
    if np.isnan(spearman) and expected_scores == actual_scores:
        spearman = 1.0  # a constant column: monotone by definition

    mask = None
    if mask_col is not None:
        def significant(row):
            return not missing(row[mask_col]) and float(row[mask_col]) <= ALPHA

        expected_mask = [significant(expected[key]) for key in both]
        actual_mask = [significant(actual[key]) for key in both]
        mask = {
            "column": mask_col,
            "alpha": ALPHA,
            "significant_expected": int(sum(expected_mask)),
            "agreement": sum(e == a for e, a in zip(expected_mask, actual_mask)) / len(both),
        }

    # disagreement taxonomy: one-sided rows, differing cells (by value), and
    # text-only differences (equal value, different spelling).
    differing, text_only = [], {}
    for key in both:
        for column, expected_raw in expected[key].items():
            if column in KEY_COLS:
                continue
            actual_raw = actual[key][column]
            if expected_raw == actual_raw:
                continue
            if same_cell(expected_raw, actual_raw):
                text_only[column] = text_only.get(column, 0) + 1
            else:
                differing.append(
                    {
                        "key": list(key),
                        "column": column,
                        "expected": expected_raw,
                        "actual": actual_raw,
                        "delta": repr(float(actual_raw) - float(expected_raw))
                        if _numeric(expected_raw, actual_raw)
                        else None,
                    }
                )
    taxonomy = {
        "rows_only_in_expected": len(set(expected) - set(actual)),
        "rows_only_in_actual": len(set(actual) - set(expected)),
        "differing_cells": len(differing),
        "text_only_diffs": text_only,
        "examples": differing[:5],
    }

    return {
        "method": method,
        "score": score,
        "higher_is_better": higher_better,
        "rows_expected": len(expected_rows),
        "rows_actual": len(actual_rows),
        "keys_matched": len(both),
        "cell_type_pairs": len(pairs),
        "top_k": TOP_K,
        "top_k_identical_pairs": overlapping,
        "spearman": float(spearman),
        "mask": mask,
        "taxonomy": taxonomy,
        "row_order_identical": [tuple(r[c] for c in KEY_COLS) for r in expected_rows]
        == [tuple(r[c] for c in KEY_COLS) for r in actual_rows],
        "bytes_identical": pathlib.Path(expected_path).read_bytes()
        == pathlib.Path(actual_path).read_bytes(),
    }


def _numeric(expected_raw, actual_raw):
    try:
        float(expected_raw), float(actual_raw)
        return True
    except ValueError:
        return False


summary = {"dataset": DATASET, "dataset_sha256": dataset_sha, "label_key": LABEL_KEY,
           "resource": RESOURCE, "n_perms": N_PERMS, "seed": SEED,
           "liana_version": li.__version__, "n_obs": int(adata.n_obs),
           "n_vars": int(adata.n_vars),
           "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
           "platform": platform.platform(), "methods": {}}

failures = 0
print(f"{'method':<18} {'rows':>5} {'keys':>5} {'top-k':>7} {'spearman':>9} {'mask':>11} {'sig':>5} {'text':>5} {'gate':>5}")
for method in METHODS:
    expected_path = expected_dir / f"kang__{method}__p{N_PERMS}.csv"
    rust_path = out_dir / f"kang__{method}__p{N_PERMS}.rust.csv"

    result = getattr(li.mt, method)(
        adata, groupby=LABEL_KEY, resource_name=RESOURCE, n_perms=N_PERMS,
        seed=SEED, n_jobs=1, inplace=False, verbose=False,
    )
    result.to_csv(expected_path, index=False)

    subprocess.run(
        [rust_bin, "run", "--h5ad", str(dataset), "--label-key", LABEL_KEY,
         "--resource", RESOURCE, "--method", method, "--n-perms", str(N_PERMS),
         "--seed", str(SEED), "--threads", "0", "--out", str(rust_path)],
        check=True,
    )

    gate = subprocess.run(
        [sys.executable, str(repo / "scripts/parity_diff.py"), "--expected",
         str(expected_path), "--actual", str(rust_path), "--rtol", "0"],
        check=False, capture_output=True, text=True,
    )
    passed = gate.returncode == 0
    failures += not passed
    if not passed:
        print(gate.stdout[-2000:], file=sys.stderr)

    stats = measure(method, expected_path, rust_path)
    stats["gate_rtol_0"] = "PASS" if passed else "FAIL"
    summary["methods"][method] = stats

    mask = stats["mask"]
    mask_text = f"{mask['agreement']:.4f}" if mask else "n/a"
    significant = mask["significant_expected"] if mask else 0
    print(
        f"{method:<18} {stats['rows_expected']:>5} {stats['keys_matched']:>5} "
        f"{stats['top_k_identical_pairs']:>3}/{stats['cell_type_pairs']:<3} "
        f"{stats['spearman']:>9.4f} {mask_text:>11} {significant:>5} "
        f"{sum(stats['taxonomy']['text_only_diffs'].values()):>5} "
        f"{stats['gate_rtol_0']:>5}"
    )

summary_path = out_dir / f"summary_p{N_PERMS}.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"real_g2: summary -> {summary_path.relative_to(repo)}")
sys.exit(1 if failures else 0)
PY
