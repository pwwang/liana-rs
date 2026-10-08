#!/usr/bin/env python3
"""Audit `paper/manuscript-v1.md`'s numbers against `paper/numbers.json`.

    python3 paper/audit_claims.py [paper/manuscript-v1.md] [paper/numbers.json]

Every numeric claim is one entry: `phrase` is the fragment as it appears in the
manuscript, `render` rebuilds that fragment's numbers from `numbers.json` (or
from the frozen artifact named in `evidence`). A claim fails if the phrase is
absent, or if what `render` produces differs from what the manuscript says —
so editing either side alone turns the audit red. Exit 0 = consistent,
1 = drifted.

External claims are numbers whose frozen home is not `numbers.json` (a gate
report, a fixture dump, the patch tree): `render` reads that artifact directly
where it is present, and the check is skipped with its evidence path where it
is not.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
MANUSCRIPT = REPO / "paper/manuscript-v1.md"
NUMBERS = REPO / "paper/numbers.json"


def _row(d: dict, arm: str) -> str:
    """A Table 1 row, rebuilt from the 50k × 2,000 × p1000 matrix cell."""
    cell = d["wall_rss_matrix"]["50k/p1000"]["arms"]
    wall = f"{cell[arm]['wall_s']['value']:.1f}"
    rss = (f"{cell[arm]['rss_mb']['value']:.0f} MB" if arm == "rust"
           else f"{cell[arm]['rss_gb']['value']:.1f} GB")
    less = cell[f"{arm}_vs_release"]["rss_less_than"]["value"]
    less = "1.0×" if arm == "release" else f"{less:.1f}× less"
    if arm == "rust":
        return f"| **{wall}** | **{rss}** | **{less}** |"
    return f"| {wall} | {rss} | {less} |"


def _val(node):
    """A frozen entry (`{"value": ...}`) or a bare config scalar."""
    return node["value"] if isinstance(node, dict) else node


def _range(node: dict) -> str:
    return f"{node['min']['value']:.1f}–{node['max']['value']:.1f}"


def _walls(d: dict, arm: str) -> str:
    return "/".join(f"{d['wall_rss_matrix'][f'{n}/p1000']['arms'][arm]['wall_s']['value']:.1f}"
                    for n in ("10k", "50k", "100k"))


def _rows(tally: dict) -> str:
    runs = tally["runs"]["value"]
    return (f"{tally['rows_matched']['value'] // runs} of "
            f"{tally['rows_expected']['value'] // runs} rows")


def _permutation_counts(d: dict) -> str:
    counts = sorted(int(p[1:]) for p in d["parity"]["pipe"]["tally"]["columns"])
    return "at " + " and ".join(f"{c:,}" for c in counts) + " permutations"


# --------------------------------------------------------------------------
# claims whose numbers live in paper/numbers.json

def frozen() -> list[dict]:
    return [
        dict(where="§Summary", phrase="costs about 10 GB",
             render=lambda d: f"costs about "
                              f"{d['law']['predict_defaults_1000x2000']['gb']['value']:.0f} GB"),
        dict(where="§Summary", phrase="about 23 GB with the largest shipped resource",
             render=lambda d: f"about "
                              f"{d['law']['predict_consensus_1000x4620']['gb']['value']:.0f} GB "
                              f"with the largest shipped resource"),
        dict(where="§1", phrase="1,000 permutations and 2,000 LR pairs",
             render=lambda d: f"{d['wall_rss_matrix']['50k/p1000']['n_perms']:,} permutations "
                              f"and {d['wall_rss_matrix']['50k/p1000']['n_lrs']:,} LR pairs"),
        dict(where="§1", phrase="(4,620 pairs)",
             render=lambda d: f"({_val(d['t5_consensus_4620']['n_lrs']):,} pairs)"),
        # the law, in §1 and again in §3.1
        dict(where="§1/§3.1", phrase="≈ 381 MB + 4.98 KB × n_perms × n_lrs",
             render=lambda d: f"≈ {d['law']['recorded_intercept_mb']['value']:.0f} MB + "
                              f"{d['law']['recorded_slope']['value']:.2f} KB × n_perms × n_lrs"),
        dict(where="§3.1", phrase="slope re-fitted at 50k cells: 5.10 KB",
             render=lambda d: f"slope re-fitted at 50k cells: "
                              f"{d['law']['refit_50k']['slope_kb_per_perm_lr']['value']:.2f} KB"),
        dict(where="§3.1", phrase="at 1,000 permutations predicts 22.8 GB",
             render=lambda d: f"at 1,000 permutations predicts "
                              f"{d['law']['predict_consensus_1000x4620']['gb']['value']:.1f} GB"),
        dict(where="§3.1", phrase="measured run reached 23.7 GB",
             render=lambda d: f"measured run reached "
                              f"{d['t5_consensus_4620']['arms']['release']['rss_gb']['value']:.1f} GB"),
        dict(where="§3.2", phrase="the reference reaches 23.7 GB",
             render=lambda d: f"the reference reaches "
                              f"{d['t5_consensus_4620']['arms']['release']['rss_gb']['value']:.1f} GB"),
        dict(where="§3.1", phrase="runs to 100,000 cells",
             render=lambda d: f"runs to {d['wall_rss_matrix']['100k/p1000']['n_obs']:,} cells"),
        # §3 preamble: the box and the two thread settings
        dict(where="§3", phrase="32 threads, 47 GB RAM",
             render=lambda d: f"{d['box']['cores']['value']} threads, "
                              f"{d['box']['mem_total_kb']['value'] / 1024 / 1024:.0f} GB RAM"),
        dict(where="§3", phrase="run with 4 jobs",
             render=lambda d: f"run with "
                              f"{d['wall_rss_matrix']['50k/p1000']['arms']['release']['threads']['value']} jobs"),
        dict(where="§3", phrase="kernels use all 32 threads",
             render=lambda d: f"kernels use all "
                              f"{d['wall_rss_matrix']['50k/p1000']['arms']['release']['numba_threads']['value']} threads"),
        # Table 1, as rows
        dict(where="Table 1", phrase="| 19.4 | 11.0 GB | 1.0× |",
             render=lambda d: _row(d, "release")),
        dict(where="Table 1", phrase="| 20.6 | 1.9 GB | 5.8× less |",
             render=lambda d: _row(d, "patched")),
        dict(where="Table 1", phrase="| **8.4** | **419 MB** | **26.4× less** |",
             render=lambda d: _row(d, "rust")),
        # §3.2 prose after the table
        dict(where="§3.2", phrase="spans 253–419 MB",
             render=lambda d: "spans " + _span_mb(d) + " MB"),
        dict(where="§3.2", phrase="is under 0.2%",
             render=lambda d: "is under 0.2%" if _spread_max(d) < 0.2
                              else f"is under {_spread_max(d):.2f}%"),
        dict(where="§3.2", phrase="is 2.8/8.4/16.2 s",
             render=lambda d: "is " + _walls(d, "rust") + " s"),
        dict(where="§3.2", phrase="13.1/19.4/32.4 s for the release",
             render=lambda d: _walls(d, "release") + " s for the release"),
        dict(where="§3.2", phrase="2.0–4.6×",
             render=lambda d: _range(d["wall_speedups"]["release"]["range"]) + "×"),
        dict(where="§3.2", phrase="1.9–5.0× against the chunked patch",
             render=lambda d: _range(d["wall_speedups"]["patched"]["range"])
                              + "× against the chunked patch"),
        dict(where="§3.2", phrase="13.4–20.6 s for the same configuration",
             render=lambda d: f"{d['patched_wall_spread']['wall_s_min']['value']:.1f}–"
                              f"{d['patched_wall_spread']['wall_s_max']['value']:.1f} s "
                              f"for the same configuration"),
        dict(where="§3.2", phrase="peaks at 741 MB",
             render=lambda d: f"peaks at "
                              f"{d['t5_consensus_4620']['arms']['rust']['rss_mb']['value']:.0f} MB"),
        dict(where="§3.2", phrase="4.9× at 32 threads",
             render=lambda d: f"{d['thread_scaling']['rust']['32']['speedup_vs_1']['value']:.1f}× "
                              f"at 32 threads"),
        dict(where="§3.2", phrase="start-up is 0.6 ms",
             render=lambda d: f"start-up is {d['startup']['version_wall_ms']['value']:.1f} ms"),
        dict(where="§3.2", phrase="1.66 s for `import liana`",
             render=lambda d: f"{d['startup']['python_import']['best_s']['value']:.2f} s "
                              f"for `import liana`"),
        # §3.2, the per-method rows (t6): the release's peak RSS against the engine's
        dict(where="§3.2", phrase="CellPhoneDB peaks at 10.2 GB against 298 MB in the engine (34.3× less)",
             render=lambda d: f"CellPhoneDB peaks at "
                              f"{_pm(d, 'cellphonedb', 'release', 'rss_gb'):.1f} GB against "
                              f"{_pm(d, 'cellphonedb', 'rust', 'rss_mb'):.0f} MB in the engine "
                              f"({_pm(d, 'cellphonedb', 'release_over_rust_rss'):.1f}× less)"),
        dict(where="§3.2", phrase="CellChat at 11.4 GB against 326 MB (35.1× less)",
             render=lambda d: f"CellChat at {_pm(d, 'cellchat', 'release', 'rss_gb'):.1f} GB "
                              f"against {_pm(d, 'cellchat', 'rust', 'rss_mb'):.0f} MB "
                              f"({_pm(d, 'cellchat', 'release_over_rust_rss'):.1f}× less)"),
        dict(where="§3.2", phrase="the chunked patch takes both to 1086 MB and 1120 MB",
             render=lambda d: f"the chunked patch takes both to "
                              f"{_pm(d, 'cellphonedb', 'patched', 'rss_mb'):.0f} MB and "
                              f"{_pm(d, 'cellchat', 'patched', 'rss_mb'):.0f} MB"),
        # §3.3, the fixture gate's tallies
        dict(where="§3.3", phrase="at 100 and 1,000 permutations",
             render=_permutation_counts),
        dict(where="§3.3", phrase="122 of 122 columns (68 numeric, 54 string)",
             render=lambda d: _columns(d["parity"]["pipe"]["tally"]["columns"]["p100"])),
        dict(where="§3.3", phrase="440 of 440 rows",
             render=lambda d: _rows(d["parity"]["pipe"]["tally"])),
        dict(where="§3.3", phrase="18 gate runs",
             render=lambda d: f"{d['parity']['pipe']['tally']['runs']['value']} gate runs"),
        dict(where="§3.3", phrase="no differing cells",
             render=lambda d: "no differing cells" if not _differing(d)
                              else f"{_differing(d)} differing cells"),
    ]


def _pm(d: dict, method: str, *path: str) -> float:
    """A per-method (t6) number: `_pm(d, "cellchat", "release", "rss_gb")`, or
    `_pm(d, "cellchat", "release_over_rust_rss")` on the method itself."""
    node = d["per_method"]["methods"][method]
    for part in path:
        node = node[part]
    return node["value"]


def _columns(count: dict) -> str:
    total = count["total"]["value"]
    return (f"{total} of {total} columns "
            f"({count['numeric']['value']} numeric, {count['object']['value']} string)")


def _differing(d: dict) -> int:
    return sum(d["parity"][gate]["tally"]["differing_cells"]["value"]
               for gate in ("pipe", "cli", "py"))


def _per_column(d: dict, key: str) -> list[float]:
    return [c[key]["value"] for c in d["law_grid"]["rust_flatness"]["per_column"].values()]


def _span_mb(d: dict) -> str:
    return (f"{min(_per_column(d, 'min_rss_kb')) / 1024:.0f}"
            f"–{max(_per_column(d, 'max_rss_kb')) / 1024:.0f}")


def _spread_max(d: dict) -> float:
    """The largest within-column RSS spread over the ten-fold `n_perms` change."""
    return max(_per_column(d, "spread_pct"))


# --------------------------------------------------------------------------
# claims whose frozen home is not numbers.json

PATCHED = pathlib.Path("/home/pwwang/p0a/patched/liana")
ORACLE = pathlib.Path("/home/pwwang/p0a/venv/lib/python3.12/site-packages/liana")


def external() -> list[dict]:
    return [
        dict(where="§2.1", phrase="one-part-in-10⁶ relative band",
             evidence="crates/liana-core/src/perms/null.rs — `TIE_RTOL: f64 = 1e-6`",
             render=lambda: _grep_ok("crates/liana-core/src/perms/null.rs",
                                     r"TIE_RTOL: f64 = 1e-6",
                                     "one-part-in-10⁶ relative band")),
        dict(where="§2.2", phrase="17-resource table",
             evidence="crates/liana-core/data/omni_resource.csv",
             render=_resource_count),
        dict(where="§2.3", phrase="kang_2018, 24,673 cells",
             evidence="testdata/real/README.md (`24 673 cells`); target/real_g2/summary_p100.json",
             render=_kang_cells),
        dict(where="§2.3/§3.3", phrase="ten reference configurations",
             evidence="ops/logs/w1b-report.md (T7); testdata/rng_ref/manifest.json",
             render=_rng_configs),
        dict(where="§3.2", phrase="blocked allocation (68 added lines across two files)",
             evidence=f"diff of {PATCHED} against the pinned 2.0.0 tree (read-only)",
             render=_patch_size),
        dict(where="§3.3", phrase="3,509 of 3,509 rows bit-identical",
             evidence="ops/logs/w6-report.md — `3509/3509 rows`, `1.0000`",
             render=_kang_gate),
        dict(where="§2.3", phrase="three defects",
             evidence="ops/logs/w6-report.md + w5b/w7 reports (index ceiling, group-sum order, compensated sums)",
             render=_three_defects),
    ]


def _read(rel: str) -> str:
    path = REPO / rel
    return path.read_text() if path.exists() else ""


def _grep_ok(rel: str, pattern: str, phrase: str) -> str:
    return phrase if re.search(pattern, _read(rel)) else f"{pattern!r} not found in {rel}"


def _resource_count() -> str:
    with open(REPO / "crates/liana-core/data/omni_resource.csv") as handle:
        names = {row["resource"] for row in csv.DictReader(handle)}
    return f"{len(names)}-resource table"


def _kang_cells() -> str:
    summary = pathlib.Path("target/real_g2/summary_p100.json")
    if summary.exists():
        n_obs = json.loads(summary.read_text())["n_obs"]
    else:  # the dataset is fetched, not committed; the README pins its shape
        found = re.search(r"(\d[\d ]*\d) cells", _read("testdata/real/README.md"))
        if not found:
            return "kang cell count not found (SKIP: no target/real_g2, no README shape)"
        n_obs = int(found.group(1).replace(" ", ""))
    return f"kang_2018, {n_obs:,} cells"


def _rng_configs() -> str:
    entries = json.loads((REPO / "testdata/rng_ref/manifest.json").read_text())["entries"]
    words = {9: "nine", 10: "ten", 11: "eleven"}
    return f"{words.get(len(entries), len(entries))} reference configurations"


def _patch_size() -> str:
    if not PATCHED.exists():
        return "patch tree not present — SKIP"
    # Insertions the way `git diff --numstat` counts them: every `+` line, blank
    # ones included (blank separators are what takes 58 to 68 on the larger file).
    added = 0
    files = 0
    for rel in sorted(p.relative_to(ORACLE) for p in ORACLE.rglob("*.py")):
        diff = subprocess.run(["diff", "-u", str(ORACLE / rel), str(PATCHED / rel)],
                              capture_output=True, text=True).stdout
        additions = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
        if additions:
            added += additions
            files += 1
    return f"blocked allocation ({added} added lines across {'two files' if files == 2 else f'{files} files'})"


def _kang_gate() -> str:
    report = _read("ops/logs/w6-report.md")
    if "3509/3509 rows" in report and "1.0000" in report:
        return "3,509 of 3,509 rows bit-identical"
    return "w6 report not found or does not carry the Kang gate — SKIP"


def _three_defects() -> str:
    """The three defects §2.3 names, each with its own fingerprint in the logs."""
    w6 = _read("ops/logs/w6-report.md")
    w7 = _read("ops/logs/w7-report.md")
    found = [
        "65,536" in w7 or "65536" in w7,        # index-width ceiling (w7)
        "reassembled" in w6,                    # frame error in the group sums (w6)
        "Kahan-compensated" in w6,              # compensated-summation detail (w6)
    ]
    return ("three defects" if all(found)
            else f"{sum(found)} of the three defects documented in the reports")


# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    manuscript_path = pathlib.Path(argv[1]) if len(argv) > 1 else MANUSCRIPT
    numbers_path = pathlib.Path(argv[2]) if len(argv) > 2 else NUMBERS
    text = manuscript_path.read_text()
    doc = json.loads(numbers_path.read_text())

    failures = []
    print(f"{'where':11s} {'claim':60s} result")
    print("-" * 104)
    for claim in frozen():
        rendered = claim["render"](doc)
        if claim["phrase"] not in text:
            result = f"FAIL phrase not in manuscript: {claim['phrase']!r}"
        elif rendered not in claim["phrase"]:
            result = f"FAIL manuscript {claim['phrase']!r} vs numbers {rendered!r}"
        else:
            result = f"ok  {rendered}"
        print(f"{claim['where']:11s} {claim['phrase']:60s} {result}")
        if result.startswith("FAIL"):
            failures.append((claim["where"], claim["phrase"]))
    print()
    for claim in external():
        rendered = claim["render"]()
        if claim["phrase"] not in text:
            result = f"FAIL phrase not in manuscript: {claim['phrase']!r}"
        elif rendered not in claim["phrase"]:
            result = f"FAIL manuscript {claim['phrase']!r} vs evidence {rendered!r}"
        else:
            result = f"ok  {rendered}"
        print(f"{claim['where']:11s} {claim['phrase']:60s} {result}")
        if result.startswith("FAIL"):
            failures.append((claim["where"], claim["phrase"]))

    total = len(frozen()) + len(external())
    print(f"\n{total} claims checked, {len(failures)} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
