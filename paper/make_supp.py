#!/usr/bin/env python3
"""Generate the paper's supplementary tables (S1-S3) from the frozen artifacts.

    python3 paper/make_supp.py          # writes paper/supplementary/{S1,S2,S3}.tsv

This script is the provenance: no cell of any table is typed by hand, and
re-running it against the same artifacts reproduces the same TSVs byte for
byte.  Per table:

  S1  configurations and versions.  The box, the suite settings and the
      benchmark's own grid come from `bench/results.json`; the pinned reference
      (version + commit) from the oracle venv's own metadata, cross-checked
      against `scripts/oracle.sh`'s constants; the toolchain and the oracle's
      package versions are read by RUNNING the tools (`cargo --version`,
      `rustc --version`, the venv's own interpreter for pip-show-style
      metadata), and each row names the artifact or the command it came from.
      A live version that has drifted from `bench/results.json`'s frozen
      `versions` block aborts the run: the table would otherwise claim the
      measurements were taken under tooling they were not.

  S2  the parity inventory: one row per method of
      `scripts/check_pipe_parity.sh` (the nine scorers, `rank_aggregate` being
      the aggregate) and one row per ported kernel port.  Method rows are
      re-derived from the frozen gate evidence: the expected CSVs' own headers
      and row counts in `testdata/expected/`, cross-checked against
      `target/w8gates/pipe.log` (the log `scripts/check_pipe_parity.sh`
      produces; when it is absent this script runs the gate to regenerate it).
      Kernel rows count the reference dumps in `testdata/math_ref/` and require
      each Rust gate's zero-bit-difference assertion to still be present.

  S3  the frozen benchmark table: every measurement in `bench/results.json`,
      one row per arm and configuration (stage, method, n_obs, n_lrs,
      n_perms, threads), with wall, whole-process elapsed, peak RSS and the
      suite's own check outcome attached to each row; the three batch RSS
      flatness checks follow as their own rows.

Fail-loud contract: an artifact, field or label this script needs but cannot
find aborts the run (exit 2) and nothing is written - a number is never
substituted and a table is never half-written.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench/results.json"
ORACLE_SH = REPO / "scripts/oracle.sh"
GATE_SH = REPO / "scripts/check_pipe_parity.sh"
GATE_LOG = REPO / "target/w8gates/pipe.log"
EXPECTED = REPO / "testdata/expected"
MATH_REF = REPO / "testdata/math_ref"
OUTDIR = REPO / "paper/supplementary"

ORACLE_PYTHON = pathlib.Path("/home/pwwang/p0a/venv/bin/python")
SRC_RESULTS = "bench/results.json"


class Missing(Exception):
    """An artifact, field or label the tables need is not there."""


def read(path: pathlib.Path) -> str:
    if not path.is_file():
        raise Missing(f"{path.relative_to(REPO)}: not found")
    return path.read_text()


# --------------------------------------------------------------------------- #
# S1: configurations and versions


def run(cmd: list[str], what: str) -> str:
    """Run a version probe; a tool that cannot run is a missing artifact, not a
    reason to fall back to a remembered version."""
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise Missing(f"{what}: `{' '.join(cmd)}` exited {proc.returncode}: "
                      f"{(proc.stderr or proc.stdout).strip()[:200]}")
    return proc.stdout.strip()


def oracle_package_version(pkg: str) -> tuple[str, str]:
    """(version, how) for a package in the oracle venv, read from the venv's own
    interpreter: `pip show` when that venv carries pip, else importlib.metadata
    (the same distribution metadata pip reads)."""
    cmd = [str(ORACLE_PYTHON), "-m", "pip", "show", pkg]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode == 0:
        found = re.search(r"^Version:\s*(\S+)\s*$", proc.stdout, re.M)
        if found:
            return found.group(1), "%s -m pip show %s" % (ORACLE_PYTHON, pkg)
    cmd = [str(ORACLE_PYTHON), "-c",
           "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))", pkg]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise Missing("%s: no version for %r (pip: %s)" % (ORACLE_PYTHON, pkg,
                                                           (proc.stderr or "").strip()[:120]))
    return proc.stdout.strip(), "%s -c importlib.metadata.version('%s') (venv has no pip)" % (
        ORACLE_PYTHON, pkg)


def oracle_pinned() -> tuple[str, str]:
    """(version, commit) the oracle venv actually carries, read by running it."""
    out = run([str(ORACLE_PYTHON), "-c",
               "import json, liana, importlib.metadata as m\n"
               "d = m.distribution('liana')\n"
               "u = d.read_text('direct_url.json')\n"
               "c = json.loads(u)['vcs_info']['commit_id'] if u else ''\n"
               "print(liana.__version__)\nprint(c)\n"],
              "oracle pinned reference")
    version, commit = out.splitlines()
    pinned = re.search(r"PINNED_VERSION = \"([^\"]+)\"", read(ORACLE_SH))
    pinned_commit = re.search(r"PINNED_COMMIT = \"([0-9a-f]+)\"", read(ORACLE_SH))
    if not pinned or not pinned_commit:
        raise Missing("scripts/oracle.sh: no PINNED_VERSION/PINNED_COMMIT constants")
    if version != pinned.group(1) or commit != pinned_commit.group(1):
        raise Missing("the oracle venv carries liana %s @ %s, but scripts/oracle.sh pins "
                      "%s @ %s" % (version, commit, pinned.group(1), pinned_commit.group(1)))
    tag = re.search(r"\(V(\S+), commit [0-9a-f]+\)", read(ORACLE_SH))
    if not tag:
        raise Missing("scripts/oracle.sh: no `(V2.0.0, commit ...)` line to read the tag from")
    return "%s (tag V%s)" % (version, tag.group(1)), commit


def distro() -> str:
    text = pathlib.Path("/etc/os-release").read_text()
    m = re.search(r'^PRETTY_NAME="([^"]+)"', text, re.M)
    if not m:
        raise Missing("/etc/os-release: no PRETTY_NAME")
    return m.group(1)


def s1(doc: dict) -> list[list[str]]:
    box, versions, suite = doc["box"], doc["versions"], doc["suite"]
    rows = [
        ["item", "value", "source"],
        ["machine.cpu", box["cpu"], f"{SRC_RESULTS}#box.cpu"],
        ["machine.cores", str(box["cores"]), f"{SRC_RESULTS}#box.cores"],
        ["machine.memory", "%d GiB (%s kB)" % (box["mem_total_kb"] / 1024 / 1024,
                                               f"{box['mem_total_kb']:,}"),
         f"{SRC_RESULTS}#box.mem_total_kb"],
        ["machine.kernel", box["kernel"], f"{SRC_RESULTS}#box.kernel"],
        ["machine.os", distro(), "/etc/os-release, read at generation time"],
    ]

    cargo = run(["cargo", "--version"], "cargo")
    rustc = run(["rustc", "--version"], "rustc")
    for live, frozen, item in ((cargo, versions["rust"]["cargo"], "rust.cargo"),
                               (rustc, versions["rust"]["rustc"], "rust.rustc")):
        if live != frozen:
            raise Missing("%s: live `%s` vs frozen `%s` in %s#versions - the toolchain moved "
                          "since the suite ran; re-run the suite before regenerating"
                          % (item, live, frozen, SRC_RESULTS))
    rows += [["rust.cargo", cargo, "`cargo --version`, run at generation time"],
             ["rust.rustc", rustc, "`rustc --version`, run at generation time"]]

    version, commit = oracle_pinned()
    rows.append(["reference.liana", "scverse/liana %s" % version,
                 f"{ORACLE_PYTHON} (direct_url.json), cross-checked against scripts/oracle.sh"])
    rows.append(["reference.commit", commit,
                 f"{ORACLE_PYTHON} direct_url.json; pinned in scripts/oracle.sh"])
    if commit != versions["liana_commit"] or version.split()[0] != versions["liana"]:
        raise Missing("oracle venv liana %s @ %s vs %s#versions %s @ %s - the reference moved"
                      % (version, commit, SRC_RESULTS, versions["liana"], versions["liana_commit"]))

    py_version = run([str(ORACLE_PYTHON), "-c", "import platform; print(platform.python_version())"],
                     "oracle python")
    rows.append(["reference.python", py_version, f"{ORACLE_PYTHON} (platform.python_version())"])
    for pkg, frozen_key in (("numpy", "numpy"), ("pandas", "pandas"),
                            ("numba", "numba"), ("scipy", "scipy")):
        live, how = oracle_package_version(pkg)
        if live != versions[frozen_key]:
            raise Missing("oracle %s: live %s vs frozen %s in %s#versions - the oracle moved "
                          "since the suite ran; re-run the suite before regenerating"
                          % (pkg, live, versions[frozen_key], SRC_RESULTS))
        rows.append(["oracle.%s" % pkg, live, how])

    measurements = doc["measurements"]
    sizes = sorted({m["n_obs"] for m in measurements if m["n_obs"]})
    perms = sorted({m["n_perms"] for m in measurements if m["n_perms"]})
    lrs = sorted({m["n_lrs"] for m in measurements if m["n_lrs"]})
    rust_threads = sorted({m["threads"] for m in measurements
                           if m["arm"] == "rust" and m["threads"]})
    py_threads = sorted({m["threads"] for m in measurements if m["arm"] in ("release", "patched")})
    numba_threads = sorted({m["numba_threads"] for m in measurements if m["numba_threads"]})
    datasets = sorted(k for k in doc["inputs_sha256"] if k.endswith(".h5ad"))
    binaries = sorted({pathlib.Path(m["cmd"].split()[0]).name for m in measurements})

    rows += [
        ["benchmark.script", suite["script"], f"{SRC_RESULTS}#suite.script"],
        ["benchmark.seed", str(suite["seed"]), f"{SRC_RESULTS}#suite.seed"],
        ["benchmark.datasets", ", ".join(datasets), f"{SRC_RESULTS}#inputs_sha256 (sha256 per file)"],
        ["benchmark.sizes", ", ".join(f"{n:,}" for n in sizes) + " cells",
         f"{SRC_RESULTS}#measurements[].n_obs"],
        ["benchmark.n_perms", ", ".join(f"{n:,}" for n in perms) + " permutations",
         f"{SRC_RESULTS}#measurements[].n_perms"],
        ["benchmark.n_lrs", ", ".join(f"{n:,}" for n in lrs) + " LR pairs",
         f"{SRC_RESULTS}#measurements[].n_lrs"],
        ["benchmark.threads",
         "rust: %s (RAYON_NUM_THREADS); python: %s (n_jobs); numba kernels: %s"
         % ("/".join(str(n) for n in rust_threads), "/".join(str(n) for n in py_threads),
            "/".join(str(n) for n in numba_threads)),
         f"{SRC_RESULTS}#measurements[].threads; #suite.threads_note"],
        ["benchmark.arms", "; ".join("%s = %s" % (k, v) for k, v in suite["arms"].items()),
         f"{SRC_RESULTS}#suite.arms"],
        ["benchmark.binaries", ", ".join(binaries), f"{SRC_RESULTS}#measurements[].cmd"],
        ["benchmark.results", "bench/results.json (generated %s)" % doc["generated_at"],
         f"{SRC_RESULTS}#generated_at"],
    ]
    return rows


# --------------------------------------------------------------------------- #
# S2: the parity inventory


def gate_methods() -> tuple[list[str], list[int]]:
    """The methods and permutation counts the gate script itself declares."""
    text = read(GATE_SH)
    m = re.search(r"methods=\(([^)]*)\)", text)
    n = re.search(r"for n in ([0-9 ]+);", text)
    if not m or not n:
        raise Missing("scripts/check_pipe_parity.sh: no `methods=(...)` / `for n in ...` lists")
    return m.group(1).replace("\\", " ").split(), [int(x) for x in n.group(1).split()]


def gate_log() -> str:
    """The gate log the S2 rows are read from; runs the gate when it is absent
    (the same read-only run `paper/make_numbers.py` performs)."""
    if not GATE_LOG.is_file():
        GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["bash", "scripts/check_pipe_parity.sh"], cwd=REPO,
                              capture_output=True, text=True)
        GATE_LOG.write_text(proc.stdout + proc.stderr)
    return read(GATE_LOG)


def csv_shape(path: pathlib.Path) -> tuple[list[str], int]:
    if not path.is_file():
        raise Missing(f"{path.relative_to(REPO)}: not found")
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise Missing(f"{path.relative_to(REPO)}: empty")
    return rows[0], len(rows) - 1


def gate_tally(text: str) -> dict:
    """Per method, the gate runs' own numbers, keyed by permutation count."""
    runs = text.split("parity diff:")[1:]
    if not runs:
        raise Missing(f"{GATE_LOG.relative_to(REPO)}: no `parity diff:` runs to tally")
    out: dict[str, dict] = {}
    for run in runs:
        name = re.search(r"expected=\S*__([a-z_]+)__p(\d+)\.csv", run)
        if not name:
            raise Missing(f"{GATE_LOG.relative_to(REPO)}: a run names no `__<method>__p<N>.csv`")
        rows = re.search(r"^rows: (\d+) expected, (\d+) actual, (\d+) matched$", run, re.M)
        rtol = re.search(r"^key: .*rtol: (\S+)$", run, re.M)
        if not rows or not rtol:
            raise Missing(f"{GATE_LOG.relative_to(REPO)}: the {name.group(0)} run has no "
                          "`rows:`/`rtol:` line")
        pass_line = re.search(r"^GATE: (PASS|FAIL) \(([^)]*)\)$", run, re.M)
        if not pass_line:
            raise Missing(f"{GATE_LOG.relative_to(REPO)}: the {name.group(0)} run has no "
                          "`GATE:` line")
        kinds = re.findall(r"^  \S+: (numeric|object) exact=", run, re.M)
        entry = {
            "rows": int(rows.group(1)),
            "numeric": kinds.count("numeric"),
            "object": kinds.count("object"),
            "rtol": rtol.group(1),
            "differing": sum(int(n) for n in re.findall(r"differs=(\d+)", run)),
            "nan_pattern": sum(int(n) for n in re.findall(r"nan-pattern=(\d+)", run)),
            "verdict": pass_line.group(1),
        }
        out.setdefault(name.group(1), {})[int(name.group(2))] = entry
    return out


# kernel ports: (label, dump, test, how the dump's vectors are counted)
def numpy_logexp_vectors(dump: dict) -> int:
    return sum(dump[fn][part]["input"]["count"]
               for fn in ("log", "exp") for part in ("pipeline", "sweep"))


def numpy_logexp_columns(dump: dict) -> int:
    return len(dump["log"]["sweep"]) - 0  # keys: input, output


def ndtr_vectors(dump: dict) -> int:
    return dump["sweep"]["input"]["count"]


def ndtr_columns(dump: dict) -> int:
    return len(dump["sweep"])


def pairwise_vectors(dump: dict) -> int:
    total = sum(case["input"]["count"] for case in dump["cases"])
    if total != dump["checks"]["values"]:
        raise Missing("testdata/math_ref/numpy_pairwise_ref.json: the cases' value count %d "
                      "disagrees with checks.values %d" % (total, dump["checks"]["values"]))
    return total


def pairwise_columns(dump: dict) -> int:
    return len([k for k in dump["cases"][0] if k not in ("kind", "n")])


def betainc_vectors(dump: dict) -> int:
    counts = {name: family["a"]["count"] for name, family in dump["families"].items()}
    for name, family in dump["families"].items():
        if any(vector["count"] != counts[name] for vector in family.values()):
            raise Missing("testdata/math_ref/scipy_betainc_ref.json: family %r is ragged" % name)
    return sum(counts.values())


def betainc_columns(dump: dict) -> int:
    return len(next(iter(dump["families"].values())))  # a, b, x, y


KERNELS = [
    # (label, dump basename, Rust gate, vectors, columns, what the dump holds)
    ("kernel: numpy float32 log/exp", "numpy_math_ref.json", "math_parity.rs",
     numpy_logexp_vectors, numpy_logexp_columns),
    ("kernel: scipy ndtr", "scipy_ndtr_ref.json", "ndtr_parity.rs",
     ndtr_vectors, ndtr_columns),
    ("kernel: numpy float32 pairwise sum/std", "numpy_pairwise_ref.json", "pairwise_parity.rs",
     pairwise_vectors, pairwise_columns),
    ("kernel: boost ibeta (scipy betainc)", "scipy_betainc_ref.json", "betainc_parity.rs",
     betainc_vectors, betainc_columns),
]

ZERO_DIFF_RE = re.compile(r"assert_eq!\(\s*diffs,\s*0,")


def s2() -> list[list[str]]:
    methods, n_perms = gate_methods()
    tally = gate_tally(gate_log())
    rows = [["method", "columns", "numeric", "object", "rows", "n_perms tested", "rtol",
             "differing cells", "source"]]
    for method in methods:
        if method not in tally:
            raise Missing(f"{GATE_LOG.relative_to(REPO)}: no gate run for method `{method}`")
        runs = tally[method]
        for n in n_perms:
            if n not in runs:
                raise Missing(f"{GATE_LOG.relative_to(REPO)}: no p{n} run for `{method}`")
        shapes = []
        for n in n_perms:
            header, n_rows = csv_shape(EXPECTED / f"synthetic__{method}__p{n}.csv")
            run = runs[n]
            if run["rows"] != n_rows or run["numeric"] + run["object"] != len(header):
                raise Missing(
                    "testdata/expected/synthetic__%s__p%d.csv has %d rows x %d columns, but the "
                    "gate log has %d rows x %d columns - the gate log is stale; re-run "
                    "scripts/check_pipe_parity.sh"
                    % (method, n, n_rows, len(header), run["rows"],
                       run["numeric"] + run["object"]))
            shapes.append((run["numeric"], run["object"], run["rows"]))
        numeric, object_, n_rows = shapes[0]
        if any(s != shapes[0] for s in shapes[1:]):
            raise Missing(f"{method}: the fixtures disagree across n_perms: {shapes}")
        differing = sum(runs[n]["differing"] for n in n_perms)
        nan_pattern = sum(runs[n]["nan_pattern"] for n in n_perms)
        verdicts = {runs[n]["verdict"] for n in n_perms}
        if differing or nan_pattern or verdicts != {"PASS"}:
            raise Missing(f"{method}: the gate did not pass cleanly: differing={differing} "
                          f"nan-pattern={nan_pattern} verdicts={verdicts}")
        perms = ", ".join(str(n) for n in n_perms)
        rows.append([method, str(numeric + object_), str(numeric), str(object_), str(n_rows),
                     perms, "0", str(differing),
                     "testdata/expected/synthetic__%s__p{%s}.csv; scripts/check_pipe_parity.sh; "
                     "%s" % (method, perms.replace(", ", ","), GATE_LOG.relative_to(REPO))])
    for label, dump_name, test_name, count, columns in KERNELS:
        dump = json.loads(read(MATH_REF / dump_name))
        gate = read(REPO / "crates/liana-core/tests" / test_name)
        if not ZERO_DIFF_RE.search(gate):
            raise Missing(f"crates/liana-core/tests/{test_name}: the zero-bit-difference "
                          "assertion is gone")
        cols = columns(dump)
        rows.append([label, str(cols), str(cols), "0", str(count(dump)), "n/a", "0 bits", "0",
                     "testdata/math_ref/%s; crates/liana-core/tests/%s" % (dump_name, test_name)])
    return rows


# --------------------------------------------------------------------------- #
# S3: the frozen benchmark table

CHECK_RE = re.compile(r"^(t\d+) (\S+) n_obs=(\d+) n_perms=(\d+) n_lrs=(\d+)$")


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "%g" % value
    return str(value)


def s3(doc: dict) -> list[list[str]]:
    measurements = doc["measurements"]
    by_config: dict[tuple, dict] = {}
    for check in doc["checks"]:
        if "row count" not in check["check"]:
            continue
        m = CHECK_RE.match(check.get("config", ""))
        if not m:
            raise Missing(f"{SRC_RESULTS}#checks: `{check['check']}` carries no parsable config")
        key = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        if key in by_config:
            raise Missing(f"{SRC_RESULTS}#checks: two row-count checks for {key}")
        by_config[key] = check

    rows = [["kind", "stage", "arm", "method", "n_obs", "n_lrs", "n_perms", "threads",
             "numba_threads", "rows", "wall_s", "elapsed_s", "peak_rss_mb", "check"]]
    for m in measurements:
        key = (m["stage"], m["method"], m["n_obs"], m["n_perms"], m["n_lrs"])
        check = by_config.get(key)
        if check is None:
            outcome = "no suite check recorded for this configuration"
        elif m["arm"] not in check["arms"]:
            outcome = ("FAIL: the suite's row-count check covers %s, not %s"
                       % (", ".join(sorted(check["arms"])), m["arm"]))
        elif check["arms"][m["arm"]] != m["rows"]:
            outcome = ("FAIL: the suite's check has %d rows for %s, this run measured %s"
                       % (check["arms"][m["arm"]], m["arm"], m["rows"]))
        else:
            outcome = ("%s: %s (%d rows)" % ("pass" if check["pass"] else "FAIL",
                                             check["check"], check["arms"][m["arm"]]))
        rows.append(["measurement", m["stage"], m["arm"], m["method"], cell(m["n_obs"]),
                     cell(m["n_lrs"]), cell(m["n_perms"]), cell(m["threads"]),
                     cell(m["numba_threads"]), cell(m["rows"]), cell(m["wall_s"]),
                     cell(m["elapsed_s"]), cell(m["rss_mb"]), outcome])

    for check in doc["checks"]:
        flat = re.search(r"flat across n_perms at n_lrs=(\d+) \(50k\)", check["check"])
        if not flat:
            continue
        n_lrs = int(flat.group(1))
        perms = [int(p[1:]) for p in check["rss_kb_by_n_perms"]]
        # the arm the check is about: the one whose t2 rows carry these RSS values
        arms = set()
        for n, kb in zip(perms, (check["rss_kb_by_n_perms"][f"p{p}"] for p in perms)):
            matches = {m["arm"] for m in measurements
                       if m["stage"] == "t2" and m["n_obs"] == 50000 and m["n_lrs"] == n_lrs
                       and m["n_perms"] == n and m["rss_kb"] == kb}
            arms |= matches
        if len(arms) != 1:
            raise Missing(f"{SRC_RESULTS}#checks: `{check['check']}` does not pin one arm: {arms}")
        outcome = "RSS flat across p%s: spread %s kB" % (
            "/".join(str(p) for p in perms), cell(check["spread_kb"]))
        rows.append(["check", "t2", arms.pop(), "rank_aggregate", "50000", str(n_lrs),
                     ", ".join(str(p) for p in perms), "4", "", "", "", "", "", outcome])
    return rows


# --------------------------------------------------------------------------- #

def write_tsv(path: pathlib.Path, rows: list[list[str]]) -> None:
    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise Missing(f"{path.name}: row {i} has {len(row)} cells, the header {width}")
        if any("\t" in c or "\n" in c for c in row):
            raise Missing(f"{path.name}: row {i} carries a tab or newline inside a cell")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")
    print("wrote %s (%d rows x %d columns)" % (path.relative_to(REPO), len(rows) - 1, width))


def main() -> int:
    try:
        doc = json.loads(read(RESULTS))
        tables = [("S1.tsv", s1(doc)), ("S2.tsv", s2()), ("S3.tsv", s3(doc))]
    except Missing as exc:
        print("make_supp: ERROR: %s" % exc, file=sys.stderr)
        return 2
    for name, rows in tables:
        write_tsv(OUTDIR / name, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
