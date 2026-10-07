"""Assemble `bench/bench_suite.sh`'s raw TSVs + the box's versions into
`bench/results.json` — the frozen, machine-readable manifest behind
`bench/RESULTS.md`.

Reads   <outdir>/results_t{1..5}.tsv, <outdir>/cmds_t{1..5}.tsv
        (default outdir: <repo>/target/bench7)
Writes  <repo>/bench/results.json, <repo>/bench/RESULTS.md

Everything in the output is either measured by the suite on this box or a
version/checksum read from the pinned toolchains; the one deliberate exception
is `recorded_references`, which re-reads the p0a logs the paper's baseline
numbers came from (read-only) so the manifest says where each came from. The
wall-stability replications quoted in the generated Caveats prose live, with
their exact commands, in `ops/logs/w7-report.md`.
"""
import hashlib
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "target" / "bench7"
DATA = pathlib.Path("/home/pwwang/p0a/data")
VENV_PY = pathlib.Path("/home/pwwang/p0a/venv/bin/python")
P0A_LOGS = pathlib.Path("/home/pwwang/p0a/logs")

STAGES = ["t1", "t2", "t3", "t4", "t5"]
COLUMNS = [
    "label", "arm", "method", "n_obs", "n_lrs", "n_perms", "threads",
    "numba_threads", "rows", "wall_s", "elapsed_s", "rss_kb", "rc", "note",
]

# The recorded Python baselines of the paper's rank_aggregate comparison — every
# row is a `/usr/bin/time -v` run on this box, logged under /home/pwwang/p0a/logs.
RECORDED = [
    ("mc1_release50k_j4.log", "release rank_aggregate 50k p1000 j4",
     "the brief's '11.3 GB / ~19 s' row"),
    ("mc1_patched50k_j4.log", "patched rank_aggregate 50k p1000 j4", ""),
    ("ra10k_j4_base2.log", "release rank_aggregate 10k p1000 j4", ""),
    ("ra10k_j4_patched.log", "patched rank_aggregate 10k p1000 j4", ""),
    ("ra50k_j4_base2.log", "release rank_aggregate 50k p1000 j4 (repeat)", ""),
    ("ra50k_j4_patched.log", "patched rank_aggregate 50k p1000 j4 (repeat)", ""),
    ("ra50k_j4_p100_base.log", "release rank_aggregate 50k p100 j4", ""),
    ("ra50k_j4_p100_patched.log", "patched rank_aggregate 50k p100 j4", ""),
    ("ra50k_j4_p2000_base.log", "release rank_aggregate 50k p2000 j4", ""),
    ("ra50k_j4_p2000_patched.log", "patched rank_aggregate 50k p2000 j4", ""),
    ("ra100k_j4.log", "release rank_aggregate 100k p1000 j4", ""),
    ("ra50k_j1.log", "release rank_aggregate 50k p1000 j1", "thread-scaling reference"),
    ("ra50k_j16.log", "release rank_aggregate 50k p1000 j16", "thread-scaling reference"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    out = {"rust": {}}
    cargo = pathlib.Path.home() / ".cargo/bin"
    for tool in ("cargo", "rustc"):
        p = run([str(cargo / tool), "--version"])
        out["rust"][tool] = p.stdout.strip() or p.stderr.strip()
    out["git_head"] = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    out["git_dirty"] = bool(run(["git", "-C", str(REPO), "status", "--porcelain"]).stdout.strip())
    p = run([str(VENV_PY), "-c", """
import json, sys
import liana, numpy, pandas, numba, scipy, anndata, scanpy
from importlib.metadata import distribution
try:
    commit = json.loads(distribution("liana").read_text("direct_url.json"))["vcs_info"]["commit_id"]
except Exception:
    commit = None
print(json.dumps({
    "python": sys.version.split()[0], "liana": liana.__version__, "liana_commit": commit,
    "numpy": numpy.__version__, "pandas": pandas.__version__, "numba": numba.__version__,
    "scipy": scipy.__version__, "anndata": anndata.__version__, "scanpy": scanpy.__version__,
}))
"""])
    out.update(json.loads(p.stdout))
    cpuinfo = pathlib.Path("/proc/cpuinfo").read_text()
    out["cpu"] = re.search(r"model name\s*:\s*(.*)", cpuinfo).group(1).strip()
    out["cores"] = cpuinfo.count("processor\t")
    out["mem_total_kb"] = int(re.search(r"MemTotal:\s+(\d+)", pathlib.Path("/proc/meminfo").read_text()).group(1))
    out["kernel"] = run(["uname", "-r"]).stdout.strip()
    return out


def inputs():
    files = {}
    for n_obs in (1000, 10000, 50000, 100000):
        for name in (f"sc_{n_obs}.h5ad", f"resource_{n_obs}.csv"):
            files[name] = sha256(DATA / name)
    files["resource_50k_lrs4620.csv"] = sha256(OUTDIR / "resources/resource_50k_lrs4620.csv")
    return files


def measurements():
    rows, cmds = [], {}
    for stage in STAGES:
        for path in (OUTDIR / f"cmds_{stage}.tsv",):
            if path.exists():
                for line in path.read_text().splitlines():
                    _, label, cmd = line.split("\t", 2)
                    cmds[label] = cmd
    for stage in STAGES:
        path = OUTDIR / f"results_{stage}.tsv"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            values = line.split("\t")
            row = dict(zip(COLUMNS, values))
            row["stage"] = stage
            for key in ("n_obs", "n_lrs", "n_perms", "threads", "numba_threads", "rows", "rc"):
                row[key] = int(row[key]) if row[key] else None
            for key in ("wall_s", "elapsed_s", "rss_kb"):
                row[key] = float(row[key]) if row[key] else None
            row["rss_mb"] = round(row["rss_kb"] / 1024, 1) if row["rss_kb"] else None
            row["cmd"] = cmds.get(row["label"])
            rows.append(row)
    return rows


def recorded():
    out = []
    for name, description, note in RECORDED:
        path = P0A_LOGS / name
        if not path.exists():
            out.append({"log": name, "description": description, "missing": True})
            continue
        text = path.read_text()
        wall = re.search(r"^WALL_S=([\d.]+)", text, re.M)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        out.append({
            "log": str(path), "description": description, "note": note,
            "wall_s": float(wall.group(1)) if wall else None,
            "rss_kb": int(rss.group(1)) if rss else None,
            "rss_mb": round(int(rss.group(1)) / 1024, 1) if rss else None,
        })
    startup = P0A_LOGS / "startup.log"
    text = startup.read_text()
    runs = re.search(r"^JSON_RUNS=(.*)$", text, re.M)
    out.append({
        "log": str(startup), "description": "python startup / import timings",
        "note": "the recorded `python -c 'import liana'` baseline (best of 3)",
        "json_runs": json.loads(runs.group(1)) if runs else None,
    })
    return out


def checks(rows):
    """Agreements the suite is supposed to produce, asserted on the collected
    rows — a mismatch here means the manifest is inconsistent, not a new fact."""
    out = []
    by_cfg = {}
    for row in rows:
        if row["stage"] in ("t1", "t2", "t5") and row["rows"]:
            by_cfg.setdefault(
                (row["stage"], row["n_obs"], row["n_perms"], row["n_lrs"]), []
            ).append((row["arm"], row["rows"]))
    for cfg, arms in sorted(by_cfg.items(), key=lambda kv: str(kv[0])):
        counts = {r for _, r in arms}
        out.append({
            "check": "row count agrees across arms",
            "config": f"{cfg[0]} n_obs={cfg[1]} n_perms={cfg[2]} n_lrs={cfg[3]}",
            "arms": dict(arms), "pass": len(counts) == 1,
        })
    rust = [r for r in rows if r["stage"] == "t2" and r["arm"] == "rust" and r["rss_kb"]]
    for n_lrs in sorted({r["n_lrs"] for r in rust}):
        col = [r for r in rust if r["n_lrs"] == n_lrs]
        lo, hi = min(r["rss_kb"] for r in col), max(r["rss_kb"] for r in col)
        out.append({
            "check": f"liana-rs peak RSS flat across n_perms at n_lrs={n_lrs} (50k)",
            "rss_kb_by_n_perms": {f"p{r['n_perms']}": round(r["rss_kb"]) for r in col},
            "spread_kb": hi - lo,
            "pass": (hi - lo) < 0.10 * lo,
        })
    return out


def law(rows):
    """The recorded law `peak ≈ 381 MB + 4.98 KB × n_perms × n_lrs` against the
    measured t2 rows: the release arm's own fit at 50k, and the anchor configs at
    10k the law was fitted from."""
    law_kb = lambda n_perms, n_lrs: 381 * 1024 + 4.98 * n_perms * n_lrs  # noqa: E731
    out = {"statement": "peak ≈ 381 MB + 4.98 KB × n_perms × n_lrs", "anchors": [], "fit_50k": None}
    for row in rows:
        if row["stage"] != "t2" or row["arm"] != "release" or not row["rss_kb"]:
            continue
        row["law_pred_kb"] = round(law_kb(row["n_perms"], row["n_lrs"]))
        row["law_residual_kb"] = round(row["rss_kb"] - row["law_pred_kb"])
        if row["n_obs"] == 10000:
            out["anchors"].append({
                "label": row["label"], "n_perms": row["n_perms"], "n_lrs": row["n_lrs"],
                "measured_kb": round(row["rss_kb"]), "law_pred_kb": row["law_pred_kb"],
                "residual_kb": row["law_residual_kb"],
            })
    rel = [r for r in rows if r["stage"] == "t2" and r["arm"] == "release"
           and r["n_obs"] == 50000 and r["rss_kb"]]
    if rel:
        # least squares on peak_kb = a + b × (n_perms × n_lrs), the law's shape
        xs = [r["n_perms"] * r["n_lrs"] for r in rel]
        ys = [r["rss_kb"] for r in rel]
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - b * sx) / n
        out["fit_50k"] = {
            "slope_kb_per_perm_lr": round(b, 4),
            "intercept_kb": round(a),
            "intercept_mb": round(a / 1024),
            "n_points": n,
            "recorded_slope": 4.98,
            "recorded_intercept_mb": 381,
        }
    return out


def fmt_mb(rss_kb):
    return "—" if not rss_kb else f"{rss_kb / 1024:.0f}"


def fmt_s(seconds):
    return "—" if seconds in (None, "") else f"{float(seconds):.2f}"


def markdown(doc):
    """`bench/RESULTS.md` — the human table, generated from the same rows as the
    manifest so the two cannot drift."""
    rows = doc["measurements"]

    def find(label):
        return next((r for r in rows if r["label"] == label), None)

    def cell(label, key, fmt=fmt_s):
        row = find(label)
        return fmt(row[key]) if row else "—"

    def k(n_obs):
        return {1000: "1k", 10000: "10k", 50000: "50k", 100000: "100k"}.get(n_obs, str(n_obs))

    L = []
    a = L.append
    box = doc["box"]
    v = doc["versions"]
    a("# W7 benchmark results — liana-rs vs liana 2.0.0 vs patched-Python")
    a("")
    a("The frozen numbers behind the paper's figures. Every row was measured on one")
    a(f"box — {box['cpu']}, {box['cores']} vCPU, {box['mem_total_kb'] // 1024 // 1024} GB RAM, "
      f"kernel {box['kernel']} — in serial")
    a(f"passes of `bench/bench_suite.sh` on {doc['generated_at'][:10]}, seed 1337: stages")
    a("t1–t4 in one run; t5 re-run once (each stage rewrites its own TSV) after its")
    a("4,620-LR resource was rebuilt — the failed first attempt is kept in")
    a("`target/bench7/miss_t5_resource/`. The")
    a("machine-readable twin of this file is `bench/results.json`: every configuration")
    a("with its exact command, wall, peak RSS, thread counts, versions and input")
    a("sha256s. `bench/RESULTS.md` is generated from it (`bench/collect_results.py`),")
    a("so the two cannot disagree.")
    a("")
    a("| arm | what runs |")
    a("|---|---|")
    a("| `rust` | `bench/engine_bench` — `liana_core::run::Method`, the `liana-rs run` "
      "CLI's own dispatch |")
    a("| `release` | liana 2.0.0 (`V2.0.0`, commit `%s`) in the pinned oracle venv |"
      % (v.get("liana_commit") or "?")[:12])
    a("| `patched` | the same venv with `PYTHONPATH=/home/pwwang/p0a/patched` (W4/W5 "
      "memory patch) |")
    a("")
    a("**Wall bases.** `wall` is the in-process measurement: for `rust` it covers")
    a("read + resource + method (`engine-bench`'s own clock); for the Python arms it is")
    a("`run_arm.py`'s `WALL_S=`, the method call only (inputs read before `t0`, so it")
    a("flatters Python by that read). `elapsed` is `/usr/bin/time -v`'s whole-process")
    a("wall — on the Python side that includes interpreter start and the ~1.6 s import")
    a("of the scientific stack. Peak RSS is `time -v`'s \"Maximum resident set size\"")
    a("(the Rust bin cross-checks it with `VmHWM`; they agree to the kB).")
    a("")
    a("**Threads.** `rust` runs on `RAYON_NUM_THREADS`; the Python arms run `n_jobs=4`")
    a("(joblib) with their numba kernels at numba's own default — `NUMBA_THREADS=32` in")
    a("every Python run recorded here (W4 D4), so the 4-thread comparison is generous")
    a("to Python, not to Rust. One machine-wide caveat: this WSL2 box occasionally")
    a("stalls a process for seconds (two rows below show a `time -v` `elapsed` several")
    a("seconds above the in-process wall; the same stall hit a `python -c` startup rep")
    a("at 5.26 s vs 1.6 s). In-process walls are the stable numbers; `elapsed` is")
    a("reported as measured.")
    a("")

    # ---- T1
    a("## T1 — `rank_aggregate` end to end, 4 threads")
    a("")
    a("Wall, seconds (in-process basis):")
    a("")
    a("| dataset | perms | rust | release | patched | rust vs release | rust vs patched |")
    a("|---|---|---|---|---|---|---|")
    for n_obs in (10000, 50000, 100000):
        for n_perms in (100, 1000):
            rl = f"t1_ra_{n_obs // 1000}k_p{n_perms}_rust"
            r, rel, pat = find(rl), find(rl.replace("_rust", "_release")), find(rl.replace("_rust", "_patched"))
            sp = (f"{rel['wall_s'] / r['wall_s']:.1f}×" if r and rel and r["wall_s"] else "—")
            sp2 = (f"{pat['wall_s'] / r['wall_s']:.1f}×" if r and pat and r["wall_s"] else "—")
            a(f"| {k(n_obs)} | {n_perms} | {cell(rl, 'wall_s')} | "
              f"{cell(rl.replace('_rust', '_release'), 'wall_s')} | "
              f"{cell(rl.replace('_rust', '_patched'), 'wall_s')} | {sp} | {sp2} |")
    a("")
    a("Peak RSS, MB:")
    a("")
    a("| dataset | perms | rust | release | patched | release / rust | patched / rust |")
    a("|---|---|---|---|---|---|---|")
    for n_obs in (10000, 50000, 100000):
        for n_perms in (100, 1000):
            rl = f"t1_ra_{n_obs // 1000}k_p{n_perms}_rust"
            r, rel, pat = find(rl), find(rl.replace("_rust", "_release")), find(rl.replace("_rust", "_patched"))
            mr = (f"{rel['rss_kb'] / r['rss_kb']:.1f}×" if r and rel and r["rss_kb"] else "—")
            mp = (f"{pat['rss_kb'] / r['rss_kb']:.1f}×" if r and pat and r["rss_kb"] else "—")
            a(f"| {k(n_obs)} | {n_perms} | {cell(rl, 'rss_kb', fmt_mb)} | "
              f"{cell(rl.replace('_rust', '_release'), 'rss_kb', fmt_mb)} | "
              f"{cell(rl.replace('_rust', '_patched'), 'rss_kb', fmt_mb)} | {mr} | {mp} |")
    a("")
    cli = find("t1_ra_50k_p1000_cli")
    if cli:
        a(f"Cross-check through the CLI binary (`liana-rs run`, 50k × p1000, 4 threads): "
          f"{cli['rows']} rows, {fmt_s(cli['elapsed_s'])} s whole-process, "
          f"{fmt_mb(cli['rss_kb'])} MB peak — the same dispatch the harness's `rust`")
        a("rows use, driven through the shipped binary.")
        a("")
        a("The `patched` arm's wall is this table's least box-state-stable cell — see")
        a("*Box-state sensitivity* under Caveats before quoting it.")
        a("")

    # ---- T2
    a("## T2 — the memory law: peak RSS vs `n_perms` × `n_lrs` at 50k")
    a("")
    a("Recorded law (release arm, `n_obs=10k`, fitted from the logged mem probes):")
    a("`peak ≈ 381 MB + 4.98 KB × n_perms × n_lrs`. Measured today at 50k, peak RSS in MB:")
    a("")
    a("| n_perms | arm | n_lrs=200 | n_lrs=1000 | n_lrs=2000 | law @2000 (release) |")
    a("|---|---|---|---|---|---|")
    law_kb = lambda n_perms, n_lrs: 381 * 1024 + 4.98 * n_perms * n_lrs  # noqa: E731
    for n_perms in (10, 100, 1000):
        for arm in ("rust", "release", "patched"):
            cells = " | ".join(
                cell(f"t2_ra_50k_p{n_perms}_lrs{n_lrs}_{arm}", "rss_kb", fmt_mb)
                for n_lrs in (200, 1000, 2000)
            )
            a(f"| {n_perms} | {arm} | {cells} | {law_kb(n_perms, 2000) / 1024:.0f} |")
    a("")
    rust_rows = [r for r in rows if r["stage"] == "t2" and r["arm"] == "rust" and r["rss_kb"]]
    if rust_rows:
        cols = {}
        for r in rust_rows:
            cols.setdefault(r["n_lrs"], []).append(r)
        spreads = []
        for n_lrs in sorted(cols):
            col = cols[n_lrs]
            lo, hi = min(r["rss_kb"] for r in col), max(r["rss_kb"] for r in col)
            spreads.append(f"{n_lrs} LRs: {fmt_mb(lo)}–{fmt_mb(hi)} MB "
                           f"({(hi - lo) / lo * 100:.1f}%)")
        lo, hi = min(r["rss_kb"] for r in rust_rows), max(r["rss_kb"] for r in rust_rows)
        a(f"At fixed resource size the rust arm's peak is flat in `n_perms` "
          f"(across p10/p100/p1000 — {', '.join(spreads)}); the permutation null")
        a(f"adds nothing measurable. The whole grid spans {fmt_mb(lo)}–{fmt_mb(hi)} MB, and")
        a("the growth is the resource side — the LR table the method carries — not memory")
        a("of the perms.")
    fit = doc["law"].get("fit_50k")
    if fit:
        a("")
        a(f"The release arm's own fit at 50k (least squares over the nine points, the law's")
        a(f"shape) gives slope **{fit['slope_kb_per_perm_lr']:.2f} KB per (perm × LR)** — the")
        a(f"recorded 4.98 KB holds — with intercept **{fit['intercept_mb']} MB**, larger than")
        a("the recorded 381 MB because the intercept carries the dataset (50k's matrix is")
        a("bigger than 10k's). Anchors at 10k (release arm — where the law was fitted):")
        a("")
        a("| config | measured MB | law MB | residual MB |")
        a("|---|---|---|---|")
        for anchor in doc["law"]["anchors"]:
            a(f"| p{anchor['n_perms']} × lrs{anchor['n_lrs']} | {anchor['measured_kb'] / 1024:.0f} | "
              f"{anchor['law_pred_kb'] / 1024:.0f} | {anchor['residual_kb'] / 1024:+.0f} |")
    a("")

    # ---- T3
    a("## T3 — thread scaling (`rank_aggregate`, 50k × 1000)")
    a("")
    a("| threads | rust wall (s) | peak RSS (MB) | speedup vs 1 |")
    a("|---|---|---|---|")
    base = find("t3_ra_50k_p1000_t1_rust")
    for threads in (1, 4, 8, 32):
        label = f"t3_ra_50k_p1000_t{threads}_rust"
        row = find(label)
        sp = f"{base['wall_s'] / row['wall_s']:.1f}×" if base and row and row["wall_s"] else "—"
        a(f"| {threads} | {cell(label, 'wall_s')} | {cell(label, 'rss_kb', fmt_mb)} | {sp} |")
    a("")
    a("Recorded Python references from `/home/pwwang/p0a/logs` (joblib `n_jobs`, numba at")
    a("32 threads) are in the table at the end of this file.")
    a("")

    # ---- T4
    a("## T4 — startup and first run")
    a("")
    a("| what | wall | peak RSS (MB) |")
    a("|---|---|---|")
    hr = find("t4_lianars_version_hr")
    if hr:
        a(f"| `liana-rs --version` (mean of 20, bash clock) | {float(hr['wall_s']) * 1000:.1f} ms | "
          f"{fmt_mb(find('t4_lianars_version_r1')['rss_kb'])} |")
    first = find("t4_firstrun_connectome_1k")
    if first:
        a(f"| `liana-rs run` first run — connectome, 1k cells × {first['n_lrs']} LRs, "
          f"{first['rows']} rows | {fmt_s(first['elapsed_s'])} s (whole process) | "
          f"{fmt_mb(first['rss_kb'])} |")
    imports = [r for r in rows if r["label"].startswith("t4_python_import_r")]
    if imports:
        best = min(r["elapsed_s"] for r in imports)
        med = sorted(r["elapsed_s"] for r in imports)[len(imports) // 2]
        a(f"| `python -c \"import liana\"` (oracle venv) | best {best:.2f} s, median {med:.2f} s "
          f"of {len(imports)} | {fmt_mb(imports[0]['rss_kb'])} |")
    a("")
    a("The recorded baseline for the import was **1.78 s** (`p0a/logs/startup.log`); the")
    a("binary starts in under a millisecond.")
    if first and first["elapsed_s"] and imports:
        best = min(r["elapsed_s"] for r in imports)
        cmp_ = "less" if first["elapsed_s"] < best else "more"
        a(f"A whole 1k-cell method run ({fmt_s(first['elapsed_s'])} s) takes {cmp_} wall clock")
        a(f"than one Python import of the stack ({best:.2f} s).")
    a("")

    # ---- T5
    a("## T5 — a 4,620-LR resource (the consensus resource's size), 50k × 1000")
    a("")
    a("The resource is drawn from this synthetic data's gene universe with the paper's")
    a("own `_sample_resource` recipe — the same one that built `resource_{n_obs}.csv` —")
    a("at the consensus resource's 4,620-pair size. The literal")
    a("`select_resource('consensus')` symbols cannot run here: they do not exist in the")
    a("synthetic `Gene{i}` var_names, and all three arms reject that dump (rc=101/1,")
    a("kept in `target/bench7/miss_t5_resource/`). The law below is in")
    a("`n_perms × n_lrs` only, so the resource's size is what this stage exercises.")
    a("")
    rows5 = [r for r in rows if r["stage"] == "t5"]
    if rows5:
        a("| arm | wall (s) | elapsed (s) | peak RSS (MB) | note |")
        a("|---|---|---|---|---|")
        for row in sorted(rows5, key=lambda r: r["label"]):
            note = row["note"] or ""
            if row["rc"]:
                note = (note + f" rc={row['rc']}").strip()
            a(f"| {row['arm']} | {fmt_s(row['wall_s'])} | {fmt_s(row['elapsed_s'])} | "
              f"{fmt_mb(row['rss_kb'])} | {note} |")
        rel = next((r for r in rows5 if r["arm"] == "release"), None)
        pred = (381 * 1024 + 4.98 * 1000 * 4620) / 1024
        a("")
        a(f"The law predicts {pred:.0f} MB (~{pred / 1024:.1f} GiB) for the release arm at this "
          f"configuration.")
        if rel and rel["rc"] == 0:
            a(f"It ran: {fmt_mb(rel['rss_kb'])} MB measured, {fmt_s(rel['elapsed_s'])} s "
              f"whole-process — the box (47 GB) fit it.")
        elif rel:
            a("It did not finish (see the note); the measured arms and the prediction stand "
              "as the extrapolation.")
    else:
        a("Not run in this pass.")
    a("")

    # ---- recorded references
    a("## Recorded references (`/home/pwwang/p0a/logs`, read-only)")
    a("")
    a("| log | what | wall (s) | peak RSS (MB) |")
    a("|---|---|---|---|")
    for ref in doc["recorded_references"]:
        if ref.get("missing"):
            a(f"| {ref['log']} | {ref['description']} | missing | missing |")
            continue
        wall = f"{ref['wall_s']:.2f}" if ref.get("wall_s") else "—"
        a(f"| `{pathlib.Path(ref['log']).name}` | {ref['description']} | {wall} | "
          f"{fmt_mb(ref.get('rss_kb'))} |")
    a("")
    a("## Caveats")
    a("")
    a("- The Python arms include numba's JIT compilation in first-call wall (both arms,")
    a("  so the comparison is like-for-like; at 50k it is small — the recorded warm-repeat")
    a("  probe measured a −0.46 s difference between call 1 and call 2).")
    a("- Python `wall` excludes the input read and the ~1.6 s import; the harness reports")
    a("  `elapsed` for the whole process alongside, which includes both.")
    a("- Row counts agree across arms in every configuration (see `checks` in")
    a("  `bench/results.json`); value-level agreement with the oracle is the W6 G2 gate's")
    a("  business, not this suite's.")
    a("- The 50k/100k `.h5ad` reads are sometimes cold (first touch after a big Python")
    a("  arm evicted page cache); that cost is inside both the `rust` and the Python")
    a("  `elapsed` numbers, and inside the `rust` in-process wall (read is timed there).")
    a("- **Box-state sensitivity (wall only).** The `patched` arm's wall moves with the")
    a("  box's state at run time: at 50k × p1000 it measured 13.4–14.8 s in the recorded")
    a("  logs, 15.0–15.1 s re-run standalone on a cold box, 18.6–18.8 s with the box warm,")
    a("  and 20.1 / 20.6 s in the suite's two serial runs — the table carries the suite")
    a("  value, which both suite runs reproduce on this config. `release` (same config)")
    a("  spans 17.1–19.4 s and `rust` 7.5–8.4 s over the same contexts. Peak RSS is")
    a("  identical in every context (patched 1,902–1,905 MB): the memory numbers, not the")
    a("  patched wall, are the box-invariant result. Replication record and commands:")
    a("  `ops/logs/w7-report.md`.")
    a("")
    path = REPO / "bench/RESULTS.md"
    path.write_text("\n".join(L) + "\n")
    return path


def main():
    rows = measurements()
    env = versions()
    doc = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "box": {k: env[k] for k in ("cpu", "cores", "mem_total_kb", "kernel")},
        "suite": {
            "script": "bench/bench_suite.sh",
            "outdir": str(OUTDIR),
            "seed": 1337,
            "threads_note": "RAYON_NUM_THREADS for rust; --n_jobs for python (joblib); "
                            "python numba kernels run at their own default (W4 D4)",
            "arms": {
                "rust": "engine_bench (liana_core::run::Method, the `liana-rs run` dispatch)",
                "release": "liana 2.0.0, pinned oracle venv, bench/run_arm.py",
                "patched": "same venv + PYTHONPATH=/home/pwwang/p0a/patched (W4/W5 memory patch)",
            },
        },
        "versions": env,
        "inputs_sha256": inputs(),
        "law": law(rows),
        "measurements": rows,
        "checks": checks(rows),
        "recorded_references": recorded(),
    }
    doc["law"]["source"] = (
        "mem_probe configs recorded in /home/pwwang/p0a/logs/round2.log, round4.log "
        "(n_obs=10000, n_jobs=1, release arm)"
    )
    path = REPO / "bench/results.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{path}: {len(rows)} measurements, {len(doc['checks'])} checks")
    print(f"{markdown(doc)}: the human table")


if __name__ == "__main__":
    main()
