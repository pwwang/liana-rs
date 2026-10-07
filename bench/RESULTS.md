# W7 benchmark results — liana-rs vs liana 2.0.0 vs patched-Python

The frozen numbers behind the paper's figures. Every row was measured on one
box — 13th Gen Intel(R) Core(TM) i9-13900, 32 vCPU, 47 GB RAM, kernel 6.18.33.2-microsoft-standard-WSL2 — in serial
passes of `bench/bench_suite.sh` on 2026-10-06, seed 1337: stages
t1–t4 in one run; t5 re-run once (each stage rewrites its own TSV) after its
4,620-LR resource was rebuilt — the failed first attempt is kept in
`target/bench7/miss_t5_resource/`. The
machine-readable twin of this file is `bench/results.json`: every configuration
with its exact command, wall, peak RSS, thread counts, versions and input
sha256s. `bench/RESULTS.md` is generated from it (`bench/collect_results.py`),
so the two cannot disagree.

| arm | what runs |
|---|---|
| `rust` | `bench/engine_bench` — `liana_core::run::Method`, the `liana-rs run` CLI's own dispatch |
| `release` | liana 2.0.0 (`V2.0.0`, commit `c59472ccc9de`) in the pinned oracle venv |
| `patched` | the same venv with `PYTHONPATH=/home/pwwang/p0a/patched` (W4/W5 memory patch) |

**Wall bases.** `wall` is the in-process measurement: for `rust` it covers
read + resource + method (`engine-bench`'s own clock); for the Python arms it is
`run_arm.py`'s `WALL_S=`, the method call only (inputs read before `t0`, so it
flatters Python by that read). `elapsed` is `/usr/bin/time -v`'s whole-process
wall — on the Python side that includes interpreter start and the ~1.6 s import
of the scientific stack. Peak RSS is `time -v`'s "Maximum resident set size"
(the Rust bin cross-checks it with `VmHWM`; they agree to the kB).

**Threads.** `rust` runs on `RAYON_NUM_THREADS`; the Python arms run `n_jobs=4`
(joblib) with their numba kernels at numba's own default — `NUMBA_THREADS=32` in
every Python run recorded here (W4 D4), so the 4-thread comparison is generous
to Python, not to Rust. One machine-wide caveat: this WSL2 box occasionally
stalls a process for seconds (two rows below show a `time -v` `elapsed` several
seconds above the in-process wall; the same stall hit a `python -c` startup rep
at 5.26 s vs 1.6 s). In-process walls are the stable numbers; `elapsed` is
reported as measured.

## T1 — `rank_aggregate` end to end, 4 threads

Wall, seconds (in-process basis):

| dataset | perms | rust | release | patched | rust vs release | rust vs patched |
|---|---|---|---|---|---|---|
| 10k | 100 | 2.07 | 5.17 | 5.07 | 2.5× | 2.4× |
| 10k | 1000 | 2.82 | 13.08 | 14.16 | 4.6× | 5.0× |
| 50k | 100 | 3.19 | 7.90 | 8.04 | 2.5× | 2.5× |
| 50k | 1000 | 8.44 | 19.37 | 20.57 | 2.3× | 2.4× |
| 100k | 100 | 4.42 | 12.06 | 11.31 | 2.7× | 2.6× |
| 100k | 1000 | 16.22 | 32.42 | 30.26 | 2.0× | 1.9× |

Peak RSS, MB:

| dataset | perms | rust | release | patched | release / rust | patched / rust |
|---|---|---|---|---|---|---|
| 10k | 100 | 293 | 1622 | 1064 | 5.5× | 3.6× |
| 10k | 1000 | 293 | 10336 | 1178 | 35.2× | 4.0× |
| 50k | 100 | 419 | 2351 | 1812 | 5.6× | 4.3× |
| 50k | 1000 | 419 | 11042 | 1902 | 26.4× | 4.5× |
| 100k | 100 | 577 | 3245 | 2809 | 5.6× | 4.9× |
| 100k | 1000 | 576 | 11948 | 2809 | 20.7× | 4.9× |

Cross-check through the CLI binary (`liana-rs run`, 50k × p1000, 4 threads): 200000 rows, 11.88 s whole-process, 422 MB peak — the same dispatch the harness's `rust`
rows use, driven through the shipped binary.

The `patched` arm's wall is this table's least box-state-stable cell — see
*Box-state sensitivity* under Caveats before quoting it.

## T2 — the memory law: peak RSS vs `n_perms` × `n_lrs` at 50k

Recorded law (release arm, `n_obs=10k`, fitted from the logged mem probes):
`peak ≈ 381 MB + 4.98 KB × n_perms × n_lrs`. Measured today at 50k, peak RSS in MB:

| n_perms | arm | n_lrs=200 | n_lrs=1000 | n_lrs=2000 | law @2000 (release) |
|---|---|---|---|---|---|
| 10 | rust | 253 | 294 | 419 | 478 |
| 10 | release | 782 | 1364 | 1599 | 478 |
| 10 | patched | 782 | 1365 | 1599 | 478 |
| 100 | rust | 253 | 293 | 419 | 1354 |
| 100 | release | 782 | 1625 | 2352 | 1354 |
| 100 | patched | 782 | 1422 | 1812 | 1354 |
| 1000 | rust | 253 | 294 | 419 | 10108 |
| 1000 | release | 1641 | 5985 | 11041 | 10108 |
| 1000 | patched | 1096 | 1627 | 1900 | 10108 |

At fixed resource size the rust arm's peak is flat in `n_perms` (across p10/p100/p1000 — 200 LRs: 253–253 MB (0.1%), 1000 LRs: 293–294 MB (0.2%), 2000 LRs: 419–419 MB (0.1%)); the permutation null
adds nothing measurable. The whole grid spans 253–419 MB, and
the growth is the resource side — the LR table the method carries — not memory
of the perms.

The release arm's own fit at 50k (least squares over the nine points, the law's
shape) gives slope **5.10 KB per (perm × LR)** — the
recorded 4.98 KB holds — with intercept **1054 MB**, larger than
the recorded 381 MB because the intercept carries the dataset (50k's matrix is
bigger than 10k's). Anchors at 10k (release arm — where the law was fitted):

| config | measured MB | law MB | residual MB |
|---|---|---|---|
| p10 × lrs2000 | 830 | 478 | +352 |
| p100 × lrs2000 | 1628 | 1354 | +275 |
| p1000 × lrs2000 | 10337 | 10108 | +230 |
| p1000 × lrs200 | 1432 | 1354 | +79 |

## T3 — thread scaling (`rank_aggregate`, 50k × 1000)

| threads | rust wall (s) | peak RSS (MB) | speedup vs 1 |
|---|---|---|---|
| 1 | 22.43 | 419 | 1.0× |
| 4 | 7.94 | 419 | 2.8× |
| 8 | 6.09 | 420 | 3.7× |
| 32 | 4.55 | 418 | 4.9× |

Recorded Python references from `/home/pwwang/p0a/logs` (joblib `n_jobs`, numba at
32 threads) are in the table at the end of this file.

## T4 — startup and first run

| what | wall | peak RSS (MB) |
|---|---|---|
| `liana-rs --version` (mean of 20, bash clock) | 0.6 ms | 3 |
| `liana-rs run` first run — connectome, 1k cells × 2000 LRs, 186513 rows | 1.75 s (whole process) | 206 |
| `python -c "import liana"` (oracle venv) | best 1.66 s, median 1.74 s of 5 | 345 |

The recorded baseline for the import was **1.78 s** (`p0a/logs/startup.log`); the
binary starts in under a millisecond.
A whole 1k-cell method run (1.75 s) takes more wall clock
than one Python import of the stack (1.66 s).

## T5 — a 4,620-LR resource (the consensus resource's size), 50k × 1000

The resource is drawn from this synthetic data's gene universe with the paper's
own `_sample_resource` recipe — the same one that built `resource_{n_obs}.csv` —
at the consensus resource's 4,620-pair size. The literal
`select_resource('consensus')` symbols cannot run here: they do not exist in the
synthetic `Gene{i}` var_names, and all three arms reject that dump (rc=101/1,
kept in `target/bench7/miss_t5_resource/`). The law below is in
`n_perms × n_lrs` only, so the resource's size is what this stage exercises.

| arm | wall (s) | elapsed (s) | peak RSS (MB) | note |
|---|---|---|---|---|
| patched | 30.73 | 32.62 | 2122 | wall=WALL_S (method call only) |
| release | 40.35 | 42.36 | 23747 | wall=WALL_S (method call only) |
| rust | 10.60 | 10.62 | 741 |  |

The law predicts 22849 MB (~22.3 GiB) for the release arm at this configuration.
It ran: 23747 MB measured, 42.36 s whole-process — the box (47 GB) fit it.

## Recorded references (`/home/pwwang/p0a/logs`, read-only)

| log | what | wall (s) | peak RSS (MB) |
|---|---|---|---|
| `mc1_release50k_j4.log` | release rank_aggregate 50k p1000 j4 | 18.86 | 11043 |
| `mc1_patched50k_j4.log` | patched rank_aggregate 50k p1000 j4 | 13.42 | 1903 |
| `ra10k_j4_base2.log` | release rank_aggregate 10k p1000 j4 | 13.38 | 10336 |
| `ra10k_j4_patched.log` | patched rank_aggregate 10k p1000 j4 | 8.87 | 1181 |
| `ra50k_j4_base2.log` | release rank_aggregate 50k p1000 j4 (repeat) | 19.39 | 11045 |
| `ra50k_j4_patched.log` | patched rank_aggregate 50k p1000 j4 (repeat) | 14.83 | 1902 |
| `ra50k_j4_p100_base.log` | release rank_aggregate 50k p100 j4 | 6.99 | 2339 |
| `ra50k_j4_p100_patched.log` | patched rank_aggregate 50k p100 j4 | 10.35 | 1811 |
| `ra50k_j4_p2000_base.log` | release rank_aggregate 50k p2000 j4 | 32.03 | 20708 |
| `ra50k_j4_p2000_patched.log` | patched rank_aggregate 50k p2000 j4 | 22.97 | 2035 |
| `ra100k_j4.log` | release rank_aggregate 100k p1000 j4 | 31.88 | 11949 |
| `ra50k_j1.log` | release rank_aggregate 50k p1000 j1 | 31.57 | 11040 |
| `ra50k_j16.log` | release rank_aggregate 50k p1000 j16 | 17.09 | 11040 |
| `startup.log` | python startup / import timings | — | — |

## Caveats

- The Python arms include numba's JIT compilation in first-call wall (both arms,
  so the comparison is like-for-like; at 50k it is small — the recorded warm-repeat
  probe measured a −0.46 s difference between call 1 and call 2).
- Python `wall` excludes the input read and the ~1.6 s import; the harness reports
  `elapsed` for the whole process alongside, which includes both.
- Row counts agree across arms in every configuration (see `checks` in
  `bench/results.json`); value-level agreement with the oracle is the W6 G2 gate's
  business, not this suite's.
- The 50k/100k `.h5ad` reads are sometimes cold (first touch after a big Python
  arm evicted page cache); that cost is inside both the `rust` and the Python
  `elapsed` numbers, and inside the `rust` in-process wall (read is timed there).
- **Box-state sensitivity (wall only).** The `patched` arm's wall moves with the
  box's state at run time: at 50k × p1000 it measured 13.4–14.8 s in the recorded
  logs, 15.0–15.1 s re-run standalone on a cold box, 18.6–18.8 s with the box warm,
  and 20.1 / 20.6 s in the suite's two serial runs — the table carries the suite
  value, which both suite runs reproduce on this config. `release` (same config)
  spans 17.1–19.4 s and `rust` 7.5–8.4 s over the same contexts. Peak RSS is
  identical in every context (patched 1,902–1,905 MB): the memory numbers, not the
  patched wall, are the box-invariant result. Replication record and commands:
  `ops/logs/w7-report.md`.

