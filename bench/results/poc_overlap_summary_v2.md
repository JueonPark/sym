# Nsight Systems overlap-trace evidence v2 (issue #67, D3 re-measurement, criterion 1)

Supersedes (does not edit) `poc_overlap_summary.md`. That v1 capture was taken
against a `-O0` build (`build/cuda`) with 256 MiB chunks, before D1 (parallel
per-chunk gather) and D2 (64 MiB chunk shrink). This v2 capture is taken
against the **Release build** (`build/cuda-release`, Task 1 of this campaign)
with the current chunking (64 MiB chunks, `n_streams=2`), to answer: has the
pipeline's bottleneck flipped from CPU-gather-bound to PCIe-copy-bound?

## Environment

- `NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0` (`nsys --version`)
- Driver `595.79`, GPU `NVIDIA GeForce RTX 4070 Ti SUPER` (`nvidia-smi --query-gpu=name,driver_version --format=csv`)
- Binary: `build/cuda-release/bench/bench-poc-transpose` (Release/optimized, per Task 1 — **not** the `-O0` tree v1 used)

## Capture command

```bash
/usr/local/cuda/bin/nsys profile -t cuda,osrt --force-overwrite true \
  -o bench/results/poc_overlap_v2 \
  build/cuda-release/bench/bench-poc-transpose --n 16384 --warmup 2 --iters 5 --reruns 1 --json -
```

`nsys` attached and captured cleanly under WSL2 on the first attempt — no
retry needed.

Driver JSON (same invocation, captured from stdout):

```
poc_transpose N=16384: ours 113.42 ms (spread 0.00%), baseline 47.81 ms wall / 43.37 ms gpu (spread 0.00%) -> speedup 0.42x
{
  "config": {"benchmark": "poc_transpose", "plan": "reference", "N": 16384, "dtype": "f32", "bytes": 1073741824, "n_buffers": 2, "n_streams": 2, "chunk_bytes": 67108864, "warmup": 2, "iters": 5, "reruns": 1, "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER", "verified": false},
  "methods": {
    "ours_pipeline_2buf_2stream": {"wall_ms": {"reruns": [{"median": 113.424, "q1": 109.943, "q3": 115.429, "iqr": 5.48572, "n": 5}], "median_spread_pct": 0}},
    "baseline_pinned_memcpy_naive_kernel": {"wall_ms": {"reruns": [{"median": 47.806, "q1": 47.3098, "q3": 47.8116, "iqr": 0.501785, "n": 5}], "median_spread_pct": 0}, "gpu_ms": {"reruns": [{"median": 43.3735, "q1": 42.9172, "q3": 43.4078, "iqr": 0.490623, "n": 5}], "median_spread_pct": 0}}
  },
  "speedup_wall_median": 0.421481
}
```

`speedup_wall_median = 0.421481` — consistent with the 0.43x figure already
recorded from Tasks 1-3; the pipelined method is still slower than baseline
wall-clock at N=16384 on this Release tree.

## CSV extraction

```bash
/usr/local/cuda/bin/nsys stats --report cuda_gpu_trace --format csv \
  --output bench/results/poc_overlap_v2 bench/results/poc_overlap_v2.nsys-rep
```

Header and H2D-row naming match v1's documented quirk: `Start (ns), Duration
(ns), ..., Strm, Name`, H2D rows named `[CUDA memcpy Host-to-Device]`
(filtered on lowercase substring `host-to-device`, not `HtoD`).

## Analysis script

`bench/analyze_overlap.py` (committed with this doc) formalizes the ad-hoc
v1 analysis into a reusable CLI:

```bash
python3 bench/analyze_overlap.py bench/results/poc_overlap_v2_cuda_gpu_trace.csv \
  --warmup-iters 2 --timed-iters 5
```

Output (verbatim):

```
inferred chunks/iteration: 16 (streams: pipeline 13,14; baseline 15)
pipeline (timed): 80 H2D copies, busy 220.06 ms over span 561.25 ms -> utilization 39.2%
chunk-issue period ms: mean 7.070, median 6.909, min 6.480, max 8.979
copy duration ms: mean 2.751, median 2.748
per-iteration wall ms (first-chunk-start deltas): 109.26, 119.41, 115.05, 113.52
baseline (timed): 5 H2D copies, busy 219.95 ms over span 234.27 ms -> utilization 93.9%
```

Shape matches the pre-registered expectation exactly: raw per-stream H2D
counts are 56 + 56 (streams 13/14, pipeline) = 112 = 7 iters × 16 chunks of
64 MiB, and 7 (stream 15, baseline) = 7 iters × 1 whole-tensor copy; 80 of
the 112 pipeline copies are the 5 timed iterations after 2 warmup iterations
are excluded.

## Computed numbers

- **Pipeline copy-engine utilization (timed): 39.2%** (220.06 ms busy / 561.25 ms span) — up from v1's 25.6%, but well short of the ≳80% "near-continuous" bar.
- **Chunk-issue period: mean 7.070 ms** (median 6.909, range 6.480–8.979 ms) — this is the producer cadence, i.e. the CPU-gather-dominated time between successive chunk H2D issues.
- **Copy duration: mean 2.751 ms** (median 2.748 ms) — each 64 MiB H2D copy is fast and consistent.
- Ratio period/copy = 7.070 / 2.751 ≈ **2.57×** — the gather is still more than double the length of the copy it's meant to hide. (For contrast, v1's ratio was ~43/10.6 ≈ 4.06×.)
- Baseline phase utilization 93.9% (span≈busy, single contiguous whole-tensor copy per iteration — included for contrast only, not overlap evidence).

## Criterion 1 verdict rule (pre-registered, quoted verbatim from the brief)

> PASS iff (a) pipeline timed utilization is near-continuous (≳80%, vs 25.6%
> in v1) AND (b) mean copy duration ≥ mean chunk-issue-period − copy-duration
> slack, i.e. the copy — not the gather — is the long pole (in v1 gather was
> ~43 ms vs ~10.6 ms copies; the flip means the period collapses to ≈ the
> copy duration). If either fails → STOP after committing the summary:
> criterion 1 failed, no headline is quoted, #67 goes back to design.

Applying the rule to the v2 numbers:

- **(a) Utilization ≳80%?** 39.2% measured. **FAILS** (39.2% < 80%, though
  a real improvement over v1's 25.6%).
- **(b) Copy is the long pole (mean copy duration ≥ mean period)?** 2.751 ms
  (copy) vs 7.070 ms (period). **FAILS** — the period has not collapsed to
  the copy duration; a ~4.3 ms/chunk gap remains, meaning the CPU gather is
  still the longer of the two and the copy is still fully hidden underneath
  it, not the reverse.

Both (a) and (b) fail.

## Criterion 1: **FAIL**

No headline speedup is quoted from this trace. The bottleneck has **not**
flipped from CPU-gather-bound to PCIe-copy-bound. Per the pre-registered
rule, #67 goes back to design.

## Interpretation

D1 (parallel per-chunk gather) and D2 (64 MiB chunk shrink, down from 256
MiB) together produced a large, real improvement in absolute terms: the
chunk-issue period dropped from v1's ~43 ms to ~7.07 ms (≈6.1× faster), the
per-chunk copy dropped from ~10.6 ms to ~2.75 ms (≈3.9× faster, tracking the
4× smaller chunk size, as expected for a bandwidth-bound copy), and
copy-engine utilization roughly-tripled-in-relative-terms from 25.6% to
39.2%. Recall v1's numbers were measured on a `-O0` build; part of the
absolute-time improvement here also reflects the switch to the Release tree,
not D1/D2 alone, though the utilization and period/copy-ratio metrics (being
ratios of durations already on the same hardware and stream architecture)
still speak directly to whether the *design* — not just the *build flags* —
closed the gap.

It did not close it far enough. The gather/copy ratio improved from ~4.06×
in v1 to ~2.57× in v2 — real progress, and directionally exactly what
D1+D2 were meant to produce — but the CPU gather is still comfortably the
longer pole of the two, and utilization remains under half of the ≳80%
near-continuous bar the rule requires. This is mechanistically consistent
with, and explains, the still-sub-1x `speedup_wall_median = 0.421481` at
N=16384 (matching the 0.43x already recorded in Tasks 1-3): the pipeline is
still throttled by single-threaded-per-chunk CPU gather throughput, not by
PCIe bandwidth, so double-buffering the H2D copies has diminishing returns
until the gather itself is faster, or gather and copy are truly comparable
in duration.

Per the global constraint, this verdict is committed as-is: criterion 1
**FAILS**, and issue #67 goes back to design rather than proceeding to a
headline "bottleneck flipped" claim.

## Local artifacts (not committed)

`bench/results/poc_overlap_v2.nsys-rep`, `bench/results/poc_overlap_v2.sqlite`,
`bench/results/poc_overlap_v2_cuda_gpu_trace.csv` — same as v1, binary
capture artifacts are not committed to the repo.
