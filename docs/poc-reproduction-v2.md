# PoC re-measurement v2: 32768² fp32 blocked transpose — criterion-1 failure report (issue #67)

**Acceptance-criteria verdicts (issue #67, pre-registered, priority order):**

1. **Criterion 1 (bottleneck flip): FAIL — the pipeline remains gather-bound.**
   The Nsight trace shows H2D copy-engine utilization 39.2% (up from v1's
   25.6%, but well short of the ≳80% "near-continuous" bar), and the per-chunk
   gather period (mean 7.07 ms) is still ~2.6× longer than the copy it is meant
   to hide (mean 2.75 ms). The copies are still fully hidden behind the CPU
   gather, not the reverse. Trace: `bench/results/poc_overlap_summary_v2.md`.
2. **Criterion 2 (gather scaling measured): reported.** At N = 32768,
   single-thread 13.0–13.3 GB/s vs 16-thread 14.3–14.5 GB/s — ratio ≈ 1.10×.
   Feeds E2/P3; see Gather scaling below.
3. **Criterion 3 (≥ 1× vs baseline): FAIL, ~0.48×.** Every rerun median of ours
   is ~2× the baseline wall, so the sub-1× verdict is unambiguous even though
   the run was unstable (see below). Ours is roughly 2× *slower* than the
   baseline, not faster.

**Because criterion 1 failed, no speedup number is blessed as "the" v2
headline.** Per issue #67's pre-registration, a criterion-1 failure means the
speedup is not quoted as a headline and the design (#67) goes back to the
board; #63 is **not** closed. The pre-registered "genuine crossover / E4–E5
input" framing does **not** apply here — that clause required criteria 1 and 2
to hold, and criterion 1 did not.

**This supersedes `docs/poc-reproduction.md` (0.36×, as measured at `-O0`) as
the status of record.** It does not edit it; the v1 doc and its JSON are left
untouched for archaeology. The v1↔v2 comparison spans a build-config change
(`-O0` → Release) and is **not** attributable to D1+D2 alone — see below.

## What changed since v1

- **D1 — parallel per-chunk gather** (#65, merged via #68): the pipeline's
  per-chunk gather now runs on a persistent worker pool rather than one CPU
  thread. This benchmark ran with `threads_multi = 16`.
- **D2 — chunk-size heuristic** (#66, merged via #69): the default chunk shrank
  from 256 MiB to 64 MiB at N = 32768. Confirmed live in every v2 JSON:
  `chunk_bytes` is 4 / 16 / 64 / 64 MiB at N = 4096 / 8192 / 16384 / 32768
  (v1's N = 32768 run used 256 MiB).
- **Release build** (the `-O0` discovery, reported plainly): the v1 tree
  (`build/cuda`) had an empty `CMAKE_BUILD_TYPE`, so all host C++ — including
  the gather — was compiled at `-O0`. The v1 0.36× headline and D1's committed
  `gather_bw_n4096.json` were both measured under that tree, which biased every
  CPU-vs-GPU comparison against the CPU path (the baseline's cost is mostly
  GPU-side work nvcc optimizes regardless of host flags). All v2 numbers come
  from a fresh `build/cuda-release` tree (`-DCMAKE_BUILD_TYPE=Release`, `-O3`),
  with **both** methods re-measured under it, so the v2 ratios are internally
  consistent. The old `build/cuda` tree was left in place, untouched.

Protocol unchanged from v1: same flags, 10 warmup + 50 timed iterations,
median + IQR, 3 reruns, `< 5%` rerun-spread stability bar, byte-exact
`--verify` in the same invocation, same frozen "reference" golden plan (4D
blocked view + `transpose(0,1)`, `divisible(N, 64)`).

## Numbers

### Headline config, N = 32768, fp32 (4 GiB tensor) — unstable, reported as-is

| method | rerun 1 median (ms) | rerun 2 | rerun 3 | IQR trend (ms) | rerun spread |
|---|---|---|---|---|---|
| ours (pipeline, 2 buf × 2 streams) | 419.4 | 393.8 | 392.6 | 28.9 → 15.8 → 7.6 | 6.83% |
| baseline (pinned memcpy + kernel) | 188.5 | 188.5 | 188.5 | 0.65 / 0.30 / 0.36 | 0.03% |

`speedup_wall_median = 0.479` (median-of-medians, `188.5 / 393.8`). Raw:
`bench/results/poc_transpose_n32768_v2.json`.

**The ours run at N = 32768 did not clear the 5% stability bar.** After the one
protocol-permitted re-run, the three rerun medians were 419.4 / 393.8 /
392.6 ms — a 6.83% spread, above the 5% bar. The pattern is a systematic
first-rerun warm-up outlier: the IQR tightens monotonically across reruns
(28.9 → 15.8 → 7.6 ms), the first rerun median sits high and the later two
agree closely, and this shape reproduced in both invocations (the headline run
and the Nsight capture). It is absent in the baseline (188.5 ms, 0.03% spread)
and absent at every smaller N (spreads ≤ 2.9%). Because of this, **no single
ours median is blessed as "the" v2 number.** The criterion-3 verdict is
nevertheless unambiguous: every rerun median is ~2× the baseline, so the
speedup is < 1× (~0.48×) regardless of which median one picks.

### Size-scaling sweep (v2, Release) vs the v1 (`-O0`) results

| N | tensor bytes | chunk_bytes | ours median (ms) | baseline median (ms) | speedup v2 (Release) | v1 (`-O0`) |
|---|---|---|---|---|---|---|
| 4096 | 64 MiB | 4 MiB | 13.6 | 3.21 | 0.236 | 0.13–0.23× (smoke)† |
| 8192 | 256 MiB | 16 MiB | 34.4 | 12.0 | 0.350 | 0.13–0.23× (smoke)† |
| 16384 | 1 GiB | 64 MiB | 111.4 | 47.5 | 0.426 | — (not run in v1) |
| 32768 | 4 GiB | 64 MiB | 393.8 | 188.5 | 0.479 | 0.362× (committed)‡ |

Speedup rises monotonically with N (0.236 → 0.350 → 0.426 → 0.479) but never
crosses 1×. † The v1 doc reported N = 4096/8192 as 0.13–0.23× smokes under
`-O0`; those were not committed as per-size JSONs, so no exact v1 number exists
for direct contrast at those sizes. ‡ Only N = 32768 has a committed v1 JSON
(`bench/results/poc_transpose_n32768.json`: ours 499.31 / baseline 180.95 →
0.362×, at 256 MiB chunks, `-O0`). The v2-vs-v1 delta at N = 32768 (0.362 →
0.479) spans the `-O0` → Release build change and D1+D2 together, and is not
decomposable into a single cause from these numbers alone.

Raw data: `bench/results/poc_transpose_n{4096,8192,16384,32768}_v2.json`.

## Overlap trace (criterion 1 evidence)

Captured at N = 16384 on the Release tree (full detail and the pre-registered
verdict rule in `bench/results/poc_overlap_summary_v2.md`):

- H2D copy-engine utilization (timed iterations): **39.2%** (v1: 25.6%) — a
  real improvement, but under half the ≳80% near-continuous bar.
- Chunk-issue period: mean **7.07 ms** (the CPU-gather-dominated producer
  cadence). Per-chunk copy duration: mean **2.75 ms**. Ratio ≈ **2.57×** (v1:
  ~4.06×) — the gather is still comfortably the long pole.
- Effective in-pipeline gather rate ≈ **9.5 GB/s** (64 MiB / 7.07 ms);
  per-chunk copy ≈ 24 GB/s (64 MiB / 2.75 ms), roughly the PCIe rate.

Interpretation (grounded in the numbers above):

- **The gather is host-memory-bandwidth-bound at Release.** 16 threads buy only
  1.10× over 1 thread (see below), so D1's parallelism cannot close the gap to
  the ~24 GB/s PCIe copy rate the pipeline would need to become copy-bound.
- **The `-O0` build had made the gather look compute-bound.** v1's committed
  `gather_bw_n4096.json` (`-O0`) showed 2.04 → 6.5 GB/s, a 3.2× gain from 32
  threads — the kind of scaling a compute-bound workload shows, and the premise
  D1's design rested on. At Release the optimizer delivered the ~6× that
  thread-parallelism was supposed to deliver, and the remaining wall is memory
  bandwidth, which more threads do not move.
- **The in-pipeline effective gather (~9.5 GB/s) is below the pure-CPU bench
  (14.3 GB/s).** Concurrent H2D DMA traffic competes with the gather for the
  same host memory bandwidth, so the gather runs slower inside the live
  pipeline than it does in isolation.

This is why the pipeline stays sub-1×: double-buffering the H2D copies has
diminishing returns while the gather feeding them is the slower stage, and the
gather is already at the host memory-bandwidth wall.

## Gather scaling (criterion 2; feeds E2/P3, supersedes the `-O0` bench)

`bench-gather-bw` on the reference plan at N = 32768 (`min_rows_per_worker = 1`),
Release tree:

| config | GB/s (3 reruns) | wall median (ms) | rerun spread |
|---|---|---|---|
| 1 thread | 13.03 / 13.11 / 13.29 | 323–330 | 2.02% |
| 16 threads | 14.33 / 14.49 / 14.51 | 296–300 | 1.25% |

Multi-thread speedup ≈ **1.10×**. Both configs cleared the 5% stability bar.
This is the memory-bandwidth ceiling that explains criterion 1's failure: with
only 1.10× headroom from parallelism, the gather cannot be sped up to the copy
rate by adding threads.

These v2 numbers supersede D1's committed `gather_bw_n4096.json` for E2/P3
purposes: that file was an `-O0` measurement (N = 4096, `threads_multi = 32`,
2.04 → 6.5 GB/s) and its apparent 3.2× thread scaling was a `-O0` artifact, not
a Release-build property. Note also the thread-count discrepancy — v1's bench
used `threads_multi = 32`, v2 reports `threads_multi = 16`; recorded here as an
environment observation, not interpreted further.

Raw data: `bench/results/gather_bw_n32768_v2.json`.

## Honest-reporting notes

- **Instability, reported explicitly.** The N = 32768 ours run did not clear
  the < 5% rerun-spread bar: 6.83% after the one protocol-permitted re-run
  (medians 419.4 / 393.8 / 392.6 ms). Per the failure-handling rule, the
  instability is reported rather than a single number blessed. It is a
  systematic first-rerun warm-up outlier (monotonic IQR tightening
  28.9 → 15.8 → 7.6 ms; the shape reproduced in both invocations; absent in the
  baseline at 0.03% and absent at all N ≤ 16384). The criterion-3 sub-1×
  verdict does not depend on the resolution of this instability — every rerun
  median is ~2× baseline.
- **No tuning.** Nothing was adjusted in response to any result: no chunk
  sizes, buffer counts, stream counts, thread counts, iteration counts, or
  method definitions were changed after seeing a number. The D2 heuristic and
  D1 pool were as merged in #69/#68; the instability was reported, not tuned
  away.
- **Verification.** Every committed transpose JSON is `"verified": true` —
  byte-exact `--verify` against the CPU reference passed in the same invocation
  for both methods at all four sizes. No correctness regression under Release.
- **WSL2 standing caveat.** All measurements were taken under WSL2
  (RTX 4070 Ti SUPER, CUDA 13.2). `nsys` attached cleanly on the first attempt.
  WSL2 is the standing environment caveat for anyone reproducing these numbers
  on bare-metal Linux, and is the most likely (unproven) suspect for the
  N = 32768 first-rerun warm-up instability.

## Bare-metal confirmation

*Pending: no bare-metal box was available at measurement time (2026-07-11). To
be appended if one becomes available; WSL2 remains the standing caveat until
then.*

## Raw-data pointers (all committed on `d3-remeasurement`)

- Transpose sweep: `bench/results/poc_transpose_n4096_v2.json`,
  `bench/results/poc_transpose_n8192_v2.json`,
  `bench/results/poc_transpose_n16384_v2.json`,
  `bench/results/poc_transpose_n32768_v2.json`
- Overlap trace + criterion-1 verdict: `bench/results/poc_overlap_summary_v2.md`
- Gather scaling: `bench/results/gather_bw_n32768_v2.json`
- Superseded v1 (untouched): `docs/poc-reproduction.md`,
  `bench/results/poc_transpose_n32768.json`,
  `bench/results/poc_overlap_summary.md`, `bench/results/gather_bw_n4096.json`
