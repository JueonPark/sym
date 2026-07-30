# D3: C7 re-measurement + result-doc supersession (issue #67)

**Date:** 2026-07-11
**Issue:** #67 — depends on D1 (#65, merged via #68) and D2 (#66, merged via #69).
**Base commit:** `28a0743` on `main` (contains D1 + D2). Working branch: `d3-remeasurement`.

## Goal

Replace the 0.36× headline (`docs/poc-reproduction.md`) with a measurement of
the revised design — D1 parallel per-chunk gather + D2 64 MiB chunk heuristic —
under the same frozen protocol, with the acceptance criteria pre-registered in
issue #67. The old doc and JSON are superseded, never edited.

## Acceptance criteria (from #67, priority order, all reported explicitly)

1. **Bottleneck flip verified** in the Nsight trace — pipeline is PCIe-bound,
   no longer gather-bound. If this fails: stop, no headline quoted, back to the
   design board.
2. **Gather scaling measured** — single- vs multi-thread GB/s reported (feeds
   E2/P3).
3. **≥ 1× vs baseline at the reference config** — reported as-is either way.
   If 1–2 hold but 3 fails, that is a genuine crossover result (this cell
   favors the GPU path) and becomes E4/E5 input with exactly that framing —
   **not** grounds for another tuning loop.

## Environment

WSL2 (RTX 4070 Ti SUPER, CUDA 13.2) is the standing environment caveat,
recorded in the doc. No bare-metal box is available now ("maybe later"): the
doc carries an explicitly marked empty **bare-metal confirmation** slot to be
appended if one becomes available.

### Build configuration (amended 2026-07-11)

Discovery during planning: the existing `build/cuda` tree has an empty
`CMAKE_BUILD_TYPE` — all host C++ (including the gather) was compiled at
`-O0` with `-D_DEBUG -UNDEBUG`. The C7 0.36× headline and D1's committed
`gather_bw_n4096.json` were measured under that tree, biasing every
CPU-vs-GPU comparison against the CPU path (the baseline's cost is mostly
GPU-side work nvcc optimizes regardless of host flags).

**Decision (user): Release only.** Treat `-O0` as a measurement bug. All D3
measurements run from a fresh `build/cuda-release` tree configured with
`-DCMAKE_BUILD_TYPE=Release`; both methods (ours *and* baseline) are re-run
under it, so the reported ratio is internally consistent. The v2 doc reports
the `-O0` discovery and the build-type change prominently: the v1 headline is
quoted as "0.36× as measured at `-O0`", and the v2 vs v1 comparison is
explicitly flagged as spanning a build-config change. The old `build/cuda`
tree is left untouched for archaeology.

## Execution plan

### 1. Preflight

- GPU idle, no other CUDA processes; RAM headroom ≥ 14 GiB (C7's floor).
- Fresh Release tree: `cmake -G Ninja -S . -B build/cuda-release
  -DCMAKE_BUILD_TYPE=Release
  -DMLIR_DIR=build/llvm-project/build/lib/cmake/mlir -DRELOC_ENABLE_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=89`; build `bench-poc-transpose`,
  `bench-gather-bw`, `libreloc-test`.
- Correctness gate under Release before any timing: `libreloc-test` passes
  (guards against optimization-revealed bugs, e.g. races the -O0 build hid).
- Confirm D2 defaults are live: output JSON's `chunk_bytes` must show 64 MiB
  at N = 32768 (not the old 256 MiB).

### 2. Headline run (criterion 3)

`bench-poc-transpose --n 32768 --verify`, frozen protocol: 10 warmup + 50
timed iterations, median + IQR, 3 re-runs, stability bar < 5% rerun spread,
byte-exact verification in the same invocation. Same flags as C7.
Output → `bench/results/poc_transpose_n32768_v2.json` (new file; the C7 JSON
stays untouched).

### 3. Size sweep

Same protocol and `--verify` at N ∈ {4096, 8192, 16384}, each →
`bench/results/poc_transpose_n<N>_v2.json`. Purpose: locate where the
crossover sits versus the Task-4 smokes (0.13–0.23× at N = 4096/8192 under the
old design), not just the N = 32768 endpoint. Reported as a size-scaling table
in the v2 doc. The headline remains N = 32768; the sweep is supporting data
and is not a tuning input.

### 4. Nsight capture at N = 16384 (criterion 1)

Same brief as #47's trace (`bench/results/poc_overlap_summary.md`):

```
nsys profile -t cuda,osrt --force-overwrite true -o bench/results/poc_overlap_v2 \
  build/cuda-release/bench/bench-poc-transpose --n 16384 --warmup 2 --iters 5 --reruns 1 --json -
```

CSV extraction via `nsys stats --report cuda_gpu_trace` with the
`host-to-device` name filter already documented in the v1 summary. Computed
evidence, timed iterations only:

- H2D copy-engine utilization over the pipeline phase (was 25.6%; "near-
  continuous" expected, ≳ 80% rule of thumb from #47's brief).
- Chunk-issue period vs per-chunk copy duration: copies (not gather) should be
  the long pole; per-chunk gather slack visible.

Verdict on criterion 1 comes from these two numbers together. Written up as
`bench/results/poc_overlap_summary_v2.md` (supersedes, does not edit, v1).
Binary artifacts (`.nsys-rep`, `.sqlite`, extracted CSV) stay local/uncommitted
as in C7.

### 5. Gather scaling (criterion 2)

`bench-gather-bw` single-thread vs worker-pool on the reference plan at the
largest N whose ~2× footprint fits host RAM (target N = 32768 ≈ 8 GiB; if it
doesn't fit, the largest that does, stated in the doc) →
`bench/results/gather_bw_n<N>_v2.json`. Reported as a GB/s pair
(1 thread vs T threads) for E2/P3. D1's committed `gather_bw_n4096.json` was
an `-O0` measurement; the v2 numbers supersede it for E2/P3 purposes.

### 6. `docs/poc-reproduction-v2.md`

Supersedes `docs/poc-reproduction.md` (v1 not edited). Contents:

- Headline result vs C7's 0.36×, with the three criteria verdicts in priority
  order, each explicit.
- What changed since v1: D1 parallel gather (#68), D2 64 MiB chunking (#69),
  and the Release build (v1 was `-O0`; see Build configuration above);
  bench protocol (flags, iteration counts, verification) unchanged.
- Size-scaling table (step 3) with the old small-N smokes for contrast.
- Gather-scaling GB/s pair (step 5).
- Honest-reporting notes in the v1 style: deviations (or none), stability
  spreads, WSL2 standing caveat, marked empty bare-metal slot.

### 7. Issue updates

Draft comments for #47 and #63 with the outcome; close #63 only if criteria
1–2 hold. **Drafts are shown to the user before anything is posted to
GitHub.** Doc + JSONs land via a PR from `d3-remeasurement` referencing #67.

## Non-goals

- No tuning: no chunk-size, buffer-count, stream-count, or protocol changes in
  response to any result (pre-registered in #67).
- No edits to v1 artifacts (`poc-reproduction.md`, `poc_overlap_summary.md`,
  C7 JSONs).
- No bare-metal run in this pass.

## Failure handling

- Criterion 1 fails → stop after step 4, report trace evidence, no headline
  quoted; #67 goes back to design discussion.
- Rerun spread ≥ 5% on any config → re-run once per protocol; if still
  unstable, report the instability explicitly instead of the number.
- `--verify` failure at any size → stop; that is a correctness regression,
  not a measurement result.
