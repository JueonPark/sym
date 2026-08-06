# BP1 — Method::BPipelined: overlap-fair Method B in the rtrack harness

**Issue**: #114 (BP1), parent tracking #108. **Date**: 2026-08-06.
**Deliverable shape**: regular PR on branch `bp1-bpipelined` — harness only, no measurement
artifacts (`bp_*` files are #BP3's); ~1–1.5 days.

## Context

R4 proved every Method-B transform kernel hides under the transfer (m ≪ ratio), yet the
committed `b_fair` baseline launches its kernel monolithically after the full transfer
(`runMethodBFair`: chunked DMA loop, then one `enqueueReceiveKernels` bracketed by a single
kBeg/kEnd pair). A reviewer combines those in one sentence. `B_pipelined` is the Method B a
competent implementation would ship: chunked `cudaMemcpyAsync` from the pinned source
(identical DMA path to `b_fair`) with the per-chunk receive/transform kernel issued
in-stream after each chunk's copy — the same shape Method A's recv loop already uses.

**The chunkability audit is recorded on issue #114** (comment, posted before coding, per
the issue's first task): T3/T5 chunk trivially (elementwise, precomputed `dInv`); T3R100's
kernel leg is N/A by construction (DMA-only, kept as the BP-G1 admissibility row);
blocked/nchw plans chunk via axis-group-aligned sub-plans with no kernel-class change;
transposePlan rows (T1, T2R100) chunk as dst column bands with a recorded perf-class
caveat (SMEM→naive fallback; hiding preserved, m≈2.97 ≪ ratio); **T2 is N/A — not
approximated** (its per-channel quantize leg cannot run per chunk with existing wrappers;
decision taken with the user). Alignment violations hard-fail, never approximate.

## §1 Harness (`bench/rtrack/rtrack_bench.cu`, plus `bench/rtrack/csv.h`)

- **Method enum**: `Method::BPipelined`, CSV tag `"b_pipelined"` (the name
  `cm4_registered_predictions.json`'s `placement_map` reserves for the Overlapped
  placement), CLI `--method bpipe`, included in `all` (and usage string). `b` (BStaged)
  and `b_fair` (BFair) are untouched — three baseline generations coexist for provenance.
- **Fixture predicates**: `needBPipe` joins `needBFair` in gating `pinnedSrc` allocation,
  and joins `needB || needBFair` for `dLin`/`dTmp`. `methodBDmaDst` applies unchanged
  (GpuStage::None lands directly in `dOut`).
- **`runMethodBPipelined(f, ck, pl)`** — per chunk: `h2dBeg[c]` → `cudaMemcpyAsync(
  methodBDmaDst(f)+off ← pinnedSrc+off)` → `h2dEnd[c]` → `kernBeg[c]` →
  `launchChunkKernel(f, chunk)` → `kernEnd[c]`, all on `pl.stream`; `evStart`/`evStop`
  around the loop. No staging buffers (pinned source read directly), hence no `c>=2`
  reuse gate. `t.cpu = 0`; `t.h2d = sumH2dMs()`; `t.kern = sumKernMs()`;
  `t.wall`/`t.gpu` as in the other methods. Stream ordering makes chunk c's kernel wait
  for chunk c's copy; `dLin` chunk regions are disjoint across c.
- **`Pipeline`**: gains a `withKern` flag allocating per-chunk `kernBeg/kernEnd` event
  vectors (size `nChunks`) and a `sumKernMs()` helper — structurally parallel to the
  A-only `recvBeg/recvEnd`/`sumRecvMs()`, kept separate because `gpu_recv_ms` is Method
  A's semantic. `b_pipelined` constructs with `allocStaging=false, withKern=true`.
- **Per-chunk kernel dispatch** — a bench-local helper (libreloc unmodified; "bounded
  change"): given the workload and the chunk's byte range, launch the audit's per-family
  form:
  - identityPlan (T3 Quantize, T5 ConvertF16): pointer + channel offsets
    (`quantizeF32S8(dLin+off, dOut+off, chunkRows, channelSize, dInv+rowBegin)`;
    `launchConvertF32F16(dLin+eOff, dOut+eOff, chunkElems)`). No sub-plan.
  - blockedTransposePlan (T1b, T1bR100, TW) / nchwToNhwcPlan (T4, T4R100): a locally
    constructed sub-`BoundPlan` (copy of `f.bound` with the sliced axis extent replaced)
    + shifted `dSrc`/`dDst` bases → `relocateF32`; T4's quantize leg chunks with
    `channels = chunkImages·…` and `dInv + chanBegin`.
  - transposePlan (T1, T2R100): the column-band sub-plan (`extents {n, j1-j0}`, shifted
    bases) → `relocateF32` (falls back to naive per the audit's recorded caveat).
  - GpuStage::None (T3R100): the kernel leg is skipped entirely — no `kernBeg/kernEnd`
    events recorded, `t.kern = 0` (a comment notes the row is DMA-only by construction;
    recording empty event pairs would only add launch noise to the h2d stream).
  - **Alignment guard**: if a chunk boundary is not aligned to the family's slab group
    (row: 4·n bytes; 64-row group / image: 256·n bytes), print
    `error: b_pipelined chunk misaligned for <id> …` and exit nonzero — never
    approximate. All committed (N, chunk) grids are aligned.
- **T2 N/A**: in `run()`'s method loop, `b_pipelined` on workload T2 prints
  `b_pipelined N/A for T2 (chunkability audit, issue #114)` to stderr and continues —
  an explicit, loud skip, never a silent omission and never an approximated hybrid.

## §2 CSV, verification, deliverable

- **CSV**: `gpu_kernel_ms` for `b_pipelined` = the per-chunk kernel sum (consistent with
  its existing "Method B kernel time" semantic; `b`/`b_fair` keep their monolithic
  kBeg→kEnd value). One new column appended after `wire` (name-based DictReader consumers
  unaffected): `h2d_occupancy` = `sumH2dMs / gpu_pipeline_ms` (both event-derived — the
  issue's "overlap occupancy (h2d-busy / wall) from events", the WSL2-compatible
  substitute for a trace; noted in a comment). Populated for every method (a and the two
  legacy B methods get it too — same formula, no special cases).
- **Verification**: the existing per-(workload × N × chunk × method) gate applies
  automatically — non-A methods memcmp against `f.ref` (the scalar-CPU Method-B
  artifact), a failure aborts the sweep. `b_pipelined` rides it with zero new machinery.
  Every runnable config must print `[verified]`.
- **Downstream (out of scope, recorded in the PR)**: `gates.py --exp bp` bars are #BP2;
  `figure_rstar.py --b-method b_pipelined` (→ `"overlapped"` in its
  `B_METHOD_PLACEMENT`) lands with #BP2/#BP3 when a consumer actually needs it;
  `gates.py:74`'s `B_METHODS` tuple likewise.
- **Verification, end-to-end**: smoke run on this box (2080 Ti), small N, `--method all`:
  all runnable rows `[verified]`; the T2 `b_pipelined` N/A skip line appears exactly
  once per (N, chunk); `h2d_occupancy ∈ (0, 1]` everywhere and visibly higher for
  `b_pipelined` than `b_fair` on kernel-bearing rows; `b`/`b_fair` rows byte-compare
  against a pre-change smoke CSV (three-generations-intact proof). Full C++ + pytest
  suites green (the harness is bench-only; the library is untouched). No `bp_*` artifact
  is created (BP3's job).

**Acceptance mapping** (#114): all configs `[verified]` = the existing gate + smoke
evidence; stage split + occupancy in the CSV schema = `gpu_kernel_ms` (per-chunk sum) +
`h2d_ms` + the new `h2d_occupancy`; N/A cells listed = the T2 loud skip + the audit
comment on the issue; chunkability audit recorded before coding = the issue comment
(posted 2026-08-06, before any implementation commit).

## Amendment (2026-08-06, during implementation)

### 1. Two-stream redesign (wording vs. intent)

§1's `runMethodBPipelined` design followed issue #114's literal wording — the per-chunk
kernel issued "in-stream after each chunk's copy", all on `pl.stream` — matching Method
A's loop shape exactly. Task 2's `h2d_occupancy` semantics check falsified this the
moment it ran: on one CUDA stream, submission order is FIFO regardless of data
dependencies, so chunk c+1's copy could not start until chunk c's kernel finished.
Measured occupancy came out statistically identical to `b_fair` (0.9242 vs 0.9245 on
`blocked_transpose`), and `h2d_ms + gpu_kernel_ms ≈ gpu_pipeline_ms` for both — i.e. the
literal single-stream reading reproduces `b_fair`'s non-overlapping behavior plus
per-chunk kernel-launch overhead, not #108's pre-registered expectation
`wall ≈ max(h2d, kern)`. The finding (with the same numbers) was recorded as a comment
on issue #114 before any reimplementation.

Per the user's ruling that the pre-registered **intent** governs over the issue's
literal phrasing (the method is named *overlap-fair*, its acceptance metric is *overlap
occupancy*, and #108's expectation assumes hiding), `runMethodBPipelined` and `Pipeline`
were corrected to a **two-stream** design: `Pipeline` gains `kernStream` (a second
non-blocking stream, created only when `withKern`), and each chunk's kernel is gated by
`cudaStreamWaitEvent(pl.kernStream, pl.h2dEnd[c], 0)` instead of shared-stream submission
order — so chunk c's kernel waits only for chunk c's bytes, and chunk c+1's copy proceeds
on the copy stream immediately, concurrently with chunk c's kernel on the kernel stream.
`evStop` moves to `kernStream` when the row has a kernel leg (its last kernel is itself
ordered after the last copy via that chunk's wait-event, so it still covers both legs);
both streams are synchronized before computing `t.wall`/`t.gpu`. Verification is
unaffected (bit-exact memcmp still gates every config; the two streams are fully
synchronized before `runMethodBPipelined` returns, so the blocking `cudaMemcpy`
readback in `runConfig` sees completed work either way; `dLin`/`dOut` chunk regions stay
disjoint across `c`).

Re-measured after the fix (2080 Ti, `--n 16384`, `--transform T1b,T3`, non-degenerate
chunking so every chunk-mib point yields `n_chunks > 1`):

| transform | method | h2d_ms | kern_ms | gpu_pipeline_ms | h2d_occupancy |
|---|---|---|---|---|---|
| blocked_transpose (4 MiB, 256 chunks) | b_fair | 82.88 | 4.47 | 87.76 | 0.944 |
| blocked_transpose (4 MiB, 256 chunks) | b_pipelined | 82.94 | 6.79 | 83.38 | 0.995 |
| quant (4 MiB, 256 chunks) | b_fair | 82.84 | 2.53 | 85.78 | 0.966 |
| quant (4 MiB, 256 chunks) | b_pipelined | 82.88 | 3.90 | 83.30 | 0.995 |

`b_pipelined`'s `gpu_pipeline_ms` now sits within ~0.5 ms of `h2d_ms` alone (≈
`max(h2d, kern)`), not near `h2d_ms + kern_ms` (`b_fair`'s behavior, and the old
single-stream `b_pipelined`'s behavior) — the intended overlap is realized. This holds
at every chunk granularity tested (4 through 256 chunks), not just the one above.

One residual, non-regression caveat found during re-verification: at the smoke's default
`--n 2048`, the `16 MiB` chunk-sweep point equals the workload's full 16 MiB input, so
`n_chunks = 1` there — a config where cross-chunk overlap is structurally unavailable to
*any* design (there is no next chunk to overlap the one kernel against). Task 2's
semantics-check script picks the CSV's last-written row per (method, transform), which
at `--n 2048` happens to be this degenerate single-chunk point, so re-running that exact
script unmodified at `--n 2048` still reports near-identical occupancy for that one point
— correctly, since no implementation can do better there. The `--n 16384` re-run above
uses chunk sizes that all divide the input into `n_chunks > 1`, exercising the design the
fix targets, and passes the identical assertion logic unmodified.

The chunkability audit (transposePlan column-band slicing, blocked/nchw group-aligned
sub-plans, the T2 N/A verdict) is unaffected by this amendment — it governs *what* each
chunk's kernel computes, not which stream it runs on or when it starts.

### 2. Uncheckable byte-compare replaced with function-body-diff proof

§2's "b/b_fair rows byte-compare against a pre-change smoke CSV" was uncheckable as
written — timing medians differ across runs (CSV rows can never byte-match across runs).
The three-generations-intact proof used instead: the `runMethodB` and `runMethodBFair`
function bodies are byte-identical between `main` and this branch (extracted via `awk
'/^StageTimes runMethodB\(/,/^}/'` / the `runMethodBFair` equivalent, diffed — zero
hunks for either function; separately confirmed that `git diff main` touches no line
inside either function's pre-change body, only inserts `runMethodBPipelined` after
`runMethodBFair`'s closing brace), plus both methods still print `[verified]` in the
`--method all` smoke.
