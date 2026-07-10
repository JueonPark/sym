# PoC reproduction: 32768² fp32 blocked transpose (issue #47)

**Result: ours 499.31 ms vs baseline 180.95 ms (wall,
median of 3×50) → 0.36× speedup.** Run-to-run median spread:
ours 1.20%, baseline 0.14% (stability bar: < 5%).

## Setup

- GPU NVIDIA GeForce RTX 4070 Ti SUPER, WSL2 (6.18.33.2-microsoft-standard-WSL2), CUDA 13.2 (nvcc V13.2.78).
- Plan: the frozen "reference" golden (4D blocked view + transpose(0,1),
  `divisible(N, 64)`), N = 32768, fp32 → 4 GiB src/dst.
- Ours: strategy-4 pipeline — 2 pinned staging buffers (caller-owned pool,
  256 MiB chunks) × 2 CUDA streams, event-recycled double
  buffering (`executeH2DPipelined`).
- Baseline (the PoC baseline): whole-tensor `cudaMemcpyAsync` from pinned
  host + naive strided relocate kernel on device. GPU time (CUDA events):
  170.09 ms median.
- Both methods read the same pinned host source; the baseline's
  pinned+contiguous precondition is granted for free (not charged), so the
  comparison is generous to the baseline.
- Protocol: `bench/protocol.h` — 10 warmup + 50 timed iterations,
  median + IQR, 3 re-runs; wall time via `steady_clock`, baseline GPU time
  via CUDA events. Byte-exact verification against CPU `executeH2D` ran
  in the same invocation (`--verify`).

## Numbers

| method | wall median (ms) | IQR (ms) | rerun spread |
|---|---|---|---|
| ours (pipeline, 2 buf × 2 streams) | 499.31 | 10.36 | 1.20% |
| baseline (pinned memcpy + kernel) | 180.95 | 0.07 | 0.14% |

Raw data: `bench/results/poc_transpose_n32768.json`.

## Honest-reporting notes

**The measured speedup is 0.36×, i.e. ours is roughly 2.8× *slower* than
the baseline, not ~2× faster.** This is materially below the ~2× PoC
expectation stated in issue #47. Per the issue's acceptance bar, this is
reported as-is and is flagged as a **schedule-discussion trigger**, not
something silently tuned away: no chunk sizes, buffer counts, iteration
counts, or method definitions were adjusted in response to this result.

Stability was not a concern here: both methods cleared the < 5% bar
comfortably on the first run (ours 1.20%, baseline 0.14%), so no re-run
was needed and no timing-noise explanation is required for the headline
number.

Context from Task 4's small-N smokes: N = 4096/8192 showed 0.13–0.23×
(single-thread CPU-style gather semantics vs a GPU-parallel baseline
kernel at small scale). At full size (N = 32768, 4 GiB), the ratio
improves somewhat (0.36× vs 0.13–0.23×) but ours remains substantially
slower than the baseline rather than crossing 1× or reaching the ~2×
expectation. The likely qualitative explanation (offered as context for
the schedule discussion, not as a justification to re-tune the
benchmark): the baseline is a single whole-tensor `cudaMemcpyAsync` +
one kernel launch over the full 4 GiB, so its fixed per-call overhead is
paid once, whereas the pipelined strategy splits the same 4 GiB into
256 MiB chunks (16 chunks at N = 32768) across 2 buffers/2 streams,
paying host-side coordination and per-chunk launch overhead repeatedly;
at this problem size that repeated overhead is not being hidden well
enough by the double buffering to beat the baseline's single large
transfer + kernel.

No deviations from the planned protocol: preflight passed cleanly
(GPU idle, no other CUDA processes; ~19 GiB RAM available, above the
14 GiB floor), so the full `--verify` run at N = 32768 was executed
directly — no substitution with a smaller `--verify` run plus an
unverified full-size run was necessary. Verification passed
byte-exact against the CPU reference for both methods. All
measurements were taken under WSL2; no additional WSL2-specific timing
anomalies were observed (both spreads were far under the 5% bar), though
WSL2 is noted here as the standing environment caveat for anyone trying
to reproduce these numbers on bare-metal Linux.

The ~2× PoC figure is the expectation, not the pass bar (issue #47); this
note reports the measured value as-is. Protocol reuse: the same header
drives `bench/bind_cost.cpp` (#C3 bind cost, `bench/results/
bind_cost_n4096.json`: median 0.0069 ms). Gather/H2D overlap
trace: see `bench/results/poc_overlap_summary.md` (issue #47 comment).
