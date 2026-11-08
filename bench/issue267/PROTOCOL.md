# Issue 267 qualification protocol

This protocol compares the merged #266 baseline (`c5085cd9`) with automatic
host-chain specialization. The immutable campaign manifest pins all source,
runtime, helper and compiler bytes. Pilots and debugger profiles are excluded
from acceptance. No performance result is asserted before the complete campaign.

## Controls and matrix

`contract.py` fixes 36 fresh processes: three rounds × one/eight CPU workers ×
baseline/candidate runtimes × main/heldout/consumer suites. Runtime order alternates
by round. Each main process covers the original six E1 families at N=33/512/8192,
M=128. Held-out cases fix eight additional shape/chain combinations before tuning,
including outer rank3 extents 3/5, seven cast roundtrips, dequantization followed
by narrowing and widening, and BF16/quantize fallbacks.

Every original E1 case compares true per-operation materialization, whole-chain
folding, the original handwritten fused control, Torch eager and Torch Inductor.
`--no-canonicalize` is not a no-fold control and is not used here. New held-out
chains/outer extents without an existing manual implementation omit that control
explicitly. Each arm writes the same pinned final wire and submits one identical
H2D copy to the same preallocated device output. No placement/pipeline advantage
is enabled. `emulate_precision_casts=True` preserves Inductor's intermediate casts.

Each case has five warmups and 30 samples per mode. Inputs change by negation
before every sample; arm order rotates. Every timed result is checked outside
timing: exact non-NaN bits and NaN classification. Preparation/first calls and
warmups are also checked. BF16 intermediate native buffers are exposed as opaque
uint16 arrays, preserving their original bits.

Main/heldout modes are producer-only and completed producer+H2D+device
synchronization. The consumer suite adds a deterministic batch-32 linear+GELU
consumer of the transferred dequantized or padded weight matrix. Its timer covers
producer, H2D, compute and synchronization, with the same GPU computation and
output checks across arms. This is a scoped transfer/consumer experiment, not
full-model or server throughput, and is reported separately from E1.

## Environment, costs and interpretation

GPU0, CPU affinity4–7/20–23, one/eight aggregate Torch/native CPU workers,
interop1, passive OpenMP waits and one Inductor compilation worker are fixed.
Every process has a fresh Inductor disk cache. Baseline/candidate shared-library
paths are checked against `/proc/self/maps`. The existing compiler is unchanged;
its binary hash is pinned alongside the two host runtimes.
Each runtime separately records the source revision used to build its binaries;
the common measurement-source inventory is not the old runtime's build source.

Records retain source compile/bind/native preparation, common allocation, first
completed call, warmup costs, every sample, plan sizes, intermediate bytes and per-case
Inductor cache file/byte growth. The AOT runtime creates no per-shape executable
code/cache. Amortized totals must name their request count and included setup;
Inductor lazy compilation is included in its first completed call. Derived
Total(N) is shared allocation + arm-specific compile/bind/setup/storage + first
completed call + (N−1)×warm median. Warmup costs are retained separately and are
excluded from this amortization model. Unfused intermediate-buffer allocation is
charged only to that arm. CPU input generation and CUDA context initialization
are shared preconditions, not included setup costs.
Analytical traffic and observed elapsed time do not measure DRAM bandwidth.

Debugger stack samples and before/after stage/copy operation counts support
bottleneck attribution. `perf` is unavailable and hardware counters are not
claimed. Baseline profiling rejected the initial generic-walker hypothesis:
rank3_chain is already coalesced to a short-row tiled transpose. Tiny stage calls,
buffer copies and repeated widening dominate its sampled stacks. Padding retains
its stage-correct fill contract. Numerical stages may be register-fused but not
algebraically removed.

Acceptance requires 60 quiet seconds and continuous build/GPU/sustained-CPU
audits. Jobs execute serially, with no concurrent builds or profiling. Contaminated
attempts are preserved with hashes before an idle retry; clean results are never
repeated based on their timings. Unexpected worker failures stop the campaign.
The analyzer rejects altered membership, provenance, settings and output counts.
All regressions, fallbacks and losses remain in the final tables. Three rounds
on one machine do not establish universal speedups or statistical generality.
