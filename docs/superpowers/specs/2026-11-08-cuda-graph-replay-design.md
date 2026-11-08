# Bounded GPU-region replay (#268, includes #252)

## Intent and boundary

Qualify CUDA Graph replay on completed matched E2/E5 requests after #264–267.
The merged baseline is b319a0ab94dbc8c8264e4fa093ffbbb796e2c046. No graph runtime
exists yet. CPU transformations, dynamic routing, transfers, grouping and
prefetch remain outside capture. Replay is opt-in and inference-only.

## Approaches and decision

1. Enabling Inductor's whole-graph switch does not make native transfers capture
safe and supplies no common ownership contract. Keep that rejection.
2. Automatic arbitrary FX partitioning adds a compiler project and implicit
state inference before replay can be qualified. Defer it explicitly.
3. Implement an explicit pure GPU callable boundary, used with prepared transfer
outputs and available identically to eager Torch and Inductor. Selected by the
implementer: this directly matches the existing model GPU function boundaries.

## API and ownership

`GPUReplayPolicy(enabled=True, slots=2, max_buckets=8, max_bytes=256<<20)`
configures `GPUReplay(function, policy=..., device='cuda:0')`. Calls accept nested
Tensor/None/bool/int/float/string positional and keyword arguments. All tensors
must be nonempty dense contiguous ordinary CUDA tensors on the declared device.
Nonmatching metadata/device, unsupported arguments, bucket exhaustion, busy slots
and memory admission failures use the original callable and record the reason.
The function must be a pure, deterministic, functional GPU region: all mutable
parameters/state are explicit arguments; no mutation, RNG, host side effects,
CPU transfers, host synchronization or captured mutable closure/module state.
This contract is explicit, not an automatic proof of capture safety.

Keys include complete pytree structure, static scalar values and tensor shape,
dtype and stride. Incoming addresses may change: copy every tensor argument to
stable storage before launch, including parameters. Exact shapes are buckets;
no padding or speculative shape generalization. Bucket count and slots bound
retention; unknown shapes fall back once full. `invalidate()` retires all graphs
for immutable revision/function changes; `close()` retires and forbids reuse.
Forked-process use is rejected before CUDA. Device changes fall back without
reusing a graph. Autograd-enabled calls are rejected.

Each slot owns static inputs, a graph-private pool, output references and a
completion event. A process-wide lock serializes cold capture. Capture requires
quiescent GPU work in the process (documented PyTorch restriction); prepare all
regions before concurrent serving. `prepare(*args, **kwargs)` creates all slots
for a bucket; calls can lazily prepare only before explicit `freeze()`.
`freeze()` prevents capture in serving and sends new shapes to fallback.
Warm calls choose an idle slot, execute copies/replay/output clones on the caller
stream, and record completion after the clones. Inputs are recorded on that
stream to preserve their allocator lifetime. Outputs are independent ordinary
Torch tensors, safe to hold across replay, aliasing and close. Slot reuse waits
for completion; occupied slots fall back, without synchronization or eviction.
`enabled=False` executes identical input-copy/output-copy slot machinery with the
same function/kernels, exposing replay attribution independently of storage.

Capture failure retires partial state and disables that bucket; a clean capture
failure may fall back. CUDA execution/replay errors propagate and poison the
owner, never duplicate a possibly launched request. Warmup/function errors must
propagate. Retire events before destroying graphs/storage; never clear the global
Torch allocator cache during serving.

## Memory

Policy bounds persistent input storage plus full graph-private allocator segment
reservation, measured via Torch allocator snapshots and graph pool IDs. Admission
is checked before input allocation and after capture; transient capture can
exceed the declared retained limit and OOM is possible. Unknown allocator pool
accounting disables capture rather than claiming zero. Inputs are ordinary
allocator storage outside private pools. Returned clones, caller-held outputs,
external transfer arenas/scratch, library/context state and allocator caching
are separate scopes; measure actual Torch peaks and sampled NVML device memory.
Do not present this owner limit as a physical device OOM guarantee.

## Qualification

Tests cover changing values/parameter addresses, explicit revision invalidation,
held aliases and outputs after close, all occupied slots, producer/caller streams,
metadata/device fallback, bounded shape growth, capture failure, budget refusal,
freeze, process mismatch and backend-owned lifecycle integration.
`RelocBackend.prepare_gpu_region()` owns wrappers and closes them with the backend;
prepared transfer execution remains outside the region.

Use the original E2 workload mathematics, FP16, native dequant controls, grouped
KV and two-slot whole-request prefetch. Measure four model cases with Torch/Sym
x eager/Inductor x replay off/on (and unwrapped continuity references). E5 uses
original decode and sparse MoE, changing requests and 1/2/4 callers, with prewarm
before concurrent frozen serving. Three independent rounds, fresh processes,
fixed affinity and aggregate CPU budget. Save first/preparation/capture costs,
CPU submission, completed p50/p95, memory, correctness, output retention and
break-even. Include low-memory refusal and shape-variation rows; do not omit
negative cells. A diagnostic Torch/CUPTI trace must show graph launches replacing
ordinary kernel launch submissions with same mathematical work. Use the existing
60-second quiet gate and interference audits. Freeze sources/protocol before
acceptance and verify provenance and matrix completeness in a stdlib analyzer.

## Delivery

English docs and ready PR with `Closes #268` (and #252 if its implemented scope
and evidence are met). No manual issue closure or merge. Public immutable Wiki
artifact with raw data, sources, dependencies and anonymous verification; no
Releases or SVG files. Original checkout remains unchanged.
