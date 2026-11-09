# Consumer-aware KV layout elimination (#269)

## Objective and choices

Implement and qualify the first target explicitly prioritized by #251/#269:
remove a private inverse KV permutation materialization before concatenation.
Profiles in #263 show shared native kernels and residual layout/consumer work;
#268 shows launch savings alone do not establish a model win. This candidate
must preserve the exact cache presented to downstream attention, and demonstrate
removed work separately from completed latency.

The executor selects generic FX clone/contiguous elimination before functional
concatenation, plus an explicit packed-wire restore boundary. Alternatives were
handwritten dequantizing GEMM (different arithmetic/kernel qualification, deferred)
and opaque fused operations (prevent downstream compiler optimization, rejected).
This is an initial supported lowering, not arbitrary consumer fusion or new wire
compression. Existing dequantization and FP16 GEMM math remain unchanged.

## Architecture

1. New `consumer_layout.py` recognizes a closed permutation→materialization→cat
   pattern on GPU in FX graphs. Every use of the materialized tensor must be a
   functional, fresh-output cat with the same dtype/device; otherwise preserve it.
   A chain can contain transpose/permute views, but no channel gather, padding,
   reshape, casts or value transforms. Source metadata must be ordinary dense
   strided, contiguous rank2/rank3 input with a valid storage offset (including
   packed-buffer views); unknown/symbolic metadata
   conservatively excludes the candidate initially. Aliasing writes, opaque
   operations or externally visible intermediate/layout users exclude rewriting.
   The result is ordinary graph operations, not an opaque custom operator.
2. Expose immutable eligibility/cost records: rule/source/materialization/consumer
   nodes, logical eliminated allocation bytes and read+write bytes, exclusion
   reason. These are logical estimates, not physical peak predictions or a latency
   model. A policy enables the pass and applies minimum saved bytes; default off.
   Integrate as `RelocBackend(consumer_layout=False|ConsumerLayoutPolicy(...))`
   before transfer import, with reports in stats and original graph fallback.
3. Add explicit `KVCacheSwap.restore_wire(stored)` returning fresh dense GPU
   `[S,H,D]` wire tensors. A shared helper also supplies Torch controls. Adjacent
   pinned views with common backing storage use one H2D copy; other legal dense
   inputs use safe per-tensor copies. Validate plain dtype/rank/device/strides and
   storage bounds before any submission. Host inputs remain owned until completion;
   returned outputs own storage, work on the current stream, and survive close.
   Existing `restore`/evict public behavior remains unchanged. Wire restore never
   exposes a pointer-based graph shortcut or reuses an output arena.
4. The benchmark's transparent wrapper inserts `wire.transpose(0,1).contiguous()`
   before the unchanged decoder. The graph pass proves the intermediate private
   and removes it where every use reaches cat. Both Torch and Sym receive the same
   lowering, raw-copy helper, kernels, grouping, budgets and replay opportunities.
   This adapter is explicit workload integration; automatic whole-program region
   discovery is not claimed. Inductor may already eliminate the same intermediate;
   report that outcome rather than attributing its existing fusion to Sym.

## Legality and numerical contract

The eligible rule only removes a copy. Functional cat must produce the same
contiguous output shape/stride/dtype and bitwise values, including signed zeros,
NaNs and integer payloads. Attention sees identical cache bits/layout. INT8→FP32→
FP16 rounding, GEMM accumulation policy, TF32 flags and token/expert decisions stay
unchanged. General whole-model floating results use original tolerance; exact
cache equality and token decisions are separate checks. No bitwise claim is made
across different eager/Inductor arithmetic implementations.

Reject or materialize: graph outputs exposing the intermediate, alias/view users,
CPU-visible/mixed consumers, mutation, noncontiguous sources, gather/channel
maps, padding, unknown operators, dtype/device changes and missing metadata.
Tests must prove exclusion preserves original semantics, including held outputs.
Training is outside scope; activation/forward transforms must not silently change
semantics. Runtime specialization relies on Dynamo guards; direct graph rewrite
is explicitly tied to the provided metadata. Default paths are unchanged.

## Evidence

Freeze exact matrix/source/runtime before acceptance. Three fresh-process rounds,
five warmups, at least30 completed samples, rotated variants, CPU affinity4–7,
20–23/eight threads, one interop thread, RTX2080Ti and pinned Python3.14.7/Torch2.14.
Observe60 quiet seconds before each matched cell and audit foreign CPU/GPU jobs.
Never build/test/profile concurrently with accepted timing. Retain failures,
negative results, warm compilation, setup costs and excluded pilots.

Model matrix: original decode(prompt8,new_tokens4), held-out decode(prompt64,
new_tokens4), prefill(prompt256,new_tokens1; no restore negative control). Compare
current native-materialized, transparent optimized-materialized, lowered candidate
for Torch/Sym with eager/Inductor compute. Resident-weight/cache baselines use
identical math where feasible, with residency/preparation disclosed. Include a
matched replay-enabled original-decode subset using #268's identical wrapper and
quotas for all variants, alongside unwrapped main runs. No per-sample selection.

Micro qualification: canonical and held-out heads/dim/lengths, mixed-consumer
fallback, large/low-memory conditions; exact cache values/held outputs. Diagnostic
profiles count allocations/materialization kernels and observed bytes; separately
measure cat/consumer slowdown, raw transfer, complete requests, preparation,
compile/first/warmup, p50/p95, Torch allocator and sampled physical memory. Do not
subtract isolated phases from model latency or equate logical bytes with peaks.

Report per-round comparisons and parent5%-below-both-controls test; preserve equal
opportunity controls and the fastest observed alternative. Three rounds/oneGPU
support only declared configuration. Saved-data analysis uses stdlib and enforces
matrix, hashes, numerical counts, resource contracts and observed isolation.

## Delivery

Work in the existing isolated worktree on merged3a62084; original checkout stays
untouched. English source/report/PR; Wiki artifacts with anonymous hash/analysis/
relocated-runtime verification; no Releases or SVG in source PR. Link #269/#251
for closure only on merge. One builder design check and one independent final
whole-change review; inline implementation. Scope judgments belong in the ledger.

## Builder-check resolution

The fresh builder identified packed-buffer views with nonzero offsets. Accept
validated dense offset views: only logical values are read, and cat creates a new
owned output. The source remains owned and no pointer arithmetic is synthesized
by the graph pass. Tests cover shared-storage offset views and intervening alias
mutation rejection. This is the executor's correction to the initial zero-offset
restriction; no extra device copy is inserted to manufacture zero offsets.
