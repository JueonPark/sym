# Prepared compact weight snapshots (#224)

`PreparedWireWeights` owns an inference checkpoint in the final INT8 wire
layout. A dense CPU `q[in, out]` and FP32 `scale[out]` become one private INT8
`[out, in]` allocation and immutable native scale metadata. Loads upload INT8
and dequantize into fresh contiguous FP32 `[out, in]` outputs. Arithmetic is
exactly `float32(q) * scale`, with zero point zero and output channel axis zero.
No implicit quantization, FP16 conversion, or FP32 weight expansion occurs.

```python
from reloc_torch import PreparedWireWeights

with PreparedWireWeights("cuda:0", max_prepared_bytes=512 << 20,
                         max_pinned_bytes=256 << 20) as weights:
    weights.prepare("layer0.up", q, scale)  # ready, stable CPU inputs during this call
    up = weights.load("layer0.up")         # completed, fresh GPU tensor
    # Group and prefetch at the model's existing consumer boundaries:
    with weights.prefetch([["layer0.up"], ["layer0.up"]]) as window:
        for (up,) in window:
            consume(up)                   # enqueue use before advancing the iterator
```

`prepare()` performs one offline Torch CPU transpose/copy and binds the
compiler-produced identity/dequantization recipe. It obeys the caller's Torch
CPU thread setting. The symbolic recipe compiles once per owner. This is
explicit value preparation; it does not claim that symbolic compilation alone
folds checkpoint values. Warm loads never scan the original weights or scales.
They validate fresh buffer descriptors and reuse the immutable dispatch template.
The native `cpu_stages_cuda_stages@0` row now uploads identity-layout inputs
without a second host gather/staging copy; its report says `direct_dense` and
zero host transformation bytes. Other layouts and nonzero stage boundaries
retain their existing transformations.

## Freshness and lifetime

This is a snapshot, not a live module-slot cache. After `prepare()` returns,
mutation (including `.data` and NumPy aliases), parameter/buffer replacement,
`load_state_dict`, changed scale values, and changes to tied weights do not
change that snapshot. Call `prepare(name, new_q, new_scale)` explicitly to
publish the new revision. Inputs must remain ready and stable during preparation.
CPU FP32 scales requiring gradients and unsupported representations are rejected.
Preparation failure leaves the previous revision accessible.

Replacement reserves old plus new storage before publishing. `invalidate(name)`
removes future lookup, while already submitted work retains its revision.
`invalidate()` removes all names. `submit(names)` exposes completion handles;
`load_many(names)` returns a completed `GroupResult` with transfer byte counts.
`prefetch()` resolves names when submitting each group, so updates between groups
can change later revisions. Use explicit context managers, including early exit.
Close rejects new work, drains outstanding transfers/consumers and clears entries.
Unknown CUDA completion quarantines buffers and keeps their budget charges.

`describe()` returns immutable scalar metadata with no writable CPU tensor alias.
GPU outputs are fresh; modifying one does not corrupt the checkpoint. Outputs
retained after their completion closes are caller-owned outside the queue budget.

## Storage and wire accounting

For `q[I,O]`, each load carries **I×O INT8 bytes + 4×O scale bytes**, and produces
4×I×O FP32 bytes. Equally prepacked Torch carries the same payload. Scale uploads
can be reused within a group by the existing runtime; report actual parameter
uploads separately. No 4× reduction relative to INT8 Torch is claimed.

`max_prepared_bytes` charges I×O + 8×O + 4 per live revision: INT8 storage,
the bound plan and arithmetic scale copies, and the scalar zero point.
`max_pinned_bytes` charges I×O when `pin_memory=True`. `max_entries` also counts
retired revisions. Replacing an entry at capacity may require invalidating it
and closing its outstanding handles first. The default limits are 1 GiB prepared,
1 GiB pinned, and 4,096 revisions. Stats include live/peak charges and queue counters.

Binding temporarily copies O(O) parameter bytes; Python/native plan metadata and
allocator caches are additional overhead. Limits describe live owned payload,
not process RSS. The queue separately bounds retained/live transfer scratch
(default 64 MiB), output reservations (128 MiB), and outstanding groups (two).
Its staging, including parameter staging, can add pinned bytes up to its scratch
cap. Preparation can run without CUDA using `pin_memory=False`; loading requires
CUDA. Owners are process-local and reject use after fork.

## Existing APIs

`prepare_weights(..., stable=True)` follows live module slots and checks a full
byte snapshot on every reuse. That is necessary for its existing mutation
contract, including aliases that evade Torch's version counter. It is unchanged.
The older `prefold_s8_image` bridge explicitly quantizes FP32 into INT8 and returns
an image; its automatic typed frontend capability is still gated. Native typed
prefold supports checked quantization, but does not implement this owned,
pretransposed INT8 checkpoint lifecycle. The [older V4 measurements](v4-prefold.md)
are not evidence for this API.

## Measurement protocol

[prepared_wire.py](../bench/issue224/prepared_wire.py) fixes the matrix regimes
before measurement: `[256,1024]`, `[1024,4096]`, `[4096,1024]`. It compares raw and
prepacked Sym, Torch eager, and Inductor conversion. All use identical INT8/FP32
semantics, pinned checkpoint inputs, fresh outputs, completed transfers, and
matched 128 MiB output / 64 MiB transfer scratch limits. Inductor compiles the
transfer conversion; model compute remains common Torch eager. It is not a
whole-model `torch.compile` comparison.

Three independent processes rotate path order; each collects 30 warm samples
and five completed total-time samples at reuse counts 1,2,4,8,16,32,64. Total reuse
time includes preparation, each completed load, and invalidation. Separate
series report atomic replacement and invalidate/reprepare costs. Artifact setup,
first preparation, and first completed loads (including first JIT use) remain
separate. Model cases are four-layer GPT decode (four tokens) and top-2 MoE with
one token; checkpoint preparation and first calls are separate from warm model
timing. Source generation/pinning and exact comparisons are outside timing.

First calls may reuse existing Inductor disk caches; they are process-first
loads, not cold-cache compiler benchmarks. Matrix measurements host-wait
completion and register each output with the consumer stream before returning;
model measurements use the same lookahead scheduling on every path.

The [correlated DMA trace](../bench/issue224/evidence/wire-trace.json) confirms
weight/scale uploads of 262,144/4,096, 4,194,304/16,384, and 4,194,304/4,096 bytes
for the three matrix shapes, respectively. Each call launches one
`dequantS8F32Kernel`; there is no FP32-expanded weight upload. This trace was
collected separately under competing CPU load and establishes representation
and kernel identity only, not timing or overlap performance.
