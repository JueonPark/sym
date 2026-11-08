# Budgeted transfer memory

`PackedOutputArena` and `plan_transfer_memory` are opt-in controls for dense
transfer outputs. They separate output ownership from transfer completion and
make prefetch/context choices explicit. They require the matching native Torch
bridge; ordinary transfers retain their existing allocation behavior.

## Ownership and completion

An arena packs a group into one allocation, with 256-byte alignment between
outputs. Ordinary tensors, views, detached tensors and retained storage handles
keep that allocation live. Neither a completed transfer nor `wait_stream()`
permits overwriting it. Recycling requires the final storage alias to disappear
and a device-wide completion barrier before the address can be reused. This
conservative barrier can reduce overlap and increase latency.

`reuse=False` isolates packing: outputs use ordinary Torch allocator storage,
and the arena keeps only weak lifetime trackers. It adds no recycling barrier.
With recycling enabled, the native owner retains the original Torch allocation.
It checks both TensorImpl and Storage strong ownership, discounting a canonical
Python wrapper only when Torch itself holds its sole Python reference. Storage
handles held by callers therefore prevent reuse just like tensor aliases.
The allocator DataPtr and StorageImpl remain unchanged throughout the lifetime;
close completes pending device work and drops only the owner's reference.
Concurrent consumers and later `record_stream()` calls use ordinary allocator
semantics, including work queued after close. Uncertain CUDA completion poisons
and quarantines backing storage rather than freeing or reusing it; destruction
does not retry the transfer. Quarantine can retain memory until process exit.

The arena never spills into an unbounded allocation. A group larger than its
cap, or a cap exhausted by held aliases, raises `MemoryBudgetError` before a
transfer launches. `FastPrefetch` defers such a speculative submission while a
pending result can make progress. Otherwise it surfaces the refusal; it does
not wait indefinitely for the caller to drop an alias. Actual CUDA OOMs and
execution errors are propagated without replay.

Callers must retain sources unchanged until transfer completion, and retain
tensors needed by future raw-pointer operations. Weak storage handles are not
ownership leases after the final strong consumer releases the storage. Exported
storage is made non-resizable before publication so it cannot grow outside the
arena's capacity. CUDA graph capture is rejected
for arena allocations; graph replay is outside this contract. Inherited arena
use after fork is rejected before touching locks or CUDA. CPU arenas are
supported for ordinary dense storage but cannot be attached to a CUDA owner.
Zero-size, noncanonical-stride, mixed-device and quantized outputs are rejected.

## Planning and application

`plan_transfer_memory` is a pure finite search over resident prefixes, prefetch
depths and whole-group versus single-member staging. Its byte inputs must come
from the qualified row and include alignment. It prefers the largest allowed
resident prefix, then the deepest feasible window, then whole-group staging.
The output envelope includes `slots + 1` groups, covering prefetched work and
the consumer's previous group. Additional held outputs belong in
`retained_bytes`. No feasible candidate raises `MemoryBudgetError`.

```python
from reloc_torch import FastTransfers, FastWeights, PackedOutputArena, plan_transfer_memory

plan = plan_transfer_memory(
    group_output_bytes, group_device_scratch_bytes,
    available_bytes=physical_capacity_bytes,
    workspace_bytes=workspace_and_library_allowance_bytes,
    retained_bytes=other_held_outputs_bytes,
    max_slots=2, max_resident_groups=0,
    subgroup_scratch_bytes=single_member_scratch_bytes,
)
arena = PackedOutputArena("cuda:0", max_bytes=plan.output_bytes)
try:
    with FastTransfers("cuda:0", contexts=plan.contexts,
                       max_scratch_bytes=plan.native_scratch_bytes) as owner:
        loader = FastWeights(owner, snapshots, output_arena=arena,
                             max_group_members=plan.subgroup_members,
                             max_prefetch_slots=plan.slots)
        with loader.prefetch(name_groups) as window:
            for outputs in window:
                consume(outputs)
finally:
    arena.close()
```

This example requires at least one nonresident group. Fully resident plans
have zero slots/contexts/output-window bytes: preload the chosen prefix into
separately owned resident storage, then omit the transfer window. Resident
preload still needs its own staging capacity; the steady-state estimate does
not promise that preparation fits. Include that cost and peak in qualification.

`FastWeights` borrows its arena and can cap members per submission and prefetch
depth. It validates the full output group, including cross-subgroup aliases,
before launching any subgroup. A later subgroup failure drains earlier
completions and propagates the error without replay. The full returned group
keeps its ordinary tensor lifetime. Subdivision changes staging and submission
overhead, not the mathematical output. Preparation and validation costs remain
part of the caller's cost.

A weight-specific arena leaves KV transfers on the same native owner with
their usual output allocation. Account for KV storage in the separate workspace
allowance. Alternatively, `FastTransfers(..., output_arena=arena)` applies
packing to every automatically allocated typed/layout group, including explicit
prepared handles. Unsupported groups, such as D2H into a CUDA arena, fail
before launch. Explicit `outputs=` remain caller-owned and receive normal
shape, stride, dtype, bounds and alias validation; they are outside arena
capacity accounting. Transfer owners and loaders never close a borrowed arena.

## Estimate versus physical memory

The plan reports resident storage, output-window capacity, device scratch per
context, workspace, retained bytes and safety margin separately. The native
scratch quota includes host and device storage, so `native_scratch_bytes` is
twice the device estimate (at least one byte for an active zero-scratch context).
The planner's physical estimate includes only the device share. For concurrent
callers, partition a common aggregate allowance; do not assign the entire
budget independently to every caller. Replanning creates a new owner after
draining the old one; it does not resize active contexts.

The input `available_bytes` is explicit: choose and document whether it means
remaining free space or a capacity envelope, and avoid counting existing
allocations twice. The estimate is not an allocation guarantee. Allocator
rounding/cache state, library workspaces, other callers and asynchronous device
use can invalidate it. Record admission refusal, predicted fit and actual OOM
as different outcomes.

`arena.stats()` reports logical allocated/live/reusable capacity, allocations,
reuses, evictions, refusals, completion barriers, quarantine and exported bytes
at close. `FastWeights.stats()` reports its arena and subgroup counts;
`FastTransfers.stats()` reports its own optional arena separately from native
resources. These counters overlap Torch allocations and must not be summed
into a physical peak. Native scratch, Torch allocated/reserved bytes, pinned
snapshots, retained outputs and sampled device memory have different scopes.
Zero warm native allocations alone does not establish lower memory use.

| Storage | Owner and lifetime | Measurement scope |
| --- | --- | --- |
| Native staging/scratch | Transfer contexts retain buffers between calls; owner close drains and retires them | Native device/host/retained counters; direct CUDA allocation is outside Torch counters |
| Packed weight outputs | Arena plus every strong tensor/storage alias; eligible for reuse only after external owners release it | Arena capacity overlaps Torch allocated bytes |
| Explicit outputs and returned request outputs | Caller-owned tensors, which may survive transfer-owner closure | Torch allocated bytes; caller must include their retention in its allowance |
| Torch cached blocks | Caching allocator, potentially retained after tensors die | Torch reserved bytes include active and cached allocations |
| Pinned snapshots | Snapshot/checkpoint owners retain immutable host source images | Snapshot pinned bytes and host allocator counters have separate ownership scopes |
| Context/library/compute/KV state | CUDA context, libraries and application tensors/workspaces | Physical samples and explicit workspace allowance; no exact sum from the other counters |

See PyTorch's [CUDA memory and stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda)
and [memory snapshot scope](https://docs.pytorch.org/docs/stable/torch_cuda_memory).
Qualification and limitations are recorded in [issue 266's experiments](../bench/issue266/README.md).
