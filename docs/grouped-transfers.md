# Consumer-sized transfer groups

Issue [#220](https://github.com/JueonPark/sym/issues/220) adds explicit groups of
related typed weights and layout-only KV transfers. A group uses one private
CUDA queue, orders once after the caller's current stream, and completes all
members at one barrier. It keeps separate output allocations and does not pack
or concatenate source payloads. CPU H2D transforms can proceed while earlier
members' copies are queued; D2H transforms run after the group's copies finish.

This is a blocking API. Choose the group at a consumer boundary: a layer's
weights, an expert's two matrices, or a K/V pair. Split groups when consumers
need different completion times. Separate owners and caller streams permit
independent concurrent groups. Preparing a group neither submits data nor
captures a stream; execution reads the current stream, leaving the request
structure suitable for a future separate submission/completion API. It does
not add asynchronous prefetch or automatically cross model dependencies.

## API and output lifetime

Use existing preparations, including the immutable execution templates from
[#219](request-overhead.md), then execute one group:

```python
from reloc_torch import (
    TransferResources, prepare_transfer_group, execute_transfer_group,
)
from reloc_torch.transport import prepare_transfer

# kv_recipe describes the desired forward transpose and transfer direction.
# Its symbolic dimensions may bind differently for each member.
requests = [prepare_transfer(kv_recipe, tensor, "cuda:0") for tensor in (k, v)]
group = prepare_transfer_group(requests)
assert len(group.destinations) == 2  # shape, strides, dtype, device per output
with TransferResources() as resources:
    result = execute_transfer_group(group, resources=resources)
k_gpu, v_gpu = result.tensors
# Outputs own separate storage and remain valid after group/resources release.
```

Groups accept 1–256 fresh layout or typed preparations, including mixed shapes,
dtypes, transforms and H2D/D2H directions, on one canonical CUDA device. No
member may appear twice or have been consumed. Execution acquires every member's
execution guard, rechecks storage and typed parameter snapshots, allocates new
outputs and validates all native views before submitting any member. The native
boundary rejects overlapping outputs and output/input aliases.

All member requests become consumed when native execution is attempted. A group cannot
be executed again, nor can its members later be executed individually. A stale
source/parameter failure before the execution boundary leaves the group fresh.
Keep sources and parameters unchanged while preparing/executing the group, just
as for individual calls. Earlier returned outputs may stay live and be mutated
independently of later groups.

The workload helper also exposes an explicit layer/expert entry point:

```python
# Existing workloads/common.py WeightFetcher; the caller chooses the group.
weights = fetcher.fetch_many(
    [layer["qkv"], layer["proj"], layer["fc1"], layer["fc2"]], "cuda:0"
)
# Or: w1, w2 = fetcher.fetch_many([expert["w1"], expert["w2"]], device)
```

`fetch()` remains the individual-call control. `fetch_many()` shares the same
owner per device, applies the same calibration/selection and records logical
bytes plus scalar group counters. Existing model examples keep their individual
call sites; grouping is opt-in and its transfer benchmarks do not imply a
whole-model speedup.

## Scale reuse and bounds

Within a group, compatible device float parameter uploads are keyed by exact
uploaded bytes, including length. Separate tensors with identical values can
share an upload; different values or extents cannot. Each kernel still receives
its own shape/channel arguments. Upload memoization lasts only for that group,
so changing a scale between groups always uploads its fresh value. Request
preparation still validates every parameter's dtype, extents and values, and
execution still checks the snapshots.

The default `max_scratch_bytes=64 << 20` bounds total owned host/device scratch
while the group runs. It must be positive. The resource owner's
`max_typed_live_bytes`, when nonzero, applies too; the stricter cap wins. A
smaller group cap evicts excess idle capacity before use. Retention between
calls remains bounded by `max_typed_retained_bytes`. Scratch cannot be recycled
within a group, so grouping trades greater simultaneous scratch for fewer
completion operations. Output/source tensors, immutable template metadata and
small request/parameter-key bookkeeping are outside the scratch budget.

Groups use the typed resource owner even for layout-only members, with one
private stream. `gather_threads` defaults to eight; an explicit gather pool can
be supplied. Groups composed entirely of direct GPU relocation/dequantization
rows do not create CPU gather workers. Pinning policy and its optional threshold
are the existing per-execution settings. Groups submit each payload independently; packing bytes
are zero. This avoids introducing packing cost or shared output-storage
lifetimes without evidence that coalescing helps.

## Failure and concurrency

A resource owner leases its context exclusively; concurrent calls on one owner
serialize, and different owners allow independent progress. No member can be
executed concurrently with its group. Every scratch block stays busy until the
whole group finishes, including temporary buffers whose local helpers have
returned.

A later allocation, upload or launch can fail after earlier members were
submitted. The group returns an error, never partial outputs or an automatic
retry. The owner then proves completion on all queues. With a successful drain,
all scratch is released and the owner remains reusable. If completion is
unknown, it quarantines the entire context **and all source/output owners**,
and permanently disables that owner. No staging-report pointer survives in an
idle or quarantined context. Tests inject failures in event recording and the
third copy, with both successful and failed drain proofs.

## Reports and measurement

`GroupResult` exposes `tensors`, immutable logical `descriptors` and a scalar
`report` with per-member implementation/byte reports. Group fields include:

- Logical transfer and native request counts, payload copies, parameter
  uploads/reuses and uploaded bytes, kernel launches.
- Backend event-record/wait and producer-ordering counts. These count backend
  operations; CUDA tracing additionally counts the actual CUDA API calls,
  including producer-event creation/record/wait/destruction.
- Packing bytes (zero), required CPU transform bytes, output bytes, staging
  decisions and peak live scratch bytes.

`TransferResources.stats()['typed']` also reports cumulative copy/event/producer
counts and scratch high-water bytes for individual and grouped controls. Torch
allocator peaks do not include native `cudaMalloc` allocations; the benchmark
reports Torch allocator changes and native host/device scratch separately.

Run `bench/issue220/grouped_transfers.py` for small/large KV in both directions,
small/large weights and heterogeneous groups, with individual Sym and similarly
batched Torch controls. The Torch control submits nonblocking copies and member
kernels on one current stream, reuses shared scale uploads within that group,
and synchronizes once. First calls, warm samples, resource/memory counters,
correctness and environment/binary hashes are recorded separately. Recipe
compilation and input mutations/oracles are outside completed-call timing.

The [qualification report](../bench/results/grouped-transfers-220/README.md)
contains matched current-main and Torch controls, independent raw rounds,
CUDA API/event counts, memory tradeoffs and fault-test evidence.
