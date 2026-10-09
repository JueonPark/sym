# Typed context reuse and concurrency

`TransferResources` defaults to one typed context. Opt in to multiple contexts
when alternating GPUs or submitting independent work from different threads or
queues. A context is compatible with its CUDA ordinal, stream count, effective
owned worker count, borrowed worker participation, and submitting thread's CPU
affinity. Caller CUDA streams and tensor values are not part of the cache key:
every call captures current ordering and binds fresh inputs, parameters and
outputs.

```python
from reloc_torch import TransferResources, TransferQueue

with TransferResources(
    max_typed_contexts=2, max_typed_contexts_per_device=1,
    max_typed_retained_bytes=64 << 20, max_typed_live_bytes=64 << 20,
    max_typed_background_workers=6, max_typed_streams=2,
) as resources:
    with TransferQueue('cuda:0', resources=resources, gather_threads=4,
                       max_scratch_bytes=32 << 20) as q0, \
         TransferQueue('cuda:1', resources=resources, gather_threads=4,
                       max_scratch_bytes=32 << 20) as q1:
        # Submit separately prepared, single-device groups to either queue.
        # Each group may contain several weights. Keep sources unchanged until
        # its handle completes. Queue close drains that queue's handles.
        pass
```

The same owner can be passed to blocking typed and grouped calls. Set
`max_typed_contexts_per_device=2` to allow two independent leases on one GPU.
The per-device cap counts resident contexts, including idle configurations.
Closing a queue that borrows resources does not close the shared owner. Close
queues before their owner; owner `close()` also drains pending native work.

## Budgets and admission

Typed retained bytes, nonzero live bytes, background workers and streams are
**aggregate limits for this owner**, divided into fixed slot quotas. The first
slots receive any integer remainder. A slot cannot borrow an unused slot's
quota. For the example, each slot permits 32 MiB of live/retained scratch,
three background workers plus its submitting thread, and one runtime stream.
`max_typed_streams` must be at least `max_typed_contexts`. Zero live bytes keeps
the existing unlimited-live behavior; use a positive value for a hard limit.

These limits cover native typed staging and owned workers. They exclude
sources, fresh outputs, Torch's allocators, queue ordering streams, the separate
layout resource cache and other owners. Sum these separately at process level.
Each queue retains its own `max_output_bytes` reservation limit. An external
`GatherPool` reserves its participating background workers against a slot's
quota while leased; its lifetime and unrelated uses remain its owner's
responsibility. Caller threads also count toward the application's CPU budget.

Admission is FIFO. A compatible idle slot is reused; otherwise the least
recently used eligible idle slot is replaced. The per-device cap can require
replacing another configuration on that device. A queued call cannot bypass
an older waiter, even if another device is free. When all eligible contexts
are leased, admission can complete an older asynchronous submission. A lease
ends at host-observed completion, not at event recording or `wait_stream()`.

`typed_acquire_timeout_ms` optionally bounds admission waiting (`0` fails
immediately when unavailable); execution after admission has no deadline.
As with existing execution failures, Python prepared requests/groups are
one-shot after an execution attempt. Prepare a fresh request for a retry.
`clear()` pauses admission, drains active work, retires contexts and resumes
waiters. `close()` rejects queued/new calls and drains every active context.
Unknown completion quarantines the exact failed context and its owners for
the process lifetime, disables the shared owner, and still permits healthy
pending siblings to complete. Forked children reject inherited use before
native locking or CUDA access.

## Observability and tuning

`resources.stats()['typed']` exposes aggregate `limits`, per-slot `contexts`
with device, affinity and quotas, `active_contexts`, `peak_active_contexts`,
`queued`, `admission_waits` and `evictions`. Cumulative execution counters are
snapshots of completed leases, so reading stats never waits on a busy GPU.
`retained_bytes` counts idle scratch; `live_bytes` and `peak_live_bytes` count
actual simultaneous allocations, including active and quarantined scratch.
Worker/stream gauges on active slots are reservations; borrowed workers are
listed separately in each context. These are resource statistics, not proof
of simultaneous CPU or DMA execution.

Choose per-call threads so all active callers plus background workers fit the
CPU budget. Pin the submitting thread **before** context creation; owned
workers inherit that affinity. Allocate/first-touch CPU sources near the target
GPU, and keep affinity stable to retain compatible contexts. More contexts,
workers or streams cannot bypass host-memory bandwidth or a shared PCIe root.
The pool does not migrate pages or automatically choose NUMA nodes, chunks or
thread counts. See the [measured scheduling controls](../bench/issue228/README.md)
for qualified settings and limitations.
