# Request preparation and execution templates

Issue [#219](https://github.com/JueonPark/sym/issues/219) reduces repeated host
preparation for completed KV and typed weight transfers. The
[baseline diagnosis](../bench/results/request-overhead-219/baseline-diagnosis.md)
was recorded before implementing these changes. It distinguishes exclusive
host phases from overlapping CUDA work; its historical percentages are not
assumed to apply to later revisions.
The [matched results](../bench/results/request-overhead-219/README.md) include
first-call latency, warmed distributions, phase costs and reproduction commands.

## What is reused

Each `CompiledRecipe` owns an LRU of immutable execution metadata. Layout-only
entries retain the native bound layout, keyed by checked symbol bindings.
Typed entries retain the bound program, checked arithmetic, capability rows and
selected implementation. A typed key contains:

- Symbol names and concrete values. The owning compiled artifact fixes the
  recipe, source/destination descriptors, layout and value transforms.
- Every runtime parameter's name, dtype, concrete extents and **exact current
  bytes**. Tensor identity and data pointers are not validity keys.
- Transfer direction, source device type/index and canonical destination CUDA
  device (or source CUDA index for D2H).
- Selection policy, explicit implementation, selection thread count and the
  calibration object. Python calibrations are immutable; distinct calibration
  objects have distinct keys even when their machine names match.

Parameter extents also use a separate 32-entry immutable metadata cache, like
the existing destination-descriptor cache. Per-request dictionaries and
capability/selection reports are fresh copies.

The native `prepare_dispatch_template` API combines checked-program creation,
capability discovery and selection. `prepare_dispatch_from_template` validates
new buffer views and creates a new request with independent consumed/report
state. It does not repeat selection. The existing native preparation API is
still available. Python exposes template observations as read-only properties;
C++ callers must treat a successfully prepared `DispatchTemplate` as immutable.

## What is checked on every call

Runtime eligibility reads live tensor fields directly instead of constructing
an observation record and querying unused pinning metadata. Within one
frontend invocation, preflight may reuse its checked symbols and destination
metadata only after comparing source identity, a fresh storage/metadata snapshot
and gradient state. This scope is thread-local and ends with that invocation.
Direct bridge calls perform normal admission and binding checks.

Typed preparation snapshots the current parameter bytes even on a hit. Before
execution it rechecks parameter bytes and source storage/metadata. Fresh buffer
views, capacity checks, CUDA pointer ownership checks, caller-stream handling,
output allocation and completion remain in the execution path. Invalid or
changed descriptors/parameters cannot use a stale cached program. Invalid
selection argument types are rejected before lookup.

The cache owns no source/output tensors, buffer pointers, streams, resource
owners or executable requests. Source payloads are read anew and outputs are
independent allocations. Buffer/stream counts, pinning mode, gather threads and
resource budgets are applied fresh during execution; they do not change the
cached selection. Selection's `threads` option **is** in the key.

## Bounds, lifetime and concurrency

The per-artifact execution cache retains at most 32 entries and 256 KiB of
parameter **key bytes** in total. Oversized parameter snapshots bypass retention.
This is not a 256 KiB total process-memory guarantee: native metadata also owns
parameter copies and fixed program structures. Their sizes are bounded by the
entry/key limits and the owning artifact's topology. Live requests retain their
own metadata until released and are outside the cache's retention budget.

Cache lookup/insertion uses a lock. Factories run outside it; concurrent misses
may construct equivalent templates, and insertion keeps one winner. Native
requests and reports are never shared between calls. Forked children start with
an empty process-local cache rather than acquiring an inherited lock; this does
not make CUDA execution after fork supported.

`compiled.execution_cache_info()` returns entry count, parameter-key bytes,
limits, hits, misses, evictions and bypasses. `compiled.clear_execution_cache()`
discards retained execution metadata without invalidating already prepared
requests or resetting lifetime counters. Concurrent calls may repopulate it.
The existing frontend `symbol_binds` diagnostic continues to count logical
layout bind requests, including cache hits; it is not a native bind-call count.

## Measurement

Use [the harness](../bench/issue219/request_overhead.py) with both revisions under
the same CPU affinity, precision, resource budgets and source mutations. It
records completed first calls, uninstrumented warmed p50/p95, raw samples,
separate exclusive host-phase diagnostics, runtime hashes and environment.
Compilation of the first frontend KV call is included; weight recipe compilation
occurs before its first-call timer. First calls are in an initialized process,
not a process-cold CUDA comparison. CUDA traces are separate diagnostics.

These measurements concern completed transfers, not complete model inference.
They do not establish a win over Torch or Torch Inductor, and do not change the
blocking API, batch requests or remove output allocation/completion costs.
