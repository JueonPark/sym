# Explicit transfer preparation

`PreparedTransfers` compiles a finite mapping of recipes before serving, then
warms caller-selected shapes through ordinary blocking transfers. It supports
layout H2D/D2H, typed transfers and indexed H2D. It does not capture a Dynamo
graph or compile surrounding model computation; use the application's explicit
`torch.compile` warmup separately when composing with Inductor.

```python
from reloc_torch import CompilerClient, PreparedTransfers

compiler = CompilerClient.from_environment(
    cache_dir="/var/cache/my-service/sym",  # opt-in, trusted local POSIX directory
    cache_max_bytes=64 << 20, cache_max_entries=128,
    cache_namespace="deployment-v1",
)
with PreparedTransfers(
    {"activations": recipe}, compiler=compiler,
    resource_options={
        "max_retained_bytes": 64 << 20,
        "max_live_staging_bytes": 64 << 20,
        "max_typed_retained_bytes": 64 << 20,
        "max_typed_live_bytes": 64 << 20,
        "max_background_workers": 7,
        "max_typed_background_workers": 7,
    },
    transfer_options={"gather_threads": 8, "n_buffers": 2,
                      "n_streams": 1, "pinning": "pinned"},
    max_output_bytes=64 << 20,
) as prepared:
    prepared.initialize_cuda("cuda:0")
    prepared.warm("activations", representative_input, "cuda:0")
    # Binding, source/parameter/index validation and output allocation stay live.
    output = prepared.transfer("activations", new_input, "cuda:0")
    diagnostics = prepared.stats()
retirement_ms = prepared.stats()["retirement_ms"]
```

`warm` executes **exactly one completed transfer** for the supplied example and
discards its output. Calling it for several shapes is explicit; it never times
alternative implementations, selects a policy from measurements, or silently
repeats the example. It initializes the device if `initialize_cuda` was omitted.
Pass `parameters={...}` for typed bindings, `indices=...` for indexed recipes,
and the same `threads`, `policy` and `implementation` used in serving. Defaults
match the existing dispatch API. There is no immutable-weight snapshot: changed
sources, indices and scale values are read on each call, and every returned
output has independent storage. Example tensors are not retained.

Preparation creates a bounded native resource owner. Defaults bound layout
live scratch to 256 MiB and typed live scratch to 64 MiB; each has its existing
separate retention, stream, worker and context limits. Zero/unlimited live
limits are rejected. `max_output_bytes` bounds each fresh output (default
256 MiB); it does not limit outputs the caller retains, the caller's input
storage, parameter snapshots, driver memory, or Torch's caching allocator.
`max_recipes` defaults to 32. The session serializes its calls; use the existing
`TransferResources`/`TransferQueue` interfaces for concurrent submission.

Warming is not a capacity reservation or a guarantee of allocation-free serving.
Changed device, affinity, execution options, pinning class or larger shapes can
replace/grow contexts within the hard limits. Multiple warmed recipes can evict
each other's incompatible contexts. Inspect `stats()["resources"]` before and
after serving; tune the existing context limits for the intended sequence.
Session execution options are copied at construction; to change those options,
create a new session. Invalid shapes/values still fail normal validation.
`close` is idempotent, measures retirement, and follows native completion and
quarantine rules. Fork-inherited sessions reject access before taking locks.

## Compiled-artifact cache

`CompilerClient` leaves caching disabled by default. An explicit `cache_dir`
allows another process/client to reuse a compiled recipe. This also works with
`RelocBackend(compiler=compiler)`, direct `.compile(recipe)` and eager frontends
that accept the client. No environment variable silently enables persistence.

The key covers the complete recipe (including direction and parameter schema),
cache format, frontend source/schema/serializer/binder content, exporter binary
SHA-256 and caller's configuration namespace. Binary stat identity includes inode
and ctime to detect replacements before rehashing. Exporter changes during a
lookup/compile are rejected. Every hit repeats portable artifact admission,
including plan/input digests, schema/wire versions and recipe association.
Corrupt entries become misses; compiler failures/rejections are not persisted.

Only portable plans are cached. Placement decisions, live values, mutable
parameter tensors, requests, streams, allocations and execution handles are
never persisted. The supported exporter links compiler logic into the executable;
if a custom wrapper depends on external configuration, plugins or dynamic
compiler libraries, version those dependencies in `cache_namespace`. This cache
is for a trusted local POSIX filesystem, not an untrusted interchange or network
filesystem. Compatible processes serialize access with an advisory file lock,
compile a missing key once, and publish atomically. Entry/byte bounds evict least
recently used artifacts; oversize plans execute without being cached. Abandoned
partial writes are removed on the next access. Concurrent users of a directory
should use the same bounds. `cache_stats()` reports hits, misses, invalid entries,
evictions and on-disk bytes, and enforces the caller's configured bounds.

## Accounting

Diagnostics separate compiler preparation, resource-owner construction, explicit
CUDA initialization, example execution/resource warming, first real execution,
subsequent execution totals and retirement. Warming time includes binding, fresh
output allocation, CPU/GPU work and completion; it is not allocation-only time.
`first_execution_ms` is the first **serving** call even after explicit warming.
Use `(serving_calls - 1)` to interpret subsequent totals, not as a latency percentile.

Measure the complete preparation envelope externally too: imports, client
construction/cache hashing, caller input creation and driver initialization may
happen outside individual phase timers. Native retained memory is in resource
stats; account separately for process RSS and Torch/driver allocations. Torch's
allocator can retain warm output memory after its tensor is released. Preparation
moves work before readiness; it does not promise a smaller total cost or improved
steady state. Include preparation and eventual retirement in short/long horizon
comparisons, and grant Torch/Inductor the same explicit warmup opportunity.
