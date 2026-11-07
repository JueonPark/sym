# Automatic prepared execution

`RelocBackend(prepared_execution=True)` lowers supported captured H2D regions to
prepared execution. It consumes the artifact produced by the existing FX importer
and compiler, including the compiler's layout, value stages and parameter
coordinates. It does not construct workload-specific recipes or register a
handwritten producer as generated fusion. The compatibility default remains
`False`; qualification results and the admission policy are recorded in
`bench/issue265/`.

```python
import torch
from reloc_torch import RelocBackend, PreparedExecutionPolicy

backend = RelocBackend(
    prepared_execution=PreparedExecutionPolicy(),
    transfer_options={"pinning": "pinned", "n_streams": 1},
)
run = torch.compile(
    lambda x: x.transpose(0, 1).contiguous().to("cuda:0", torch.float16),
    backend=backend, fullgraph=True, dynamic=True,
)
with torch.no_grad():
    result = run(torch.randn(512, 1024))
print(backend.stats()["prepared_execution"])
backend.close()
```

## Admission and execution

The frontend first applies its existing metadata, inference, symbol, extent and
promised-output guards. Empty, unsupported stride/offset/dtype and singleton
specializations retain the original-region behavior. The prepared adapter then
applies a generic byte/reuse policy; rejection runs the saved original once.
There is no runtime timing probe. Size and reuse thresholds are heuristics, not a
universal performance guarantee.

For typed artifacts, native capability determines which rows are legal for the
current shape and exact parameter values. The `device` policy prefers a device
layout, then fewer host stages, then smaller wire/temp requirements among legal
accelerated rows. `row_policy="cpu"` is a qualification control. In particular,
nonzero zero points can exclude a GPU dequantization row even when shape and
parameter addresses are unchanged. This is checked on every new value revision.

Parameter-free typed and layout regions reuse a prepared native group and its
validated metadata. Current source storage and newly allocated output storage
are bound on each call. Indexed regions reuse a descriptor specialization, but
snapshot and validate the current indices and construct an indexed native program
on **every** call. Index program construction and report conversion are counted;
this path does not claim native group reuse.

Current scale/zero-point tensors are snapshotted before revision lookup and
compared again immediately before execution. A prepared call is single-use.
Source metadata, parameter values and index values may not change between
preflight and execution; callers must not concurrently mutate operands during a
transfer. Held outputs remain independent of later calls and resource retirement.
Errors after the launch decision are terminal and never replay the saved graph.

## Lifetimes and bounds

The existing compiler-artifact cache, automatic specialization cache and native
scratch owner have separate lifetimes. Defaults bound the automatic cache to
32 entries and 256 KiB of copied parameter keys. Admission and negative entries
share these bounds. Keys contain plan bytes, bindings, tensor metadata, device
and exact parameter dtype/extents/bytes. They contain no live tensor references.
An oversized parameter revision falls back before insertion. Native copies of
parameters and artifact metadata are additional memory, measured separately.

Eviction drops the cache's reference. A preflighted call retains its entry until
completion, so eviction cannot invalidate that call. Consequently outstanding
caller-owned calls can retain more entries than the cache limit. Owner close
drains all surviving handles; subsequent submission is rejected. Inherited use
is rejected by PID checks before acquiring Python locks. Execution is serialized
within one automatic manager, while independent adapters may share the native
resource owner.

The adapter borrows its existing `TransferResources` for prepared submission.
There is no second scratch/worker budget. Each used device/kind has a private
ordering stream; scratch contexts, staging, workers and internal streams remain
subject to the supplied native resource limits. `FastTransfers(resources=owner)`
also exposes borrowed ownership: closing the wrapper drains its handles without
closing the supplied owner.

Explicit placement, D2H, per-call resources (`transfer_resources=None`), custom
gather pools, explicit multiple streams, automatic thread count and configured
pinning thresholds use the legacy adapter with a recorded bypass reason. A custom
runtime owns its policy and rejects backend-level prepared configuration.
Unconfigured `pinning="auto"` remains conservative/pageable. Disabling prepared
execution preserves the original adapter and its configuration.

## Diagnostics

`stats()["prepared_execution"]` reports attempts at the adapter boundary,
preparations, ready-entry hits, misses, evictions, exclusions, legacy bypasses,
retained entries/key bytes, live parameter snapshot bytes/time, preparation time,
per-call index binding, successful executions and actual native dispatch rows.
Frontend exclusions before this boundary remain in the existing fallback and
capture-exclusion maps: these counts describe different events and must not be
added together as a request count.

`preparation_ns` includes snapshot and cache/admission work, including excluded
requests, but excludes earlier frontend guards and compilation. Snapshot timing
is a subset, not additive. Parameter rechecks remain included in completed-call
latency. Dynamo/Inductor/plan compilation counts are independent of preparation.

Typed execution/payload/dispatch totals merge compact native aggregates into the
backend snapshot. Parameter-free prepared groups need no detailed report
materialization. Indexed calls still materialize one report each. Per-call legacy
staging explanations remain legacy-only; prepared staging is exposed through
aggregate native resource counters and explicitly marked by `staging_reporting`.
This prevents a missing detailed explanation from being presented as zero staging
work. Close retains cumulative execution counters and clears live cache gauges.
