# Typed CPU transforms with chunked H2D

Individual `cpu_reference` H2D transfers and `cpu_stages_cuda_stages@k`
transfers produce wire-format chunks directly into a retained staging ring.
After the workers finish chunk n, the driver submits its copy and starts the
next CPU transform. A slot's previous copy event must complete before any
worker overwrites that slot. The call returns only after its work completes.

The CPU/GPU split still retains a device wire tensor and launches its remaining
GPU stages after all chunks arrive. This change overlaps CPU transformation
with DMA; it does not pipeline the remaining GPU kernels.

```python
from reloc_torch import TransferResources, dispatch

with TransferResources(max_typed_retained_bytes=64 << 20,
                       max_typed_live_bytes=64 << 20) as resources:
    request = dispatch.prepare_typed_transfer(
        compiled, source, "cuda:0", implementation="cpu_reference", threads=8)
    result = dispatch.execute_typed_transfer(
        request, resources=resources, gather_threads=8,
        pinning="pinned", n_buffers=2, n_streams=1)
```

`TransferResources` retains the staging allocations and worker pool across
calls. Pinning allocation cost remains inside the first completed call. With
no owner, staging is ephemeral. All native scratch allocations, including the
device wire tensor for a split, count against the owner's live/retained limits.
Outputs are separately owned Torch allocations and never alias staging.

## Configuration and controls

- `pipeline=True` is the default for individual typed transfers.
  `pipeline=False` retains the previous whole-host-buffer path for comparisons.
- `chunk_size=None` keeps wire payloads up to 4 MiB whole. Larger payloads
  target 32 chunks, clamped to 1–64 MiB per chunk. An explicit positive byte
  target overrides this heuristic, including for small transfers. Whole
  physical outer rows are indivisible; an individual row can exceed the target.
- Typed Python dispatch defaults to two buffers; layout-only defaults are
  unchanged. `n_buffers=1` uses exactly the same chunk schedule and worker/pinning
  configuration, but waits for each copy before producing the next chunk. Four
  buffers are supported, but were not faster consistently in qualification.
  Raw C++ callers set `TransferOptions.nBuffers` explicitly (its shared default
  remains four).
- `gather_threads` or a supplied pool controls CPU parallelism. This change
  does not increase the caller's worker budget. The qualification uses eight
  CPU threads and one copy stream; those are measured settings, not universal
  performance guarantees.
- Existing pinning policy applies to each ring allocation. `pinning="auto"`
  without `min_pinned_bytes` remains pageable. A calibrated/configured threshold
  must fit the actual chunk size and the owner's retention budget to select
  pinned memory. Pageable copies need not overlap CPU work with DMA; smaller
  host working sets can still change their completed latency.

The new `pipeline` and `chunk_size` keywords belong to the direct typed API.
`TransportAdapter` uses the typed defaults; its shared layout/typed option set
continues to expose the existing buffer, worker, stream and pinning controls.
Row selection and its cost model are unchanged. These measurements do not
establish that the automatic policy picks the best implementation for every
workload.

## Semantics and fallback

Chunks cover disjoint, contiguous physical destination row windows, including
padding. CPU execution rebases writes to each local allocation while retaining
global output coordinates for per-channel parameters. Pad fills are converted
at the selected wire boundary using the same arithmetic as whole execution.
SIMD casts operate on the same valid contiguous runs, including scalar tails.
Parameter snapshots, shape binding and request ownership remain per invocation.

If physical rows cannot be partitioned safely, execution retains the whole
buffer path. Single-chunk schedules also use that path. Transfer groups keep
their existing whole-member staging and shared completion barrier. D2H and
GPU-producing rows are unaffected by this feature.

The report adds `host_pipeline` (`chunked`, `whole_disabled`,
`whole_single_chunk`, `whole_non_partitionable`, `whole_group`, or
`not_applicable`), `host_chunks`, `host_chunk_bytes` (maximum logical slot size)
and `host_buffers`. Actual retained bytes can exceed logical slot bytes because
the arena reserves allocation growth headroom; inspect
`resources.stats()['typed']` and `report['staging']` for capacities and pinning.

Caller-stream ordering precedes private work. On partial failure, pending
copies are drained before scratch or tensor owners are released. If completion
cannot be proven, the arena, every staging allocation, and source/output owners
are quarantined together; the resource owner cannot be reused. No error path
returns partially copied outputs or retries the consumed request.

## Qualification

`test_typed_pipeline.py` exercises exact oracle bytes, shape/value/parameter
changes, padding, per-channel operations, CPU/GPU splits, one/two/four slots,
pinning policies, caller-stream ordering, memory limits and injected copy/event
failures. Native tests check local window bounds and the non-partitionable
fallback against the established whole-buffer oracle.

`bench/issue221/typed_pipeline.py` compares the prior runtime, the whole-buffer
control, identical one/two/four-buffer schedules, eager Torch, Torch with a
retained pinned conversion destination, and Inductor-compiled CPU transforms
followed by the same wire-format upload. Both cast stages in the split case
remain observable; no baseline bypasses intermediate f16 rounding or sends a
different payload. Timings include binding, allocation and completion, but
exclude recipe/Inductor compilation, input changes and correctness checks.

Native NVTX ranges are enabled only with `RELOC_ENABLE_NVTX=ON`. The
`reloc.typed.transform.work` range runs inside each actual CPU worker;
`reloc.typed.fill` covers padding initialization. Their integer payload is the
chunk index. `reloc.typed.h2d.submit` identifies the matching CUDA submission.
`bench/issue221/analyze_trace.py` correlates these ranges with actual GPU memcpy
activities by CUDA correlation ID, unions parallel worker intervals, and
checks overlap against the **next** chunk. A one-buffer trace is the
serialization control. Profiling runs are separate from headline timings.

See the [measurement report](../bench/results/typed-h2d-221/README.md) for
configuration sweeps, independent timing rounds, memory footprints, correlated
overlap evidence and remaining regressions. Raw profiler databases and verbose
test logs are deliberately excluded from the PR.
