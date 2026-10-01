# Estimated code changes for transfer resource reuse

This is the historical change estimate for the [resource retention design](transfer-resource-reuse.md)
and [issue #165](https://github.com/JueonPark/sym/issues/165). The snippets are
schematic review examples, not an applied or compilable implementation patch.
Use the [enablement record](transfer-resource-enablement.md) and linked user
guide for the delivered implementation, current APIs and measured results.
Names and line counts below describe the original proposal.

## Compatibility with the merged transpose kernel

[PR #164](https://github.com/JueonPark/sym/pull/164) merged as `13d3447`.
Comparing its committed tree with the design's original baseline, `d019d1b`,
produces no differences. Resource reuse can be developed directly on the
merged main branch.

The execution chain remains:

```text
frontend -> leased transfer resources -> H2D chunk pipeline
         -> gatherChunk -> tryGatherTranspose32 -> tiled CPU transpose
```

`Transpose.cpp`, `Transpose.h`, and the fast-path call in `Execute.cpp` need
no planned changes. `Pipeline.cpp` still gathers chunk n+1 while a previous
chunk transfers. The design changes ownership, capacity checks, and error
handling around that work. CMake source lists, test lists, and documentation
are additive edits on the merged baseline.

There is a separate compatibility consideration: the proposed `quiesce()`
method changes the native `CopyBackend` interface. Its implementations and
C++ consumers must be rebuilt together. Keep the existing transfer overloads
and serialized plan formats; this interface change is unrelated to the
transpose kernel.

## Approximate diff size

These are planning ranges for the full design, including bounded admission,
H2D and forward D2H, close/clear, diagnostics, and exceptional ownership.
Counts mean added/deleted lines, not net lines. They exclude this proposal,
other prose documentation, and generated files. Expect roughly 30-35 source,
test, benchmark, and build files, depending on helper placement.

| Area | Main files | Added lines | Deleted/refactored lines |
| --- | --- | ---: | ---: |
| Native resource ownership and cache | New `TransferResources.h/.cpp` | 580-900 | 0 |
| Transfer and pipeline integration | `Transfer.h/.cpp`, `Pipeline.h/.cpp` | 140-240 | 70-130 |
| Completion and lifecycle fixes | Backend headers, `CudaBackend.cu`, `HostBackend.cpp`, `PinnedBufferPool.h/.cpp`, `GatherPool.cpp` | 100-180 | 20-50 |
| Python native bindings | `PyTransfer.h/.cpp` | 180-300 | 20-40 |
| Frontend ownership and configuration | New `reloc_torch/resources.py`; `transport.py`, `runtime.py`, `backend.py`, package exports | 280-420 | 20-50 |
| Native and Python tests | Resource, transfer, backend, worker, and CUDA frontend tests | 700-1100 | Small adjustments |
| Reproducible benchmark | New `bench/transfer_resource_reuse.py` | 200-350 | 0 |
| Build registration | Runtime and test CMake lists | 20-40 | 0 |
| **Total** | | **2200-3530** | **About 130-270, plus test adjustments** |

Production C++/Python accounts for approximately 1280-2040 added lines.
Tests and benchmarks account for another 900-1450. Most complexity comes
from bounded concurrency, resource accounting, and failure ownership, rather
than changes to the copy algorithm.

## New native ownership surface

The header would expose the cache policy and cached execution interface;
context and lease bookkeeping should stay private to the implementation.
Omitted definitions below include normalized execution options, stats, backend
factory injection, and the error/completion result.

```diff
+++ libreloc/include/reloc/TransferResources.h
+struct TransferResourceLimits {
+  size_t maxRetainedBytes = 256ull << 20;
+  size_t maxContexts = 4;
+  size_t maxContextsPerDevice = 2;
+  size_t maxBackgroundWorkers = 64;
+  std::optional<size_t> maxLiveStagingBytes;
+  std::optional<std::chrono::milliseconds> acquireTimeout;
+};
+
+class TransferResourceCache {
+public:
+  explicit TransferResourceCache(TransferResourceLimits limits);
+  ~TransferResourceCache();
+  TransferResourceCache(const TransferResourceCache &) = delete;
+  TransferResourceCache &operator=(const TransferResourceCache &) = delete;
+  void clear();
+  std::optional<TransferError> close();
+  TransferResourceStats stats() const;
+private:
+  struct Impl;
+  std::unique_ptr<Impl> impl_;
+};
+
+TransferOutcome executeTransferCached(
+    TransferRequest &request, TransferResourceCache &resources,
+    const CachedTransferOptions &options,
+    std::shared_ptr<void> bufferOwners);
```

`TransferResources.cpp` implements compatibility lookup, reservations,
exclusive leases, capacity growth, LRU eviction, waiters, clear/close
generations, failure retirement, and accounting. Construction remains lazy:
making the cache does not create streams, pinned memory, or workers.

## Transfer execution and the existing pipeline

Extract one checked execution core shared by ephemeral and cached paths.
Keep source/destination validation, single-use request semantics, caller-stream
ordering, and forward-D2H behavior in that core.

```diff
--- libreloc/src/Transfer.cpp
+++ libreloc/src/Transfer.cpp
@@ Resource acquisition for the cached path
-  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
-  GatherPool gather(options.gatherThreads);
+  auto requirements = describeTransfer(request, options);
+  auto acquired = resources.acquire(requirements, bufferOwners);
+  // On rejection: return the admission error, without launching work.
+  auto lease = takeLease(acquired);
+  auto &backend = lease.backend();
+  auto &pool = lease.staging();
+  auto *gather = lease.gather();
@@ Shared execution
-  executeH2DPipelined(bound, src, dst, backend, pool,
-                      options.chunkSizeOverride, &gather);
+  auto result = executePreparedTransfer(
+      request, backend, pool, gather, requirements, options);
+  return lease.finish(result);
```

This illustrates moving allocations out of the per-call path, not deleting
the legacy executor. `describeTransfer` validates capacity arithmetic and
fixes the schedule before acquisition. `lease.finish` handles event-proven
success, queue quiescence after an error, and owner retention if completion
remains unknown. It cannot unconditionally return a failed context to the cache.

The pipeline gains a checked internal entry point accepting the precomputed
schedule. Runtime capacity checks replace reliance on an assertion for this
path. The existing public wrappers delegate to the same core.

```diff
--- libreloc/src/Pipeline.cpp
+++ libreloc/src/Pipeline.cpp
@@ Checked core receives schedule and active slot count
-  ChunkSchedule sched = planChunks(bound, pool.nBuffers(), chunkSizeOverride);
-  assert(pool.bufferBytes() >= sched.maxChunkBytes);
+  if (!pool.valid() || pool.bufferBytes() < sched.maxChunkBytes)
+    return insufficientStaging();
@@ Chunk execution
   int i = pool.acquire();
+  if (backend.failed())
+    return pipelineFailure();
   // Fill staging and dispatch gatherChunk over this chunk's row ranges.
   // gatherChunk continues to select the merged transpose kernel.
   backend.copyAsync(q, deviceChunk, staging, c.bytes, CopyDir::HostToDevice);
-  pool.setEvent(i, backend.recordEvent(q));
+  EventHandle event = backend.recordEvent(q);
+  if (event == 0 || backend.failed())
+    return pipelineFailure(); // owning execution guard quiesces the queues
+  pool.setEvent(i, event);
@@ End of request
   pool.drain();
+  return backend.failed() ? pipelineFailure() : pipelineComplete();
```

The complete implementation also checks failure before submission and validates
the active slot count. There is no new per-chunk stream synchronization after
recording an event. The next chunk can begin gathering immediately into another
available staging buffer; the completion drain remains at the end.

## Backend and worker lifetime fixes

```diff
--- libreloc/include/reloc/Backend.h
+++ libreloc/include/reloc/Backend.h
+enum class QueueCompletion { Complete, Unknown };
 class CopyBackend {
 public:
+  // Attempts completion even when failed() already holds a sticky error.
+  virtual QueueCompletion quiesce() = 0;
 };
```

CUDA implements this over private streams and preserves the original error.
HostBackend waits for both queued and executing work and retires consumed
event records. Staging destruction is allowed only after a successful
completion proof. A failed event recording after copy submission must therefore
reach the execution guard, not buffer reuse.

For externally supplied gather pools, qualify close racing with dispatch entry
in both debug and release builds. Already-running jobs finish; remaining ranges
can run inline if the external owner closes its workers. Cache-owned worker
pools remain alive until all leases using them complete.

## Binding and frontend changes

`PyTransfer.cpp` registers the resource class, validates arguments, creates an
owner token before releasing the GIL, and selects the cached or ephemeral
execution path. The cached token retains both tensor allocations across error
cleanup. Admission and close release the GIL; normal owner destruction reacquires
it. Python package exports include the new native class and facade.

```diff
--- libreloc/python/reloc_torch/transport.py
+++ libreloc/python/reloc_torch/transport.py
-def execute_transfer(request, *, n_buffers=4, n_streams=2,
-                     gather_threads=1, gather_pool=None):
+def execute_transfer(request, *, n_buffers=4, n_streams=2,
+                     gather_threads=1, gather_pool=None, resources=None):
     # Existing source recheck, fresh output allocation and make_transfer.
     pyreloc.execute_transfer(
         native,
+        resources=None if resources is None else resources.native,
+        owners=None if resources is None else (request.source, out),
         caller_stream=compat.cuda_stream_handle(cuda_device),
         n_buffers=n_buffers,
         n_streams=n_streams,
         gather_threads=gather_threads,
         gather_pool=gather_pool,
     )
```

`TransportAdapter` resolves `AUTO` to an owned cache, records borrowed ownership
otherwise, and forwards it only through the layout-transfer branch:

```diff
--- libreloc/python/reloc_torch/runtime.py
+++ libreloc/python/reloc_torch/runtime.py
@@ TransportAdapter.execute, after the existing typed-dispatch branch
-    return self._module.execute_transfer(call.request)
+    return self._module.execute_transfer(
+        call.request, resources=self._resources, **self._transfer_options,
+    )
```

`RelocBackend` exposes the policy, creates its default adapter with those
settings, reports resource stats, and closes an adapter it owns after
invalidating entries. Borrowed adapters and caches remain caller-owned.
`resources.py` provides explicit close/clear, context-manager support, PID
checks, argument validation, and immutable configuration. New options must be
thread-safe across lazy initialization and close races.

## Implementation batches

The [implementation plan](transfer-resource-implementation-plan.md) divides
these broad areas into eight tracked subissues with acceptance checks and
stacked-PR dependencies:

1. R1–R2: backend completion, native resource ownership, checked execution, and
   failure-injection tests. Keep the existing kernel and pipeline dispatch.
2. R3–R5: bounded cache admission, Python bindings, and opt-in direct/compiled/
   eager reuse, including capacity changes, close races, and retained outputs.
3. R6–R8: the committed benchmark, integrated qualification, and documentation;
   make `AUTO` the frontend default only after both qualification gates pass.

Full acceptance includes a profiler trace showing gather of chunk n+1 during
DMA of chunk n, with the same chunk size and worker budget as the control.
The CPU transpose tests from #164 remain part of regression coverage.
