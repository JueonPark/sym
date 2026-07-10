//===- Pipeline.h - Strategy-4 chunked pinned/stream pipeline ---*- C++ -*-===//
//
// executeH2DPipelined gathers each dst chunk into an event-recycled pinned
// staging buffer and issues an async copy to the device, round-robining over
// the backend's queues. Output is byte-identical to executeH2D.
// executeD2HPipelined is the symmetric inverse (Task 5). Written once against
// CopyBackend, so HostBackend exercises the whole algorithm in CI.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PIPELINE_H
#define RELOC_PIPELINE_H

#include "reloc/Backend.h"
#include "reloc/Bind.h"

#include <cstddef>

namespace reloc {

class PinnedBufferPool;
class GatherPool;

/// Per-worker byte floor for parallel per-chunk gather/scatter (issue #65):
/// a worker's sub-range must cover at least this many staging bytes
/// (converted to rows via the chunk row size), so tiny chunks run inline
/// instead of shredding into per-row tasks. E2 retunes via this constant.
constexpr size_t kMinGatherBytesPerWorker = 1ull << 20; // 1 MiB

/// Strategy 4 (H2D). `deviceDst` is sized bound.totalBytes (a device pointer
/// for CudaBackend, plain host for HostBackend). `nBuffers` in {1,2,4};
/// number of streams = backend.numQueues(). chunkSizeOverride == 0 uses the
/// size heuristic. gatherThreads partitions each chunk's gather across a
/// per-call GatherPool (0 = hardware concurrency); 1 is bit-identical to
/// the pre-D1 pipeline (no pool, no threads). Byte-identical to executeH2D.
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0,
                         unsigned gatherThreads = 1);

/// As above with a CALLER-OWNED GatherPool so long-lived callers (pybind)
/// amortize worker startup across calls.
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather);

/// Strategy 4 (H2D) with a CALLER-OWNED staging pool: buffers are reused
/// across calls instead of allocated per call, so steady-state latency is
/// measurable (issue #47's benchmark) and long-lived callers amortize
/// pinned allocation. Buffer count comes from pool.nBuffers().
/// Precondition (asserted): pool.bufferBytes() >=
/// planChunks(bound, pool.nBuffers(), chunkSizeOverride).maxChunkBytes,
/// and `pool` was created against this `backend`. gather == nullptr gathers
/// inline (single thread). Byte-identical to the pool-per-call overload.
/// (No D2H twin yet -- add when a caller needs it.)
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend,
                         PinnedBufferPool &pool, size_t chunkSizeOverride = 0,
                         GatherPool *gather = nullptr);

/// Strategy 4 inverse (D2H). Added in Task 5.
void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0);

} // namespace reloc

#endif // RELOC_PIPELINE_H
