//===- Pipeline.cpp - Strategy-4 chunked pinned/stream pipeline -----------===//

#include "reloc/Pipeline.h"
#include "Trace.h"
#include "TransferInternal.h"

#include "reloc/ChunkSchedule.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/PinnedBufferPool.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace reloc {
namespace {

// Splat the single pad pattern across a contiguous staging window (mirrors
// fillDst restricted to one byte window). No-op without pads.
void fillStagingWindow(const BoundPlan &b, void *staging, size_t bytes) {
  if (b.padRegions.empty())
    return;
  const uint32_t es = b.elementSize;
  uint64_t bits = b.padRegions.front().fillBits;
  for (const PadRegion &p : b.padRegions)
    assert(p.fillBits == bits && "v0 supports a single fill value");
  auto *p = static_cast<uint8_t *>(staging);
  for (size_t off = 0; off + es <= bytes; off += es)
    std::memcpy(p + off, &bits, es);
}

// Byte pointer at which dst element offset 0 would land for a staging window
// that starts at physical outer row `paddedBegin` (see gatherChunk's rebase
// contract in Execute.h).
uint8_t *rebase(void *staging, const BoundPlan &b, int64_t paddedBegin) {
  return static_cast<uint8_t *>(staging) -
         static_cast<size_t>(paddedBegin) *
             static_cast<size_t>(b.dstStrides[0]) * b.elementSize;
}

// Run op over rows [begin, end) through `gather` when parallelism is safe
// and the pool is real; inline otherwise. The per-worker floor is
// kMinGatherBytesPerWorker expressed in rows of this chunk schedule's
// rowBytes, so a worker never receives less than ~1 MiB of gather work.
void dispatchRows(GatherPool *gather, bool parallelSafe, int64_t rowBytes,
                  int64_t begin, int64_t end,
                  const std::function<void(int64_t, int64_t)> &op) {
  if (!gather || !parallelSafe || gather->threadCount() <= 1) {
    op(begin, end);
    return;
  }
  const int64_t minRows =
      std::max<int64_t>(1, static_cast<int64_t>(kMinGatherBytesPerWorker) /
                               std::max<int64_t>(1, rowBytes));
  gather->parallelFor(begin, end, minRows, op);
}

// Sufficient condition for parallel scatter: valid index tuples map
// injectively to src element offsets, so workers holding disjoint outer
// ranges write disjoint src bytes -- interleaved (e.g. transposed src) is
// fine, colliding is not. Standard mixed-radix test over stride-sorted
// axes; broadcast/degenerate strides (<= 0 with extent > 1) fail.
// Conservative: false only costs parallelism (scatter runs inline), never
// correctness.
bool srcRowsWriteDisjoint(const BoundPlan &b) {
  std::vector<std::pair<int64_t, int64_t>> ax; // (stride, extent), extent > 1
  for (size_t k = 0; k < b.extents.size(); ++k) {
    if (b.extents[k] <= 1)
      continue;
    if (b.srcStrides[k] <= 0)
      return false;
    ax.emplace_back(b.srcStrides[k], b.extents[k]);
  }
  std::sort(ax.begin(), ax.end());
  int64_t span = 0; // max element offset reachable by smaller-stride axes
  for (const auto &se : ax) {
    if (se.first <= span)
      return false;
    span += (se.second - 1) * se.first;
  }
  return true;
}

void d2hPipelinedImpl(const BoundPlan &bound, const void *deviceSrc,
                      void *srcBaseV, CopyBackend &backend, int nBuffers,
                      size_t chunkSizeOverride, GatherPool *gather) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  const int nStreams = backend.numQueues();
  if (!pool.valid() || backend.failed() || nStreams < 1)
    throw std::runtime_error("inverse pipeline staging/backend unavailable");
  const bool parallelSafe = srcRowsWriteDisjoint(bound);

  struct InFlight {
    int buf;
    EventHandle ev;
    size_t chunk;
  };
  std::deque<InFlight> inflight;

  // Wait for a chunk's D2H copy to land, then scatter its valid cells from the
  // staging buffer into srcBaseV. Frees the staging buffer for reuse. The
  // dispatchRows barrier returns only after every sub-range completed, so the
  // buffer-reuse reasoning below is unchanged by parallel scatter.
  auto scatterOne = [&](const InFlight &f) {
    const Chunk &c = sched.chunks[f.chunk];
    backend.waitEvent(f.ev);
    if (backend.failed())
      throw std::runtime_error("inverse copy completion failed: " +
                               backend.error());
    if (c.validEnd > c.validBegin) {
      const uint8_t *dstBase = rebase(pool.buffer(f.buf), bound, c.paddedBegin);
      dispatchRows(gather, parallelSafe, sched.rowBytes, c.validBegin,
                   c.validEnd, [&](int64_t rb, int64_t re) {
                     scatterChunk(bound, dstBase, srcBaseV, rb, re);
                   });
    }
  };

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(); // blocks on this buffer's prior copy event
    if (i < 0)
      throw std::runtime_error("inverse staging reuse failed: " +
                               backend.error());
    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    pool.markPending();
    backend.copyAsync(q, pool.buffer(i),
                      static_cast<const uint8_t *>(deviceSrc) + c.byteOffset,
                      c.bytes, CopyDir::DeviceToHost);
    EventHandle ev = backend.recordEvent(q);
    pool.setEvent(i, ev);
    if (!ev || backend.failed())
      throw std::runtime_error("inverse copy submission failed: " +
                               backend.error());
    inflight.push_back({i, ev, k});
    // Keep at most pool.nBuffers() copies outstanding; drain the oldest
    // (which uses the buffer we are about to reuse next) before it is
    // overwritten. The deferred scatterOne(front) below (waitEvent + scatter)
    // runs synchronously on this single driver thread strictly before the
    // next acquire() reuses that same buffer, so the buffer is fully drained
    // and scattered before any new D2H copy overwrites it.
    if (static_cast<int>(inflight.size()) == pool.nBuffers()) {
      scatterOne(inflight.front());
      inflight.pop_front();
    }
  }
  for (const InFlight &f : inflight)
    scatterOne(f);
  pool.drain();
}

} // namespace

std::optional<TransferError> detail::executeH2DPrepared(
    const BoundPlan &bound, const void *srcBase, void *deviceDst,
    CopyBackend &backend, PinnedBufferPool &pool, const ChunkSchedule &sched,
    int activeSlots, GatherPool *gather, TransferCompletion &completion,
    const StackedSource *stacked) {
  if (!pool.usesBackend(backend) || !pool.valid() || activeSlots < 1 ||
      activeSlots > pool.nBuffers() || sched.chunks.empty() ||
      pool.bufferBytes() < sched.maxChunkBytes)
    return TransferError{"insufficient_capacity",
                         "invalid staging for schedule"};
  const int nStreams = backend.numQueues();
  if (nStreams < 1 || backend.failed())
    return TransferError{"backend_failure",
                         "backend unavailable: " + backend.error()};
  for (const auto &c : sched.chunks)
    if (c.byteOffset < 0 || size_t(c.byteOffset) > size_t(bound.totalBytes) ||
        c.bytes > size_t(bound.totalBytes) - size_t(c.byteOffset) ||
        c.bytes > sched.maxChunkBytes || c.validBegin < 0 ||
        c.validEnd < c.validBegin || c.validEnd > bound.extents[0])
      return TransferError{"insufficient_capacity",
                           "chunk exceeds transfer capacity"};
  // A serialized schedule means outer rows are NOT provably disjoint in dst
  // (see planChunks): partitioning the whole-tensor chunk across workers
  // would race, so gather stays inline -- same fallback executeH2DThreaded
  // makes.
  const bool parallelSafe = !sched.serialized;

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(activeSlots); // gate reuse on the prior copy
    if (i < 0 || backend.failed())
      return TransferError{"backend_failure",
                           "staging reuse failed: " + backend.error()};
    void *staging = pool.buffer(i);

    fillStagingWindow(bound, staging, c.bytes);
    if (c.validEnd > c.validBegin) {
      uint8_t *dstBase = rebase(staging, bound, c.paddedBegin);
      // The counting barrier inside dispatchRows completes before copyAsync
      // may read the staging bytes.
      dispatchRows(gather, parallelSafe, sched.rowBytes, c.validBegin,
                   c.validEnd, [&](int64_t rb, int64_t re) {
                     detail::TraceRange work("reloc.gather.work", k);
                     if (stacked)
                       gatherChunk(bound, *stacked, dstBase, rb, re);
                     else
                       gatherChunk(bound, srcBase, dstBase, rb, re);
                   });
    }

    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    completion = TransferCompletion::Unknown;
    pool.markPending();
    {
      detail::TraceRange submit("reloc.h2d.submit", k);
      backend.copyAsync(q, static_cast<uint8_t *>(deviceDst) + c.byteOffset,
                        staging, c.bytes, CopyDir::HostToDevice);
    }
    if (backend.failed())
      return TransferError{"backend_failure",
                           "copy submission failed: " + backend.error()};
    const EventHandle event = backend.recordEvent(q);
    pool.setEvent(i, event);
    if (!event || backend.failed())
      return TransferError{"backend_failure",
                           "event recording failed: " + backend.error()};
  }
  pool.drain();
  if (backend.failed())
    return TransferError{"backend_failure",
                         "copy completion failed: " + backend.error()};
  completion = TransferCompletion::Complete;
  return std::nullopt;
}

namespace {
// Compatibility wrappers use the same loop and error-completion guard. These
// raw-pointer APIs cannot retain the caller's backend or tensor allocations.
void runH2DPrepared(const BoundPlan &bound, const void *src, void *dst,
                    CopyBackend &backend, PinnedBufferPool &pool,
                    const ChunkSchedule &schedule, GatherPool *gather) {
  TransferCompletion completion = TransferCompletion::NotLaunched;
  std::optional<TransferError> error;
  try {
    detail::CompletionGuard guard(backend, completion);
    error = detail::executeH2DPrepared(bound, src, dst, backend, pool, schedule,
                                       pool.nBuffers(), gather, completion);
  } catch (...) {
    if (completion == TransferCompletion::Complete)
      pool.markComplete();
    throw;
  }
  if (completion == TransferCompletion::Complete)
    pool.markComplete();
  if (error)
    throw std::runtime_error((completion == TransferCompletion::Unknown
                                  ? "completion_unknown: "
                                  : "") +
                             error->message);
}
} // namespace

void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend,
                         PinnedBufferPool &pool, size_t chunkSizeOverride,
                         GatherPool *gather) {
  const auto sched = planChunks(bound, pool.nBuffers(), chunkSizeOverride);
  runH2DPrepared(bound, srcBase, deviceDst, backend, pool, sched, gather);
}

void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, unsigned gatherThreads) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  if (gatherThreads == 1) {
    // Regression guard (issue #65): threads == 1 must not even construct a
    // GatherPool -- bit-identical behavior to the pre-D1 pipeline.
    runH2DPrepared(bound, srcBase, deviceDst, backend, pool, sched, nullptr);
    return;
  }
  GatherPool gather(gatherThreads);
  runH2DPrepared(bound, srcBase, deviceDst, backend, pool, sched, &gather);
}

void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  runH2DPrepared(bound, srcBase, deviceDst, backend, pool, sched, &gather);
}

void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, unsigned gatherThreads) {
  if (gatherThreads == 1) {
    d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                     chunkSizeOverride, /*gather=*/nullptr);
    return;
  }
  GatherPool gather(gatherThreads);
  d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                   chunkSizeOverride, &gather);
}

void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather) {
  d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                   chunkSizeOverride, &gather);
}

} // namespace reloc
