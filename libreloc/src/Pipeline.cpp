//===- Pipeline.cpp - Strategy-4 chunked pinned/stream pipeline -----------===//

#include "reloc/Pipeline.h"

#include "reloc/ChunkSchedule.h"
#include "reloc/Execute.h"
#include "reloc/PinnedBufferPool.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>

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

} // namespace

void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  const int nStreams = backend.numQueues();

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(); // blocks on this buffer's prior copy event
    void *staging = pool.buffer(i);

    fillStagingWindow(bound, staging, c.bytes);
    if (c.validEnd > c.validBegin)
      gatherChunk(bound, srcBase, rebase(staging, bound, c.paddedBegin),
                  c.validBegin, c.validEnd);

    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    backend.copyAsync(q,
                      static_cast<uint8_t *>(deviceDst) + c.byteOffset, staging,
                      c.bytes, CopyDir::HostToDevice);
    pool.setEvent(i, backend.recordEvent(q));
  }
  pool.drain();
}

void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  const int nStreams = backend.numQueues();

  struct InFlight {
    int buf;
    EventHandle ev;
    size_t chunk;
  };
  std::deque<InFlight> inflight;

  // Wait for a chunk's D2H copy to land, then scatter its valid cells from the
  // staging buffer into srcBaseV. Frees the staging buffer for reuse.
  auto scatterOne = [&](const InFlight &f) {
    const Chunk &c = sched.chunks[f.chunk];
    backend.waitEvent(f.ev);
    if (c.validEnd > c.validBegin)
      scatterChunk(bound, rebase(pool.buffer(f.buf), bound, c.paddedBegin),
                   srcBaseV, c.validBegin, c.validEnd);
  };

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(); // blocks on this buffer's prior copy event
    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    backend.copyAsync(q, pool.buffer(i),
                      static_cast<const uint8_t *>(deviceSrc) + c.byteOffset,
                      c.bytes, CopyDir::DeviceToHost);
    EventHandle ev = backend.recordEvent(q);
    pool.setEvent(i, ev);
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

} // namespace reloc
