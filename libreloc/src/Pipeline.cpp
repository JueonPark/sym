//===- Pipeline.cpp - Strategy-4 chunked pinned/stream pipeline -----------===//

#include "reloc/Pipeline.h"

#include "reloc/ChunkSchedule.h"
#include "reloc/Execute.h"
#include "reloc/PinnedBufferPool.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace reloc {
namespace {

// Splat the single pad pattern across a contiguous staging window (mirrors
// fillDst restricted to one byte window). No-op without pads.
void fillStagingWindow(const BoundPlan &b, void *staging, size_t bytes) {
  if (b.padRegions.empty())
    return;
  const uint32_t es = b.elementSize;
  uint64_t bits = b.padRegions.front().fillBits;
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

// Temporary stub; implemented in Task 5.
void executeD2HPipelined(const BoundPlan &, const void *, void *, CopyBackend &,
                         int, size_t) {}

} // namespace reloc
