//===- Execute.cpp - CPU relocation executors -----------------------------===//

#include "reloc/Execute.h"

#include "reloc/CopyRun.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <thread>

namespace reloc {
namespace {

// Splat the low `es` bytes of `bits` across `count` elements at `dst`.
void fillPattern(uint8_t *dst, uint64_t bits, uint32_t es, int64_t count) {
  for (int64_t e = 0; e < count; ++e)
    std::memcpy(dst + e * es, &bits, es);
}

// Copy the innermost axis (r-1) for indices [iBegin, iEnd): a contiguous
// copyRun when unit-stride on both sides, else a scalar strided loop.
// `innerLo` is the leading pad on the innermost axis (0 if unpadded).
void copyRun1D(const BoundPlan &b, const uint8_t *src, uint8_t *dst,
               int64_t innerLo, int64_t srcOff, int64_t dstOff, int64_t iBegin,
               int64_t iEnd) {
  const size_t d = b.extents.size() - 1;
  const uint32_t es = b.elementSize;
  if (b.srcStrides[d] == 1 && b.dstStrides[d] == 1) {
    copyRun(dst + (dstOff + iBegin + innerLo) * es,
            src + (srcOff + iBegin) * es,
            static_cast<size_t>(iEnd - iBegin) * es);
  } else {
    for (int64_t i = iBegin; i < iEnd; ++i)
      std::memcpy(dst + (dstOff + (i + innerLo) * b.dstStrides[d]) * es,
                  src + (srcOff + i * b.srcStrides[d]) * es, es);
  }
}

// Walk axis `depth` over [iBegin, iEnd), accumulating element offsets, and
// recurse to the innermost run. Only the TOP axis (depth 0) is ranged to
// the chunk; deeper axes always cover their full extent. Correct for r==1
// (the single axis is both top and innermost, ranged to the chunk) and
// r>=2. dst indices are shifted by each axis's leading pad `lo`.
void walk(const BoundPlan &b, const uint8_t *src, uint8_t *dst,
          const std::vector<int64_t> &lo, size_t depth, int64_t iBegin,
          int64_t iEnd, int64_t srcOff, int64_t dstOff) {
  const size_t r = b.extents.size();
  if (depth == r - 1) {
    copyRun1D(b, src, dst, lo[depth], srcOff, dstOff, iBegin, iEnd);
    return;
  }
  for (int64_t i = iBegin; i < iEnd; ++i)
    walk(b, src, dst, lo, depth + 1, 0, b.extents[depth + 1],
         srcOff + i * b.srcStrides[depth],
         dstOff + (i + lo[depth]) * b.dstStrides[depth]);
}

} // namespace

void fillDst(const BoundPlan &bound, void *dstBaseV) {
  if (bound.padRegions.empty())
    return;
  const uint32_t es = bound.elementSize;
  uint64_t bits = bound.padRegions.front().fillBits;
  for (const PadRegion &p : bound.padRegions)
    assert(p.fillBits == bits && "v0 supports a single fill value");
  // Compute the physical (padded) element count directly from
  // extents + padRegions rather than trusting bound.totalBytes: a
  // hand-built BoundPlan (bypassing bind()) may leave totalBytes at its
  // default 0, which would silently turn this into a no-op. This mirrors
  // bind()'s own padded-extent computation, so it agrees with totalBytes
  // whenever bind() did populate it.
  std::vector<int64_t> padded = bound.extents;
  for (const PadRegion &p : bound.padRegions)
    padded[p.axis] += p.lo + p.hi;
  int64_t count = 1;
  for (int64_t e : padded)
    count *= e;
  fillPattern(static_cast<uint8_t *>(dstBaseV), bits, es, count);
}

void gatherChunk(const BoundPlan &bound, const void *srcBaseV, void *dstBaseV,
                 int64_t outerBegin, int64_t outerEnd) {
  const auto *src = static_cast<const uint8_t *>(srcBaseV);
  auto *dst = static_cast<uint8_t *>(dstBaseV);
  const size_t r = bound.extents.size();
  assert(r >= 1 && "bound plan has no axes");

  std::vector<int64_t> lo(r, 0);
  for (const PadRegion &p : bound.padRegions)
    lo[p.axis] = p.lo;

  // Valid cells only for outer indices [outerBegin, outerEnd); pads were
  // handled by fillDst. walk ranges the top axis to the chunk.
  walk(bound, src, dst, lo, /*depth=*/0, outerBegin, outerEnd,
       /*srcOff=*/0, /*dstOff=*/0);
}

void executeH2D(const BoundPlan &bound, const void *srcBase, void *dstBase) {
  fillDst(bound, dstBase);
  gatherChunk(bound, srcBase, dstBase, 0, bound.extents[0]);
}

void executeH2DThreaded(const BoundPlan &bound, const void *srcBase,
                        void *dstBase, unsigned threads) {
  const int64_t outer = bound.extents[0];
  // Fill pads once, single-threaded, before any worker writes valid cells
  // (gatherChunk no longer fills). No race: happens-before all workers.
  fillDst(bound, dstBase);
  if (threads == 0)
    threads = std::thread::hardware_concurrency();
  if (threads == 0)
    threads = 1;
  // Never spawn more workers than outer rows; each row is disjoint in dst
  // (distinct outer index -> distinct dst offset range), so no data race.
  int64_t workers = std::min<int64_t>(threads, outer);
  if (workers <= 1) {
    gatherChunk(bound, srcBase, dstBase, 0, outer);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  int64_t per = (outer + workers - 1) / workers; // ceil
  for (int64_t w = 0; w < workers; ++w) {
    int64_t begin = w * per;
    int64_t end = std::min(begin + per, outer);
    if (begin >= end)
      break;
    auto body = [&bound, srcBase, dstBase, begin, end]() {
      gatherChunk(bound, srcBase, dstBase, begin, end);
    };
    if (w + 1 == workers)
      body(); // run the last partition on this thread
    else
      pool.emplace_back(body);
  }
  for (std::thread &t : pool)
    t.join();
}

ViewDescriptor executeView(const BoundPlan &bound, const void *srcBase) {
  assert(bound.noCopy && "executeView requires no_copy");
  assert(bound.padRegions.empty() && "a no_copy view cannot carry pads");
  return ViewDescriptor{srcBase, bound.extents, bound.srcStrides,
                        bound.elementSize};
}

} // namespace reloc
