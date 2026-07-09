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

  // Partitioning by outer index is race-free only when distinct outer rows
  // provably write disjoint dst byte ranges. With all strides >= 0 (an
  // invariant bind() guarantees), the dst offsets touched while holding
  // the outer index fixed at i0 span [base(i0), base(i0) + innerSpan]
  // where innerSpan = sum over inner axes k in [1, rank) of
  // (extents[k]-1)*dstStrides[k] (the per-axis pad shift `lo` is identical
  // for every outer row and cancels out of this disjointness check, so we
  // can ignore it). Adjacent rows i0 and i0+1 are guaranteed disjoint iff
  // dstStrides[0] >= innerSpan + 1. This is a conservative (sufficient,
  // not necessary) test: canonical bind() output is dense/row-major-ish
  // and always satisfies it, but a hand-built BoundPlan can violate it
  // (e.g. dstStrides[0] smaller than the inner block span), in which case
  // distinct outer indices alias the same dst bytes and splitting them
  // across workers would race. When the check fails we fall back to a
  // single gatherChunk call over the whole outer range -- i.e. exactly
  // executeH2D's single-thread path -- so the result stays deterministic
  // and bit-identical to executeH2D instead of racy UB.
  int64_t innerSpan = 0;
  for (size_t k = 1; k < bound.dstStrides.size(); ++k)
    innerSpan += (bound.extents[k] - 1) * bound.dstStrides[k];
  const bool rowsDisjoint = bound.dstStrides[0] >= innerSpan + 1;

  int64_t workers = std::min<int64_t>(threads, outer);
  if (!rowsDisjoint || workers <= 1) {
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
    if (w + 1 == workers) {
      body(); // run the last partition on this thread
    } else {
      // Exceptions are enabled for this target (see libreloc/README.md's
      // "Linkage contract"): if std::thread's constructor throws (e.g.
      // resource exhaustion) partway through this loop, join every
      // already-spawned worker before rethrowing so stack unwinding does
      // not destroy a still-joinable std::thread and call
      // std::terminate().
      try {
        pool.emplace_back(body);
      } catch (...) {
        for (std::thread &t : pool)
          if (t.joinable())
            t.join();
        throw;
      }
    }
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
