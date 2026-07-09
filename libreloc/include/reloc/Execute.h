//===- Execute.h - CPU relocation executors ---------------------*- C++ -*-===//
//
// Consume a BoundPlan and move bytes. gatherChunk is the primitive
// (#C5's per-chunk materialization); the strategies are the
// "one chunk = whole tensor" case. All buffers are raw element-0
// pointers; offsets scale by BoundPlan::elementSize. No exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_EXECUTE_H
#define RELOC_EXECUTE_H

#include "reloc/Bind.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reloc {

/// A published strided view (Strategy 1, no_copy). No data was moved.
struct ViewDescriptor {
  const void *base;
  std::vector<int64_t> extents;
  std::vector<int64_t> strides; // element strides into `base`
  uint32_t elementSize;
};

/// Strategy 1: no_copy -> a descriptor over the source, no movement.
/// Precondition (asserted): bound.noCopy and bound.padRegions empty.
ViewDescriptor executeView(const BoundPlan &bound, const void *srcBase);

/// Splat the single pad fill pattern across the entire physical dst
/// (bound.totalBytes). No-op when bound.padRegions is empty. Call before
/// gatherChunk so outer-axis pad rows are covered; #C5 calls this on its
/// staging buffer. Asserts all pad regions share one fillBits (v0).
void fillDst(const BoundPlan &bound, void *dstBase);

/// The primitive: write ONLY the valid dst cells for outer-axis indices
/// [outerBegin, outerEnd) (does not fill pads — call fillDst first).
/// `dstBase` is the address at which dst element offset 0 would land
/// (rebase it for a staging buffer). Scalar inner copy in this task;
/// Task 2 vectorises the contiguous run.
void gatherChunk(const BoundPlan &bound, const void *srcBase, void *dstBase,
                 int64_t outerBegin, int64_t outerEnd);

/// The D2H primitive: reconstruct ONLY the valid src cells for outer-axis
/// indices [outerBegin, outerEnd), reading from a dst-layout buffer. Mirror of
/// gatherChunk. `dstBaseV` is the address at which dst element offset 0 would
/// land (rebase it for a staging chunk); `srcBaseV` is the full valid buffer.
void scatterChunk(const BoundPlan &bound, const void *dstBaseV, void *srcBaseV,
                  int64_t outerBegin, int64_t outerEnd);

/// Strategy 2: single-thread H2D copy = gatherChunk over the whole outer
/// range. `dstBase` sized bound.totalBytes.
void executeH2D(const BoundPlan &bound, const void *srcBase, void *dstBase);

/// Strategy 3: multi-thread tiled H2D. Partitions the outer axis into
/// contiguous ranges, one gatherChunk per worker. `threads == 0` uses
/// std::thread::hardware_concurrency(). Falls back to a single
/// single-threaded gatherChunk call (no threads spawned) when distinct
/// outer rows are not provably disjoint in dst -- e.g. dstStrides[0]
/// smaller than the inner block span -- so the result is always
/// bit-identical to executeH2D and never racy.
void executeH2DThreaded(const BoundPlan &bound, const void *srcBase,
                        void *dstBase, unsigned threads = 0);

/// D2H: reconstruct the original source from a dst-layout buffer over the
/// valid (unpadded) index space. Realizes the inverse relocation by
/// role-swapping src/dst strides over the coalesced valid index space (a
/// scatter mirror of the H2D walk) rather than evaluating the wire
/// `inverse` PUSH_DIM streams -- see libreloc/README.md's Surface entry.
/// `dstBaseV` is a dst-layout buffer (as produced by executeH2D);
/// `srcBaseV` is sized for the valid region only (no pad cells).
void executeD2H(const BoundPlan &bound, const void *dstBaseV, void *srcBaseV);

} // namespace reloc

#endif // RELOC_EXECUTE_H
