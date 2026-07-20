//===- plans.h - hand-authored R-track workload plans -----------*- C++ -*-===//
//
// R0.3 (issue #76). Every plan is authored directly as a BoundPlan (the
// CudaKernelsTest convention) instead of decoding bench/reference_plan.h:
// the frozen golden blob on main does not encode the intended blocked
// transpose for N > 4096 (issue #63), and RtrackTest verifies each builder
// here against independent index math -- which a decode-then-execute
// self-check cannot do. Axes are authored in coalesced dst-descending
// order, packed dst, elementSize 4 (fp32 source). L is set to the true
// innermost contiguous run of the authored axes (only bind() writes it;
// executors derive runs from the strides, so this is documentation).
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_PLANS_H
#define BENCH_RTRACK_PLANS_H

#include "reloc/Bind.h"

#include <cassert>
#include <cstdint>

namespace bench {
namespace rtrack {

/// [N, N] contiguous copy (T3/T5's host-read pattern; no relocation).
inline reloc::BoundPlan identityPlan(int64_t n) {
  reloc::BoundPlan b;
  b.extents = {n, n};
  b.srcStrides = {n, 1};
  b.dstStrides = {n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = n;
  return b;
}

/// Plain 2-D transpose: dst row i = src column i (T1/T2; the tiled-SMEM
/// shape for relocateF32).
inline reloc::BoundPlan transposePlan(int64_t n) {
  reloc::BoundPlan b;
  b.extents = {n, n};
  b.srcStrides = {1, n};
  b.dstStrides = {n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = 1;
  return b;
}

/// The sym#63 anchor: x.view(N/64, 64, 64, N/64).transpose(0, 1), with the
/// CORRECTED axes from the issue-#63 fix, pre-coalesced to rank 3 exactly
/// as bind() would merge them (b1+n1 -> one src- and dst-contiguous run of
/// N elements).
inline reloc::BoundPlan blockedTransposePlan(int64_t n) {
  assert(n % 64 == 0);
  const int64_t m = n / 64;
  reloc::BoundPlan b;
  b.extents = {64, m, n};
  b.srcStrides = {n, 64 * n, 1};
  b.dstStrides = {n * m, n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = n;
  return b;
}

/// T4: NCHW -> NHWC with (B, C, H, W) = (N/64, 64, 64, N/64) so the total
/// stays N^2. Coalesced dst order (b, h, w, c); src strides index NCHW.
inline reloc::BoundPlan nchwToNhwcPlan(int64_t n) {
  assert(n % 64 == 0);
  const int64_t C = 64, H = 64, W = n / 64;
  reloc::BoundPlan b;
  b.extents = {n / 64, H, W, C};
  b.srcStrides = {C * H * W, W, 1, H * W};
  b.dstStrides = {H * W * C, W * C, C, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = 1;
  return b;
}

/// Largest source element offset reachable through the plan (buffer
/// sizing; equals N^2 - 1 for every bijective builder above).
inline int64_t maxSrcOffset(const reloc::BoundPlan &b) {
  int64_t off = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    off += (b.extents[k] - 1) * b.srcStrides[k];
  return off;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_PLANS_H
