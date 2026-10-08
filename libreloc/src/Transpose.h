//===- Transpose.h - internal CPU transpose kernels -------------*- C++ -*-===//

#ifndef RELOC_SRC_TRANSPOSE_H
#define RELOC_SRC_TRANSPOSE_H

#include <cstdint>

namespace reloc {
struct BoundPlan;
struct StackedSource;

namespace detail {

// Copy opaque four-byte elements: dst[r * columns + c] =
// src[c * srcStride + r]. Both pointers address the first element of this
// window; source rows may extend beyond it. Buffers must not overlap.
// Kept separately callable so the portable implementation is tested on x86.
void transpose32Scalar(const uint8_t *src, uint8_t *dst, int64_t rows,
                       int64_t columns, int64_t srcStride);

// Recognize an unpadded dense rank-2 transpose and materialize only the
// requested destination rows. Uses gatherChunk's rebased-destination contract.
// Returns false without touching buffers when the plan is not supported.
bool tryGatherTranspose32(const BoundPlan &bound, const uint8_t *src,
                          uint8_t *dst, int64_t outerBegin, int64_t outerEnd);

// The same kernel for a last-dim torch.stack: the plan is the 32-bit unpadded
// [N, Z] -> [Z, N] transpose of the logical source, so source row c is input
// c (StackedSource::bases[c]). Returns false without touching buffers
// otherwise.
bool tryGatherTranspose32Stacked(const BoundPlan &bound,
                                 const StackedSource &source, uint8_t *dst,
                                 int64_t outerBegin, int64_t outerEnd);

} // namespace detail
} // namespace reloc

#endif // RELOC_SRC_TRANSPOSE_H
