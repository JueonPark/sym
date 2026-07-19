//===- QuantKernels.h - internal per-variant kernel decls -------*- C++ -*-===//
//
// Shared between the dispatch TU (Quant.cpp) and the per-ISA TUs
// (QuantAVX2.cpp, QuantAVX512.cpp). SIMD symbols are only DEFINED when the
// build adds those TUs (RELOC_QUANT_HAVE_X86_SIMD); dispatch code must
// guard references accordingly.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_QUANT_QUANTKERNELS_H
#define RELOC_QUANT_QUANTKERNELS_H

#include <cmath>
#include <cstdint>

namespace reloc {
namespace quant {
namespace detail {

/// The scalar quant contract. fmax first so NaN -> -128, matching the
/// vector MAXPS(v, lo) operand order; nearbyintf under the default FP env
/// is round-to-nearest-even, matching CVTPS2DQ.
inline int8_t quantOne(float x, float invScale) {
  float y = x * invScale;
  y = std::fmax(y, -128.0f);
  y = std::fmin(y, 127.0f);
  return static_cast<int8_t>(std::nearbyintf(y));
}

void quantizePackScalar(const float *src, int8_t *dst, int64_t n,
                        float invScale);

} // namespace detail
} // namespace quant
} // namespace reloc

#endif // RELOC_QUANT_QUANTKERNELS_H
