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

uint16_t f32ToF16Scalar(float f);
void convertF32F16Scalar(const float *src, uint16_t *dst, int64_t n);

inline uint8_t nibbleSat(int8_t v) {
  int x = v < -8 ? -8 : (v > 7 ? 7 : v);
  return static_cast<uint8_t>(x & 0xF);
}

void packS8S4Scalar(const int8_t *src, uint8_t *dst, int64_t pairs);

/// One innermost run of the fused kernel: n elements read at srcStride,
/// written contiguously. Implementations must be bit-identical.
using QuantRunFn = void (*)(const float *src, int64_t srcStride, int8_t *dst,
                            int64_t n, float invScale);

void quantRunScalar(const float *src, int64_t srcStride, int8_t *dst, int64_t n,
                    float invScale);

#if defined(RELOC_QUANT_HAVE_X86_SIMD)
// Defined in QuantAVX2.cpp / QuantAVX512.cpp (per-TU -m flags).
void quantizePackAVX2(const float *src, int8_t *dst, int64_t n, float invScale);
void quantizePackAVX512(const float *src, int8_t *dst, int64_t n,
                        float invScale);
void convertF32F16F16C(const float *src, uint16_t *dst, int64_t n);
void convertF32F16AVX512(const float *src, uint16_t *dst, int64_t n);
void packS8S4AVX512(const int8_t *src, uint8_t *dst, int64_t pairs);
void quantRunAVX512(const float *src, int64_t srcStride, int8_t *dst, int64_t n,
                    float invScale);
void quantRunAVX512Pf(const float *src, int64_t srcStride, int8_t *dst,
                      int64_t n, float invScale);
#endif

} // namespace detail
} // namespace quant
} // namespace reloc

#endif // RELOC_QUANT_QUANTKERNELS_H
