//===- QuantAVX2.cpp - AVX2/FMA/F16C quant kernel variants ----------------===//
// Compiled with -mavx2 -mfma -mf16c (see libreloc/CMakeLists.txt); callers
// must gate on cpuSupports(Variant::AVX2).

#include "QuantKernels.h"

#include <immintrin.h>

namespace reloc {
namespace quant {
namespace detail {

void quantizePackAVX2(const float *src, int8_t *dst, int64_t n,
                      float invScale) {
  const __m256 vinv = _mm256_set1_ps(invScale);
  const __m256 vlo = _mm256_set1_ps(-128.0f);
  const __m256 vhi = _mm256_set1_ps(127.0f);
  // packs_epi32 + packs_epi16 interleave 128-bit lanes; this permutation
  // restores source order (dword k of the result = elements 4k..4k+3).
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i q[4];
    for (int k = 0; k < 4; ++k) {
      __m256 v = _mm256_loadu_ps(src + i + 8 * k);
      v = _mm256_mul_ps(v, vinv);
      v = _mm256_max_ps(v, vlo); // NaN -> vlo (MAXPS second-operand rule)
      v = _mm256_min_ps(v, vhi);
      q[k] = _mm256_cvtps_epi32(v); // RNE
    }
    __m256i p01 = _mm256_packs_epi32(q[0], q[1]);
    __m256i p23 = _mm256_packs_epi32(q[2], q[3]);
    __m256i p = _mm256_packs_epi16(p01, p23);
    p = _mm256_permutevar8x32_epi32(p, order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), p);
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

} // namespace detail
} // namespace quant
} // namespace reloc
