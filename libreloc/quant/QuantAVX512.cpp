//===- QuantAVX512.cpp - AVX-512 (F+BW) quant kernel variants -------------===//
// Compiled with -mavx512f -mavx512bw (see libreloc/CMakeLists.txt); callers
// must gate on cpuSupports(Variant::AVX512).

#include "QuantKernels.h"

#include <cassert>
#include <immintrin.h>

namespace reloc {
namespace quant {
namespace detail {

void quantizePackAVX512(const float *src, int8_t *dst, int64_t n,
                        float invScale) {
  const __m512 vinv = _mm512_set1_ps(invScale);
  const __m512 vlo = _mm512_set1_ps(-128.0f);
  const __m512 vhi = _mm512_set1_ps(127.0f);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    v = _mm512_mul_ps(v, vinv);
    v = _mm512_max_ps(v, vlo); // NaN -> vlo (MAXPS second-operand rule)
    v = _mm512_min_ps(v, vhi);
    __m512i q = _mm512_cvtps_epi32(v); // RNE
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i),
                     _mm512_cvtsepi32_epi8(q));
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

} // namespace detail
} // namespace quant
} // namespace reloc
