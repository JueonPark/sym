//===- QuantAVX512.cpp - AVX-512 (F+BW) quant kernel variants -------------===//
// Compiled with -mavx512f -mavx512bw (see libreloc/CMakeLists.txt); callers
// must gate on cpuSupports(Variant::AVX512).

#include "QuantKernels.h"

#include <climits>
#include <cstdint>
#include <immintrin.h>

namespace {

// Shared body: quantize 16 gathered floats and store. `Prefetch` pulls the
// next-but-one iteration's lines (each strided element usually sits on its
// own cache line, so one prefetch per lane).
template <bool Prefetch>
void quantRunAVX512Impl(const float *src, int64_t srcStride, int8_t *dst,
                        int64_t n, float invScale) {
  using reloc::quant::detail::quantOne;
  if (srcStride == 1) {
    reloc::quant::detail::quantizePackAVX512(src, dst, n, invScale);
    return;
  }
  if (srcStride <= 0 || srcStride > (INT32_MAX / 16)) {
    // i32gather indices are signed 32-bit scaled by 4 bytes; out-of-range
    // strides take the scalar path so the variant contract stays
    // unconditional in release builds too.
    reloc::quant::detail::quantRunScalar(src, srcStride, dst, n, invScale);
    return;
  }
  const __m512 vinv = _mm512_set1_ps(invScale);
  const __m512 vlo = _mm512_set1_ps(-128.0f);
  const __m512 vhi = _mm512_set1_ps(127.0f);
  const __m512i lane =
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i vidx =
      _mm512_mullo_epi32(lane, _mm512_set1_epi32(static_cast<int>(srcStride)));
  constexpr int64_t kPfDist = 32; // elements ahead = 2 vector iterations
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    if (Prefetch && i + kPfDist + 16 <= n) {
      const float *pf = src + (i + kPfDist) * srcStride;
      for (int k = 0; k < 16; ++k)
        _mm_prefetch(reinterpret_cast<const char *>(pf + k * srcStride),
                     _MM_HINT_T0);
    }
    __m512 v = _mm512_i32gather_ps(vidx, src + i * srcStride, 4);
    v = _mm512_mul_ps(v, vinv);
    v = _mm512_max_ps(v, vlo);
    v = _mm512_min_ps(v, vhi);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i),
                     _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v)));
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i * srcStride], invScale);
}

} // namespace

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

void convertF32F16AVX512(const float *src, uint16_t *dst, int64_t n) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + i),
        _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  for (; i < n; ++i)
    dst[i] = f32ToF16Scalar(src[i]);
}

void packS8S4AVX512(const int8_t *src, uint8_t *dst, int64_t pairs) {
  const __m512i lo8 = _mm512_set1_epi8(-8);
  const __m512i hi7 = _mm512_set1_epi8(7);
  const __m512i mask = _mm512_set1_epi16(0x0F0F);
  int64_t i = 0;
  for (; i + 32 <= pairs; i += 32) { // 64 int8 in -> 32 packed bytes out
    __m512i v = _mm512_loadu_si512(src + 2 * i);
    v = _mm512_max_epi8(v, lo8);
    v = _mm512_min_epi8(v, hi7);
    v = _mm512_and_si512(v, mask);
    // Per epi16 lane: bits 0-3 = even nibble, bits 8-11 = odd nibble.
    // v | (v >> 4) puts the odd nibble at bits 4-7; VPMOVWB truncates each
    // lane to its low byte = the packed result.
    const __m512i t = _mm512_or_si512(v, _mm512_srli_epi16(v, 4));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i),
                        _mm512_cvtepi16_epi8(t));
  }
  for (; i < pairs; ++i)
    dst[i] = static_cast<uint8_t>(nibbleSat(src[2 * i]) |
                                  (nibbleSat(src[2 * i + 1]) << 4));
}

void quantRunAVX512(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale) {
  quantRunAVX512Impl<false>(src, srcStride, dst, n, invScale);
}

void quantRunAVX512Pf(const float *src, int64_t srcStride, int8_t *dst,
                      int64_t n, float invScale) {
  quantRunAVX512Impl<true>(src, srcStride, dst, n, invScale);
}

} // namespace detail
} // namespace quant
} // namespace reloc
