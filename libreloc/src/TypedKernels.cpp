//===- TypedKernels.cpp - tiled layout and buffered value stages
//------------===//
#include "TypedKernels.h"
#include "Transpose.h"
#include "reloc/Quant.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RELOC_TYPED_AVX2 1
#endif

namespace reloc::typed::detail {
namespace {

// These two stages complement the existing qualified quantize/narrow SIMD
// kernels. Keep arithmetic operations separate: no dequant/narrow fusion that
// loses the intermediate binary32 rounding, and no FMA contraction.
#ifdef RELOC_TYPED_AVX2
__attribute__((target("avx2,f16c"))) void widen(const uint16_t *src, float *dst,
                                                int64_t count) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    __m256 value = _mm256_cvtph_ps(h);
    // Preserve the scalar reference's signaling/quiet NaN bits as well.
    __m256i bits = _mm256_cvtepu16_epi32(h);
    __m256i nan =
        _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fff)),
                           _mm256_set1_epi32(0x7c00));
    __m256i payload = _mm256_or_si256(
        _mm256_set1_epi32(0x7f800000),
        _mm256_or_si256(
            _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)),
                              16),
            _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x3ff)),
                              13)));
    _mm256_storeu_ps(dst + i,
                     _mm256_blendv_ps(value, _mm256_castsi256_ps(payload),
                                      _mm256_castsi256_ps(nan)));
  }
  for (; i < count; ++i)
    dst[i] = quant::widenF16F32(src[i]);
}

__attribute__((target("avx2"))) void dequantize(const int8_t *src, float *dst,
                                                int64_t count, float scale,
                                                int32_t zp) {
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i));
    __m256i d =
        _mm256_sub_epi32(_mm256_cvtepi8_epi32(q), _mm256_set1_epi32(zp));
    _mm256_storeu_ps(
        dst + i, _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(scale)));
  }
  for (; i < count; ++i)
    dst[i] = static_cast<float>(static_cast<int32_t>(src[i]) - zp) * scale;
}
#endif

// Actual typed arrays, not type-punned byte scratch: all loads/stores obey
// alignment and object-lifetime rules, even for byte-offset external views.
struct Buffer {
  alignas(64) float f32[kTypedRun];
  alignas(64) uint16_t f16[kTypedRun];
  alignas(64) int8_t s8[kTypedRun];
  void *data(ElementType type) {
    return type.kind == ElementTypeKind::Integer ? static_cast<void *>(s8)
           : type.bitwidth == 16                 ? static_cast<void *>(f16)
                                                 : static_cast<void *>(f32);
  }
};

void stage(const StageArithmetic &s, Buffer &input, Buffer &output,
           int64_t count, size_t channel) {
  switch (s.transform) {
  case ValueTransformKind::Cast:
    if (s.input.bitwidth == 32) {
      quant::convertF32F16(input.f32, output.f16, count);
      return;
    }
#ifdef RELOC_TYPED_AVX2
    if (quant::cpuSupports(quant::Variant::AVX2)) {
      widen(input.f16, output.f32, count);
      return;
    }
#endif
    for (int64_t i = 0; i < count; ++i)
      output.f32[i] = quant::widenF16F32(input.f16[i]);
    return;
  case ValueTransformKind::Quantize:
    quant::quantizePackF32S8(input.f32, output.s8, 1, count,
                             &s.invScale[channel]);
    return;
  case ValueTransformKind::Dequantize: {
    const float scale = s.scale[channel];
    const int32_t zp = s.zeroPoint[s.zeroPoint.size() == 1 ? 0 : channel];
#ifdef RELOC_TYPED_AVX2
    if (quant::cpuSupports(quant::Variant::AVX2)) {
      dequantize(input.s8, output.f32, count, scale, zp);
      return;
    }
#endif
    for (int64_t i = 0; i < count; ++i)
      output.f32[i] =
          static_cast<float>(static_cast<int32_t>(input.s8[i]) - zp) * scale;
    return;
  }
  }
}

template <size_t Width>
void transposeTile(const uint8_t *src, uint8_t *dst, int64_t rows,
                   int64_t columns, int64_t stride) {
  for (int64_t c = 0; c < columns; ++c)
    for (int64_t r = 0; r < rows; ++r)
      std::memcpy(dst + (r * columns + c) * Width,
                  src + (c * stride + r) * Width, Width);
}
} // namespace

bool simpleChannels(const Program &p, uint32_t from, uint32_t to,
                    int64_t innerSpan) {
  for (uint32_t k = from; k < to; ++k) {
    const auto &s = p.stages[k];
    if (s.perChannel &&
        (!s.channelIsDim || p.resultStrides[s.channelDim] < innerSpan ||
         p.plan.resultExtents[s.channelDim] > s.channelLength))
      return false;
  }
  return true;
}

void stages(const Program &p, uint32_t from, uint32_t to, const uint8_t *src,
            uint8_t *dst, int64_t count, int64_t offset) {
  const uint32_t inWidth = widthAt(p, from), outWidth = widthAt(p, to);
  if (from == to) {
    std::memcpy(dst, src, count * inWidth);
    return;
  }
  Buffer a, b;
  for (int64_t begin = 0; begin < count;) {
    int64_t n = std::min(kTypedRun, count - begin);
    // Padding or a coalesced layout can put a row inside a channel block.
    // Split exactly at its next boundary instead of guessing a channel axis.
    for (uint32_t k = from; k < to; ++k)
      if (p.stages[k].perChannel) {
        const int64_t stride = p.resultStrides[p.stages[k].channelDim];
        n = std::min(n, stride - (offset + begin) % stride);
      }
    Buffer *input = &a, *output = &b;
    std::memcpy(input->data(typeAt(p, from)), src + begin * inWidth,
                n * inWidth);
    for (uint32_t k = from; k < to; ++k) {
      const auto &s = p.stages[k];
      const size_t channel =
          s.perChannel ? (offset + begin) / p.resultStrides[s.channelDim] %
                             p.plan.resultExtents[s.channelDim]
                       : 0;
      stage(s, *input, *output, n, channel);
      std::swap(input, output);
    }
    std::memcpy(dst + begin * outWidth, input->data(typeAt(p, to)),
                n * outWidth);
    begin += n;
  }
}

void transpose(const Program &p, uint32_t from, uint32_t to, const uint8_t *src,
               uint8_t *dst, int64_t first, int64_t last, int64_t origin,
               int64_t outerLo, int64_t innerLo) {
  const auto &layout = p.plan.layout;
  const int64_t columns = layout.extents[1];
  const uint32_t inWidth = widthAt(p, from), outWidth = widthAt(p, to);
  alignas(64) uint8_t tile[kTypedTile * kTypedTile * 4];
  for (int64_t r = first; r < last; r += kTypedTile) {
    const int64_t rows = std::min(kTypedTile, last - r);
    for (int64_t c = 0; c < columns; c += kTypedTile) {
      const int64_t cols = std::min(kTypedTile, columns - c);
      const uint8_t *source = src + (c * layout.srcStrides[1] + r) * inWidth;
      if (inWidth == 4)
        reloc::detail::transpose32(source, tile, rows, cols,
                                   layout.srcStrides[1]);
      else if (inWidth == 2)
        transposeTile<2>(source, tile, rows, cols, layout.srcStrides[1]);
      else
        transposeTile<1>(source, tile, rows, cols, layout.srcStrides[1]);
      for (int64_t row = 0; row < rows; ++row) {
        const int64_t offset =
            (r + row + outerLo) * layout.dstStrides[0] + c + innerLo;
        stages(p, from, to, tile + row * cols * inWidth,
               dst + (offset - origin) * outWidth, cols, offset);
      }
    }
  }
}
void stridedRuns(const Program &p, uint32_t from, uint32_t to,
                 const uint8_t *src, uint8_t *dst, int64_t first, int64_t last,
                 int64_t origin, bool narrowCast) {
  const auto &l = p.plan.layout;
  const int64_t middle = l.extents[1], run = l.extents[2];
  const int64_t ss0 = l.srcStrides[0], ss1 = l.srcStrides[1];
  const int64_t ds0 = l.dstStrides[0], ds1 = l.dstStrides[1];
  const uint32_t inWidth = widthAt(p, from), outWidth = widthAt(p, to);
  auto one = [&](int64_t i, int64_t j) {
    const uint8_t *s = src + (i * ss0 + j * ss1) * inWidth;
    const int64_t offset = i * ds0 + j * ds1;
    uint8_t *d = dst + (offset - origin) * outWidth;
    if (narrowCast && reinterpret_cast<uintptr_t>(s) % alignof(float) == 0 &&
        reinterpret_cast<uintptr_t>(d) % alignof(uint16_t) == 0) {
      quant::convertF32F16(reinterpret_cast<const float *>(s),
                           reinterpret_cast<uint16_t *>(d), run);
      return;
    }
    stages(p, from, to, s, d, run, offset);
  };
  if (ss1 <= ss0) {
    // Source rows are contiguous along the middle axis (or identity).
    for (int64_t i = first; i < last; ++i)
      for (int64_t j = 0; j < middle; ++j)
        one(i, j);
  } else {
    // Source is contiguous along the outer axis: stream it, scatter writes.
    for (int64_t j = 0; j < middle; ++j)
      for (int64_t i = first; i < last; ++i)
        one(i, j);
  }
}
} // namespace reloc::typed::detail
