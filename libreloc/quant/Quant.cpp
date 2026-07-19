//===- Quant.cpp - R0.1 quant kernel dispatch + scalar variants -----------===//

#include "reloc/Quant.h"
#include "QuantKernels.h"

#include <cassert>
#include <cstring>

namespace {

// Mirror of Execute.cpp's walk() for the fused kernel: only the top axis is
// ranged to the chunk; deeper axes cover their full extent. The per-channel
// invScale is picked at depth 0 and constant below it.
void quantWalk(const reloc::BoundPlan &b, const float *src, int8_t *dst,
               const float *invScales, size_t depth, int64_t iBegin,
               int64_t iEnd, int64_t srcOff, int64_t dstOff, float invScale,
               reloc::quant::detail::QuantRunFn run) {
  const size_t r = b.extents.size();
  if (depth == r - 1) {
    run(src + srcOff + iBegin * b.srcStrides[depth], b.srcStrides[depth],
        dst + dstOff + iBegin, iEnd - iBegin, invScale);
    return;
  }
  for (int64_t i = iBegin; i < iEnd; ++i)
    quantWalk(b, src, dst, invScales, depth + 1, 0, b.extents[depth + 1],
              srcOff + i * b.srcStrides[depth],
              dstOff + i * b.dstStrides[depth],
              depth == 0 ? invScales[i] : invScale, run);
}

} // namespace

namespace reloc {
namespace quant {

bool cpuSupports(Variant v) {
  switch (v) {
  case Variant::Auto:
  case Variant::Scalar:
    return true;
  case Variant::AVX2:
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
  case Variant::AVX512:
  case Variant::AVX512Pf:
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
  }
  return false;
}

bool kernelHasVariant(Kernel k, Variant v) {
  if (v == Variant::Auto || v == Variant::Scalar)
    return true;
  switch (k) {
  case Kernel::QuantizePack:
    return v == Variant::AVX2 || v == Variant::AVX512;
  case Kernel::GatherQuantize:
    return v == Variant::AVX512 || v == Variant::AVX512Pf;
  case Kernel::PackS8S4:
    return v == Variant::AVX512;
  case Kernel::ConvertF32F16:
    return v == Variant::AVX2 || v == Variant::AVX512;
  }
  return false;
}

Variant resolveFor(Kernel k, Variant v) {
  assert(kernelHasVariant(k, v) && "variant not implemented for this kernel");
  assert(cpuSupports(v) && "variant unsupported on this host");
  if (v != Variant::Auto)
    return v;
  if (kernelHasVariant(k, Variant::AVX512) && cpuSupports(Variant::AVX512))
    return Variant::AVX512;
  if (kernelHasVariant(k, Variant::AVX2) && cpuSupports(Variant::AVX2))
    return Variant::AVX2;
  return Variant::Scalar;
}

namespace detail {

void quantizePackScalar(const float *src, int8_t *dst, int64_t n,
                        float invScale) {
  for (int64_t i = 0; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

uint16_t f32ToF16Scalar(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  x &= 0x7fffffffu;
  if (x > 0x7f800000u) // NaN: quiet bit + truncated payload (VCVTPS2PH)
    return static_cast<uint16_t>(sign | 0x7e00u | ((x >> 13) & 0x3ffu));
  if (x >= 0x477ff000u) // inf, or >= 65520 which RNE-rounds to inf
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (x >= 0x38800000u) {
    // Normal half. Add 0xfff + (bit 13) to RNE the 13 dropped mantissa
    // bits; the carry ripples into the exponent field correctly.
    const uint32_t r = x + 0xfffu + ((x >> 13) & 1u);
    return static_cast<uint16_t>(sign | ((r - 0x38000000u) >> 13));
  }
  if (x <= 0x33000000u) // <= 2^-25: ties-to-even lands on zero
    return sign;
  // Subnormal half: value = mant24 * 2^(exp - 150), result ulp = 2^-24.
  const uint32_t shift = 126u - (x >> 23); // in [14, 24] here
  const uint32_t mant = (x & 0x7fffffu) | 0x800000u;
  uint32_t r = mant >> shift;
  const uint32_t rem = mant & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (r & 1u)))
    ++r; // may carry into 0x400: that IS the smallest normal -- correct
  return static_cast<uint16_t>(sign | r);
}

void convertF32F16Scalar(const float *src, uint16_t *dst, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    dst[i] = f32ToF16Scalar(src[i]);
}

void packS8S4Scalar(const int8_t *src, uint8_t *dst, int64_t pairs) {
  for (int64_t i = 0; i < pairs; ++i)
    dst[i] = static_cast<uint8_t>(nibbleSat(src[2 * i]) |
                                  (nibbleSat(src[2 * i + 1]) << 4));
}

void quantRunScalar(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale) {
  if (srcStride == 1) {
    quantizePackScalar(src, dst, n, invScale);
    return;
  }
  for (int64_t i = 0; i < n; ++i)
    dst[i] = quantOne(src[i * srcStride], invScale);
}

} // namespace detail

void quantizePackF32S8(const float *src, int8_t *dst, int64_t channels,
                       int64_t channelSize, const float *invScales,
                       Variant v) {
  const Variant r = resolveFor(Kernel::QuantizePack, v);
  for (int64_t c = 0; c < channels; ++c) {
    const float *s = src + c * channelSize;
    int8_t *d = dst + c * channelSize;
    switch (r) {
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    case Variant::AVX512:
      detail::quantizePackAVX512(s, d, channelSize, invScales[c]);
      break;
    case Variant::AVX2:
      detail::quantizePackAVX2(s, d, channelSize, invScales[c]);
      break;
#endif
    default:
      detail::quantizePackScalar(s, d, channelSize, invScales[c]);
      break;
    }
  }
}

void convertF32F16(const float *src, uint16_t *dst, int64_t count,
                   Variant v) {
  const Variant r = resolveFor(Kernel::ConvertF32F16, v);
  switch (r) {
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  case Variant::AVX512:
    detail::convertF32F16AVX512(src, dst, count);
    break;
  case Variant::AVX2:
    detail::convertF32F16F16C(src, dst, count);
    break;
#endif
  default:
    detail::convertF32F16Scalar(src, dst, count);
    break;
  }
}

void packS8S4(const int8_t *src, uint8_t *dst, int64_t pairs, Variant v) {
  const Variant r = resolveFor(Kernel::PackS8S4, v);
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  if (r == Variant::AVX512) {
    detail::packS8S4AVX512(src, dst, pairs);
    return;
  }
#endif
  (void)r;
  detail::packS8S4Scalar(src, dst, pairs);
}

void gatherQuantizeF32S8(const BoundPlan &bound, const float *srcBase,
                         int8_t *dstBase, const float *invScales,
                         int64_t outerBegin, int64_t outerEnd, Variant v) {
  assert(bound.elementSize == 4 && "fused quantize is fp32-only");
  assert(bound.padRegions.empty() && "pads unsupported in gatherQuantize v0");
  assert(bound.extents.size() >= 2 &&
         "per-channel scale needs a distinct outer axis");
  assert(bound.dstStrides.back() == 1 &&
         "dst innermost axis must be contiguous");
  const Variant r = resolveFor(Kernel::GatherQuantize, v);
  detail::QuantRunFn run = detail::quantRunScalar;
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  if (r == Variant::AVX512)
    run = detail::quantRunAVX512;
  else if (r == Variant::AVX512Pf)
    run = detail::quantRunAVX512Pf;
#endif
  quantWalk(bound, srcBase, dstBase, invScales, /*depth=*/0, outerBegin,
            outerEnd, /*srcOff=*/0, /*dstOff=*/0, /*invScale=*/0.0f, run);
}

} // namespace quant
} // namespace reloc
