//===- Quant.cpp - R0.1 quant kernel dispatch + scalar variants -----------===//

#include "reloc/Quant.h"
#include "QuantKernels.h"

#include <cassert>

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

} // namespace quant
} // namespace reloc
