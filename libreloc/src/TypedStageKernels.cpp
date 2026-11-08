//===- TypedStageKernels.cpp - register execution of ordered value stages ===//
#include "TypedStageKernels.h"
#include "reloc/Quant.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RELOC_STAGE_AVX2 1
#endif

namespace reloc::typed::detail {
namespace {
bool f32(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 32;
}
bool f16(ElementType t) {
  return t.kind == ElementTypeKind::Float && t.bitwidth == 16;
}

void scalar(const StageRun &run, const uint8_t *src, uint8_t *dst,
            int64_t count, float scale, int32_t zp) {
  const bool dequant = run.kind == StageRun::Kind::DequantCastChain;
  const int inWidth = dequant ? 1 : run.inputHalf ? 2 : 4;
  const int outWidth = run.outputHalf ? 2 : 4;
  for (int64_t i = 0; i < count; ++i) {
    float value;
    if (dequant) {
      const int8_t q = static_cast<int8_t>(src[i]);
      value = static_cast<float>(static_cast<int32_t>(q) - zp) * scale;
    } else if (run.inputHalf) {
      uint16_t bits;
      std::memcpy(&bits, src + i * inWidth, 2);
      value = quant::widenF16F32(bits);
    } else {
      std::memcpy(&value, src + i * inWidth, 4);
    }
    for (uint32_t k = 0; k < run.roundTrips; ++k)
      value = quant::widenF16F32(quant::narrowF32F16(value));
    if (run.outputHalf) {
      const uint16_t bits = quant::narrowF32F16(value);
      std::memcpy(dst + i * outWidth, &bits, 2);
    } else {
      std::memcpy(dst + i * outWidth, &value, 4);
    }
  }
}

#ifdef RELOC_STAGE_AVX2
// Hardware widening quiets signaling NaNs; the existing CPU reference keeps
// their payload bits. Restore these bits for an initial half input. Half
// values produced by narrowing are already quiet and need no correction.
__attribute__((target("avx2,f16c"))) __m256 widenExact(__m128i half) {
  const __m256 value = _mm256_cvtph_ps(half);
  const __m256i bits = _mm256_cvtepu16_epi32(half);
  const __m256i nan =
      _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fff)),
                         _mm256_set1_epi32(0x7c00));
  const __m256i payload = _mm256_or_si256(
      _mm256_set1_epi32(0x7f800000),
      _mm256_or_si256(
          _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x8000)),
                            16),
          _mm256_slli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x3ff)),
                            13)));
  return _mm256_blendv_ps(value, _mm256_castsi256_ps(payload),
                          _mm256_castsi256_ps(nan));
}

template <bool Dequant, bool InputHalf, bool OutputHalf>
__attribute__((target("avx2,f16c"))) void
vectorRun(const StageRun &run, const uint8_t *src, uint8_t *dst, int64_t count,
          float scale, int32_t zp) {
  constexpr int inWidth = Dequant ? 1 : InputHalf ? 2 : 4;
  constexpr int outWidth = OutputHalf ? 2 : 4;
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 value;
    if constexpr (Dequant) {
      const __m128i q =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i));
      const __m256i difference =
          _mm256_sub_epi32(_mm256_cvtepi8_epi32(q), _mm256_set1_epi32(zp));
      // Explicit binary32 multiply followed by explicit half conversions.
      value =
          _mm256_mul_ps(_mm256_cvtepi32_ps(difference), _mm256_set1_ps(scale));
    } else if constexpr (InputHalf) {
      value = widenExact(_mm_loadu_si128(
          reinterpret_cast<const __m128i *>(src + i * inWidth)));
    } else {
      value =
          _mm256_loadu_ps(reinterpret_cast<const float *>(src + i * inWidth));
    }
    for (uint32_t k = 0; k < run.roundTrips; ++k)
      value = _mm256_cvtph_ps(_mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT |
                                                         _MM_FROUND_NO_EXC));
    if constexpr (OutputHalf) {
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * outWidth),
                       _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT |
                                                  _MM_FROUND_NO_EXC));
    } else {
      _mm256_storeu_ps(reinterpret_cast<float *>(dst + i * outWidth), value);
    }
  }
  scalar(run, src + i * inWidth, dst + i * outWidth, count - i, scale, zp);
}

void vectorDispatch(const StageRun &run, const uint8_t *src, uint8_t *dst,
                    int64_t count, float scale, int32_t zp) {
  if (run.kind == StageRun::Kind::DequantCastChain) {
    if (run.outputHalf)
      vectorRun<true, false, true>(run, src, dst, count, scale, zp);
    else
      vectorRun<true, false, false>(run, src, dst, count, scale, zp);
  } else if (run.inputHalf) {
    if (run.outputHalf)
      vectorRun<false, true, true>(run, src, dst, count, scale, zp);
    else
      vectorRun<false, true, false>(run, src, dst, count, scale, zp);
  } else if (run.outputHalf) {
    vectorRun<false, false, true>(run, src, dst, count, scale, zp);
  } else {
    vectorRun<false, false, false>(run, src, dst, count, scale, zp);
  }
}
#endif
} // namespace

StageRun prepareStageRun(const Program &p, uint32_t from, uint32_t to) {
  StageRun run;
  run.from = from;
  run.to = to;
  if (from >= to || to > p.stages.size())
    return run;
  uint32_t castFrom = from;
  const auto &first = p.stages[from];
  const bool dequant = first.transform == ValueTransformKind::Dequantize;
  if (dequant) {
    if (first.input.kind != ElementTypeKind::Integer ||
        first.input.bitwidth != 8 || !f32(first.output) ||
        first.policy != NumericPolicyKind::Affine || to == from + 1)
      return run;
    if (first.perChannel) {
      if (!first.channelIsDim || first.channelDim >= p.resultStrides.size() ||
          p.resultStrides[first.channelDim] <= 0 ||
          p.plan.resultExtents[first.channelDim] <= 0 ||
          p.plan.resultExtents[first.channelDim] > first.channelLength)
        return run;
      run.channelStride = p.resultStrides[first.channelDim];
      run.channelExtent = p.plan.resultExtents[first.channelDim];
    }
    ++castFrom;
  }
  for (uint32_t k = castFrom; k < to; ++k) {
    const auto &s = p.stages[k];
    if (s.transform != ValueTransformKind::Cast || s.perChannel ||
        !((f32(s.input) && f16(s.output) &&
           s.policy == NumericPolicyKind::IeeeRne) ||
          (f16(s.input) && f32(s.output) &&
           s.policy == NumericPolicyKind::Exact)))
      return StageRun{from, to};
    if (k > from && (p.stages[k - 1].output.kind != s.input.kind ||
                     p.stages[k - 1].output.bitwidth != s.input.bitwidth))
      return StageRun{from, to};
  }
  run.inputHalf = !dequant && f16(first.input);
  run.outputHalf = f16(p.stages[to - 1].output);
  uint32_t casts = to - castFrom;
  if (run.inputHalf)
    --casts; // The first exact widening loads the working binary32 register.
  run.roundTrips = casts / 2;
  run.kind =
      dequant ? StageRun::Kind::DequantCastChain : StageRun::Kind::CastChain;
  return run;
}

const char *stageRunName(const StageRun &run) {
  switch (run.kind) {
  case StageRun::Kind::Buffered:
    return "buffered";
  case StageRun::Kind::CastChain:
    return "register_cast_chain";
  case StageRun::Kind::DequantCastChain:
    return "register_dequant_cast_chain";
  }
  return "buffered";
}

bool runStageRun(const Program &p, const StageRun &run, const uint8_t *src,
                 uint8_t *dst, int64_t count, int64_t offset) {
  if (run.kind == StageRun::Kind::Buffered)
    return false;
  const bool dequant = run.kind == StageRun::Kind::DequantCastChain;
  const int inWidth = dequant ? 1 : run.inputHalf ? 2 : 4;
  const int outWidth = run.outputHalf ? 2 : 4;
  for (int64_t begin = 0; begin < count;) {
    int64_t n = count - begin;
    size_t channel = 0;
    if (run.channelStride) {
      n = std::min(n, run.channelStride - (offset + begin) % run.channelStride);
      channel = (offset + begin) / run.channelStride % run.channelExtent;
    }
    const auto &first = p.stages[run.from];
    const float scale = dequant ? first.scale[channel] : 1.f;
    const int32_t zp =
        dequant ? first.zeroPoint[first.zeroPoint.size() == 1 ? 0 : channel]
                : 0;
#ifdef RELOC_STAGE_AVX2
    if (quant::cpuSupports(quant::Variant::AVX2))
      vectorDispatch(run, src + begin * inWidth, dst + begin * outWidth, n,
                     scale, zp);
    else
#endif
      scalar(run, src + begin * inWidth, dst + begin * outWidth, n, scale, zp);
    begin += n;
  }
  return true;
}
} // namespace reloc::typed::detail
