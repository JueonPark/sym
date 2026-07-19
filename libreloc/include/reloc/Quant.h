//===- Quant.h - R0.1 CPU transform kernels (issue #74) ---------*- C++ -*-===//
//
// The R-track's CPU-side transform kernels: contiguous per-channel int8
// quantize, fused strided-gather + quantize over a BoundPlan (the Case-1a
// kernel), int4 nibble pack, and fp32->fp16 convert. Every kernel has a
// scalar reference variant plus SIMD tiers selected at runtime, and every
// SIMD tier is bit-identical to scalar (tests enforce memcmp equality).
//
// Quantize semantics (all variants): q = (int8) rne(clamp(x * invScale,
// -128.0f, 127.0f)); clamp is max-then-min so NaN maps to -128 (the x86
// MAXPS(v, lo) rule); rne is round-to-nearest-even (default FP env).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_QUANT_H
#define RELOC_QUANT_H

#include "reloc/Bind.h"

#include <cstdint>

namespace reloc {

class GatherPool;

namespace quant {

/// SIMD tier. AVX2 also implies FMA + F16C (co-resident on every AVX2 CPU
/// since Haswell; checked at runtime). AVX512 means F + BW. AVX512Pf is the
/// software-prefetch gather variant of gatherQuantizeF32S8 only, never
/// chosen by Auto -- the bench decides whether it earns its keep.
enum class Variant : int { Auto = 0, Scalar, AVX2, AVX512, AVX512Pf };

/// The four issue-#74 kernels, for per-kernel variant queries.
enum class Kernel : int { QuantizePack, GatherQuantize, PackS8S4,
                          ConvertF32F16 };

/// True when this host can execute `v`. Auto and Scalar are always true.
bool cpuSupports(Variant v);

/// True when `k` implements tier `v` (issue #74's variant table). Auto and
/// Scalar are true for every kernel.
bool kernelHasVariant(Kernel k, Variant v);

/// Auto -> the best implemented+supported plain tier (never AVX512Pf);
/// explicit variants return themselves. Asserts kernelHasVariant(k, v) and
/// cpuSupports(v).
Variant resolveFor(Kernel k, Variant v);

/// K1, issue #74's `quantize_pack_f32_s8`: contiguous fp32 -> contiguous
/// int8 with a per-channel scale. `src`/`dst` hold channels * channelSize
/// elements; channel c covers [c * channelSize, (c+1) * channelSize) and
/// is quantized with invScales[c] (q = rne(clamp(x * invScale))).
void quantizePackF32S8(const float *src, int8_t *dst, int64_t channels,
                       int64_t channelSize, const float *invScales,
                       Variant v = Variant::Auto);

/// K4, issue #74's `convert_f32_f16`: contiguous fp32 -> IEEE binary16
/// (round-to-nearest-even, subnormals preserved, overflow -> inf). Scalar
/// output is bit-identical to F16C/AVX-512 VCVTPS2PH for non-NaN inputs.
void convertF32F16(const float *src, uint16_t *dst, int64_t count,
                   Variant v = Variant::Auto);

/// K3, issue #74's `pack_s8_s4`: int8 -> int4 nibble pack for the r=0.125
/// sweep. Each input is saturated to [-8, 7]; output byte i = low nibble
/// from src[2i], high nibble from src[2i+1]. Reads 2*pairs, writes pairs.
void packS8S4(const int8_t *src, uint8_t *dst, int64_t pairs,
              Variant v = Variant::Auto);

} // namespace quant
} // namespace reloc

#endif // RELOC_QUANT_H
