//===- QuantTest.cpp - R0.1 CPU quant kernels (issue #74) -----------------===//

#include "reloc/Quant.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

using reloc::quant::Kernel;
using reloc::quant::Variant;

TEST(QuantDispatch, ScalarAndAutoAlwaysSupported) {
  EXPECT_TRUE(reloc::quant::cpuSupports(Variant::Scalar));
  EXPECT_TRUE(reloc::quant::cpuSupports(Variant::Auto));
}

TEST(QuantDispatch, KernelVariantTableMatchesIssue74) {
  // quantize_pack: scalar / AVX2 / AVX-512
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX2));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX512));
  EXPECT_FALSE(reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX512Pf));
  // gather_quantize: scalar / AVX-512 gather / prefetch+tiled
  EXPECT_FALSE(reloc::quant::kernelHasVariant(Kernel::GatherQuantize, Variant::AVX2));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::GatherQuantize, Variant::AVX512));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::GatherQuantize, Variant::AVX512Pf));
  // pack_s8_s4: AVX-512 (+ scalar reference)
  EXPECT_FALSE(reloc::quant::kernelHasVariant(Kernel::PackS8S4, Variant::AVX2));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::PackS8S4, Variant::AVX512));
  // convert_f32_f16: F16C (the AVX2 tier) / AVX-512
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX2));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX512));
}

TEST(QuantDispatch, ResolveAutoPicksAnImplementedSupportedTier) {
  for (Kernel k : {Kernel::QuantizePack, Kernel::GatherQuantize,
                   Kernel::PackS8S4, Kernel::ConvertF32F16}) {
    Variant r = reloc::quant::resolveFor(k, Variant::Auto);
    EXPECT_NE(r, Variant::Auto);
    EXPECT_NE(r, Variant::AVX512Pf) << "prefetch tier is opt-in only";
    EXPECT_TRUE(reloc::quant::cpuSupports(r));
    EXPECT_TRUE(reloc::quant::kernelHasVariant(k, r));
  }
}

TEST(QuantDispatch, ResolveExplicitIsIdentity) {
  EXPECT_EQ(reloc::quant::resolveFor(Kernel::QuantizePack, Variant::Scalar),
            Variant::Scalar);
  if (reloc::quant::cpuSupports(Variant::AVX512)) {
    EXPECT_EQ(
        reloc::quant::resolveFor(Kernel::GatherQuantize, Variant::AVX512Pf),
        Variant::AVX512Pf);
  }
}

// Independent reformulation of the quant contract (Global Constraints).
int8_t refQuantOne(float x, float invScale) {
  float y = x * invScale;
  if (std::isnan(y))
    return -128;
  if (y < -128.0f)
    y = -128.0f;
  if (y > 127.0f)
    y = 127.0f;
  return static_cast<int8_t>(std::lrintf(y)); // FE_TONEAREST default = RNE
}

std::vector<float> randomFloats(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float &x : v)
    x = d(rng);
  return v;
}

TEST(QuantizePack, ScalarMatchesReferencePerChannel) {
  const int64_t channels = 5, chSize = 67; // 67 = 4*16 + 3: remainder tail
  std::vector<float> src = randomFloats(channels * chSize, 1, -300.f, 300.f);
  std::vector<float> inv(channels);
  for (int64_t c = 0; c < channels; ++c)
    inv[c] = 0.05f + 0.9f * static_cast<float>(c);
  std::vector<int8_t> dst(channels * chSize, 42);
  reloc::quant::quantizePackF32S8(src.data(), dst.data(), channels, chSize,
                                  inv.data(), Variant::Scalar);
  for (int64_t c = 0; c < channels; ++c)
    for (int64_t i = 0; i < chSize; ++i)
      ASSERT_EQ(dst[c * chSize + i], refQuantOne(src[c * chSize + i], inv[c]))
          << "c=" << c << " i=" << i;
}

TEST(QuantizePack, ScalarSpecialValues) {
  // invScale = 1: RNE ties go to even; saturation clamps; NaN -> -128.
  const float src[] = {0.0f,   -0.0f,   0.5f, -0.5f, 1.5f,      2.5f, -2.5f,
                       200.0f, -200.0f, HUGE_VALF, -HUGE_VALF, NAN,  126.6f};
  const int8_t want[] = {0, 0, 0, 0, 2, 2, -2, 127, -128, 127, -128, -128, 127};
  const int64_t n = sizeof(src) / sizeof(src[0]);
  std::vector<int8_t> dst(n, 42);
  const float inv = 1.0f;
  reloc::quant::quantizePackF32S8(src, dst.data(), 1, n, &inv,
                                  Variant::Scalar);
  for (int64_t i = 0; i < n; ++i)
    EXPECT_EQ(dst[i], want[i]) << "i=" << i;
}

} // namespace
