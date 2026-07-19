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

TEST(QuantizePack, SimdVariantsBitExactVsScalar) {
  bool ranAny = false;
  for (Variant v : {Variant::AVX2, Variant::AVX512}) {
    if (!reloc::quant::cpuSupports(v))
      continue;
    ranAny = true;
    for (int64_t n : {1, 15, 16, 17, 31, 32, 33, 64, 1000, 4099}) {
      std::vector<float> src =
          randomFloats(n, static_cast<uint32_t>(7 + n), -300.f, 300.f);
      // Poke the clamp/NaN lanes inside a full vector when room allows.
      if (n >= 33) {
        src[3] = NAN;
        src[17] = HUGE_VALF;
        src[32] = -HUGE_VALF;
      }
      std::vector<int8_t> a(n, 0), b(n, 0);
      const float inv = 0.37f;
      reloc::quant::quantizePackF32S8(src.data(), a.data(), 1, n, &inv,
                                      Variant::Scalar);
      reloc::quant::quantizePackF32S8(src.data(), b.data(), 1, n, &inv, v);
      ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n))
          << "variant=" << static_cast<int>(v) << " n=" << n;
    }
  }
  if (!ranAny)
    GTEST_SKIP() << "no SIMD tier supported on this host";
}

TEST(ConvertF32F16, ScalarSpecials) {
  struct Case {
    float in;
    uint16_t want;
  } cases[] = {
      {0.0f, 0x0000},          {-0.0f, 0x8000},
      {1.0f, 0x3C00},          {-2.0f, 0xC000},
      {65504.0f, 0x7BFF},      {65519.0f, 0x7BFF}, // below RNE-to-inf cut
      {65520.0f, 0x7C00},      {HUGE_VALF, 0x7C00},
      {-HUGE_VALF, 0xFC00},
      {5.9604645e-8f, 0x0001}, // 2^-24, smallest half subnormal
      {2.9802322e-8f, 0x0000}, // 2^-25: tie, rounds to even (zero)
      {4.4703484e-8f, 0x0001}, // 1.5 * 2^-25: rounds up
      {6.1035156e-5f, 0x0400}, // 2^-14, smallest half normal
      {0.1f, 0x2E66},          {3.14159265f, 0x4248},
  };
  for (const Case &c : cases) {
    uint16_t out = 0;
    reloc::quant::convertF32F16(&c.in, &out, 1, Variant::Scalar);
    EXPECT_EQ(out, c.want) << "in=" << c.in;
  }
  float nan = NAN;
  uint16_t h = 0;
  reloc::quant::convertF32F16(&nan, &h, 1, Variant::Scalar);
  EXPECT_EQ(h & 0x7C00u, 0x7C00u);
  EXPECT_NE(h & 0x3FFu, 0u); // still a NaN, not inf
}

TEST(ConvertF32F16, SimdVariantsBitExactVsScalar) {
  bool ranAny = false;
  for (Variant v : {Variant::AVX2, Variant::AVX512}) {
    if (!reloc::quant::cpuSupports(v))
      continue;
    ranAny = true;
    std::mt19937 rng(11);
    for (int64_t n : {1, 7, 8, 9, 16, 33, 100000}) {
      // Random BIT PATTERNS: covers normals, subnormals, inf, both signs,
      // every exponent. This empirically pins the scalar converter to the
      // hardware VCVTPS2PH result. NaNs are tested semantically below.
      std::vector<float> src(n);
      for (float &f : src) {
        uint32_t bits = rng();
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        f = std::isnan(x) ? 1.0f : x;
      }
      std::vector<uint16_t> a(n, 0xDEAD), b(n, 0xBEEF);
      reloc::quant::convertF32F16(src.data(), a.data(), n, Variant::Scalar);
      reloc::quant::convertF32F16(src.data(), b.data(), n, v);
      ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n * sizeof(uint16_t)))
          << "variant=" << static_cast<int>(v) << " n=" << n;
    }
    float nan = NAN;
    uint16_t h = 0;
    reloc::quant::convertF32F16(&nan, &h, 1, v);
    EXPECT_EQ(h & 0x7C00u, 0x7C00u);
    EXPECT_NE(h & 0x3FFu, 0u);
  }
  if (!ranAny)
    GTEST_SKIP() << "no SIMD tier supported on this host";
}

// Hand-built 2-D transpose-style plan: dst is rows x cols dense row-major,
// src is read with swapped strides (mirrors what bind() produces for
// reloc.transpose; ExecuteTest builds plans the same way).
reloc::BoundPlan transposePlan(int64_t rows, int64_t cols) {
  reloc::BoundPlan b;
  b.extents = {rows, cols};
  b.srcStrides = {1, rows};
  b.dstStrides = {cols, 1};
  b.elementSize = 4;
  b.totalBytes = rows * cols * 4;
  return b;
}

int64_t maxSrcOffset(const reloc::BoundPlan &b) {
  int64_t off = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    off += (b.extents[k] - 1) * b.srcStrides[k];
  return off;
}

// Naive full-index-space walk: the independent oracle for the fused kernel.
void refGatherQuantize(const reloc::BoundPlan &b, const float *src,
                       int8_t *dst, const float *invScales) {
  const size_t r = b.extents.size();
  std::vector<int64_t> idx(r, 0);
  while (true) {
    int64_t so = 0, dso = 0;
    for (size_t k = 0; k < r; ++k) {
      so += idx[k] * b.srcStrides[k];
      dso += idx[k] * b.dstStrides[k];
    }
    dst[dso] = refQuantOne(src[so], invScales[idx[0]]);
    size_t k = r;
    for (;;) {
      if (k == 0)
        return;
      --k;
      if (++idx[k] < b.extents[k])
        break;
      idx[k] = 0;
    }
  }
}

TEST(GatherQuantize, ScalarMatchesNaiveWalk2D) {
  auto b = transposePlan(5, 67); // strided inner reads + remainder tail
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 3, -300.f, 300.f);
  std::vector<float> inv = {0.9f, 0.1f, 1.7f, 0.03f, 2.5f};
  const size_t total = 5 * 67;
  std::vector<int8_t> got(total, 42), want(total, 24);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), total));
}

TEST(GatherQuantize, ContiguousInnerFastPath) {
  // srcStrides.back() == 1: rows with a gap between them (row-major copy
  // out of a larger parent buffer) -- exercises the stride-1 fast path.
  reloc::BoundPlan b;
  b.extents = {4, 33};
  b.srcStrides = {40, 1};
  b.dstStrides = {33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 33 * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 5, -300.f, 300.f);
  std::vector<float> inv = {1.0f, 0.5f, 0.25f, 2.0f};
  std::vector<int8_t> got(4 * 33, 0), want(4 * 33, 1);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size()));
}

TEST(GatherQuantize, ScalarMatchesNaiveWalk3D) {
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100}; // arbitrary positive, strided innermost
  b.dstStrides = {198, 33, 1}; // dense row-major dst
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 9, -300.f, 300.f);
  std::vector<float> inv = {0.4f, 1.1f, 0.7f, 3.0f};
  std::vector<int8_t> got(4 * 6 * 33, 0), want(4 * 6 * 33, 1);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size()));
}

TEST(GatherQuantize, ChunkedEqualsWholeRange) {
  auto b = transposePlan(5, 67);
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 3, -300.f, 300.f);
  std::vector<float> inv = {0.9f, 0.1f, 1.7f, 0.03f, 2.5f};
  const size_t total = 5 * 67;
  std::vector<int8_t> whole(total, 0), chunked(total, 1);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), whole.data(), inv.data(),
                                    0, 5, Variant::Scalar);
  for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}})
    reloc::quant::gatherQuantizeF32S8(b, src.data(), chunked.data(),
                                      inv.data(), lo, hi, Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(whole.data(), chunked.data(), total));
}

uint8_t refNibble(int8_t v) {
  int x = v < -8 ? -8 : (v > 7 ? 7 : v);
  return static_cast<uint8_t>(x & 0xF);
}

TEST(PackS8S4, ScalarPacksAndSaturates) {
  const std::vector<int8_t> src = {0,   1,    -1, 7, -8, 8,
                                   -9,  127,  -128, 3, 5,  -6};
  const int64_t pairs = 6;
  std::vector<uint8_t> dst(pairs, 0xAA);
  reloc::quant::packS8S4(src.data(), dst.data(), pairs, Variant::Scalar);
  for (int64_t i = 0; i < pairs; ++i) {
    const uint8_t want = static_cast<uint8_t>(
        refNibble(src[2 * i]) | (refNibble(src[2 * i + 1]) << 4));
    EXPECT_EQ(dst[i], want) << "i=" << i;
  }
}

TEST(PackS8S4, Avx512BitExactVsScalar) {
  if (!reloc::quant::cpuSupports(Variant::AVX512))
    GTEST_SKIP() << "AVX-512 unsupported on this host";
  std::mt19937 rng(13);
  for (int64_t pairs : {1, 31, 32, 33, 100, 100003}) {
    std::vector<int8_t> src(2 * pairs);
    for (int8_t &b : src)
      b = static_cast<int8_t>(rng()); // full int8 range incl. saturating
    std::vector<uint8_t> a(pairs, 0), b(pairs, 1);
    reloc::quant::packS8S4(src.data(), a.data(), pairs, Variant::Scalar);
    reloc::quant::packS8S4(src.data(), b.data(), pairs, Variant::AVX512);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), pairs)) << "pairs=" << pairs;
  }
}

} // namespace
