//===- QuantTest.cpp - R0.1 CPU quant kernels (issue #74) -----------------===//

#include "reloc/Quant.h"

#include "reloc/GatherPool.h"

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
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX2));
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX512));
  EXPECT_FALSE(
      reloc::quant::kernelHasVariant(Kernel::QuantizePack, Variant::AVX512Pf));
  // gather_quantize: scalar / AVX-512 gather / prefetch+tiled
  EXPECT_FALSE(
      reloc::quant::kernelHasVariant(Kernel::GatherQuantize, Variant::AVX2));
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::GatherQuantize, Variant::AVX512));
  EXPECT_TRUE(reloc::quant::kernelHasVariant(Kernel::GatherQuantize,
                                             Variant::AVX512Pf));
  // pack_s8_s4: AVX-512 (+ scalar reference)
  EXPECT_FALSE(reloc::quant::kernelHasVariant(Kernel::PackS8S4, Variant::AVX2));
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::PackS8S4, Variant::AVX512));
  // convert_f32_f16: F16C (the AVX2 tier) / AVX-512
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX2));
  EXPECT_TRUE(
      reloc::quant::kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX512));
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
  const float src[] = {0.0f,       -0.0f, 0.5f,   -0.5f,   1.5f,
                       2.5f,       -2.5f, 200.0f, -200.0f, HUGE_VALF,
                       -HUGE_VALF, NAN,   126.6f};
  const int8_t want[] = {0, 0, 0, 0, 2, 2, -2, 127, -128, 127, -128, -128, 127};
  const int64_t n = sizeof(src) / sizeof(src[0]);
  std::vector<int8_t> dst(n, 42);
  const float inv = 1.0f;
  reloc::quant::quantizePackF32S8(src, dst.data(), 1, n, &inv, Variant::Scalar);
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
      {0.0f, 0x0000},
      {-0.0f, 0x8000},
      {1.0f, 0x3C00},
      {-2.0f, 0xC000},
      {65504.0f, 0x7BFF},
      {65519.0f, 0x7BFF}, // below RNE-to-inf cut
      {65520.0f, 0x7C00},
      {HUGE_VALF, 0x7C00},
      {-HUGE_VALF, 0xFC00},
      {5.9604645e-8f, 0x0001}, // 2^-24, smallest half subnormal
      {2.9802322e-8f, 0x0000}, // 2^-25: tie, rounds to even (zero)
      {4.4703484e-8f, 0x0001}, // 1.5 * 2^-25: rounds up
      {6.1035156e-5f, 0x0400}, // 2^-14, smallest half normal
      {0.1f, 0x2E66},
      {3.14159265f, 0x4248},
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
void refGatherQuantize(const reloc::BoundPlan &b, const float *src, int8_t *dst,
                       const float *invScales) {
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
  b.srcStrides = {2, 9, 100};  // arbitrary positive, strided innermost
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
  reloc::quant::gatherQuantizeF32S8(b, src.data(), whole.data(), inv.data(), 0,
                                    5, Variant::Scalar);
  for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}})
    reloc::quant::gatherQuantizeF32S8(b, src.data(), chunked.data(), inv.data(),
                                      lo, hi, Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(whole.data(), chunked.data(), total));
}

TEST(GatherQuantize, SimdVariantsBitExactVsScalar) {
  if (!reloc::quant::cpuSupports(Variant::AVX512))
    GTEST_SKIP() << "AVX-512 unsupported on this host";
  struct PlanCase {
    const char *name;
    reloc::BoundPlan b;
  };
  std::vector<PlanCase> plans;
  plans.push_back({"transpose 129x517", transposePlan(129, 517)});
  plans.push_back({"transpose 16x16", transposePlan(16, 16)});
  {
    reloc::BoundPlan b; // contiguous inner fast path
    b.extents = {7, 133};
    b.srcStrides = {140, 1};
    b.dstStrides = {133, 1};
    b.elementSize = 4;
    b.totalBytes = 7 * 133 * 4;
    plans.push_back({"contiguous inner", b});
  }
  {
    reloc::BoundPlan b; // large stride: distinct cache line per gather lane
    b.extents = {3, 65};
    b.srcStrides = {1, 8192};
    b.dstStrides = {65, 1};
    b.elementSize = 4;
    b.totalBytes = 3 * 65 * 4;
    plans.push_back({"stride 8192", b});
  }
  for (auto &pc : plans) {
    std::vector<float> src =
        randomFloats(maxSrcOffset(pc.b) + 1, 21, -300.f, 300.f);
    std::vector<float> inv(pc.b.extents[0]);
    for (size_t c = 0; c < inv.size(); ++c)
      inv[c] = 0.03f + 0.11f * static_cast<float>(c);
    const size_t total = static_cast<size_t>(pc.b.totalBytes / 4);
    std::vector<int8_t> ref(total, 0);
    reloc::quant::gatherQuantizeF32S8(pc.b, src.data(), ref.data(), inv.data(),
                                      0, pc.b.extents[0], Variant::Scalar);
    for (Variant v : {Variant::AVX512, Variant::AVX512Pf}) {
      std::vector<int8_t> got(total, 1);
      reloc::quant::gatherQuantizeF32S8(pc.b, src.data(), got.data(),
                                        inv.data(), 0, pc.b.extents[0], v);
      ASSERT_EQ(0, std::memcmp(ref.data(), got.data(), total))
          << pc.name << " variant=" << static_cast<int>(v);
    }
  }
}

uint8_t refNibble(int8_t v) {
  int x = v < -8 ? -8 : (v > 7 ? 7 : v);
  return static_cast<uint8_t>(x & 0xF);
}

TEST(PackS8S4, ScalarPacksAndSaturates) {
  const std::vector<int8_t> src = {0, 1, -1, 7, -8, 8, -9, 127, -128, 3, 5, -6};
  const int64_t pairs = 6;
  std::vector<uint8_t> dst(pairs, 0xAA);
  reloc::quant::packS8S4(src.data(), dst.data(), pairs, Variant::Scalar);
  for (int64_t i = 0; i < pairs; ++i) {
    const uint8_t want = static_cast<uint8_t>(refNibble(src[2 * i]) |
                                              (refNibble(src[2 * i + 1]) << 4));
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

//===----------------------------------------------------------------------===//
// C1 witness vectors (docs/reloc-typed-semantics.md, "Witness vectors"): pin
// the existing kernels to the declared `symmetric_rne` / `ieee_rne` policies
// on the exact inputs the contract names, so a kernel change that silently
// alters a policy fails here before any typed plan can rely on it.
//===----------------------------------------------------------------------===//

// Unit scale, zero point 0. The tie vector must produce [-2,-2,0,0,2,2]:
// round-to-nearest-even. Saturation clamps the limits; NaN maps to -128 and
// the infinities saturate.
const float kQuantWitness[] = {-2.5f, -1.5f,  -0.5f,     0.5f,       1.5f,
                               2.5f,  -129.f, -128.f,    -127.f,     126.f,
                               127.f, 128.f,  HUGE_VALF, -HUGE_VALF, NAN};
const int8_t kQuantWant[] = {-2,   -2,  0,   0,   2,   2,    -128, -128,
                             -127, 126, 127, 127, 127, -128, -128};
constexpr int64_t kQuantWitnessCount =
    sizeof(kQuantWitness) / sizeof(kQuantWitness[0]);

void expectSymmetricRneWitness(Variant v) {
  // Cycle the witnesses through 64 lanes so SIMD bodies (32 per AVX2 step,
  // 16 per AVX-512 step) see them inside full vectors, not only in the tail.
  const int64_t n = 64;
  std::vector<float> src(n);
  std::vector<int8_t> want(n);
  for (int64_t i = 0; i < n; ++i) {
    src[i] = kQuantWitness[i % kQuantWitnessCount];
    want[i] = kQuantWant[i % kQuantWitnessCount];
  }
  const float inv = 1.0f;
  std::vector<int8_t> dst(n, 42);
  reloc::quant::quantizePackF32S8(src.data(), dst.data(), 1, n, &inv, v);
  for (int64_t i = 0; i < n; ++i)
    EXPECT_EQ(dst[i], want[i])
        << "variant=" << static_cast<int>(v) << " i=" << i << " x=" << src[i];
}

TEST(TypedSemantics, SymmetricRneQuantizeWitnessVectors) {
  expectSymmetricRneWitness(Variant::Scalar);
  for (Variant v : {Variant::AVX2, Variant::AVX512})
    if (reloc::quant::cpuSupports(v))
      expectSymmetricRneWitness(v);
}

TEST(TypedSemantics, ReciprocalScaleIsFormedOnceInBinary32) {
  // The contract is t = fl32(x * inv) with inv = fl32(1 / scale): two
  // binary32 roundings. x / scale is a single correctly rounded division and
  // is NOT bit-equivalent; the kernels consume inv, and so does the
  // reference. Evidence for the non-equivalence at scale 0.3 (the same
  // witness the pinned PyTorch build reproduces: 3 of 8 differ).
  const float scale = 0.3f;
  const float inv = 1.0f / scale;
  const float xs[] = {0.1f, 0.3f, 0.7f, 1.1f, 2.9f, 10.1f, 100.3f, 1000.7f};
  int differ = 0;
  for (float x : xs) {
    const float byInv = x * inv, byDiv = x / scale;
    uint32_t a, b;
    std::memcpy(&a, &byInv, sizeof(a));
    std::memcpy(&b, &byDiv, sizeof(b));
    differ += a != b;
  }
  EXPECT_EQ(differ, 3);
  std::vector<int8_t> dst(8, 42);
  reloc::quant::quantizePackF32S8(xs, dst.data(), 1, 8, &inv, Variant::Scalar);
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(dst[i], refQuantOne(xs[i], inv)) << "i=" << i;
}

TEST(TypedSemantics, PerChannelAxisIsTheOperandAxisNotTheKernelChannel) {
  // Witness: shape [2, 3, 4], channel axis 1, scales {0.5, 1, 2},
  // x[i] = i - 11.5. The declared semantics index the parameter by the
  // operand's axis-1 coordinate. The existing kernels only know
  // "channel = outermost contiguous block" (quantizePackF32S8) or
  // "channel = outermost coalesced plan axis" (gatherQuantizeF32S8), so a
  // non-outer channel axis must be reached by slicing or by a plan that
  // moves the channel axis outermost -- the mapping C2 must carry through
  // folds and R3 must establish before dispatch.
  float x[24];
  for (int i = 0; i < 24; ++i)
    x[i] = static_cast<float>(i) - 11.5f;
  const float inv[3] = {1.0f / 0.5f, 1.0f / 1.0f, 1.0f / 2.0f}; // exact
  int8_t want[24];
  for (int a = 0; a < 2; ++a)
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 4; ++k)
        want[a * 12 + c * 4 + k] = refQuantOne(x[a * 12 + c * 4 + k], inv[c]);
  // torch.quantize_per_channel(x, scales, zeros, axis=1, qint8).int_repr()[0]
  // on the pinned PyTorch 2.14.0 CPU build (docs table):
  const int8_t torchSlice0[12] = {-23, -21, -19, -17, -8, -6,
                                  -6,  -4,  -2,  -1,  -1, 0};
  for (int i = 0; i < 12; ++i)
    EXPECT_EQ(want[i], torchSlice0[i]) << "i=" << i;

  // Slice loop: one contiguous kernel call per outer index.
  int8_t sliced[24];
  for (int a = 0; a < 2; ++a)
    reloc::quant::quantizePackF32S8(x + a * 12, sliced + a * 12, 3, 4, inv,
                                    Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(want, sliced, sizeof(want)));

  // Plan form: a relocation whose outermost axis IS the channel axis
  // (dst layout [c][a][k]) feeds the fused kernel directly.
  reloc::BoundPlan b;
  b.extents = {3, 2, 4};
  b.srcStrides = {4, 12, 1};
  b.dstStrides = {8, 4, 1};
  b.elementSize = 4;
  b.totalBytes = 24 * 4;
  int8_t relocated[24];
  reloc::quant::gatherQuantizeF32S8(b, x, relocated, inv, 0, 3,
                                    Variant::Scalar);
  for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 2; ++a)
      for (int k = 0; k < 4; ++k)
        EXPECT_EQ(relocated[c * 8 + a * 4 + k], want[a * 12 + c * 4 + k])
            << "c=" << c << " a=" << a << " k=" << k;
}

TEST(TypedSemantics, IeeeRneNarrowingWitnessVectors) {
  struct Case {
    float in;
    uint16_t want;
  } cases[] = {
      {0.0f, 0x0000},                // +0
      {-0.0f, 0x8000},               // -0 (sign preserved)
      {0x1p-24f, 0x0001},            // 2^-24: smallest binary16 subnormal
      {0x1p-25f, 0x0000},            // 2^-25: tie between 0 and 2^-24 -> even
      {0x1.8p-25f, 0x0001},          // above the tie: rounds up
      {0x1p-14f, 0x0400},            // 2^-14: smallest binary16 normal
      {65504.0f, 0x7BFF},            // largest finite binary16
      {65519.99f, 0x7BFF},           // below the overflow tie
      {65520.0f, 0x7C00},            // tie between 65504 and 2^16 -> inf
      {1e5f, 0x7C00},                // overflow -> +inf, never saturates
      {HUGE_VALF, 0x7C00},           // +inf
      {-HUGE_VALF, 0xFC00},          // -inf
      {1.0f + 0x1p-11f, 0x3C00},     // tie between 1.0 and 1+2^-10 -> even
      {1.0f + 3 * 0x1p-12f, 0x3C01}, // above the tie: rounds up
  };
  std::vector<Variant> variants = {Variant::Scalar};
  for (Variant v : {Variant::AVX2, Variant::AVX512})
    if (reloc::quant::cpuSupports(v))
      variants.push_back(v);
  for (Variant v : variants) {
    for (const Case &c : cases) {
      // Eight copies fill one F16C vector so the SIMD body handles the value.
      float in[8];
      uint16_t out[8];
      std::fill(in, in + 8, c.in);
      reloc::quant::convertF32F16(in, out, 8, v);
      for (int i = 0; i < 8; ++i)
        EXPECT_EQ(out[i], c.want)
            << "variant=" << static_cast<int>(v) << " in=" << c.in;
    }
    // NaN: the result must be a NaN; payload and sign are outside conformance.
    uint32_t payloadBits = 0x7fc12345u;
    float payloadNan;
    std::memcpy(&payloadNan, &payloadBits, sizeof(payloadNan));
    uint16_t h = 0;
    reloc::quant::convertF32F16(&payloadNan, &h, 1, v);
    EXPECT_EQ(h & 0x7C00u, 0x7C00u);
    EXPECT_NE(h & 0x3FFu, 0u);
  }
}

TEST(TypedSemantics, AffineDequantizeReferenceBits) {
  // out = fl32((q - zero_point) * scale): the integer difference is exact
  // (|q - zp| <= 255), then one binary32 rounding. libreloc has no CPU
  // dequantize kernel (an R3 gap); this pins the reference arithmetic to the
  // bits the pinned PyTorch 2.14.0 build produces for q * 0.3 (CPU and CUDA
  // agree, and torch.dequantize matches).
  const int8_t q[] = {-128, -1, 0, 1, 127};
  const uint32_t want[] = {0xc219999au, 0xbe99999au, 0x00000000u, 0x3e99999au,
                           0x42186667u};
  for (int i = 0; i < 5; ++i) {
    const float out = static_cast<float>(q[i]) * 0.3f;
    uint32_t bits;
    std::memcpy(&bits, &out, sizeof(bits));
    EXPECT_EQ(bits, want[i]) << "q=" << static_cast<int>(q[i]);
  }
  // Nonzero zero point: q = -128, zp = 127 -> (-255) * 0.3f.
  const float shifted =
      static_cast<float>(static_cast<int32_t>(-128) - 127) * 0.3f;
  uint32_t shiftedBits;
  std::memcpy(&shiftedBits, &shifted, sizeof(shiftedBits));
  EXPECT_EQ(shiftedBits, 0xc2990000u);
}

TEST(QuantParallel, AllWrappersMatchSerial) {
  reloc::GatherPool pool(4);
  // quantize_pack: 1037 channels x 1031 elements: > 2 chunks past the 1
  // MiB/worker floor, odd split boundaries
  {
    const int64_t ch = 1037, cs = 1031;
    std::vector<float> src = randomFloats(ch * cs, 31, -300.f, 300.f);
    std::vector<float> inv(ch, 0.21f);
    std::vector<int8_t> a(ch * cs, 0), b(ch * cs, 1);
    reloc::quant::quantizePackF32S8(src.data(), a.data(), ch, cs, inv.data());
    reloc::quant::quantizePackF32S8Parallel(pool, src.data(), b.data(), ch, cs,
                                            inv.data());
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
  }
  // gather_quantize on a transpose plan: 4099 rows x 263 cols: > 2 chunks
  // past the 1 MiB/worker floor, odd split boundaries
  {
    auto p = transposePlan(4099, 263);
    std::vector<float> src =
        randomFloats(maxSrcOffset(p) + 1, 33, -300.f, 300.f);
    std::vector<float> inv(4099);
    for (size_t c = 0; c < inv.size(); ++c)
      inv[c] = 0.02f + 0.001f * static_cast<float>(c);
    std::vector<int8_t> a(4099 * 263, 0), b(4099 * 263, 1);
    reloc::quant::gatherQuantizeF32S8(p, src.data(), a.data(), inv.data(), 0,
                                      p.extents[0]);
    reloc::quant::gatherQuantizeF32S8Parallel(pool, p, src.data(), b.data(),
                                              inv.data());
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
  }
  // pack_s8_s4: 2097157 pairs: > 2 chunks past the 1 MiB/worker floor, odd
  // split boundaries
  {
    const int64_t pairs = 2097157;
    std::mt19937 rng(35);
    std::vector<int8_t> src(2 * pairs);
    for (int8_t &x : src)
      x = static_cast<int8_t>(rng());
    std::vector<uint8_t> a(pairs, 0), b(pairs, 1);
    reloc::quant::packS8S4(src.data(), a.data(), pairs);
    reloc::quant::packS8S4Parallel(pool, src.data(), b.data(), pairs);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), pairs));
  }
  // convert_f32_f16
  {
    const int64_t n = (1 << 20) + 37;
    std::vector<float> src = randomFloats(n, 37, -70000.f, 70000.f);
    std::vector<uint16_t> a(n, 0), b(n, 1);
    reloc::quant::convertF32F16(src.data(), a.data(), n);
    reloc::quant::convertF32F16Parallel(pool, src.data(), b.data(), n);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n * sizeof(uint16_t)));
  }
  pool.close();
}

TEST(QuantParallel, NonDisjointDstRowsFallBackInline) {
  // dstStrides[0] < inner span: outer rows alias in dst, so the parallel
  // wrapper must serialize (same guard as executeH2DThreaded) and still
  // produce exactly the serial result.
  reloc::BoundPlan b;
  b.extents = {6, 8};
  b.srcStrides = {8, 1};
  b.dstStrides = {4, 1}; // rows overlap: span 7 >= stride 4
  b.elementSize = 4;
  b.totalBytes = (5 * 4 + 7 + 1) * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 41, -10.f, 10.f);
  std::vector<float> inv(6, 1.0f);
  const size_t total = 5 * 4 + 7 + 1;
  std::vector<int8_t> a(total, 0), c(total, 1);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), a.data(), inv.data(), 0, 6,
                                    Variant::Scalar);
  reloc::GatherPool pool(4);
  reloc::quant::gatherQuantizeF32S8Parallel(pool, b, src.data(), c.data(),
                                            inv.data(), Variant::Scalar);
  pool.close();
  EXPECT_EQ(0, std::memcmp(a.data(), c.data(), total));
}

} // namespace
