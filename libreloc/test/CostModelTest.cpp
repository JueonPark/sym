//===- CostModelTest.cpp - P3b cost-model component (issue #97) -----------===//

#include "reloc/CostModel.h"

#include "gtest/gtest.h"

#include <string>
#include <variant>

namespace {

using reloc::costmodel::CostModel;

const char *kMinimal = R"(# costmodel calibration v0
# machine: testbox
pcie.h2d_gbps 13.07
cpu.t8.contiguous.quantize_pack_gbps 22.76  # trailing comment
hbm.bw_gbps 544
)";

CostModel mustParse(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<CostModel>(r)) << std::get<std::string>(r);
  return std::get<CostModel>(r);
}

std::string mustFail(const std::string &text) {
  auto r = CostModel::parse(text);
  EXPECT_TRUE(std::holds_alternative<std::string>(r));
  return std::holds_alternative<std::string>(r) ? std::get<std::string>(r)
                                                : std::string();
}

TEST(CostModelParse, MinimalFileRoundTrips) {
  CostModel m = mustParse(kMinimal);
  EXPECT_EQ(m.machine(), "testbox");
  EXPECT_TRUE(m.has("pcie.h2d_gbps"));
  EXPECT_DOUBLE_EQ(m.at("pcie.h2d_gbps"), 13.07);
  EXPECT_DOUBLE_EQ(m.at("cpu.t8.contiguous.quantize_pack_gbps"), 22.76);
  EXPECT_FALSE(m.has("absent.key"));
  EXPECT_DOUBLE_EQ(m.get("absent.key", -1.0), -1.0);
}

TEST(CostModelParse, RejectsMissingOrWrongVersionHeader) {
  EXPECT_NE(mustFail("pcie.h2d_gbps 13.07\n").find("version"),
            std::string::npos);
  EXPECT_NE(mustFail("# costmodel calibration v1\nk 1\n").find("version"),
            std::string::npos);
}

TEST(CostModelParse, RejectsMalformedLines) {
  const std::string base = "# costmodel calibration v0\n";
  // No value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps\n").find("line 2"),
            std::string::npos);
  // Non-numeric value.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps fast\n").find("line 2"),
            std::string::npos);
  // Trailing junk after the value that is not a comment.
  EXPECT_NE(mustFail(base + "pcie.h2d_gbps 13.07 junk\n").find("line 2"),
            std::string::npos);
}

TEST(CostModelParse, RejectsDuplicateKeys) {
  const std::string text = "# costmodel calibration v0\n"
                           "pcie.h2d_gbps 13.07\n"
                           "pcie.h2d_gbps 26.79\n";
  EXPECT_NE(mustFail(text).find("duplicate"), std::string::npos);
}

TEST(CostModelParse, LoadReportsUnreadablePath) {
  auto r = CostModel::load("/nonexistent/path.cal");
  ASSERT_TRUE(std::holds_alternative<std::string>(r));
}

using reloc::costmodel::classify;
using reloc::costmodel::cpuBw;
using reloc::costmodel::pathCosts;
using reloc::costmodel::Pattern;

reloc::BoundPlan planWithL(int64_t elems, int64_t L) {
  reloc::BoundPlan b;
  b.extents = {elems / L > 0 ? elems / L : 1, L};
  b.srcStrides = {1, elems / L > 0 ? elems / L : 1}; // shape-only fixture
  b.dstStrides = {L, 1};
  b.elementSize = 4;
  b.totalBytes = elems * 4;
  b.L = L;
  return b;
}

TEST(CostModelClassify, MapsLToPattern) {
  EXPECT_EQ(classify(planWithL(4096, 4096)), Pattern::Contiguous);
  EXPECT_EQ(classify(planWithL(4096, 1)), Pattern::SingleElement);
  EXPECT_EQ(classify(planWithL(4096, 64)), Pattern::Blocked); // == floor
  EXPECT_EQ(classify(planWithL(4096, 512)), Pattern::Blocked);
  EXPECT_EQ(classify(planWithL(4096, 8)), Pattern::Tiled);
}

// Synthetic calibration with easy numbers: link 10 GB/s, contiguous CPU
// kernels all 20 GB/s, blocked gather 5 GB/s, HBM m=2 with BW 100.
const char *kSynth = R"(# costmodel calibration v0
pcie.h2d_gbps 10
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.contiguous.convert_f32_f16_gbps 20
cpu.t8.contiguous.quantize_pack_gbps 20
cpu.t8.contiguous.pack_s8_s4_gbps 20
cpu.t8.blocked.gather_f32_gbps 5
cpu.t8.blocked.convert_f32_f16_gbps 20
cpu.t8.blocked.gather_quantize_gbps 4
cpu.t8.blocked.pack_s8_s4_gbps 20
hbm.bw_gbps 100
hbm.m.contiguous 1
hbm.m.blocked 2
multigpu.delivery_gbps.k4 30
overhead.a_ms 0.5
overhead.b_ms 0.1
)";

TEST(CostModelCpuBw, FigureRstarComposition) {
  CostModel m = mustParse(kSynth);
  // Contiguous r=1.0 -> contig_read.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Contiguous, 1.0, 8), 20.0);
  // Contiguous r=0.25 -> quantize_pack.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Contiguous, 0.25, 8), 20.0);
  // Strided r=0.5 -> harmonic(gather_f32, convert): 1/(1/5+1/20) = 4.
  EXPECT_DOUBLE_EQ(*cpuBw(m, Pattern::Blocked, 0.5, 8), 4.0);
  // Strided r=0.125 -> harmonic(gather_quantize, 4*pack):
  // 1/(1/4 + 1/80) = 80/21.
  EXPECT_NEAR(*cpuBw(m, Pattern::Blocked, 0.125, 8), 80.0 / 21.0, 1e-12);
  // Missing tier -> nullopt.
  EXPECT_FALSE(cpuBw(m, Pattern::Blocked, 1.0, 1).has_value());
  // Unmodelled r -> nullopt (not a crash).
  EXPECT_FALSE(cpuBw(m, Pattern::Blocked, 0.3, 8).has_value());
}

TEST(CostModelPathCosts, AffineFormsAndK) {
  CostModel m = mustParse(kSynth);
  const int64_t S = 1000000000; // 1 GB -> 1 s per 1 GB/s: easy arithmetic
  // Contiguous r=0.25, K=1: A slope = max(1/20, 0.25/10)=0.05 ms/MB ->
  // tA = 0.5 + 1000*0.05... in ms: S/1e9 * 1e3 * max(1/20, 0.025) = 50ms.
  auto pc = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 1, false);
  ASSERT_TRUE(pc.has_value());
  EXPECT_NEAR(pc->tAMs, 0.5 + 50.0, 1e-9);
  // B: max(1/10, 1/100) = 0.1 s -> 100 ms + 0.1 intercept.
  EXPECT_NEAR(pc->tBMs, 0.1 + 100.0, 1e-9);
  // Affine decomposition consistent: t = intercept + slope*S.
  EXPECT_NEAR(pc->aInterceptMs + pc->aSlopeMsPerByte * S, pc->tAMs, 1e-9);
  EXPECT_NEAR(pc->bInterceptMs + pc->bSlopeMsPerByte * S, pc->tBMs, 1e-9);
  // K=4 scatter: delivery 30 GB/s aggregate. B ships S at 30 -> 33.3ms;
  // A CPU still 50ms (flat in K), A DMA r*S at 30 -> 8.3ms -> max = CPU.
  auto pc4 = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 4, false);
  ASSERT_TRUE(pc4.has_value());
  EXPECT_NEAR(pc4->tBMs, 0.1 + 1000.0 / 30.0, 1e-6);
  EXPECT_NEAR(pc4->tAMs, 0.5 + 50.0, 1e-6);
  // K=4 broadcast: B ships 4S -> 133.3ms; A ships 4*r*S=1S -> 33.3ms DMA,
  // CPU 50 -> max 50.
  auto pb4 = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 4, true);
  ASSERT_TRUE(pb4.has_value());
  EXPECT_NEAR(pb4->tBMs, 0.1 + 4000.0 / 30.0, 1e-6);
  EXPECT_NEAR(pb4->tAMs, 0.5 + 50.0, 1e-6);
  // Missing K key -> nullopt.
  EXPECT_FALSE(
      pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 2, false).has_value());
}

using reloc::costmodel::decide;
using reloc::costmodel::MethodDecision;

TEST(CostModelDecide, PicksWinnerAndThreshold) {
  CostModel m = mustParse(kSynth);
  // Contiguous r=0.25: slopes A=0.05ms/MB? (see PathCosts test) -- A slope
  // 5e-8 ms/B, B slope 1e-7 ms/B; intercepts A=0.5, B=0.1. Lines cross at
  // S* = (0.5-0.1)/(1e-7-5e-8) = 8e6 bytes. Below: B wins; above: A.
  auto small = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8);
  ASSERT_TRUE(small.has_value());
  EXPECT_EQ(small->method, MethodDecision::Method::B);
  EXPECT_NEAR(small->thresholdBytes, 8e6, 1.0);
  auto big = decide(m, Pattern::Contiguous, 1 << 30, 0.25, 8);
  ASSERT_TRUE(big.has_value());
  EXPECT_EQ(big->method, MethodDecision::Method::A);
  EXPECT_NEAR(big->thresholdBytes, 8e6, 1.0);
}

TEST(CostModelDecide, SizeIndependentDecisionHasNoThreshold) {
  // Same slopes ordering as intercepts ordering -> no crossing.
  const char *cal = R"(# costmodel calibration v0
pcie.h2d_gbps 10
cpu.t8.contiguous.quantize_pack_gbps 40
hbm.bw_gbps 100
hbm.m.contiguous 1
overhead.a_ms 0.05
overhead.b_ms 0.1
)";
  CostModel m = mustParse(cal);
  // A slope max(1/40, .25/10)=0.025 < B slope 0.1; A intercept smaller too.
  auto d = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->method, MethodDecision::Method::A);
  EXPECT_DOUBLE_EQ(d->thresholdBytes, -1);
}

TEST(CostModelDecide, PrefoldArmDelegatesToV4Rule) {
  CostModel m = mustParse(kSynth);
  // nReuse=16 amortizes the 50ms CPU pass to ~3.1ms/load: APrefold wins.
  auto d = decide(m, Pattern::Contiguous, 1000000000, 0.25, 8, 1, 16);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->method, MethodDecision::Method::APrefold);
  EXPECT_LT(d->tAMs, 30.0); // amortized per-load, way under B's 100ms
  // nReuse=1 (cold single-use): prefoldWins says no -> plain A vs B.
  auto d1 = decide(m, Pattern::Contiguous, 1000000000, 0.25, 8, 1, 1);
  ASSERT_TRUE(d1.has_value());
  EXPECT_NE(d1->method, MethodDecision::Method::APrefold);
}

TEST(CostModelDecide, ThresholdAgreesWithBruteForce) {
  // Issue #97 acceptance: threshold precompute vs brute-force agreement.
  CostModel m = mustParse(kSynth);
  for (double r : {1.0, 0.5, 0.25, 0.125}) {
    for (Pattern p : {Pattern::Contiguous, Pattern::Blocked}) {
      auto probe = decide(m, p, 1 << 20, r, 8);
      if (!probe.has_value())
        continue;
      const double thr = probe->thresholdBytes;
      // Brute-force every S against the direct tAMs/tBMs comparison, and
      // separately confirm the stored boundary is the *only* place the
      // method flips across the whole scanned range.
      MethodDecision::Method prevMethod = MethodDecision::Method::B;
      bool havePrev = false;
      int flips = 0;
      for (int64_t S = 1 << 12; S <= (1ll << 34); S <<= 1) {
        auto d = decide(m, p, S, r, 8);
        ASSERT_TRUE(d.has_value());
        auto pc = pathCosts(m, p, S, r, 8, 1, false);
        ASSERT_TRUE(pc.has_value());
        EXPECT_EQ(d->method == MethodDecision::Method::A, pc->tAMs <= pc->tBMs)
            << patternName(p) << " r=" << r << " S=" << S;
        if (havePrev && d->method != prevMethod)
          ++flips;
        prevMethod = d->method;
        havePrev = true;
      }
      EXPECT_EQ(flips, thr > 0 ? 1 : 0)
          << patternName(p) << " r=" << r << " thr=" << thr;
    }
  }
}

} // namespace
