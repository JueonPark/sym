//===- RtrackTest.cpp - R0.3 harness CPU-side tests (CI) ------------------===//
//
// Everything here runs without a GPU: plan builders vs independent
// index-math oracles (the issue-#63 lesson: never verify a plan against
// its own executor), chunk math, the 5+30 stats summary, CSV formatting,
// the workload table, and the quant round-trip error bound.
//
//===----------------------------------------------------------------------===//

#include "rtrack/chunking.h"
#include "rtrack/csv.h"
#include "rtrack/plans.h"
#include "rtrack/rstats.h"
#include "rtrack/workloads.h"

#include "reloc/Execute.h"
#include "reloc/Quant.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using bench::rtrack::blockedTransposePlan;
using bench::rtrack::identityPlan;
using bench::rtrack::maxSrcOffset;
using bench::rtrack::nchwToNhwcPlan;
using bench::rtrack::transposePlan;

// Walk the plan's full dst index space; call check(idx, dstOff, srcOff).
template <class Fn>
void forEachCell(const reloc::BoundPlan &b, Fn check) {
  const size_t r = b.extents.size();
  std::vector<int64_t> idx(r, 0);
  while (true) {
    int64_t so = 0, dso = 0;
    for (size_t k = 0; k < r; ++k) {
      so += idx[k] * b.srcStrides[k];
      dso += idx[k] * b.dstStrides[k];
    }
    check(idx, dso, so);
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

// src offsets must be a bijection onto [0, n^2) -- the exact property the
// pre-fix golden reference plan violated for N > 4096 (issue #63).
void expectBijective(const reloc::BoundPlan &b, int64_t total) {
  std::vector<int64_t> offs;
  offs.reserve(static_cast<size_t>(total));
  forEachCell(b, [&](const std::vector<int64_t> &, int64_t, int64_t so) {
    offs.push_back(so);
  });
  ASSERT_EQ(static_cast<int64_t>(offs.size()), total);
  std::sort(offs.begin(), offs.end());
  for (int64_t i = 0; i < total; ++i)
    ASSERT_EQ(offs[static_cast<size_t>(i)], i);
}

TEST(RtrackPlans, IdentityIsIdentity) {
  const int64_t n = 64;
  auto b = identityPlan(n);
  forEachCell(b, [&](const std::vector<int64_t> &, int64_t dso, int64_t so) {
    ASSERT_EQ(dso, so);
  });
  expectBijective(b, n * n);
  EXPECT_EQ(maxSrcOffset(b), n * n - 1);
}

TEST(RtrackPlans, TransposeMatchesIndexMath) {
  const int64_t n = 64;
  auto b = transposePlan(n);
  // dst (i, j) holds src (j, i): srcOff = j * n + i.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t dso, int64_t so) {
    ASSERT_EQ(dso, idx[0] * n + idx[1]);
    ASSERT_EQ(so, idx[1] * n + idx[0]);
  });
  expectBijective(b, n * n);
}

TEST(RtrackPlans, BlockedTransposeMatchesViewTransposeOracle) {
  const int64_t n = 128, m = n / 64;
  auto b = blockedTransposePlan(n);
  // out = x.view(N/64, 64, 64, N/64).transpose(0, 1). The plan is authored
  // rank-3 (a, bq, j) with j the merged (c, d) inner pair: c = j / m,
  // d = j % m. x_view strides (row-major): (64*n, n, m, 1), so
  // src = bq*64n + a*n + c*m + d, computed here WITHOUT the plan's strides.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t, int64_t so) {
    const int64_t a = idx[0], bq = idx[1], j = idx[2];
    const int64_t c = j / m, d = j % m;
    ASSERT_EQ(so, bq * 64 * n + a * n + c * m + d);
  });
  expectBijective(b, n * n);
}

TEST(RtrackPlans, NchwToNhwcMatchesIndexMath) {
  const int64_t n = 128;
  const int64_t C = 64, H = 64, W = n / 64;
  auto b = nchwToNhwcPlan(n);
  // dst (b, h, w, c) packed NHWC; src NCHW: b*CHW + c*HW + h*W + w.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t dso, int64_t so) {
    const int64_t bb = idx[0], h = idx[1], w = idx[2], c = idx[3];
    ASSERT_EQ(dso, ((bb * H + h) * W + w) * C + c);
    ASSERT_EQ(so, ((bb * C + c) * H + h) * W + w);
  });
  expectBijective(b, n * n);
}

// The blocked plan must also round-trip through the library executor --
// authored strides and executeH2D agree on a real buffer.
TEST(RtrackPlans, BlockedTransposeExecutesBijectively) {
  const int64_t n = 128;
  auto b = blockedTransposePlan(n);
  std::vector<float> src(static_cast<size_t>(n * n));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<float>(i);
  std::vector<float> dst(src.size(), -1.0f);
  reloc::executeH2D(b, src.data(), dst.data());
  std::vector<float> sorted = dst;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i)
    ASSERT_EQ(sorted[i], static_cast<float>(i));
}

TEST(RtrackChunks, RowChunksCoverExactly) {
  auto c = bench::rtrack::planRowChunks(/*rows=*/100, /*rowBytes=*/1000,
                                        /*chunkBytes=*/4096);
  EXPECT_EQ(c.rowsPerChunk, 4);
  EXPECT_EQ(c.nChunks, 25);
  EXPECT_EQ(c.stagingBytes, 4000);
}

TEST(RtrackChunks, OversizedRowGetsOwnChunk) {
  auto c = bench::rtrack::planRowChunks(8, /*rowBytes=*/1 << 24,
                                        /*chunkBytes=*/1 << 22);
  EXPECT_EQ(c.rowsPerChunk, 1);
  EXPECT_EQ(c.nChunks, 8);
  EXPECT_EQ(c.stagingBytes, 1 << 24); // staging grows past the request
}

TEST(RtrackChunks, ChunkLargerThanTensorIsOneChunk) {
  auto c = bench::rtrack::planRowChunks(16, 64, 1 << 20);
  EXPECT_EQ(c.rowsPerChunk, 16);
  EXPECT_EQ(c.nChunks, 1);
  auto bc = bench::rtrack::planByteChunks(1024, 1 << 20);
  EXPECT_EQ(bc.bytesPerChunk, 1024);
  EXPECT_EQ(bc.nChunks, 1);
}

TEST(RtrackChunks, ByteChunksLastShort) {
  auto bc = bench::rtrack::planByteChunks(10 << 20, 4 << 20);
  EXPECT_EQ(bc.bytesPerChunk, 4 << 20);
  EXPECT_EQ(bc.nChunks, 3); // 4 + 4 + 2 MiB
}

TEST(RtrackStats, ProtocolConstantsMatchIssue76) {
  EXPECT_EQ(bench::rtrack::kWarmup, 5);
  EXPECT_EQ(bench::rtrack::kIters, 30);
  EXPECT_DOUBLE_EQ(bench::rtrack::kIqrFlagPct, 5.0);
}

TEST(RtrackStats, MedianMinP95) {
  std::vector<double> s;
  for (int i = 20; i >= 1; --i)
    s.push_back(i); // 20..1, summarize sorts
  auto r = bench::rtrack::summarizeSamples(s);
  EXPECT_DOUBLE_EQ(r.median, 10.5);
  EXPECT_DOUBLE_EQ(r.min, 1.0);
  EXPECT_DOUBLE_EQ(r.p95, 19.05); // numpy-linear percentile
  EXPECT_EQ(r.n, 20u);
}

TEST(RtrackStats, UnstableFlagAtFivePercent) {
  // numpy-linear quartiles: q1 99.25, q3 103 -> IQR/median = 3.75%: clean.
  std::vector<double> tight = {99, 99, 100, 100, 104, 104};
  EXPECT_FALSE(bench::rtrack::summarizeSamples(tight).unstable);
  // q1 96.25, q3 103.75 -> IQR/median = 7.5%: flagged.
  std::vector<double> wide = {90, 95, 100, 100, 105, 111};
  EXPECT_TRUE(bench::rtrack::summarizeSamples(wide).unstable);
}

TEST(RtrackStats, NowMsMonotonic) {
  double a = bench::rtrack::nowMs();
  double b = bench::rtrack::nowMs();
  EXPECT_GE(b, a);
}

size_t commaCount(const std::string &s) {
  return static_cast<size_t>(std::count(s.begin(), s.end(), ','));
}

TEST(RtrackCsv, HeaderAndRowFieldCountsMatch) {
  bench::rtrack::CsvRow row;
  EXPECT_EQ(commaCount(bench::rtrack::csvHeaderLine()),
            commaCount(bench::rtrack::csvRowLine(row)));
}

TEST(RtrackCsv, RowGolden) {
  bench::rtrack::CsvRow r;
  r.machine = "epyc-2080ti";
  r.gpu = "NVIDIA GeForce RTX 2080 Ti";
  r.method = "a";
  r.transform = "transpose";
  r.dtypeOut = "s8";
  r.n = 8192;
  r.r = 0.25;
  r.threads = 8;
  r.chunkReqBytes = 4ll << 20;
  r.stagingBytes = 4ll << 20;
  r.nChunks = 16;
  r.wall = {12.5, 12.0, 13.75, 2.4, false, 30};
  r.gpuPipe = {12.4, 0, 0, 0, false, 30};
  r.cpuStage = {8.0, 0, 0, 0, false, 30};
  r.h2d = {2.75, 0, 0, 0, false, 30};
  r.gpuKernel = {0.0, 0, 0, 0, false, 30};
  r.gpuRecv = {0.0, 0, 0, 0, false, 30};
  r.h2dOcc = {0.92, 0, 0, 0, false, 30};
  r.variant = "matrix";
  r.wire = "s8";
  r.effectiveInputGbps = 21.47;
  r.verified = true;
  EXPECT_EQ(bench::rtrack::csvRowLine(r),
            "epyc-2080ti,NVIDIA GeForce RTX 2080 Ti,a,transpose,8192,s8,0.25,"
            "8,4,4194304,16,12.5,12,13.75,2.4,0,21.47,12.4,8,2.75,0,0,1,"
            "matrix,s8,0.92");
}

TEST(RtrackCsv, CommaInFieldSanitized) {
  // The -DNDEBUG benchmarking build has no asserts; a comma in --machine
  // or a GPU name must not shift the downstream columns.
  bench::rtrack::CsvRow r;
  r.machine = "EPYC 7351, 4 nodes";
  std::string line = bench::rtrack::csvRowLine(r);
  EXPECT_EQ(commaCount(line), commaCount(bench::rtrack::csvHeaderLine()));
  EXPECT_NE(line.find("EPYC 7351; 4 nodes"), std::string::npos);
}

TEST(RtrackWorkloads, TableConsistent) {
  const auto &ws = bench::rtrack::allWorkloads();
  ASSERT_EQ(ws.size(), 22u);
  for (const auto &w : ws) {
    SCOPED_TRACE(w.id);
    auto b = w.makePlan(128);
    EXPECT_EQ(b.totalBytes, 128 * 128 * 4);
    EXPECT_EQ(maxSrcOffset(b), 128 * 128 - 1);
    // r is exactly the wire width ratio (fp32 in).
    EXPECT_DOUBLE_EQ(w.r, bench::rtrack::wireRatio(w.wire));
    // Packed dst rows: outer stride == product of inner extents (the
    // chunked-staging rebase and the per-channel quantize rely on this).
    int64_t inner = 1;
    for (size_t k = 1; k < b.extents.size(); ++k)
      inner *= b.extents[k];
    EXPECT_EQ(b.dstStrides[0], inner);
    EXPECT_EQ(bench::rtrack::findWorkload(w.id), &w);
    if (std::string(w.variant) == "matrix") {
      // R1 rows: artifact == wire, no receive stage, B always runs.
      EXPECT_STREQ(bench::rtrack::wireName(w.wire),
                   bench::rtrack::dtypeName(w.dtypeOut));
      EXPECT_TRUE(w.recvStage == bench::rtrack::RecvStage::None);
      EXPECT_TRUE(w.methodB);
    } else {
      // R2 rows: fixed-f32 artifact; only the r=1.0 row measures B; the
      // receive stage exists exactly when the wire is compressed.
      EXPECT_STREQ(w.variant, "rsweep");
      EXPECT_TRUE(w.dtypeOut == bench::rtrack::DtypeOut::F32);
      EXPECT_EQ(w.methodB, w.wire == bench::rtrack::Wire::F32);
      EXPECT_EQ(w.recvStage == bench::rtrack::RecvStage::None,
                w.wire == bench::rtrack::Wire::F32);
    }
  }
  EXPECT_EQ(bench::rtrack::findWorkload("nope"), nullptr);
}

TEST(RtrackWorkloads, WireBytesMath) {
  using bench::rtrack::Wire;
  using bench::rtrack::wireBytes;
  EXPECT_EQ(wireBytes(Wire::F32, 128), 512);
  EXPECT_EQ(wireBytes(Wire::F16, 128), 256);
  EXPECT_EQ(wireBytes(Wire::S8, 128), 128);
  EXPECT_EQ(wireBytes(Wire::S4, 128), 64);
  EXPECT_DOUBLE_EQ(bench::rtrack::wireRatio(Wire::F32), 1.0);
  EXPECT_DOUBLE_EQ(bench::rtrack::wireRatio(Wire::F16), 0.5);
  EXPECT_DOUBLE_EQ(bench::rtrack::wireRatio(Wire::S8), 0.25);
  EXPECT_DOUBLE_EQ(bench::rtrack::wireRatio(Wire::S4), 0.125);
}

TEST(RtrackWorkloads, S4QuantRoundTripMaxAbsErrBound) {
  // R2 r=0.125 contract: with invScale = 7/maxAbs the s8 values land in
  // [-7, 7], the s4 nibble pack is lossless on them, and the dequantized
  // roundtrip obeys |x - q*scale| <= scale/2.
  const int64_t n = 4096;
  std::vector<float> src(static_cast<size_t>(n));
  float maxAbs = 0;
  for (int64_t i = 0; i < n; ++i) {
    src[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.37f) * 100.0f;
    maxAbs = std::max(maxAbs, std::fabs(src[static_cast<size_t>(i)]));
  }
  const float invScale = 7.0f / maxAbs, scale = 1.0f / invScale;
  std::vector<int8_t> q(src.size());
  reloc::quant::quantizePackF32S8(src.data(), q.data(), 1, n, &invScale,
                                  reloc::quant::Variant::Scalar);
  std::vector<uint8_t> packed(src.size() / 2);
  reloc::quant::packS8S4(q.data(), packed.data(),
                         static_cast<int64_t>(packed.size()),
                         reloc::quant::Variant::Scalar);
  double worst = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    const uint8_t b = packed[i / 2];
    const int8_t nib =
        (i % 2 == 0) ? static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4)
                     : static_cast<int8_t>(static_cast<int8_t>(b) >> 4);
    ASSERT_EQ(nib, q[i]); // in-range: pack is lossless
    worst = std::max(worst, std::fabs(static_cast<double>(src[i]) -
                                      static_cast<double>(nib) * scale));
  }
  EXPECT_LE(worst, 0.5 * scale * 1.0001);
}

TEST(RtrackWorkloads, GatherQuantizeOnIdentityEqualsQuantizePack) {
  // T3's Method-A kernel is quantizePackF32S8; the driver's s8 reference
  // is gatherQuantizeF32S8 over the plan. On the identity plan they must
  // agree bit-exactly.
  const int64_t n = 128;
  auto b = identityPlan(n);
  std::vector<float> src(static_cast<size_t>(n * n));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  std::vector<float> inv(static_cast<size_t>(n), 1.0f / 3.0f);
  std::vector<int8_t> a(src.size()), g(src.size());
  reloc::quant::quantizePackF32S8(src.data(), a.data(), n, n, inv.data(),
                                  reloc::quant::Variant::Scalar);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), g.data(), inv.data(), 0, n,
                                    reloc::quant::Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(a.data(), g.data(), a.size()));
}

TEST(RtrackWorkloads, QuantRoundTripMaxAbsErrBound) {
  // R0 exit criterion: |x - dequant(quant(x))| <= scale/2 for unsaturated
  // inputs when invScale = 127 / maxAbs.
  const int64_t n = 4096;
  std::vector<float> src(static_cast<size_t>(n));
  float maxAbs = 0;
  for (int64_t i = 0; i < n; ++i) {
    src[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.37f) * 100.0f;
    maxAbs = std::max(maxAbs, std::fabs(src[static_cast<size_t>(i)]));
  }
  const float invScale = 127.0f / maxAbs, scale = maxAbs / 127.0f;
  std::vector<int8_t> q(src.size());
  reloc::quant::quantizePackF32S8(src.data(), q.data(), 1, n, &invScale,
                                  reloc::quant::Variant::Scalar);
  double worst = 0;
  for (size_t i = 0; i < src.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(src[i]) -
                                      static_cast<double>(q[i]) * scale));
  EXPECT_LE(worst, 0.5 * scale * 1.0001);
}

} // namespace
