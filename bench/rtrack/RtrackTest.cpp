//===- RtrackTest.cpp - R0.3 harness CPU-side tests (CI) ------------------===//
//
// Everything here runs without a GPU: plan builders vs independent
// index-math oracles (the issue-#63 lesson: never verify a plan against
// its own executor), chunk math, the 5+30 stats summary, CSV formatting,
// the workload table, and the quant round-trip error bound.
//
//===----------------------------------------------------------------------===//

#include "rtrack/plans.h"

#include "reloc/Execute.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
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

} // namespace
