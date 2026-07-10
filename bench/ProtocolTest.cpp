//===- ProtocolTest.cpp - protocol.h statistics/harness tests -------------===//

#include "protocol.h"
#include "gtest/gtest.h"

#include <vector>

namespace {

TEST(Protocol, ConstantsMatchBuildDocSection3) {
  EXPECT_EQ(bench::kWarmupIters, 10);
  EXPECT_EQ(bench::kTimedIters, 50);
  EXPECT_EQ(bench::kReruns, 3);
}

TEST(Protocol, MedianOddAndEven) {
  EXPECT_DOUBLE_EQ(bench::summarize({3, 1, 2}).median, 2.0);
  EXPECT_DOUBLE_EQ(bench::summarize({4, 1, 3, 2}).median, 2.5);
}

TEST(Protocol, QuartilesLinearInterpolation) {
  // numpy 'linear' percentiles of [1,2,3,4]: q1 = 1.75, q3 = 3.25.
  bench::Stats s = bench::summarize({4, 2, 1, 3});
  EXPECT_DOUBLE_EQ(s.q1, 1.75);
  EXPECT_DOUBLE_EQ(s.q3, 3.25);
  EXPECT_DOUBLE_EQ(s.iqr, 1.5);
  EXPECT_EQ(s.n, 4u);
}

TEST(Protocol, SingleSampleDegenerate) {
  bench::Stats s = bench::summarize({7.5});
  EXPECT_DOUBLE_EQ(s.median, 7.5);
  EXPECT_DOUBLE_EQ(s.q1, 7.5);
  EXPECT_DOUBLE_EQ(s.q3, 7.5);
  EXPECT_DOUBLE_EQ(s.iqr, 0.0);
}

TEST(Protocol, RunOnceCountsWarmupAndTimedVoid) {
  int calls = 0;
  bench::RerunSamples r =
      bench::runOnce([&] { ++calls; }, /*warmup=*/3, /*iters=*/5);
  EXPECT_EQ(calls, 8);             // warmup + timed both execute
  EXPECT_EQ(r.wall_ms.size(), 5u); // only timed iterations are recorded
  EXPECT_TRUE(r.extra_ms.empty()); // void callable -> no extra metric
  for (double w : r.wall_ms)
    EXPECT_GE(w, 0.0);
}

TEST(Protocol, RunOnceRecordsCallerMetric) {
  int calls = 0;
  bench::RerunSamples r = bench::runOnce(
      [&]() -> double { return 10.0 + (calls++, 0); }, /*warmup=*/2,
      /*iters=*/4);
  EXPECT_EQ(r.wall_ms.size(), 4u);
  ASSERT_EQ(r.extra_ms.size(), 4u);
  for (double g : r.extra_ms)
    EXPECT_DOUBLE_EQ(g, 10.0);
}

TEST(Protocol, AnalyzeRerunsSpread) {
  // Rerun medians 10.0 and 10.5 -> spread = 100*(10.5-10)/10 = 5%.
  bench::Series s = bench::analyzeReruns({{10, 10, 10}, {10.5, 10.5, 10.5}});
  ASSERT_EQ(s.reruns.size(), 2u);
  EXPECT_DOUBLE_EQ(s.medianSpreadPct, 5.0);
}

TEST(Protocol, JsonEmission) {
  bench::Stats s = bench::summarize({1, 2, 3, 4});
  EXPECT_EQ(bench::statsToJson(s),
            "{\"median\": 2.5, \"q1\": 1.75, \"q3\": 3.25, \"iqr\": 1.5, "
            "\"n\": 4}");
  bench::Series series = bench::analyzeReruns({{2, 2, 2}, {2, 2, 2}});
  EXPECT_EQ(bench::seriesToJson(series),
            "{\"reruns\": [{\"median\": 2, \"q1\": 2, \"q3\": 2, \"iqr\": 0, "
            "\"n\": 3}, {\"median\": 2, \"q1\": 2, \"q3\": 2, \"iqr\": 0, "
            "\"n\": 3}], \"median_spread_pct\": 0}");
}

} // namespace
