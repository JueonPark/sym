//===- protocol.h - E4 measurement-protocol skeleton (issue #47) -*- C++
//-*-===//
//
// The build-doc §3 measurement protocol as a reusable header: warmup +
// timed iterations, median + IQR summaries, and re-run analysis for
// run-to-run variance (the <5% stability bar). Header-only and std-only by
// contract -- E4 inherits this file. GPU timing (CUDA events) belongs
// INSIDE the caller's iteration callable, which may return its own
// per-iteration milliseconds; the harness always records steady_clock wall
// time around every timed iteration regardless.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_PROTOCOL_H
#define BENCH_PROTOCOL_H

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bench {

/// Build-doc §3 protocol constants.
inline constexpr int kWarmupIters = 10;
inline constexpr int kTimedIters = 50;
inline constexpr int kReruns = 3;

struct Stats {
  double median = 0;
  double q1 = 0;
  double q3 = 0;
  double iqr = 0;
  size_t n = 0;
};

/// Linear-interpolation percentile over a sorted sample vector (numpy's
/// 'linear' method): position = p * (n - 1), interpolate between ranks.
inline double percentileSorted(const std::vector<double> &sorted, double p) {
  if (sorted.empty())
    return 0.0;
  if (sorted.size() == 1)
    return sorted.front();
  double pos = p * static_cast<double>(sorted.size() - 1);
  size_t lo = static_cast<size_t>(pos);
  double frac = pos - static_cast<double>(lo);
  if (lo + 1 >= sorted.size())
    return sorted.back();
  return sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
}

inline Stats summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Stats s;
  s.n = samples.size();
  s.median = percentileSorted(samples, 0.50);
  s.q1 = percentileSorted(samples, 0.25);
  s.q3 = percentileSorted(samples, 0.75);
  s.iqr = s.q3 - s.q1;
  return s;
}

/// Samples from one rerun. extra_ms holds the caller's own per-iteration
/// metric (e.g. CUDA-event GPU ms) when the callable returns double; it is
/// empty for void callables.
struct RerunSamples {
  std::vector<double> wall_ms;
  std::vector<double> extra_ms;
};

/// One rerun: `warmup` untimed calls, then `iters` timed calls. The
/// callable is invoked identically in both phases.
template <class F>
RerunSamples runOnce(F &&iter, int warmup = kWarmupIters,
                     int iters = kTimedIters) {
  using Clock = std::chrono::steady_clock;
  constexpr bool kHasMetric = !std::is_void_v<std::invoke_result_t<F &>>;
  RerunSamples out;
  out.wall_ms.reserve(static_cast<size_t>(iters));
  if constexpr (kHasMetric)
    out.extra_ms.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < warmup; ++i)
    (void)iter();
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    if constexpr (kHasMetric) {
      double metric = iter();
      out.wall_ms.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      out.extra_ms.push_back(metric);
    } else {
      iter();
      out.wall_ms.push_back(
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
  }
  return out;
}

/// Per-metric re-run analysis: one Stats per rerun plus the spread of the
/// rerun medians -- the issue's "run-to-run median spread < 5%" bar.
struct Series {
  std::vector<Stats> reruns;
  double medianSpreadPct = 0;
};

inline Series analyzeReruns(const std::vector<std::vector<double>> &perRerun) {
  Series s;
  double minMed = 0, maxMed = 0;
  for (size_t i = 0; i < perRerun.size(); ++i) {
    Stats st = summarize(perRerun[i]);
    if (i == 0) {
      minMed = maxMed = st.median;
    } else {
      minMed = std::min(minMed, st.median);
      maxMed = std::max(maxMed, st.median);
    }
    s.reruns.push_back(st);
  }
  s.medianSpreadPct = minMed > 0 ? 100.0 * (maxMed - minMed) / minMed : 0.0;
  return s;
}

// --- minimal JSON emission (drivers compose full documents) --------------

inline std::string jsonNumber(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

inline std::string statsToJson(const Stats &s) {
  std::string out = "{\"median\": " + jsonNumber(s.median);
  out += ", \"q1\": " + jsonNumber(s.q1);
  out += ", \"q3\": " + jsonNumber(s.q3);
  out += ", \"iqr\": " + jsonNumber(s.iqr);
  out += ", \"n\": " + std::to_string(s.n) + "}";
  return out;
}

inline std::string seriesToJson(const Series &s) {
  std::string out = "{\"reruns\": [";
  for (size_t i = 0; i < s.reruns.size(); ++i) {
    if (i)
      out += ", ";
    out += statsToJson(s.reruns[i]);
  }
  out += "], \"median_spread_pct\": " + jsonNumber(s.medianSpreadPct) + "}";
  return out;
}

} // namespace bench

#endif // BENCH_PROTOCOL_H
