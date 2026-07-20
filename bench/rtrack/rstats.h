//===- rstats.h - issue #76 measurement protocol ----------------*- C++ -*-===//
//
// 5 warmup + 30 timed iterations; report median, min, p95; flag any config
// with IQR/median > 5%. Percentiles reuse bench/protocol.h's numpy-linear
// interpolation. nowMs() is steady_clock bracketed by compiler fences so a
// stage boundary cannot be reordered around the read.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_RSTATS_H
#define BENCH_RTRACK_RSTATS_H

#include "protocol.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace bench {
namespace rtrack {

inline constexpr int kWarmup = 5;
inline constexpr int kIters = 30;
inline constexpr double kIqrFlagPct = 5.0;

struct RStats {
  double median = 0;
  double min = 0;
  double p95 = 0;
  double iqrOverMedianPct = 0;
  bool unstable = false;
  size_t n = 0;
};

inline RStats summarizeSamples(std::vector<double> samples) {
  RStats r;
  if (samples.empty())
    return r;
  std::sort(samples.begin(), samples.end());
  r.n = samples.size();
  r.min = samples.front();
  r.median = bench::percentileSorted(samples, 0.50);
  r.p95 = bench::percentileSorted(samples, 0.95);
  const double iqr = bench::percentileSorted(samples, 0.75) -
                     bench::percentileSorted(samples, 0.25);
  r.iqrOverMedianPct = r.median > 0 ? 100.0 * iqr / r.median : 0.0;
  r.unstable = r.iqrOverMedianPct > kIqrFlagPct;
  return r;
}

inline double nowMs() {
  asm volatile("" ::: "memory");
  const double t = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  asm volatile("" ::: "memory");
  return t;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_RSTATS_H
