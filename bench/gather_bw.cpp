//===- gather_bw.cpp - D1 gather-bandwidth micro-benchmark ----------------===//
//
// Issue #65's benchmark (and E2's seed data point): gatherChunk bandwidth on
// the golden reference plan, single-thread vs a GatherPool of T workers,
// through bench/protocol.h. Pure CPU gather into a malloc'd dst-layout
// buffer -- no CopyBackend -- so the number isolates exactly the primitive
// the pipeline parallelizes, dispatched with the same per-worker byte floor
// the pipeline applies.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace {

// The pipeline's per-worker floor (Pipeline.h) in rows of this plan, so the
// bench measures the exact dispatch executeH2DPipelined performs per chunk.
int64_t minRowsPerWorker(const reloc::BoundPlan &b) {
  int64_t rowBytes = b.dstStrides[0] * static_cast<int64_t>(b.elementSize);
  return std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowBytes));
}

struct Measurement {
  bench::Series wall;
  std::vector<double> gbPerS; // per rerun, from that rerun's median
};

Measurement measureGather(const reloc::BoundPlan &b, const uint8_t *src,
                          uint8_t *dst, unsigned threads, int warmup,
                          int iters, int reruns) {
  reloc::GatherPool pool(threads);
  const int64_t outer = b.extents[0];
  const int64_t minRows = minRowsPerWorker(b);
  std::vector<std::vector<double>> wallPerRerun;
  for (int r = 0; r < reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(
        [&] {
          pool.parallelFor(0, outer, minRows, [&](int64_t rb, int64_t re) {
            reloc::gatherChunk(b, src, dst, rb, re);
          });
        },
        warmup, iters);
    wallPerRerun.push_back(std::move(s.wall_ms));
  }
  Measurement m;
  m.wall = bench::analyzeReruns(wallPerRerun);
  for (const bench::Stats &st : m.wall.reruns)
    m.gbPerS.push_back(st.median > 0
                           ? static_cast<double>(b.totalBytes) /
                                 (st.median * 1e-3) / 1e9
                           : 0.0);
  return m;
}

std::string gbToJson(const std::vector<double> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += bench::jsonNumber(v[i]);
  }
  return out + "]";
}

int run(int64_t n, unsigned threads, const char *jsonPath, int warmup,
        int iters, int reruns) {
  std::vector<uint8_t> bytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan) {
    std::fprintf(stderr, "error: golden reference plan failed to decode\n");
    return 1;
  }
  auto boundResult = reloc::bind(*plan, {{"N", n}});
  auto *b = std::get_if<reloc::BoundPlan>(&boundResult);
  if (!b) {
    std::fprintf(stderr, "error: bind failed for N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }

  // Source sized by the max element offset reachable via srcStrides.
  int64_t maxOff = 0;
  for (size_t k = 0; k < b->extents.size(); ++k)
    maxOff += (b->extents[k] - 1) * b->srcStrides[k];
  std::vector<uint8_t> src(static_cast<size_t>(maxOff + 1) * b->elementSize);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>((i * 131) & 0xff);
  std::vector<uint8_t> dst(static_cast<size_t>(b->totalBytes), 0);

  // Correctness gate before timing: the pool-dispatched gather must match
  // executeH2D byte-for-byte (a wrong benchmark is worse than none).
  {
    std::vector<uint8_t> ref(static_cast<size_t>(b->totalBytes), 0);
    reloc::executeH2D(*b, src.data(), ref.data());
    reloc::GatherPool pool(threads);
    pool.parallelFor(0, b->extents[0], minRowsPerWorker(*b),
                     [&](int64_t rb, int64_t re) {
                       reloc::gatherChunk(*b, src.data(), dst.data(), rb, re);
                     });
    if (std::memcmp(ref.data(), dst.data(), ref.size()) != 0) {
      std::fprintf(stderr, "error: parallel gather mismatch vs executeH2D\n");
      return 1;
    }
  }

  Measurement single =
      measureGather(*b, src.data(), dst.data(), 1, warmup, iters, reruns);
  reloc::GatherPool probe(threads); // resolve 0 -> hw for reporting
  const int threadsResolved = probe.threadCount();
  probe.close();
  Measurement multi = measureGather(*b, src.data(), dst.data(), threads,
                                    warmup, iters, reruns);

  std::string doc =
      "{\n  \"config\": {\"benchmark\": \"gather_bw\", \"plan\": "
      "\"reference\", \"N\": " +
      std::to_string(n) +
      ", \"total_bytes\": " + std::to_string(b->totalBytes) +
      ", \"threads_multi\": " + std::to_string(threadsResolved) +
      ", \"min_rows_per_worker\": " + std::to_string(minRowsPerWorker(*b)) +
      ", \"warmup\": " + std::to_string(warmup) +
      ", \"iters\": " + std::to_string(iters) +
      ", \"reruns\": " + std::to_string(reruns) +
      "},\n  \"methods\": {\n    \"gather_1thread\": {\"wall_ms\": " +
      bench::seriesToJson(single.wall) +
      ", \"gb_per_s\": " + gbToJson(single.gbPerS) +
      "},\n    \"gather_multithread\": {\"wall_ms\": " +
      bench::seriesToJson(multi.wall) +
      ", \"gb_per_s\": " + gbToJson(multi.gbPerS) + "}\n  }\n}\n";
  if (std::strcmp(jsonPath, "-") == 0) {
    std::fputs(doc.c_str(), stdout);
  } else {
    std::FILE *f = std::fopen(jsonPath, "w");
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", jsonPath);
      return 1;
    }
    std::fputs(doc.c_str(), f);
    std::fclose(f);
  }
  std::fprintf(stderr,
               "gather_bw: N=%lld 1T %.2f GB/s, %dT %.2f GB/s (%.2fx), "
               "rerun spread %.2f%% / %.2f%%\n",
               static_cast<long long>(n), single.gbPerS.front(),
               threadsResolved, multi.gbPerS.front(),
               multi.gbPerS.front() / single.gbPerS.front(),
               single.wall.medianSpreadPct, multi.wall.medianSpreadPct);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  int64_t n = 4096;
  unsigned threads = 0; // 0 = hardware concurrency
  const char *jsonPath = "-";
  int warmup = bench::kWarmupIters, iters = bench::kTimedIters,
      reruns = bench::kReruns;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--n")
      n = std::atoll(next());
    else if (a == "--threads")
      threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--json")
      jsonPath = next();
    else if (a == "--warmup")
      warmup = std::atoi(next());
    else if (a == "--iters")
      iters = std::atoi(next());
    else if (a == "--reruns")
      reruns = std::atoi(next());
    else {
      std::fprintf(stderr,
                   "usage: bench-gather-bw [--n N] [--threads T] "
                   "[--json PATH|-] [--warmup W] [--iters I] [--reruns R]\n");
      return 2;
    }
  }
  if (n <= 0 || n % 64 != 0) {
    std::fprintf(stderr,
                 "error: N must be positive and divisible by 64 (got %lld)\n",
                 static_cast<long long>(n));
    return 2;
  }
  if (warmup < 0 || iters < 1 || reruns < 1) {
    std::fprintf(stderr,
                 "error: warmup must be >= 0 and iters/reruns must be >= 1 "
                 "(got warmup=%d, iters=%d, reruns=%d)\n",
                 warmup, iters, reruns);
    return 2;
  }
  return run(n, threads, jsonPath, warmup, iters, reruns);
}
