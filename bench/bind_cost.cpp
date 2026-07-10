//===- bind_cost.cpp - #C3 bind-cost micro-benchmark ----------------------===//
//
// The protocol-reuse proof required by issue #47: measures reloc::bind()
// latency for the golden reference plan through bench/protocol.h, emitting
// the same JSON shape as the PoC driver. CPU-only; runs as a ctest smoke
// so CI proves the protocol header generalizes.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/Decode.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace {

// Keep the optimizer from deleting the bind() under test.
volatile int64_t gSink = 0;

int run(int64_t n, const char *jsonPath, int warmup, int iters, int reruns) {
  std::vector<uint8_t> bytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan) {
    std::fprintf(stderr, "error: golden reference plan failed to decode\n");
    return 1;
  }

  std::vector<std::vector<double>> wallPerRerun;
  for (int r = 0; r < reruns; ++r) {
    bench::RerunSamples samples = bench::runOnce(
        [&] {
          auto bound = reloc::bind(*plan, {{"N", n}});
          auto *bp = std::get_if<reloc::BoundPlan>(&bound);
          if (!bp)
            std::abort(); // divisibility holds for the chosen N by design
          gSink += bp->totalBytes;
        },
        warmup, iters);
    wallPerRerun.push_back(std::move(samples.wall_ms));
  }
  bench::Series wall = bench::analyzeReruns(wallPerRerun);

  std::string doc = "{\n  \"config\": {\"benchmark\": \"bind_cost\", "
                    "\"plan\": \"reference\", \"N\": " +
                    std::to_string(n) +
                    ", \"warmup\": " + std::to_string(warmup) +
                    ", \"iters\": " + std::to_string(iters) +
                    ", \"reruns\": " + std::to_string(reruns) +
                    "},\n  \"methods\": {\"bind_reference_plan\": "
                    "{\"wall_ms\": " +
                    bench::seriesToJson(wall) + "}}\n}\n";
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
               "bind_cost: N=%lld median %.4f ms (rerun spread %.2f%%)\n",
               static_cast<long long>(n), wall.reruns.front().median,
               wall.medianSpreadPct);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  int64_t n = 4096;
  const char *jsonPath = "-";
  int warmup = bench::kWarmupIters, iters = bench::kTimedIters,
      reruns = bench::kReruns;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--n")
      n = std::atoll(next());
    else if (a == "--json")
      jsonPath = next();
    else if (a == "--warmup")
      warmup = std::atoi(next());
    else if (a == "--iters")
      iters = std::atoi(next());
    else if (a == "--reruns")
      reruns = std::atoi(next());
    else {
      std::fprintf(stderr, "usage: bench-bind-cost [--n N] [--json PATH|-] "
                           "[--warmup W] [--iters I] [--reruns R]\n");
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
  return run(n, jsonPath, warmup, iters, reruns);
}
