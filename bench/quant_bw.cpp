//===- quant_bw.cpp - R0.1 quant-kernel bandwidth micro-benchmark ---------===//
//
// Issue #74's exit measurement and the R1 stage-roofline feeder: bandwidth
// of each libreloc/quant kernel (snake_case reporting names from the issue)
// plus the gather_f32 baseline (existing gatherChunk), at a chosen SIMD
// variant and GatherPool thread count, through bench/protocol.h. Every
// timed configuration is verified byte-exact against the serial scalar
// path first (a wrong benchmark is worse than none).
//
// Thread pinning is external by design (the pool does not pin):
//   taskset -c 0-7 ./bench-quant-bw --kernel all --threads 8 ...
// Pin to distinct physical cores; avoid SMT sibling pairs for T <= 4.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::quant::Kernel;
using reloc::quant::Variant;

const char *variantName(Variant v) {
  switch (v) {
  case Variant::Auto:
    return "auto";
  case Variant::Scalar:
    return "scalar";
  case Variant::AVX2:
    return "avx2";
  case Variant::AVX512:
    return "avx512";
  case Variant::AVX512Pf:
    return "avx512pf";
  }
  return "?";
}

bool parseVariant(const std::string &s, Variant &out) {
  for (Variant v : {Variant::Auto, Variant::Scalar, Variant::AVX2,
                    Variant::AVX512, Variant::AVX512Pf})
    if (s == variantName(v)) {
      out = v;
      return true;
    }
  return false;
}

struct Timing {
  bench::Series wall;
  std::vector<double> inGbps, outGbps;
};

template <typename Fn>
Timing timeIt(Fn &&fn, int64_t inBytes, int64_t outBytes, int warmup,
              int iters, int reruns) {
  std::vector<std::vector<double>> per;
  for (int r = 0; r < reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(fn, warmup, iters);
    per.push_back(std::move(s.wall_ms));
  }
  Timing t;
  t.wall = bench::analyzeReruns(per);
  for (const bench::Stats &st : t.wall.reruns) {
    const double sec = st.median * 1e-3;
    t.inGbps.push_back(sec > 0 ? static_cast<double>(inBytes) / sec / 1e9
                               : 0.0);
    t.outGbps.push_back(sec > 0 ? static_cast<double>(outBytes) / sec / 1e9
                                : 0.0);
  }
  return t;
}

std::string vecToJson(const std::vector<double> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += bench::jsonNumber(v[i]);
  }
  return out + "]";
}

std::string entryJson(const std::string &kernel, const std::string &label,
                      int64_t inB, int64_t outB, const Timing &t) {
  return "    \"" + kernel + "\": {\"variant\": \"" + label +
         "\", \"in_bytes\": " + std::to_string(inB) +
         ", \"out_bytes\": " + std::to_string(outB) +
         ", \"wall_ms\": " + bench::seriesToJson(t.wall) +
         ", \"in_gb_per_s\": " + vecToJson(t.inGbps) +
         ", \"out_gb_per_s\": " + vecToJson(t.outGbps) + "}";
}

std::vector<float> makeFloats(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  return v;
}

struct Config {
  int64_t n = 4096;
  unsigned threads = 1;
  Variant variant = Variant::Auto;
  int warmup = bench::kWarmupIters;
  int iters = bench::kTimedIters;
  int reruns = bench::kReruns;
};

// Returns std::nullopt on a correctness-gate failure (caller exits 1).
std::optional<Timing> runQuantizePack(reloc::GatherPool &pool,
                                      const Config &c, int64_t &inB,
                                      int64_t &outB) {
  const int64_t ch = c.n, cs = c.n; // n x n, per-row scale
  std::vector<float> src = makeFloats(static_cast<size_t>(ch * cs));
  std::vector<float> inv(static_cast<size_t>(ch), 1.0f / 127.0f);
  std::vector<int8_t> dst(static_cast<size_t>(ch * cs), 0);
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::quantizePackF32S8(src.data(), ref.data(), ch, cs,
                                    inv.data(), Variant::Scalar);
    reloc::quant::quantizePackF32S8Parallel(pool, src.data(), dst.data(), ch,
                                            cs, inv.data(), c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = ch * cs * 4;
  outB = ch * cs;
  return timeIt(
      [&] {
        reloc::quant::quantizePackF32S8Parallel(pool, src.data(), dst.data(),
                                                ch, cs, inv.data(), c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runConvert(reloc::GatherPool &pool, const Config &c,
                                 int64_t &inB, int64_t &outB) {
  const int64_t n = c.n * c.n;
  std::vector<float> src = makeFloats(static_cast<size_t>(n));
  std::vector<uint16_t> dst(static_cast<size_t>(n), 0);
  {
    std::vector<uint16_t> ref(dst.size(), 1);
    reloc::quant::convertF32F16(src.data(), ref.data(), n, Variant::Scalar);
    reloc::quant::convertF32F16Parallel(pool, src.data(), dst.data(), n,
                                        c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size() * 2) != 0)
      return std::nullopt;
  }
  inB = n * 4;
  outB = n * 2;
  return timeIt(
      [&] {
        reloc::quant::convertF32F16Parallel(pool, src.data(), dst.data(), n,
                                            c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runPack(reloc::GatherPool &pool, const Config &c,
                              int64_t &inB, int64_t &outB) {
  const int64_t pairs = c.n * c.n / 2;
  std::vector<int8_t> src(static_cast<size_t>(2 * pairs));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<int8_t>((i * 37) & 0xff);
  std::vector<uint8_t> dst(static_cast<size_t>(pairs), 0);
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::quant::packS8S4(src.data(), ref.data(), pairs, Variant::Scalar);
    reloc::quant::packS8S4Parallel(pool, src.data(), dst.data(), pairs,
                                   c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = 2 * pairs;
  outB = pairs;
  return timeIt(
      [&] {
        reloc::quant::packS8S4Parallel(pool, src.data(), dst.data(), pairs,
                                       c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

// The two plan-driven kernels share the golden reference plan (gather_bw's
// setup) bound at N = c.n.
struct PlanFixture {
  reloc::BoundPlan bound;
  std::vector<float> src; // fp32 elements, sized by max reachable offset
};

std::optional<PlanFixture> makePlanFixture(int64_t n) {
  std::vector<uint8_t> bytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan)
    return std::nullopt;
  auto boundResult = reloc::bind(*plan, {{"N", n}});
  auto *b = std::get_if<reloc::BoundPlan>(&boundResult);
  if (!b)
    return std::nullopt;
  PlanFixture f;
  f.bound = *b;
  int64_t maxOff = 0;
  for (size_t k = 0; k < f.bound.extents.size(); ++k)
    maxOff += (f.bound.extents[k] - 1) * f.bound.srcStrides[k];
  f.src = makeFloats(static_cast<size_t>(maxOff + 1));
  return f;
}

std::optional<Timing> runGatherQuantize(reloc::GatherPool &pool,
                                        const PlanFixture &f, const Config &c,
                                        int64_t &inB, int64_t &outB) {
  const reloc::BoundPlan &b = f.bound;
  std::vector<int8_t> dst(static_cast<size_t>(b.totalBytes / 4), 0);
  std::vector<float> inv(static_cast<size_t>(b.extents[0]), 1.0f / 127.0f);
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::gatherQuantizeF32S8(b, f.src.data(), ref.data(), inv.data(),
                                      0, b.extents[0], Variant::Scalar);
    reloc::quant::gatherQuantizeF32S8Parallel(pool, b, f.src.data(),
                                              dst.data(), inv.data(),
                                              c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = b.totalBytes; // fp32 read side
  outB = b.totalBytes / 4;
  return timeIt(
      [&] {
        reloc::quant::gatherQuantizeF32S8Parallel(pool, b, f.src.data(),
                                                  dst.data(), inv.data(),
                                                  c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runGatherF32(reloc::GatherPool &pool,
                                   const PlanFixture &f, const Config &c,
                                   int64_t &inB, int64_t &outB) {
  const reloc::BoundPlan &b = f.bound;
  const auto *srcBytes = reinterpret_cast<const uint8_t *>(f.src.data());
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0);
  const int64_t rowBytes = b.dstStrides[0] * 4;
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowBytes));
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::executeH2D(b, srcBytes, ref.data());
    pool.parallelFor(0, b.extents[0], minRows, [&](int64_t rb, int64_t re) {
      reloc::gatherChunk(b, srcBytes, dst.data(), rb, re);
    });
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = b.totalBytes;
  outB = b.totalBytes;
  return timeIt(
      [&] {
        pool.parallelFor(0, b.extents[0], minRows,
                         [&](int64_t rb, int64_t re) {
                           reloc::gatherChunk(b, srcBytes, dst.data(), rb, re);
                         });
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

const char *const kAllKernels[] = {"quantize_pack_f32_s8",
                                   "gather_quantize_f32_s8", "pack_s8_s4",
                                   "convert_f32_f16", "gather_f32"};

std::optional<Kernel> quantKernelFor(const std::string &name) {
  if (name == "quantize_pack_f32_s8")
    return Kernel::QuantizePack;
  if (name == "gather_quantize_f32_s8")
    return Kernel::GatherQuantize;
  if (name == "pack_s8_s4")
    return Kernel::PackS8S4;
  if (name == "convert_f32_f16")
    return Kernel::ConvertF32F16;
  return std::nullopt; // gather_f32: not a quant kernel, variant ignored
}

int run(const std::string &kernelArg, const Config &c, const char *jsonPath) {
  std::vector<std::string> kernels;
  if (kernelArg == "all")
    kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
  else
    kernels.push_back(kernelArg);

  // Validate the variant per requested quant kernel BEFORE any setup.
  for (const std::string &k : kernels) {
    if (auto qk = quantKernelFor(k)) {
      if (!reloc::quant::kernelHasVariant(*qk, c.variant) ||
          !reloc::quant::cpuSupports(c.variant)) {
        std::fprintf(stderr,
                     "error: variant %s not available for kernel %s on this "
                     "host\n",
                     variantName(c.variant), k.c_str());
        return 3;
      }
    }
  }

  std::optional<PlanFixture> plan;
  for (const std::string &k : kernels)
    if (k == "gather_quantize_f32_s8" || k == "gather_f32") {
      plan = makePlanFixture(c.n);
      if (!plan) {
        std::fprintf(stderr, "error: reference plan decode/bind failed\n");
        return 1;
      }
      break;
    }

  reloc::GatherPool pool(c.threads);
  std::string body;
  for (const std::string &k : kernels) {
    int64_t inB = 0, outB = 0;
    std::optional<Timing> t;
    if (k == "quantize_pack_f32_s8")
      t = runQuantizePack(pool, c, inB, outB);
    else if (k == "convert_f32_f16")
      t = runConvert(pool, c, inB, outB);
    else if (k == "pack_s8_s4")
      t = runPack(pool, c, inB, outB);
    else if (k == "gather_quantize_f32_s8")
      t = runGatherQuantize(pool, *plan, c, inB, outB);
    else if (k == "gather_f32")
      t = runGatherF32(pool, *plan, c, inB, outB);
    else {
      std::fprintf(stderr, "error: unknown kernel %s\n", k.c_str());
      return 2;
    }
    if (!t) {
      std::fprintf(stderr, "error: %s mismatch vs serial scalar reference\n",
                   k.c_str());
      return 1;
    }
    auto qk = quantKernelFor(k);
    const std::string label =
        qk ? variantName(reloc::quant::resolveFor(*qk, c.variant)) : "n/a";
    if (!body.empty())
      body += ",\n";
    body += entryJson(k, label, inB, outB, *t);
    std::fprintf(stderr,
                 "quant_bw: %-24s %8s T=%d in %.2f GB/s out %.2f GB/s "
                 "(spread %.2f%%)\n",
                 k.c_str(), label.c_str(), pool.threadCount(),
                 t->inGbps.front(), t->outGbps.front(),
                 t->wall.medianSpreadPct);
  }

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"quant_bw\", \"N\": " +
      std::to_string(c.n) +
      ", \"threads\": " + std::to_string(pool.threadCount()) +
      ", \"variant\": \"" + variantName(c.variant) +
      "\", \"warmup\": " + std::to_string(c.warmup) +
      ", \"iters\": " + std::to_string(c.iters) +
      ", \"reruns\": " + std::to_string(c.reruns) +
      "},\n  \"kernels\": {\n" + body + "\n  }\n}\n";
  pool.close();
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
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string kernel = "all";
  Config c;
  const char *jsonPath = "-";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kernel")
      kernel = next();
    else if (a == "--n")
      c.n = std::atoll(next());
    else if (a == "--threads")
      c.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--variant") {
      if (!parseVariant(next(), c.variant)) {
        std::fprintf(stderr, "error: bad --variant (auto|scalar|avx2|avx512|"
                             "avx512pf)\n");
        return 2;
      }
    } else if (a == "--json")
      jsonPath = next();
    else if (a == "--warmup")
      c.warmup = std::atoi(next());
    else if (a == "--iters")
      c.iters = std::atoi(next());
    else if (a == "--reruns")
      c.reruns = std::atoi(next());
    else {
      std::fprintf(stderr,
                   "usage: bench-quant-bw [--kernel NAME|all] [--n N] "
                   "[--threads T] [--variant V] [--json PATH|-] [--warmup W] "
                   "[--iters I] [--reruns R]\n");
      return 2;
    }
  }
  if (c.n <= 0 || c.n % 64 != 0) {
    std::fprintf(stderr,
                 "error: N must be positive and divisible by 64 (got %lld)\n",
                 static_cast<long long>(c.n));
    return 2;
  }
  if (c.warmup < 0 || c.iters < 1 || c.reruns < 1) {
    std::fprintf(stderr, "error: bad warmup/iters/reruns\n");
    return 2;
  }
  return run(kernel, c, jsonPath);
}
