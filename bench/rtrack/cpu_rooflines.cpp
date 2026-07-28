//===- cpu_rooflines.cpp - R1 stage rooflines on the rtrack plans ---------===//
//
// Issue #82's independent stage measurements, CPU side: strided gather BW,
// fused gather+quantize BW, contiguous quantize/convert BW, and plain
// contiguous read BW, at a chosen GatherPool thread count. Kernels are
// timed in isolation (heap destination, no pipeline) so these are the
// roofline inputs for P3b and the D-track bottleneck-flip check.
//
// This intentionally does NOT reuse bench/quant_bw.cpp's plan fixture:
// that driver decodes the frozen golden blob, which on main still encodes
// the pre-#63-fix non-injective pattern. Plans here come from
// rtrack/plans.h (oracle-verified in RtrackTest). Every timed config is
// verified byte-exact against the serial scalar path first.
//
// Thread pinning is external (taskset); see docs/m0-2080ti-bringup.md.
//
//===----------------------------------------------------------------------===//

#include "rtrack/plans.h"
#include "rtrack/rstats.h"

#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace bench::rtrack;

struct Config {
  int64_t n = 8192;
  unsigned threads = 1;
  reloc::quant::Variant variant = reloc::quant::Variant::Auto;
  int warmup = kWarmup;
  int iters = kIters;
};

struct Result {
  RStats wall;
  double inGbps = 0, outGbps = 0;
};

template <typename Fn>
Result timeIt(Fn &&fn, int64_t inBytes, int64_t outBytes, const Config &c) {
  std::vector<double> wall;
  wall.reserve(static_cast<size_t>(c.iters));
  for (int i = 0; i < c.warmup; ++i)
    fn();
  for (int i = 0; i < c.iters; ++i) {
    const double t0 = nowMs();
    fn();
    wall.push_back(nowMs() - t0);
  }
  Result r;
  r.wall = summarizeSamples(wall);
  const double sec = r.wall.median * 1e-3;
  r.inGbps = sec > 0 ? static_cast<double>(inBytes) / sec / 1e9 : 0;
  r.outGbps = sec > 0 ? static_cast<double>(outBytes) / sec / 1e9 : 0;
  return r;
}

std::vector<float> makeFloats(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  return v;
}

reloc::BoundPlan planByName(const std::string &name, int64_t n, bool &ok) {
  ok = true;
  if (name == "transpose")
    return transposePlan(n);
  if (name == "blocked")
    return blockedTransposePlan(n);
  if (name == "nchw")
    return nchwToNhwcPlan(n);
  if (name == "identity")
    return identityPlan(n);
  ok = false;
  return {};
}

int64_t minRowsFor(const reloc::BoundPlan &b) {
  const int64_t rowSrcBytes = b.dstStrides[0] * 4;
  return std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowSrcBytes));
}

// gather_f32: strided fp32 gather over the plan into a heap dst.
std::optional<Result> runGatherF32(reloc::GatherPool &pool,
                                   const reloc::BoundPlan &b,
                                   const std::vector<float> &src,
                                   const Config &c, int64_t &inB,
                                   int64_t &outB) {
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0);
  const int64_t minRows = minRowsFor(b);
  auto par = [&] {
    pool.parallelFor(0, b.extents[0], minRows, [&](int64_t rb, int64_t re) {
      reloc::gatherChunk(b, src.data(), dst.data(), rb, re);
    });
  };
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::executeH2D(b, src.data(), ref.data());
    par();
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = outB = b.totalBytes;
  return timeIt(par, inB, outB, c);
}

// gather_quantize_f32_s8 over the plan into a heap int8 dst image.
std::optional<Result> runGatherQuantize(reloc::GatherPool &pool,
                                        const reloc::BoundPlan &b,
                                        const std::vector<float> &src,
                                        const Config &c, int64_t &inB,
                                        int64_t &outB) {
  const int64_t rows = b.extents[0];
  std::vector<float> inv(static_cast<size_t>(rows), 1.0f / 127.0f);
  std::vector<int8_t> dst(static_cast<size_t>(b.totalBytes / 4), 0);
  const int64_t minRows = minRowsFor(b);
  auto par = [&] {
    pool.parallelFor(0, rows, minRows, [&](int64_t rb, int64_t re) {
      reloc::quant::gatherQuantizeF32S8(b, src.data(), dst.data(), inv.data(),
                                        rb, re, c.variant);
    });
  };
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::gatherQuantizeF32S8(b, src.data(), ref.data(), inv.data(), 0,
                                      rows, reloc::quant::Variant::Scalar);
    par();
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = b.totalBytes;
  outB = b.totalBytes / 4;
  return timeIt(par, inB, outB, c);
}

// quantize_pack_f32_s8, contiguous.
std::optional<Result> runQuantizePack(reloc::GatherPool &pool, const Config &c,
                                      const std::vector<float> &src,
                                      int64_t &inB, int64_t &outB) {
  const int64_t ch = c.n, cs = c.n;
  std::vector<float> inv(static_cast<size_t>(ch), 1.0f / 127.0f);
  std::vector<int8_t> dst(static_cast<size_t>(ch * cs), 0);
  auto par = [&] {
    reloc::quant::quantizePackF32S8Parallel(pool, src.data(), dst.data(), ch,
                                            cs, inv.data(), c.variant);
  };
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::quantizePackF32S8(src.data(), ref.data(), ch, cs, inv.data(),
                                    reloc::quant::Variant::Scalar);
    par();
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = ch * cs * 4;
  outB = ch * cs;
  return timeIt(par, inB, outB, c);
}

// convert_f32_f16, contiguous.
std::optional<Result> runConvert(reloc::GatherPool &pool, const Config &c,
                                 const std::vector<float> &src, int64_t &inB,
                                 int64_t &outB) {
  const int64_t total = c.n * c.n;
  std::vector<uint16_t> dst(static_cast<size_t>(total), 0);
  auto par = [&] {
    reloc::quant::convertF32F16Parallel(pool, src.data(), dst.data(), total,
                                        c.variant);
  };
  {
    std::vector<uint16_t> ref(dst.size(), 1);
    reloc::quant::convertF32F16(src.data(), ref.data(), total,
                                reloc::quant::Variant::Scalar);
    par();
    if (std::memcmp(ref.data(), dst.data(), dst.size() * 2) != 0)
      return std::nullopt;
  }
  inB = total * 4;
  outB = total * 2;
  return timeIt(par, inB, outB, c);
}

// contig_read: pure contiguous fp32 read (4 independent accumulators per
// range for ILP). Each range's partial sum is keyed by its begin index,
// so the parallel result is deterministic and the verify step can replay
// the identical ranges serially and compare bit-for-bit.
std::optional<Result> runContigRead(reloc::GatherPool &pool, const Config &c,
                                    const std::vector<float> &src, int64_t &inB,
                                    int64_t &outB) {
  const int64_t total = c.n * c.n;
  const int64_t minElems =
      static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) / 4;
  auto rangeSum = [&](int64_t rb, int64_t re) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int64_t i = rb;
    for (; i + 4 <= re; i += 4) {
      a0 += src[static_cast<size_t>(i)];
      a1 += src[static_cast<size_t>(i + 1)];
      a2 += src[static_cast<size_t>(i + 2)];
      a3 += src[static_cast<size_t>(i + 3)];
    }
    for (; i < re; ++i)
      a0 += src[static_cast<size_t>(i)];
    return a0 + a1 + a2 + a3;
  };
  std::mutex mu;
  std::map<int64_t, std::pair<int64_t, float>> parts; // rb -> (re, sum)
  auto par = [&] {
    pool.parallelFor(0, total, minElems, [&](int64_t rb, int64_t re) {
      float s = rangeSum(rb, re);
      std::lock_guard<std::mutex> lock(mu);
      parts[rb] = {re, s};
    });
  };
  {
    parts.clear();
    par();
    int64_t covered = 0;
    for (const auto &kv : parts) {
      if (rangeSum(kv.first, kv.second.first) != kv.second.second)
        return std::nullopt;
      covered += kv.second.first - kv.first;
    }
    if (covered != total)
      return std::nullopt; // partition did not cover the tensor
  }
  inB = total * 4;
  outB = 0;
  return timeIt(par, inB, outB, c);
}

// pack_s8_s4: the r=0.125 sweep's second pass, contiguous s8 -> nibbles.
std::optional<Result> runPackS8S4(reloc::GatherPool &pool, const Config &c,
                                  int64_t &inB, int64_t &outB) {
  const int64_t total = c.n * c.n;
  std::vector<int8_t> src(static_cast<size_t>(total));
  for (int64_t i = 0; i < total; ++i)
    src[static_cast<size_t>(i)] =
        static_cast<int8_t>(static_cast<int>((i * 7) & 0xF) - 8);
  std::vector<uint8_t> dst(static_cast<size_t>(total / 2), 0);
  auto par = [&] {
    reloc::quant::packS8S4Parallel(pool, src.data(), dst.data(), total / 2,
                                   c.variant);
  };
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::quant::packS8S4(src.data(), ref.data(), total / 2,
                           reloc::quant::Variant::Scalar);
    par();
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = total;
  outB = total / 2;
  return timeIt(par, inB, outB, c);
}

const char *const kAll[] = {"gather_f32",    "gather_quantize",
                            "quantize_pack", "convert_f32_f16",
                            "contig_read",   "pack_s8_s4"};

int run(const std::string &kernelArg, const std::string &planName,
        const Config &c, const char *jsonPath) {
  std::vector<std::string> kernels;
  if (kernelArg == "all")
    kernels.assign(std::begin(kAll), std::end(kAll));
  else
    kernels.push_back(kernelArg);

  bool ok = true;
  reloc::BoundPlan plan = planByName(planName, c.n, ok);
  if (!ok) {
    std::fprintf(stderr,
                 "error: bad --plan (transpose|blocked|nchw|identity)\n");
    return 2;
  }
  std::vector<float> src = makeFloats(static_cast<size_t>(c.n * c.n));
  reloc::GatherPool pool(c.threads);

  std::string body;
  for (const std::string &k : kernels) {
    int64_t inB = 0, outB = 0;
    std::optional<Result> r;
    if (k == "gather_f32")
      r = runGatherF32(pool, plan, src, c, inB, outB);
    else if (k == "gather_quantize")
      r = runGatherQuantize(pool, plan, src, c, inB, outB);
    else if (k == "quantize_pack")
      r = runQuantizePack(pool, c, src, inB, outB);
    else if (k == "convert_f32_f16")
      r = runConvert(pool, c, src, inB, outB);
    else if (k == "contig_read")
      r = runContigRead(pool, c, src, inB, outB);
    else if (k == "pack_s8_s4")
      r = runPackS8S4(pool, c, inB, outB);
    else {
      std::fprintf(stderr, "error: unknown kernel %s\n", k.c_str());
      return 2;
    }
    if (!r) {
      std::fprintf(stderr, "error: %s mismatch vs serial reference\n",
                   k.c_str());
      return 1;
    }
    if (!body.empty())
      body += ",\n";
    body += "    \"" + k + "\": {\"in_bytes\": " + std::to_string(inB) +
            ", \"out_bytes\": " + std::to_string(outB) +
            ", \"median_ms\": " + bench::jsonNumber(r->wall.median) +
            ", \"min_ms\": " + bench::jsonNumber(r->wall.min) +
            ", \"p95_ms\": " + bench::jsonNumber(r->wall.p95) +
            ", \"iqr_over_median_pct\": " +
            bench::jsonNumber(r->wall.iqrOverMedianPct) +
            ", \"in_gb_per_s\": " + bench::jsonNumber(r->inGbps) +
            ", \"out_gb_per_s\": " + bench::jsonNumber(r->outGbps) + "}";
    std::fprintf(stderr,
                 "cpu_rooflines: %-16s plan=%-9s T=%d in %6.2f GB/s "
                 "(iqr %4.1f%%)\n",
                 k.c_str(), planName.c_str(), pool.threadCount(), r->inGbps,
                 r->wall.iqrOverMedianPct);
  }

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"cpu_rooflines\", \"N\": " +
      std::to_string(c.n) + ", \"plan\": \"" + planName +
      "\", \"threads\": " + std::to_string(pool.threadCount()) +
      ", \"warmup\": " + std::to_string(c.warmup) +
      ", \"iters\": " + std::to_string(c.iters) + "},\n  \"kernels\": {\n" +
      body + "\n  }\n}\n";
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
  std::string kernel = "all", planName = "blocked";
  Config c;
  const char *jsonPath = "-";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kernel")
      kernel = next();
    else if (a == "--plan")
      planName = next();
    else if (a == "--n")
      c.n = std::atoll(next());
    else if (a == "--threads")
      c.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--warmup")
      c.warmup = std::atoi(next());
    else if (a == "--iters")
      c.iters = std::atoi(next());
    else if (a == "--json")
      jsonPath = next();
    else {
      std::fprintf(stderr,
                   "usage: bench-cpu-rooflines [--kernel NAME|all] "
                   "[--plan transpose|blocked|nchw|identity] [--n N] "
                   "[--threads T] [--warmup W] [--iters I] [--json PATH|-]\n");
      return 2;
    }
  }
  if (c.n <= 0 || c.n % 64 != 0 || c.warmup < 0 || c.iters < 1) {
    std::fprintf(stderr, "error: bad --n/--warmup/--iters\n");
    return 2;
  }
  return run(kernel, planName, c, jsonPath);
}
