//===- multigpu_reloc.cu - R3/EXP-3 multi-GPU amortization (#84) ----------===//
//
// Method A (CPU-transform-once, ship int8) vs Method B (ship fp32, GPU
// transform per GPU) across K GPUs, in the two issue-#84 scenarios:
//
//   broadcast: the SAME relocated+quantized tensor to all K GPUs. A does
//     one CPU gather+quant pass (blocked-transpose plan) and DMAs the r*S
//     int8 image to each GPU; B DMAs the full fp32 S to each GPU and runs
//     relocate+quant on-device. Final artifact per GPU: the int8 tensor.
//
//   scatter: one tensor sharded across K GPUs (tensor-parallel weight
//     loading). A does one CPU quant pass over the whole tensor and DMAs
//     each GPU its int8 shard (r*S total); B DMAs each GPU its fp32 shard
//     (S total) and quantizes on-device. Plan is the identity/quant case
//     so shards partition source and destination cleanly. This is gate G5.
//
// Concurrency: single process, one host thread per GPU (own stream + device
// buffers), spin-barrier start -- the methodology validated by M0 (cross-die
// pairs scaled 1.98x, 4-GPU 3.35x; CUDA context serialization would forbid
// both). Baselines: B_xK (K concurrent) and B_staged (serialized DMAs).
//
// Primary metric: aggregate wall clock for the whole delivery INCLUDING
// Method A's single CPU transform (the issue's Method A includes it), plus
// a delivery-only (DMA) reading. Every GPU's final int8 is verified against
// the CPU scalar reference before timing.
//
//===----------------------------------------------------------------------===//

#include "rtrack/plans.h"
#include "rtrack/rstats.h"

#include "reloc/CudaKernels.h"
#include "reloc/GatherPool.h"
#include "reloc/Quant.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err_), #x);                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

using namespace bench::rtrack;

enum class Scenario { Broadcast, Scatter };
enum class Method { A, BxK, BStaged };

const char *scenarioName(Scenario s) {
  return s == Scenario::Broadcast ? "broadcast" : "scatter";
}
const char *methodName(Method m) {
  return m == Method::A ? "a" : (m == Method::BxK ? "bxk" : "bstaged");
}

struct Options {
  Scenario scenario = Scenario::Scatter;
  int64_t n = 8192;
  std::vector<int> ks = {1, 2, 4};
  unsigned threads = 8; // CPU transform pool
  int warmup = 3;
  int iters = 20;
  const char *jsonPath = "-";
};

// Per-GPU device/stream state. Buffers are sized to the GPU's share of the
// scenario (whole tensor for broadcast, a shard for scatter).
struct GpuCtx {
  int dev = -1;
  cudaStream_t stream = nullptr;
  int8_t *dInt8 = nullptr;  // A receive; also B's quantized output
  float *dF32 = nullptr;    // B receive (fp32)
  float *dTmp = nullptr;    // B broadcast: relocated fp32 before quant
  float *dInv = nullptr;    // per-channel invScales (uniform here)
  int64_t elems = 0;        // this GPU's element count
  int64_t hostElemOffset = 0; // offset into the host buffers (scatter)
};

// Shared host state: pinned fp32 source (B's DMA source + A's transform
// input) and pinned int8 image (A's DMA source), plus the CPU reference.
struct HostState {
  reloc::BoundPlan plan;
  int64_t totalElems = 0, rows = 0, channelSize = 0;
  bool contiguousQuant = false; // scatter/identity: use the fast pack kernel
  float *hF32 = nullptr;      // pinned, totalElems
  int8_t *hInt8 = nullptr;    // pinned, totalElems (A's transformed image)
  std::vector<int8_t> ref;    // CPU scalar reference, totalElems
  std::vector<float> invScales;
};

void buildHost(HostState &h, Scenario sc, int64_t n) {
  h.plan = sc == Scenario::Broadcast ? blockedTransposePlan(n) : identityPlan(n);
  h.contiguousQuant = sc == Scenario::Scatter;
  h.totalElems = n * n;
  h.rows = h.plan.extents[0];
  h.channelSize = h.totalElems / h.rows;
  CUDA_CHECK(cudaHostAlloc(&h.hF32, static_cast<size_t>(h.totalElems) * 4,
                           cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&h.hInt8, static_cast<size_t>(h.totalElems),
                           cudaHostAllocDefault));
  for (int64_t i = 0; i < h.totalElems; ++i)
    h.hF32[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  h.invScales.assign(static_cast<size_t>(h.rows), 1.0f / 127.0f);
  // CPU scalar reference: the whole transformed int8 tensor (dst layout).
  h.ref.resize(static_cast<size_t>(h.totalElems));
  reloc::quant::gatherQuantizeF32S8(h.plan, h.hF32, h.ref.data(),
                                    h.invScales.data(), 0, h.rows,
                                    reloc::quant::Variant::Scalar);
}

void freeHost(HostState &h) {
  cudaFreeHost(h.hF32);
  cudaFreeHost(h.hInt8);
}

// Run the CPU transform once into h.hInt8 (Method A's shared pass). Scatter
// is the identity/contiguous case, where quantize_pack (~23 GB/s on this
// AVX2 host) is bit-identical to the plan-walking gather_quantize (proven
// in RtrackTest) but ~7x faster -- using the plan kernel there would
// penalize Method A for no reason. Broadcast is a genuine relocation, so it
// needs the fused gather_quantize.
void cpuTransform(HostState &h, reloc::GatherPool &pool,
                  reloc::quant::Variant variant) {
  if (h.contiguousQuant)
    reloc::quant::quantizePackF32S8Parallel(pool, h.hF32, h.hInt8, h.rows,
                                            h.channelSize, h.invScales.data(),
                                            variant);
  else
    reloc::quant::gatherQuantizeF32S8Parallel(pool, h.plan, h.hF32, h.hInt8,
                                              h.invScales.data(), variant);
}

// GPU transform for Method B: fp32 (src layout) in dF32 -> int8 (dst
// layout) in dInt8, matching the CPU reference. Broadcast relocates first.
void gpuTransform(Scenario sc, GpuCtx &g, const HostState &h) {
  if (sc == Scenario::Broadcast) {
    reloc::cuda::relocateF32(h.plan, g.dF32, g.dTmp, g.stream);
    reloc::cuda::quantizeF32S8(g.dTmp, g.dInt8, h.rows, h.channelSize, g.dInv,
                               g.stream);
  } else {
    // identity plan: quantize contiguously. channels = this shard's rows.
    reloc::cuda::quantizeF32S8(g.dF32, g.dInt8, g.elems / h.channelSize,
                               h.channelSize, g.dInv, g.stream);
  }
}

// Verify a GPU's final int8 against the reference slice; returns false on
// mismatch. `off`/`n` in elements.
bool verifyGpu(const GpuCtx &g, const HostState &h) {
  std::vector<int8_t> got(static_cast<size_t>(g.elems));
  CUDA_CHECK(cudaSetDevice(g.dev));
  CUDA_CHECK(cudaMemcpy(got.data(), g.dInt8, static_cast<size_t>(g.elems),
                        cudaMemcpyDeviceToHost));
  return std::memcmp(got.data(), h.ref.data() + g.hostElemOffset,
                     static_cast<size_t>(g.elems)) == 0;
}

struct RunTimes {
  RStats wall;      // end-to-end incl. CPU transform (Method A)
  double dmaMs = 0; // delivery-only median (DMA aggregate span)
};

// One Method-A iteration: CPU transform once, then K concurrent DMAs of the
// int8 image (whole for broadcast, shard slice for scatter). Returns wall ms.
double iterA(HostState &h, std::vector<GpuCtx> &ctxs, reloc::GatherPool &pool,
             reloc::quant::Variant variant, double &dmaMs) {
  const double t0 = nowMs();
  cpuTransform(h, pool, variant);
  const double tDma = nowMs();
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> ts;
  for (size_t i = 0; i < ctxs.size(); ++i)
    ts.emplace_back([&, i] {
      GpuCtx &g = ctxs[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) {
      }
      CUDA_CHECK(cudaMemcpyAsync(g.dInt8, h.hInt8 + g.hostElemOffset,
                                 static_cast<size_t>(g.elems),
                                 cudaMemcpyHostToDevice, g.stream));
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
    });
  while (ready.load() != static_cast<int>(ctxs.size())) {
  }
  go.store(true, std::memory_order_release);
  for (std::thread &t : ts)
    t.join();
  const double end = nowMs();
  dmaMs = end - tDma;
  return end - t0;
}

// One Method-B iteration. staged=false runs the K GPUs concurrently
// (B_xK); staged=true serializes them (B_staged). Each GPU: DMA fp32 ->
// GPU transform -> int8.
double iterB(Scenario sc, HostState &h, std::vector<GpuCtx> &ctxs,
             bool staged) {
  auto oneGpu = [&](GpuCtx &g) {
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemcpyAsync(g.dF32, h.hF32 + g.hostElemOffset,
                               static_cast<size_t>(g.elems) * 4,
                               cudaMemcpyHostToDevice, g.stream));
    gpuTransform(sc, g, h);
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
  };
  const double t0 = nowMs();
  if (staged) {
    for (GpuCtx &g : ctxs)
      oneGpu(g);
  } else {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    for (size_t i = 0; i < ctxs.size(); ++i)
      ts.emplace_back([&, i] {
        GpuCtx &g = ctxs[i];
        CUDA_CHECK(cudaSetDevice(g.dev));
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) {
        }
        oneGpu(g);
      });
    while (ready.load() != static_cast<int>(ctxs.size())) {
    }
    go.store(true, std::memory_order_release);
    for (std::thread &t : ts)
      t.join();
  }
  return nowMs() - t0;
}

// Allocate device buffers for K GPUs under a scenario. Broadcast: each GPU
// holds the whole tensor; scatter: each GPU holds one shard (rows split K
// ways -> elems split K ways, since dst is packed).
void allocCtxs(std::vector<GpuCtx> &ctxs, Scenario sc, const HostState &h,
               int K, int nDev) {
  ctxs.resize(static_cast<size_t>(K));
  const int64_t shardElems = h.totalElems / K; // K divides rows -> divides elems
  for (int k = 0; k < K; ++k) {
    GpuCtx &g = ctxs[static_cast<size_t>(k)];
    g.dev = k % nDev;
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    if (sc == Scenario::Broadcast) {
      g.elems = h.totalElems;
      g.hostElemOffset = 0;
    } else {
      g.elems = shardElems;
      g.hostElemOffset = static_cast<int64_t>(k) * shardElems;
    }
    void *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(g.elems)));
    g.dInt8 = static_cast<int8_t *>(p);
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(g.elems) * 4));
    g.dF32 = static_cast<float *>(p);
    if (sc == Scenario::Broadcast) {
      CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(g.elems) * 4));
      g.dTmp = static_cast<float *>(p);
    }
    const int64_t channels =
        sc == Scenario::Broadcast ? h.rows : g.elems / h.channelSize;
    std::vector<float> inv(static_cast<size_t>(channels), 1.0f / 127.0f);
    CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(channels) * 4));
    g.dInv = static_cast<float *>(p);
    CUDA_CHECK(cudaMemcpy(g.dInv, inv.data(),
                          static_cast<size_t>(channels) * 4,
                          cudaMemcpyHostToDevice));
  }
}

void freeCtxs(std::vector<GpuCtx> &ctxs) {
  for (GpuCtx &g : ctxs) {
    cudaSetDevice(g.dev);
    cudaFree(g.dInt8);
    cudaFree(g.dF32);
    cudaFree(g.dTmp);
    cudaFree(g.dInv);
    cudaStreamDestroy(g.stream);
  }
  ctxs.clear();
}

std::string entryJson(Scenario sc, Method m, int K, int64_t n,
                     const HostState &h, const RunTimes &t, double speedup) {
  const int64_t S = h.totalElems * 4;
  // Logical input delivered: one tensor (both scenarios move one tensor's
  // worth of logical data; broadcast replicates it K times physically).
  const double effGbps = t.wall.median > 0
                             ? static_cast<double>(S) /
                                   (t.wall.median * 1e-3) / 1e9
                             : 0.0;
  std::string j =
      "    {\"scenario\": \"" + std::string(scenarioName(sc)) +
      "\", \"method\": \"" + methodName(m) + "\", \"K\": " +
      std::to_string(K) + ", \"N\": " + std::to_string(n) +
      ", \"wall_median_ms\": " + bench::jsonNumber(t.wall.median) +
      ", \"wall_min_ms\": " + bench::jsonNumber(t.wall.min) +
      ", \"wall_p95_ms\": " + bench::jsonNumber(t.wall.p95) +
      ", \"iqr_over_median_pct\": " +
      bench::jsonNumber(t.wall.iqrOverMedianPct) +
      ", \"dma_ms\": " + bench::jsonNumber(t.dmaMs) +
      ", \"eff_input_gb_per_s\": " + bench::jsonNumber(effGbps);
  if (speedup >= 0)
    j += ", \"speedup_vs_bxk\": " + bench::jsonNumber(speedup);
  j += "}";
  return j;
}

int run(const Options &opt) {
  int nDev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&nDev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  HostState h;
  buildHost(h, opt.scenario, opt.n);
  reloc::GatherPool pool(opt.threads);
  const reloc::quant::Variant variant = reloc::quant::Variant::Auto;

  std::string rows;
  auto emit = [&](const std::string &j) {
    if (!rows.empty())
      rows += ",\n";
    rows += j;
  };

  for (int K : opt.ks) {
    if (K < 1 || h.rows % K != 0) {
      std::fprintf(stderr, "skip K=%d (rows %lld not divisible)\n", K,
                   static_cast<long long>(h.rows));
      continue;
    }
    std::vector<GpuCtx> ctxs;
    allocCtxs(ctxs, opt.scenario, h, K, nDev);

    // --- verify every method before timing --------------------------------
    double dma = 0;
    (void)iterA(h, ctxs, pool, variant, dma);
    for (const GpuCtx &g : ctxs)
      if (!verifyGpu(g, h)) {
        std::fprintf(stderr, "VERIFY FAILED: A %s K=%d\n",
                     scenarioName(opt.scenario), K);
        return 1;
      }
    for (bool staged : {false, true}) {
      for (GpuCtx &g : ctxs) {
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaMemset(g.dInt8, 0xAB, static_cast<size_t>(g.elems)));
      }
      (void)iterB(opt.scenario, h, ctxs, staged);
      for (const GpuCtx &g : ctxs)
        if (!verifyGpu(g, h)) {
          std::fprintf(stderr, "VERIFY FAILED: B%s %s K=%d\n",
                       staged ? "_staged" : "_xK",
                       scenarioName(opt.scenario), K);
          return 1;
        }
    }

    // --- time A, B_xK, B_staged -------------------------------------------
    auto timeIt = [&](Method m) {
      std::vector<double> wall, dmas;
      auto once = [&]() -> double {
        double d = 0;
        double w;
        if (m == Method::A)
          w = iterA(h, ctxs, pool, variant, d);
        else
          w = iterB(opt.scenario, h, ctxs, m == Method::BStaged);
        dmas.push_back(d);
        return w;
      };
      for (int i = 0; i < opt.warmup; ++i)
        (void)once();
      for (int i = 0; i < opt.iters; ++i)
        wall.push_back(once());
      RunTimes rt;
      rt.wall = summarizeSamples(wall);
      rt.dmaMs = summarizeSamples(dmas).median;
      return rt;
    };

    RunTimes ta = timeIt(Method::A);
    RunTimes tbk = timeIt(Method::BxK);
    RunTimes tbs = timeIt(Method::BStaged);
    const double sa = tbk.wall.median > 0 ? tbk.wall.median / ta.wall.median
                                          : 0.0;
    emit(entryJson(opt.scenario, Method::A, K, opt.n, h, ta, sa));
    emit(entryJson(opt.scenario, Method::BxK, K, opt.n, h, tbk, -1));
    emit(entryJson(opt.scenario, Method::BStaged, K, opt.n, h, tbs, -1));
    std::fprintf(stderr,
                 "%s K=%d: A %7.2f ms | BxK %7.2f ms | Bstaged %7.2f ms | "
                 "A/BxK %.2fx%s\n",
                 scenarioName(opt.scenario), K, ta.wall.median,
                 tbk.wall.median, tbs.wall.median, sa,
                 (opt.scenario == Scenario::Scatter && K == 4)
                     ? (sa >= 1.3 ? "  [G5 PASS]" : "  [G5 fail]")
                     : "");
    freeCtxs(ctxs);
  }
  pool.close();

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"multigpu_reloc\", \"gpu\": \"" +
      std::string(prop.name) + "\", \"scenario\": \"" +
      scenarioName(opt.scenario) + "\", \"N\": " + std::to_string(opt.n) +
      ", \"cpu_threads\": " + std::to_string(opt.threads) +
      ", \"warmup\": " + std::to_string(opt.warmup) +
      ", \"iters\": " + std::to_string(opt.iters) +
      "},\n  \"rows\": [\n" + rows + "\n  ]\n}\n";
  freeHost(h);
  if (std::strcmp(opt.jsonPath, "-") == 0) {
    std::fputs(doc.c_str(), stdout);
  } else {
    std::FILE *f = std::fopen(opt.jsonPath, "w");
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", opt.jsonPath);
      return 1;
    }
    std::fputs(doc.c_str(), f);
    std::fclose(f);
  }
  return 0;
}

std::vector<int> parseInts(const std::string &s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t next = s.find(',', pos);
    if (next == std::string::npos)
      next = s.size();
    if (next > pos)
      out.push_back(std::atoi(s.substr(pos, next - pos).c_str()));
    pos = next + 1;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--scenario") {
      std::string s = next();
      if (s == "broadcast")
        opt.scenario = Scenario::Broadcast;
      else if (s == "scatter")
        opt.scenario = Scenario::Scatter;
      else {
        std::fprintf(stderr, "error: --scenario broadcast|scatter\n");
        return 2;
      }
    } else if (a == "--n")
      opt.n = std::atoll(next());
    else if (a == "--k")
      opt.ks = parseInts(next());
    else if (a == "--threads")
      opt.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--json")
      opt.jsonPath = next();
    else {
      std::fprintf(stderr,
                   "usage: bench-multigpu-reloc [--scenario broadcast|scatter] "
                   "[--n N] [--k 1,2,4] [--threads T] [--warmup W] "
                   "[--iters I] [--json PATH|-]\n");
      return 2;
    }
  }
  if (opt.n <= 0 || opt.n % 64 != 0) {
    std::fprintf(stderr, "error: N must be positive and divisible by 64\n");
    return 2;
  }
  if (opt.warmup < 0 || opt.iters < 1)
    return 2;
  return run(opt);
}
