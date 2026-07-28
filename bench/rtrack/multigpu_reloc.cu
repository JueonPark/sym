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

#include "reloc/CudaBackend.h"
#include "reloc/CudaKernels.h"
#include "reloc/GatherPool.h"
#include "reloc/Prefold.h"
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

// broadcast_contig (issue #99): same fan-out placement as broadcast (every
// GPU receives the whole tensor) but with the identity plan, so Method A's
// CPU stage is quantize-only (no strided gather). Separates "broadcast is
// CPU-gather-bound" from "broadcast is fan-out-bound".
enum class Scenario { Broadcast, Scatter, BroadcastContig };
enum class Method { A, APrefold, BxK, BStaged };

const char *scenarioName(Scenario s) {
  switch (s) {
  case Scenario::Broadcast:
    return "broadcast";
  case Scenario::Scatter:
    return "scatter";
  case Scenario::BroadcastContig:
    return "broadcast_contig";
  }
  return "?";
}

// Placement: both broadcast variants ship the whole tensor to every GPU.
bool broadcastPlacement(Scenario s) { return s != Scenario::Scatter; }
const char *methodName(Method m) {
  switch (m) {
  case Method::A:
    return "a";
  case Method::APrefold:
    return "aprefold";
  case Method::BxK:
    return "bxk";
  case Method::BStaged:
    return "bstaged";
  }
  return "?";
}

struct Options {
  Scenario scenario = Scenario::Scatter;
  int64_t n = 8192;
  std::vector<int> ks = {1, 2, 4};
  unsigned threads = 8; // CPU transform pool
  int warmup = 3;
  int iters = 20;
  const char *jsonPath = "-";
  std::vector<int> reuse;  // --reuse: amortization sweep points (empty = skip)
  bool streaming = false;  // --streaming: re-fold-per-load counter-case
  std::vector<int> devices; // --devices: GPU ordinals used round-robin
                            // (empty = 0..nDev-1, the pre-#99 behavior).
                            // Lets K=2 target the uncontended cross-die
                            // pair {2,3} instead of the shared-root {0,1}.
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
  h.contiguousQuant = sc != Scenario::Broadcast;
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

// P4 (issue #98): fold once through the library component. The artifact
// lands in CudaBackend staging = pinned memory, so aprefold's DMAs read
// it directly. Cold cost (allocStaging + fold) is timed by the caller.
reloc::prefold::PrefoldArtifact foldOnce(HostState &h,
                                         reloc::CudaBackend &backend,
                                         reloc::GatherPool &pool,
                                         reloc::quant::Variant variant) {
  const auto spec = h.contiguousQuant
                        ? reloc::prefold::OutputSpec::S8QuantPack
                        : reloc::prefold::OutputSpec::S8GatherQuant;
  return reloc::prefold::prefoldArtifact(h.plan, h.hF32, spec,
                                         h.invScales.data(), backend, pool,
                                         variant);
}

// Touch one value per channel so a re-fold is genuinely required (the
// invScales here are constant 1/127, so mutation never invalidates them).
void mutateSource(HostState &h, int iter) {
  for (int64_t c = 0; c < h.rows; ++c)
    h.hF32[c * h.channelSize] =
        (static_cast<float>(((c + iter) * 131) & 0xff) - 128.0f) * 0.9f;
}

// GPU transform for Method B: fp32 (src layout) in dF32 -> int8 (dst
// layout) in dInt8, matching the CPU reference. Broadcast relocates first.
void gpuTransform(Scenario sc, GpuCtx &g, const HostState &h) {
  if (sc == Scenario::Broadcast) {
    reloc::cuda::relocateF32(h.plan, g.dF32, g.dTmp, g.stream);
    reloc::cuda::quantizeF32S8(g.dTmp, g.dInt8, h.rows, h.channelSize, g.dInv,
                               g.stream);
  } else {
    // identity plan (scatter shard or broadcast_contig whole tensor):
    // quantize contiguously. channels = this GPU's rows.
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

// One aprefold iteration: the transform happened at fold time, so the
// timed work is ONLY the K concurrent int8 DMAs from the pinned artifact
// (same spin-barrier protocol as iterA).
double iterAPrefold(const reloc::prefold::PrefoldArtifact &art,
                    std::vector<GpuCtx> &ctxs) {
  const int8_t *src = static_cast<const int8_t *>(art.data());
  const double t0 = nowMs();
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
      CUDA_CHECK(cudaMemcpyAsync(g.dInt8, src + g.hostElemOffset,
                                 static_cast<size_t>(g.elems),
                                 cudaMemcpyHostToDevice, g.stream));
      CUDA_CHECK(cudaStreamSynchronize(g.stream));
    });
  while (ready.load() != static_cast<int>(ctxs.size())) {
  }
  go.store(true, std::memory_order_release);
  for (std::thread &t : ts)
    t.join();
  return nowMs() - t0;
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
               int K, const std::vector<int> &devices) {
  ctxs.resize(static_cast<size_t>(K));
  const int64_t shardElems = h.totalElems / K; // K divides rows -> divides elems
  for (int k = 0; k < K; ++k) {
    GpuCtx &g = ctxs[static_cast<size_t>(k)];
    g.dev = devices[static_cast<size_t>(k) % devices.size()];
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    if (broadcastPlacement(sc)) {
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
      // Only the relocating broadcast needs the fp32 intermediate.
      CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(g.elems) * 4));
      g.dTmp = static_cast<float *>(p);
    }
    // Equal for both broadcast placements (g.elems == totalElems -> rows).
    const int64_t channels = g.elems / h.channelSize;
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
                     const HostState &h, const RunTimes &t, double speedup,
                     double speedupVsA = -1) {
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
  if (speedupVsA >= 0)
    j += ", \"speedup_vs_a\": " + bench::jsonNumber(speedupVsA);
  j += "}";
  return j;
}

int run(const Options &opt) {
  int nDev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&nDev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Effective device list (issue #99): --devices verbatim, else 0..nDev-1
  // (the pre-#99 round-robin). Validated here, not in allocCtxs, so a bad
  // ordinal fails before any allocation.
  std::vector<int> devices = opt.devices;
  if (devices.empty())
    for (int d = 0; d < nDev; ++d)
      devices.push_back(d);
  for (int d : devices)
    if (d < 0 || d >= nDev) {
      std::fprintf(stderr, "error: --devices ordinal %d out of range [0, %d)\n",
                   d, nDev);
      return 2;
    }

  HostState h;
  buildHost(h, opt.scenario, opt.n);
  reloc::GatherPool pool(opt.threads);
  const reloc::quant::Variant variant = reloc::quant::Variant::Auto;
  reloc::CudaBackend backend(/*numStreams=*/1); // used for pinned staging only

  std::string rows;
  auto emit = [&](const std::string &j) {
    if (!rows.empty())
      rows += ",\n";
    rows += j;
  };

  std::string reuseRows;
  auto emitReuse = [&](const char *mode, Scenario sc, int64_t n, int K,
                       int nReuse, int loadsPerTrial, double aPerLoad,
                       double pfPerLoad, double tTransform,
                       double tPrefoldCold, bool predicted, bool measured) {
    if (!reuseRows.empty())
      reuseRows += ",\n";
    reuseRows +=
        "    {\"mode\": \"" + std::string(mode) + "\", \"scenario\": \"" +
        scenarioName(sc) + "\", \"N\": " + std::to_string(n) +
        ", \"K\": " + std::to_string(K) +
        ", \"n_reuse\": " + std::to_string(nReuse) +
        ", \"loads_per_trial\": " + std::to_string(loadsPerTrial) +
        ", \"a_per_load_ms\": " + bench::jsonNumber(aPerLoad) +
        ", \"prefold_per_load_ms\": " + bench::jsonNumber(pfPerLoad) +
        ", \"t_transform_ms\": " + bench::jsonNumber(tTransform) +
        ", \"t_prefold_cold_ms\": " + bench::jsonNumber(tPrefoldCold) +
        ", \"predicted_prefold_wins\": " +
        (predicted ? "true" : "false") +
        ", \"measured_prefold_wins\": " + (measured ? "true" : "false") +
        "}";
  };

  for (int K : opt.ks) {
    if (K < 1 || h.rows % K != 0) {
      std::fprintf(stderr, "skip K=%d (rows %lld not divisible)\n", K,
                   static_cast<long long>(h.rows));
      continue;
    }
    std::vector<GpuCtx> ctxs;
    allocCtxs(ctxs, opt.scenario, h, K, devices);

    // --- verify every method before timing --------------------------------
    double dma = 0;
    (void)iterA(h, ctxs, pool, variant, dma);
    for (const GpuCtx &g : ctxs)
      if (!verifyGpu(g, h)) {
        std::fprintf(stderr, "VERIFY FAILED: A %s K=%d\n",
                     scenarioName(opt.scenario), K);
        return 1;
      }

    // aprefold: fold once via the library component, verify the delivery.
    reloc::prefold::PrefoldArtifact art =
        foldOnce(h, backend, pool, variant);
    if (!art.valid()) {
      std::fprintf(stderr, "error: prefoldArtifact failed (%s K=%d)\n",
                   scenarioName(opt.scenario), K);
      return 1;
    }
    for (GpuCtx &g : ctxs) {
      CUDA_CHECK(cudaSetDevice(g.dev));
      CUDA_CHECK(cudaMemset(g.dInt8, 0xAB, static_cast<size_t>(g.elems)));
    }
    (void)iterAPrefold(art, ctxs);
    for (const GpuCtx &g : ctxs)
      if (!verifyGpu(g, h)) {
        std::fprintf(stderr, "VERIFY FAILED: aprefold %s K=%d\n",
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

    // --- time A, APrefold, B_xK, B_staged -----------------------------------
    auto timeIt = [&](Method m) {
      std::vector<double> wall, dmas;
      auto once = [&]() -> double {
        double d = 0;
        double w;
        if (m == Method::A)
          w = iterA(h, ctxs, pool, variant, d);
        else if (m == Method::APrefold) {
          w = iterAPrefold(art, ctxs);
          d = w; // the whole iteration IS the DMA leg
        } else
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
    RunTimes tap = timeIt(Method::APrefold);
    RunTimes tbk = timeIt(Method::BxK);
    RunTimes tbs = timeIt(Method::BStaged);
    const double sa = tbk.wall.median > 0 ? tbk.wall.median / ta.wall.median
                                          : 0.0;
    const double sap = tbk.wall.median > 0
                           ? tbk.wall.median / tap.wall.median
                           : 0.0;
    const double sapVsA = ta.wall.median > 0
                              ? ta.wall.median / tap.wall.median
                              : 0.0;
    emit(entryJson(opt.scenario, Method::A, K, opt.n, h, ta, sa));
    emit(entryJson(opt.scenario, Method::APrefold, K, opt.n, h, tap, sap,
                   sapVsA));
    emit(entryJson(opt.scenario, Method::BxK, K, opt.n, h, tbk, -1));
    emit(entryJson(opt.scenario, Method::BStaged, K, opt.n, h, tbs, -1));
    std::fprintf(stderr,
                 "%s K=%d: A %7.2f | Apre %7.2f | BxK %7.2f | Bstg %7.2f ms "
                 "| Apre/BxK %.2fx%s\n",
                 scenarioName(opt.scenario), K, ta.wall.median,
                 tap.wall.median, tbk.wall.median, tbs.wall.median, sap,
                 (opt.scenario == Scenario::Scatter && K == 4 &&
                  opt.n == 8192)
                     ? (sap >= 3.0 ? "  [V4-G1 PASS]" : "  [V4-G1 fail]")
                     : "");

    // --- V4 reuse sweep: amortization rule vs measured reuse counts -------
    // Trial(n): prefold = cold fold (allocStaging + transform) + n DMAs;
    //           A       = n x (transform + DMA).
    // The cold fold makes n_reuse=1 the cold single-use counter-case: the
    // rule input is tPrefoldCold (fold incl. pinned alloc) with penalty
    // folded in, i.e. prefoldWins(n, tTransform, tPrefoldCold, 0).
    if (!opt.reuse.empty()) {
      const int sweepIters = 7;
      // Median transform-only and cold-fold times feed the rule.
      std::vector<double> tT, tP;
      for (int i = 0; i < sweepIters; ++i) {
        const double t0 = nowMs();
        cpuTransform(h, pool, variant);
        tT.push_back(nowMs() - t0);
        const double t1 = nowMs();
        reloc::prefold::PrefoldArtifact cold =
            foldOnce(h, backend, pool, variant);
        tP.push_back(nowMs() - t1);
        if (!cold.valid())
          return 1;
      }
      const double tTransform = summarizeSamples(tT).median;
      const double tPrefoldCold = summarizeSamples(tP).median;
      for (int n : opt.reuse) {
        if (n < 1)
          continue;
        std::vector<double> pf, aa;
        for (int i = 0; i < sweepIters; ++i) {
          const double t0 = nowMs();
          reloc::prefold::PrefoldArtifact trialArt =
              foldOnce(h, backend, pool, variant);
          for (int j = 0; j < n; ++j)
            (void)iterAPrefold(trialArt, ctxs);
          pf.push_back((nowMs() - t0) / n);
          const double t1 = nowMs();
          for (int j = 0; j < n; ++j) {
            double d = 0;
            (void)iterA(h, ctxs, pool, variant, d);
          }
          aa.push_back((nowMs() - t1) / n);
        }
        const double pfMed = summarizeSamples(pf).median;
        const double aMed = summarizeSamples(aa).median;
        const bool predicted =
            reloc::prefold::prefoldWins(n, tTransform, tPrefoldCold, 0.0);
        const bool measured = pfMed < aMed;
        emitReuse("reuse", opt.scenario, opt.n, K, n, /*loadsPerTrial=*/n,
                  aMed, pfMed, tTransform, tPrefoldCold, predicted,
                  measured);
        std::fprintf(stderr,
                     "reuse n=%2d K=%d: A %7.2f ms/load | prefold %7.2f "
                     "ms/load | predicted %s measured %s\n",
                     n, K, aMed, pfMed, predicted ? "PREFOLD" : "A",
                     measured ? "PREFOLD" : "A");
      }
    }

    // --- V4 streaming counter-case: source mutates every load -------------
    // Model swapping: the artifact is never reusable, so every load pays a
    // fresh fold (incl. its pinned allocation). n_reuse = 1 per artifact;
    // the rule predicts A.
    if (opt.streaming) {
      const int loads = 4, sweepIters = 7;
      std::vector<double> pf, aa, tT, tP;
      for (int i = 0; i < sweepIters; ++i) {
        const double t0 = nowMs();
        for (int j = 0; j < loads; ++j) {
          mutateSource(h, i * loads + j);
          reloc::prefold::PrefoldArtifact sArt =
              foldOnce(h, backend, pool, variant);
          (void)iterAPrefold(sArt, ctxs);
        }
        pf.push_back((nowMs() - t0) / loads);
        const double t1 = nowMs();
        for (int j = 0; j < loads; ++j) {
          mutateSource(h, 1000 + i * loads + j);
          double d = 0;
          (void)iterA(h, ctxs, pool, variant, d);
        }
        aa.push_back((nowMs() - t1) / loads);
        // Rule inputs, re-measured under mutation (they match the sweep's).
        const double t2 = nowMs();
        cpuTransform(h, pool, variant);
        tT.push_back(nowMs() - t2);
        const double t3 = nowMs();
        reloc::prefold::PrefoldArtifact cold =
            foldOnce(h, backend, pool, variant);
        tP.push_back(nowMs() - t3);
        if (!cold.valid())
          return 1;
      }
      const double tTransform = summarizeSamples(tT).median;
      const double tPrefoldCold = summarizeSamples(tP).median;
      const bool predicted =
          reloc::prefold::prefoldWins(1, tTransform, tPrefoldCold, 0.0);
      const double pfMed = summarizeSamples(pf).median;
      const double aMed = summarizeSamples(aa).median;
      emitReuse("streaming", opt.scenario, opt.n, K, 1, loads, aMed, pfMed,
                tTransform, tPrefoldCold, predicted, pfMed < aMed);
      // Post-mutation correctness: recompute the reference once and check
      // the last delivery (the timed loop itself is unverified by design).
      reloc::quant::gatherQuantizeF32S8(h.plan, h.hF32, h.ref.data(),
                                        h.invScales.data(), 0, h.rows,
                                        reloc::quant::Variant::Scalar);
      reloc::prefold::PrefoldArtifact finalArt =
          foldOnce(h, backend, pool, variant);
      (void)iterAPrefold(finalArt, ctxs);
      for (const GpuCtx &g : ctxs)
        if (!verifyGpu(g, h)) {
          std::fprintf(stderr, "VERIFY FAILED: streaming %s K=%d\n",
                       scenarioName(opt.scenario), K);
          return 1;
        }
    }

    freeCtxs(ctxs);
  }
  pool.close();

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"multigpu_reloc\", \"gpu\": \"" +
      std::string(prop.name) + "\", \"scenario\": \"" +
      scenarioName(opt.scenario) + "\", \"N\": " + std::to_string(opt.n) +
      ", \"cpu_threads\": " + std::to_string(opt.threads) +
      ", \"warmup\": " + std::to_string(opt.warmup) +
      ", \"iters\": " + std::to_string(opt.iters) + ", \"devices\": [" +
      [&] {
        std::string ds;
        for (int d : devices)
          ds += (ds.empty() ? "" : ", ") + std::to_string(d);
        return ds;
      }() +
      "]},\n  \"rows\": [\n" + rows +
      "\n  ],\n  \"reuse_rows\": [\n" + reuseRows + "\n  ]\n}\n";
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
      else if (s == "broadcast_contig")
        opt.scenario = Scenario::BroadcastContig;
      else {
        std::fprintf(stderr,
                     "error: --scenario broadcast|scatter|broadcast_contig\n");
        return 2;
      }
    } else if (a == "--n")
      opt.n = std::atoll(next());
    else if (a == "--k")
      opt.ks = parseInts(next());
    else if (a == "--devices")
      opt.devices = parseInts(next());
    else if (a == "--reuse")
      opt.reuse = parseInts(next());
    else if (a == "--streaming")
      opt.streaming = true;
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
                   "usage: bench-multigpu-reloc "
                   "[--scenario broadcast|scatter|broadcast_contig] "
                   "[--n N] [--k 1,2,4] [--devices 0,1,2,3] "
                   "[--reuse 1,2,4,16] [--streaming] "
                   "[--threads T] [--warmup W] [--iters I] [--json PATH|-]\n");
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
