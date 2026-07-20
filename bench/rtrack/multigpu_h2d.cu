//===- multigpu_h2d.cu - M0 per-GPU and aggregate pinned H2D/D2H ----------===//
//
// Issue #73's M0 bring-up numbers (and the EXP-3 seed): pinned-buffer
// copy bandwidth per GPU measured alone, then concurrently on an
// arbitrary GPU subset (one host thread per GPU, its own stream and
// pinned buffer, a start barrier, cudaMemcpyAsync loops). The aggregate
// vs sum-of-single ratio is the pre-registered EXP-3 gate input; pair
// subsets probe which GPUs share a root port behaviorally (this box:
// GPU0/GPU1 sit on one die per nvidia-smi topo).
//
// NUMA caveat (documented in the M0 report): without libnuma the pinned
// buffers are all allocated from the main thread, so their page placement
// follows that thread's node, not each GPU's. Interpret per-GPU asymmetry
// with the topology in hand.
//
//===----------------------------------------------------------------------===//

#include "rtrack/rstats.h"

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

using bench::rtrack::nowMs;

struct Options {
  std::vector<int> gpus; // empty = all
  int64_t bytes = 256ll << 20;
  int iters = 20;
  int warmup = 3;
  int rounds = 3;   // repeat each measurement; report the best round
  bool d2h = false; // also measure D2H
};

struct GpuCtx {
  int dev = -1;
  void *hBuf = nullptr; // pinned
  void *dBuf = nullptr;
  cudaStream_t stream = nullptr;
};

void initCtx(GpuCtx &g, int dev, int64_t bytes) {
  g.dev = dev;
  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaHostAlloc(&g.hBuf, static_cast<size_t>(bytes),
                           cudaHostAllocDefault));
  std::memset(g.hBuf, 0x5A, static_cast<size_t>(bytes));
  CUDA_CHECK(cudaMalloc(&g.dBuf, static_cast<size_t>(bytes)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
}

void freeCtx(GpuCtx &g) {
  cudaSetDevice(g.dev);
  cudaStreamDestroy(g.stream);
  cudaFree(g.dBuf);
  cudaFreeHost(g.hBuf);
}

// One timed copy loop on this GPU; returns GB/s over the whole loop.
double copyLoop(GpuCtx &g, int64_t bytes, int iters, bool h2d) {
  CUDA_CHECK(cudaSetDevice(g.dev));
  const double t0 = nowMs();
  for (int i = 0; i < iters; ++i) {
    if (h2d)
      CUDA_CHECK(cudaMemcpyAsync(g.dBuf, g.hBuf, static_cast<size_t>(bytes),
                                 cudaMemcpyHostToDevice, g.stream));
    else
      CUDA_CHECK(cudaMemcpyAsync(g.hBuf, g.dBuf, static_cast<size_t>(bytes),
                                 cudaMemcpyDeviceToHost, g.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g.stream));
  const double sec = (nowMs() - t0) * 1e-3;
  return static_cast<double>(bytes) * iters / sec / 1e9;
}

// Concurrent run over all ctxs: one thread per GPU, spin barrier, then
// each runs the same copy loop. Returns per-GPU GB/s (aggregate = sum).
std::vector<double> concurrentRun(std::vector<GpuCtx> &ctxs, int64_t bytes,
                                  int iters, bool h2d) {
  std::vector<double> gbps(ctxs.size(), 0.0);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(ctxs.size());
  for (size_t i = 0; i < ctxs.size(); ++i) {
    threads.emplace_back([&, i] {
      CUDA_CHECK(cudaSetDevice(ctxs[i].dev));
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) {
      }
      gbps[i] = copyLoop(ctxs[i], bytes, iters, h2d);
    });
  }
  while (ready.load() != static_cast<int>(ctxs.size())) {
  }
  go.store(true, std::memory_order_release);
  for (std::thread &t : threads)
    t.join();
  return gbps;
}

std::string gpuList(const std::vector<GpuCtx> &ctxs) {
  std::string s;
  for (const GpuCtx &g : ctxs)
    s += (s.empty() ? "" : ",") + std::to_string(g.dev);
  return s;
}

int run(const Options &opt) {
  int nDev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&nDev));
  std::vector<int> devs = opt.gpus;
  if (devs.empty())
    for (int d = 0; d < nDev; ++d)
      devs.push_back(d);
  for (int d : devs)
    if (d < 0 || d >= nDev) {
      std::fprintf(stderr, "error: GPU %d out of range (%d devices)\n", d,
                   nDev);
      return 2;
    }

  std::vector<GpuCtx> ctxs(devs.size());
  for (size_t i = 0; i < devs.size(); ++i)
    initCtx(ctxs[i], devs[i], opt.bytes);

  std::string json = "{\n  \"benchmark\": \"multigpu_h2d\", \"bytes\": " +
                     std::to_string(opt.bytes) +
                     ", \"iters\": " + std::to_string(opt.iters) +
                     ", \"gpus\": [" + gpuList(ctxs) + "],\n";

  for (bool h2d : {true, false}) {
    if (!h2d && !opt.d2h)
      break;
    const char *dir = h2d ? "h2d" : "d2h";
    // Per-GPU, alone. Best of `rounds` timed loops: transient dips (link
    // retrain, clock ramp, IF traffic from elsewhere) only ever depress a
    // copy loop, so max-of-rounds is the honest capability number.
    std::vector<double> single(ctxs.size(), 0.0);
    for (size_t i = 0; i < ctxs.size(); ++i) {
      (void)copyLoop(ctxs[i], opt.bytes, opt.warmup, h2d);
      for (int r = 0; r < opt.rounds; ++r)
        single[i] =
            std::max(single[i], copyLoop(ctxs[i], opt.bytes, opt.iters, h2d));
      std::fprintf(stderr, "%s GPU%d alone: %6.2f GB/s\n", dir, ctxs[i].dev,
                   single[i]);
    }
    // All requested GPUs concurrently; keep the round with the highest
    // aggregate (per-GPU splits within a round stay consistent).
    (void)concurrentRun(ctxs, opt.bytes, opt.warmup, h2d);
    std::vector<double> conc;
    double bestAgg = -1;
    for (int r = 0; r < opt.rounds; ++r) {
      std::vector<double> round =
          concurrentRun(ctxs, opt.bytes, opt.iters, h2d);
      double agg = 0;
      for (double g : round)
        agg += g;
      if (agg > bestAgg) {
        bestAgg = agg;
        conc = round;
      }
    }
    double aggregate = 0, sumSingle = 0;
    std::string singleJson, concJson;
    for (size_t i = 0; i < ctxs.size(); ++i) {
      aggregate += conc[i];
      sumSingle += single[i];
      singleJson += (i ? ", " : "") + bench::jsonNumber(single[i]);
      concJson += (i ? ", " : "") + bench::jsonNumber(conc[i]);
      std::fprintf(stderr, "%s GPU%d concurrent: %6.2f GB/s\n", dir,
                   ctxs[i].dev, conc[i]);
    }
    std::fprintf(stderr,
                 "%s aggregate %6.2f GB/s vs sum-of-single %6.2f GB/s "
                 "(scaling %.2fx of %zux)\n",
                 dir, aggregate, sumSingle,
                 aggregate / (sumSingle / static_cast<double>(ctxs.size())),
                 ctxs.size());
    json += std::string("  \"") + dir + "\": {\"single_gbps\": [" +
            singleJson + "], \"concurrent_gbps\": [" + concJson +
            "], \"aggregate_gbps\": " + bench::jsonNumber(aggregate) +
            ", \"sum_single_gbps\": " + bench::jsonNumber(sumSingle) + "},\n";
  }
  json += "  \"end\": true\n}\n";
  std::fputs(json.c_str(), stdout);

  for (GpuCtx &g : ctxs)
    freeCtx(g);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--gpus") {
      std::string s = next();
      size_t pos = 0;
      while (pos <= s.size()) {
        size_t nextC = s.find(',', pos);
        if (nextC == std::string::npos)
          nextC = s.size();
        if (nextC > pos)
          opt.gpus.push_back(std::atoi(s.substr(pos, nextC - pos).c_str()));
        pos = nextC + 1;
      }
    } else if (a == "--mib")
      opt.bytes = std::atoll(next()) << 20;
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--rounds")
      opt.rounds = std::atoi(next());
    else if (a == "--d2h")
      opt.d2h = true;
    else {
      std::fprintf(stderr, "usage: bench-multigpu-h2d [--gpus 0,1,...] "
                           "[--mib M] [--iters I] [--warmup W] [--rounds R] "
                           "[--d2h]\n");
      return 2;
    }
  }
  if (opt.bytes < (1 << 20) || opt.iters < 1 || opt.warmup < 0 ||
      opt.rounds < 1) {
    std::fprintf(stderr, "error: bad --mib/--iters/--warmup/--rounds\n");
    return 2;
  }
  return run(opt);
}
