//===- poc_transpose.cu - C7 PoC reproduction driver (issue #47) ----------===//
//
// N x N fp32 blocked transpose (the golden "reference" plan: 4D view +
// transpose(0,1), divisible(N, 64)):
//   ours     strategy-4 pipeline -- CudaBackend with 2 streams, a
//            caller-owned pinned pool with 2 buffers, event-recycled
//            double buffering. Metric: end-to-end wall ms (host-blocking).
//   baseline whole-tensor pinned cudaMemcpyAsync H2D + naive strided
//            relocate kernel (the PoC baseline; CuTe arrives with E3).
//            Metrics: CUDA-event GPU ms and wall ms.
// Protocol: bench/protocol.h (build doc §3). Speedup = baseline wall
// median / ours wall median, medians taken over the per-rerun medians.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/ChunkSchedule.h"
#include "reloc/CudaBackend.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/PinnedBufferPool.h"
#include "reloc/Pipeline.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,         \
                   __LINE__, cudaGetErrorString(err_), #x);                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kMaxRank = 8;

// The bound plan's coalesced axes, passed to the kernel by value.
struct Axes {
  int64_t ext[kMaxRank];
  int64_t srcStride[kMaxRank];
  int64_t dstStride[kMaxRank];
  int rank;
};

// Naive strided relocate: one thread per valid element; decompose the
// linear index over the coalesced extents (row-major), gather via
// srcStrides, scatter via dstStrides. This IS the plan executed on
// device -- generic, unfused, deliberately naive (the PoC baseline).
__global__ void relocateKernel(const float *src, float *dst, Axes a,
                               int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  int64_t rem = i, srcOff = 0, dstOff = 0;
  for (int k = a.rank - 1; k >= 0; --k) {
    int64_t c = rem % a.ext[k];
    rem /= a.ext[k];
    srcOff += c * a.srcStride[k];
    dstOff += c * a.dstStride[k];
  }
  dst[dstOff] = src[srcOff];
}

struct Options {
  int64_t n = 32768;
  int warmup = bench::kWarmupIters;
  int iters = bench::kTimedIters;
  int reruns = bench::kReruns;
  const char *jsonPath = "-";
  bool verify = false;
};

double medianOfRerunMedians(const bench::Series &s) {
  std::vector<double> meds;
  for (const bench::Stats &st : s.reruns)
    meds.push_back(st.median);
  return bench::summarize(meds).median;
}

int run(const Options &opt) {
  // --- plan: decode the golden reference blob and bind at N ---------------
  std::vector<uint8_t> planBytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(planBytes.data(), planBytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan) {
    std::fprintf(stderr, "error: reference plan failed to decode\n");
    return 1;
  }
  auto boundResult = reloc::bind(*plan, {{"N", opt.n}});
  auto *bound = std::get_if<reloc::BoundPlan>(&boundResult);
  if (!bound) {
    std::fprintf(stderr, "error: bind failed: %s\n",
                 std::get<reloc::BindError>(boundResult).message.c_str());
    return 1;
  }
  const size_t bytes = static_cast<size_t>(bound->totalBytes);
  const int64_t totalElems = opt.n * opt.n;
  if (bytes != static_cast<size_t>(totalElems) * 4) {
    std::fprintf(stderr, "error: unexpected plan footprint\n");
    return 1;
  }
  if (bound->extents.size() > kMaxRank) {
    std::fprintf(stderr, "error: plan rank exceeds kernel max\n");
    return 1;
  }

  // --- buffers -------------------------------------------------------------
  float *hSrc = nullptr; // pinned; shared by both methods (see plan note)
  if (cudaHostAlloc(&hSrc, bytes, cudaHostAllocDefault) != cudaSuccess) {
    std::fprintf(stderr,
                 "error: pinned alloc of %.1f GiB failed (WSL2 pinned "
                 "limit?) -- retry with a smaller --n\n",
                 static_cast<double>(bytes) / (1ull << 30));
    return 1;
  }
  for (int64_t i = 0; i < totalElems; ++i) {
    uint32_t bits = static_cast<uint32_t>(i) * 2654435761u;
    std::memcpy(&hSrc[i], &bits, 4);
  }
  float *dDst = nullptr, *dSrc = nullptr;
  CUDA_CHECK(cudaMalloc(&dDst, bytes));
  CUDA_CHECK(cudaMalloc(&dSrc, bytes)); // baseline's linear device copy

  // --- ours: backend + caller-owned pool (2 buffers, 2 streams) -----------
  reloc::CudaBackend backend(/*numStreams=*/2);
  reloc::ChunkSchedule sched = reloc::planChunks(*bound, /*nBuffers=*/2);
  reloc::PinnedBufferPool pool(backend, /*nBuffers=*/2, sched.maxChunkBytes);

  // --- baseline: stream + events + kernel launch geometry ------------------
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t evStart, evStop;
  CUDA_CHECK(cudaEventCreate(&evStart));
  CUDA_CHECK(cudaEventCreate(&evStop));
  Axes axes{};
  axes.rank = static_cast<int>(bound->extents.size());
  for (int k = 0; k < axes.rank; ++k) {
    axes.ext[k] = bound->extents[k];
    axes.srcStride[k] = bound->srcStrides[k];
    axes.dstStride[k] = bound->dstStrides[k];
  }
  const int kBlock = 256;
  const int64_t kGrid = (totalElems + kBlock - 1) / kBlock;

  auto oursIter = [&] {
    reloc::executeH2DPipelined(*bound, hSrc, dDst, backend, pool);
  };
  auto baselineIter = [&]() -> double {
    CUDA_CHECK(cudaEventRecord(evStart, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(dSrc, hSrc, bytes, cudaMemcpyHostToDevice, stream));
    relocateKernel<<<static_cast<unsigned>(kGrid), kBlock, 0, stream>>>(
        dSrc, dDst, axes, totalElems);
    CUDA_CHECK(cudaEventRecord(evStop, stream));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    return static_cast<double>(ms);
  };

  // --- verify (optional): both methods vs CPU executeH2D -------------------
  if (opt.verify) {
    std::fprintf(stderr, "verify: computing CPU reference (%.1f GiB)...\n",
                 static_cast<double>(bytes) / (1ull << 30));
    std::vector<uint8_t> ref(bytes), got(bytes);
    reloc::executeH2D(*bound, hSrc, ref.data());
    CUDA_CHECK(cudaMemset(dDst, 0, bytes));
    oursIter();
    CUDA_CHECK(cudaMemcpy(got.data(), dDst, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(got.data(), ref.data(), bytes) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: ours != CPU reference\n");
      return 1;
    }
    CUDA_CHECK(cudaMemset(dDst, 0, bytes));
    (void)baselineIter();
    CUDA_CHECK(cudaMemcpy(got.data(), dDst, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(got.data(), ref.data(), bytes) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: baseline != CPU reference\n");
      return 1;
    }
    std::fprintf(stderr, "verify: both methods byte-exact vs CPU\n");
  }

  // --- measure --------------------------------------------------------------
  std::vector<std::vector<double>> oursWall, baseWall, baseGpu;
  for (int r = 0; r < opt.reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(oursIter, opt.warmup, opt.iters);
    oursWall.push_back(std::move(s.wall_ms));
  }
  for (int r = 0; r < opt.reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(baselineIter, opt.warmup, opt.iters);
    baseWall.push_back(std::move(s.wall_ms));
    baseGpu.push_back(std::move(s.extra_ms));
  }
  bench::Series oursWallS = bench::analyzeReruns(oursWall);
  bench::Series baseWallS = bench::analyzeReruns(baseWall);
  bench::Series baseGpuS = bench::analyzeReruns(baseGpu);
  double speedup =
      medianOfRerunMedians(baseWallS) / medianOfRerunMedians(oursWallS);

  // --- report ----------------------------------------------------------------
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::string doc =
      "{\n  \"config\": {\"benchmark\": \"poc_transpose\", \"plan\": "
      "\"reference\", \"N\": " +
      std::to_string(opt.n) +
      ", \"dtype\": \"f32\", \"bytes\": " + std::to_string(bytes) +
      ", \"n_buffers\": 2, \"n_streams\": 2, "
      "\"chunk_bytes\": " +
      std::to_string(sched.maxChunkBytes) +
      ", \"warmup\": " + std::to_string(opt.warmup) +
      ", \"iters\": " + std::to_string(opt.iters) +
      ", \"reruns\": " + std::to_string(opt.reruns) + ", \"gpu\": \"" +
      prop.name + "\", \"verified\": " + (opt.verify ? "true" : "false") +
      "},\n  \"methods\": {\n    \"ours_pipeline_2buf_2stream\": "
      "{\"wall_ms\": " +
      bench::seriesToJson(oursWallS) +
      "},\n    \"baseline_pinned_memcpy_naive_kernel\": {\"wall_ms\": " +
      bench::seriesToJson(baseWallS) +
      ", \"gpu_ms\": " + bench::seriesToJson(baseGpuS) +
      "}\n  },\n  \"speedup_wall_median\": " + bench::jsonNumber(speedup) +
      "\n}\n";
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
  std::fprintf(
      stderr,
      "poc_transpose N=%lld: ours %.2f ms (spread %.2f%%), baseline %.2f "
      "ms wall / %.2f ms gpu (spread %.2f%%) -> speedup %.2fx\n",
      static_cast<long long>(opt.n), medianOfRerunMedians(oursWallS),
      oursWallS.medianSpreadPct, medianOfRerunMedians(baseWallS),
      medianOfRerunMedians(baseGpuS), baseWallS.medianSpreadPct, speedup);

  CUDA_CHECK(cudaEventDestroy(evStart));
  CUDA_CHECK(cudaEventDestroy(evStop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dSrc));
  CUDA_CHECK(cudaFree(dDst));
  CUDA_CHECK(cudaFreeHost(hSrc));
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--n")
      opt.n = std::atoll(next());
    else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--reruns")
      opt.reruns = std::atoi(next());
    else if (a == "--json")
      opt.jsonPath = next();
    else if (a == "--verify")
      opt.verify = true;
    else {
      std::fprintf(stderr,
                   "usage: bench-poc-transpose [--n N] [--warmup W] "
                   "[--iters I] [--reruns R] [--json PATH|-] [--verify]\n"
                   "  N must be divisible by 64. --verify needs ~2x extra "
                   "host RAM.\n");
      return 2;
    }
  }
  if (opt.n <= 0 || opt.n % 64 != 0) {
    std::fprintf(stderr, "error: N must be positive and divisible by 64\n");
    return 2;
  }
  if (opt.warmup < 0 || opt.iters < 1 || opt.reruns < 1) {
    std::fprintf(stderr,
                 "error: --warmup must be >= 0, --iters and --reruns must "
                 "be >= 1\n");
    return 2;
  }
  return run(opt);
}
