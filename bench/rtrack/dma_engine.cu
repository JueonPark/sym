//===- dma_engine.cu - R5/EXP-5 DMA-engine negative result (#86) ----------===//
//
// Why the copy engine and zero-copy cannot substitute for the CPU-side
// gather in Method A: both collapse when the access is fine-grained/strided.
//
//   part 1 -- cudaMemcpy2DAsync H2D at a swept row width, dense
//     (spitch = width) and strided (spitch = 2*width, the engine reads
//     `width` bytes then skips `width`). Total useful bytes fixed, so
//     effective BW = total / time exposes the per-row descriptor floor at
//     small widths.
//   part 2 -- UVA zero-copy: a GPU kernel reads host-pinned MAPPED memory
//     at a swept element stride and reduces it; delivered BW =
//     (elements read * 4) / time falls far below the pinned-copy ceiling
//     and collapses with stride (each strided read still pulls a full
//     PCIe burst, most of it wasted).
//
// Both parts verify before timing (2D: byte-exact copy; zero-copy: the
// all-ones sum equals the element count exactly, via a double accumulator).
// GPU-only, never in CI. Hard 2-day box: one table, done.
//
//===----------------------------------------------------------------------===//

#include "rtrack/rstats.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

// Zero-copy strided read: each thread reads p[i*stride] over a grid-stride
// loop and atomically adds its partial to a double accumulator (sm_75 has
// double atomicAdd). All-ones input -> sum == count, an exact verify.
__global__ void zeroCopyReadKernel(const float *p, int64_t count,
                                   int64_t stride, double *out) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  double acc = 0.0;
  for (; i < count; i += step)
    acc += static_cast<double>(p[i * stride]);
  atomicAdd(out, acc);
}

struct Timing {
  RStats ms;
  double gbps = 0; // usefulBytes / time
};

template <typename Launch>
Timing timeIt(Launch &&launch, int64_t usefulBytes, int warmup, int iters,
              cudaStream_t stream) {
  cudaEvent_t beg, end;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(beg, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    samples.push_back(ms);
  }
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  Timing t;
  t.ms = summarizeSamples(samples);
  t.gbps = t.ms.median > 0
               ? static_cast<double>(usefulBytes) / (t.ms.median * 1e-3) / 1e9
               : 0.0;
  return t;
}

struct Options {
  int64_t totalMiB = 256; // fixed useful bytes for the 2D sweep
  int64_t zcMiB = 256;    // zero-copy buffer size
  int warmup = 3;
  int iters = 20;
  const char *jsonPath = "-";
};

// part 1: cudaMemcpy2DAsync row-width sweep, dense + strided.
std::string runMemcpy2D(const Options &opt, cudaStream_t stream) {
  const int64_t total = opt.totalMiB << 20;
  const int64_t widths[] = {64,        256,       1 << 10,  4 << 10,
                            16 << 10,  64 << 10,  256 << 10, 1 << 20};
  std::string rows;
  for (int strided = 0; strided < 2; ++strided) {
    for (int64_t W : widths) {
      const int64_t H = total / W;         // rows; total useful stays fixed
      const int64_t spitch = strided ? 2 * W : W;
      const int64_t srcBytes = spitch * H; // host footprint
      void *hSrc = nullptr, *dDst = nullptr;
      CUDA_CHECK(cudaHostAlloc(&hSrc, static_cast<size_t>(srcBytes),
                               cudaHostAllocDefault));
      std::memset(hSrc, 0x3C, static_cast<size_t>(srcBytes));
      CUDA_CHECK(cudaMalloc(&dDst, static_cast<size_t>(W * H)));
      auto launch = [&] {
        CUDA_CHECK(cudaMemcpy2DAsync(dDst, W, hSrc, spitch, W, H,
                                     cudaMemcpyHostToDevice, stream));
      };
      // verify: first row copied byte-exact.
      launch();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<uint8_t> got(static_cast<size_t>(W));
      CUDA_CHECK(cudaMemcpy(got.data(), dDst, static_cast<size_t>(W),
                            cudaMemcpyDeviceToHost));
      std::vector<uint8_t> want(static_cast<size_t>(W), 0x3C);
      if (std::memcmp(got.data(), want.data(), static_cast<size_t>(W)) != 0) {
        std::fprintf(stderr, "VERIFY FAILED: memcpy2d W=%lld strided=%d\n",
                     static_cast<long long>(W), strided);
        std::exit(1);
      }
      Timing t = timeIt(launch, total, opt.warmup, opt.iters, stream);
      cudaFreeHost(hSrc);
      cudaFree(dDst);
      if (!rows.empty())
        rows += ",\n";
      rows += "    {\"mode\": \"" + std::string(strided ? "strided" : "dense") +
              "\", \"row_width_bytes\": " + std::to_string(W) +
              ", \"rows\": " + std::to_string(H) +
              ", \"median_ms\": " + bench::jsonNumber(t.ms.median) +
              ", \"gb_per_s\": " + bench::jsonNumber(t.gbps) + "}";
      std::fprintf(stderr, "memcpy2d %-7s W=%6lldB  %7.3f GB/s\n",
                   strided ? "strided" : "dense", static_cast<long long>(W),
                   t.gbps);
    }
  }
  return rows;
}

// part 2: UVA zero-copy strided read sweep.
std::string runZeroCopy(const Options &opt, cudaStream_t stream) {
  const int64_t n = (opt.zcMiB << 20) / 4; // float count
  float *hMap = nullptr;
  CUDA_CHECK(cudaHostAlloc(&hMap, static_cast<size_t>(n) * 4,
                           cudaHostAllocMapped));
  for (int64_t i = 0; i < n; ++i)
    hMap[i] = 1.0f;
  float *dMap = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dMap, hMap, 0));
  double *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, sizeof(double)));

  const int64_t strides[] = {1, 2, 4, 8, 16, 32};
  const int kThreads = 256;
  std::string rows;
  for (int64_t S : strides) {
    const int64_t count = n / S; // elements actually read
    const int64_t blocks =
        (count + kThreads - 1) / kThreads > 65535
            ? 65535
            : (count + kThreads - 1) / kThreads;
    auto launch = [&] {
      CUDA_CHECK(cudaMemsetAsync(dOut, 0, sizeof(double), stream));
      zeroCopyReadKernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                           stream>>>(dMap, count, S, dOut);
      CUDA_CHECK(cudaGetLastError());
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double got = 0;
    CUDA_CHECK(cudaMemcpy(&got, dOut, sizeof(double), cudaMemcpyDeviceToHost));
    if (got != static_cast<double>(count)) {
      std::fprintf(stderr,
                   "VERIFY FAILED: zerocopy stride=%lld sum %.1f != %lld\n",
                   static_cast<long long>(S), got,
                   static_cast<long long>(count));
      std::exit(1);
    }
    Timing t = timeIt(launch, count * 4, opt.warmup, opt.iters, stream);
    if (!rows.empty())
      rows += ",\n";
    rows += "    {\"stride_elems\": " + std::to_string(S) +
            ", \"elems_read\": " + std::to_string(count) +
            ", \"median_ms\": " + bench::jsonNumber(t.ms.median) +
            ", \"delivered_gb_per_s\": " + bench::jsonNumber(t.gbps) + "}";
    std::fprintf(stderr, "zerocopy stride=%3lld  %6.3f GB/s delivered\n",
                 static_cast<long long>(S), t.gbps);
  }
  cudaFree(dOut);
  cudaFreeHost(hMap);
  return rows;
}

int run(const Options &opt) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  if (!prop.canMapHostMemory) {
    std::fprintf(stderr, "error: device cannot map host memory (zero-copy)\n");
    return 1;
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  std::string m2d = runMemcpy2D(opt, stream);
  std::string zc = runZeroCopy(opt, stream);

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"dma_engine\", \"gpu\": \"" +
      std::string(prop.name) +
      "\", \"total_mib\": " + std::to_string(opt.totalMiB) +
      ", \"zerocopy_mib\": " + std::to_string(opt.zcMiB) +
      ", \"warmup\": " + std::to_string(opt.warmup) +
      ", \"iters\": " + std::to_string(opt.iters) +
      "},\n  \"memcpy2d\": [\n" + m2d + "\n  ],\n  \"zerocopy\": [\n" + zc +
      "\n  ]\n}\n";
  cudaStreamDestroy(stream);
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

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--total-mib")
      opt.totalMiB = std::atoll(next());
    else if (a == "--zc-mib")
      opt.zcMiB = std::atoll(next());
    else if (a == "--warmup")
      opt.warmup = std::atoi(next());
    else if (a == "--iters")
      opt.iters = std::atoi(next());
    else if (a == "--json")
      opt.jsonPath = next();
    else {
      std::fprintf(stderr, "usage: bench-dma-engine [--total-mib M] "
                           "[--zc-mib M] [--warmup W] [--iters I] "
                           "[--json PATH|-]\n");
      return 2;
    }
  }
  if (opt.totalMiB < 1 || opt.zcMiB < 1 || opt.warmup < 0 || opt.iters < 1)
    return 2;
  return run(opt);
}
