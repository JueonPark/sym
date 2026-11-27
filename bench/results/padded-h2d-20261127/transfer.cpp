#include "reloc/ChunkSchedule.h"
#include "reloc/CudaBackend.h"
#include "reloc/PinnedBufferPool.h"
#include "reloc/Pipeline.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MEASURE_PHASES
extern long long phaseNs, phaseCalls;
#endif

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char **argv) {
  if (argc != 2 ||
      (std::string(argv[1]) != "padded" && std::string(argv[1]) != "plain"))
    return 2;
  const int pad = std::string(argv[1]) == "padded" ? 1 : 0;
  constexpr int rows = 4096, cols = 1024;
  constexpr uint32_t fill = 0xc0200000; // -2.5f, as in pad_2d_outer.json
  reloc::BoundPlan plan;
  plan.extents = {rows - 2 * pad, cols};
  plan.dstStrides = plan.srcStrides = {cols, 1};
  plan.elementSize = sizeof(uint32_t);
  plan.totalBytes = rows * cols * sizeof(uint32_t);
  plan.L = cols;
  if (pad)
    plan.padRegions.push_back({0, 1, 1, fill});
  std::vector<uint32_t> src(plan.extents[0] * cols);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = uint32_t(i) * 2654435761u;
  std::vector<uint32_t> expected(rows * cols, fill), actual(rows * cols);
  std::copy(src.begin(), src.end(), expected.begin() + pad * cols);
  reloc::CudaBackend backend(2, 0);
  auto chunks = reloc::planChunks(plan, 2, 0);
  reloc::PinnedBufferPool pool(backend, 2, chunks.maxChunkBytes);
  void *device = nullptr;
  check(cudaMalloc(&device, plan.totalBytes));
  check(cudaMemset(device, 0xcc, plan.totalBytes));
  check(cudaDeviceSynchronize());
  using Clock = std::chrono::steady_clock;
  std::puts("iteration,wall_ns,fill_ns");
  for (int i = -10; i < 50; ++i) {
    src[0] = expected[pad * cols] = static_cast<uint32_t>(i);
#ifdef MEASURE_PHASES
    phaseCalls = phaseNs = 0;
#endif
    auto start = Clock::now();
    reloc::executeH2DPipelined(plan, src.data(), device, backend, pool);
    auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - start)
                    .count();
    long long fillNs = 0;
#ifdef MEASURE_PHASES
    if (phaseCalls != static_cast<long long>(chunks.chunks.size()))
      throw std::runtime_error("phase timer did not cover every chunk");
    fillNs = phaseNs;
#endif
    if (i == -10 || i == 49) {
      check(cudaMemcpy(actual.data(), device, plan.totalBytes,
                       cudaMemcpyDeviceToHost));
      if (actual != expected)
        throw std::runtime_error(
            "device result differs from independent oracle");
    }
    if (i >= 0)
      std::printf("%d,%lld,%lld\n", i, static_cast<long long>(wall), fillNs);
  }
  check(cudaFree(device));
}
