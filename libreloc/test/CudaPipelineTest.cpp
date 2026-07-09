//===- CudaPipelineTest.cpp - GPU pipeline round-trip (local only) --------===//
//
// Compiled and run only under RELOC_ENABLE_CUDA on a machine with a GPU; never
// in CI. Proves executeH2DPipelined lands the same bytes on the device as the
// CPU executeH2D reference, and D2H(H2D(x)) == x, across buffer counts.
//
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Bind.h"
#include "reloc/CudaBackend.h"
#include "reloc/Execute.h"
#include "reloc/Pipeline.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace {

using reloc::BoundPlan;

int64_t product(const std::vector<int64_t> &v) {
  return std::accumulate(v.begin(), v.end(), int64_t(1),
                         std::multiplies<int64_t>());
}

std::vector<uint8_t> iotaBytes(int64_t elements, uint32_t elementSize) {
  std::vector<uint8_t> buf(static_cast<size_t>(elements) * elementSize);
  for (int64_t e = 0; e < elements; ++e)
    for (uint32_t b = 0; b < elementSize; ++b)
      buf[e * elementSize + b] =
          static_cast<uint8_t>((e * 131 + b * 17) & 0xff);
  return buf;
}

BoundPlan transposeN(int64_t n) {
  // dst [n,n] <- src [n,n] transpose (axis0 src 1 dst n, axis1 src n dst 1).
  BoundPlan b;
  b.extents = {n, n};
  b.srcStrides = {1, n};
  b.dstStrides = {n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = 1;
  return b;
}

TEST(CudaPipeline, H2DMatchesCpuAndRoundTrips) {
  const int64_t n = 8192; // acceptance: reference-shaped plan at N = 8192
  BoundPlan b = transposeN(n);
  int64_t elems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(elems, b.elementSize);
  std::vector<uint8_t> cpuRef(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), cpuRef.data());

  void *dev = nullptr;
  ASSERT_EQ(cudaMalloc(&dev, static_cast<size_t>(b.totalBytes)), cudaSuccess);

  for (int nBuffers : {1, 2, 4}) {
    reloc::CudaBackend backend(2);
    ASSERT_EQ(cudaMemset(dev, 0, static_cast<size_t>(b.totalBytes)),
              cudaSuccess);
    reloc::executeH2DPipelined(b, src.data(), dev, backend, nBuffers,
                               /*override=*/0);
    std::vector<uint8_t> devBack(static_cast<size_t>(b.totalBytes), 0);
    ASSERT_EQ(cudaMemcpy(devBack.data(), dev, static_cast<size_t>(b.totalBytes),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(devBack, cpuRef) << "H2D nBuffers=" << nBuffers;

    std::vector<uint8_t> host(static_cast<size_t>(elems) * b.elementSize, 0xCD);
    reloc::executeD2HPipelined(b, dev, host.data(), backend, nBuffers,
                               /*override=*/0);
    EXPECT_EQ(host, src) << "D2H round-trip nBuffers=" << nBuffers;
  }
  cudaFree(dev);
}

} // namespace

#endif // RELOC_ENABLE_CUDA
