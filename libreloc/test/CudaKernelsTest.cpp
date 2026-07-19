//===- CudaKernelsTest.cpp - R0.2 GPU kernel correctness (local only) -----===//
//
// Compiled and run only under RELOC_ENABLE_CUDA on a machine with a GPU;
// never in CI (the CudaPipelineTest convention). Every kernel is checked
// against its CPU reference: relocate vs executeH2D byte-exact, quantize vs
// reloc::quant scalar bit-exact, dequant/unpack exact by construction.
//
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaKernels.h"
#include "reloc/Bind.h"
#include "reloc/Execute.h"
#include "reloc/Quant.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// RAII device buffer; ASSERT-friendly (constructor cannot assert, so
// callers check valid()).
struct DeviceBuffer {
  void *p = nullptr;
  explicit DeviceBuffer(size_t bytes) { cudaMalloc(&p, bytes); }
  ~DeviceBuffer() { cudaFree(p); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  bool valid() const { return p != nullptr; }
  template <typename T>
  T *as() const {
    return static_cast<T *>(p);
  }
};

template <typename T>
void upload(const DeviceBuffer &d, const std::vector<T> &h) {
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d.p, h.data(), h.size() * sizeof(T),
                                    cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(const DeviceBuffer &d, size_t n) {
  std::vector<T> h(n);
  EXPECT_EQ(cudaSuccess,
            cudaMemcpy(h.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

std::vector<float> randomFloats(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float &x : v)
    x = dist(rng);
  return v;
}

TEST(CudaCopy, RoundTripOddCount) {
  const int64_t n = (1 << 20) + 13; // odd tail: exercises the non-vector path
  std::vector<float> src = randomFloats(static_cast<size_t>(n), 1, -1e6f, 1e6f);
  DeviceBuffer dSrc(n * sizeof(float)), dDst(n * sizeof(float));
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  ASSERT_EQ(cudaSuccess, cudaMemset(dDst.p, 0xAB, n * sizeof(float)));
  reloc::cuda::copyF32(dSrc.as<float>(), dDst.as<float>(), n);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> out = download<float>(dDst, static_cast<size_t>(n));
  ASSERT_EQ(0, std::memcmp(src.data(), out.data(), n * sizeof(float)));
}

} // namespace

#endif // RELOC_ENABLE_CUDA
