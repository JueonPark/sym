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

reloc::BoundPlan transposePlan(int64_t rows, int64_t cols) {
  reloc::BoundPlan b;
  b.extents = {rows, cols};
  b.srcStrides = {1, rows};
  b.dstStrides = {cols, 1};
  b.elementSize = 4;
  b.totalBytes = rows * cols * 4;
  return b;
}

int64_t maxSrcOffset(const reloc::BoundPlan &b) {
  int64_t off = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    off += (b.extents[k] - 1) * b.srcStrides[k];
  return off;
}

// CPU oracle: executeH2D over the same plan, on the same host data.
std::vector<float> cpuRelocate(const reloc::BoundPlan &b,
                               const std::vector<float> &src) {
  std::vector<float> dst(static_cast<size_t>(b.totalBytes / 4), 0.0f);
  reloc::executeH2D(b, src.data(), dst.data());
  return dst;
}

void expectRelocateMatchesCpu(const reloc::BoundPlan &b, uint32_t seed,
                              bool tiled) {
  std::vector<float> src =
      randomFloats(static_cast<size_t>(maxSrcOffset(b) + 1), seed, -1e6f, 1e6f);
  std::vector<float> want = cpuRelocate(b, src);
  DeviceBuffer dSrc(src.size() * 4), dDst(want.size() * 4);
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  ASSERT_EQ(cudaSuccess, cudaMemset(dDst.p, 0xCD, want.size() * 4));
  if (tiled)
    reloc::cuda::relocateF32(b, dSrc.as<float>(), dDst.as<float>());
  else
    reloc::cuda::relocateNaiveF32(b, dSrc.as<float>(), dDst.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> got = download<float>(dDst, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * 4));
}

TEST(CudaRelocateNaive, TransposeMatchesCpu) {
  expectRelocateMatchesCpu(transposePlan(129, 517), 3, /*tiled=*/false);
  expectRelocateMatchesCpu(transposePlan(512, 512), 5, /*tiled=*/false);
}

TEST(CudaRelocateNaive, Rank3MatchesCpu) {
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100};
  b.dstStrides = {198, 33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  expectRelocateMatchesCpu(b, 7, /*tiled=*/false);
}

TEST(CudaRelocateTiled, TransposeShapesMatchCpu) {
  // multiples of 32, remainder tiles both axes, tiny, tall/wide
  expectRelocateMatchesCpu(transposePlan(1024, 1024), 11, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(129, 517), 13, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(32, 32), 17, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(1, 4096), 19, /*tiled=*/true);
}

TEST(CudaRelocateTiled, NonTransposePlanFallsBackBitExact) {
  // 3-D plan: not 2-D-transpose-shaped -> must take the naive fallback and
  // still match the CPU oracle.
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100};
  b.dstStrides = {198, 33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  expectRelocateMatchesCpu(b, 23, /*tiled=*/true);
}

TEST(CudaRelocateTiled, TiledIdenticalToNaive) {
  auto b = transposePlan(801, 333); // remainder tiles, non-square
  std::vector<float> src =
      randomFloats(static_cast<size_t>(maxSrcOffset(b) + 1), 29, -1e6f, 1e6f);
  DeviceBuffer dSrc(src.size() * 4), dA(static_cast<size_t>(b.totalBytes)),
      dB(static_cast<size_t>(b.totalBytes));
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dA.valid());
  ASSERT_TRUE(dB.valid());
  upload(dSrc, src);
  reloc::cuda::relocateF32(b, dSrc.as<float>(), dA.as<float>());
  reloc::cuda::relocateNaiveF32(b, dSrc.as<float>(), dB.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  size_t n = static_cast<size_t>(b.totalBytes / 4);
  std::vector<float> a = download<float>(dA, n), c = download<float>(dB, n);
  ASSERT_EQ(0, std::memcmp(a.data(), c.data(), n * 4));
}

TEST(CudaQuantize, BitExactVsCpuScalar) {
  const int64_t channels = 5, chSize = 1031;
  std::vector<float> src =
      randomFloats(static_cast<size_t>(channels * chSize), 31, -300.f, 300.f);
  // Poke the clamp/NaN/tie lanes.
  src[0] = NAN;
  src[1] = HUGE_VALF;
  src[2] = -HUGE_VALF;
  src[3] = 0.5f;
  src[4] = 2.5f;
  src[5] = -2.5f;
  src[6] = 200.0f;
  src[7] = -200.0f;
  std::vector<float> inv(channels);
  for (int64_t c = 0; c < channels; ++c)
    inv[c] = 0.05f + 0.9f * static_cast<float>(c);
  std::vector<int8_t> want(static_cast<size_t>(channels * chSize), 0);
  reloc::quant::quantizePackF32S8(src.data(), want.data(), channels, chSize,
                                  inv.data(), reloc::quant::Variant::Scalar);
  DeviceBuffer dSrc(src.size() * 4), dInv(inv.size() * 4), dDst(want.size());
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dInv.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  upload(dInv, inv);
  reloc::cuda::quantizeF32S8(dSrc.as<float>(), dDst.as<int8_t>(), channels,
                             chSize, dInv.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<int8_t> got = download<int8_t>(dDst, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size()));
}

} // namespace

#endif // RELOC_ENABLE_CUDA
