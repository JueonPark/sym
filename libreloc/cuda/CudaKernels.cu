//===- CudaKernels.cu - R0.2 GPU transform kernels ------------------------===//
//
// Issue #75's kernel set. Launch geometry: 256-thread blocks, grid-stride
// where the body is trivially divisible. Kernels are correctness-first
// (R0.2); bandwidth work belongs to the R0.3 harness (issue #76).
//
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaKernels.h"

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

namespace reloc {
namespace cuda {
namespace {

constexpr int kThreads = 256;

cudaStream_t asStream(void *p) { return static_cast<cudaStream_t>(p); }

int64_t gridFor(int64_t work) { return (work + kThreads - 1) / kThreads; }

__global__ void copyF32Vec4Kernel(const float4 *src, float4 *dst,
                                  int64_t count4) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < count4; i += stride)
    dst[i] = src[i];
}

__global__ void copyF32TailKernel(const float *src, float *dst, int64_t begin,
                                  int64_t count) {
  int64_t i = begin + threadIdx.x;
  if (i < count)
    dst[i] = src[i];
}

constexpr int kMaxRank = 8;

struct Axes {
  int64_t ext[kMaxRank];
  int64_t srcStride[kMaxRank];
  int64_t dstStride[kMaxRank];
  int rank;
};

Axes packAxes(const reloc::BoundPlan &b) {
  assert(b.elementSize == 4 && "fp32 kernels only");
  assert(b.padRegions.empty() && "pads unsupported in R0.2 kernels");
  assert(!b.extents.empty() &&
         b.extents.size() <= static_cast<size_t>(kMaxRank) &&
         "rank out of kernel range");
  Axes a;
  a.rank = static_cast<int>(b.extents.size());
  for (int k = 0; k < a.rank; ++k) {
    a.ext[k] = b.extents[k];
    a.srcStride[k] = b.srcStrides[k];
    a.dstStride[k] = b.dstStrides[k];
  }
  return a;
}

int64_t totalElements(const reloc::BoundPlan &b) {
  int64_t total = 1;
  for (int64_t e : b.extents)
    total *= e;
  return total;
}

// One thread per valid element: decompose the linear index over the
// extents (row-major), gather via srcStrides, scatter via dstStrides —
// the bench/poc_transpose.cu baseline kernel, now a library citizen.
__global__ void relocateNaiveKernel(const float *src, float *dst, Axes a,
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

} // namespace

void copyF32(const float *dSrc, float *dDst, int64_t count, void *stream) {
  const int64_t count4 = count / 4;
  if (count4 > 0) {
    int64_t blocks = std::min<int64_t>(gridFor(count4), 65535);
    copyF32Vec4Kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                        asStream(stream)>>>(
        reinterpret_cast<const float4 *>(dSrc),
        reinterpret_cast<float4 *>(dDst), count4);
  }
  if (count % 4 != 0)
    copyF32TailKernel<<<1, 4, 0, asStream(stream)>>>(dSrc, dDst, count4 * 4,
                                                     count);
}

void relocateNaiveF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                      void *stream) {
  Axes a = packAxes(bound);
  int64_t total = totalElements(bound);
  relocateNaiveKernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                        asStream(stream)>>>(dSrc, dDst, a, total);
}

// Task 3 replaces this forward with the tiled dispatch.
void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream) {
  relocateNaiveF32(bound, dSrc, dDst, stream);
}

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
