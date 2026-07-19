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

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
