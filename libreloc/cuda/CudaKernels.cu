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

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

// Coalesced 2-D transpose: read a 32x32 tile of `in` coalesced, write it
// back transposed and coalesced; +1 padding kills SMEM bank conflicts.
// `in` is inRows x inCols row-major; out[c][r] = in[r][c].
__global__ void transposeTiledKernel(const float *in, float *out,
                                     int64_t inRows, int64_t inCols) {
  __shared__ float tile[kTile][kTile + 1];
  int64_t x = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.x;
  int64_t y = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.y;
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inCols && y + j < inRows)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * inCols + x];
  __syncthreads();
  x = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.x; // out col
  y = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.y; // out row
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inRows && y + j < inCols)
      out[(y + j) * inRows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// relocate_f32's fast path applies when the coalesced plan is exactly a
// 2-D transpose: dst [R,C] dense row-major, src read column-major.
bool isTranspose2D(const reloc::BoundPlan &b) {
  return b.extents.size() == 2 && b.srcStrides[0] == 1 &&
         b.srcStrides[1] == b.extents[0] && b.dstStrides[1] == 1 &&
         b.dstStrides[0] == b.extents[1];
}

__global__ void quantizeF32S8Kernel(const float *src, int8_t *dst,
                                    int64_t channelSize, const float *invScales,
                                    int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  float y = src[i] * invScales[i / channelSize];
  y = fmaxf(y, -128.0f); // NaN -> -128, matching the CPU quantOne contract
  y = fminf(y, 127.0f);
  dst[i] = static_cast<int8_t>(__float2int_rn(y));
}

__global__ void dequantS8F32Kernel(const int8_t *src, float *dst,
                                   int64_t channelSize, const float *scales,
                                   int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  dst[i] = static_cast<float>(src[i]) * scales[i / channelSize];
}

__global__ void unpackS4S8Kernel(const uint8_t *src, int8_t *dst,
                                 int64_t pairs) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= pairs)
    return;
  const uint8_t b = src[i];
  dst[2 * i] = static_cast<int8_t>(static_cast<uint8_t>(b << 4)) >> 4;
  dst[2 * i + 1] = static_cast<int8_t>(b) >> 4;
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

void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream) {
  if (!isTranspose2D(bound)) {
    relocateNaiveF32(bound, dSrc, dDst, stream);
    return;
  }
  // Plan dst[r][c] = src[c*R + r]: view src as (C x R) row-major `in`,
  // dst as out with out[r][c] = in[c][r] -> launch with inRows=C, inCols=R.
  const int64_t R = bound.extents[0], C = bound.extents[1];
  dim3 block(kTile, kBlockRows);
  dim3 grid(static_cast<unsigned>((R + kTile - 1) / kTile),
            static_cast<unsigned>((C + kTile - 1) / kTile));
  transposeTiledKernel<<<grid, block, 0, asStream(stream)>>>(dSrc, dDst,
                                                             /*inRows=*/C,
                                                             /*inCols=*/R);
}

void quantizeF32S8(const float *dSrc, int8_t *dDst, int64_t channels,
                   int64_t channelSize, const float *dInvScales, void *stream) {
  int64_t total = channels * channelSize;
  quantizeF32S8Kernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                        asStream(stream)>>>(dSrc, dDst, channelSize, dInvScales,
                                            total);
}

void dequantS8F32(const int8_t *dSrc, float *dDst, int64_t channels,
                  int64_t channelSize, const float *dScales, void *stream) {
  int64_t total = channels * channelSize;
  dequantS8F32Kernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                       asStream(stream)>>>(dSrc, dDst, channelSize, dScales,
                                           total);
}

void unpackS4S8(const uint8_t *dSrc, int8_t *dDst, int64_t pairs,
                void *stream) {
  unpackS4S8Kernel<<<static_cast<unsigned>(gridFor(pairs)), kThreads, 0,
                     asStream(stream)>>>(dSrc, dDst, pairs);
}

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
