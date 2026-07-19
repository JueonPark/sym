//===- CudaKernels.h - R0.2 GPU transform kernels ---------------*- C++ -*-===//
//
// Host-callable launch wrappers for issue #75's kernel set (Method-B
// transforms, Method-A receive paths, and the EXP-4 calibration kernels).
// Streams are type-erased to void* (nullptr = default stream) so this
// header pulls in no CUDA headers — the CudaBackend.h convention. All
// launches are asynchronous; the caller synchronizes. All device pointers
// come from cudaMalloc (16-byte aligned). Compiled only under
// RELOC_ENABLE_CUDA, for sm_75 (Turing) and sm_89 (Ada).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_CUDAKERNELS_H
#define RELOC_CUDAKERNELS_H

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Bind.h"

#include <cstdint>

namespace reloc {
namespace cuda {

/// `copy_f32`: device->device element copy, the JustCopy bandwidth ceiling
/// (EXP-4 calibration, Colfax-comparable). Vectorized float4 body + scalar
/// tail.
void copyF32(const float *dSrc, float *dDst, int64_t count,
             void *stream = nullptr);

/// `relocate_naive_f32`: GMEM->GMEM strided permute over the BoundPlan's
/// stride sets, one thread per valid element (the hiding-ratio floor for
/// EXP-4). Preconditions (asserted host-side): elementSize == 4, no pads,
/// 1 <= rank <= 8.
void relocateNaiveF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                      void *stream = nullptr);

/// `relocate_f32`: Method B's transform. SMEM-tiled + padded 32x32
/// transpose when the coalesced plan is 2-D-transpose-shaped; falls back
/// to the naive kernel otherwise. Output bit-identical to
/// relocateNaiveF32 either way. Same preconditions.
void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream = nullptr);

/// `quantize_f32_s8` (GPU side, Method B for EXP-2's quantize workloads):
/// contiguous per-channel int8 quantize, BIT-IDENTICAL to the CPU scalar
/// contract (fmaxf-then-fminf clamp so NaN -> -128; __float2int_rn = RNE).
/// dInvScales: device array of `channels` floats.
void quantizeF32S8(const float *dSrc, int8_t *dDst, int64_t channels,
                   int64_t channelSize, const float *dInvScales,
                   void *stream = nullptr);

/// `dequant_s8_f32` (Method A dtype-only receive path): contiguous
/// per-channel int8 -> fp32, out = (float)s8 * scale[channel]. Exact fp32
/// arithmetic. dScales: device array of `channels` floats.
void dequantS8F32(const int8_t *dSrc, float *dDst, int64_t channels,
                  int64_t channelSize, const float *dScales,
                  void *stream = nullptr);

/// `unpack_s4_s8` (Method A r=0.125 receive path): int4 nibble unpack,
/// exact inverse of the CPU packS8S4 on saturated values. Byte i ->
/// dDst[2i] = sign-extended low nibble, dDst[2i+1] = high nibble.
void unpackS4S8(const uint8_t *dSrc, int8_t *dDst, int64_t pairs,
                void *stream = nullptr);

/// `dequant_relocate_s8_f32` (Method A's fused receive path; the R0 exit
/// test's PCIe-hiding candidate on Turing): int8 in the plan's SRC layout
/// -> fp32 in the plan's DST layout, scaled per coalesced outer channel
/// (dScales: device array of extents[0] floats). Preconditions (asserted
/// host-side): elementSize == 4, no pads, 2 <= rank <= 8.
void dequantRelocateS8F32(const BoundPlan &bound, const int8_t *dSrc,
                          float *dDst, const float *dScales,
                          void *stream = nullptr);

/// `scatter_random_f32` (EXP-4's pathological data-dependent case):
/// dDst[dIdx[i]] = dSrc[i]. dIdx: device array of `count` int64 indices;
/// must be a permutation of [0, count) for a deterministic, race-free
/// result (caller's contract).
void scatterRandomF32(const float *dSrc, const int64_t *dIdx, float *dDst,
                      int64_t count, void *stream = nullptr);

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
#endif // RELOC_CUDAKERNELS_H
