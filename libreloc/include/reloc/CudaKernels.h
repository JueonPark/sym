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

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
#endif // RELOC_CUDAKERNELS_H
