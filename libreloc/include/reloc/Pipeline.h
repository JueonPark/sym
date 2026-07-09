//===- Pipeline.h - Strategy-4 chunked pinned/stream pipeline ---*- C++ -*-===//
//
// executeH2DPipelined gathers each dst chunk into an event-recycled pinned
// staging buffer and issues an async copy to the device, round-robining over
// the backend's queues. Output is byte-identical to executeH2D.
// executeD2HPipelined is the symmetric inverse (Task 5). Written once against
// CopyBackend, so HostBackend exercises the whole algorithm in CI.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PIPELINE_H
#define RELOC_PIPELINE_H

#include "reloc/Backend.h"
#include "reloc/Bind.h"

#include <cstddef>

namespace reloc {

/// Strategy 4 (H2D). `deviceDst` is sized bound.totalBytes (a device pointer
/// for CudaBackend, plain host for HostBackend). `nBuffers` in {1,2,4};
/// number of streams = backend.numQueues(). chunkSizeOverride == 0 uses the
/// size heuristic. Byte-identical to executeH2D.
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0);

/// Strategy 4 inverse (D2H). Added in Task 5.
void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0);

} // namespace reloc

#endif // RELOC_PIPELINE_H
