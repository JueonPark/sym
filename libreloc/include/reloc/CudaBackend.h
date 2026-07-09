//===- CudaBackend.h - CopyBackend over the CUDA Runtime API ----*- C++ -*-===//
//
// Pinned staging (cudaHostAlloc), cudaMemcpyAsync on non-blocking streams, and
// cudaEvent completion. Compiled only when RELOC_ENABLE_CUDA; its tests run
// locally on the desktop GPU, never in CI. CUDA handles are erased to void* so
// this header pulls in no CUDA headers (keeps the MLIR-free include scan and
// non-CUDA translation units clean).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_CUDABACKEND_H
#define RELOC_CUDABACKEND_H

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Backend.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace reloc {

class CudaBackend : public CopyBackend {
public:
  explicit CudaBackend(int numStreams = 2);
  ~CudaBackend() override;

  CudaBackend(const CudaBackend &) = delete;
  CudaBackend &operator=(const CudaBackend &) = delete;

  void *allocStaging(size_t bytes) override;
  void freeStaging(void *p) override;
  int numQueues() const override { return static_cast<int>(streams_.size()); }
  void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                 CopyDir dir) override;
  EventHandle recordEvent(int queue) override;
  void waitEvent(EventHandle ev) override;
  bool queryEvent(EventHandle ev) override;

private:
  std::vector<void *> streams_;                    // cudaStream_t erased
  std::unordered_map<EventHandle, void *> events_; // handle -> cudaEvent_t
  uint64_t nextEvent_ = 1;
};

} // namespace reloc

#endif // RELOC_ENABLE_CUDA
#endif // RELOC_CUDABACKEND_H
