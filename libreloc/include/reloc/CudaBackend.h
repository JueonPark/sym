//===- CudaBackend.h - CopyBackend over the CUDA Runtime API ----*- C++ -*-===//
//
// Pinned staging (cudaHostAlloc), cudaMemcpyAsync on non-blocking streams, and
// cudaEvent completion. Compiled only when RELOC_ENABLE_CUDA; its tests run
// locally on the desktop GPU, never in CI. CUDA handles are erased to void* so
// this header pulls in no CUDA headers (keeps the MLIR-free include scan and
// non-CUDA translation units clean). Every CUDA call's status is checked and
// the first failure is recorded in the sticky error state (R2, issue #146);
// resources are created and used under the backend's device.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_CUDABACKEND_H
#define RELOC_CUDABACKEND_H

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace reloc {

class CudaBackend : public CopyBackend {
public:
  /// `device` < 0 uses the current device at construction and pins it.
  explicit CudaBackend(int numStreams = 2, int device = -1);
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
  bool waitStream(const void *externalStream) override;
  bool failed() const override { return !error_.empty(); }
  const std::string &error() const override { return error_; }
  int device() const override { return device_; }

private:
  /// Record the first failing status; returns true when `status` is success.
  bool check(int status, const char *what);

  std::vector<void *> streams_;                    // cudaStream_t erased
  std::unordered_map<EventHandle, void *> events_; // handle -> cudaEvent_t
  uint64_t nextEvent_ = 1;
  int device_ = -1;
  std::string error_;
};

/// Resolve the CUDA device that owns `pointer` (cudaPointerGetAttributes).
/// Returns false with `error` set for host or unknown memory, so callers can
/// prove a declared device ordinal instead of trusting raw addresses.
bool cudaPointerDevice(const void *pointer, int &device, std::string &error);

} // namespace reloc

#endif // RELOC_ENABLE_CUDA
#endif // RELOC_CUDABACKEND_H
