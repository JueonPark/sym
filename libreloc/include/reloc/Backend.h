//===- Backend.h - copy-backend interface (design decision 3) ---*- C++ -*-===//
//
// The Strategy-4 pipeline (PinnedBufferPool + StreamPipeline) is written once
// against this interface. HostBackend implements it with plain memory + worker
// threads (CI-testable); CudaBackend with pinned alloc + cudaMemcpyAsync +
// events (RELOC_ENABLE_CUDA, tested locally). No exceptions cross this
// interface -- failures are reported by return value.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_BACKEND_H
#define RELOC_BACKEND_H

#include <cstddef>
#include <cstdint>

namespace reloc {

/// Direction of an async copy. For HostToDevice, `dst` is device and `src` is
/// staging; DeviceToHost swaps the roles.
enum class CopyDir { HostToDevice, DeviceToHost };

/// Opaque completion handle. 0 means "no event" (waits/queries are no-ops).
using EventHandle = uint64_t;

class CopyBackend {
public:
  virtual ~CopyBackend() = default;

  /// Allocate a staging buffer of `bytes` (host memory; pinned for CUDA).
  /// Returns nullptr on failure -- never throws.
  virtual void *allocStaging(size_t bytes) = 0;
  virtual void freeStaging(void *p) = 0;

  /// Number of logical queues (== CUDA streams). Always >= 1.
  virtual int numQueues() const = 0;

  /// Enqueue an async copy on `queue`. Copies on one queue complete in
  /// enqueue order.
  virtual void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                         CopyDir dir) = 0;

  /// Record an event after every copy enqueued so far on `queue`. The returned
  /// handle completes once those copies have finished.
  virtual EventHandle recordEvent(int queue) = 0;

  /// Block the calling thread until `ev` completes. ev == 0 returns at once.
  virtual void waitEvent(EventHandle ev) = 0;

  /// Non-blocking: true iff `ev` has completed (or ev == 0).
  virtual bool queryEvent(EventHandle ev) = 0;
};

} // namespace reloc

#endif // RELOC_BACKEND_H
