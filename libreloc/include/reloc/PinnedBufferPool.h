//===- PinnedBufferPool.h - event-gated staging ring ------------*- C++ -*-===//
//
// A ring of `nBuffers` equal-size staging buffers allocated through a
// CopyBackend. Reuse is event-gated: acquire() blocks on the buffer's
// outstanding event before returning it, so a caller can never overwrite bytes
// that an in-flight copy still reads.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PINNEDBUFFERPOOL_H
#define RELOC_PINNEDBUFFERPOOL_H

#include "reloc/Backend.h"

#include <cstddef>
#include <vector>

namespace reloc {

class PinnedBufferPool {
public:
  PinnedBufferPool(CopyBackend &backend, int nBuffers, size_t bufferBytes);
  ~PinnedBufferPool();

  PinnedBufferPool(const PinnedBufferPool &) = delete;
  PinnedBufferPool &operator=(const PinnedBufferPool &) = delete;

  int nBuffers() const { return static_cast<int>(buffers_.size()); }
  size_t bufferBytes() const { return bufferBytes_; }

  /// Next buffer index (round-robin). Blocks until that buffer's pending event
  /// (if any) has completed, then clears it.
  int acquire();

  void *buffer(int index) { return buffers_[index]; }

  /// Record the event that must complete before `index` may be reused.
  void setEvent(int index, EventHandle ev) { events_[index] = ev; }

  /// Wait for every still-pending event to complete.
  void drain();

private:
  CopyBackend &backend_;
  std::vector<void *> buffers_;
  std::vector<EventHandle> events_; // 0 == no pending event
  size_t bufferBytes_;
  int next_ = -1;
};

} // namespace reloc

#endif // RELOC_PINNEDBUFFERPOOL_H
