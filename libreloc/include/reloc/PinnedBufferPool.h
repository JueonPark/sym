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

  /// True iff every staging buffer was allocated. The constructor cannot
  /// report an allocation failure (no exceptions cross CopyBackend), so
  /// callers that must fail by value check this before the first acquire().
  bool valid() const;
  bool usesBackend(const CopyBackend &backend) const {
    return &backend_ == &backend;
  }

  /// Next buffer index (round-robin). Blocks until that buffer's pending event
  /// (if any) has completed, then clears it.
  int acquire();
  /// Checked active prefix for a precomputed schedule. -1 on invalid capacity
  /// or a failed wait; the caller must stop before writing the returned slot.
  int acquire(int activeSlots);

  void *buffer(int index) { return buffers_[index]; }

  /// Record the event that must complete before `index` may be reused.
  void setEvent(int index, EventHandle ev) {
    events_[index] = ev;
    pending_ = true;
    missingEvent_ |= ev == 0;
  }

  /// Mark before submission (copyAsync itself can fail after enqueueing).
  void markPending() { pending_ = true; }
  /// Only after event-proven success or explicit successful quiescence.
  void markComplete() { pending_ = missingEvent_ = false; }

  /// Wait for every still-pending event to complete.
  void drain();

private:
  CopyBackend &backend_;
  std::vector<void *> buffers_;
  std::vector<EventHandle> events_; // 0 == no pending event
  size_t bufferBytes_;
  int next_ = -1;
  bool pending_ = false;
  bool missingEvent_ = false;
};

} // namespace reloc

#endif // RELOC_PINNEDBUFFERPOOL_H
