//===- PinnedBufferPool.cpp - event-gated staging ring --------------------===//

#include "reloc/PinnedBufferPool.h"

#include <limits>
#include <stdexcept>

namespace reloc {

PinnedBufferPool::PinnedBufferPool(CopyBackend &backend, int nBuffers,
                                   size_t bufferBytes)
    : backend_(backend), bufferBytes_(bufferBytes) {
  if (nBuffers < 1)
    nBuffers = 1;
  if (bufferBytes == 0 ||
      bufferBytes > std::numeric_limits<size_t>::max() / size_t(nBuffers))
    throw std::invalid_argument("invalid staging capacity");
  buffers_.resize(nBuffers, nullptr);
  events_.resize(nBuffers, 0);
  try {
    for (int i = 0; i < nBuffers && !backend_.failed(); ++i) {
      buffers_[i] = backend_.allocStaging(bufferBytes);
      if (!buffers_[i])
        break;
    }
  } catch (...) {
    for (void *p : buffers_)
      if (p)
        backend_.freeStaging(p);
    throw;
  }
}

PinnedBufferPool::~PinnedBufferPool() {
  // Normally drain() already proved completion. On an exceptional raw-pipeline
  // exit do not free bytes that may still be read by DMA. Owned transfer APIs
  // retain the entire pool/backend/owners instead of reaching this fallback.
  if (pending_) {
    try {
      if (backend_.quiesce() != QueueCompletion::Complete)
        return;
    } catch (...) {
      return;
    }
  }
  for (void *p : buffers_)
    if (p)
      backend_.freeStaging(p);
}

bool PinnedBufferPool::valid() const {
  for (void *p : buffers_)
    if (p == nullptr)
      return false;
  return true;
}

int PinnedBufferPool::acquire() { return acquire(nBuffers()); }

int PinnedBufferPool::acquire(int activeSlots) {
  if (activeSlots < 1 || activeSlots > nBuffers() || !valid() ||
      backend_.failed() || missingEvent_)
    return -1;
  next_ = (next_ + 1) % activeSlots;
  if (events_[next_] != 0) {
    backend_.waitEvent(events_[next_]);
    events_[next_] = 0;
  }
  return backend_.failed() ? -1 : next_;
}

void PinnedBufferPool::drain() {
  for (int i = 0; i < nBuffers(); ++i)
    if (events_[i] != 0) {
      backend_.waitEvent(events_[i]);
      events_[i] = 0;
    }
  if (!backend_.failed() && !missingEvent_)
    markComplete();
}

} // namespace reloc
