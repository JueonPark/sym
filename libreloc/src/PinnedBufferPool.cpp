//===- PinnedBufferPool.cpp - event-gated staging ring --------------------===//

#include "reloc/PinnedBufferPool.h"

namespace reloc {

PinnedBufferPool::PinnedBufferPool(CopyBackend &backend, int nBuffers,
                                   size_t bufferBytes)
    : backend_(backend), bufferBytes_(bufferBytes) {
  if (nBuffers < 1)
    nBuffers = 1;
  buffers_.resize(nBuffers, nullptr);
  events_.resize(nBuffers, 0);
  for (int i = 0; i < nBuffers; ++i)
    buffers_[i] = backend_.allocStaging(bufferBytes);
}

PinnedBufferPool::~PinnedBufferPool() {
  for (void *p : buffers_)
    backend_.freeStaging(p);
}

int PinnedBufferPool::acquire() {
  next_ = (next_ + 1) % nBuffers();
  if (events_[next_] != 0) {
    backend_.waitEvent(events_[next_]);
    events_[next_] = 0;
  }
  return next_;
}

void PinnedBufferPool::drain() {
  for (int i = 0; i < nBuffers(); ++i)
    if (events_[i] != 0) {
      backend_.waitEvent(events_[i]);
      events_[i] = 0;
    }
}

} // namespace reloc
