// Explicit, bounded ownership for blocking CUDA typed dispatch.
#ifndef RELOC_DISPATCHRESOURCES_H
#define RELOC_DISPATCHRESOURCES_H
#include "reloc/Dispatch.h"
#include <memory>

namespace reloc::dispatch {
struct ResourceStats {
  uint64_t requests = 0, hits = 0, contexts = 0;
  uint64_t deviceAllocations = 0, hostAllocations = 0, frees = 0;
  size_t retainedBytes = 0, deviceBytes = 0, hostBytes = 0;
  unsigned backgroundWorkers = 0;
  int streams = 0;
  bool closed = false, quarantined = false, processValid = true;
};

// One exclusively leased context per owner. Concurrent calls serialize; use
// separate owners for independent concurrency. A device/stream/CPU-affinity or
// worker-count change retires the old context. Only scratch allocations are
// cached: data, parameters, requests and caller streams are always refreshed.
// maxRetainedBytes bounds combined host + device scratch BETWEEN calls;
// maxLiveBytes (0 = unlimited) also caps scratch during a call. Oversize
// scratch is ephemeral and released only after completion. Streams/workers have
// their own hard limits. Unknown completion quarantines ALL scratch and buffer
// owners and permanently disables this owner. Inherited owners reject use after
// fork.
class Resources {
public:
  explicit Resources(size_t maxRetainedBytes = size_t(256) << 20,
                     size_t maxLiveBytes = 0,
                     unsigned maxBackgroundWorkers = 64,
                     unsigned maxStreams = 8);
  ~Resources();
  Resources(const Resources &) = delete;
  Resources &operator=(const Resources &) = delete;
  TransferOutcome execute(DispatchRequest &, int device, int streams,
                          const TransferOptions &,
                          std::shared_ptr<void> owners);
  ResourceStats stats() const;
  std::optional<TransferError> clear();
  std::optional<TransferError> close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace reloc::dispatch
#endif
