// Explicit, bounded ownership for blocking and asynchronous CUDA dispatch.
#ifndef RELOC_DISPATCHRESOURCES_H
#define RELOC_DISPATCHRESOURCES_H
#include "reloc/Dispatch.h"
#include <memory>
#include <vector>

namespace reloc::dispatch {
class Completion {
  friend class Resources;

public:
  struct State;
  explicit Completion(std::shared_ptr<State>);
  ~Completion(); // abandoning a handle drains; never releases pending owners
  Completion(const Completion &) = delete;
  Completion &operator=(const Completion &) = delete;
  TransferOutcome wait();
  std::variant<bool, TransferError> query();
  /// H2D only: GPU-side dependency, without waiting on the host.
  std::optional<TransferError> waitStream(const void *consumer);

private:
  std::shared_ptr<State> state_;
};
struct ContextResourceStats {
  int device = -1;
  bool active = false;
  unsigned backgroundWorkers = 0, streams = 0, borrowedWorkers = 0;
  size_t retainedBytes = 0, liveLimit = 0, retainedLimit = 0;
  std::vector<unsigned long> affinity;
};
struct ResourceStats {
  uint64_t requests = 0, hits = 0, contexts = 0;
  uint64_t deviceAllocations = 0, hostAllocations = 0, frees = 0;
  size_t retainedBytes = 0, deviceBytes = 0, hostBytes = 0;
  unsigned backgroundWorkers = 0;
  uint64_t copyCalls = 0, eventRecords = 0, eventWaits = 0, callerWaits = 0;
  size_t peakLiveBytes = 0;
  int streams = 0;
  bool closed = false, quarantined = false, processValid = true;
  uint64_t admissionWaits = 0, evictions = 0;
  size_t liveBytes = 0;
  unsigned activeContexts = 0, peakActiveContexts = 0, queued = 0;
  size_t retainedLimit = 0, liveLimit = 0;
  unsigned workerLimit = 0, streamLimit = 0, contextLimit = 0,
           perDeviceLimit = 0;
  std::vector<ContextResourceStats> contextDetails;
};

// A bounded pool keyed by device, stream count, CPU affinity and workers.
// The default one-slot pool preserves serial execution; opt in to more slots
// for alternating-device reuse and independent concurrent leases. Aggregate
// byte/worker/stream limits are divided into fixed per-slot quotas. Admission
// is FIFO; an idle incompatible slot is evicted in LRU order. Only allocations
// are cached: data, parameters, requests and caller streams are always
// refreshed. maxRetainedBytes bounds combined host + device scratch BETWEEN
// calls; maxLiveBytes (0 = unlimited) also caps scratch during a call. Oversize
// scratch is ephemeral and released only after completion. Streams/workers have
// their own hard limits. Unknown completion quarantines ALL scratch and buffer
// owners and permanently disables this owner. Inherited owners reject use after
// fork.
class Resources {
public:
  explicit Resources(size_t maxRetainedBytes = size_t(256) << 20,
                     size_t maxLiveBytes = 0,
                     unsigned maxBackgroundWorkers = 64,
                     unsigned maxStreams = 8, unsigned maxContexts = 1,
                     unsigned maxContextsPerDevice = 1,
                     std::optional<uint64_t> acquireTimeoutMs = {});
  ~Resources();
  Resources(const Resources &) = delete;
  Resources &operator=(const Resources &) = delete;
  TransferOutcome execute(DispatchRequest &, int device, int streams,
                          const TransferOptions &,
                          std::shared_ptr<void> owners);
  /// One owned queue, one producer ordering and one completion barrier for a
  /// group. Every scratch block and every buffer owner stays live through the
  /// barrier or quarantine. maxScratchBytes must be positive for this API.
  TransferOutcome execute(GroupRequest &, const TransferOptions &,
                          std::shared_ptr<void> owners);
  /// CPU preparation runs here; GPU work and D2H host completion stay pending.
  /// A submission leases its context until completion. Admission drains an
  /// older submission when needed, never reusing its scratch while pending.
  std::variant<std::shared_ptr<Completion>, TransferError>
  submit(GroupRequest &, const TransferOptions &, std::shared_ptr<void> owners);
  ResourceStats stats() const;
  std::optional<TransferError> clear();
  std::optional<TransferError> close();

private:
  friend class Completion;
  friend struct ResourcePoolTestAccess;
  TransferOutcome executeImpl(DispatchRequest *, GroupRequest *, int device,
                              int streams, const TransferOptions &,
                              std::shared_ptr<void> owners,
                              std::shared_ptr<Completion::State> pending = {});
  struct Impl;
  struct Pool;
  explicit Resources(std::shared_ptr<Impl>);
  std::shared_ptr<Impl> impl_;
  std::shared_ptr<Pool> pool_;
};
} // namespace reloc::dispatch
#endif
