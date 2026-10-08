#include "reloc/DispatchResources.h"
#include "DispatchGroupInternal.h"
#include "TransferResourcePolicy.h"
#include "reloc/GatherPool.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace reloc::dispatch {
namespace {
TransferError fail(const char *code, const char *message) {
  return {code, message};
}
#ifdef RELOC_ENABLE_CUDA
// A freed scratch block is NOT made available within the same dispatch: a
// launch on another queue may still be reading it. Only finish() recycles it.
class ScratchBackend : public CudaBackend {
  struct Block {
    void *pointer = nullptr;
    size_t bytes = 0;
    bool device = false, pinned = false, busy = false;
  };
  std::vector<Block> blocks_;
  size_t live_ = 0, retainedLimit_, liveLimit_;
  std::string scratchError_;
  TransferOptions options_;
  size_t requestPeak_ = 0;
  size_t liveLimit() const {
    if (!options_.maxScratchBytes)
      return liveLimit_;
    return liveLimit_ ? std::min(liveLimit_, options_.maxScratchBytes)
                      : options_.maxScratchBytes;
  }
  void release(Block &b) {
    if (b.device)
      CudaBackend::freeDevice(b.pointer);
    else if (b.pinned)
      CudaBackend::freeStaging(b.pointer);
    else
      std::free(b.pointer);
    live_ -= b.bytes;
    ++frees;
    b.pointer = nullptr;
  }
  void *allocate(size_t bytes, bool device) {
    if (failed())
      return nullptr;
    const size_t limit = liveLimit();
    size_t busyBytes = 0;
    for (const auto &b : blocks_)
      if (b.busy)
        busyBytes += b.bytes;
    auto fitsRetention = [&](size_t capacity) {
      return capacity <= retainedLimit_ &&
             busyBytes <= retainedLimit_ - capacity;
    };
    auto decision = selectStaging(options_, bytes, fitsRetention(bytes));
    const bool pinned = !device && decision.pinned;
    auto record = [&](size_t capacity, bool reused) {
      if (!device && options_.staging) {
        decision.capacityBytes = capacity;
        decision.buffers = 1;
        decision.reused = reused;
        decision.retentionEligible = fitsRetention(capacity);
        options_.staging->push_back(decision);
      }
    };
    Block *best = nullptr;
    for (auto &b : blocks_)
      if (!b.busy && b.device == device && b.pinned == pinned &&
          b.bytes >= bytes &&
          (!pinned || options_.pinning != PinningPolicy::Auto ||
           fitsRetention(b.bytes)) &&
          (!best || b.bytes < best->bytes))
        best = &b;
    if (best) {
      best->busy = true;
      record(best->bytes, true);
      return best->pointer;
    }
    // Small shape changes should not pay another CUDA/pinned allocation.
    // Add 12.5% headroom, rounded to 256 KiB, only when both budgets allow it.
    // Pinning above used requested wire bytes, not this spare capacity.
    size_t capacity = bytes;
    constexpr size_t quantum = size_t(256) << 10;
    if (bytes >= (size_t(1) << 20) &&
        bytes <= std::numeric_limits<size_t>::max() - bytes / 8 - quantum) {
      size_t grown = ((bytes + bytes / 8 + quantum - 1) / quantum) * quantum;
      if (fitsRetention(grown) && (!limit || grown <= limit))
        capacity = grown;
    }
    // Drop idle capacity before growing. Busy blocks stay owned until the
    // entire dispatch completes, including allocations freed by its helpers.
    for (auto &b : blocks_)
      if (!b.busy &&
          (capacity > retainedLimit_ || live_ > retainedLimit_ - capacity ||
           (limit && (capacity > limit || live_ > limit - capacity))))
        release(b);
    blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                                 [](const Block &b) { return !b.pointer; }),
                  blocks_.end());
    if (limit && (capacity > limit || live_ > limit - capacity))
      capacity = bytes; // rounding must not reject an otherwise fitting request
    if (limit && (bytes > limit || live_ > limit - bytes)) {
      scratchError_ = "typed scratch live-byte limit exceeded";
      return nullptr;
    }
    if (failed())
      return nullptr;
    blocks_.reserve(blocks_.size() + 1); // no throwing bookkeeping after alloc
    void *p = device   ? CudaBackend::allocDevice(capacity)
              : pinned ? CudaBackend::allocStaging(capacity)
                       : std::malloc(capacity);
    if (!p) {
      scratchError_ = "typed scratch allocation failed";
      return nullptr;
    }
    blocks_.push_back({p, capacity, device, pinned, true});
    live_ += capacity;
    requestPeak_ = std::max(requestPeak_, live_);
    peakLiveBytes = std::max(peakLiveBytes, live_);
    if (device)
      ++deviceAllocations;
    else
      ++hostAllocations;
    record(capacity, false);
    return p;
  }

public:
  uint64_t deviceAllocations = 0, hostAllocations = 0, frees = 0;
  uint64_t copyCalls = 0, eventRecords = 0, eventWaits = 0, callerWaits = 0;
  size_t peakLiveBytes = 0;
  ScratchBackend(int streams, int device, size_t retained, size_t live)
      : CudaBackend(streams, device), retainedLimit_(retained),
        liveLimit_(live) {}
  ~ScratchBackend() override {
    for (auto &b : blocks_)
      release(b);
  }
  void begin(const TransferOptions &options) {
    options_ = options;
    // A smaller per-group live budget also excludes old idle capacity.
    const auto limit = liveLimit();
    for (auto &b : blocks_)
      if (limit && live_ > limit && !b.busy)
        release(b);
    blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                                 [](const Block &b) { return !b.pointer; }),
                  blocks_.end());
    requestPeak_ = live_;
  }
  size_t requestPeak() const { return requestPeak_; }
  void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                 CopyDir dir) override {
    ++copyCalls;
    CudaBackend::copyAsync(queue, dst, src, bytes, dir);
  }
  EventHandle recordEvent(int queue) override {
    ++eventRecords;
    return CudaBackend::recordEvent(queue);
  }
  void waitEvent(EventHandle event) override {
    ++eventWaits;
    CudaBackend::waitEvent(event);
  }
  bool waitStream(const void *stream) override {
    ++callerWaits;
    return CudaBackend::waitStream(stream);
  }
  void addMetrics(ResourceStats &s) const {
    s.copyCalls += copyCalls;
    s.eventRecords += eventRecords;
    s.eventWaits += eventWaits;
    s.callerWaits += callerWaits;
    s.peakLiveBytes = std::max(s.peakLiveBytes, peakLiveBytes);
  }
  void endReporting() { options_.staging = nullptr; }
  void *allocDevice(size_t bytes) override { return allocate(bytes, true); }
  void freeDevice(void *) override {}
  void *allocStaging(size_t bytes) override { return allocate(bytes, false); }
  void freeStaging(void *) override {}
  bool failed() const override {
    return !scratchError_.empty() || CudaBackend::failed();
  }
  const std::string &error() const override {
    return scratchError_.empty() ? CudaBackend::error() : scratchError_;
  }
  void finish() {
    for (auto &b : blocks_) {
      b.busy = false;
      if (live_ > retainedLimit_)
        release(b);
    }
    blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                                 [](const Block &b) { return !b.pointer; }),
                  blocks_.end());
    // No request's caller stream or borrowed pool remains in an idle owner.
    options_ = {};
  }
  void snapshot(ResourceStats &s) const {
    addMetrics(s);
    s.retainedBytes = live_;
    s.deviceBytes = s.hostBytes = 0;
    for (auto &b : blocks_)
      (b.device ? s.deviceBytes : s.hostBytes) += b.bytes;
  }
};
struct Context : detail::QuarantineNode {
  ScratchBackend backend;
  std::unique_ptr<GatherPool> workers;
  std::vector<unsigned long> affinity;
  unsigned threads;
  std::shared_ptr<void> owners;
  Context(int streams, int device, size_t retained, size_t live,
          unsigned threads, std::vector<unsigned long> affinity)
      : backend(streams, device, retained, live), affinity(std::move(affinity)),
        threads(threads) {
    if (threads > 1 && !backend.failed())
      workers = std::make_unique<GatherPool>(threads);
  }
};
#endif
} // namespace

struct Resources::Impl {
  detail::ProcessIdentity process = detail::captureProcessIdentity();
  mutable std::mutex mu;
  size_t retainedLimit, liveLimit;
  unsigned maxWorkers, maxStreams;
  ResourceStats counters;
#ifdef RELOC_ENABLE_CUDA
  std::unique_ptr<Context> context;
  void retire() {
    if (context) {
      context->backend.addMetrics(counters);
      counters.deviceAllocations += context->backend.deviceAllocations;
      counters.hostAllocations += context->backend.hostAllocations;
      // All retained allocations are freed with the retired context.
      counters.frees +=
          context->backend.deviceAllocations + context->backend.hostAllocations;
      context.reset();
    }
  }
#endif
  bool valid() const { return process == detail::currentProcessIdentity(); }
  Impl(size_t retained, size_t live, unsigned workers, unsigned streams)
      : retainedLimit(retained), liveLimit(live), maxWorkers(workers),
        maxStreams(streams) {}
};
Resources::Resources(size_t retained, size_t live, unsigned workers,
                     unsigned streams)
    : impl_(std::make_unique<Impl>(retained, live, workers, streams)) {
  if (!streams)
    throw std::invalid_argument("max_streams must be positive");
}
Resources::~Resources() {
  if (!impl_->valid())
    (void)impl_.release(); // no inherited mutex/CUDA cleanup
}
ResourceStats Resources::stats() const {
  if (!impl_->valid()) {
    ResourceStats s;
    s.processValid = false;
    return s;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto s = impl_->counters;
#ifdef RELOC_ENABLE_CUDA
  if (impl_->context) {
    const auto &c = *impl_->context;
    c.backend.snapshot(s);
    s.deviceAllocations += c.backend.deviceAllocations;
    s.hostAllocations += c.backend.hostAllocations;
    s.frees += c.backend.frees;
    s.backgroundWorkers = c.threads - 1;
    s.streams = c.backend.numQueues();
  }
#endif
  return s;
}
std::optional<TransferError> Resources::clear() {
  if (!impl_->valid())
    return fail("process_mismatch",
                "typed resources belong to another process");
  if (GatherPool::inCallback())
    return fail("reentrant_call", "cannot clear from a gather callback");
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->counters.quarantined)
    return fail("completion_unknown", "typed resources quarantined");
  if (impl_->counters.closed)
    return fail("resources_closed", "typed resources closed");
#ifdef RELOC_ENABLE_CUDA
  impl_->retire();
#endif
  return std::nullopt;
}
std::optional<TransferError> Resources::close() {
  if (!impl_->valid())
    return fail("process_mismatch",
                "typed resources belong to another process");
  if (GatherPool::inCallback())
    return fail("reentrant_call", "cannot close from a gather callback");
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->counters.closed = true;
  if (impl_->counters.quarantined)
    return fail("completion_unknown", "typed resources quarantined");
#ifdef RELOC_ENABLE_CUDA
  impl_->retire();
#endif
  return std::nullopt;
}
TransferOutcome Resources::execute(DispatchRequest &request, int device,
                                   int streams, const TransferOptions &options,
                                   std::shared_ptr<void> owners) {
  return executeImpl(&request, nullptr, device, streams, options,
                     std::move(owners));
}
TransferOutcome Resources::execute(GroupRequest &group,
                                   const TransferOptions &options,
                                   std::shared_ptr<void> owners) {
  if (!options.maxScratchBytes)
    return {
        fail("invalid_options", "group scratch byte limit must be positive")};
  if (group.device < 0)
    return {fail("backend_mismatch", "groups require a CUDA device")};
  return executeImpl(nullptr, &group, group.device, 1, options,
                     std::move(owners));
}
TransferOutcome Resources::executeImpl(DispatchRequest *request,
                                       GroupRequest *group, int device,
                                       int streams,
                                       const TransferOptions &options,
                                       std::shared_ptr<void> owners) {
  if (!impl_->valid())
    return {
        fail("process_mismatch", "typed resources belong to another process")};
  if (GatherPool::inCallback())
    return {fail("reentrant_call", "cannot dispatch from a gather callback")};
  if (!owners)
    return {
        fail("invalid_options", "typed execution needs strong buffer owners")};
  auto gpuRow = [](const DispatchRequest &r) {
    return r.selected.id == kCudaDequantRelocate ||
           r.selected.id == kCudaRelocateF32;
  };
  const bool gpuOnly =
      (group || options.directDenseUpload) &&
      (request ? gpuRow(*request)
               : std::all_of(group->items.begin(), group->items.end(),
                             [&](const auto &item) {
                               const auto *r =
                                   std::get_if<DispatchRequest>(&item);
                               return r && gpuRow(*r);
                             }));
  unsigned threads = options.gather || gpuOnly ? 1 : options.gatherThreads;
  if (!threads)
    threads = std::max(1u, std::thread::hardware_concurrency());
  if (threads - 1 > impl_->maxWorkers || streams < 1 ||
      unsigned(streams) > impl_->maxStreams)
    return {fail("resource_limit", "typed stream/worker limit exceeded")};
  auto affinity = detail::currentCpuAffinity();
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (request ? request->consumed : group->consumed)
    return {fail("already_executed", "typed request already executed")};
  auto &s = impl_->counters;
  if (s.quarantined)
    return {fail("completion_unknown", "typed resources quarantined")};
  if (s.closed)
    return {fail("resources_closed", "typed resources closed")};
  ++s.requests;
#ifdef RELOC_ENABLE_CUDA
  if (impl_->context && (impl_->context->backend.device() != device ||
                         impl_->context->backend.numQueues() != streams ||
                         impl_->context->threads != threads ||
                         impl_->context->affinity != affinity))
    impl_->retire();
  if (!impl_->context) {
    impl_->context = std::make_unique<Context>(
        streams, device, impl_->retainedLimit, impl_->liveLimit, threads,
        std::move(affinity));
    ++s.contexts;
  } else
    ++s.hits;
  auto &c = *impl_->context;
  c.owners = std::move(owners);
  auto execution = options;
  if (!execution.gather)
    execution.gather = c.workers.get();
  c.backend.begin(execution);
  TransferOutcome out;
  ResourceStats before;
  c.backend.addMetrics(before);
  try {
    out.error = request
                    ? executeDispatch(*request, c.backend, execution)
                    : group_detail::executeGroup(*group, c.backend, execution);
  } catch (const std::exception &e) {
    out.error = TransferError{"backend_failure", e.what()};
  } catch (...) {
    out.error = fail("backend_failure", "typed dispatch threw");
  }
  if (group) {
    ResourceStats after;
    c.backend.addMetrics(after);
    group->report.copyCalls = after.copyCalls - before.copyCalls;
    group->report.eventRecords = after.eventRecords - before.eventRecords;
    group->report.eventWaits = after.eventWaits - before.eventWaits;
    group->report.callerWaits = after.callerWaits - before.callerWaits;
    group->report.scratchPeakBytes = c.backend.requestPeak();
  }
  // Error quarantine must not keep a pointer into the caller's request report.
  c.backend.endReporting();
  // Healthy dispatch waits for each submitted copy/kernel. Errors and throwing
  // helpers require an independent completion proof before freeing anything.
  if (out.error && c.backend.quiesce() == QueueCompletion::Unknown) {
    out.completion = TransferCompletion::Unknown;
    c.backend.snapshot(s);
    s.deviceAllocations += c.backend.deviceAllocations;
    s.hostAllocations += c.backend.hostAllocations;
    s.frees += c.backend.frees;
    s.backgroundWorkers = c.threads - 1;
    s.streams = streams;
    s.quarantined = true;
    detail::quarantine(std::move(impl_->context));
    out.error = fail("completion_unknown",
                     "typed dispatch and its owners were quarantined");
    return out;
  }
  out.completion = (request ? request->consumed : group->consumed)
                       ? TransferCompletion::Complete
                       : TransferCompletion::NotLaunched;
  c.owners.reset();
  if (out.error)
    impl_->retire();
  else {
    c.backend.finish();
    if (c.backend.failed()) {
      out.error = TransferError{"backend_failure", c.backend.error()};
      impl_->retire();
    }
  }
  return out;
#else
  (void)device;
  return {fail("backend_failure", "typed CUDA resources require a CUDA build")};
#endif
}
} // namespace reloc::dispatch
