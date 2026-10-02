#include "reloc/DispatchResources.h"
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
    bool device = false, busy = false;
  };
  std::vector<Block> blocks_;
  size_t live_ = 0, retainedLimit_, liveLimit_;
  std::string scratchError_;
  void release(Block &b) {
    if (b.device)
      CudaBackend::freeDevice(b.pointer);
    else
      CudaBackend::freeStaging(b.pointer);
    live_ -= b.bytes;
    ++frees;
    b.pointer = nullptr;
  }
  void *allocate(size_t bytes, bool device) {
    if (failed())
      return nullptr;
    Block *best = nullptr;
    for (auto &b : blocks_)
      if (!b.busy && b.device == device && b.bytes >= bytes &&
          (!best || b.bytes < best->bytes))
        best = &b;
    if (best) {
      best->busy = true;
      return best->pointer;
    }
    // Small shape changes should not pay another CUDA/pinned allocation.
    // Add 12.5% headroom, rounded to 256 KiB, only when both budgets allow it.
    size_t capacity = bytes;
    constexpr size_t quantum = size_t(256) << 10;
    if (bytes >= (size_t(1) << 20) &&
        bytes <= std::numeric_limits<size_t>::max() - bytes / 8 - quantum) {
      size_t grown = ((bytes + bytes / 8 + quantum - 1) / quantum) * quantum;
      if (grown <= retainedLimit_ && (!liveLimit_ || grown <= liveLimit_))
        capacity = grown;
    }
    // Drop idle capacity before growing. Busy blocks stay owned until the
    // entire dispatch completes, including allocations freed by its helpers.
    for (auto &b : blocks_)
      if (!b.busy &&
          (capacity > retainedLimit_ || live_ > retainedLimit_ - capacity ||
           (liveLimit_ &&
            (capacity > liveLimit_ || live_ > liveLimit_ - capacity))))
        release(b);
    blocks_.erase(std::remove_if(blocks_.begin(), blocks_.end(),
                                 [](const Block &b) { return !b.pointer; }),
                  blocks_.end());
    if (liveLimit_ && (capacity > liveLimit_ || live_ > liveLimit_ - capacity))
      capacity = bytes; // rounding must not reject an otherwise fitting request
    if (liveLimit_ && (bytes > liveLimit_ || live_ > liveLimit_ - bytes)) {
      scratchError_ = "typed scratch live-byte limit exceeded";
      return nullptr;
    }
    if (failed())
      return nullptr;
    blocks_.reserve(blocks_.size() + 1); // no throwing bookkeeping after alloc
    void *p = device ? CudaBackend::allocDevice(capacity)
                     : CudaBackend::allocStaging(capacity);
    if (!p) {
      scratchError_ = "typed scratch allocation failed";
      return nullptr;
    }
    blocks_.push_back({p, capacity, device, true});
    live_ += capacity;
    if (device)
      ++deviceAllocations;
    else
      ++hostAllocations;
    return p;
  }

public:
  uint64_t deviceAllocations = 0, hostAllocations = 0, frees = 0;
  ScratchBackend(int streams, int device, size_t retained, size_t live)
      : CudaBackend(streams, device), retainedLimit_(retained),
        liveLimit_(live) {}
  ~ScratchBackend() override {
    for (auto &b : blocks_)
      release(b);
  }
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
  }
  void snapshot(ResourceStats &s) const {
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
  if (!impl_->valid())
    return {
        fail("process_mismatch", "typed resources belong to another process")};
  if (GatherPool::inCallback())
    return {fail("reentrant_call", "cannot dispatch from a gather callback")};
  if (!owners)
    return {
        fail("invalid_options", "typed execution needs strong buffer owners")};
  const bool gpuOnly = options.directDenseUpload &&
                       (request.selected.id == kCudaDequantRelocate ||
                        request.selected.id == kCudaRelocateF32);
  unsigned threads = options.gather || gpuOnly ? 1 : options.gatherThreads;
  if (!threads)
    threads = std::max(1u, std::thread::hardware_concurrency());
  if (threads - 1 > impl_->maxWorkers || streams < 1 ||
      unsigned(streams) > impl_->maxStreams)
    return {fail("resource_limit", "typed stream/worker limit exceeded")};
  auto affinity = detail::currentCpuAffinity();
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (request.consumed)
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
  TransferOutcome out;
  try {
    out.error = executeDispatch(request, c.backend, execution);
  } catch (const std::exception &e) {
    out.error = TransferError{"backend_failure", e.what()};
  } catch (...) {
    out.error = fail("backend_failure", "typed dispatch threw");
  }
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
  out.completion = request.consumed ? TransferCompletion::Complete
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
