// Private admission/lease layer over single-context typed execution.
#ifndef RELOC_DISPATCHRESOURCEPOOL_H
#define RELOC_DISPATCHRESOURCEPOOL_H
#include "TransferResourcePolicy.h"
#include "reloc/DispatchResources.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

namespace reloc::dispatch {
struct ScratchGauge {
  std::atomic<size_t> live{0}, peak{0};
  void add(size_t bytes) {
    const size_t now = live.fetch_add(bytes) + bytes;
    size_t old = peak.load();
    while (old < now && !peak.compare_exchange_weak(old, now)) {
    }
  }
  void remove(size_t bytes) { live.fetch_sub(bytes); }
};

struct Resources::Pool : std::enable_shared_from_this<Resources::Pool> {
  struct Key {
    int device, streams;
    unsigned threads, borrowedWorkers;
    std::vector<unsigned long> affinity;
    bool operator==(const Key &other) const;
  };
  struct Slot {
    std::shared_ptr<Resources> resources;
    std::optional<Key> key;
    std::weak_ptr<Completion> pending;
    ResourceStats completed;
    bool active = false;
    uint64_t lastUse = 0;
    size_t retained, live;
    unsigned workers, streams;
  };
  struct Lease {
    std::shared_ptr<Pool> pool;
    std::shared_ptr<Resources> resources;
    size_t slot;
    Lease(std::shared_ptr<Pool> p, std::shared_ptr<Resources> r, size_t s)
        : pool(std::move(p)), resources(std::move(r)), slot(s) {}
    ~Lease();
  };
  detail::ProcessIdentity process = detail::captureProcessIdentity();
  mutable std::mutex mu;
  std::condition_variable cv;
  std::vector<Slot> slots;
  std::deque<uint64_t> waiters;
  std::shared_ptr<ScratchGauge> gauge = std::make_shared<ScratchGauge>();
  unsigned perDevice, active = 0, peakActive = 0;
  uint64_t clock = 0, waits = 0, evictions = 0;
  std::optional<uint64_t> timeout;
  bool closed = false, clearing = false, quarantined = false;

  Pool(size_t retained, size_t live, unsigned workers, unsigned streams,
       unsigned contexts, unsigned perDevice, std::optional<uint64_t> timeout);
  bool valid() const { return process == detail::currentProcessIdentity(); }
  std::variant<std::shared_ptr<Lease>, TransferError> acquire(Key);
  void submitted(const Lease &, const std::shared_ptr<Completion> &);
  void release(size_t);
  ResourceStats stats() const;
  std::optional<TransferError> drain(bool close);
  // Impl is defined in DispatchResources.cpp; create private one-slot engines
  // there so the admission layer cannot reach mutable CUDA context internals.
  std::shared_ptr<Resources> create(const Slot &);
};
} // namespace reloc::dispatch
#endif
