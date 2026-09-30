#include "TransferResourcePolicy.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/TransferResources.h"
#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#endif

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>

namespace reloc {
namespace {
using detail::CacheRequest;
using detail::ResourceKey;
using detail::TransferContextAccess;
enum class ResourceState { Building, Leased, Idle, Retiring, Quarantined };
enum class ClearResult { Complete, Closed, GenerationExhausted, Quarantined };

std::optional<TransferError> clearError(ClearResult result) {
  switch (result) {
  case ClearResult::Complete:
    return std::nullopt;
  case ClearResult::Closed:
    return TransferError{"resources_closed", "resource cache is closed"};
  case ClearResult::GenerationExhausted:
    return TransferError{"resource_limit", "resource generation exhausted"};
  case ClearResult::Quarantined:
    return TransferError{"completion_unknown",
                         "quarantined resources remain charged"};
  }
  std::terminate();
}

struct Entry {
  explicit Entry(const CacheRequest &r)
      : key(r.key), capacity(r.slotCapacity), bytes(r.bytes),
        workers(r.workers), cacheable(r.cacheable) {}
  ResourceKey key;
  size_t capacity, bytes;
  unsigned workers;
  bool cacheable;
  size_t allocated = 0, allocatedWorkers = 0, streams = 0, events = 0;
  uint64_t generation = 0, lastUse = 0;
  ResourceState state = ResourceState::Building;
  std::unique_ptr<TransferContext> context;
};

// No allocations for the recursive-call guard, including failure cleanup.
struct CallScope {
  static thread_local CallScope *current;
  const void *cache;
  CallScope *previous = current;
  explicit CallScope(const void *cache) : cache(cache) { current = this; }
  ~CallScope() { current = previous; }
  static bool contains(const void *cache) {
    for (auto *p = current; p; p = p->previous)
      if (p->cache == cache)
        return true;
    return GatherPool::inCallback();
  }
};
thread_local CallScope *CallScope::current = nullptr;

bool exceeds(size_t used, size_t added, size_t limit) {
  return added > limit || used > limit - added;
}
size_t live(const TransferResourceUsage &u) {
  return u.allocatedStagingBytes + u.reservedStagingBytes;
}
size_t retained(const TransferResourceUsage &u) {
  return u.retainedAllocatedBytes + u.retainedReservedBytes;
}
size_t workers(const TransferResourceUsage &u) {
  return u.backgroundWorkers + u.reservedWorkers;
}

struct CacheState : std::enable_shared_from_this<CacheState> {
  CacheState(TransferResourceLimits limits, TransferBackendFactory factory)
      : limits(limits), factory(std::move(factory)) {}
  std::mutex mu;
  std::condition_variable cv;
  const TransferResourceLimits limits;
  TransferBackendFactory factory;
  std::list<std::unique_ptr<Entry>> entries;
  TransferResourceStats counters;
  bool closed = false;
  uint64_t generation = 0, clock = 0;

  struct Waiter {
    CacheState &state;
    Waiter *prev, *next = nullptr;
    bool linked = true;
    explicit Waiter(CacheState &s) : state(s), prev(s.tail) {
      if (prev)
        prev->next = this;
      else
        state.head = this;
      state.tail = this;
      ++state.counters.waiters;
    }
    void remove() {
      if (!linked)
        return;
      if (prev)
        prev->next = next;
      else
        state.head = next;
      if (next)
        next->prev = prev;
      else
        state.tail = prev;
      --state.counters.waiters;
      linked = false;
      state.cv.notify_all();
    }
    ~Waiter() { remove(); } // caller's unique_lock still holds mu
  };
  Waiter *head = nullptr, *tail = nullptr;

  TransferResourceUsage usage(const ResourceKey *device = nullptr,
                              bool onlyQuarantine = false) const {
    TransferResourceUsage u;
    for (const auto &entry : entries) {
      const auto &e = *entry;
      if ((device && !e.key.sameDevice(*device)) ||
          (onlyQuarantine && e.state != ResourceState::Quarantined))
        continue;
      ++u.contexts;
      switch (e.state) {
      case ResourceState::Building:
        ++u.building;
        break;
      case ResourceState::Leased:
        ++u.leased;
        break;
      case ResourceState::Idle:
        ++u.idle;
        break;
      case ResourceState::Retiring:
        ++u.retiring;
        break;
      case ResourceState::Quarantined:
        ++u.quarantined;
        u.quarantineBytes += e.allocated;
        break;
      }
      u.allocatedStagingBytes += e.allocated;
      u.reservedStagingBytes += e.bytes - e.allocated;
      if (e.cacheable) {
        u.retainedAllocatedBytes += e.allocated;
        u.retainedReservedBytes += e.bytes - e.allocated;
      }
      u.backgroundWorkers += e.allocatedWorkers;
      u.reservedWorkers += e.workers - e.allocatedWorkers;
      u.streams += e.streams;
      u.outstandingEvents += e.events;
    }
    return u;
  }

  void peaks() {
    auto u = usage();
    counters.peakLiveStagingBytes =
        std::max(counters.peakLiveStagingBytes, live(u));
    counters.peakAllocatedStagingBytes =
        std::max(counters.peakAllocatedStagingBytes, u.allocatedStagingBytes);
  }

  bool fits(const CacheRequest &r, const Entry *grow,
            const TransferResourceUsage &all,
            const TransferResourceUsage &device) const {
    size_t contexts = grow ? 0 : 1;
    size_t bytes = grow ? r.bytes - grow->bytes : r.bytes;
    return !exceeds(all.contexts, contexts, limits.maxContexts) &&
           !exceeds(device.contexts, contexts, limits.maxContextsPerDevice) &&
           !exceeds(workers(all), grow ? 0 : r.workers,
                    limits.maxBackgroundWorkers) &&
           !exceeds(retained(all), r.cacheable ? bytes : 0,
                    limits.maxRetainedBytes) &&
           !exceeds(live(all), bytes,
                    limits.maxLiveStagingBytes.value_or(
                        std::numeric_limits<size_t>::max()));
  }

  // Called with an exclusive Retiring entry, never under mu. Accounting is
  // released only after the context and its allocation machinery are gone.
  void retire(Entry *e) noexcept {
    bool quarantined =
        e->context && e->context->close() == TransferCompletion::Unknown;
    if (!quarantined)
      e->context.reset();
    std::lock_guard<std::mutex> lock(mu);
    if (quarantined) {
      e->state = ResourceState::Quarantined;
      ++counters.quarantines;
    } else {
      auto it = std::find_if(entries.begin(), entries.end(),
                             [&](const auto &p) { return p.get() == e; });
      entries.erase(it);
    }
    cv.notify_all();
  }

  void finish(Entry *e, bool healthy) noexcept {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (healthy && e->cacheable && !closed && e->generation == generation &&
          !usage(&e->key, true).quarantined) {
        e->state = ResourceState::Idle;
        e->lastUse = ++clock;
        cv.notify_all();
        return;
      }
      e->state = ResourceState::Retiring;
    }
    retire(e);
  }

  struct Lease {
    std::shared_ptr<CacheState> owner;
    Entry *entry;
    bool build, healthy = false;
    Lease(std::shared_ptr<CacheState> owner, Entry *entry, bool build)
        : owner(std::move(owner)), entry(entry), build(build) {}
    Lease(Lease &&other) noexcept
        : owner(std::move(other.owner)), entry(other.entry), build(other.build),
          healthy(other.healthy) {
      other.entry = nullptr;
    }
    Lease(const Lease &) = delete;
    ~Lease() {
      if (entry)
        owner->finish(entry, healthy);
    }
  };

  std::variant<Lease, TransferError> acquire(const CacheRequest &r) {
    // Allocate bookkeeping before taking the admission lock. No backend,
    // worker or staging resources exist until reservations have been published.
    auto fresh = std::make_unique<Entry>(r);
    auto deadline = std::chrono::steady_clock::time_point::max();
    if (limits.acquireTimeout) {
      auto now = std::chrono::steady_clock::now();
      deadline =
          now + std::min(*limits.acquireTimeout,
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - now));
    }
    std::unique_lock<std::mutex> lock(mu);
    Waiter waiter(*this);
    bool waited = false;
    for (;;) {
      if (closed)
        return TransferError{"resources_closed", "resource cache is closed"};
      auto permanent = usage(nullptr, true);
      auto devicePermanent = usage(&r.key, true);
      if (devicePermanent.quarantined)
        return TransferError{"resources_disabled",
                             "device has quarantined work"};
      if (!fits(r, nullptr, permanent, devicePermanent))
        return TransferError{"resource_limit",
                             "request exceeds available hard limits"};
      if (waited && std::chrono::steady_clock::now() >= deadline) {
        ++counters.timeouts;
        return TransferError{"resource_timeout",
                             "resource admission timed out"};
      }
      if (head == &waiter) {
        Entry *sufficient = nullptr, *grow = nullptr;
        if (r.cacheable)
          for (auto &e : entries) {
            if (e->state != ResourceState::Idle || !(e->key == r.key))
              continue;
            if (e->capacity >= r.slotCapacity) {
              if (!sufficient || e->capacity < sufficient->capacity)
                sufficient = e.get();
            } else if (!grow || e->capacity > grow->capacity)
              grow = e.get();
          }
        if (sufficient) {
          sufficient->state = ResourceState::Leased;
          ++counters.hits;
          waiter.remove();
          return Lease(shared_from_this(), sufficient, false);
        }
        auto all = usage(), device = usage(&r.key);
        if (fits(r, grow, all, device)) {
          Entry *entry = grow;
          if (grow) {
            grow->capacity = r.slotCapacity;
            grow->bytes =
                r.bytes; // reserve delta while old bytes remain allocated
            grow->state = ResourceState::Building;
            ++counters.growths;
          } else {
            entry = fresh.get();
            entries.push_back(std::move(fresh));
            ++counters.misses;
            if (!r.cacheable)
              ++counters.ephemeral;
          }
          entry->generation = generation;
          peaks();
          waiter.remove();
          return Lease(shared_from_this(), entry, true);
        }
        // Evict the oldest idle context that can relieve a binding limit.
        size_t delta = grow ? r.bytes - grow->bytes : r.bytes;
        bool needContexts =
            exceeds(all.contexts, grow ? 0 : 1, limits.maxContexts);
        bool needDevice =
            exceeds(device.contexts, grow ? 0 : 1, limits.maxContextsPerDevice);
        bool needWorkers = exceeds(workers(all), grow ? 0 : r.workers,
                                   limits.maxBackgroundWorkers);
        bool needRetained = exceeds(retained(all), r.cacheable ? delta : 0,
                                    limits.maxRetainedBytes);
        bool needLive = exceeds(live(all), delta,
                                limits.maxLiveStagingBytes.value_or(
                                    std::numeric_limits<size_t>::max()));
        Entry *victim = nullptr;
        for (auto &e : entries) {
          if (e.get() == grow || e->state != ResourceState::Idle)
            continue;
          bool helps = needContexts ||
                       (needDevice && e->key.sameDevice(r.key)) ||
                       (needWorkers && e->workers) ||
                       (needRetained && e->cacheable) || needLive;
          if (helps && (!victim || e->lastUse < victim->lastUse))
            victim = e.get();
        }
        if (victim) {
          victim->state = ResourceState::Retiring;
          ++counters.evictions;
          lock.unlock();
          retire(victim);
          lock.lock();
          continue;
        }
      }
      if (!waited) {
        waited = true;
        ++counters.waits;
      }
      if (std::chrono::steady_clock::now() >= deadline)
        continue;
      cv.wait_until(lock, deadline);
    }
  }

  ClearResult clear(bool closing) {
    std::unique_lock<std::mutex> lock(mu);
    if (!closing && closed)
      return ClearResult::Closed;
    if (!closed) {
      if (generation == std::numeric_limits<uint64_t>::max())
        return ClearResult::GenerationExhausted;
      ++generation;
    }
    const auto cutoff = generation;
    if (closing)
      closed = true;
    cv.notify_all();
    for (;;) {
      Entry *victim = nullptr;
      for (auto &e : entries)
        if (e->state == ResourceState::Idle && e->generation < cutoff) {
          victim = e.get();
          break;
        }
      if (victim) {
        victim->state = ResourceState::Retiring;
        ++counters.evictions;
        lock.unlock();
        retire(victim);
        lock.lock();
        continue;
      }
      if (!closing)
        return ClearResult::Complete;
      auto u = usage();
      if (u.building || u.leased || u.retiring || counters.waiters) {
        cv.wait(lock);
        continue;
      }
      if (u.quarantined)
        return ClearResult::Quarantined;
      return ClearResult::Complete;
    }
  }

  TransferResourceStats snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    auto result = counters;
    static_cast<TransferResourceUsage &>(result) = usage();
    result.closed = closed;
    result.generation = generation;
    for (const auto &e : entries) {
      if (std::any_of(result.devices.begin(), result.devices.end(),
                      [&](const auto &d) {
                        return d.kind == e->key.backend.kind &&
                               d.device == e->key.backend.device;
                      }))
        continue;
      TransferDeviceStats d;
      static_cast<TransferResourceUsage &>(d) = usage(&e->key);
      d.kind = e->key.backend.kind;
      d.device = e->key.backend.device;
      d.disabled = d.quarantined != 0;
      result.devices.push_back(d);
    }
    return result;
  }
};

// Allocation accounting shares the admission mutex but never holds it while
// invoking the backend. The target charge is reserved before any allocation.
class CountedBackend : public CopyBackend {
public:
  CountedBackend(std::unique_ptr<CopyBackend> backend,
                 std::shared_ptr<CacheState> state, Entry &entry)
      : backend_(std::move(backend)), state_(std::move(state)), entry_(entry) {
    std::lock_guard<std::mutex> lock(state_->mu);
    entry_.streams = backend_->numQueues();
    state_->counters.streamCreations += entry_.streams;
  }
  ~CountedBackend() override {
    backend_.reset();
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->counters.streamDestructions += entry_.streams;
    state_->counters.workerJoins += entry_.allocatedWorkers;
    state_->counters.eventRetirements += entry_.events;
    entry_.streams = entry_.allocatedWorkers = entry_.events = 0;
  }
  void *allocStaging(size_t bytes) override {
    void *p = backend_->allocStaging(bytes);
    if (!p)
      return nullptr;
    try {
      sizes_.emplace(p, bytes);
    } catch (...) {
      backend_->freeStaging(p);
      throw;
    }
    std::lock_guard<std::mutex> lock(state_->mu);
    entry_.allocated += bytes;
    ++state_->counters.stagingAllocations;
    state_->peaks();
    return p;
  }
  void freeStaging(void *p) override {
    if (!p)
      return;
    size_t bytes = sizes_.at(p);
    backend_->freeStaging(p);
    sizes_.erase(p);
    std::lock_guard<std::mutex> lock(state_->mu);
    entry_.allocated -= bytes;
    ++state_->counters.stagingFrees;
  }
  int numQueues() const override { return backend_->numQueues(); }
  int device() const override { return backend_->device(); }
  bool failed() const override { return backend_->failed(); }
  const std::string &error() const override { return backend_->error(); }
  bool waitStream(const void *s) override { return backend_->waitStream(s); }
  QueueCompletion quiesce() override { return backend_->quiesce(); }
  void copyAsync(int q, void *dst, const void *src, size_t bytes,
                 CopyDir dir) override {
    backend_->copyAsync(q, dst, src, bytes, dir);
  }
  EventHandle recordEvent(int q) override {
    auto ev = backend_->recordEvent(q);
    if (ev) {
      std::lock_guard<std::mutex> lock(state_->mu);
      ++entry_.events;
      ++state_->counters.eventCreations;
    }
    return ev;
  }
  void waitEvent(EventHandle ev) override {
    backend_->waitEvent(ev);
    if (ev && !backend_->failed()) {
      std::lock_guard<std::mutex> lock(state_->mu);
      --entry_.events;
      ++state_->counters.eventRetirements;
    }
  }
  bool queryEvent(EventHandle ev) override { return backend_->queryEvent(ev); }

private:
  std::unique_ptr<CopyBackend> backend_;
  std::shared_ptr<CacheState> state_;
  Entry &entry_;
  std::map<void *, size_t> sizes_;
};

std::unique_ptr<CopyBackend> makeBackend(const TransferBackendConfig &config) {
  if (config.kind == MemoryKind::Host)
    return std::make_unique<HostBackend>(config.streams);
#ifdef RELOC_ENABLE_CUDA
  return std::make_unique<CudaBackend>(config.streams, config.device);
#else
  throw std::runtime_error("CUDA backend is not available in this build");
#endif
}

TransferOutcome run(TransferRequest &request, const CacheRequest &r,
                    const std::shared_ptr<CacheState> &state,
                    std::shared_ptr<void> owners,
                    TransferCompletion &completion) {
  auto acquired = state->acquire(r);
  if (auto *error = std::get_if<TransferError>(&acquired))
    return {*error, TransferCompletion::NotLaunched};
  auto &lease = std::get<CacheState::Lease>(acquired);
  auto &entry = *lease.entry;
  try {
    if (lease.build) {
      if (!entry.context) {
        auto backend = state->factory(r.key.backend);
        if (!backend || backend->failed() ||
            backend->device() != r.key.backend.device ||
            backend->numQueues() != r.key.backend.streams)
          throw std::runtime_error(
              "backend construction failed or configuration differs");
        entry.context = std::make_unique<TransferContext>(
            std::make_unique<CountedBackend>(std::move(backend), state, entry));
      }
      TransferContextAccess::prepare(*entry.context, request.direction,
                                     r.execution, entry.capacity);
      const auto prepared = entry.context->stats();
      std::lock_guard<std::mutex> lock(state->mu);
      state->counters.workerCreations +=
          prepared.backgroundWorkers - entry.allocatedWorkers;
      entry.allocatedWorkers = prepared.backgroundWorkers;
      if (state->closed)
        return {TransferError{"resources_closed",
                              "cache closed during construction"},
                TransferCompletion::NotLaunched};
      if (state->usage(&r.key, true).quarantined)
        return {TransferError{"resources_disabled",
                              "device disabled during construction"},
                TransferCompletion::NotLaunched};
      entry.state = ResourceState::Leased;
    }
    auto result = TransferContextAccess::execute(
        request, *entry.context, r.options, r.execution, std::move(owners));
    lease.healthy = !result.error && entry.context->stats().reusable;
    return result;
  } catch (...) {
    if (entry.context)
      completion = TransferContextAccess::completion(*entry.context);
    throw;
  }
}
} // namespace

struct TransferResourceCache::Impl {
  detail::ProcessIdentity identity = detail::captureProcessIdentity();
  std::shared_ptr<CacheState> state;
  Impl(TransferResourceLimits limits, TransferBackendFactory factory)
      : state(std::make_shared<CacheState>(limits, factory ? std::move(factory)
                                                           : makeBackend)) {}
  bool validProcess() const {
    return identity == detail::currentProcessIdentity();
  }
};

TransferResourceCache::TransferResourceCache(TransferResourceLimits limits,
                                             TransferBackendFactory factory) {
  if (limits.acquireTimeout && limits.acquireTimeout->count() < 0)
    throw std::invalid_argument(
        "resource acquisition timeout must be nonnegative");
  impl_ = std::make_unique<Impl>(limits, std::move(factory));
}

TransferResourceCache::~TransferResourceCache() {
  // In a fork child, even destroying an inherited mutex/backend can deadlock.
  // Abandon the inherited copy without touching its locks or CUDA machinery.
  if (!impl_->validProcess()) {
    (void)impl_.release();
    return;
  }
  // Destruction performs the same drain without allocating an error message.
  // As with other C++ owners, the wrapper must outlive all calls using it.
  CallScope scope(impl_->state.get());
  (void)impl_->state->clear(true);
}

std::optional<TransferError> TransferResourceCache::clear() {
  if (!impl_->validProcess())
    return TransferError{"process_mismatch",
                         "resource cache belongs to another process"};
  if (CallScope::contains(impl_->state.get()))
    return TransferError{"resource_reentrant",
                         "recursive cache lifecycle operation"};
  CallScope scope(impl_->state.get());
  return clearError(impl_->state->clear(false));
}

std::optional<TransferError> TransferResourceCache::close() {
  if (!impl_->validProcess())
    return TransferError{"process_mismatch",
                         "resource cache belongs to another process"};
  if (CallScope::contains(impl_->state.get()))
    return TransferError{"resource_reentrant",
                         "recursive cache lifecycle operation"};
  CallScope scope(impl_->state.get());
  return clearError(impl_->state->clear(true));
}

TransferResourceStats TransferResourceCache::stats() const {
  if (!impl_->validProcess()) {
    TransferResourceStats invalid;
    invalid.closed = true;
    invalid.processValid = false;
    return invalid;
  }
  return impl_->state->snapshot();
}

TransferOutcome executeTransferCached(TransferRequest &request,
                                      TransferResourceCache &resources,
                                      const CachedTransferOptions &options,
                                      std::shared_ptr<void> bufferOwners) {
  auto reject = [](const char *code, const char *message) {
    return TransferOutcome{TransferError{code, message},
                           TransferCompletion::NotLaunched};
  };
  if (!resources.impl_->validProcess())
    return reject("process_mismatch",
                  "resource cache belongs to another process");
  auto state = resources.impl_->state;
  if (CallScope::contains(state.get()))
    return reject("resource_reentrant",
                  "recursive resource admission is unsupported");
  if (!bufferOwners)
    return reject("invalid_options", "a buffer owner token is required");
  // Keep external workers alive through admission, execution, and retirement.
  auto gatherOwner = options.gather;
  CallScope scope(state.get());
  bool claimed = false;
  TransferOutcome result;
  TransferCompletion completion = TransferCompletion::NotLaunched;
  try {
    auto described =
        detail::describeCachedTransfer(request, options, state->limits);
    if (auto *error = std::get_if<TransferError>(&described))
      return {*error, TransferCompletion::NotLaunched};
    request.consumed = claimed = true;
    {
      std::lock_guard<std::mutex> lock(state->mu);
      ++state->counters.requests;
    }
    result = run(request, std::get<CacheRequest>(described), state,
                 std::move(bufferOwners), completion);
  } catch (const std::exception &e) {
    result.completion = completion;
    result.error = TransferError{completion == TransferCompletion::Unknown
                                     ? "completion_unknown"
                                     : "backend_failure",
                                 e.what()};
  } catch (...) {
    result.completion = completion;
    result.error = TransferError{completion == TransferCompletion::Unknown
                                     ? "completion_unknown"
                                     : "backend_failure",
                                 "unexpected cached transfer failure"};
  }
  if (claimed && result.error) {
    std::lock_guard<std::mutex> lock(state->mu);
    ++state->counters.failures;
  }
  return result;
}
} // namespace reloc
