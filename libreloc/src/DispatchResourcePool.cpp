#include "DispatchResourcePool.h"
#include "reloc/GatherPool.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace reloc::dispatch {
namespace {
TransferError fail(const char *code, const char *message) {
  return {code, message};
}
size_t share(size_t total, unsigned count, unsigned index) {
  return total / count + (index < total % count);
}
} // namespace

bool Resources::Pool::Key::operator==(const Key &o) const {
  return device == o.device && streams == o.streams && threads == o.threads &&
         borrowedWorkers == o.borrowedWorkers && affinity == o.affinity;
}
Resources::Pool::Pool(size_t retained, size_t live, unsigned workers,
                      unsigned streams, unsigned contexts, unsigned perDevice,
                      std::optional<uint64_t> timeout)
    : perDevice(perDevice), timeout(timeout) {
  if (!contexts || !perDevice || perDevice > contexts || streams < contexts ||
      (live && live < contexts) ||
      (timeout &&
       *timeout > uint64_t(std::numeric_limits<int64_t>::max() / 2000000)))
    throw std::invalid_argument("invalid typed context limits or timeout");
  slots.reserve(contexts);
  for (unsigned i = 0; i < contexts; ++i) {
    Slot slot{};
    slot.retained = share(retained, contexts, i);
    slot.live = share(live, contexts, i);
    slot.workers = share(workers, contexts, i);
    slot.streams = share(streams, contexts, i);
    slots.push_back(std::move(slot));
  }
}
Resources::Pool::Lease::~Lease() { pool->release(slot); }

std::variant<std::shared_ptr<Resources::Pool::Lease>, TransferError>
Resources::Pool::acquire(Key key) {
  if (!valid())
    return fail("process_mismatch",
                "typed resources belong to another process");
  if (GatherPool::inCallback())
    return fail("reentrant_call", "cannot acquire from a gather callback");
  auto fits = [&](const Slot &s) {
    return key.streams > 0 && unsigned(key.streams) <= s.streams &&
           key.threads - 1 + key.borrowedWorkers <= s.workers;
  };
  if (!std::any_of(slots.begin(), slots.end(), fits))
    return fail("resource_limit",
                "typed stream/worker per-context quota exceeded");
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  std::unique_lock<std::mutex> lock(mu);
  const uint64_t ticket = ++clock;
  waiters.push_back(ticket);
  cv.notify_all();
  auto remove = [&] {
    waiters.erase(std::find(waiters.begin(), waiters.end(), ticket));
    cv.notify_all();
  };
  bool counted = false;
  try {
    for (;;) {
      if (closed || quarantined) {
        remove();
        return quarantined
                   ? fail("completion_unknown", "typed resources quarantined")
                   : fail("resources_closed", "typed resources closed");
      }
      std::shared_ptr<Completion> pending;
      if (!clearing && waiters.front() == ticket) {
        unsigned deviceCount = 0;
        for (auto &s : slots)
          deviceCount += s.key && s.key->device == key.device;
        size_t choice = slots.size();
        for (size_t i = 0; i < slots.size(); ++i) {
          auto &s = slots[i];
          if (s.active || !fits(s))
            continue;
          if (s.key && *s.key == key) {
            choice = i;
            break;
          }
          if (deviceCount >= perDevice &&
              (!s.key || s.key->device != key.device))
            continue;
          if (choice == slots.size() || !s.key ||
              (slots[choice].key && s.lastUse < slots[choice].lastUse))
            choice = i;
        }
        if (choice != slots.size()) {
          auto &s = slots[choice];
          if (!s.resources)
            s.resources = create(s);
          // Allocate the lease before marking the slot active. A failed host
          // allocation cannot leave an occupied slot or submitted work behind.
          auto lease =
              std::make_shared<Lease>(shared_from_this(), s.resources, choice);
          if (s.key && !(*s.key == key))
            ++evictions;
          s.key = std::move(key);
          s.active = true;
          s.lastUse = ++clock;
          ++active;
          peakActive = std::max(active, peakActive);
          remove();
          return lease;
        }
        // A pending submission owns its slot until host completion. Progress
        // it outside the admission mutex, without taking another lease.
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (auto &s : slots)
          if (s.active && s.lastUse < oldest &&
              (deviceCount < perDevice || s.key->device == key.device))
            if (auto work = s.pending.lock()) {
              pending = std::move(work);
              oldest = s.lastUse;
            }
      }
      if (!counted) {
        ++waits;
        counted = true;
      }
      if (timeout &&
          Clock::now() - start >= std::chrono::milliseconds(*timeout)) {
        remove();
        return fail("acquire_timeout", "typed context admission timed out");
      }
      if (pending) {
        lock.unlock();
        std::optional<TransferError> error;
        bool done = false;
        try {
          if (timeout) {
            auto result = pending->query();
            if (auto *failure = std::get_if<TransferError>(&result))
              error = *failure;
            else
              done = std::get<bool>(result);
          } else {
            error = pending->wait().error;
            done = true;
          }
        } catch (...) {
          lock.lock();
          throw;
        }
        lock.lock();
        if (error) {
          remove();
          return *error;
        }
        if (done)
          continue;
      }
      if (timeout) {
        const auto deadline = start + std::chrono::milliseconds(*timeout);
        cv.wait_until(
            lock, pending ? std::min(deadline, Clock::now() +
                                                   std::chrono::milliseconds(1))
                          : deadline);
      } else
        cv.wait(lock);
    }
  } catch (...) {
    remove();
    throw;
  }
}

void Resources::Pool::submitted(const Lease &lease,
                                const std::shared_ptr<Completion> &completion) {
  std::lock_guard<std::mutex> lock(mu);
  slots[lease.slot].pending = completion;
  cv.notify_all();
}
void Resources::Pool::release(size_t index) {
  // No engine mutex may be held here: stats and maintenance take the pool
  // mutex before visiting engines. Completion releases its token after unlock.
  auto completed = slots[index].resources->stats();
  std::lock_guard<std::mutex> lock(mu);
  auto &s = slots[index];
  s.pending.reset();
  s.active = false;
  s.lastUse = ++clock;
  --active;
  quarantined |= completed.quarantined;
  s.completed = std::move(completed);
  cv.notify_all();
}

ResourceStats Resources::Pool::stats() const {
  ResourceStats out;
  if (!valid()) {
    out.processValid = false;
    return out;
  }
  std::lock_guard<std::mutex> lock(mu);
  out.closed = closed;
  out.quarantined = quarantined;
  out.activeContexts = active;
  out.peakActiveContexts = peakActive;
  out.queued = waiters.size();
  out.admissionWaits = waits;
  out.evictions = evictions;
  out.contextLimit = slots.size();
  out.perDeviceLimit = perDevice;
  for (const auto &slot : slots) {
    out.retainedLimit += slot.retained;
    out.liveLimit += slot.live;
    out.workerLimit += slot.workers;
    out.streamLimit += slot.streams;
    if (!slot.resources)
      continue;
    // Never wait on a busy engine while holding admission: a slow GPU must
    // not keep unrelated callers from taking free slots. Execution counters
    // are completion snapshots; live scratch comes from the atomic gauge.
    const auto &s = slot.completed;
    out.requests += s.requests;
    out.hits += s.hits;
    out.contexts += s.contexts;
    out.deviceAllocations += s.deviceAllocations;
    out.hostAllocations += s.hostAllocations;
    out.frees += s.frees;
    out.copyCalls += s.copyCalls;
    out.eventRecords += s.eventRecords;
    out.eventWaits += s.eventWaits;
    out.callerWaits += s.callerWaits;
    if (!slot.active) {
      out.retainedBytes += s.retainedBytes;
      out.deviceBytes += s.deviceBytes;
      out.hostBytes += s.hostBytes;
    }
    const unsigned workers =
        slot.active ? slot.key->threads - 1 : s.backgroundWorkers;
    const unsigned streams =
        slot.active ? unsigned(slot.key->streams) : unsigned(s.streams);
    out.backgroundWorkers += workers;
    out.streams += streams;
    out.quarantined |= s.quarantined;
    if (slot.key)
      out.contextDetails.push_back(
          {slot.key->device, slot.active, workers, streams,
           slot.active ? slot.key->borrowedWorkers : 0,
           slot.active ? 0 : s.retainedBytes, slot.live, slot.retained,
           slot.key->affinity});
  }
  out.liveBytes = gauge->live.load();
  out.peakLiveBytes = gauge->peak.load();
  return out;
}

std::optional<TransferError> Resources::Pool::drain(bool close) {
  if (!valid())
    return fail("process_mismatch",
                "typed resources belong to another process");
  if (GatherPool::inCallback())
    return fail("reentrant_call", "cannot drain from a gather callback");
  std::unique_lock<std::mutex> lock(mu);
  // Serialize maintenance, while close immediately rejects queued admissions.
  if (close) {
    closed = true;
    cv.notify_all();
  }
  cv.wait(lock, [&] { return !clearing; });
  if (closed && !close)
    return fail("resources_closed", "typed resources closed");
  clearing = true;
  cv.notify_all();
  std::optional<TransferError> error;
  try {
    while (active) {
      std::shared_ptr<Completion> pending;
      for (auto &s : slots)
        if (s.active && (pending = s.pending.lock()))
          break;
      if (!pending) {
        cv.wait(lock);
        continue;
      }
      lock.unlock();
      TransferOutcome outcome;
      try {
        outcome = pending->wait();
      } catch (...) {
        lock.lock();
        throw;
      }
      lock.lock();
      if (outcome.error && !error)
        error = outcome.error;
    }
    for (auto &s : slots) {
      if (!s.resources)
        continue;
      auto failure = close ? s.resources->close() : s.resources->clear();
      if (failure && !error)
        error = failure;
      s.completed = s.resources->stats();
      quarantined |= s.completed.quarantined;
      if (!s.completed.quarantined)
        s.key.reset();
    }
  } catch (...) {
    clearing = false;
    cv.notify_all();
    throw;
  }
  clearing = false;
  cv.notify_all();
  if (quarantined)
    return fail("completion_unknown", "typed resources quarantined");
  return error;
}
} // namespace reloc::dispatch
