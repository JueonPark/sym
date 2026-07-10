//===- GatherPool.cpp - persistent gather/scatter worker pool -------------===//

#include "reloc/GatherPool.h"

#include <algorithm>
#include <cassert>

namespace reloc {

GatherPool::GatherPool(unsigned threads) {
  if (threads == 0)
    threads = std::thread::hardware_concurrency();
  if (threads == 0)
    threads = 1;
  threads_ = static_cast<int>(threads);
  workers_.reserve(static_cast<size_t>(threads_ - 1));
  for (int w = 0; w + 1 < threads_; ++w) {
    // Same rationale as executeH2DThreaded: if a std::thread constructor
    // throws partway through, join the already-spawned workers before
    // rethrowing so unwinding never destroys a joinable std::thread.
    try {
      workers_.emplace_back([this] { workerLoop(); });
    } catch (...) {
      close();
      throw;
    }
  }
}

GatherPool::~GatherPool() { close(); }

void GatherPool::close() {
  // Serialize concurrent close() calls (pybind exposes close() with the
  // GIL released): the loser blocks until the winner has joined every
  // worker, so "close() returned" always means "threads are gone".
  std::lock_guard<std::mutex> closeLk(closeMu_);
  if (closed_)
    return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    assert(pending_.empty() && outstanding_ == 0 &&
           "close() during an in-flight parallelFor");
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread &t : workers_)
    if (t.joinable())
      t.join();
  workers_.clear();
  closed_ = true;
}

void GatherPool::workerLoop() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
    if (pending_.empty())
      return; // stop_ set and nothing left to claim
    Range r = pending_.back();
    pending_.pop_back();
    const auto *fn = fn_;
    lk.unlock();
    (*fn)(r.begin, r.end);
    lk.lock();
    if (--outstanding_ == 0)
      done_.notify_one();
  }
}

void GatherPool::parallelFor(int64_t begin, int64_t end, int64_t minPerWorker,
                             const std::function<void(int64_t, int64_t)> &fn) {
  assert(!closed_ && "parallelFor on a closed GatherPool");
  const int64_t n = end - begin;
  if (n <= 0)
    return;
  if (minPerWorker < 1)
    minPerWorker = 1;
  const int64_t parts =
      std::min<int64_t>(threads_, std::max<int64_t>(1, n / minPerWorker));
  const int64_t per = (n + parts - 1) / parts; // ceil
  // Sub-ranges 1..parts-1 go to the workers; range 0 runs inline below.
  // (ceil rounding can make trailing ranges empty; skip them.)
  std::vector<Range> rest;
  for (int64_t p = 1; p < parts; ++p) {
    int64_t b = begin + p * per;
    int64_t e = std::min(begin + (p + 1) * per, end);
    if (b < e)
      rest.push_back({b, e});
  }
  if (rest.empty()) {
    fn(begin, end); // single sub-range: bit-identical to a direct call
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    pending_ = rest;
    outstanding_ = static_cast<int>(rest.size());
  }
  cv_.notify_all();
  fn(begin, std::min(begin + per, end));
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return outstanding_ == 0; });
  fn_ = nullptr; // still under mu_: workers only read fn_ under the lock
}

} // namespace reloc
