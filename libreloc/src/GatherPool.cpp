//===- GatherPool.cpp - persistent gather/scatter worker pool -------------===//

#include "reloc/GatherPool.h"

#include <algorithm>
#include <cassert>

namespace reloc {
namespace {
thread_local unsigned callbackDepth = 0;
void invoke(const std::function<void(int64_t, int64_t)> &fn, int64_t begin,
            int64_t end) {
  struct Guard {
    Guard() { ++callbackDepth; }
    ~Guard() { --callbackDepth; }
  } guard;
  fn(begin, end);
}
} // namespace

bool GatherPool::inCallback() { return callbackDepth != 0; }

GatherPool::GatherPool(unsigned threads) {
  if (threads == 0)
    threads = std::thread::hardware_concurrency();
  if (threads == 0)
    threads = 1;
  threads_ = static_cast<int>(threads);
  assigned_.assign(static_cast<size_t>(threads_), Range{0, 0});
  workers_.reserve(static_cast<size_t>(threads_ - 1));
  for (int w = 0; w + 1 < threads_; ++w) {
    // Same rationale as executeH2DThreaded: if a std::thread constructor
    // throws partway through, join the already-spawned workers before
    // rethrowing so unwinding never destroys a joinable std::thread.
    try {
      workers_.emplace_back([this, w] { workerLoop(w + 1); });
    } catch (...) {
      close();
      throw;
    }
  }
}

GatherPool::~GatherPool() { close(); }

void GatherPool::close() {
  // Serialize against other close() calls AND in-flight dispatches (pybind
  // exposes both with the GIL released): once close() returns, every worker
  // has been joined and no dispatch is using workers. Inline calls need no
  // worker state and can continue after close().
  std::lock_guard<std::mutex> driverLk(driverMu_);
  if (closed_)
    return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    assert(outstanding_ == 0 && "close() during an in-flight parallelFor");
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread &t : workers_)
    if (t.joinable())
      t.join();
  workers_.clear();
  closed_ = true;
}

void GatherPool::workerLoop(int worker) {
  std::unique_lock<std::mutex> lk(mu_);
  const size_t index = static_cast<size_t>(worker);
  // Index the slot under the lock on every use: assigned_ is rewritten by
  // each dispatch, so no reference into it may outlive the lock.
  auto pending = [&] { return assigned_[index].begin < assigned_[index].end; };
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || pending(); });
    if (!pending())
      return; // stop_ set and nothing assigned
    Range r = assigned_[index];
    assigned_[index] = Range{0, 0};
    const auto *fn = fn_;
    lk.unlock();
    std::exception_ptr failure;
    try {
      invoke(*fn, r.begin, r.end);
    } catch (...) {
      failure = std::current_exception();
    }
    lk.lock();
    if (failure && !failure_)
      failure_ = failure;
    if (--outstanding_ == 0)
      done_.notify_one();
  }
}

void GatherPool::parallelFor(int64_t begin, int64_t end, int64_t minPerWorker,
                             const std::function<void(int64_t, int64_t)> &fn) {
  const int64_t n = end - begin;
  if (n <= 0)
    return;
  if (minPerWorker < 1)
    minPerWorker = 1;
  const int64_t parts =
      std::min<int64_t>(threads_, std::max<int64_t>(1, n / minPerWorker));
  const int64_t per = (n + parts - 1) / parts; // ceil
  // Sub-range p goes to worker p; range 0 runs inline below. (ceil rounding
  // can make trailing ranges empty; skip them.)
  std::vector<Range> rest(static_cast<size_t>(threads_), Range{0, 0});
  int count = 0;
  for (int64_t p = 1; p < parts; ++p) {
    int64_t b = begin + p * per;
    int64_t e = std::min(begin + (p + 1) * per, end);
    if (b < e) {
      rest[static_cast<size_t>(p)] = {b, e};
      ++count;
    }
  }
  if (count == 0) {
    invoke(fn, begin, end); // single sub-range: bit-identical to a direct call
    return;
  }
  // Serialize whole dispatches against each other and against close():
  // Python threads may share one pool. Held across the barrier wait below.
  std::lock_guard<std::mutex> driverLk(driverMu_);
  if (closed_) {
    // Lost a race with close(), possibly between chunks of a borrowed-pool
    // transfer: the workers are gone, so run the whole range inline
    // rather than parking work no one will drain.
    invoke(fn, begin, end);
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    for (size_t i = 0; i < assigned_.size(); ++i)
      assigned_[i] = rest[i];
    outstanding_ = count;
    failure_ = nullptr;
  }
  cv_.notify_all();
  std::exception_ptr failure;
  try {
    invoke(fn, begin, std::min(begin + per, end));
  } catch (...) {
    failure = std::current_exception();
  }
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return outstanding_ == 0; });
  fn_ = nullptr; // still under mu_: workers only read fn_ under the lock
  if (!failure)
    failure = failure_;
  failure_ = nullptr;
  lk.unlock();
  if (failure)
    std::rethrow_exception(failure);
}

} // namespace reloc
