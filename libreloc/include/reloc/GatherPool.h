//===- GatherPool.h - persistent gather/scatter worker pool -----*- C++ -*-===//
//
// D1 (issue #65): a long-lived pool of worker threads. parallelFor runs a
// row-range callable over [begin, end) partitioned into <= threadCount()
// contiguous sub-ranges (one runs inline on the calling thread) and returns
// only after every sub-range completed (counting barrier) -- the pipeline
// relies on that barrier before copyAsync reads the staging buffer. Explicit
// close() lifecycle so pybind callers can tear the pool down
// deterministically: no threads outlive the interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_GATHERPOOL_H
#define RELOC_GATHERPOOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace reloc {

class GatherPool {
public:
  /// threads == 0 resolves to std::thread::hardware_concurrency() (>= 1).
  /// Spawns threadCount() - 1 OS workers; the calling thread acts as the
  /// last worker of every dispatch, so no worker idles while the caller
  /// blocks.
  explicit GatherPool(unsigned threads = 0);
  ~GatherPool(); // close()s if still open

  GatherPool(const GatherPool &) = delete;
  GatherPool &operator=(const GatherPool &) = delete;

  int threadCount() const { return threads_; }
  bool closed() const { return closed_; }

  /// Join every worker. Idempotent, and safe against concurrent close()
  /// calls or an in-flight parallelFor: it serializes behind them, so
  /// "close() returned" always means "workers are gone".
  void close();

  /// Partition [begin, end) into <= threadCount() contiguous sub-ranges of
  /// at least minPerWorker rows each and run fn(subBegin, subEnd) across the
  /// workers; blocks until every sub-range completed. Collapses to a plain
  /// inline fn(begin, end) -- no locks, no worker wakeup -- when only one
  /// sub-range results, so the threads==1 path is bit-identical in behavior
  /// to calling fn directly. Concurrent dispatches from multiple driver
  /// threads are serialized internally (each runs to completion before the
  /// next starts), and a dispatch that loses a race with close() runs fn
  /// inline instead of handing work to joined workers. Not reentrant from
  /// within fn; fn must not throw. Calling after close() is a contract
  /// violation (asserted in debug); in release it degrades to the inline
  /// path.
  void parallelFor(int64_t begin, int64_t end, int64_t minPerWorker,
                   const std::function<void(int64_t, int64_t)> &fn);

private:
  struct Range {
    int64_t begin, end;
  };

  void workerLoop();

  int threads_ = 1;
  std::atomic<bool> closed_{false};
  std::vector<std::thread> workers_; // threads_ - 1 entries

  std::mutex driverMu_; // serializes whole dispatches and close() against
                        // each other (pybind exposes both to arbitrary
                        // Python threads with the GIL released)
  std::mutex mu_;       // guards everything below
  std::condition_variable cv_;   // wakes workers (new work or stop)
  std::condition_variable done_; // wakes the parallelFor barrier
  const std::function<void(int64_t, int64_t)> *fn_ = nullptr; // live dispatch
  std::vector<Range> pending_; // sub-ranges not yet claimed
  int outstanding_ = 0;        // handed to workers, not yet finished
  bool stop_ = false;
};

} // namespace reloc

#endif // RELOC_GATHERPOOL_H
