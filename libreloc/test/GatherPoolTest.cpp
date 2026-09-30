//===- GatherPoolTest.cpp - D1 worker-pool unit tests ---------------------===//
//
// The pool contract (issue #65): parallelFor covers [begin, end) exactly
// once whatever the thread/floor combination, the min-rows floor stops tiny
// ranges from shredding into per-row tasks, one pool is reusable across many
// dispatches, threads == 1 never leaves the calling thread, and close() is
// an idempotent, observable teardown.
//
//===----------------------------------------------------------------------===//

#include "reloc/GatherPool.h"
#include "BlockingGate.h"
#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

using reloc::GatherPool;

TEST(GatherPool, ParallelForCoversRangeExactlyOnce) {
  for (unsigned threads : {1u, 2u, 4u, 16u})
    for (int64_t n :
         {int64_t(0), int64_t(1), int64_t(7), int64_t(64), int64_t(1000)})
      for (int64_t minPer : {int64_t(1), int64_t(4), int64_t(100)}) {
        GatherPool pool(threads);
        std::vector<std::atomic<int>> hits(static_cast<size_t>(n) + 1);
        for (auto &h : hits)
          h.store(0);
        pool.parallelFor(0, n, minPer, [&](int64_t b, int64_t e) {
          for (int64_t i = b; i < e; ++i)
            hits[static_cast<size_t>(i)].fetch_add(1);
        });
        for (int64_t i = 0; i < n; ++i)
          EXPECT_EQ(hits[static_cast<size_t>(i)].load(), 1)
              << "i=" << i << " threads=" << threads << " n=" << n
              << " minPer=" << minPer;
      }
}

TEST(GatherPool, MinRowsFloorLimitsPartitionCount) {
  GatherPool pool(8);
  // 10 rows / 4-row floor: at most 2 sub-ranges despite 8 threads.
  std::atomic<int> calls{0};
  pool.parallelFor(0, 10, 4, [&](int64_t, int64_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 2);
  // Entirely below the floor: exactly one inline call.
  calls = 0;
  pool.parallelFor(0, 3, 4, [&](int64_t, int64_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 1);
}

TEST(GatherPool, SingleThreadRunsInlineOnCaller) {
  GatherPool pool(1);
  EXPECT_EQ(pool.threadCount(), 1);
  std::mutex mu;
  std::set<std::thread::id> ids;
  pool.parallelFor(0, 100, 1, [&](int64_t, int64_t) {
    std::lock_guard<std::mutex> lk(mu);
    ids.insert(std::this_thread::get_id());
  });
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(*ids.begin(), std::this_thread::get_id());
}

TEST(GatherPool, ReusableAcrossDispatches) {
  GatherPool pool(4);
  for (int round = 0; round < 50; ++round) {
    std::atomic<int64_t> sum{0};
    pool.parallelFor(0, 128, 1, [&](int64_t b, int64_t e) {
      int64_t s = 0;
      for (int64_t i = b; i < e; ++i)
        s += i;
      sum.fetch_add(s);
    });
    EXPECT_EQ(sum.load(), 128 * 127 / 2) << "round " << round;
  }
}

TEST(GatherPool, ConcurrentDispatchesSerialize) {
  // Two driver threads share one pool (the pybind surface allows exactly
  // this; release builds have no asserts). Internal serialization must keep
  // every dispatch's coverage and barrier intact.
  GatherPool pool(4);
  std::vector<std::atomic<int>> hits(2048);
  for (auto &h : hits)
    h.store(0);
  auto driver = [&](int64_t base) {
    for (int r = 0; r < 20; ++r)
      pool.parallelFor(base, base + 1024, 1, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i)
          hits[static_cast<size_t>(i)].fetch_add(1);
      });
  };
  std::thread other(driver, 1024);
  driver(0);
  other.join();
  for (size_t i = 0; i < hits.size(); ++i)
    ASSERT_EQ(hits[i].load(), 20) << "index " << i;
}

TEST(GatherPool, CloseIsIdempotentAndObservable) {
  GatherPool pool(4);
  EXPECT_FALSE(pool.closed());
  pool.close();
  EXPECT_TRUE(pool.closed());
  pool.close(); // second close must be a no-op, not a crash
  EXPECT_TRUE(pool.closed());
}

TEST(GatherPool, ZeroThreadsResolvesToHardwareConcurrency) {
  GatherPool pool(0);
  EXPECT_GE(pool.threadCount(), 1);
}

TEST(GatherPool, BorrowedPoolClosedAfterAdmissionRunsLaterDispatchInline) {
  for (unsigned threads : {1u, 4u}) {
    GatherPool pool(threads);
    std::promise<void> admitted;
    BlockingGate resume;
    auto transfer = std::async(std::launch::async, [&] {
      // Model the binding's initial closed() check, before it releases the GIL.
      EXPECT_FALSE(pool.closed());
      admitted.set_value();
      resume.arriveAndWait();
      int calls = 0;
      auto driver = std::this_thread::get_id();
      pool.parallelFor(0, 128, 1, [&](int64_t b, int64_t e) {
        EXPECT_EQ(std::this_thread::get_id(), driver);
        EXPECT_EQ(b, 0);
        EXPECT_EQ(e, 128);
        ++calls;
      });
      EXPECT_EQ(calls, 1);
    });
    admitted.get_future().wait();
    pool.close();
    resume.release();
    transfer.get();
  }
}

TEST(GatherPool, ConcurrentCloseWaitsForRunningWorkers) {
  GatherPool pool(4);
  BlockingGate jobs;
  std::atomic<int> completed{0};
  auto dispatch = std::async(std::launch::async, [&] {
    pool.parallelFor(0, 128, 1, [&](int64_t, int64_t) {
      jobs.arriveAndWait();
      ++completed;
    });
  });
  EXPECT_TRUE(jobs.waitForArrivals(4));
  auto closer = [&] {
    pool.close();
    EXPECT_EQ(completed.load(), 4);
  };
  auto first = std::async(std::launch::async, closer);
  auto second = std::async(std::launch::async, closer);
  EXPECT_EQ(first.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  jobs.release();
  dispatch.get();
  first.get();
  second.get();
  EXPECT_TRUE(pool.closed());
}

} // namespace
