#include "../src/DispatchResourcePool.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <future>
#include <thread>

namespace reloc::dispatch {
struct ResourcePoolTestAccess {
  using Pool = Resources::Pool;
};
} // namespace reloc::dispatch
namespace {
using Pool = reloc::dispatch::ResourcePoolTestAccess::Pool;
using Error = reloc::TransferError;
using Lease = std::shared_ptr<Pool::Lease>;

Pool::Key key(int device, int streams = 1) {
  return {device, streams, 1, 0, {1}};
}
Lease acquire(const std::shared_ptr<Pool> &pool, Pool::Key k) {
  auto result = pool->acquire(std::move(k));
  if (auto *error = std::get_if<Error>(&result)) {
    ADD_FAILURE() << error->code << ": " << error->message;
    return {};
  }
  return std::get<Lease>(result);
}
bool queued(const std::shared_ptr<Pool> &pool, size_t count) {
  std::unique_lock<std::mutex> lock(pool->mu);
  return pool->cv.wait_for(lock, std::chrono::seconds(5),
                           [&] { return pool->waiters.size() == count; });
}

TEST(DispatchResourcePool, ReusesCompatibleSlotsAndEvictsLeastRecentlyUsed) {
  auto pool = std::make_shared<Pool>(8192, 16384, 6, 4, 2, 1, std::nullopt);
  auto first = acquire(pool, key(0));
  auto engine0 = first->resources;
  first.reset();
  auto second = acquire(pool, key(1));
  auto engine1 = second->resources;
  second.reset();
  first = acquire(pool, key(0));
  EXPECT_EQ(first->resources, engine0);
  first.reset();
  auto third = acquire(pool, key(2));
  EXPECT_EQ(third->resources, engine1);
  third.reset();
  auto stats = pool->stats();
  EXPECT_EQ(stats.evictions, 1u);
  ASSERT_EQ(stats.contextDetails.size(), 2u);
  std::vector<int> devices;
  for (const auto &context : stats.contextDetails)
    devices.push_back(context.device);
  std::sort(devices.begin(), devices.end());
  EXPECT_EQ(devices, (std::vector<int>{0, 2}));
  EXPECT_EQ(stats.contextDetails[0].retainedLimit, 4096u);
  EXPECT_EQ(stats.contextDetails[0].liveLimit, 8192u);
  EXPECT_EQ(stats.workerLimit, 6u);
  auto changed = acquire(pool, key(0, 2));
  EXPECT_EQ(changed->resources, engine0); // same-device resident limit is one
  EXPECT_EQ(pool->stats().evictions, 2u);
}

TEST(DispatchResourcePool, FairAdmissionDoesNotPermitNewcomersToBypassWaiters) {
  auto pool = std::make_shared<Pool>(0, 1024, 0, 1, 1, 1, std::nullopt);
  auto held = acquire(pool, key(0));
  std::vector<int> order;
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i] {
      auto lease = acquire(pool, key(i));
      if (lease)
        order.push_back(i); // exclusive lease synchronizes this vector
    });
    EXPECT_TRUE(queued(pool, i + 1));
  }
  held.reset();
  for (auto &t : threads)
    t.join();
  EXPECT_EQ(order, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(pool->stats().admissionWaits, 4u);
  EXPECT_EQ(pool->stats().peakActiveContexts, 1u);
}

TEST(DispatchResourcePool, CloseRejectsQueuedCallsAndDrainsActiveLease) {
  auto pool = std::make_shared<Pool>(0, 1024, 0, 1, 1, 1, std::nullopt);
  auto held = acquire(pool, key(0));
  auto waiter =
      std::async(std::launch::async, [&] { return pool->acquire(key(1)); });
  ASSERT_TRUE(queued(pool, 1));
  auto close =
      std::async(std::launch::async, [&] { return pool->drain(true); });
  auto result = waiter.get();
  ASSERT_TRUE(std::holds_alternative<Error>(result));
  EXPECT_EQ(std::get<Error>(result).code, "resources_closed");
  EXPECT_EQ(close.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  held.reset();
  EXPECT_FALSE(close.get());
  EXPECT_TRUE(pool->stats().closed);
  EXPECT_EQ(pool->stats().activeContexts, 0u);
  EXPECT_TRUE(pool->stats().contextDetails.empty());
}

TEST(DispatchResourcePool, TimeoutAndQuotaFailureLeaveNoQueuedOrActiveWork) {
  auto pool = std::make_shared<Pool>(0, 1024, 0, 2, 2, 1, 0);
  auto held = acquire(pool, key(0));
  auto blocked = pool->acquire(key(0)); // per-device limit despite a free slot
  ASSERT_TRUE(std::holds_alternative<Error>(blocked));
  EXPECT_EQ(std::get<Error>(blocked).code, "acquire_timeout");
  auto independent = acquire(pool, key(1));
  EXPECT_EQ(pool->stats().peakActiveContexts, 2u);
  auto excessive = pool->acquire(key(2, 2));
  ASSERT_TRUE(std::holds_alternative<Error>(excessive));
  EXPECT_EQ(std::get<Error>(excessive).code, "resource_limit");
  EXPECT_EQ(pool->stats().queued, 0u);
  held.reset();
  independent.reset();
  EXPECT_FALSE(pool->drain(false));
  EXPECT_FALSE(pool->stats().closed);
  EXPECT_TRUE(pool->stats().contextDetails.empty());
  EXPECT_TRUE(acquire(pool, key(0)));
}

TEST(DispatchResourcePool, AffinityAndWorkersArePartOfCompatibility) {
  auto pool = std::make_shared<Pool>(0, 2048, 2, 2, 2, 2, std::nullopt);
  auto a = key(0), b = a;
  b.threads = 2;
  b.affinity = {2};
  auto first = acquire(pool, a), second = acquire(pool, b);
  auto e0 = first->resources, e1 = second->resources;
  EXPECT_NE(e0, e1);
  first.reset();
  second.reset();
  EXPECT_EQ(acquire(pool, a)->resources, e0);
  EXPECT_EQ(acquire(pool, b)->resources, e1);
  EXPECT_EQ(pool->stats().evictions, 0u);
}

TEST(DispatchResourcePool, ClearPausesAdmissionThenResumesQueuedCalls) {
  auto pool = std::make_shared<Pool>(0, 1024, 0, 1, 1, 1, std::nullopt);
  auto held = acquire(pool, key(0));
  auto clear =
      std::async(std::launch::async, [&] { return pool->drain(false); });
  {
    std::unique_lock<std::mutex> lock(pool->mu);
    pool->cv.wait_for(lock, std::chrono::seconds(5),
                      [&] { return pool->clearing; });
  }
  auto waiter =
      std::async(std::launch::async, [&] { return acquire(pool, key(1)); });
  EXPECT_TRUE(queued(pool, 1));
  held.reset();
  EXPECT_FALSE(clear.get());
  auto resumed = waiter.get();
  EXPECT_TRUE(resumed);
  EXPECT_FALSE(pool->stats().closed);
}
} // namespace
