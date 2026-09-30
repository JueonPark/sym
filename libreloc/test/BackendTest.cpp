//===- BackendTest.cpp - HostBackend unit tests ---------------------------===//

#include "BlockingGate.h"
#include "reloc/HostBackend.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

namespace {

using reloc::CopyDir;
using reloc::EventHandle;
using reloc::HostBackend;
using reloc::QueueCompletion;

class EventFailureHostBackend : public HostBackend {
public:
  using HostBackend::HostBackend;
  EventHandle recordEvent(int) override {
    error_ = "injected event recording failure";
    return 0;
  }
  bool failed() const override { return !error_.empty(); }
  const std::string &error() const override { return error_; }

private:
  std::string error_;
};

TEST(Backend, AllocCopyEventMovesBytes) {
  HostBackend backend(1);
  const size_t n = 256;
  void *staging = backend.allocStaging(n);
  ASSERT_NE(staging, nullptr);
  std::vector<uint8_t> src(n), dst(n, 0);
  for (size_t i = 0; i < n; ++i)
    src[i] = static_cast<uint8_t>(i * 3 + 1);
  std::memcpy(staging, src.data(), n);

  backend.copyAsync(0, dst.data(), staging, n, CopyDir::HostToDevice);
  EventHandle ev = backend.recordEvent(0);
  backend.waitEvent(ev);

  EXPECT_EQ(std::memcmp(dst.data(), src.data(), n), 0);
  EXPECT_TRUE(backend.queryEvent(ev));
  backend.freeStaging(staging);
}

TEST(Backend, ZeroEventIsAlwaysComplete) {
  HostBackend backend(1);
  backend.waitEvent(0); // must not hang
  EXPECT_TRUE(backend.queryEvent(0));
}

TEST(Backend, CopyHookGatesCompletion) {
  // A hook that blocks until released keeps the copy -- and therefore the
  // event recorded after it -- pending. This is the mechanism the pool's
  // event-gating test relies on.
  HostBackend backend(1);
  std::atomic<bool> release{false};
  std::atomic<bool> hookEntered{false};
  backend.setCopyHook([&] {
    hookEntered = true;
    while (!release.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });

  const size_t n = 64;
  void *staging = backend.allocStaging(n);
  std::vector<uint8_t> dst(n, 0);
  backend.copyAsync(0, dst.data(), staging, n, CopyDir::HostToDevice);
  EventHandle ev = backend.recordEvent(0);

  // Wait until the worker is parked inside the hook, then assert the event is
  // still pending (the copy has not finished).
  while (!hookEntered.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_FALSE(backend.queryEvent(ev));

  release = true;
  backend.waitEvent(ev);
  EXPECT_TRUE(backend.queryEvent(ev));
  backend.freeStaging(staging);
}

TEST(Backend, MultipleQueuesReportCount) {
  HostBackend backend(3);
  EXPECT_EQ(backend.numQueues(), 3);
}

TEST(Backend, QuiesceDrainsExecutingAndQueuedCopiesAfterEventFailure) {
  for (int queues : {1, 2}) {
    for (int copies : {1, 2}) {
      SCOPED_TRACE(::testing::Message()
                   << queues << " queues, " << copies << " copies each");
      // One copy leaves an empty deque while executing; two also leave queued
      // work. No successfully recorded event is available to prove completion.
      std::vector<int> src(queues * copies, 42), dst(src.size(), 0);
      EventFailureHostBackend backend(queues);
      BlockingGate gate;
      backend.setCopyHook([&] { gate.arriveAndWait(); });
      for (int q = 0; q < queues; ++q)
        for (int c = 0; c < copies; ++c) {
          int i = q * copies + c;
          backend.copyAsync(q, &dst[i], &src[i], sizeof(int),
                            CopyDir::HostToDevice);
        }
      EXPECT_TRUE(gate.waitForArrivals(queues));
      EXPECT_EQ(backend.recordEvent(0), 0u);
      EXPECT_TRUE(backend.failed());
      const auto originalError = backend.error();
      std::promise<void> entered;
      auto drained = std::async(std::launch::async, [&] {
        entered.set_value();
        return backend.quiesce();
      });
      entered.get_future().wait();
      EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(50)),
                std::future_status::timeout);
      gate.release();
      EXPECT_EQ(drained.get(), QueueCompletion::Complete);
      // Also join via real events so a failing quiesce regression cannot make
      // the test itself race when reading or releasing the copy buffers.
      for (int q = 0; q < queues; ++q)
        backend.waitEvent(backend.HostBackend::recordEvent(q));
      EXPECT_EQ(dst, src);
      EXPECT_EQ(backend.error(), originalError);
      EXPECT_EQ(backend.outstandingEvents(), 0u);
      EXPECT_EQ(backend.quiesce(), QueueCompletion::Complete);
    }
  }
}

TEST(Backend, SuccessfulWaitsBoundEventMetadataAndAllowRepeatedWaits) {
  HostBackend backend(3);
  for (int round = 0; round < 1000; ++round) {
    std::vector<EventHandle> events;
    for (int q = 0; q < backend.numQueues(); ++q)
      events.push_back(backend.recordEvent(q));
    EXPECT_EQ(backend.outstandingEvents(), events.size());
    for (size_t i = 0; i < events.size(); ++i) {
      backend.waitEvent(events[i]);
      EXPECT_EQ(backend.outstandingEvents(), events.size() - i - 1);
      backend.waitEvent(events[i]);
      EXPECT_TRUE(backend.queryEvent(events[i]));
    }
  }
}

TEST(Backend, ConcurrentWaitersCanObserveTheSameRetiredEvent) {
  int src = 7, dst = 0;
  HostBackend backend;
  BlockingGate gate;
  backend.setCopyHook([&] { gate.arriveAndWait(); });
  backend.copyAsync(0, &dst, &src, sizeof(int), CopyDir::HostToDevice);
  auto event = backend.recordEvent(0);
  EXPECT_TRUE(gate.waitForArrivals(1));
  auto first =
      std::async(std::launch::async, [&] { backend.waitEvent(event); });
  auto second =
      std::async(std::launch::async, [&] { backend.waitEvent(event); });
  gate.release();
  first.get();
  second.get();
  EXPECT_EQ(dst, src);
  EXPECT_EQ(backend.outstandingEvents(), 0u);
}

} // namespace
