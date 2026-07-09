//===- PoolTest.cpp - PinnedBufferPool tests ------------------------------===//

#include "reloc/PinnedBufferPool.h"
#include "reloc/HostBackend.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {

using reloc::CopyDir;
using reloc::EventHandle;
using reloc::HostBackend;
using reloc::PinnedBufferPool;

TEST(Pool, AllocatesDistinctBuffersRoundRobin) {
  HostBackend backend(1);
  PinnedBufferPool pool(backend, 4, 128);
  EXPECT_EQ(pool.nBuffers(), 4);
  EXPECT_EQ(pool.bufferBytes(), 128u);
  // Round-robin: 0,1,2,3,0 with no pending events (never blocks).
  EXPECT_EQ(pool.acquire(), 0);
  EXPECT_EQ(pool.acquire(), 1);
  EXPECT_EQ(pool.acquire(), 2);
  EXPECT_EQ(pool.acquire(), 3);
  EXPECT_EQ(pool.acquire(), 0);
  // Distinct addresses.
  EXPECT_NE(pool.buffer(0), pool.buffer(1));
}

TEST(Pool, AcquireIsEventGated) {
  // Prove buffer reuse is gated on event completion: with a single buffer
  // whose event is held pending by a blocked copy hook, a second acquire()
  // must not return until the event fires.
  HostBackend backend(1);
  std::atomic<bool> release{false};
  std::atomic<bool> hookEntered{false};
  backend.setCopyHook([&] {
    hookEntered = true;
    while (!release.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  });

  PinnedBufferPool pool(backend, 1, 64);
  int i = pool.acquire(); // buffer 0, no prior event
  ASSERT_EQ(i, 0);

  // Simulate an in-flight copy out of buffer 0 and gate its reuse on the event.
  std::vector<uint8_t> devDst(64, 0);
  backend.copyAsync(0, devDst.data(), pool.buffer(0), 64,
                    CopyDir::HostToDevice);
  EventHandle ev = backend.recordEvent(0);
  pool.setEvent(0, ev);

  while (!hookEntered.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::atomic<bool> acquired{false};
  std::thread t([&] {
    pool.acquire(); // must block until ev completes
    acquired = true;
  });

  // While the event is pending, acquire must NOT have returned.
  for (int k = 0; k < 30; ++k) {
    EXPECT_FALSE(acquired.load());
    EXPECT_FALSE(backend.queryEvent(ev));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  release = true; // let the copy finish -> event fires -> acquire returns
  t.join();
  EXPECT_TRUE(acquired.load());
  EXPECT_TRUE(backend.queryEvent(ev));
}

TEST(Pool, DrainWaitsForAllEvents) {
  HostBackend backend(2);
  PinnedBufferPool pool(backend, 2, 64);
  std::vector<uint8_t> a(64, 0), b(64, 0);
  int i0 = pool.acquire();
  backend.copyAsync(0, a.data(), pool.buffer(i0), 64, CopyDir::HostToDevice);
  pool.setEvent(i0, backend.recordEvent(0));
  int i1 = pool.acquire();
  backend.copyAsync(1, b.data(), pool.buffer(i1), 64, CopyDir::HostToDevice);
  pool.setEvent(i1, backend.recordEvent(1));
  pool.drain(); // must return only after both events complete
  SUCCEED();
}

} // namespace
