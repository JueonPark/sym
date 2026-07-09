//===- BackendTest.cpp - HostBackend unit tests ---------------------------===//

#include "reloc/HostBackend.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using reloc::CopyDir;
using reloc::EventHandle;
using reloc::HostBackend;

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

} // namespace
