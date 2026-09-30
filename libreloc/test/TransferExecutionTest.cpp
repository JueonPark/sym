#include "../src/TransferInternal.h"
#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "reloc/PinnedBufferPool.h"
#include "gtest/gtest.h"

#include <limits>

namespace {
using namespace transfer_test;

TEST(TransferExecution, RequirementsPreserveScheduleAndReduceOnlyActiveSlots) {
  auto b = layout({128, 16384}, {16384, 1}, {16384, 1}); // 8 MiB / 2 chunks
  auto req =
      request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(1));
  TransferOptions options;
  options.nBuffers = 4;
  auto described = detail::describeTransfer(req, options);
  ASSERT_TRUE(std::holds_alternative<detail::TransferRequirements>(described));
  const auto &r = std::get<detail::TransferRequirements>(described);
  const auto expected = planChunks(b, 4);
  EXPECT_EQ(r.activeSlots, 2);
  EXPECT_EQ(r.stagingBytes, 8u << 20);
  ASSERT_EQ(r.schedule.chunks.size(), expected.chunks.size());
  for (size_t i = 0; i < expected.chunks.size(); ++i) {
    EXPECT_EQ(r.schedule.chunks[i].byteOffset, expected.chunks[i].byteOffset);
    EXPECT_EQ(r.schedule.chunks[i].bytes, expected.chunks[i].bytes);
  }
}

TEST(TransferExecution, LargeRowsSerializedSchedulesAndHugeOverrideAreChecked) {
  const int64_t columns = kMaxChunkBytes / 4 + 1;
  for (auto strides :
       {std::vector<int64_t>{columns, 1}, std::vector<int64_t>{1, 2}}) {
    auto b = layout({2, columns}, {columns, 1}, strides);
    auto req =
        request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(1));
    auto described = detail::describeTransfer(req, {});
    ASSERT_TRUE(
        std::holds_alternative<detail::TransferRequirements>(described));
    auto r = std::get<detail::TransferRequirements>(described);
    EXPECT_GT(r.slotBytes, kMaxChunkBytes);
    EXPECT_EQ(r.activeSlots, strides[0] == 1 ? 1 : 2);
  }
  auto b = layout({std::numeric_limits<int64_t>::max()}, {1}, {1}, 1);
  auto req =
      request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(1));
  TransferOptions options;
  options.chunkSizeOverride = std::numeric_limits<size_t>::max();
  auto described = detail::describeTransfer(req, options);
  ASSERT_TRUE(std::holds_alternative<detail::TransferRequirements>(described));
  auto r = std::get<detail::TransferRequirements>(described);
  EXPECT_EQ(r.schedule.chunks.size(), 1u);
  EXPECT_EQ(r.slotBytes, size_t(b.totalBytes));

  b = layout({1, 4}, {4, 1}, {std::numeric_limits<int64_t>::max(), 1});
  req = request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(1));
  described = detail::describeTransfer(req, {});
  ASSERT_TRUE(std::holds_alternative<TransferError>(described));
  EXPECT_EQ(std::get<TransferError>(described).code, "integer_overflow");
}

TEST(TransferExecution,
     CheckedCoreRejectsUndersizedOrForeignStagingBeforeCopy) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  Buffers bytes(b);
  auto req = request(b, bytes.src.data(), bytes.dst.data());
  auto r =
      std::get<detail::TransferRequirements>(detail::describeTransfer(req, {}));
  auto m = std::make_shared<Metrics>();
  Backend backend(m), other(std::make_shared<Metrics>());
  for (bool foreign : {false, true}) {
    PinnedBufferPool pool(foreign ? other : backend, 1,
                          foreign ? r.slotBytes : r.slotBytes - 1);
    TransferCompletion completion = TransferCompletion::NotLaunched;
    auto error = detail::executePreparedTransfer(req, r, {}, backend, pool,
                                                 nullptr, completion);
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, "insufficient_capacity");
    EXPECT_EQ(completion, TransferCompletion::NotLaunched);
  }
  EXPECT_EQ(m->copies, 0);
  EXPECT_THROW(
      PinnedBufferPool(backend, 3, std::numeric_limits<size_t>::max() / 2),
      std::invalid_argument);
}

TEST(TransferExecution,
     AliasedSerializedDestinationIsRejectedBeforeAllocation) {
  auto b = layout({3, 3}, {3, 1}, {1, 1});
  Buffers bytes(b);
  auto req = request(b, bytes.src.data(), bytes.dst.data());
  auto m = std::make_shared<Metrics>();
  Backend backend(m);
  auto error = executeTransfer(req, backend, {});
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, "unsupported_layout");
  EXPECT_FALSE(req.consumed);
  EXPECT_EQ(m->allocations, 0);
}

TEST(TransferExecution, ErrorsAfterEnqueueDrainBeforeFreeAndNeverReplay) {
  for (int failure = 0; failure < 4; ++failure) {
    SCOPED_TRACE(failure);
    auto b = layout({7, 13}, {1, 7}, {13, 1});
    Buffers bytes(b);
    auto req = request(b, bytes.src.data(), bytes.dst.data());
    auto m = std::make_shared<Metrics>();
    Backend backend(m);
    backend.failEvent = failure == 0;
    backend.failCopy = failure == 1;
    backend.throwCopy = failure == 2;
    backend.failWait = failure == 3;
    backend.gateCopies();
    TransferOptions options;
    options.nBuffers = 1;
    options.chunkSizeOverride = 52;
    auto running = std::async(std::launch::async, [&] {
      return executeTransfer(req, backend, options);
    });
    EXPECT_TRUE(backend.gate->waitForArrivals(1));
    EXPECT_EQ(running.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);
    EXPECT_EQ(m->copies, 1);
    EXPECT_EQ(m->frees, 0);
    backend.gate->release();
    auto error = running.get();
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, "backend_failure");
    EXPECT_TRUE(req.consumed);
    EXPECT_EQ(m->copies, 1);
    EXPECT_EQ(m->quiesces, 1);
    EXPECT_EQ(m->frees, m->allocations);
  }
}

TEST(TransferExecution, SecondGatherOverlapsPendingCopyOnlyWithTwoBuffers) {
  for (int slots : {1, 2}) {
    auto b = layout({7, 13}, {1, 7}, {13, 1});
    Buffers bytes(b);
    std::vector<uint8_t> expected(b.totalBytes);
    executeH2D(b, bytes.src.data(), expected.data());
    auto req = request(b, bytes.src.data(), bytes.dst.data());
    auto m = std::make_shared<Metrics>();
    Backend backend(m);
    backend.gateCopies();
    auto secondCopy = m->secondCopy.get_future();
    TransferOptions options;
    options.nBuffers = slots;
    options.chunkSizeOverride = 104;
    auto running = std::async(std::launch::async, [&] {
      return executeTransfer(req, backend, options);
    });
    EXPECT_TRUE(backend.gate->waitForArrivals(1));
    // copyAsync is called only after the chunk's gather barrier. Seeing the
    // second submission while the first DMA is held proves producer overlap.
    EXPECT_EQ(secondCopy.wait_for(slots == 2 ? std::chrono::milliseconds(5000)
                                             : std::chrono::milliseconds(50)),
              slots == 2 ? std::future_status::ready
                         : std::future_status::timeout);
    backend.gate->release();
    EXPECT_FALSE(running.get());
    EXPECT_EQ(bytes.dst, expected);
    EXPECT_EQ(m->quiesces,
              0); // no extra synchronization on successful transfers
  }
}

TEST(TransferExecution,
     PartialConstructionRollsBackAndChangedViewsAreRechecked) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  Buffers bytes(b);
  for (bool throws : {false, true}) {
    auto m = std::make_shared<Metrics>();
    Backend backend(m);
    backend.failAllocation = 2;
    backend.throwAllocation = throws;
    TransferOptions options;
    options.chunkSizeOverride = 52;
    auto req = request(b, bytes.src.data(), bytes.dst.data());
    auto error = executeTransfer(req, backend, options);
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, "backend_failure");
    EXPECT_TRUE(req.consumed);
    EXPECT_EQ(m->allocations, 2);
    EXPECT_EQ(m->frees, 1);
    EXPECT_EQ(m->copies, 0);
    EXPECT_EQ(m->liveBytes, 0u);
  }
  auto req = request(b, bytes.src.data(), bytes.dst.data());
  --req.source.capacityBytes;
  auto m = std::make_shared<Metrics>();
  Backend backend(m);
  auto error = executeTransfer(req, backend, {});
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, "insufficient_capacity");
  EXPECT_FALSE(req.consumed);
  EXPECT_EQ(m->allocations, 0);
}
} // namespace
