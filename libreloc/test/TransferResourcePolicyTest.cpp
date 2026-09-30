#include "../src/TransferResourcePolicy.h"
#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "gtest/gtest.h"

#include <limits>

namespace {
using namespace transfer_test;
constexpr size_t quantum = 256u << 10;

TEST(TransferResourcePolicy, CapacityRoundingIsCheckedAndPreservesSchedule) {
  EXPECT_EQ(std::get<size_t>(detail::roundStagingCapacity(1)), quantum);
  EXPECT_EQ(std::get<size_t>(detail::roundStagingCapacity(quantum)), quantum);
  EXPECT_EQ(std::get<size_t>(detail::roundStagingCapacity(quantum + 1)),
            2 * quantum);
  for (size_t bytes : {size_t(0), std::numeric_limits<size_t>::max()})
    EXPECT_TRUE(std::holds_alternative<TransferError>(
        detail::roundStagingCapacity(bytes)));
  auto b = layout({128, 16384}, {16384, 1}, {16384, 1});
  auto req =
      request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(1));
  auto r = std::get<detail::CacheRequest>(
      detail::describeCachedTransfer(req, {}, {}));
  auto schedule = planChunks(b, 4);
  EXPECT_EQ(r.key.activeSlots, 2);
  EXPECT_EQ(r.bytes, 8u << 20);
  ASSERT_EQ(r.execution.schedule.chunks.size(), schedule.chunks.size());
  for (size_t i = 0; i < schedule.chunks.size(); ++i) {
    EXPECT_EQ(r.execution.schedule.chunks[i].byteOffset,
              schedule.chunks[i].byteOffset);
    EXPECT_EQ(r.execution.schedule.chunks[i].bytes, schedule.chunks[i].bytes);
  }
  TransferResourceLimits limits;
  limits.maxRetainedBytes = r.bytes - 1;
  EXPECT_FALSE(std::get<detail::CacheRequest>(
                   detail::describeCachedTransfer(req, {}, limits))
                   .cacheable);
  EXPECT_FALSE(req.consumed);
}

TEST(TransferResourcePolicy, KeysDescribeResourcesRatherThanRequestValues) {
  auto a = layout({17, 43}, {1, 17}, {43, 1});
  auto b = layout({9, 13}, {13, 1}, {13, 1}, 1);
  auto req =
      request(a, reinterpret_cast<void *>(1), reinterpret_cast<void *>(2));
  auto other =
      request(b, reinterpret_cast<void *>(3), reinterpret_cast<void *>(4));
  CachedTransferOptions options;
  auto key = std::get<detail::CacheRequest>(
                 detail::describeCachedTransfer(req, options, {}))
                 .key;
  options.transfer.callerStream = reinterpret_cast<void *>(100);
  options.transfer.hasCallerStream = true;
  EXPECT_TRUE(key == std::get<detail::CacheRequest>(
                         detail::describeCachedTransfer(other, options, {}))
                         .key);
  for (int field = 0; field < 5; ++field) {
    auto changed = options;
    auto changedReq = other;
    if (field == 0)
      changed.backend.streams = 3;
    if (field == 1)
      changed.placementTag = 1;
    if (field == 2)
      changed.transfer.gatherThreads = 2;
    if (field == 3)
      changed.gather = std::make_shared<GatherPool>(1);
    if (field == 4)
      changedReq.direction = TransferDirection::DeviceToHost;
    EXPECT_FALSE(key ==
                 std::get<detail::CacheRequest>(
                     detail::describeCachedTransfer(changedReq, changed, {}))
                     .key);
  }
  options.gather = std::make_shared<GatherPool>(2);
  options.transfer.gatherThreads = 100;
  auto borrowed = std::get<detail::CacheRequest>(
      detail::describeCachedTransfer(req, options, {}));
  EXPECT_EQ(borrowed.workers, 0u);
  EXPECT_EQ(borrowed.key.workers, detail::WorkerMode::Borrowed);
  options.gather->close();
  EXPECT_EQ(
      std::get<TransferError>(detail::describeCachedTransfer(req, options, {}))
          .code,
      "invalid_options");
}

TEST(TransferResourcePolicy,
     PreparedCapacityCanGrowWithoutReplanningOrRecreatingWorkers) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  auto m = std::make_shared<Metrics>();
  TransferContext context(std::make_unique<Backend>(m));
  CachedTransferOptions options;
  options.transfer.nBuffers = 2;
  options.transfer.chunkSizeOverride = 104;
  options.transfer.gatherThreads = 3;
  for (unsigned round = 0; round < 3; ++round) {
    auto bytes = std::make_shared<Buffers>(b, round);
    auto req = request(b, bytes->src.data(), bytes->dst.data());
    auto r = std::get<detail::CacheRequest>(
        detail::describeCachedTransfer(req, options, {}));
    size_t capacity = round == 0 ? quantum : 2 * quantum;
    detail::TransferContextAccess::prepare(context, req.direction, r.execution,
                                           capacity);
    EXPECT_EQ(context.stats().slotBytes, capacity);
    EXPECT_EQ(context.stats().workerPoolCreations, 1u);
    req.consumed = true;
    auto result = detail::TransferContextAccess::execute(
        req, context, r.options, r.execution, bytes);
    ASSERT_FALSE(result.error);
    std::vector<uint8_t> expected(b.totalBytes);
    executeH2D(b, bytes->src.data(), expected.data());
    EXPECT_EQ(bytes->dst, expected);
    EXPECT_EQ(bytes.use_count(), 1);
  }
  EXPECT_EQ(m->allocations, 4);
  EXPECT_EQ(m->peakBytes, 4 * quantum);
  for (size_t bytes : m->copyBytes)
    EXPECT_LE(bytes, 104u);
  EXPECT_EQ(context.stats().stagingPoolCreations, 2u);
  EXPECT_EQ(context.close(), TransferCompletion::Complete);
  EXPECT_EQ(m->allocations, m->frees);
}

TEST(TransferResourcePolicy,
     ForwardD2HCapacityUsesSourceReachAndDevicePreflight) {
  auto b = layout({8}, {3}, {1});
  auto req =
      request(b, reinterpret_cast<void *>(1), reinterpret_cast<void *>(2),
              TransferDirection::DeviceToHost);
  auto r = std::get<detail::CacheRequest>(
      detail::describeCachedTransfer(req, {}, {}));
  EXPECT_EQ(r.execution.sourceBytes, 88u);
  EXPECT_EQ(r.execution.slotBytes, 88u);
  EXPECT_EQ(r.slotCapacity, quantum);
  EXPECT_EQ(r.key.activeSlots, 1);
  req.source.kind = MemoryKind::Cuda;
  req.source.device = 1;
  EXPECT_EQ(
      std::get<TransferError>(detail::describeCachedTransfer(req, {}, {})).code,
      "device_mismatch");
}

TEST(TransferResourcePolicy, CallbackMarkerCoversWorkersInlineAndExceptions) {
  for (unsigned threads : {1u, 2u}) {
    GatherPool pool(threads);
    EXPECT_FALSE(GatherPool::inCallback());
    EXPECT_THROW(pool.parallelFor(0, 2, 1,
                                  [](int64_t begin, int64_t) {
                                    EXPECT_TRUE(GatherPool::inCallback());
                                    if (begin == 0)
                                      throw std::runtime_error("callback");
                                  }),
                 std::runtime_error);
    EXPECT_FALSE(GatherPool::inCallback());
    pool.close();
    pool.parallelFor(0, 2, 1, [](int64_t, int64_t) {
      EXPECT_TRUE(GatherPool::inCallback());
    });
    EXPECT_FALSE(GatherPool::inCallback());
  }
}
} // namespace
