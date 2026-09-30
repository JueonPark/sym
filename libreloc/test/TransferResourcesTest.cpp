#include "reloc/TransferResources.h"
#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "gtest/gtest.h"

namespace {
using namespace transfer_test;

TEST(TransferResources, RepeatedLayoutsAreExactAndOnlyResourcesAreRetained) {
  PadRegion pad{0, 2, 5};
  pad.fillBits = 0x12345678;
  const std::vector<BoundPlan> plans = {
      layout({17, 43}, {1, 17}, {43, 1}), layout({9, 13}, {1, 9}, {13, 1}, 1),
      layout({5, 7}, {7, 1}, {7, 1}, 4, {pad}),
      layout({3, 5, 7}, {7, 21, 1}, {35, 7, 1}),
      layout({9, 13}, {13, 1}, {1, 9}) // serialized, column-major destination
  };
  for (auto direction :
       {TransferDirection::HostToDevice, TransferDirection::DeviceToHost}) {
    auto m = std::make_shared<Metrics>();
    TransferContext context(std::make_unique<Backend>(m));
    std::vector<std::shared_ptr<Buffers>> outputs;
    std::vector<std::vector<uint8_t>> references;
    std::vector<std::weak_ptr<Buffers>> owners;
    for (const auto &b : plans) {
      int allocations = 0;
      for (unsigned round = 0; round < 3; ++round) {
        auto bytes = std::make_shared<Buffers>(b, round);
        auto req = request(b, bytes->src.data(), bytes->dst.data(), direction);
        std::vector<uint8_t> expected(b.totalBytes);
        executeH2D(b, bytes->src.data(), expected.data());
        TransferOptions options;
        options.chunkSizeOverride = 52;
        options.gatherThreads = 3;
        auto result = executeTransfer(req, context, options, bytes);
        ASSERT_FALSE(result.error)
            << (result.error ? result.error->message : "");
        EXPECT_EQ(result.completion, TransferCompletion::Complete);
        EXPECT_EQ(bytes->dst, expected);
        EXPECT_EQ(bytes.use_count(), 1); // context dropped successful owners
        if (round) {
          EXPECT_EQ(m->allocations, allocations);
        }
        allocations = m->allocations;
        owners.push_back(bytes);
        outputs.push_back(std::move(bytes));
        references.push_back(std::move(expected));
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i)
      EXPECT_EQ(outputs[i]->dst, references[i]);
    outputs.clear();
    for (auto &owner : owners)
      EXPECT_TRUE(owner.expired());
    EXPECT_EQ(context.stats().workerPoolCreations, 1u);
    EXPECT_EQ(m->quiesces, 0);
    EXPECT_EQ(context.close(), TransferCompletion::Complete);
    EXPECT_EQ(m->allocations, m->frees);
    EXPECT_EQ(m->destroyed, 1);
  }
}

TEST(TransferResources, GrowthKeepsBackendWorkersAndPerRequestStreamOrdering) {
  auto m = std::make_shared<Metrics>();
  TransferContext context(std::make_unique<Backend>(m));
  size_t round = 0;
  for (auto b : {layout({32, 16}, {16, 1}, {16, 1}),
                 layout({17, 128}, {128, 1}, {128, 1}),
                 layout({16, 8}, {8, 1}, {8, 1})}) {
    auto bytes = std::make_shared<Buffers>(b);
    auto req = request(b, bytes->src.data(), bytes->dst.data());
    TransferOptions options;
    options.nBuffers = 2;
    options.chunkSizeOverride = 128;
    options.gatherThreads = 3;
    options.hasCallerStream = true;
    options.callerStream = reinterpret_cast<void *>(++round);
    auto result = executeTransfer(req, context, options, bytes);
    ASSERT_FALSE(result.error);
    EXPECT_EQ(bytes->dst, bytes->src);
    auto again = executeTransfer(req, context, options, bytes);
    ASSERT_TRUE(again.error);
    EXPECT_EQ(again.error->code, "already_executed");
    EXPECT_TRUE(context.stats().reusable);
  }
  EXPECT_EQ(m->allocations, 4);
  EXPECT_EQ(m->frees, 2);
  EXPECT_EQ(m->peakBytes, 1024u); // old staging freed before larger allocation
  EXPECT_EQ(context.stats().slotBytes, 512u);
  EXPECT_EQ(context.stats().stagingBytes, 1024u);
  EXPECT_EQ(context.stats().workerPoolCreations, 1u);
  EXPECT_EQ(m->callerStreams,
            (std::vector<const void *>{reinterpret_cast<void *>(1),
                                       reinterpret_cast<void *>(2),
                                       reinterpret_cast<void *>(3)}));
  EXPECT_EQ(m->destroyed, 0);
}

TEST(TransferResources,
     LargeStagingAndParallelChunksExecuteWithActualCapacity) {
  for (const auto &b :
       {layout({1, 17 * 1024 * 1024}, {17 * 1024 * 1024, 1},
               {17 * 1024 * 1024, 1}),              // a single row above 64 MiB
        layout({2049, 8192}, {8192, 1}, {1, 2049}), // serialized, above 64 MiB
        layout({1025, 1027}, {1, 1025}, {1027, 1})}) {
    auto m = std::make_shared<Metrics>();
    TransferContext context(std::make_unique<Backend>(m));
    for (unsigned round = 0; round < 2; ++round) {
      auto bytes = std::make_shared<Buffers>(b, round);
      auto req = request(b, bytes->src.data(), bytes->dst.data());
      std::vector<uint8_t> expected(b.totalBytes);
      executeH2D(b, bytes->src.data(), expected.data());
      TransferOptions options;
      options.gatherThreads = 3;
      options.chunkSizeOverride = 2 * 1024 * 1024;
      auto result = executeTransfer(req, context, options, bytes);
      ASSERT_FALSE(result.error);
      EXPECT_EQ(bytes->dst, expected);
      EXPECT_EQ(context.stats().stagingPoolCreations, 1u);
      EXPECT_EQ(context.stats().workerPoolCreations, 1u);
    }
    EXPECT_EQ(context.close(), TransferCompletion::Complete);
    EXPECT_EQ(m->allocations, m->frees);
  }
}

TEST(TransferResources, ForwardD2HRevalidatesSourceSpanIncludingOffsetAndGaps) {
  auto b = layout({8}, {3}, {1});
  auto m = std::make_shared<Metrics>();
  TransferContext context(std::make_unique<Backend>(m));
  for (unsigned round = 0; round < 2; ++round) {
    auto bytes = std::make_shared<Buffers>(b, round);
    bytes->src.insert(bytes->src.begin(), 4, 0xAB);
    auto req = request(b, bytes->src.data(), bytes->dst.data(),
                       TransferDirection::DeviceToHost);
    req.source.capacityBytes = bytes->src.size();
    req.source.offsetBytes = 4;
    req.sourceSpanBytes = 1; // public cached fields cannot override validation
    std::vector<uint8_t> expected(b.totalBytes);
    executeH2D(b, bytes->src.data() + 4, expected.data());
    auto result = executeTransfer(req, context, {}, bytes);
    ASSERT_FALSE(result.error);
    EXPECT_EQ(bytes->dst, expected);
    EXPECT_EQ(context.stats().slotBytes, 88u);
    EXPECT_EQ(context.stats().activeSlots, 1);
  }
  EXPECT_EQ(m->allocations, 1);
  EXPECT_EQ(m->copyBytes, (std::vector<size_t>{88, 88}));
}

TEST(TransferResources, RejectionsLaunchNothingAndFailedConstructionRetires) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  auto bytes = std::make_shared<Buffers>(b);
  auto req = request(b, bytes->src.data(), bytes->dst.data());
  auto m = std::make_shared<Metrics>();
  auto backend = std::make_unique<Backend>(m);
  backend->failAllocation = 2;
  TransferContext context(std::move(backend));
  EXPECT_EQ(executeTransfer(req, context, {}, {}).error->code,
            "invalid_options");
  EXPECT_FALSE(req.consumed);
  EXPECT_EQ(m->allocations, 0);
  TransferOptions options;
  options.chunkSizeOverride = 52;
  auto result = executeTransfer(req, context, options, bytes);
  ASSERT_TRUE(result.error);
  EXPECT_EQ(result.completion, TransferCompletion::NotLaunched);
  EXPECT_TRUE(req.consumed);
  EXPECT_EQ(m->frees, 1);
  EXPECT_EQ(m->destroyed, 1);
  EXPECT_EQ(bytes.use_count(), 1);
  EXPECT_FALSE(context.stats().reusable);
  EXPECT_EQ(context.close(), TransferCompletion::Complete);
}

TEST(TransferResources, EventFailureKeepsOwnersUntilQuiescenceCompletes) {
  for (auto direction :
       {TransferDirection::HostToDevice, TransferDirection::DeviceToHost}) {
    auto b = layout({7, 13}, {1, 7}, {13, 1});
    auto bytes = std::make_shared<Buffers>(b);
    std::weak_ptr<Buffers> owner = bytes;
    auto req = request(b, bytes->src.data(), bytes->dst.data(), direction);
    auto m = std::make_shared<Metrics>();
    auto backend = std::make_unique<Backend>(m);
    backend->gateCopies();
    auto gate = backend->gate;
    backend->failEvent = true;
    TransferContext context(std::move(backend));
    auto run =
        std::async(std::launch::async, [&, bytes = std::move(bytes)]() mutable {
          return executeTransfer(req, context, {}, std::move(bytes));
        });
    EXPECT_TRUE(gate->waitForArrivals(1));
    EXPECT_EQ(run.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(m->frees, 0);
    gate->release();
    auto result = run.get();
    ASSERT_TRUE(result.error);
    EXPECT_EQ(result.completion, TransferCompletion::Complete);
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(m->copies, 1);
    EXPECT_EQ(m->allocations, m->frees);
    EXPECT_EQ(m->destroyed, 1);
  }
}

TEST(TransferResources, UnknownCompletionOutlivesContextAndDiscardedError) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  auto bytes = std::make_shared<Buffers>(b);
  std::weak_ptr<Buffers> owner = bytes;
  auto req = request(b, bytes->src.data(), bytes->dst.data());
  auto m = std::make_shared<Metrics>();
  auto backend = std::make_unique<Backend>(m);
  backend->gateCopies();
  auto gate = backend->gate;
  auto *quarantined = backend.get(); // observation only; ownership stays native
  backend->failEvent = backend->unknown = true;
  {
    TransferContext context(std::move(backend));
    {
      auto result = executeTransfer(req, context, {}, std::move(bytes));
      ASSERT_TRUE(result.error);
      EXPECT_EQ(result.error->code, "completion_unknown");
      EXPECT_EQ(result.completion, TransferCompletion::Unknown);
    }
    EXPECT_TRUE(context.stats().quarantined);
    EXPECT_GT(context.stats().stagingBytes, 0u);
    EXPECT_EQ(context.close(), TransferCompletion::Unknown);
  }
  EXPECT_FALSE(owner.expired());
  EXPECT_EQ(m->frees, 0);
  EXPECT_EQ(m->destroyed, 0);
  gate->release();
  quarantined->HostBackend::quiesce(); // finish the simulated DMA for test exit
  EXPECT_FALSE(owner.expired()); // no implicit recovery or late unsafe release
}

TEST(TransferResources, BorrowedWorkersAreNeitherClosedNorRetained) {
  auto b = layout({7, 13}, {1, 7}, {13, 1});
  auto bytes = std::make_shared<Buffers>(b);
  auto req = request(b, bytes->src.data(), bytes->dst.data());
  auto m = std::make_shared<Metrics>();
  TransferContext context(std::make_unique<Backend>(m));
  auto pool = std::make_shared<GatherPool>(3);
  std::weak_ptr<GatherPool> borrowed = pool;
  TransferOptions options;
  options.gatherThreads = 100;
  options.gather = pool.get();
  auto result = executeTransfer(req, context, options, bytes);
  ASSERT_FALSE(result.error);
  EXPECT_FALSE(pool->closed());
  EXPECT_EQ(context.stats().workerPoolCreations, 0u);
  EXPECT_EQ(context.stats().backgroundWorkers, 0u);
  pool.reset();
  EXPECT_TRUE(borrowed.expired());
  EXPECT_EQ(context.close(), TransferCompletion::Complete);
}
} // namespace
