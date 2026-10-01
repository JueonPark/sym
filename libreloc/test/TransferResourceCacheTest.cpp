#include "../src/TransferResourcePolicy.h"
#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/TransferResources.h"
#include "gtest/gtest.h"

#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace {
using namespace transfer_test;
constexpr size_t quantum = 256u << 10;

struct Physical {
  std::atomic<size_t> bytes{0}, peak{0};
};
struct ObservedBackend : Backend {
  ObservedBackend(std::shared_ptr<Metrics> m,
                  const TransferBackendConfig &config,
                  std::shared_ptr<Physical> physical)
      : Backend(std::move(m), config.streams), ordinal(config.device),
        physical(std::move(physical)) {}
  int ordinal;
  std::shared_ptr<Physical> physical;
  std::function<void()> beforeAllocation, beforeFree;
  std::map<void *, size_t> sizes;
  int device() const override { return ordinal; }
  void *allocStaging(size_t bytes) override {
    if (beforeAllocation)
      beforeAllocation();
    auto *p = Backend::allocStaging(bytes);
    if (p) {
      sizes[p] = bytes;
      size_t live = physical->bytes.fetch_add(bytes) + bytes;
      size_t peak = physical->peak.load();
      while (peak < live && !physical->peak.compare_exchange_weak(peak, live)) {
      }
    }
    return p;
  }
  void freeStaging(void *p) override {
    if (beforeFree)
      beforeFree();
    size_t bytes = sizes.at(p);
    Backend::freeStaging(p);
    sizes.erase(p);
    physical->bytes.fetch_sub(bytes);
  }
};
struct Factory {
  std::mutex mu;
  std::atomic<size_t> created{0};
  std::shared_ptr<Physical> physical = std::make_shared<Physical>();
  std::vector<std::shared_ptr<Metrics>> metrics;
  std::vector<ObservedBackend *> backends; // observation only while alive
  std::function<void(ObservedBackend &, size_t)> setup;
  std::unique_ptr<CopyBackend> operator()(const TransferBackendConfig &config) {
    auto id = created.fetch_add(1);
    auto m = std::make_shared<Metrics>();
    auto backend = std::make_unique<ObservedBackend>(m, config, physical);
    {
      std::lock_guard<std::mutex> lock(mu);
      if (metrics.size() <= id) {
        metrics.resize(id + 1);
        backends.resize(id + 1);
      }
      metrics[id] = m;
      backends[id] = backend.get();
    }
    if (setup)
      setup(*backend, id);
    return backend;
  }
  TransferBackendFactory callback() {
    return [this](const auto &config) { return (*this)(config); };
  }
  ObservedBackend *backend(size_t id) {
    std::lock_guard<std::mutex> lock(mu);
    return backends.at(id);
  }
  std::shared_ptr<Metrics> metric(size_t id) {
    std::lock_guard<std::mutex> lock(mu);
    return metrics.at(id);
  }
};

BoundPlan dense(size_t bytes) { return layout({int64_t(bytes / 4)}, {1}, {1}); }
CachedTransferOptions oneSlot() {
  CachedTransferOptions o;
  o.transfer.nBuffers = 1;
  return o;
}
void setDeviceView(TransferRequest &req, const CachedTransferOptions &options) {
  if (options.backend.kind == MemoryKind::Cuda) {
    auto &view = req.direction == TransferDirection::HostToDevice
                     ? req.destination
                     : req.source;
    view.kind = MemoryKind::Cuda;
    view.device = options.backend.device;
  }
}
TransferOutcome
execute(TransferResourceCache &cache, const BoundPlan &b,
        CachedTransferOptions options = oneSlot(), unsigned seed = 1,
        TransferDirection direction = TransferDirection::HostToDevice) {
  auto bytes = std::make_shared<Buffers>(b, seed);
  auto req = request(b, bytes->src.data(), bytes->dst.data(), direction);
  setDeviceView(req, options);
  std::vector<uint8_t> expected(b.totalBytes);
  executeH2D(b, bytes->src.data(), expected.data());
  auto result = executeTransferCached(req, cache, options, bytes);
  if (!result.error) {
    EXPECT_EQ(result.completion, TransferCompletion::Complete);
    EXPECT_EQ(bytes->dst, expected);
    EXPECT_EQ(bytes.use_count(), 1);
    EXPECT_TRUE(req.consumed);
  }
  return result;
}
bool until(const std::function<bool()> &predicate) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::yield();
  }
  return true;
}
void expectEmpty(const TransferResourceStats &s) {
  EXPECT_EQ(s.contexts, 0u);
  EXPECT_EQ(s.allocatedStagingBytes, 0u);
  EXPECT_EQ(s.reservedStagingBytes, 0u);
  EXPECT_EQ(s.backgroundWorkers, 0u);
  EXPECT_EQ(s.reservedWorkers, 0u);
  EXPECT_EQ(s.streams, 0u);
  EXPECT_EQ(s.outstandingEvents, 0u);
  EXPECT_EQ(s.waiters, 0u);
}

TEST(TransferResourceCache,
     LazyReuseProcessesCurrentDataAndRetainsOnlyResources) {
  Factory factory;
  TransferResourceCache cache({}, factory.callback());
  expectEmpty(cache.stats());
  EXPECT_EQ(factory.created, 0u);
  CachedTransferOptions options;
  options.transfer.nBuffers = 2;
  options.transfer.chunkSizeOverride = 104;
  options.transfer.gatherThreads = 3;
  for (unsigned round = 0; round < 8; ++round) {
    auto b = round % 2 ? layout({17, 43}, {1, 17}, {43, 1})
                       : layout({19, 29}, {1, 19}, {29, 1});
    auto result = execute(cache, b, options, round);
    ASSERT_FALSE(result.error) << result.error->message;
  }
  auto s = cache.stats();
  EXPECT_EQ(factory.created, 1u);
  EXPECT_EQ(s.requests, 8u);
  EXPECT_EQ(s.misses, 1u);
  EXPECT_EQ(s.hits, 7u);
  EXPECT_EQ(s.growths, 0u);
  EXPECT_EQ(s.stagingAllocations, 2u);
  EXPECT_EQ(s.streamCreations, 2u);
  EXPECT_EQ(s.workerCreations, 2u);
  EXPECT_EQ(s.allocatedStagingBytes, 2 * quantum);
  EXPECT_EQ(s.reservedStagingBytes, 0u);
  EXPECT_EQ(s.outstandingEvents, 0u);
  EXPECT_EQ(s.eventCreations, s.eventRetirements);
  EXPECT_FALSE(cache.close());
  expectEmpty(cache.stats());
  EXPECT_EQ(cache.stats().workerJoins, 2u);
  EXPECT_EQ(cache.stats().streamDestructions, 2u);
  EXPECT_EQ(factory.physical->bytes, 0u);
}

TEST(TransferResourceCache, GrowthKeepsFullReservationAcrossFreeAndAllocation) {
  Factory factory;
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  limits.maxLiveStagingBytes = 2 * quantum;
  TransferResourceCache cache(limits, factory.callback());
  auto options = oneSlot();
  options.transfer.gatherThreads = 3;
  ASSERT_FALSE(execute(cache, dense(quantum), options).error);
  BlockingGate freeing, allocating;
  auto *backend = factory.backend(0);
  backend->beforeFree = [&] { freeing.arriveAndWait(); };
  backend->beforeAllocation = [&] { allocating.arriveAndWait(); };
  auto growth = std::async(std::launch::async, [&] {
    return execute(cache, dense(2 * quantum), options);
  });
  EXPECT_TRUE(freeing.waitForArrivals(1));
  auto s = cache.stats();
  EXPECT_EQ(s.building, 1u);
  EXPECT_EQ(s.allocatedStagingBytes, quantum);
  EXPECT_EQ(s.reservedStagingBytes, quantum);
  auto waiter = std::async(std::launch::async, [&] {
    return execute(cache, dense(quantum), options, 3);
  });
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 1; }));
  freeing.release();
  EXPECT_TRUE(allocating.waitForArrivals(1));
  s = cache.stats();
  EXPECT_EQ(s.allocatedStagingBytes, 0u);
  EXPECT_EQ(s.reservedStagingBytes, 2 * quantum);
  EXPECT_EQ(s.backgroundWorkers, 2u);
  EXPECT_EQ(factory.created, 1u);
  allocating.release();
  EXPECT_FALSE(growth.get().error);
  EXPECT_FALSE(waiter.get().error);
  s = cache.stats();
  EXPECT_EQ(s.growths, 1u);
  EXPECT_EQ(s.workerCreations, 2u);
  EXPECT_EQ(s.streamCreations, 2u);
  EXPECT_EQ(s.peakLiveStagingBytes, 2 * quantum);
  EXPECT_EQ(factory.physical->peak, 2 * quantum);
  EXPECT_FALSE(cache.close());
}

TEST(TransferResourceCache, ChoosesSmallestSufficientCapacityAndEvictsIdleLru) {
  Factory factory;
  BlockingGate copy;
  factory.setup = [&](ObservedBackend &b, size_t id) {
    if (!id)
      b.setCopyHook([&] { copy.arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = limits.maxContextsPerDevice = 2;
  TransferResourceCache cache(limits, factory.callback());
  auto small =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(copy.waitForArrivals(1));
  EXPECT_FALSE(execute(cache, dense(quantum + 4)).error);
  copy.release();
  EXPECT_FALSE(small.get().error);
  EXPECT_FALSE(execute(cache, dense(128)).error);
  EXPECT_EQ(factory.metric(0)->copies, 2);
  EXPECT_EQ(factory.metric(1)->copies, 1);
  EXPECT_FALSE(execute(cache, dense(quantum + 8)).error);
  EXPECT_EQ(factory.metric(1)->copies, 2);
  auto changed = oneSlot();
  changed.placementTag = 1;
  EXPECT_FALSE(execute(cache, dense(64), changed).error);
  EXPECT_EQ(factory.metric(0)->destroyed, 1);
  EXPECT_EQ(factory.metric(1)->destroyed, 0);
  EXPECT_EQ(cache.stats().evictions, 1u);
  EXPECT_EQ(cache.stats().growths, 0u);
}

TEST(TransferResourceCache, WaitersAreFifoAndAContextIsExclusive) {
  Factory factory;
  BlockingGate copy;
  factory.setup = [&](ObservedBackend &b, size_t) {
    b.setCopyHook([&] { copy.arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  TransferResourceCache cache(limits, factory.callback());
  auto launch = [&](uintptr_t stream) {
    return std::async(std::launch::async, [&, stream] {
      auto options = oneSlot();
      options.transfer.hasCallerStream = true;
      options.transfer.callerStream = reinterpret_cast<void *>(stream);
      return execute(cache, dense(4096), options, stream);
    });
  };
  auto first = launch(1);
  EXPECT_TRUE(copy.waitForArrivals(1));
  auto second = launch(2);
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 1; }));
  auto third = launch(3);
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 2; }));
  EXPECT_EQ(factory.created, 1u);
  EXPECT_EQ(factory.metric(0)->copies, 1);
  EXPECT_EQ(cache.stats().leased, 1u);
  copy.release();
  EXPECT_FALSE(first.get().error);
  EXPECT_FALSE(second.get().error);
  EXPECT_FALSE(third.get().error);
  EXPECT_EQ(factory.metric(0)->callerStreams,
            (std::vector<const void *>{reinterpret_cast<void *>(1),
                                       reinterpret_cast<void *>(2),
                                       reinterpret_cast<void *>(3)}));
  EXPECT_EQ(cache.stats().waiters, 0u);
}

TEST(TransferResourceCache,
     ImpossibleLimitsRejectWithoutWaitingOrChangingThreads) {
  for (int constraint = 0; constraint < 4; ++constraint) {
    Factory factory;
    TransferResourceLimits limits;
    limits.acquireTimeout = std::chrono::milliseconds(10);
    auto options = oneSlot();
    if (constraint == 0)
      limits.maxContexts = 0;
    if (constraint == 1)
      limits.maxContextsPerDevice = 0;
    if (constraint == 2)
      limits.maxLiveStagingBytes = quantum - 1;
    if (constraint == 3) {
      limits.maxBackgroundWorkers = 1;
      options.transfer.gatherThreads = 3;
    }
    TransferResourceCache cache(limits, factory.callback());
    auto b = dense(64);
    auto bytes = std::make_shared<Buffers>(b);
    auto req = request(b, bytes->src.data(), bytes->dst.data());
    auto result = executeTransferCached(req, cache, options, bytes);
    ASSERT_TRUE(result.error);
    EXPECT_EQ(result.error->code, "resource_limit");
    EXPECT_EQ(result.completion, TransferCompletion::NotLaunched);
    EXPECT_TRUE(req.consumed);
    EXPECT_EQ(cache.stats().waits, 0u);
    EXPECT_EQ(cache.stats().failures, 1u);
    EXPECT_EQ(factory.created, 0u);
    expectEmpty(cache.stats());
  }
}

TEST(TransferResourceCache, TimeoutRemovesWaiterAndLaterCallsStillWork) {
  Factory factory;
  BlockingGate copy;
  factory.setup = [&](ObservedBackend &b, size_t) {
    b.setCopyHook([&] { copy.arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  limits.acquireTimeout = std::chrono::milliseconds(50);
  TransferResourceCache cache(limits, factory.callback());
  auto first =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(copy.waitForArrivals(1));
  auto result = execute(cache, dense(64));
  ASSERT_TRUE(result.error);
  EXPECT_EQ(result.error->code, "resource_timeout");
  EXPECT_EQ(cache.stats().timeouts, 1u);
  EXPECT_EQ(cache.stats().waiters, 0u);
  copy.release();
  EXPECT_FALSE(first.get().error);
  EXPECT_FALSE(execute(cache, dense(64)).error);
  EXPECT_EQ(factory.created, 1u);
}

TEST(TransferResourceCache, ClearRetiresOldBuildingAndExecutingGenerations) {
  for (bool building : {true, false}) {
    Factory factory;
    BlockingGate gate;
    factory.setup = [&](ObservedBackend &b, size_t id) {
      if (id)
        return;
      if (building)
        b.beforeAllocation = [&] { gate.arriveAndWait(); };
      else
        b.setCopyHook([&] { gate.arriveAndWait(); });
    };
    TransferResourceCache cache({}, factory.callback());
    auto call = std::async(std::launch::async,
                           [&] { return execute(cache, dense(64)); });
    EXPECT_TRUE(gate.waitForArrivals(1));
    EXPECT_FALSE(cache.clear());
    EXPECT_EQ(cache.stats().generation, 1u);
    EXPECT_EQ(call.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    gate.release();
    EXPECT_FALSE(call.get().error);
    expectEmpty(cache.stats());
    EXPECT_FALSE(execute(cache, dense(64)).error);
    EXPECT_EQ(factory.created, 2u);
    EXPECT_EQ(cache.stats().idle, 1u);
  }
}

TEST(TransferResourceCache, CloseWaitsForConstructionAndRejectsItsPublication) {
  Factory factory;
  BlockingGate allocation;
  factory.setup = [&](ObservedBackend &b, size_t) {
    b.beforeAllocation = [&] { allocation.arriveAndWait(); };
  };
  TransferResourceCache cache({}, factory.callback());
  auto call =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(allocation.waitForArrivals(1));
  auto close = std::async(std::launch::async, [&] { return cache.close(); });
  EXPECT_TRUE(until([&] { return cache.stats().closed; }));
  EXPECT_EQ(close.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  EXPECT_EQ(execute(cache, dense(64)).error->code, "resources_closed");
  allocation.release();
  auto result = call.get();
  ASSERT_TRUE(result.error);
  EXPECT_EQ(result.error->code, "resources_closed");
  EXPECT_EQ(factory.metric(0)->copies, 0);
  EXPECT_FALSE(close.get());
  expectEmpty(cache.stats());
  EXPECT_FALSE(cache.close());
}

TEST(TransferResourceCache, CloseWakesWaitersAndWaitsForTheActiveLease) {
  Factory factory;
  BlockingGate copy;
  factory.setup = [&](ObservedBackend &b, size_t) {
    b.setCopyHook([&] { copy.arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  TransferResourceCache cache(limits, factory.callback());
  auto call =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(copy.waitForArrivals(1));
  auto waiter =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 1; }));
  auto close = std::async(std::launch::async, [&] { return cache.close(); });
  auto rejected = waiter.get();
  ASSERT_TRUE(rejected.error);
  EXPECT_EQ(rejected.error->code, "resources_closed");
  EXPECT_EQ(close.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  EXPECT_EQ(factory.metric(0)->frees, 0);
  copy.release();
  EXPECT_FALSE(call.get().error);
  EXPECT_FALSE(close.get());
  expectEmpty(cache.stats());
}

TEST(TransferResourceCache,
     EvictionKeepsRetiringCapacityChargedUntilFreeFinishes) {
  Factory factory;
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  limits.maxRetainedBytes = quantum;
  limits.maxLiveStagingBytes = quantum;
  TransferResourceCache cache(limits, factory.callback());
  ASSERT_FALSE(execute(cache, dense(64)).error);
  BlockingGate freeing;
  factory.backend(0)->beforeFree = [&] { freeing.arriveAndWait(); };
  auto next = std::async(std::launch::async, [&] {
    auto options = oneSlot();
    options.placementTag = 1;
    return execute(cache, dense(64), options);
  });
  EXPECT_TRUE(freeing.waitForArrivals(1));
  auto s = cache.stats();
  EXPECT_EQ(s.retiring, 1u);
  EXPECT_EQ(s.allocatedStagingBytes, quantum);
  EXPECT_EQ(s.retainedAllocatedBytes, quantum);
  EXPECT_EQ(factory.created, 1u);
  freeing.release();
  EXPECT_FALSE(next.get().error);
  EXPECT_EQ(factory.created, 2u);
  EXPECT_EQ(factory.physical->peak, quantum);
}

TEST(TransferResourceCache, OversizeRequestsUseAccountedEphemeralContexts) {
  Factory factory;
  TransferResourceLimits limits;
  limits.maxRetainedBytes = quantum;
  limits.maxLiveStagingBytes = 2 * quantum;
  TransferResourceCache cache(limits, factory.callback());
  ASSERT_FALSE(execute(cache, dense(64)).error);
  for (unsigned seed = 0; seed < 3; ++seed) {
    EXPECT_FALSE(execute(cache, dense(quantum + 4), oneSlot(), seed,
                         TransferDirection::DeviceToHost)
                     .error);
    expectEmpty(cache.stats());
  }
  EXPECT_EQ(cache.stats().ephemeral, 3u);
  EXPECT_EQ(cache.stats().evictions, 1u);
  EXPECT_EQ(factory.created, 4u);
  EXPECT_EQ(factory.physical->peak, 2 * quantum);
  EXPECT_EQ(factory.physical->bytes, 0u);
}

TEST(TransferResourceCache, PartialConstructionRollsBackEveryReservation) {
  for (int failure = 0; failure < 3; ++failure) {
    Factory factory;
    BlockingGate secondAllocation;
    factory.setup = [&](ObservedBackend &b, size_t id) {
      if (id)
        return;
      if (failure == 0)
        throw std::runtime_error("factory failure");
      b.failAllocation = 2;
      b.throwAllocation = failure == 2;
      b.beforeAllocation = [&, observed = &b] {
        if (observed->m->allocations == 1)
          secondAllocation.arriveAndWait();
      };
    };
    TransferResourceCache cache({}, factory.callback());
    CachedTransferOptions options;
    options.transfer.nBuffers = 2;
    options.transfer.chunkSizeOverride = 104;
    options.transfer.gatherThreads = 3;
    auto b = layout({7, 13}, {1, 7}, {13, 1});
    auto call = std::async(std::launch::async,
                           [&] { return execute(cache, b, options); });
    if (failure) {
      EXPECT_TRUE(secondAllocation.waitForArrivals(1));
      auto s = cache.stats();
      EXPECT_EQ(s.building, 1u);
      EXPECT_EQ(s.allocatedStagingBytes, quantum);
      EXPECT_EQ(s.reservedStagingBytes, quantum);
      EXPECT_EQ(s.reservedWorkers, 2u);
      secondAllocation.release();
    }
    auto result = call.get();
    ASSERT_TRUE(result.error);
    EXPECT_EQ(result.error->code, "backend_failure");
    expectEmpty(cache.stats());
    EXPECT_EQ(factory.physical->bytes, 0u);
    EXPECT_EQ(factory.metric(0)->destroyed, 1);
    EXPECT_FALSE(execute(cache, b, options).error);
    EXPECT_EQ(factory.created, 2u);
  }
}

TEST(TransferResourceCache, PerDeviceLimitsAllowIndependentActiveDevices) {
  Factory factory;
  BlockingGate firstCopy, secondCopy;
  factory.setup = [&](ObservedBackend &b, size_t) {
    auto *gate = b.ordinal == 0 ? &firstCopy : &secondCopy;
    b.setCopyHook([gate] { gate->arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = 2;
  limits.maxContextsPerDevice = 1;
  TransferResourceCache cache(limits, factory.callback());
  auto options = oneSlot();
  options.backend.kind = MemoryKind::Cuda;
  options.backend.device = 0;
  auto first = std::async(std::launch::async,
                          [&] { return execute(cache, dense(64), options); });
  EXPECT_TRUE(firstCopy.waitForArrivals(1));
  auto other = options;
  other.backend.device = 1;
  auto second = std::async(std::launch::async,
                           [&] { return execute(cache, dense(64), other); });
  EXPECT_TRUE(secondCopy.waitForArrivals(1));
  auto third = std::async(std::launch::async, [&] {
    return execute(cache, dense(64), options, 3);
  });
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 1; }));
  EXPECT_EQ(cache.stats().leased, 2u);
  EXPECT_EQ(factory.created, 2u);
  firstCopy.release();
  EXPECT_FALSE(first.get().error);
  EXPECT_FALSE(third.get().error);
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(10)),
            std::future_status::timeout);
  for (const auto &device : cache.stats().devices)
    EXPECT_EQ(device.contexts, 1u);
  secondCopy.release();
  EXPECT_FALSE(second.get().error);
}

TEST(TransferResourceCache,
     QuarantineStaysChargedDisablesDeviceAndOutlivesCache) {
  Factory factory;
  factory.setup = [&](ObservedBackend &b, size_t id) {
    if (!id) {
      b.gateCopies();
      b.failEvent = b.unknown = true;
    }
  };
  TransferResourceLimits limits;
  limits.maxContexts = 2;
  limits.maxBackgroundWorkers = 2;
  limits.maxLiveStagingBytes = 2 * quantum;
  auto cache =
      std::make_unique<TransferResourceCache>(limits, factory.callback());
  auto b = dense(64);
  auto bytes = std::make_shared<Buffers>(b);
  std::weak_ptr<Buffers> owners = bytes;
  auto req = request(b, bytes->src.data(), bytes->dst.data());
  auto options = oneSlot();
  options.transfer.gatherThreads = 3;
  options.backend = {MemoryKind::Cuda, 0, 2};
  setDeviceView(req, options);
  auto call =
      std::async(std::launch::async, [&, bytes = std::move(bytes)]() mutable {
        return executeTransferCached(req, *cache, options, std::move(bytes));
      });
  EXPECT_TRUE(until([&] {
    return factory.created == 1 && cache->stats().streamCreations == 2;
  }));
  auto *quarantined = factory.backend(0);
  auto gate = quarantined->gate;
  EXPECT_TRUE(gate->waitForArrivals(1));
  auto status = call.wait_for(std::chrono::seconds(5));
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready)
    gate->release();
  {
    auto result = call.get();
    ASSERT_TRUE(result.error);
    EXPECT_EQ(result.error->code, "completion_unknown");
    EXPECT_EQ(result.completion, TransferCompletion::Unknown);
  }
  auto s = cache->stats();
  EXPECT_EQ(s.quarantined, 1u);
  EXPECT_EQ(s.quarantineBytes, quantum);
  EXPECT_EQ(s.backgroundWorkers, 2u);
  ASSERT_EQ(s.devices.size(), 1u);
  EXPECT_TRUE(s.devices[0].disabled);
  EXPECT_EQ(execute(*cache, b, options).error->code, "resources_disabled");
  options.backend.device = 1;
  EXPECT_EQ(execute(*cache, b, options).error->code, "resource_limit");
  options.transfer.gatherThreads = 1;
  EXPECT_FALSE(execute(*cache, b, options).error);
  EXPECT_FALSE(cache->clear());
  EXPECT_EQ(cache->stats().contexts, 1u);
  EXPECT_EQ(cache->close()->code, "completion_unknown");
  EXPECT_FALSE(owners.expired());
  EXPECT_EQ(factory.metric(0)->frees, 0);
  EXPECT_EQ(factory.metric(0)->destroyed, 0);
  gate->release();
  if (factory.metric(0)->destroyed == 0)
    quarantined->HostBackend::quiesce();
  cache.reset();
  EXPECT_FALSE(owners.expired());
  EXPECT_EQ(factory.physical->bytes, quantum);
}

TEST(TransferResourceCache,
     BorrowedWorkersAndRecursiveCallsHaveExplicitLifetimes) {
  TransferResourceCache cache;
  auto options = oneSlot();
  options.gather = std::make_shared<GatherPool>(2);
  std::weak_ptr<GatherPool> owner = options.gather;
  options.transfer.gatherThreads = 100;
  EXPECT_FALSE(execute(cache, dense(64), options).error);
  EXPECT_EQ(cache.stats().workerCreations, 0u);
  options.gather->parallelFor(0, 2, 1, [&](int64_t, int64_t) {
    EXPECT_EQ(execute(cache, dense(64)).error->code, "resource_reentrant");
    EXPECT_EQ(cache.close()->code, "resource_reentrant");
  });
  options.gather.reset();
  EXPECT_TRUE(owner.expired());
  EXPECT_FALSE(cache.close());

  Factory factory;
  TransferResourceCache recursive({}, factory.callback());
  factory.setup = [&](ObservedBackend &, size_t) {
    EXPECT_EQ(execute(recursive, dense(64)).error->code, "resource_reentrant");
    EXPECT_EQ(recursive.clear()->code, "resource_reentrant");
    EXPECT_EQ(recursive.stats().building,
              1u); // callbacks run without cache lock
  };
  EXPECT_FALSE(execute(recursive, dense(64)).error);
}

TEST(TransferResourceCache,
     ForkInheritedCacheRejectsBeforeLocksAndCanBeAbandoned) {
  auto cache = std::make_unique<TransferResourceCache>();
  ASSERT_FALSE(execute(*cache, dense(64)).error);
  auto identity = detail::captureProcessIdentity();
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    auto after = detail::currentProcessIdentity();
    bool ok = after.pid != identity.pid && after.epoch != identity.epoch;
    ok &= !cache->stats().processValid;
    ok &= cache->close()->code == "process_mismatch";
    ok &= cache->clear()->code == "process_mismatch";
    ok &= execute(*cache, dense(64)).error->code == "process_mismatch";
    cache.reset(); // must not destroy inherited workers or enter their locks
    _exit(ok ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_FALSE(execute(*cache, dense(64)).error);
  EXPECT_TRUE(cache->stats().processValid);
}

TEST(TransferResourceCache,
     PreflightRejectsWithoutClaimButClosedAdmissionConsumes) {
  TransferResourceCache cache;
  auto b = dense(64);
  auto bytes = std::make_shared<Buffers>(b);
  auto req = request(b, bytes->src.data(), bytes->dst.data());
  EXPECT_EQ(executeTransferCached(req, cache, {}, {}).error->code,
            "invalid_options");
  EXPECT_FALSE(req.consumed);
  auto options = oneSlot();
  options.backend.device = 0;
  EXPECT_EQ(executeTransferCached(req, cache, options, bytes).error->code,
            "invalid_options");
  EXPECT_FALSE(req.consumed);
  EXPECT_EQ(cache.stats().requests, 0u);
  EXPECT_FALSE(executeTransferCached(req, cache, {}, bytes).error);
  EXPECT_EQ(executeTransferCached(req, cache, {}, bytes).error->code,
            "already_executed");
  EXPECT_EQ(cache.stats().requests, 1u);
  EXPECT_FALSE(cache.close());
  auto next = request(b, bytes->src.data(), bytes->dst.data());
  EXPECT_EQ(executeTransferCached(next, cache, {}, bytes).error->code,
            "resources_closed");
  EXPECT_TRUE(next.consumed);
}

TEST(TransferResourceCache,
     BorrowedPoolCanCloseWhileItsRequestWaitsForAdmission) {
  Factory factory;
  BlockingGate copy;
  factory.setup = [&](ObservedBackend &b, size_t id) {
    if (!id)
      b.setCopyHook([&] { copy.arriveAndWait(); });
  };
  TransferResourceLimits limits;
  limits.maxContexts = 1;
  TransferResourceCache cache(limits, factory.callback());
  auto active =
      std::async(std::launch::async, [&] { return execute(cache, dense(64)); });
  EXPECT_TRUE(copy.waitForArrivals(1));
  auto options = oneSlot();
  options.gather = std::make_shared<GatherPool>(2);
  auto borrowed = std::async(std::launch::async, [&] {
    return execute(cache, layout({1024, 1024}, {1, 1024}, {1024, 1}), options);
  });
  EXPECT_TRUE(until([&] { return cache.stats().waiters == 1; }));
  options.gather->close();
  copy.release();
  EXPECT_FALSE(active.get().error);
  EXPECT_FALSE(borrowed.get().error);
  EXPECT_EQ(cache.stats().workerCreations, 0u);
}

TEST(TransferResourceCache, CachedExecutionPreservesTwoBufferOverlap) {
  for (int slots : {1, 2}) {
    Factory factory;
    BlockingGate copy;
    factory.setup = [&](ObservedBackend &b, size_t) {
      b.setCopyHook([&] { copy.arriveAndWait(); });
    };
    TransferResourceCache cache({}, factory.callback());
    auto options = oneSlot();
    options.transfer.nBuffers = slots;
    options.transfer.chunkSizeOverride = 104;
    auto run = std::async(std::launch::async, [&] {
      return execute(cache, layout({7, 13}, {1, 7}, {13, 1}), options);
    });
    EXPECT_TRUE(copy.waitForArrivals(1));
    auto secondCopy = factory.metric(0)->secondCopy.get_future();
    EXPECT_EQ(secondCopy.wait_for(slots == 2 ? std::chrono::milliseconds(5000)
                                             : std::chrono::milliseconds(30)),
              slots == 2 ? std::future_status::ready
                         : std::future_status::timeout);
    copy.release();
    EXPECT_FALSE(run.get().error);
    EXPECT_EQ(factory.metric(0)->quiesces, 0);
  }
}

#ifdef __linux__
TEST(TransferResourceCache, ActualCallerAffinityIsPartOfCompatibility) {
  struct AffinityGuard {
    cpu_set_t saved;
    bool valid = false;
    ~AffinityGuard() {
      if (valid)
        sched_setaffinity(0, sizeof(saved), &saved);
    }
  } guard;
  ASSERT_EQ(sched_getaffinity(0, sizeof(guard.saved), &guard.saved), 0);
  guard.valid = true;
  std::vector<int> cpus;
  for (int i = 0; i < CPU_SETSIZE && cpus.size() < 2; ++i)
    if (CPU_ISSET(i, &guard.saved))
      cpus.push_back(i);
  if (cpus.size() < 2)
    GTEST_SKIP() << "needs two allowed CPUs";
  Factory factory;
  TransferResourceCache cache({}, factory.callback());
  auto options = oneSlot();
  options.transfer.gatherThreads = 3;
  for (int cpu : {cpus[0], cpus[1], cpus[0]}) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    ASSERT_EQ(sched_setaffinity(0, sizeof(mask), &mask), 0);
    EXPECT_FALSE(execute(cache, dense(64), options).error);
  }
  EXPECT_EQ(factory.created, 2u);
  EXPECT_EQ(cache.stats().hits, 1u);
  EXPECT_EQ(cache.stats().workerCreations, 4u);
}
#endif

TEST(TransferResourceCache,
     ConcurrentChangingShapesStayWithinPhysicalAndReservedLimits) {
  Factory factory;
  TransferResourceLimits limits;
  limits.maxContexts = 3;
  limits.maxContextsPerDevice = 2;
  limits.maxBackgroundWorkers = 4;
  limits.maxRetainedBytes = 4 * quantum;
  limits.maxLiveStagingBytes = 8 * quantum;
  TransferResourceCache cache(limits, factory.callback());
  std::vector<std::future<void>> calls;
  for (unsigned thread = 0; thread < 6; ++thread)
    calls.push_back(std::async(std::launch::async, [&, thread] {
      for (unsigned round = 0; round < 20; ++round) {
        auto options = oneSlot();
        options.placementTag = (thread + round) % 3;
        options.transfer.gatherThreads = round % 2 ? 3 : 1;
        size_t bytes = (1 + (thread + round) % 6) * quantum;
        auto result = execute(cache, dense(bytes), options, round + thread,
                              round % 2 ? TransferDirection::DeviceToHost
                                        : TransferDirection::HostToDevice);
        EXPECT_FALSE(result.error)
            << (result.error ? result.error->message : "");
        auto s = cache.stats();
        EXPECT_LE(s.contexts, 2u);
        EXPECT_LE(s.allocatedStagingBytes + s.reservedStagingBytes,
                  8 * quantum);
        EXPECT_LE(s.retainedAllocatedBytes + s.retainedReservedBytes,
                  4 * quantum);
        EXPECT_LE(s.backgroundWorkers + s.reservedWorkers, 4u);
      }
    }));
  for (auto &call : calls)
    call.get();
  EXPECT_EQ(cache.stats().requests, 120u);
  EXPECT_EQ(cache.stats().failures, 0u);
  EXPECT_LE(factory.physical->peak, 8 * quantum);
  EXPECT_FALSE(cache.close());
  expectEmpty(cache.stats());
  EXPECT_EQ(factory.physical->bytes, 0u);
}

TEST(TransferResourceCache, ClearAndCloseDuringGrowthRetireTheOldGeneration) {
  for (bool close : {false, true}) {
    Factory factory;
    TransferResourceLimits limits;
    limits.maxContexts = 1;
    TransferResourceCache cache(limits, factory.callback());
    ASSERT_FALSE(execute(cache, dense(quantum)).error);
    BlockingGate allocating;
    factory.backend(0)->beforeAllocation = [&] { allocating.arriveAndWait(); };
    auto growth = std::async(
        std::launch::async, [&] { return execute(cache, dense(2 * quantum)); });
    EXPECT_TRUE(allocating.waitForArrivals(1));
    EXPECT_EQ(cache.stats().building, 1u);
    EXPECT_EQ(cache.stats().reservedStagingBytes, 2 * quantum);
    auto retiring = std::async(std::launch::async, [&] {
      return close ? cache.close() : cache.clear();
    });
    EXPECT_TRUE(until([&] {
      auto s = cache.stats();
      return close ? s.closed : s.generation == 1;
    }));
    if (close) {
      EXPECT_EQ(retiring.wait_for(std::chrono::milliseconds(10)),
                std::future_status::timeout);
    }
    allocating.release();
    auto result = growth.get();
    EXPECT_FALSE(retiring.get());
    if (close) {
      ASSERT_TRUE(result.error);
      EXPECT_EQ(result.error->code, "resources_closed");
      EXPECT_EQ(factory.metric(0)->copies, 1); // only the original warmup
    } else {
      EXPECT_FALSE(result.error);
    }
    expectEmpty(cache.stats());
    EXPECT_EQ(factory.physical->bytes, 0u);
    EXPECT_EQ(factory.metric(0)->allocations, factory.metric(0)->frees);
  }
}

TEST(TransferResourceCache,
     WarmedCopyFailuresDrainBeforeRetiringAndNeverReplay) {
  for (int failure = 0; failure < 4; ++failure) {
    SCOPED_TRACE(failure);
    Factory factory;
    TransferResourceCache cache({}, factory.callback());
    auto b = layout({17, 43}, {1, 17}, {43, 1});
    auto options = oneSlot();
    ASSERT_FALSE(execute(cache, b, options).error);
    auto *backend = factory.backend(0);
    backend->failCopy = failure == 0;
    backend->throwCopy = failure == 1;
    backend->failEvent = failure == 2;
    backend->failWait = failure == 3;
    backend->gateCopies();
    auto gate = backend->gate;
    auto bytes = std::make_shared<Buffers>(b, 9);
    std::weak_ptr<Buffers> owners = bytes;
    auto req = request(b, bytes->src.data(), bytes->dst.data());
    auto running =
        std::async(std::launch::async, [&, bytes = std::move(bytes)]() mutable {
          return executeTransferCached(req, cache, options, std::move(bytes));
        });
    EXPECT_TRUE(gate->waitForArrivals(1));
    EXPECT_EQ(running.wait_for(std::chrono::milliseconds(10)),
              std::future_status::timeout);
    EXPECT_FALSE(owners.expired());
    EXPECT_EQ(factory.metric(0)->frees, 0);
    gate->release();
    auto result = running.get();
    ASSERT_TRUE(result.error);
    EXPECT_EQ(result.completion, TransferCompletion::Complete);
    EXPECT_EQ(factory.metric(0)->copies, 2); // warmup plus one failed call
    EXPECT_TRUE(owners.expired());
    expectEmpty(cache.stats());
    EXPECT_EQ(factory.physical->bytes, 0u);
    // A fresh healthy context may serve the next request after known
    // completion.
    EXPECT_FALSE(execute(cache, b, options, 10).error);
    EXPECT_EQ(factory.created, 2u);
  }
}
} // namespace
