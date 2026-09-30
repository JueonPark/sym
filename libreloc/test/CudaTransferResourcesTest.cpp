// Reused native contexts with real DMA and a fresh producer stream per call.
#ifdef RELOC_ENABLE_CUDA
#include "TransferTestSupport.h"
#include "reloc/CudaBackend.h"
#include "reloc/Execute.h"
#include "reloc/TransferResources.h"
#include "gtest/gtest.h"
#include <cstring>
#include <cuda_runtime.h>

namespace {
using namespace transfer_test;

struct CudaBuffers : Buffers {
  using Buffers::Buffers;
  void *device = nullptr;
  void *upload = nullptr;
  ~CudaBuffers() {
    if (device)
      cudaFree(device);
    if (upload)
      cudaFreeHost(upload);
  }
};
struct ProducerStreams {
  cudaStream_t streams[2]{};
  ~ProducerStreams() {
    for (auto stream : streams)
      if (stream)
        cudaStreamDestroy(stream);
  }
};

void CUDART_CB waitForProducer(void *data) {
  static_cast<BlockingGate *>(data)->arriveAndWait();
}

void checkReuse(bool cached) {
  ProducerStreams producers;
  for (auto &stream : producers.streams)
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              cudaSuccess);
  auto b = layout({513, 517}, {1, 513}, {517, 1});
  for (auto direction :
       {TransferDirection::HostToDevice, TransferDirection::DeviceToHost}) {
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    TransferResourceCache cache;
    std::unique_ptr<TransferContext> context;
    CudaBackend *observed = nullptr;
    const void *privateStreams[2]{};
    if (!cached) {
      auto backend = std::make_unique<CudaBackend>(2);
      ASSERT_FALSE(backend->failed()) << backend->error();
      observed = backend.get();
      privateStreams[0] = backend->stream(0);
      privateStreams[1] = backend->stream(1);
      context = std::make_unique<TransferContext>(std::move(backend));
    }
    for (unsigned round = 0; round < 3; ++round) {
      auto bytes = std::make_shared<CudaBuffers>(b, round);
      ASSERT_EQ(cudaMalloc(&bytes->device, b.totalBytes), cudaSuccess);
      ASSERT_EQ(cudaHostAlloc(&bytes->upload, bytes->src.size(),
                              cudaHostAllocDefault),
                cudaSuccess);
      std::memcpy(bytes->upload, bytes->src.data(), bytes->src.size());
      std::vector<uint8_t> expected(b.totalBytes);
      executeH2D(b, bytes->src.data(), expected.data());
      const bool h2d = direction == TransferDirection::HostToDevice;
      auto req = request(b, h2d ? bytes->src.data() : bytes->device,
                         h2d ? bytes->device : bytes->dst.data(), direction);
      auto &deviceView = h2d ? req.destination : req.source;
      deviceView.kind = MemoryKind::Cuda;
      deviceView.device = device;
      auto stream = producers.streams[round % 2];
      BlockingGate gate;
      ASSERT_EQ(cudaLaunchHostFunc(stream, waitForProducer, &gate),
                cudaSuccess);
      EXPECT_TRUE(gate.waitForArrivals(1));
      if (h2d) {
        EXPECT_EQ(cudaMemsetAsync(bytes->device, 0xCC, b.totalBytes, stream),
                  cudaSuccess);
      } else {
        EXPECT_EQ(cudaMemcpyAsync(bytes->device, bytes->upload,
                                  bytes->src.size(), cudaMemcpyHostToDevice,
                                  stream),
                  cudaSuccess);
      }
      TransferOptions options;
      options.nBuffers = 2;
      options.chunkSizeOverride = 128 * 1024;
      options.gatherThreads = 3;
      options.hasCallerStream = true;
      options.callerStream = stream;
      auto run = std::async(std::launch::async, [&] {
        if (!cached)
          return executeTransfer(req, *context, options, bytes);
        CachedTransferOptions cachedOptions;
        cachedOptions.transfer = options;
        cachedOptions.backend = {MemoryKind::Cuda, device, 2};
        return executeTransferCached(req, cache, cachedOptions, bytes);
      });
      EXPECT_EQ(run.wait_for(std::chrono::milliseconds(50)),
                std::future_status::timeout);
      gate.release();
      auto result = run.get();
      ASSERT_FALSE(result.error) << (result.error ? result.error->message : "");
      EXPECT_EQ(result.completion, TransferCompletion::Complete);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      if (h2d) {
        ASSERT_EQ(cudaMemcpy(bytes->dst.data(), bytes->device, b.totalBytes,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
      }
      EXPECT_EQ(bytes->dst, expected);
      EXPECT_EQ(bytes.use_count(), 1);
      if (cached) {
        const auto stats = cache.stats();
        EXPECT_EQ(stats.misses, 1u);
        EXPECT_EQ(stats.hits, round);
        EXPECT_EQ(stats.stagingAllocations, h2d ? 2u : 1u);
        EXPECT_EQ(stats.streamCreations, 2u);
        EXPECT_EQ(stats.workerCreations, 2u);
        EXPECT_EQ(stats.outstandingEvents, 0u);
      } else {
        EXPECT_EQ(context->stats().stagingPoolCreations, 1u);
        EXPECT_EQ(context->stats().workerPoolCreations, 1u);
        EXPECT_EQ(observed->stream(0), privateStreams[0]);
        EXPECT_EQ(observed->stream(1), privateStreams[1]);
      }
    }
    if (cached) {
      EXPECT_FALSE(cache.close());
      EXPECT_EQ(cache.stats().contexts, 0u);
    } else {
      EXPECT_EQ(context->close(), TransferCompletion::Complete);
    }
  }
}
TEST(CudaTransferResources, ReusePreservesDataAndFreshProducerOrdering) {
  checkReuse(false);
}
TEST(CudaTransferResources, CacheReusesAllocationsAndOrdersEveryProducer) {
  checkReuse(true);
}
} // namespace
#endif
