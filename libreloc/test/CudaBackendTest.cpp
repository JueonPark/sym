// Focused completion tests on real CUDA streams (local GPU qualification).
#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaBackend.h"
#include "BlockingGate.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <future>

namespace {

void CUDART_CB waitAtGate(void *data) {
  static_cast<BlockingGate *>(data)->arriveAndWait();
}

void checkQuiesce(bool failCleanup) {
  reloc::CudaBackend backend(2);
  ASSERT_FALSE(backend.failed()) << backend.error();
  auto *src = static_cast<int *>(backend.allocStaging(2 * sizeof(int)));
  auto *dst = static_cast<int *>(backend.allocDevice(2 * sizeof(int)));
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  src[0] = 17;
  src[1] = 29;
  auto first = static_cast<cudaStream_t>(backend.stream(0));
  auto last = static_cast<cudaStream_t>(backend.stream(1));
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  int callerDevice =
      devices > 1 ? (backend.device() + 1) % devices : backend.device();

  BlockingGate copy;
  ASSERT_EQ(cudaLaunchHostFunc(last, waitAtGate, &copy), cudaSuccess);
  backend.copyAsync(0, dst, src, sizeof(int), reloc::CopyDir::HostToDevice);
  backend.copyAsync(1, dst + 1, src + 1, sizeof(int),
                    reloc::CopyDir::HostToDevice);
  EXPECT_TRUE(copy.waitForArrivals(1));

  // A harmless invalid argument injects a sticky operational error without
  // poisoning the device or preventing already-submitted copies from finishing.
  EXPECT_EQ(cudaSetDevice(-1), cudaErrorInvalidDevice);
  EXPECT_FALSE(backend.recordLaunchStatus("injected operation"));
  const auto originalError = backend.error();
  EXPECT_FALSE(originalError.empty());
  EXPECT_EQ(backend.recordEvent(1), 0u);

  // Synchronizing a capturing stream fails. This injects a cleanup failure on
  // the first stream; quiesce must still wait for the pending second stream.
  if (failCleanup) {
    EXPECT_EQ(cudaStreamBeginCapture(first, cudaStreamCaptureModeRelaxed),
              cudaSuccess);
  }
  std::promise<void> entered;
  auto drained = std::async(std::launch::async, [&] {
    EXPECT_EQ(cudaSetDevice(callerDevice), cudaSuccess);
    entered.set_value();
    auto result = backend.quiesce();
    int restored = -1;
    EXPECT_EQ(cudaGetDevice(&restored), cudaSuccess);
    EXPECT_EQ(restored, callerDevice);
    return result;
  });
  entered.get_future().wait();
  EXPECT_EQ(drained.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  copy.release();
  EXPECT_EQ(drained.get(), failCleanup ? reloc::QueueCompletion::Unknown
                                       : reloc::QueueCompletion::Complete);
  EXPECT_EQ(backend.error(), originalError);
  if (failCleanup) {
    cudaGraph_t graph = nullptr;
    EXPECT_EQ(cudaStreamEndCapture(first, &graph),
              cudaErrorStreamCaptureInvalidated);
    if (graph) {
      EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
    (void)cudaGetLastError();
  }
  // Explicit waits also make test teardown safe if quiesce regresses.
  EXPECT_EQ(cudaStreamSynchronize(first), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(last), cudaSuccess);
  int actual[2] = {};
  EXPECT_EQ(cudaMemcpy(actual, dst, sizeof(actual), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(actual[0], src[0]);
  EXPECT_EQ(actual[1], src[1]);
  EXPECT_EQ(backend.quiesce(), reloc::QueueCompletion::Complete);
  EXPECT_EQ(backend.error(), originalError);
  backend.freeDevice(dst);
  backend.freeStaging(src);
}

TEST(CudaBackend, QuiesceDrainsEveryStreamDespiteStickyError) {
  checkQuiesce(false);
}

TEST(CudaBackend, FailedCleanupPreservesErrorAndStillAttemptsLaterStreams) {
  checkQuiesce(true);
}

} // namespace

#endif // RELOC_ENABLE_CUDA
