//===- StackedTransferTest.cpp - stacked forward transfer requests --------===//
//
// torch.stack support: a stacked request validates every input
// before any copy, then runs through the unchanged pipeline (chunks, staging
// slots, gather workers, retained contexts) and lands exactly the bytes of
// the single-source transfer over the inputs' concatenation.
//
//===----------------------------------------------------------------------===//

#include "reloc/Transfer.h"

#include "TransferTestSupport.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "reloc/TransferResources.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#ifdef RELOC_ENABLE_CUDA
#include "reloc/CudaBackend.h"
#include <cuda_runtime.h>
#endif

namespace {

using reloc::BoundPlan;
using reloc::BufferView;
using reloc::MemoryKind;
using reloc::TransferDirection;
using reloc::TransferError;
using reloc::TransferOptions;
using reloc::TransferRequest;

BufferView hostView(const void *base, size_t capacity,
                    std::vector<int64_t> extents, uint32_t elementSize) {
  BufferView v;
  v.base = reinterpret_cast<uintptr_t>(base);
  v.capacityBytes = capacity;
  v.extents = std::move(extents);
  v.strides.assign(v.extents.size(), 1);
  for (size_t k = v.extents.size(); k-- > 1;)
    v.strides[k - 1] = v.strides[k] * v.extents[k];
  v.elementSize = elementSize;
  return v;
}

// stack(count x [rows, columns], dim) of 4-byte elements as the compiler
// folds it from the logical [count, rows, columns].
struct Stack {
  int64_t count, rows, columns;
  int dim;
  BoundPlan plan;
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<uint8_t> concatenated;

  Stack(int64_t count, int64_t rows, int64_t columns, int dim = 1)
      : count(count), rows(rows), columns(columns), dim(dim) {
    const int64_t z = rows * columns;
    if (dim == 0)
      plan = transfer_test::layout({count * z}, {1}, {1}, 4);
    else if (dim == 1)
      plan = transfer_test::layout({rows, count, columns}, {columns, z, 1},
                                   {count * columns, columns, 1}, 4);
    else
      plan = transfer_test::layout({z, count}, {1, z}, {count, 1}, 4);
    for (int64_t i = 0; i < count; ++i) {
      std::vector<uint8_t> input(static_cast<size_t>(z) * 4);
      for (size_t k = 0; k < input.size(); ++k)
        input[k] = static_cast<uint8_t>((i * 211 + k * 17 + 3) & 0xff);
      concatenated.insert(concatenated.end(), input.begin(), input.end());
      inputs.push_back(std::move(input));
    }
  }
  std::vector<BufferView> views() const {
    std::vector<BufferView> out;
    for (const auto &input : inputs)
      out.push_back(hostView(input.data(), input.size(), {rows, columns}, 4));
    return out;
  }
  std::vector<int64_t> destinationExtents() const {
    if (dim == 0)
      return {count, rows, columns};
    if (dim == 1)
      return {rows, count, columns};
    return {rows, columns, count};
  }
  std::vector<uint8_t> reference() const {
    std::vector<uint8_t> dst(static_cast<size_t>(plan.totalBytes), 0xAB);
    reloc::executeH2D(plan, concatenated.data(), dst.data());
    return dst;
  }
};

std::string codeOf(const std::variant<TransferRequest, TransferError> &r) {
  if (const auto *e = std::get_if<TransferError>(&r))
    return e->code;
  return "ok";
}

std::string codeOf(const std::variant<size_t, TransferError> &r) {
  if (const auto *e = std::get_if<TransferError>(&r))
    return e->code;
  return "ok";
}

TEST(StackedTransfer, HostForwardMatchesTheConcatenatedTransfer) {
  for (int dim : {0, 1, 2}) {
    Stack s(3, 5, 7, dim);
    std::vector<uint8_t> dst(static_cast<size_t>(s.plan.totalBytes), 0xCD);
    auto validated = reloc::validateStackedTransfer(
        s.plan, s.views(),
        hostView(dst.data(), dst.size(), s.destinationExtents(), 4),
        TransferDirection::HostToDevice);
    ASSERT_EQ(codeOf(validated), "ok") << "dim " << dim;
    auto request = std::get<TransferRequest>(validated);
    EXPECT_EQ(request.stackSources.size(), 3u);
    EXPECT_EQ(request.stackSegmentElements, 35);
    EXPECT_EQ(request.sourceSpanBytes, 3u * 35 * 4);
    reloc::HostBackend backend(2);
    ASSERT_FALSE(reloc::executeTransfer(request, backend, TransferOptions{})
                     .has_value());
    EXPECT_EQ(dst, s.reference()) << "dim " << dim;
    auto again = reloc::executeTransfer(request, backend, TransferOptions{});
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->code, "already_executed");
  }
}

TEST(StackedTransfer, ChunksAndWorkersCrossInputBoundaries) {
  // Forced small chunks cut inside inputs (dim 0) and across them (dim 1);
  // workers split each chunk's outer rows.
  for (int dim : {0, 1, 2}) {
    Stack s(4, 64, 256, dim);
    const auto expected = s.reference();
    for (unsigned threads : {1u, 4u})
      for (size_t chunk : {size_t(4096), size_t(48 * 1024)}) {
        std::vector<uint8_t> dst(static_cast<size_t>(s.plan.totalBytes), 0xCD);
        auto validated = reloc::validateStackedTransfer(
            s.plan, s.views(),
            hostView(dst.data(), dst.size(), s.destinationExtents(), 4),
            TransferDirection::HostToDevice);
        ASSERT_EQ(codeOf(validated), "ok");
        auto request = std::get<TransferRequest>(validated);
        reloc::HostBackend backend(2);
        reloc::GatherPool pool(threads);
        TransferOptions options;
        options.nBuffers = 2;
        options.chunkSizeOverride = chunk;
        options.gather = &pool;
        ASSERT_FALSE(
            reloc::executeTransfer(request, backend, options).has_value());
        EXPECT_EQ(dst, expected) << "dim " << dim << ", " << threads
                                 << " threads, " << chunk << "-byte chunks";
      }
  }
}

TEST(StackedTransfer, RetainedContextServesRepeatedStackedRequests) {
  Stack s(3, 32, 64, 1);
  const auto expected = s.reference();
  reloc::TransferContext context(std::make_unique<reloc::HostBackend>(2));
  auto owners = std::make_shared<int>(0);
  for (int call = 0; call < 3; ++call) {
    std::vector<uint8_t> dst(static_cast<size_t>(s.plan.totalBytes), 0xCD);
    auto validated = reloc::validateStackedTransfer(
        s.plan, s.views(),
        hostView(dst.data(), dst.size(), s.destinationExtents(), 4),
        TransferDirection::HostToDevice);
    ASSERT_EQ(codeOf(validated), "ok");
    auto request = std::get<TransferRequest>(validated);
    auto outcome =
        reloc::executeTransfer(request, context, TransferOptions{}, owners);
    ASSERT_FALSE(outcome.error.has_value()) << outcome.error->message;
    EXPECT_EQ(dst, expected) << "call " << call;
  }
  EXPECT_EQ(context.close(), reloc::TransferCompletion::Complete);
}

TEST(StackedTransfer, InvalidSourcesFailBeforeAnyCopy) {
  Stack s(3, 5, 7, 1);
  std::vector<uint8_t> dst(static_cast<size_t>(s.plan.totalBytes), 0xCD);
  const BufferView destination =
      hostView(dst.data(), dst.size(), s.destinationExtents(), 4);
  auto expect = [&](const std::vector<BufferView> &views,
                    TransferDirection direction, const char *code) {
    EXPECT_EQ(codeOf(reloc::validateStackedTransfer(s.plan, views, destination,
                                                    direction)),
              code);
    EXPECT_EQ(codeOf(reloc::validateStackedSources(s.plan, views, direction)),
              code);
  };
  const auto views = s.views();
  expect({}, TransferDirection::HostToDevice, "plan_mismatch");
  expect({views[0], views[1]}, TransferDirection::HostToDevice,
         "plan_mismatch");
  expect(views, TransferDirection::DeviceToHost, "direction_mismatch");
  auto unequal = views;
  unequal[2].extents = {5, 6};
  unequal[2].strides = {6, 1};
  expect(unequal, TransferDirection::HostToDevice, "plan_mismatch");
  auto strided = views;
  strided[1].extents = {7, 5};
  strided[1].strides = {1, 7};
  expect(strided, TransferDirection::HostToDevice, "unsupported_layout");
  auto narrow = views;
  narrow[0].elementSize = 2;
  narrow[0].extents = {5, 14};
  narrow[0].strides = {14, 1};
  expect(narrow, TransferDirection::HostToDevice, "plan_mismatch");
  auto truncated = views;
  truncated[1].capacityBytes -= 1;
  expect(truncated, TransferDirection::HostToDevice, "insufficient_capacity");
  auto device = views;
  device[0].kind = MemoryKind::Cuda;
  device[0].device = 0;
  expect(device, TransferDirection::HostToDevice, "direction_mismatch");
  BoundPlan typed = s.plan;
  typed.typed = true;
  EXPECT_EQ(codeOf(reloc::validateStackedSources(
                typed, views, TransferDirection::HostToDevice)),
            "typed_unsupported");
}

TEST(StackedTransfer, LogicalSizeOverflowIsReported) {
  // Four views of 2^61 four-byte elements: N * Z * 4 overflows.
  static const uint8_t byte = 0;
  std::vector<BufferView> views(
      4, hostView(&byte, SIZE_MAX, {int64_t(1) << 61}, 4));
  BoundPlan plan = transfer_test::layout({1}, {1}, {1}, 4);
  EXPECT_EQ(codeOf(reloc::validateStackedSources(
                plan, views, TransferDirection::HostToDevice)),
            "integer_overflow");
}

#ifdef RELOC_ENABLE_CUDA
TEST(StackedTransferCuda, DeviceDestinationMatchesTheConcatenatedTransfer) {
  int devices = 0, current = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "no CUDA device";
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  for (int dim : {0, 1, 2}) {
    Stack s(4, 256, 1024, dim); // 4 MiB of inputs
    const auto expected = s.reference();
    void *device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, static_cast<size_t>(s.plan.totalBytes)),
              cudaSuccess);
    BufferView destination =
        hostView(device, static_cast<size_t>(s.plan.totalBytes),
                 s.destinationExtents(), 4);
    destination.kind = MemoryKind::Cuda;
    destination.device = current;
    auto validated = reloc::validateStackedTransfer(
        s.plan, s.views(), destination, TransferDirection::HostToDevice);
    ASSERT_EQ(codeOf(validated), "ok");
    auto request = std::get<TransferRequest>(validated);
    reloc::CudaBackend backend(2);
    TransferOptions options;
    options.chunkSizeOverride = 256 * 1024; // several chunks per request
    ASSERT_FALSE(reloc::executeTransfer(request, backend, options).has_value());
    std::vector<uint8_t> back(static_cast<size_t>(s.plan.totalBytes));
    ASSERT_EQ(
        cudaMemcpy(back.data(), device, back.size(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(back, expected) << "dim " << dim;
    cudaFree(device);
  }
}
#endif

} // namespace
