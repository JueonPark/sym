//===- TransferTest.cpp - validated forward transfer requests (R2) --------===//
//
// Capacity/span validation must fail before any backend copy; the forward
// host path (both directions through HostBackend) must be byte-identical to
// executeH2D; requests are single-use and backend failures propagate.
//
//===----------------------------------------------------------------------===//

#include "reloc/Transfer.h"

#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/HostBackend.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::BoundPlan;
using reloc::BufferView;
using reloc::MemoryKind;
using reloc::PadRegion;
using reloc::TransferDirection;
using reloc::TransferError;
using reloc::TransferOptions;
using reloc::TransferRequest;

std::vector<uint8_t> iotaBytes(int64_t elements, uint32_t elementSize) {
  std::vector<uint8_t> buf(static_cast<size_t>(elements) * elementSize);
  for (int64_t e = 0; e < elements; ++e)
    for (uint32_t b = 0; b < elementSize; ++b)
      buf[e * elementSize + b] =
          static_cast<uint8_t>((e * 131 + b * 17) & 0xff);
  return buf;
}

BoundPlan makeBound(std::vector<int64_t> extents,
                    std::vector<int64_t> srcStrides,
                    std::vector<int64_t> dstStrides, uint32_t elementSize,
                    std::vector<PadRegion> pads = {}) {
  BoundPlan b;
  b.extents = std::move(extents);
  b.srcStrides = std::move(srcStrides);
  b.dstStrides = std::move(dstStrides);
  b.elementSize = elementSize;
  b.padRegions = std::move(pads);
  int64_t total = elementSize;
  std::vector<int64_t> padded = b.extents;
  for (const PadRegion &p : b.padRegions)
    padded[p.axis] += p.lo + p.hi;
  for (int64_t e : padded)
    total *= e;
  b.totalBytes = total;
  b.L = (!b.extents.empty() && b.srcStrides.back() == 1 &&
         b.dstStrides.back() == 1)
            ? b.extents.back()
            : 1;
  return b;
}

BufferView hostView(const void *base, size_t capacity, size_t offset,
                    std::vector<int64_t> extents, std::vector<int64_t> strides,
                    uint32_t elementSize) {
  BufferView v;
  v.base = reinterpret_cast<uintptr_t>(base);
  v.capacityBytes = capacity;
  v.offsetBytes = offset;
  v.extents = std::move(extents);
  v.strides = std::move(strides);
  v.elementSize = elementSize;
  v.kind = MemoryKind::Host;
  v.device = -1;
  return v;
}

std::string codeOf(const std::variant<TransferRequest, TransferError> &r) {
  if (const auto *e = std::get_if<TransferError>(&r))
    return e->code;
  return "ok";
}

// Transpose [4,6] -> [6,4] f32: src strides (1,4), dst strides (4,1) in
// coalesced dst order (extents {6,4}).
BoundPlan transposePlan() { return makeBound({6, 4}, {1, 4}, {4, 1}, 4); }

// Counting backend: copies must not happen when validation rejects.
class CountingHostBackend : public reloc::HostBackend {
public:
  explicit CountingHostBackend(int queues) : HostBackend(queues) {}
  void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                 reloc::CopyDir dir) override {
    ++copies;
    HostBackend::copyAsync(queue, dst, src, bytes, dir);
  }
  std::atomic<int> copies{0};
};

TEST(Transfer, SourceOrDestinationOneByteShortFailsBeforeAnyCopy) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes));
  CountingHostBackend backend(1);

  BufferView goodSrc = hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4);
  BufferView goodDst = hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4);
  ASSERT_EQ(codeOf(reloc::validateTransfer(b, goodSrc, goodDst,
                                           TransferDirection::HostToDevice)),
            "ok");

  BufferView shortSrc = goodSrc;
  shortSrc.capacityBytes = src.size() - 1;
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, shortSrc, goodDst,
                                           TransferDirection::HostToDevice)),
            "insufficient_capacity");
  BufferView shortDst = goodDst;
  shortDst.capacityBytes = dst.size() - 1;
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, goodSrc, shortDst,
                                           TransferDirection::HostToDevice)),
            "insufficient_capacity");
  // A nonzero offset eats into the declared capacity the same way.
  BufferView offsetSrc = goodSrc;
  offsetSrc.offsetBytes = 4;
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, offsetSrc, goodDst,
                                           TransferDirection::HostToDevice)),
            "insufficient_capacity");
  EXPECT_EQ(backend.copies.load(), 0);
}

TEST(Transfer, LargeCheckedProductsReportOverflowNotWraparound) {
  BoundPlan b = transposePlan();
  const int64_t huge = std::numeric_limits<int64_t>::max() / 2;
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes));
  BufferView dstView = hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4);
  uint8_t byte = 0;
  BufferView src = hostView(&byte, std::numeric_limits<size_t>::max(), 0,
                            {huge, 4}, {4, 1}, 4);
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, src, dstView,
                                           TransferDirection::HostToDevice)),
            "integer_overflow");
  BufferView wideStride = hostView(&byte, std::numeric_limits<size_t>::max(), 0,
                                   {3, 3}, {huge, 1}, 4);
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, wideStride, dstView,
                                           TransferDirection::HostToDevice)),
            "integer_overflow");
  BufferView equalStrides = hostView(&byte, std::numeric_limits<size_t>::max(),
                                     0, {2, 3}, {huge, huge}, 4);
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, equalStrides, dstView,
                                           TransferDirection::HostToDevice)),
            "unsupported_layout");
  BoundPlan overflowing = transposePlan();
  overflowing.extents = {huge, 4};
  overflowing.totalBytes = 96;
  std::vector<uint8_t> small = iotaBytes(24, 4);
  BufferView okSrc = hostView(small.data(), small.size(), 0, {4, 6}, {6, 1}, 4);
  EXPECT_EQ(codeOf(reloc::validateTransfer(overflowing, okSrc, dstView,
                                           TransferDirection::HostToDevice)),
            "integer_overflow");
}

TEST(Transfer, UnsupportedViewsAreRejectedWithStableCodes) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes));
  BufferView dstView = hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4);
  auto run = [&](BufferView v) {
    return codeOf(reloc::validateTransfer(b, v, dstView,
                                          TransferDirection::HostToDevice));
  };
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {4, 6}, {-6, 1}, 4)),
            "unsupported_layout");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {4, 6}, {0, 1}, 4)),
            "unsupported_layout");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {4, 6}, {1, 1}, 4)),
            "unsupported_layout");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {}, {}, 4)),
            "invalid_view");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {0, 6}, {6, 1}, 4)),
            "invalid_view");
  EXPECT_EQ(run(hostView(nullptr, src.size(), 0, {4, 6}, {6, 1}, 4)),
            "invalid_view");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 2)),
            "plan_mismatch");
  EXPECT_EQ(run(hostView(src.data(), src.size(), 0, {2, 6}, {6, 1}, 4)),
            "plan_mismatch");
  BufferView cudaSource =
      hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4);
  cudaSource.kind = MemoryKind::Cuda;
  cudaSource.device = 0;
  EXPECT_EQ(run(cudaSource), "direction_mismatch");
  BufferView cudaDestination = dstView;
  cudaDestination.kind = MemoryKind::Cuda;
  cudaDestination.device = 0;
  EXPECT_EQ(codeOf(reloc::validateTransfer(
                b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
                cudaDestination, TransferDirection::DeviceToHost)),
            "direction_mismatch");
  BufferView stridedDst = dstView;
  stridedDst.extents = {4, 6};
  stridedDst.strides = {1, 4};
  EXPECT_EQ(codeOf(reloc::validateTransfer(
                b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
                stridedDst, TransferDirection::HostToDevice)),
            "unsupported_layout");
  BufferView wrongDst = dstView;
  wrongDst.extents = {3, 4};
  wrongDst.strides = {4, 1};
  EXPECT_EQ(codeOf(reloc::validateTransfer(
                b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
                wrongDst, TransferDirection::HostToDevice)),
            "plan_mismatch");
}

TEST(Transfer, PadOnlyRegionsNeverReadSourceBytes) {
  // dst [2+1+1, 3] with axis-0 pads: the source must be exactly 6 elements.
  PadRegion pad{0, 1, 1};
  pad.fillBits = 0x7F800001;
  BoundPlan b = makeBound({2, 3}, {3, 1}, {3, 1}, 4, {pad});
  std::vector<uint8_t> src = iotaBytes(6, 4);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  BufferView dstView = hostView(dst.data(), dst.size(), 0, {4, 3}, {3, 1}, 4);
  BufferView exact = hostView(src.data(), src.size(), 0, {2, 3}, {3, 1}, 4);
  BufferView tooShort = exact;
  tooShort.capacityBytes = src.size() - 1;
  EXPECT_EQ(codeOf(reloc::validateTransfer(b, tooShort, dstView,
                                           TransferDirection::HostToDevice)),
            "insufficient_capacity");
  auto validated = reloc::validateTransfer(b, exact, dstView,
                                           TransferDirection::HostToDevice);
  ASSERT_EQ(codeOf(validated), "ok");
  auto request = std::get<TransferRequest>(validated);
  EXPECT_EQ(request.sourceSpanBytes, src.size());
  EXPECT_EQ(request.destinationBytes, dst.size());
  CountingHostBackend backend(2);
  TransferOptions options;
  options.nBuffers = 2;
  auto error = reloc::executeTransfer(request, backend, options);
  ASSERT_FALSE(error.has_value()) << error->code << ": " << error->message;
  EXPECT_EQ(dst, reference);
  EXPECT_TRUE(request.consumed);
}

TEST(Transfer, HostToHostForwardMatchesReferenceAndIsSingleUse) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
      hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4),
      TransferDirection::HostToDevice);
  ASSERT_EQ(codeOf(validated), "ok");
  auto request = std::get<TransferRequest>(validated);
  CountingHostBackend backend(2);
  TransferOptions options;
  ASSERT_FALSE(reloc::executeTransfer(request, backend, options).has_value());
  EXPECT_EQ(dst, reference);
  EXPECT_GE(backend.copies.load(), 1);
  const int copies = backend.copies.load();
  auto again = reloc::executeTransfer(request, backend, options);
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(again->code, "already_executed");
  EXPECT_EQ(backend.copies.load(), copies);
}

TEST(Transfer, DeviceToHostForwardStagesThenGathers) {
  // Same plan, opposite direction: the "device" source is host memory here,
  // so the staging copy + forward gather path runs entirely under HostBackend.
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());
  for (unsigned threads : {1u, 3u}) {
    std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
    auto validated = reloc::validateTransfer(
        b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
        hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4),
        TransferDirection::DeviceToHost);
    ASSERT_EQ(codeOf(validated), "ok");
    auto request = std::get<TransferRequest>(validated);
    CountingHostBackend backend(1);
    TransferOptions options;
    options.gatherThreads = threads;
    ASSERT_FALSE(reloc::executeTransfer(request, backend, options).has_value());
    EXPECT_EQ(dst, reference) << "threads=" << threads;
    EXPECT_EQ(backend.copies.load(), 1); // exactly one staging copy
  }
  // Caller-owned gather pool path.
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
      hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4),
      TransferDirection::DeviceToHost);
  auto request = std::get<TransferRequest>(validated);
  reloc::GatherPool pool(2);
  CountingHostBackend backend(1);
  TransferOptions options;
  options.gather = &pool;
  ASSERT_FALSE(reloc::executeTransfer(request, backend, options).has_value());
  EXPECT_EQ(dst, reference);
}

TEST(Transfer, IdentityPlanStillMovesBytes) {
  BoundPlan b = makeBound({24}, {1}, {1}, 4);
  b.noCopy = true; // layout says no copy; a cross-buffer transfer still copies
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(src.size(), 0xCD);
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {24}, {1}, 4),
      hostView(dst.data(), dst.size(), 0, {24}, {1}, 4),
      TransferDirection::HostToDevice);
  ASSERT_EQ(codeOf(validated), "ok");
  auto request = std::get<TransferRequest>(validated);
  CountingHostBackend backend(1);
  ASSERT_FALSE(
      reloc::executeTransfer(request, backend, TransferOptions{}).has_value());
  EXPECT_EQ(dst, src);
  EXPECT_GE(backend.copies.load(), 1);
}

class FailingStagingBackend : public reloc::HostBackend {
public:
  FailingStagingBackend() : HostBackend(1) {}
  void *allocStaging(size_t) override { return nullptr; }
};

TEST(Transfer, BackendFailuresPropagateWithoutRetry) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
      hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4),
      TransferDirection::DeviceToHost);
  auto request = std::get<TransferRequest>(validated);
  FailingStagingBackend backend;
  auto error = reloc::executeTransfer(request, backend, TransferOptions{});
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "backend_failure");
  EXPECT_TRUE(request.consumed);
  EXPECT_EQ(std::count(dst.begin(), dst.end(), 0xCD),
            static_cast<long>(dst.size()));
}

TEST(Transfer, HostToDeviceStagingFailureIsReportedNotDereferenced) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
      hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4),
      TransferDirection::HostToDevice);
  auto request = std::get<TransferRequest>(validated);
  FailingStagingBackend backend;
  TransferOptions options;
  options.nBuffers = 2;
  auto error = reloc::executeTransfer(request, backend, options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "backend_failure");
  EXPECT_NE(error->message.find("pinned staging allocation failed"),
            std::string::npos);
  EXPECT_TRUE(request.consumed);
  EXPECT_EQ(std::count(dst.begin(), dst.end(), 0xCD),
            static_cast<long>(dst.size()));
}

class DeviceOneBackend : public reloc::HostBackend {
public:
  DeviceOneBackend() : HostBackend(1) {}
  int device() const override { return 1; }
};

TEST(Transfer, CudaViewOnAnotherDeviceFailsBeforeAnyWork) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0xCD);
  BufferView destination =
      hostView(dst.data(), dst.size(), 0, {6, 4}, {4, 1}, 4);
  destination.kind = MemoryKind::Cuda;
  destination.device = 0;
  auto validated = reloc::validateTransfer(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4), destination,
      TransferDirection::HostToDevice);
  ASSERT_EQ(codeOf(validated), "ok");
  auto request = std::get<TransferRequest>(validated);
  DeviceOneBackend backend;
  auto error = reloc::executeTransfer(request, backend, TransferOptions{});
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->code, "device_mismatch");
  EXPECT_FALSE(request.consumed);
  EXPECT_EQ(std::count(dst.begin(), dst.end(), 0xCD),
            static_cast<long>(dst.size()));
  // A host backend (device() < 0) accepts any ordinal.
  reloc::HostBackend host(1);
  ASSERT_FALSE(
      reloc::executeTransfer(request, host, TransferOptions{}).has_value());
}

TEST(Transfer, SourceOnlyValidationMatchesFullValidation) {
  BoundPlan b = transposePlan();
  std::vector<uint8_t> src = iotaBytes(24, 4);
  auto span = reloc::validateTransferSource(
      b, hostView(src.data(), src.size(), 0, {4, 6}, {6, 1}, 4),
      TransferDirection::HostToDevice);
  ASSERT_TRUE(std::holds_alternative<size_t>(span));
  EXPECT_EQ(std::get<size_t>(span), src.size());
  auto rejected = reloc::validateTransferSource(
      b, hostView(src.data(), src.size() - 1, 0, {4, 6}, {6, 1}, 4),
      TransferDirection::HostToDevice);
  ASSERT_TRUE(std::holds_alternative<TransferError>(rejected));
  EXPECT_EQ(std::get<TransferError>(rejected).code, "insufficient_capacity");
}

TEST(Transfer, HostBackendDefaultsSatisfyTheExtendedContract) {
  reloc::HostBackend backend(1);
  EXPECT_TRUE(backend.waitStream(nullptr));
  EXPECT_FALSE(backend.failed());
  EXPECT_TRUE(backend.error().empty());
  EXPECT_EQ(backend.device(), -1);
}

} // namespace
