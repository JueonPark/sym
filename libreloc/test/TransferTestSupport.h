// Shared native transfer fixtures; failure gates do not depend on CUDA.
#ifndef RELOC_TEST_TRANSFERTESTSUPPORT_H
#define RELOC_TEST_TRANSFERTESTSUPPORT_H

#include "BlockingGate.h"
#include "reloc/HostBackend.h"
#include "reloc/Transfer.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>

namespace transfer_test {
using namespace reloc;

inline BoundPlan layout(std::vector<int64_t> extents, std::vector<int64_t> src,
                        std::vector<int64_t> dst, uint32_t width = 4,
                        std::vector<PadRegion> pads = {}) {
  BoundPlan b;
  b.extents = std::move(extents);
  b.srcStrides = std::move(src);
  b.dstStrides = std::move(dst);
  b.elementSize = width;
  b.padRegions = std::move(pads);
  auto padded = b.extents;
  for (const auto &p : b.padRegions)
    padded[p.axis] += p.lo + p.hi;
  b.totalBytes = width;
  for (auto e : padded)
    b.totalBytes *= e;
  b.L = 1;
  return b;
}

inline size_t sourceBytes(const BoundPlan &b) {
  size_t reach = 0;
  for (size_t i = 0; i < b.extents.size(); ++i)
    reach += (b.extents[i] - 1) * b.srcStrides[i];
  return (reach + 1) * b.elementSize;
}

inline TransferRequest
request(const BoundPlan &b, const void *src, void *dst,
        TransferDirection direction = TransferDirection::HostToDevice) {
  BufferView source, destination;
  source.base = reinterpret_cast<uintptr_t>(src);
  source.capacityBytes = sourceBytes(b);
  source.extents = b.extents;
  source.strides = b.srcStrides;
  source.elementSize = b.elementSize;
  destination.base = reinterpret_cast<uintptr_t>(dst);
  destination.capacityBytes = b.totalBytes;
  destination.extents = {b.totalBytes / b.elementSize};
  destination.strides = {1};
  destination.elementSize = b.elementSize;
  auto result = validateTransfer(b, source, destination, direction);
  if (auto *error = std::get_if<TransferError>(&result))
    throw std::runtime_error(error->code + ": " + error->message);
  return std::get<TransferRequest>(std::move(result));
}

struct Buffers {
  explicit Buffers(const BoundPlan &b, unsigned seed = 1)
      : src(sourceBytes(b)), dst(b.totalBytes, 0xCD) {
    for (size_t i = 0; i < src.size(); ++i)
      src[i] = uint8_t(i * 131 + seed * 17);
  }
  std::vector<uint8_t> src, dst;
};

struct Metrics {
  std::atomic<int> allocations{0}, frees{0}, copies{0}, waits{0}, events{0};
  std::atomic<int> quiesces{0}, destroyed{0};
  size_t liveBytes = 0, peakBytes = 0; // read after driver completion
  std::vector<size_t> copyBytes;
  std::vector<const void *> callerStreams;
  std::promise<void> secondCopy;
};

class Backend : public HostBackend {
public:
  explicit Backend(std::shared_ptr<Metrics> m, int queues = 2)
      : HostBackend(queues), m(std::move(m)) {}
  ~Backend() override { ++m->destroyed; }
  int failAllocation = 0;
  bool throwAllocation = false, failEvent = false, failCopy = false;
  bool throwCopy = false, failWait = false, unknown = false;
  std::shared_ptr<BlockingGate> gate;
  std::shared_ptr<Metrics> m;

  void *allocStaging(size_t bytes) override {
    if (++m->allocations == failAllocation) {
      if (throwAllocation)
        throw std::runtime_error("injected allocation failure");
      return nullptr;
    }
    auto *p = HostBackend::allocStaging(bytes);
    if (p) {
      sizes[p] = bytes;
      m->liveBytes += bytes;
      m->peakBytes = std::max(m->peakBytes, m->liveBytes);
    }
    return p;
  }
  void freeStaging(void *p) override {
    if (p) {
      ++m->frees;
      m->liveBytes -= sizes.at(p);
      sizes.erase(p);
    }
    HostBackend::freeStaging(p);
  }
  void copyAsync(int queue, void *dst, const void *src, size_t bytes,
                 CopyDir direction) override {
    const int number = ++m->copies;
    m->copyBytes.push_back(bytes);
    HostBackend::copyAsync(queue, dst, src, bytes, direction);
    if (gate && number == 1 && !gate->waitForArrivals(1))
      throw std::runtime_error("test copy did not reach its gate");
    if (number == 2)
      m->secondCopy.set_value();
    if (failCopy)
      error_ = "injected copy failure";
    if (throwCopy)
      throw std::runtime_error("injected exception after copy submission");
  }
  EventHandle recordEvent(int queue) override {
    ++m->events;
    if (failEvent) {
      error_ = "injected event failure";
      return 0;
    }
    return HostBackend::recordEvent(queue);
  }
  void waitEvent(EventHandle event) override {
    ++m->waits;
    if (failWait) {
      error_ = "injected wait failure";
      return;
    }
    HostBackend::waitEvent(event);
  }
  QueueCompletion quiesce() override {
    ++m->quiesces;
    return unknown ? QueueCompletion::Unknown : HostBackend::quiesce();
  }
  bool waitStream(const void *stream) override {
    m->callerStreams.push_back(stream);
    return true;
  }
  bool failed() const override { return !error_.empty(); }
  const std::string &error() const override { return error_; }
  void gateCopies() {
    gate = std::make_shared<BlockingGate>();
    setCopyHook([g = gate] { g->arriveAndWait(); });
  }

private:
  std::map<void *, size_t> sizes;
  std::string error_;
};
} // namespace transfer_test
#endif
