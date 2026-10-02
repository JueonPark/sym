#include "TransferResourcePolicy.h"
#include "reloc/GatherPool.h"

#include <atomic>
#include <cerrno>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <system_error>
#include <tuple>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace reloc::detail {
namespace {
std::atomic<uint64_t> processEpoch{0};
std::once_flag forkHook;

std::vector<unsigned long> affinity() {
  std::vector<unsigned long> mask;
#ifdef __linux__
  // Grow beyond CPU_SETSIZE when required by the kernel's affinity ABI.
  mask.resize(16);
  while (sched_getaffinity(0, mask.size() * sizeof(unsigned long),
                           reinterpret_cast<cpu_set_t *>(mask.data())) != 0) {
    if (errno != EINVAL || mask.size() > (1u << 20))
      throw std::system_error(errno, std::generic_category(),
                              "sched_getaffinity");
    mask.resize(mask.size() * 2);
  }
#endif
  return mask;
}
} // namespace

std::vector<unsigned long> currentCpuAffinity() { return affinity(); }

ProcessIdentity currentProcessIdentity() {
  return {int64_t(getpid()), processEpoch.load(std::memory_order_relaxed)};
}

ProcessIdentity captureProcessIdentity() {
  static_assert(std::atomic<uint64_t>::is_always_lock_free,
                "fork identity must be readable without inherited locks");
  std::call_once(forkHook, [] {
    int error = pthread_atfork(nullptr, nullptr, [] {
      processEpoch.fetch_add(1, std::memory_order_relaxed);
    });
    if (error)
      throw std::system_error(error, std::generic_category(), "pthread_atfork");
  });
  return currentProcessIdentity();
}

bool ResourceKey::sameDevice(const ResourceKey &other) const {
  return backend.kind == other.backend.kind &&
         backend.device == other.backend.device;
}

bool ResourceKey::operator==(const ResourceKey &other) const {
  return sameDevice(other) &&
         std::tie(backend.streams, direction, activeSlots, workers,
                  participants, placementTag, affinity) ==
             std::tie(other.backend.streams, other.direction, other.activeSlots,
                      other.workers, other.participants, other.placementTag,
                      other.affinity);
}

std::variant<size_t, TransferError> roundStagingCapacity(size_t bytes) {
  constexpr size_t quantum = 256u << 10;
  if (!bytes || bytes > std::numeric_limits<size_t>::max() - (quantum - 1))
    return TransferError{"integer_overflow",
                         "rounded staging capacity overflows"};
  return (bytes + quantum - 1) & ~(quantum - 1);
}

std::variant<CacheRequest, TransferError>
describeCachedTransfer(const TransferRequest &request,
                       const CachedTransferOptions &options,
                       const TransferResourceLimits &limits) {
  const auto &backend = options.backend;
  if (backend.streams < 1 ||
      (backend.kind != MemoryKind::Host && backend.kind != MemoryKind::Cuda) ||
      (backend.kind == MemoryKind::Host && backend.device != -1) ||
      (backend.kind == MemoryKind::Cuda && backend.device < 0))
    return TransferError{"invalid_options", "invalid backend configuration"};
  const auto &deviceView = request.direction == TransferDirection::HostToDevice
                               ? request.destination
                               : request.source;
  if (deviceView.kind != backend.kind)
    return TransferError{"device_mismatch",
                         "transfer storage and backend kinds differ"};
  for (const auto *view : {&request.source, &request.destination})
    if (view->kind == MemoryKind::Cuda &&
        (backend.kind != MemoryKind::Cuda || view->device != backend.device))
      return TransferError{"device_mismatch",
                           "view does not belong to cache device"};
  if ((options.transfer.gather &&
       options.transfer.gather != options.gather.get()) ||
      (options.gather && options.gather->closed()))
    return TransferError{"invalid_options",
                         "borrowed gather pool needs a live shared owner"};
  CacheRequest result;
  result.options = options.transfer;
  result.options.gather = options.gather.get();
  auto described = describeTransfer(request, result.options);
  if (auto *error = std::get_if<TransferError>(&described))
    return *error;
  result.execution = std::get<TransferRequirements>(std::move(described));
  auto rounded = roundStagingCapacity(result.execution.slotBytes);
  if (auto *error = std::get_if<TransferError>(&rounded))
    return *error;
  result.slotCapacity = std::get<size_t>(rounded);
  if (__builtin_mul_overflow(result.slotCapacity,
                             size_t(result.execution.activeSlots),
                             &result.bytes))
    return TransferError{"integer_overflow", "rounded staging ring overflows"};
  result.workers = result.execution.gatherThreads - 1;
  result.cacheable = result.bytes <= limits.maxRetainedBytes;
  result.key = {backend,
                request.direction,
                result.execution.activeSlots,
                options.gather   ? WorkerMode::Borrowed
                : result.workers ? WorkerMode::Owned
                                 : WorkerMode::Inline,
                options.gather ? unsigned(options.gather->threadCount())
                               : result.execution.gatherThreads,
                options.placementTag,
                affinity()};
  return result;
}
} // namespace reloc::detail
