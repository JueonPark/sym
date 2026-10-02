//===- TransferResources.h - reusable native transfer ownership -*- C++ -*-===//
#ifndef RELOC_TRANSFERRESOURCES_H
#define RELOC_TRANSFERRESOURCES_H

#include "reloc/Transfer.h"

#include <chrono>
#include <functional>
#include <memory>

namespace reloc {

namespace detail {
struct TransferContextAccess;
}

/// Budgets are per cache and include reservations and quarantined resources.
/// Workers count owned gather threads, excluding the calling thread and any
/// externally owned GatherPool. The timeout bounds admission waits only.
struct TransferResourceLimits {
  size_t maxRetainedBytes = size_t(256) << 20;
  size_t maxContexts = 4;
  size_t maxContextsPerDevice = 2;
  size_t maxBackgroundWorkers = 64;
  std::optional<size_t> maxLiveStagingBytes;
  std::optional<std::chrono::milliseconds> acquireTimeout;
};

struct TransferBackendConfig {
  MemoryKind kind = MemoryKind::Host;
  int device = -1; // explicit CUDA ordinal; -1 for Host
  int streams = 2;
  bool pinned = true; // resolved from policy and the request's wire bytes
};

struct CachedTransferOptions {
  TransferOptions transfer;
  TransferBackendConfig backend;
  // Strong ownership for the blocking call only. Overrides gatherThreads.
  // A raw transfer.gather without this matching owner is rejected.
  std::shared_ptr<GatherPool> gather;
  // An additional placement-policy identity, not a request to change affinity.
  // The actual calling thread's Linux CPU-affinity mask is always included.
  uint64_t placementTag = 0;
};

struct TransferContextStats {
  size_t stagingBytes = 0;
  size_t slotBytes = 0;
  int activeSlots = 0;
  unsigned backgroundWorkers = 0;
  size_t stagingPoolCreations = 0;
  size_t workerPoolCreations = 0;
  bool reusable = true;
  bool quarantined = false;
};

/// One exclusively used resource owner, not a CUDA context or a cache. Takes
/// ownership of a fresh, dedicated backend; staging/workers are lazy. All
/// methods and destruction require external serialization.
/// TransferResourceCache provides leases and bounded multi-context admission
/// around this primitive.
class TransferContext {
public:
  explicit TransferContext(std::unique_ptr<CopyBackend> backend);
  ~TransferContext();
  TransferContext(const TransferContext &) = delete;
  TransferContext &operator=(const TransferContext &) = delete;

  /// Release resources after completed execution. Unknown completion moves
  /// resources and buffer owners into process-lifetime quarantine instead.
  /// Idempotent; an explicitly closed or failed context cannot execute again.
  TransferCompletion close() noexcept;
  TransferContextStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  TransferContextStats stats_;
  friend struct detail::TransferContextAccess;
  friend TransferOutcome executeTransfer(TransferRequest &, TransferContext &,
                                         const TransferOptions &,
                                         std::shared_ptr<void>);
};

/// Blocking execution with reusable resources and exceptional buffer lifetime.
/// bufferOwners must strongly own BOTH allocations; it is released on success
/// or established completion, and retained indefinitely on Unknown completion.
/// The context stores no successful request's plan, pointers, or caller stream.
/// A borrowed options.gather must remain alive through this call only; it is
/// never owned/closed/retained by the context. Errors retire the context after
/// execution claims the request; preflight rejections leave it usable.
TransferOutcome executeTransfer(TransferRequest &request,
                                TransferContext &context,
                                const TransferOptions &options,
                                std::shared_ptr<void> bufferOwners);

// Limits and statistics describe one explicit owner, not a process-global pool.
// Reserved values are additional, not inclusive of allocated values. Their sum
// is the admission charge, held through construction, retirement and
// quarantine.
struct TransferResourceUsage {
  size_t contexts = 0, building = 0, leased = 0, idle = 0, retiring = 0,
         quarantined = 0;
  size_t allocatedStagingBytes = 0, reservedStagingBytes = 0;
  size_t retainedAllocatedBytes = 0, retainedReservedBytes = 0;
  size_t backgroundWorkers = 0, reservedWorkers = 0, streams = 0;
  size_t quarantineBytes = 0, outstandingEvents = 0;
};

struct TransferDeviceStats : TransferResourceUsage {
  MemoryKind kind = MemoryKind::Host;
  int device = -1;
  bool disabled = false;
};

struct TransferResourceStats : TransferResourceUsage {
  uint64_t requests = 0, hits = 0, growths = 0, misses = 0, evictions = 0;
  uint64_t ephemeral = 0, waits = 0, timeouts = 0, failures = 0,
           quarantines = 0;
  uint64_t stagingAllocations = 0, stagingFrees = 0;
  uint64_t streamCreations = 0, streamDestructions = 0;
  uint64_t workerCreations = 0, workerJoins = 0;
  uint64_t eventCreations = 0, eventRetirements = 0;
  size_t peakLiveStagingBytes = 0, peakAllocatedStagingBytes = 0, waiters = 0;
  uint64_t generation = 0;
  bool closed = false, processValid = true;
  std::vector<TransferDeviceStats> devices;
};

using TransferBackendFactory =
    std::function<std::unique_ptr<CopyBackend>(const TransferBackendConfig &)>;

/// Thread-safe cache with one exclusive native lease per blocking execution.
/// Construction is lazy. A factory must return a fresh dedicated backend with
/// exactly the requested device/queue count and no previously submitted work.
/// It must roll back partial construction and be safe for concurrent calls.
/// The wrapper must outlive all calls using it; close() may race with
/// execution, but destruction must not. Inherited instances reject use after
/// fork and abandon their resources on child destruction without entering
/// backend code.
class TransferResourceCache {
public:
  explicit TransferResourceCache(TransferResourceLimits limits = {},
                                 TransferBackendFactory factory = {});
  ~TransferResourceCache();
  TransferResourceCache(const TransferResourceCache &) = delete;
  TransferResourceCache &operator=(const TransferResourceCache &) = delete;

  /// Evict idle resources; older building/active leases retire on return.
  std::optional<TransferError> clear();
  /// Stop admission and wait for normal construction/execution/retirement.
  /// Unknown completion remains charged and returns completion_unknown.
  std::optional<TransferError> close();
  /// Safe without CUDA initialization. A fork-inherited object returns a
  /// processValid=false snapshot without entering inherited locks.
  TransferResourceStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend TransferOutcome executeTransferCached(TransferRequest &,
                                               TransferResourceCache &,
                                               const CachedTransferOptions &,
                                               std::shared_ptr<void>);
};

/// Validates once, then claims the request before admission. Limit, timeout,
/// construction and execution errors after that point consume it. The caller
/// must serialize access to the request and strongly own both buffers in the
/// supplied token; no successful buffer or external gather owner is cached.
TransferOutcome executeTransferCached(TransferRequest &,
                                      TransferResourceCache &,
                                      const CachedTransferOptions &,
                                      std::shared_ptr<void> bufferOwners);

} // namespace reloc
#endif
