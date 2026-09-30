//===- TransferResources.h - reusable native transfer ownership -*- C++ -*-===//
#ifndef RELOC_TRANSFERRESOURCES_H
#define RELOC_TRANSFERRESOURCES_H

#include "reloc/Transfer.h"

#include <chrono>
#include <memory>

namespace reloc {

namespace detail {
struct TransferContextAccess;
}

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
/// methods and destruction require external serialization. #168 adds leases
/// and bounded multi-context admission around this primitive.
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

} // namespace reloc
#endif
