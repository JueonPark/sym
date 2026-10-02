// Private preparation and compatibility contract for the bounded cache.
#ifndef RELOC_TRANSFERRESOURCEPOLICY_H
#define RELOC_TRANSFERRESOURCEPOLICY_H

#include "TransferInternal.h"
#include "reloc/TransferResources.h"

namespace reloc::detail {

struct TransferContextAccess {
  static TransferCompletion completion(const TransferContext &);
  // An exclusive, idle context only. No submission; may throw on construction.
  // capacity controls allocation only, never the already computed schedule.
  static void prepare(TransferContext &, TransferDirection,
                      const TransferRequirements &, size_t capacity);
  // The caller already validated and claimed this single-use request.
  static TransferOutcome execute(TransferRequest &, TransferContext &,
                                 const TransferOptions &,
                                 const TransferRequirements &,
                                 std::shared_ptr<void>);
};

enum class WorkerMode { Inline, Owned, Borrowed };

struct ResourceKey {
  TransferBackendConfig backend;
  TransferDirection direction;
  int activeSlots;
  WorkerMode workers;
  unsigned participants;
  uint64_t placementTag;
  std::vector<unsigned long> affinity;
  bool operator==(const ResourceKey &) const;
  bool sameDevice(const ResourceKey &) const;
};

struct CacheRequest {
  TransferRequirements execution;
  TransferOptions options;
  ResourceKey key;
  size_t slotCapacity = 0;
  size_t bytes = 0;
  unsigned workers = 0;
  bool cacheable = false;
};

std::variant<size_t, TransferError> roundStagingCapacity(size_t bytes);
std::variant<CacheRequest, TransferError>
describeCachedTransfer(const TransferRequest &, const CachedTransferOptions &,
                       const TransferResourceLimits &);

std::vector<unsigned long> currentCpuAffinity();

struct ProcessIdentity {
  int64_t pid;
  uint64_t epoch;
  bool operator==(const ProcessIdentity &other) const {
    return pid == other.pid && epoch == other.epoch;
  }
};
// Registration only at construction. currentProcessIdentity() takes no locks.
ProcessIdentity captureProcessIdentity();
ProcessIdentity currentProcessIdentity();

} // namespace reloc::detail
#endif
