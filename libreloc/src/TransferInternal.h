// Internal execution contract shared by ephemeral transfers and owned contexts.
#ifndef RELOC_TRANSFERINTERNAL_H
#define RELOC_TRANSFERINTERNAL_H

#include "reloc/ChunkSchedule.h"
#include "reloc/Transfer.h"

#include <memory>

namespace reloc {
class PinnedBufferPool;

namespace detail {

// Request-local only: never store a schedule, plan or buffer address in an
// idle context. The configured buffer count determines this schedule once.
struct TransferRequirements {
  ChunkSchedule schedule{};
  size_t sourceBytes = 0;
  size_t slotBytes = 0;
  size_t stagingBytes = 0;
  int activeSlots = 1;
  unsigned gatherThreads = 1; // resolved owned participants; 1 for borrowed
};

std::variant<TransferRequirements, TransferError>
describeTransfer(const TransferRequest &request,
                 const TransferOptions &options);
std::optional<TransferError>
checkTransferBackend(const TransferRequest &request,
                     const CopyBackend &backend);

// Sets completion before any possible submission and quiesces on error or
// exception. The caller owns storage/owners and retains them on Unknown.
std::optional<TransferError> executePreparedTransfer(
    const TransferRequest &request, const TransferRequirements &requirements,
    const TransferOptions &options, CopyBackend &backend,
    PinnedBufferPool &pool, GatherPool *gather, TransferCompletion &completion);

std::optional<TransferError>
executeH2DPrepared(const BoundPlan &bound, const void *src, void *dst,
                   CopyBackend &backend, PinnedBufferPool &pool,
                   const ChunkSchedule &schedule, int activeSlots,
                   GatherPool *gather, TransferCompletion &completion);

// Preallocated as part of the resource owner, so quarantine never allocates
// while handling failure. The process-lifetime list deliberately never drains.
struct QuarantineNode {
  virtual ~QuarantineNode() = default;
  QuarantineNode *next = nullptr;
};
void quarantine(std::unique_ptr<QuarantineNode> resources) noexcept;

class CompletionGuard {
public:
  CompletionGuard(CopyBackend &backend, TransferCompletion &completion)
      : backend_(backend), completion_(completion) {}
  ~CompletionGuard();

private:
  CopyBackend &backend_;
  TransferCompletion &completion_;
};

} // namespace detail
} // namespace reloc
#endif
