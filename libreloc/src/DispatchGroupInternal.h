// Only called with Resources' arena backend: freed scratch is not recycled or
// released until the whole group completes or the context is quarantined.
#ifndef RELOC_DISPATCHGROUPINTERNAL_H
#define RELOC_DISPATCHGROUPINTERNAL_H
#include "reloc/Dispatch.h"
#include <functional>
namespace reloc::dispatch::group_detail {
using HostCompletion = std::function<std::optional<TransferError>()>;
std::optional<TransferError>
executeGroup(GroupRequest &, CopyBackend &, const TransferOptions &,
             std::vector<HostCompletion> *deferred = nullptr);
} // namespace reloc::dispatch::group_detail
#endif
