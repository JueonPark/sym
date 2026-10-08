// Only called with Resources' arena backend: freed scratch is not recycled or
// released until the whole group completes or the context is quarantined.
#ifndef RELOC_DISPATCHGROUPINTERNAL_H
#define RELOC_DISPATCHGROUPINTERNAL_H
#include "reloc/Dispatch.h"
namespace reloc::dispatch::group_detail {
std::optional<TransferError> executeGroup(GroupRequest &, CopyBackend &,
                                          const TransferOptions &);
}
#endif
