//===- Prefold.cpp - P4 load-time pre-folded transform (issue #98) --------===//

#include "reloc/Prefold.h"

#include "reloc/GatherPool.h"

namespace reloc {
namespace prefold {

bool prefoldWins(int64_t nReuse, double tTransformMs, double tPrefoldMs,
                 double penaltyMs) {
  if (nReuse < 1)
    return false;
  return static_cast<double>(nReuse) * tTransformMs >
         tPrefoldMs + penaltyMs;
}

} // namespace prefold
} // namespace reloc
