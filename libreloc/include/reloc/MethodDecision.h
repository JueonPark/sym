//===- MethodDecision.h - pattern + method-decision value types -*- C++ -*-===//
//
// Split out of CostModel.h to break an include cycle: Bind.h needs
// MethodDecision (BoundPlan::decision) but CostModel.h includes Bind.h
// (classify() takes a const BoundPlan&). This header has no dependency
// on BoundPlan and may be included by both.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_METHODDECISION_H
#define RELOC_METHODDECISION_H

#include <cstdint>

namespace reloc {
namespace costmodel {

/// Host-side access pattern of Method A's transform, classified from the
/// coalesced BoundPlan alone. Misclassifying the pattern dominates every
/// other modelling error (R1 measured a >20x spread across these).
enum class Pattern { Contiguous, Blocked, SingleElement, Tiled };

const char *patternName(Pattern p);

/// L == totalElems -> Contiguous; L == 1 -> SingleElement;
/// L >= kBlockedRunFloor -> Blocked; else Tiled.
constexpr int64_t kBlockedRunFloor = 64; // elements

struct MethodDecision {
  enum class Method { A, B, APrefold };
  Method method = Method::B;
  double tAMs = 0, tBMs = 0;  // chosen-arm A is APrefold's when nReuse>0
  double thresholdBytes = -1; // argmin boundary nearest the bound S;
                              // -1 = decision is size-independent
  Pattern pattern = Pattern::Contiguous;
  int k = 1;
  int64_t nReuse = -1;
};

const char *methodName(MethodDecision::Method m);

} // namespace costmodel
} // namespace reloc

#endif // RELOC_METHODDECISION_H
