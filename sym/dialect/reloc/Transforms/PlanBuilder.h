//===- PlanBuilder.h - In-progress reloc plan state -------------*- C++ -*-===//
//
// PlanBuilder is the mutable folding state for one straight-line reloc.*
// chain (P1b). It is seeded as the identity plan over the chain's source
// tensor; transfer functions (foldTranspose; foldReshape/foldPad in later
// milestones) rewrite it in place; finalize() emits the verifier-checked
// #reloc.plan attribute.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PLAN_BUILDER_H
#define RELOC_PLAN_BUILDER_H

#include "RelocDialect.h"
#include "SymDialect.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir {
namespace reloc {

/// One plan axis in destination order. dst strides are derived from the
/// axis extents at finalize() time, so they are not stored here.
struct PlanAxis {
  std::string name;
  Attribute extent;    // sym expression
  Attribute srcStride; // sym expression: stride in the source buffer
};

/// Invariants between transfer-function calls:
/// - axes[k] describes dst axis k (destination order);
/// - perm maps dst axis k to source-view axis perm[k] and is a bijection
///   on [0, axes.size()); a folded reshape re-baselines the view, so perm
///   resets to the identity over the new rank;
/// - axis extents are the current dst extents (no pad widths until #B3);
/// - divisibility holds the constraints emitted by folds so far, in
///   emission order, deduplicated.
class PlanBuilder {
public:
  /// Seed the identity plan over `input` (rank >= 1): the src descriptor
  /// takes the type's extents with canonical (elided) row-major strides and
  /// zero offset; axis k is named "d<k>" with the row-major src stride;
  /// perm is the identity.
  explicit PlanBuilder(sym::SymbolicTensorType input);

  /// Emit the #reloc.plan for the current state: dst gets the axis extents
  /// with canonical row-major strides, the inverse map is the inverse of
  /// perm, divisibility constraints are emitted in insertion order, and
  /// contiguity flags mark axes with provably-unit source stride (no_copy
  /// detection lands in #B5). Verifier-checked; returns null after
  /// emitting an error at `loc` if verification fails.
  PlanAttr finalize(Location loc) const;

  // State is public: transfer functions are free functions over the builder.
  MLIRContext *ctx;
  TensorDescAttr src;
  SmallVector<PlanAxis> axes;
  SmallVector<int64_t> perm;
  SmallVector<DivisibilityAttr> divisibility;
};

/// #B1 transfer function: fold `reloc.transpose` with permutation `opPerm`
/// (result dim k = operand dim opPerm[k]) into `plan` by permuting the axes
/// and composing the permutations (perm[k] <- perm[opPerm[k]]). Emits no
/// constraints and never fails (the result is always success(), kept for a
/// uniform transfer-function API); `opPerm` must be a valid permutation of
/// the current rank (guaranteed by the reloc.transpose verifier).
LogicalResult foldTranspose(PlanBuilder &plan, ArrayRef<int64_t> opPerm);

/// #B2 transfer function: fold `reloc.reshape` with `targetShape` (sym
/// expression attributes, the op's target_shape) into `plan`. The current
/// dst view is dense row-major, so the reshape decomposes into per-axis
/// splits and contiguity-checked merges; a successful fold re-baselines
/// the view (perm = identity over the new rank, axes renamed d0..d(n-1))
/// and may append divisibility constraints (symbolic splits). Returns
/// failure — leaving `plan` untouched — when no factorization exists with
/// the available symbolic facts (issue #15 design decision 1: bail, never
/// a wrong plan).
LogicalResult foldReshape(PlanBuilder &plan, ArrayRef<Attribute> targetShape);

} // namespace reloc
} // namespace mlir

#endif // RELOC_PLAN_BUILDER_H
