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
///   on [0, axes.size());
/// - axis extents are the current dst extents (no pad widths until #B3).
class PlanBuilder {
public:
  /// Seed the identity plan over `input` (rank >= 1): the src descriptor
  /// takes the type's extents with canonical (elided) row-major strides and
  /// zero offset; axis k is named "d<k>" with the row-major src stride;
  /// perm is the identity.
  explicit PlanBuilder(sym::SymbolicTensorType input);

  /// Emit the #reloc.plan for the current state: dst gets the axis extents
  /// with canonical row-major strides, the inverse map is the inverse of
  /// perm, and no constraint sections are emitted (no_copy detection lands
  /// in #B5). Verifier-checked; returns null after emitting an error at
  /// `loc` if verification fails.
  PlanAttr finalize(Location loc) const;

  // State is public: transfer functions are free functions over the builder.
  MLIRContext *ctx;
  TensorDescAttr src;
  SmallVector<PlanAxis> axes;
  SmallVector<int64_t> perm;
};

/// Helper to implement PlanBuilder::finalize (defined in RelocAttributes.cpp
/// where storage definitions are available for template instantiation).
PlanAttr finalizePlanBuilder(const PlanBuilder &builder, Location loc);

} // namespace reloc
} // namespace mlir

#endif // RELOC_PLAN_BUILDER_H
