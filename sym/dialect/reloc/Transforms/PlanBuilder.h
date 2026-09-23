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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
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

/// One pending pad (at most one per dst axis; same-value re-pads merge).
struct PlanPad {
  int64_t axis;       // dst axis index (kept current across folds)
  Attribute lo;       // sym expression: leading pad width
  Attribute hi;       // sym expression: trailing pad width
  TypedAttr value;    // FUSED fill: the fill in the current element type
  TypedAttr original; // the fill as written (C2), in its entry stage's dtype
  int64_t stage = 0;  // value stages folded before this pad entered (C2)
};

/// One folded typed value stage (C2, issue #142): what the stage does and how
/// the CURRENT dst-view coordinates select its channel parameter. Transposes
/// and reshapes rewrite `channel` (they are index maps), pads shift it by
/// their leading width; finalizeTyped() turns it into the stage's channel
/// map over the logical result coordinates.
struct PlanValueStage {
  ValueTransform transform;
  NumericPolicy policy;
  Type inputType;
  Type outputType;
  SmallVector<Attribute> shape; // logical operand shape when folded
  Attribute scale;     // dense constant or #reloc.binding (casts: null)
  Attribute zeroPoint; // optional
  int64_t axis = -1;   // channel axis on `shape`; -1 per tensor
  AffineExpr channel;  // parameter index over current view dims
};

/// Invariants between transfer-function calls:
/// - axes[k] describes dst axis k (destination order);
/// - perm maps dst axis k to source-view axis perm[k] and is a bijection
///   on [0, axes.size()); a folded reshape re-baselines the view, so perm
///   resets to the identity over the new rank;
/// - axis extents are the VALID (unpadded) extents; pads holds at most one
///   {lo, hi, value} per axis, and a padded axis's dst extent is
///   (extent + lo) + hi, derived at finalize();
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
  /// with canonical row-major strides and the CURRENT element type, the
  /// inverse map is the inverse of perm, divisibility constraints are
  /// emitted in insertion order, and contiguity flags mark axes with
  /// provably-unit source stride (no_copy detection lands in #B5).
  /// Verifier-checked; returns null after emitting an error at `loc` if
  /// verification fails.
  PlanAttr finalize(Location loc) const;

  /// C2: emit the #reloc.typed_plan for a chain that folded at least one
  /// value stage: finalize()'s layout (source dtype -> current dtype, fused
  /// fills), the logical result descriptor (the layout dst at finalize time,
  /// before any canonical axis merging), the stages with their channel maps
  /// over that result's coordinates, and the original fills with their entry
  /// stages. Returns null when no stage was folded (use finalize()) or when
  /// verification fails (after emitting an error at `loc`).
  TypedPlanAttr finalizeTyped(Location loc) const;

  // State is public: transfer functions are free functions over the builder.
  MLIRContext *ctx;
  TensorDescAttr src;
  SmallVector<PlanAxis> axes;
  SmallVector<int64_t> perm;
  SmallVector<DivisibilityAttr> divisibility;
  SmallVector<PlanPad> pads;
  Type elementType;                   // current stage dtype (C2)
  SmallVector<PlanValueStage> stages; // folded value stages, in order (C2)
  SmallVector<StringRef> symbolNames; // affine symbol positions of channels
  std::string bailReason; // stable reason set by a failing transfer function

  /// The pad on dst axis `axis`, or nullptr.
  const PlanPad *findPad(int64_t axis) const {
    for (const PlanPad &pad : pads)
      if (pad.axis == axis)
        return &pad;
    return nullptr;
  }
};

/// #B1 transfer function: fold `reloc.transpose` with permutation `opPerm`
/// (result dim k = operand dim opPerm[k]) into `plan` by permuting the axes
/// and composing the permutations (perm[k] <- perm[opPerm[k]]). Emits no
/// constraints and never fails (the result is always success(), kept for a
/// uniform transfer-function API); `opPerm` must be a valid permutation of
/// the current rank (guaranteed by the reloc.transpose verifier).
/// Pads travel with their axes: their indices are renumbered through the
/// permutation.
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

/// #B3 transfer function: fold `reloc.pad` on dst axis `axis` with
/// leading/trailing widths `lo`/`hi` (sym expressions, tensor.pad
/// convention) and fill `value`. The axis's valid extent is unchanged —
/// the pad is tracked separately and grows the dst extent at finalize().
/// A second pad on the same axis merges when the fill values are equal
/// (the new pad wraps the old valid region). Fails, leaving `plan`
/// untouched, on: an out-of-range axis, non-expression or provably
/// negative widths, or a differing fill value on an already-padded axis.
LogicalResult foldPad(PlanBuilder &plan, int64_t axis, Attribute lo,
                      Attribute hi, TypedAttr value);

/// C2 transfer function: fold one typed value transform (C1's reloc.cast /
/// reloc.quantize / reloc.dequantize, already verified) into `plan`. Every
/// pending pad's fused fill is folded through the stage with the C1
/// reference arithmetic; a per-channel or runtime parameter makes a single
/// fused fill impossible, so the fold fails with bailReason
/// "fill_not_foldable". On success the stage is recorded with its channel
/// expression (the operand's channel-axis coordinate) and the element type
/// advances. Fails, leaving `plan` untouched, when the input type is not
/// the current element type or `shape` does not match the current rank.
LogicalResult foldValueStage(PlanBuilder &plan, ValueTransform transform,
                             NumericPolicy policy, Type inputType,
                             Type outputType, ArrayRef<Attribute> shape,
                             Attribute scale, Attribute zeroPoint,
                             std::optional<int64_t> axis);

/// True for ops the transfer functions can fold: the layout ops (#B1-#B3:
/// reloc.transpose, reloc.reshape, reloc.pad) and, since C2, the typed value
/// transforms (reloc.cast, reloc.quantize, reloc.dequantize).
bool isFoldableChainOp(Operation *op);

/// Dispatch one chain op through its transfer function. Returns failure
/// for a transfer-function bail or a non-foldable op; a typed bail leaves
/// its stable reason in plan.bailReason (empty for layout bails).
LogicalResult foldChainOp(PlanBuilder &plan, Operation *op);

/// #B5: canonicalize `plan` so equivalent chains yield the same attribute:
/// re-simplify every expression through sym's simplifier, merge adjacent
/// axes that are contiguous-compatible on both the src and dst sides and
/// view-adjacent (perm[k+1] == perm[k]+1; padded/aligned axes never
/// merge), rename axes d0..d(n-1), rebuild the dst descriptor (dst rank
/// collapses with the axes), and recompute no_copy from isPureView.
/// Plans outside fold-normal form (explicit dst strides or a
/// non-permutation inverse) get the safe subset: constant folding and
/// no_copy recomputation only. Idempotent. Returns null after emitting a
/// diagnostic at `loc` if the rebuilt plan fails verification.
PlanAttr canonicalizePlan(PlanAttr plan, Location loc);

/// C2: canonicalize a typed plan. The layout goes through canonicalizePlan
/// with no_copy cleared afterwards (a value program is never a view; R3 asks
/// isPureView(layout) for the index-map question), descriptors and stage
/// shapes are constant-folded, channel maps simplified, fills sorted by dst
/// axis with their bits untouched. No stage is ever removed: no C1 policy is
/// an identity and lossy sequences are never cancelled. Idempotent. Returns
/// null after emitting a diagnostic at `loc` if verification fails.
TypedPlanAttr canonicalizeTypedPlan(TypedPlanAttr plan, Location loc);

} // namespace reloc
} // namespace mlir

#endif // RELOC_PLAN_BUILDER_H
