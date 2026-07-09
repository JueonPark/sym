//===- PlanCanonicalize.cpp - Canonicalize folded reloc plans -------------===//
//
// #B5: minimize plans so equivalent chains yield byte-identical serialized
// plans. Inverse-pair elimination is emergent: transpose pairs already
// fold to the identity perm, split/merge pairs collapse via adjacent-axis
// merging, and isPureView then sets no_copy.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocUtils.h"
#include "SymDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::reloc;

/// Rebuild an expression bottom-up through sym's simplifier ("constant
/// folding ... no new simplifier", issue #25).
static Attribute resimplify(Attribute expr, MLIRContext *ctx) {
  auto binary = dyn_cast_or_null<sym::BinaryExprAttr>(expr);
  if (!binary)
    return expr;
  return sym::getSimplifiedBinaryExpr(ctx, binary.getOpcode(),
                                      resimplify(binary.getLhs(), ctx),
                                      resimplify(binary.getRhs(), ctx));
}

namespace {
struct CanonAxis {
  Attribute extent;
  Attribute srcStride;
  Attribute dstStride;
};
} // namespace

/// Contiguity on a chosen stride side, via the P1a predicate.
static bool contiguousOn(const CanonAxis &outer, const CanonAxis &inner,
                         bool srcSide, MLIRContext *ctx) {
  Attribute outerStride = srcSide ? outer.srcStride : outer.dstStride;
  Attribute innerStride = srcSide ? inner.srcStride : inner.dstStride;
  auto info = [&](Attribute extent, Attribute stride) {
    return AxisInfoAttr::get(ctx, "t", extent, stride, stride);
  };
  return isContiguousCompatible(info(outer.extent, outerStride),
                                info(inner.extent, innerStride));
}

PlanAttr mlir::reloc::canonicalizePlan(PlanAttr plan, Location loc) {
  MLIRContext *ctx = plan.getContext();
  auto simp = [&](Attribute expr) { return resimplify(expr, ctx); };
  auto mul = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Mul, lhs,
                                        rhs);
  };
  auto add = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add, lhs,
                                        rhs);
  };

  // Constant-fold the src descriptor (never merged).
  SmallVector<Attribute> srcExtents, srcStrides;
  for (Attribute extent : plan.getSrc().getExtents())
    srcExtents.push_back(simp(extent));
  for (Attribute stride : plan.getSrc().getStrides())
    srcStrides.push_back(simp(stride));
  auto src = TensorDescAttr::get(ctx, srcExtents, srcStrides,
                                 simp(plan.getSrc().getOffset()),
                                 plan.getSrc().getElementType());

  // Extract mutable state.
  SmallVector<CanonAxis> axes;
  for (AxisInfoAttr axis : plan.getAxes())
    axes.push_back({simp(axis.getExtent()), simp(axis.getSrcStride()),
                    simp(axis.getDstStride())});
  SmallVector<int64_t> perm(plan.getPerm().asArrayRef().begin(),
                            plan.getPerm().asArrayRef().end());
  SmallVector<PadFillAttr> pads;
  for (PadFillAttr pad : plan.getPadFill())
    pads.push_back(PadFillAttr::get(ctx, pad.getDstAxis(), simp(pad.getLo()),
                                    simp(pad.getHi()), pad.getValue()));
  SmallVector<AlignmentAttr> alignments(plan.getAlignment().begin(),
                                        plan.getAlignment().end());
  SmallVector<DivisibilityAttr> divisibility;
  for (DivisibilityAttr constraint : plan.getDivisibility()) {
    auto simplified = DivisibilityAttr::get(ctx, simp(constraint.getExpr()),
                                            constraint.getDivisor());
    if (!llvm::is_contained(divisibility, simplified))
      divisibility.push_back(simplified);
  }

  // Fold-normal form gate: axis merging needs a permutation inverse and a
  // canonical (elided) row-major dst.
  bool foldNormal = plan.getDst().getStrides().empty() &&
                    plan.getInverse().getValue().isPermutation();

  if (foldNormal) {
    auto referenced = [&](size_t k) {
      for (PadFillAttr pad : pads)
        if (pad.getDstAxis() == static_cast<int64_t>(k))
          return true;
      for (AlignmentAttr alignment : alignments)
        if (alignment.getAxis() == static_cast<int64_t>(k))
          return true;
      return false;
    };
    // Fixpoint adjacent-axis merging.
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t k = 0; k + 1 < axes.size(); ++k) {
        if (referenced(k) || referenced(k + 1))
          continue;
        if (perm[k + 1] != perm[k] + 1)
          continue; // not view-adjacent
        if (!contiguousOn(axes[k], axes[k + 1], /*srcSide=*/true, ctx) ||
            !contiguousOn(axes[k], axes[k + 1], /*srcSide=*/false, ctx))
          continue;
        axes[k] = {mul(axes[k].extent, axes[k + 1].extent),
                   axes[k + 1].srcStride, axes[k + 1].dstStride};
        axes.erase(axes.begin() + k + 1);
        int64_t removedView = perm[k] + 1;
        perm.erase(perm.begin() + k + 1);
        for (int64_t &view : perm)
          if (view > removedView)
            --view;
        // Guarded above: no pad/alignment ON k or k+1; remap the rest.
        SmallVector<PadFillAttr> remappedPads;
        for (PadFillAttr pad : pads)
          remappedPads.push_back(PadFillAttr::get(
              ctx,
              pad.getDstAxis() > static_cast<int64_t>(k) ? pad.getDstAxis() - 1
                                                         : pad.getDstAxis(),
              pad.getLo(), pad.getHi(), pad.getValue()));
        pads = std::move(remappedPads);
        SmallVector<AlignmentAttr> remappedAlignments;
        for (AlignmentAttr alignment : alignments)
          remappedAlignments.push_back(
              AlignmentAttr::get(ctx,
                                 alignment.getAxis() > static_cast<int64_t>(k)
                                     ? alignment.getAxis() - 1
                                     : alignment.getAxis(),
                                 alignment.getBytes()));
        alignments = std::move(remappedAlignments);
        changed = true;
        break; // indices shifted: restart the scan
      }
    }
  }

  // Rebuild. dst extents = axis extents plus pad widths, verifier
  // association (extent + lo) + hi.
  SmallVector<Attribute> dstExtents;
  for (auto [k, axis] : llvm::enumerate(axes)) {
    Attribute extent = axis.extent;
    for (PadFillAttr pad : pads)
      if (pad.getDstAxis() == static_cast<int64_t>(k))
        extent = add(add(extent, pad.getLo()), pad.getHi());
    dstExtents.push_back(extent);
  }
  SmallVector<Attribute> dstStrides;
  if (foldNormal)
    dstStrides = canonicalRowMajorStrides(dstExtents, ctx);
  else
    for (const CanonAxis &axis : axes)
      dstStrides.push_back(axis.dstStride);
  SmallVector<Attribute> dstDescStrides;
  if (!plan.getDst().getStrides().empty())
    for (Attribute stride : plan.getDst().getStrides())
      dstDescStrides.push_back(simp(stride));
  auto dst = TensorDescAttr::get(ctx, dstExtents, dstDescStrides,
                                 simp(plan.getDst().getOffset()),
                                 plan.getDst().getElementType());

  Attribute one = sym::ConstantExprAttr::get(ctx, 1);
  SmallVector<AxisInfoAttr> axisAttrs;
  SmallVector<bool> contiguous;
  for (auto [k, axis] : llvm::enumerate(axes)) {
    axisAttrs.push_back(AxisInfoAttr::get(
        ctx, ("d" + Twine(k)).str(), axis.extent, axis.srcStride,
        foldNormal ? dstStrides[k] : axis.dstStride));
    contiguous.push_back(proveEqual(axis.srcStride, one) == Proof::Proven);
  }

  AffineMapAttr inverse = plan.getInverse();
  if (foldNormal) {
    AffineMap forward = AffineMap::getPermutationMap(perm, ctx);
    inverse = AffineMapAttr::get(inversePermutation(forward));
  }

  bool runtimePadCheck = false;
  for (PadFillAttr pad : pads)
    if (provePadRange(pad, axisAttrs, dst) == Proof::Unknown)
      runtimePadCheck = true;

  // no_copy: recomputed from the P1a predicate, never trusted.
  PlanAttr candidate = PlanAttr::get(
      ctx, src, dst, DenseI64ArrayAttr::get(ctx, perm), axisAttrs,
      ArrayRef<PadFillAttr>(pads), ArrayRef<DivisibilityAttr>(divisibility),
      ArrayRef<AlignmentAttr>(alignments),
      DenseBoolArrayAttr::get(ctx, contiguous), /*noCopy=*/false,
      runtimePadCheck, inverse);
  bool noCopy = isPureView(candidate);

  return PlanAttr::getChecked(
      [&]() { return emitError(loc); }, ctx, src, dst,
      DenseI64ArrayAttr::get(ctx, perm), ArrayRef<AxisInfoAttr>(axisAttrs),
      ArrayRef<PadFillAttr>(pads), ArrayRef<DivisibilityAttr>(divisibility),
      ArrayRef<AlignmentAttr>(alignments),
      DenseBoolArrayAttr::get(ctx, contiguous), noCopy, runtimePadCheck,
      inverse);
}
