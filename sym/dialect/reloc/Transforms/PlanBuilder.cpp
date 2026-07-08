//===- PlanBuilder.cpp - In-progress reloc plan state ---------------------===//
//
// This file implements the P1b folding state and its transfer functions.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocUtils.h"
#include "SymUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::reloc;

PlanBuilder::PlanBuilder(sym::SymbolicTensorType input)
    : ctx(input.getContext()) {
  ArrayRef<Attribute> shape = input.getShape();
  assert(!shape.empty() && "rank-0 tensors have no relocation plan");
  Attribute zero = sym::ConstantExprAttr::get(ctx, 0);
  src = TensorDescAttr::get(ctx, shape, /*strides=*/ArrayRef<Attribute>(), zero,
                            input.getElementType());
  SmallVector<Attribute> srcStrides = canonicalRowMajorStrides(shape, ctx);
  for (auto [k, extent] : llvm::enumerate(shape)) {
    axes.push_back({("d" + Twine(k)).str(), extent, srcStrides[k]});
    perm.push_back(static_cast<int64_t>(k));
  }
}

PlanAttr PlanBuilder::finalize(Location loc) const {
  auto add = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add, lhs,
                                        rhs);
  };
  SmallVector<Attribute> dstExtents;
  dstExtents.reserve(axes.size());
  for (auto [k, axis] : llvm::enumerate(axes)) {
    Attribute extent = axis.extent;
    // (extent + lo) + hi: the same association the plan verifier proves
    // against, so the equality is uniqued-attribute identity.
    if (const PlanPad *pad = findPad(static_cast<int64_t>(k)))
      extent = add(add(extent, pad->lo), pad->hi);
    dstExtents.push_back(extent);
  }
  Attribute zero = sym::ConstantExprAttr::get(ctx, 0);
  auto dst = TensorDescAttr::get(ctx, dstExtents,
                                 /*strides=*/ArrayRef<Attribute>(), zero,
                                 src.getElementType());
  SmallVector<Attribute> dstStrides = canonicalRowMajorStrides(dstExtents, ctx);
  SmallVector<AxisInfoAttr> axisAttrs;
  axisAttrs.reserve(axes.size());
  for (auto [k, axis] : llvm::enumerate(axes))
    axisAttrs.push_back(AxisInfoAttr::get(ctx, axis.name, axis.extent,
                                          axis.srcStride, dstStrides[k]));
  AffineMap forward = AffineMap::getPermutationMap(perm, ctx);
  auto inverse = AffineMapAttr::get(inversePermutation(forward));
  Attribute one = sym::ConstantExprAttr::get(ctx, 1);
  SmallVector<bool> contiguous;
  contiguous.reserve(axes.size());
  for (const PlanAxis &axis : axes)
    contiguous.push_back(proveEqual(axis.srcStride, one) == Proof::Proven);
  SmallVector<PadFillAttr> padAttrs;
  padAttrs.reserve(pads.size());
  for (const PlanPad &pad : pads)
    padAttrs.push_back(
        PadFillAttr::get(ctx, pad.axis, pad.lo, pad.hi, pad.value));
  // Symbolic pad ranges are not statically provable; the verifier accepts
  // them only under runtime_pad_check.
  bool runtimePadCheck = false;
  for (PadFillAttr pad : padAttrs)
    if (provePadRange(pad, axisAttrs, dst) == Proof::Unknown)
      runtimePadCheck = true;
  return PlanAttr::getChecked(
      [&]() { return emitError(loc); }, ctx, src, dst,
      DenseI64ArrayAttr::get(ctx, perm), ArrayRef<AxisInfoAttr>(axisAttrs),
      ArrayRef<PadFillAttr>(padAttrs), ArrayRef<DivisibilityAttr>(divisibility),
      /*alignment=*/ArrayRef<AlignmentAttr>(),
      DenseBoolArrayAttr::get(ctx, contiguous),
      /*noCopy=*/false, runtimePadCheck, inverse);
}

LogicalResult mlir::reloc::foldTranspose(PlanBuilder &plan,
                                         ArrayRef<int64_t> opPerm) {
  int64_t rank = static_cast<int64_t>(plan.axes.size());
  assert(static_cast<int64_t>(opPerm.size()) == rank &&
         "transpose perm size must match plan rank");
  SmallVector<PlanAxis> newAxes;
  SmallVector<int64_t> newPerm;
  newAxes.reserve(rank);
  newPerm.reserve(rank);
  for (int64_t source : opPerm) {
    assert(source >= 0 && source < rank && "perm entry out of range");
    // New dst axis k <- old dst axis opPerm[k].
    newAxes.push_back(plan.axes[source]);
    newPerm.push_back(plan.perm[source]);
  }
  // Pads travel with their axis: renumber through the permutation.
  SmallVector<int64_t> newIndexOfOld(rank);
  for (int64_t k = 0; k < rank; ++k)
    newIndexOfOld[opPerm[k]] = k;
  for (PlanPad &pad : plan.pads)
    pad.axis = newIndexOfOld[pad.axis];
  plan.axes = std::move(newAxes);
  plan.perm = std::move(newPerm);
  return success();
}

//===----------------------------------------------------------------------===//
// foldReshape (#B2)
//===----------------------------------------------------------------------===//

/// True iff `attr` is a ConstantExprAttr, extracting its value.
static bool getConstant(Attribute attr, int64_t &value) {
  if (auto constant = dyn_cast_or_null<sym::ConstantExprAttr>(attr)) {
    value = constant.getValue();
    return true;
  }
  return false;
}

/// True iff `targets` is a valid decomposition of one axis of extent
/// `extent`. Static rule: all entries constant with product == extent.
/// One-symbolic rule: exactly one non-constant entry, logically equal to
/// `extent floordiv C` where C is the product of the remaining (constant)
/// entries — sets needsDivisibility (unless C == 1) and divisor = C.
static bool matchesSplit(Attribute extent, ArrayRef<Attribute> targets,
                         MLIRContext *ctx, bool &needsDivisibility,
                         int64_t &divisor) {
  needsDivisibility = false;
  int64_t extentValue;
  int64_t constProduct = 1;
  Attribute symbolic;
  for (Attribute target : targets) {
    int64_t value;
    if (getConstant(target, value)) {
      constProduct *= value;
      continue;
    }
    if (symbolic)
      return false; // at most one symbolic entry in v0
    symbolic = target;
  }
  if (!symbolic)
    return getConstant(extent, extentValue) && extentValue == constProduct;
  if (getConstant(extent, extentValue))
    return false; // constant extent cannot absorb a symbolic entry
  Attribute quotient = sym::getSimplifiedBinaryExpr(
      ctx, sym::SymbolicExprOp::Div, extent,
      sym::ConstantExprAttr::get(ctx, constProduct));
  if (!sym::UnificationSolver::areLogicallyEqual(symbolic, quotient))
    return false;
  needsDivisibility = constProduct > 1;
  divisor = constProduct;
  return true;
}

/// P1a contiguity predicate over builder axes: outer.srcStride ==
/// inner.srcStride * inner.extent, provably.
static bool contiguousCompatible(const PlanAxis &outer, const PlanAxis &inner,
                                 MLIRContext *ctx) {
  auto info = [&](const PlanAxis &axis) {
    return AxisInfoAttr::get(ctx, axis.name, axis.extent, axis.srcStride,
                             axis.srcStride);
  };
  return isContiguousCompatible(info(outer), info(inner));
}

/// Append the axes of a split: stride peeling right-to-left from
/// `srcStride`. Names are assigned by the caller after the full match.
static void appendSplitAxes(ArrayRef<Attribute> targets, Attribute srcStride,
                            SmallVectorImpl<PlanAxis> &out, MLIRContext *ctx) {
  SmallVector<Attribute> strides(targets.size());
  Attribute running = srcStride;
  for (int64_t t = static_cast<int64_t>(targets.size()) - 1; t >= 0; --t) {
    strides[t] = running;
    running = sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Mul,
                                           running, targets[t]);
  }
  for (auto [extent, stride] : llvm::zip(targets, strides))
    out.push_back({"", extent, stride});
}

LogicalResult mlir::reloc::foldReshape(PlanBuilder &plan,
                                       ArrayRef<Attribute> targetShape) {
  MLIRContext *ctx = plan.ctx;
  ArrayRef<PlanAxis> old = plan.axes;
  if (targetShape.empty())
    return failure();
  for (Attribute target : targetShape)
    if (!isSymExpr(target))
      return failure();
  // The frozen P1a reshape verifier only checks element-count equality, not
  // sign, so a decomposition like [24] -> [-8, -3] (matching element count)
  // passes it. Guard here so the transform layer never folds a non-positive
  // constant extent into negative extents/strides (bail-never-wrong-plan).
  for (Attribute target : targetShape) {
    int64_t value;
    if (getConstant(target, value) && value <= 0)
      return failure();
  }
  for (const PlanAxis &axis : old) {
    int64_t value;
    if (getConstant(axis.extent, value) && value <= 0)
      return failure();
  }

  auto mul = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Mul, lhs,
                                        rhs);
  };
  auto add = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add, lhs,
                                        rhs);
  };
  auto paddedExtent = [&](size_t k) -> Attribute {
    const PlanPad *pad = plan.findPad(static_cast<int64_t>(k));
    return pad ? add(add(old[k].extent, pad->lo), pad->hi) : old[k].extent;
  };
  Attribute one = sym::ConstantExprAttr::get(ctx, 1);

  SmallVector<PlanAxis> newAxes;
  SmallVector<DivisibilityAttr> emitted;
  SmallVector<int64_t> newIndexOfOld(old.size(), -1);
  size_t i = 0, j = 0;
  while (i < old.size() && j < targetShape.size()) {
    size_t iEnd = i + 1, jEnd = j + 1;
    Attribute oldProd = paddedExtent(i);
    Attribute newProd = targetShape[j];
    bool needsDivisibility = false;
    int64_t divisor = 0;

    // Grow the group until the products provably match or the split rule
    // validates; bail when no growth is possible.
    while (proveEqual(oldProd, newProd) != Proof::Proven) {
      if (iEnd == i + 1 &&
          matchesSplit(paddedExtent(i), targetShape.slice(j, jEnd - j), ctx,
                       needsDivisibility, divisor))
        break;
      int64_t oldValue, newValue;
      if (getConstant(oldProd, oldValue) && getConstant(newProd, newValue)) {
        if (oldValue < newValue) {
          if (iEnd == old.size())
            return failure();
          oldProd = mul(oldProd, paddedExtent(iEnd));
          ++iEnd;
        } else {
          if (jEnd == targetShape.size())
            return failure();
          newProd = mul(newProd, targetShape[jEnd]);
          ++jEnd;
        }
      } else if (iEnd == i + 1 && jEnd < targetShape.size()) {
        newProd = mul(newProd, targetShape[jEnd]);
        ++jEnd; // extend the candidate split run
      } else if (iEnd < old.size()) {
        oldProd = mul(oldProd, paddedExtent(iEnd));
        ++iEnd; // extend the candidate merge run
      } else {
        return failure();
      }
    }

    size_t numOld = iEnd - i, numNew = jEnd - j;
    if (needsDivisibility || (numOld == 1 && numNew > 1)) {
      // SPLIT one axis into the target run.
      if (plan.findPad(static_cast<int64_t>(i)))
        return failure(); // splitting a padded axis (design decision 3)
      appendSplitAxes(targetShape.slice(j, numNew), old[i].srcStride, newAxes,
                      ctx);
      if (needsDivisibility) {
        auto constraint = DivisibilityAttr::get(ctx, old[i].extent, divisor);
        if (!llvm::is_contained(plan.divisibility, constraint) &&
            !llvm::is_contained(emitted, constraint))
          emitted.push_back(constraint);
      }
    } else if (numOld == 1 && numNew == 1) {
      newIndexOfOld[i] = static_cast<int64_t>(newAxes.size());
      newAxes.push_back(old[i]); // KEEP (extents proven equal)
    } else {
      // MERGE the old run (contiguity-gated), then split if numNew > 1.
      for (size_t p = i; p < iEnd; ++p)
        if (plan.findPad(static_cast<int64_t>(p)))
          return failure(); // pad folded into a merged axis: inexpressible
      for (size_t p = i; p + 1 < iEnd; ++p)
        if (!contiguousCompatible(old[p], old[p + 1], ctx))
          return failure();
      Attribute mergedStride = old[iEnd - 1].srcStride;
      if (numNew == 1)
        newAxes.push_back({"", targetShape[j], mergedStride});
      else
        appendSplitAxes(targetShape.slice(j, numNew), mergedStride, newAxes,
                        ctx);
    }
    i = iEnd;
    j = jEnd;
  }

  // Absorb trailing unit dims on either side.
  while (i < old.size() && proveEqual(paddedExtent(i), one) == Proof::Proven) {
    if (plan.findPad(static_cast<int64_t>(i)))
      return failure();
    ++i;
  }
  while (j < targetShape.size() &&
         proveEqual(targetShape[j], one) == Proof::Proven) {
    newAxes.push_back({"", targetShape[j], one});
    ++j;
  }
  if (i != old.size() || j != targetShape.size())
    return failure();

  // Commit: the reshaped view becomes the new source view.
  for (size_t k = 0; k < newAxes.size(); ++k)
    newAxes[k].name = ("d" + Twine(k)).str();
  plan.axes = std::move(newAxes);
  plan.perm.clear();
  for (size_t k = 0; k < plan.axes.size(); ++k)
    plan.perm.push_back(static_cast<int64_t>(k));
  plan.divisibility.append(emitted.begin(), emitted.end());
  for (PlanPad &pad : plan.pads) {
    assert(newIndexOfOld[pad.axis] >= 0 &&
           "padded axis must survive as a 1:1 keep");
    pad.axis = newIndexOfOld[pad.axis];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// foldPad (#B3)
//===----------------------------------------------------------------------===//

LogicalResult mlir::reloc::foldPad(PlanBuilder &plan, int64_t axis,
                                   Attribute lo, Attribute hi,
                                   TypedAttr value) {
  MLIRContext *ctx = plan.ctx;
  if (axis < 0 || axis >= static_cast<int64_t>(plan.axes.size()))
    return failure();
  if (!isSymExpr(lo) || !isSymExpr(hi) || !value)
    return failure();
  int64_t width;
  if ((getConstant(lo, width) && width < 0) ||
      (getConstant(hi, width) && width < 0))
    return failure(); // provably negative width (op verifier gap defense)

  // Both widths provably zero: nothing to record.
  int64_t loValue, hiValue;
  if (getConstant(lo, loValue) && loValue == 0 && getConstant(hi, hiValue) &&
      hiValue == 0)
    return success();

  auto add = [&](Attribute lhs, Attribute rhs) {
    return sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add, lhs,
                                        rhs);
  };
  for (PlanPad &pad : plan.pads) {
    if (pad.axis != axis)
      continue;
    if (pad.value != value)
      return failure(); // one fill value per axis in the plan format
    // The new pad wraps the old valid region: widths accumulate.
    pad.lo = add(pad.lo, lo);
    pad.hi = add(pad.hi, hi);
    return success();
  }
  plan.pads.push_back({axis, lo, hi, value});
  return success();
}
