//===- PlanBuilder.cpp - In-progress reloc plan state ---------------------===//
//
// This file implements the P1b folding state and its transfer functions.
//
//===----------------------------------------------------------------------===//

#include "PlanBuilder.h"
#include "RelocUtils.h"
#include "SymUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::reloc;

PlanBuilder::PlanBuilder(sym::SymbolicTensorType input)
    : ctx(input.getContext()), elementType(input.getElementType()) {
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
  // The dst element type is the CURRENT stage dtype: identical to the source
  // for layout-only chains, the last value stage's output otherwise (C2).
  auto dst =
      TensorDescAttr::get(ctx, dstExtents,
                          /*strides=*/ArrayRef<Attribute>(), zero, elementType);
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

TypedPlanAttr PlanBuilder::finalizeTyped(Location loc) const {
  if (stages.empty())
    return {};
  PlanAttr layout = finalize(loc);
  if (!layout)
    return {};
  // The logical result is the layout dst as finalized here (full logical
  // rank); canonicalization may later merge the LAYOUT's axes but never
  // touches this descriptor or the channel maps written over it.
  TensorDescAttr result = layout.getDst();
  size_t rank = axes.size();
  SmallVector<Attribute> symbolAttrs;
  for (StringRef name : symbolNames)
    symbolAttrs.push_back(StringAttr::get(ctx, name));
  SmallVector<ValueStageAttr> stageAttrs;
  for (const PlanValueStage &stage : stages) {
    AffineMap channel;
    if (stage.channel)
      channel = AffineMap::get(
          rank, symbolNames.size(),
          simplifyAffineExpr(stage.channel, rank, symbolNames.size()), ctx);
    // Exact argument types keep the concrete getChecked overload (the
    // generic Base::getChecked needs the storage class, complete only in
    // RelocAttributes.cpp).
    ValueStageAttr attr = ValueStageAttr::getChecked(
        [&]() { return emitError(loc); }, ctx, stage.transform, stage.policy,
        stage.inputType, stage.outputType, ArrayRef<Attribute>(stage.shape),
        stage.scale, stage.zeroPoint, stage.axis, channel);
    if (!attr)
      return {};
    stageAttrs.push_back(attr);
  }
  SmallVector<TypedFillAttr> fillAttrs;
  for (const PlanPad &pad : pads)
    fillAttrs.push_back(
        TypedFillAttr::get(ctx, pad.axis, pad.stage, pad.original));
  return TypedPlanAttr::getChecked([&]() { return emitError(loc); }, ctx, src,
                                   result, ArrayAttr::get(ctx, symbolAttrs),
                                   layout, ArrayRef<ValueStageAttr>(stageAttrs),
                                   ArrayRef<TypedFillAttr>(fillAttrs));
}

/// True when any folded stage carries a channel expression (C2): only then
/// do layout folds need to rewrite coordinates.
static bool tracksChannels(const PlanBuilder &plan) {
  return llvm::any_of(plan.stages, [](const PlanValueStage &stage) {
    return static_cast<bool>(stage.channel);
  });
}

/// Rewrite every channel expression through `dimReplacements` (old dim k ->
/// dimReplacements[k]).
static void replaceChannelDims(PlanBuilder &plan,
                               ArrayRef<AffineExpr> dimReplacements) {
  for (PlanValueStage &stage : plan.stages)
    if (stage.channel)
      stage.channel = stage.channel.replaceDims(dimReplacements);
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
  // C2: channel expressions follow their coordinates through the permutation.
  if (tracksChannels(plan)) {
    SmallVector<AffineExpr> dims(rank);
    for (int64_t k = 0; k < rank; ++k)
      dims[k] = getAffineDimExpr(newIndexOfOld[k], plan.ctx);
    replaceChannelDims(plan, dims);
  }
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

  // C2: channel expressions are written over the current dims; a reshape
  // rewrites each referenced old dim as an affine expression of the new
  // dims (KEEP: the new dim; SPLIT: the row-major recombination of the new
  // run; MERGE: floordiv/mod of the merged coordinate). Only referenced dims
  // are converted so the plan's symbol table stays minimal; extents become
  // affine symbols through the builder's symbol list, committed on success.
  const bool trackChannels = tracksChannels(plan);
  SmallVector<bool> referenced(old.size(), false);
  if (trackChannels)
    for (const PlanValueStage &stage : plan.stages)
      if (stage.channel)
        stage.channel.walk([&](AffineExpr expr) {
          if (auto dim = dyn_cast<AffineDimExpr>(expr))
            referenced[dim.getPosition()] = true;
        });
  SmallVector<StringRef> symbols(plan.symbolNames);
  SmallVector<AffineExpr> oldDimExprs(old.size(),
                                      getAffineConstantExpr(0, ctx));
  auto affine = [&](Attribute expr) -> AffineExpr {
    FailureOr<AffineExpr> converted = symToAffine(expr, symbols, ctx);
    return succeeded(converted) ? *converted : AffineExpr();
  };
  // Row-major recombination of a run of new dims [first, first + n) with
  // extents `targets`: sum_t d_(first+t) * prod_(l>t) targets[l].
  auto recombine = [&](ArrayRef<Attribute> targets,
                       size_t first) -> AffineExpr {
    AffineExpr sum = getAffineConstantExpr(0, ctx);
    AffineExpr stride = getAffineConstantExpr(1, ctx);
    for (int64_t t = static_cast<int64_t>(targets.size()) - 1; t >= 0; --t) {
      sum = sum + getAffineDimExpr(first + t, ctx) * stride;
      if (t == 0)
        break; // the outermost extent is never a stride: keep symbols minimal
      AffineExpr extent = affine(targets[t]);
      if (!extent)
        return {};
      stride = stride * extent;
    }
    return sum;
  };

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
    const size_t firstNew = newAxes.size();
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
      if (trackChannels && referenced[i]) {
        // [B, 12] -> [B, 4, 3]: original channel = 3 * c_outer + c_inner.
        oldDimExprs[i] = recombine(targetShape.slice(j, numNew), firstNew);
        if (!oldDimExprs[i])
          return failure();
      }
    } else if (numOld == 1 && numNew == 1) {
      newIndexOfOld[i] = static_cast<int64_t>(newAxes.size());
      newAxes.push_back(old[i]); // KEEP (extents proven equal)
      oldDimExprs[i] = getAffineDimExpr(firstNew, ctx);
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
      if (trackChannels) {
        // [B, C] -> [B * C]: original channel = flat_index mod C; an outer
        // merged axis is flat_index floordiv (product of the inner extents).
        AffineExpr merged =
            numNew == 1 ? getAffineDimExpr(firstNew, ctx)
                        : recombine(targetShape.slice(j, numNew), firstNew);
        if (!merged)
          return failure();
        for (size_t p = i; p < iEnd; ++p) {
          if (!referenced[p])
            continue;
          AffineExpr inner = getAffineConstantExpr(1, ctx);
          for (size_t q = p + 1; q < iEnd; ++q) {
            AffineExpr extent = affine(old[q].extent);
            if (!extent)
              return failure();
            inner = inner * extent;
          }
          AffineExpr coordinate = merged.floorDiv(inner);
          if (p != i) {
            AffineExpr extent = affine(old[p].extent);
            if (!extent)
              return failure();
            coordinate = coordinate % extent;
          }
          oldDimExprs[p] = coordinate;
        }
      }
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

  // Every padded axis must have survived as a 1:1 keep (guarded above);
  // check before mutating so a future guard gap bails instead of
  // committing a wrong plan in release builds.
  for (const PlanPad &pad : plan.pads)
    if (newIndexOfOld[pad.axis] < 0)
      return failure();

  // Commit: the reshaped view becomes the new source view.
  for (size_t k = 0; k < newAxes.size(); ++k)
    newAxes[k].name = ("d" + Twine(k)).str();
  plan.axes = std::move(newAxes);
  plan.perm.clear();
  for (size_t k = 0; k < plan.axes.size(); ++k)
    plan.perm.push_back(static_cast<int64_t>(k));
  plan.divisibility.append(emitted.begin(), emitted.end());
  for (PlanPad &pad : plan.pads)
    pad.axis = newIndexOfOld[pad.axis];
  if (trackChannels) {
    replaceChannelDims(plan, oldDimExprs);
    plan.symbolNames = std::move(symbols);
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
  // C2: channel expressions of already-folded stages are written over the
  // current dst coordinates; a new leading width moves the padded axis, so
  // the stage's coordinate along it becomes d_axis - lo.
  SmallVector<StringRef> symbols(plan.symbolNames);
  AffineExpr loAffine;
  if (tracksChannels(plan) && !(getConstant(lo, loValue) && loValue == 0)) {
    FailureOr<AffineExpr> converted = symToAffine(lo, symbols, ctx);
    if (failed(converted))
      return failure();
    loAffine = *converted;
  }
  auto shiftChannels = [&]() {
    if (!loAffine)
      return;
    SmallVector<AffineExpr> dims;
    for (size_t k = 0; k < plan.axes.size(); ++k)
      dims.push_back(getAffineDimExpr(k, ctx));
    dims[axis] = dims[axis] - loAffine;
    replaceChannelDims(plan, dims);
    plan.symbolNames = std::move(symbols);
  };
  const int64_t entryStage = static_cast<int64_t>(plan.stages.size());
  for (PlanPad &pad : plan.pads) {
    if (pad.axis != axis)
      continue;
    if (pad.value != value)
      return failure(); // one fill value per axis in the plan format
    if (pad.stage != entryStage) {
      // Two pads of one axis entering at different value stages have two
      // entry points; the plan format keeps one per axis, so keep the
      // original chain instead of merging them.
      plan.bailReason = "pad_stage";
      return failure();
    }
    // The new pad wraps the old valid region: widths accumulate.
    pad.lo = add(pad.lo, lo);
    pad.hi = add(pad.hi, hi);
    shiftChannels();
    return success();
  }
  plan.pads.push_back({axis, lo, hi, value, /*original=*/value, entryStage});
  shiftChannels();
  return success();
}

//===----------------------------------------------------------------------===//
// foldValueStage (C2)
//===----------------------------------------------------------------------===//

LogicalResult mlir::reloc::foldValueStage(
    PlanBuilder &plan, ValueTransform transform, NumericPolicy policy,
    Type inputType, Type outputType, ArrayRef<Attribute> shape, Attribute scale,
    Attribute zeroPoint, std::optional<int64_t> axis) {
  MLIRContext *ctx = plan.ctx;
  plan.bailReason.clear();
  if (inputType != plan.elementType || shape.size() != plan.axes.size()) {
    plan.bailReason = "type_chain";
    return failure();
  }
  int64_t rank = static_cast<int64_t>(plan.axes.size());
  if (axis && (*axis < 0 || *axis >= rank)) {
    plan.bailReason = "channel_axis";
    return failure();
  }
  // Every pending pad's fused fill must pass through this stage as ONE
  // value: casts always fold, quantize/dequantize only with per-tensor
  // constant parameters. A per-channel stage would give every padded
  // position its own channel's code, so the fill program is absent.
  SmallVector<TypedAttr> fused;
  if (!plan.pads.empty()) {
    if (axis) {
      plan.bailReason = "fill_not_foldable";
      return failure();
    }
    // A per-tensor stage attribute (no channel map) is verifier-valid, so
    // the C1 reference arithmetic can be reused directly.
    ValueStageAttr stageAttr =
        ValueStageAttr::get(ctx, transform, policy, inputType, outputType,
                            shape, scale, zeroPoint, -1, AffineMap());
    for (const PlanPad &pad : plan.pads) {
      TypedAttr folded = foldFillThroughStage(pad.value, stageAttr);
      if (!folded) {
        plan.bailReason = "fill_not_foldable";
        return failure();
      }
      fused.push_back(folded);
    }
  }
  // Commit.
  for (auto [pad, value] : llvm::zip(plan.pads, fused))
    pad.value = value;
  PlanValueStage stage;
  stage.transform = transform;
  stage.policy = policy;
  stage.inputType = inputType;
  stage.outputType = outputType;
  stage.shape.assign(shape.begin(), shape.end());
  stage.scale = scale;
  stage.zeroPoint = zeroPoint;
  stage.axis = axis.value_or(-1);
  if (axis)
    stage.channel = getAffineDimExpr(*axis, ctx);
  plan.stages.push_back(std::move(stage));
  plan.elementType = outputType;
  return success();
}

//===----------------------------------------------------------------------===//
// Chain-op dispatch (#B4)
//===----------------------------------------------------------------------===//

bool mlir::reloc::isFoldableChainOp(Operation *op) {
  return isa<TransposeOp, ReshapeOp, PadOp>(op) || isTypedValueTransformOp(op);
}

/// Element type and logical shape of a typed op's operand/result.
template <typename OpType>
static LogicalResult
foldTypedOp(PlanBuilder &plan, OpType op, ValueTransform transform,
            Attribute scale, Attribute zeroPoint, std::optional<int64_t> axis) {
  auto input = cast<sym::SymbolicTensorType>(op.getInput().getType());
  auto result = cast<sym::SymbolicTensorType>(op.getResult().getType());
  return foldValueStage(plan, transform, op.getPolicy(), input.getElementType(),
                        result.getElementType(), input.getShape(), scale,
                        zeroPoint, axis);
}

LogicalResult mlir::reloc::foldChainOp(PlanBuilder &plan, Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](TransposeOp transpose) {
        return foldTranspose(plan, transpose.getPerm());
      })
      .Case([&](ReshapeOp reshape) {
        return foldReshape(plan, reshape.getTargetShape().getValue());
      })
      .Case([&](PadOp pad) {
        return foldPad(plan, pad.getAxis(), pad.getLo(), pad.getHi(),
                       pad.getValue());
      })
      .Case([&](CastOp cast) {
        return foldTypedOp(plan, cast, ValueTransform::Cast, Attribute(),
                           Attribute(), std::nullopt);
      })
      .Case([&](QuantizeOp quantize) {
        return foldTypedOp(plan, quantize, ValueTransform::Quantize,
                           quantize.getScale(), quantize.getZeroPointAttr(),
                           quantize.getAxis());
      })
      .Case([&](DequantizeOp dequantize) {
        return foldTypedOp(plan, dequantize, ValueTransform::Dequantize,
                           dequantize.getScale(), dequantize.getZeroPointAttr(),
                           dequantize.getAxis());
      })
      .Default([](Operation *) { return failure(); });
}
