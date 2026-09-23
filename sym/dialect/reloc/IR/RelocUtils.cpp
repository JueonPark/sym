//===- RelocUtils.cpp - Reloc dialect utilities ---------------------------===//
//
// This file implements utilities for the Reloc dialect.
//
//===----------------------------------------------------------------------===//

#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace mlir;
using namespace mlir::reloc;

//===----------------------------------------------------------------------===//
// sym <-> affine expression bridge
//===----------------------------------------------------------------------===//

FailureOr<AffineExpr> mlir::reloc::symToAffine(
    Attribute expr, SmallVectorImpl<StringRef> &symbolNames, MLIRContext *ctx) {
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(expr))
    return getAffineConstantExpr(constant.getValue(), ctx);

  if (auto symbol = dyn_cast<sym::SymbolExprAttr>(expr)) {
    StringRef name = symbol.getName();
    for (auto [index, existing] : llvm::enumerate(symbolNames))
      if (existing == name)
        return getAffineSymbolExpr(index, ctx);
    symbolNames.push_back(name);
    return getAffineSymbolExpr(symbolNames.size() - 1, ctx);
  }

  if (auto binary = dyn_cast<sym::BinaryExprAttr>(expr)) {
    FailureOr<AffineExpr> lhs = symToAffine(binary.getLhs(), symbolNames, ctx);
    if (failed(lhs))
      return failure();
    FailureOr<AffineExpr> rhs = symToAffine(binary.getRhs(), symbolNames, ctx);
    if (failed(rhs))
      return failure();
    switch (binary.getOpcode()) {
    case sym::SymbolicExprOp::Add:
      return *lhs + *rhs;
    case sym::SymbolicExprOp::Sub:
      return *lhs - *rhs; // affine encodes as lhs + rhs * -1
    case sym::SymbolicExprOp::Mul:
      return *lhs * *rhs;
    case sym::SymbolicExprOp::Div:
      return lhs->floorDiv(
          *rhs); // Div is floor division (matches affine floordiv)
    case sym::SymbolicExprOp::Mod:
      return *lhs % *rhs;
    }
    llvm_unreachable("unknown SymbolicExprOp");
  }

  return failure();
}

Attribute mlir::reloc::affineToSym(AffineExpr expr,
                                   ArrayRef<StringRef> symbolNames,
                                   MLIRContext *ctx) {
  using sym::SymbolicExprOp;

  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return sym::ConstantExprAttr::get(ctx, constant.getValue());

  if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    if (symbol.getPosition() >= symbolNames.size())
      return {};
    return sym::SymbolExprAttr::get(ctx, symbolNames[symbol.getPosition()]);
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary)
    return {}; // AffineDimExpr: no sym counterpart in A2.

  // Rebuild subtraction from affine's encodings so `a - b` survives the
  // round trip: `x + (y * -1)` -> `x - y`, and `x + (-c)` -> `x - c`.
  if (binary.getKind() == AffineExprKind::Add) {
    AffineExpr rhs = binary.getRHS();
    if (auto rhsBinary = dyn_cast<AffineBinaryOpExpr>(rhs))
      if (rhsBinary.getKind() == AffineExprKind::Mul)
        if (auto factor = dyn_cast<AffineConstantExpr>(rhsBinary.getRHS()))
          if (factor.getValue() == -1) {
            Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
            Attribute sub = affineToSym(rhsBinary.getLHS(), symbolNames, ctx);
            if (!lhs || !sub)
              return {};
            return sym::getSimplifiedBinaryExpr(ctx, SymbolicExprOp::Sub, lhs,
                                                sub);
          }
    if (auto constant = dyn_cast<AffineConstantExpr>(rhs))
      if (constant.getValue() < 0) {
        Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
        if (!lhs)
          return {};
        return sym::getSimplifiedBinaryExpr(
            ctx, SymbolicExprOp::Sub, lhs,
            sym::ConstantExprAttr::get(ctx, -constant.getValue()));
      }
  }

  SymbolicExprOp opcode;
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    opcode = SymbolicExprOp::Add;
    break;
  case AffineExprKind::Mul:
    opcode = SymbolicExprOp::Mul;
    break;
  case AffineExprKind::Mod:
    opcode = SymbolicExprOp::Mod;
    break;
  case AffineExprKind::FloorDiv:
    opcode = SymbolicExprOp::Div;
    break;
  default:
    return {}; // CeilDiv: no sym counterpart in A2.
  }

  Attribute lhs = affineToSym(binary.getLHS(), symbolNames, ctx);
  Attribute rhs = affineToSym(binary.getRHS(), symbolNames, ctx);
  if (!lhs || !rhs)
    return {};
  return sym::getSimplifiedBinaryExpr(ctx, opcode, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Plan-structure predicates
//===----------------------------------------------------------------------===//

bool mlir::reloc::isContiguousCompatible(AxisInfoAttr outer,
                                         AxisInfoAttr inner) {
  if (!outer || !inner)
    return false;
  MLIRContext *ctx = outer.getContext();
  Attribute product = sym::getSimplifiedBinaryExpr(
      ctx, sym::SymbolicExprOp::Mul, inner.getSrcStride(), inner.getExtent());
  return sym::UnificationSolver::areLogicallyEqual(outer.getSrcStride(),
                                                   product);
}

bool mlir::reloc::isPureView(PlanAttr plan) {
  if (!plan)
    return false;
  if (!plan.getPadFill().empty())
    return false;
  for (AxisInfoAttr axis : plan.getAxes())
    if (!sym::UnificationSolver::areLogicallyEqual(axis.getSrcStride(),
                                                   axis.getDstStride()))
      return false;
  return sym::UnificationSolver::areLogicallyEqual(plan.getSrc().getOffset(),
                                                   plan.getDst().getOffset());
}

bool mlir::reloc::isTypedValueTransformOp(Operation *op) {
  return isa<CastOp, QuantizeOp, DequantizeOp>(op);
}

//===----------------------------------------------------------------------===//
// Typed value transform legality (C1 rules)
//===----------------------------------------------------------------------===//

static StringRef transformName(ValueTransform transform) {
  switch (transform) {
  case ValueTransform::Cast:
    return "reloc.cast";
  case ValueTransform::Quantize:
    return "reloc.quantize";
  case ValueTransform::Dequantize:
    return "reloc.dequantize";
  }
  llvm_unreachable("unknown ValueTransform");
}

LogicalResult mlir::reloc::verifyValueTransformSignature(
    function_ref<InFlightDiagnostic()> emitError, ValueTransform transform,
    NumericPolicy policy, Type from, Type to) {
  switch (transform) {
  case ValueTransform::Cast: {
    std::optional<NumericPolicy> required;
    StringRef pair;
    if (from.isF32() && to.isF16()) {
      required = NumericPolicy::IeeeRne;
      pair = "f32 -> f16";
    } else if (from.isF16() && to.isF32()) {
      required = NumericPolicy::Exact;
      pair = "f16 -> f32";
    }
    if (!required)
      return emitError() << "cast from " << from << " to " << to
                         << " is not a supported typed conversion (f32 -> "
                            "f16, f16 -> f32)";
    if (policy != *required)
      return emitError() << "policy '" << stringifyNumericPolicy(policy)
                         << "' is not defined for the " << pair
                         << " cast (use '" << stringifyNumericPolicy(*required)
                         << "')";
    return success();
  }
  case ValueTransform::Quantize:
    if (!from.isF32() || !to.isSignlessInteger(8))
      return emitError() << "quantize expects an f32 operand and a signless i8 "
                            "result (int8 signedness is declared by the "
                            "operation, not by the storage type), but got "
                         << from << " -> " << to;
    if (policy != NumericPolicy::SymmetricRne)
      return emitError() << "policy '" << stringifyNumericPolicy(policy)
                         << "' is not defined for " << transformName(transform)
                         << " (use 'symmetric_rne')";
    return success();
  case ValueTransform::Dequantize:
    if (!from.isSignlessInteger(8) || !to.isF32())
      return emitError() << "dequantize expects a signless i8 operand and an "
                            "f32 result, but got "
                         << from << " -> " << to;
    if (policy != NumericPolicy::Affine)
      return emitError() << "policy '" << stringifyNumericPolicy(policy)
                         << "' is not defined for " << transformName(transform)
                         << " (use 'affine')";
    return success();
  }
  llvm_unreachable("unknown ValueTransform");
}

/// Constant extents print as plain integers in diagnostics; anything else
/// prints as the attribute.
static std::string describeExtent(Attribute extent) {
  std::string text;
  llvm::raw_string_ostream os(text);
  if (auto constant = dyn_cast<sym::ConstantExprAttr>(extent))
    os << constant.getValue();
  else
    os << extent;
  return text;
}

namespace {
enum class ParamRole { Scale, ZeroPoint };
} // namespace

static StringRef roleName(ParamRole role) {
  return role == ParamRole::Scale ? "scale" : "zero_point";
}

/// One quantization parameter: a dense constant or a #reloc.binding, rank 0
/// (per tensor) or rank 1 (per channel, length == the channel axis extent
/// unless undecidable), role-specific element type, valid constant values.
static LogicalResult
verifyQuantParam(function_ref<InFlightDiagnostic()> emitError, ParamRole role,
                 Attribute attr, ArrayRef<Attribute> shape,
                 std::optional<int64_t> axis) {
  StringRef name = roleName(role);
  MLIRContext *ctx = attr.getContext();
  int64_t rank = 0;
  Attribute channels; // rank-1 length as a sym expression
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    ShapedType type = dense.getType();
    rank = type.getRank();
    if (rank > 1)
      return emitError() << "parameters must have rank 0 or 1, but " << name
                         << " has rank " << rank;
    Type element = type.getElementType();
    if (role == ParamRole::Scale) {
      if (!element.isF32())
        return emitError()
               << "scale constants must have element type f32, but got "
               << element;
      for (auto [index, value] : llvm::enumerate(dense.getValues<APFloat>()))
        if (!value.isFinite() || value.isNegative() || value.isZero())
          return emitError()
                 << "scale must be finite and strictly positive, but element "
                 << index << " is " << FloatAttr::get(element, value);
    } else {
      if (!element.isSignlessInteger())
        return emitError() << "zero_point constants must have a signless "
                              "integer element type, but got "
                           << element;
      for (auto [index, value] : llvm::enumerate(dense.getValues<APInt>())) {
        int64_t zeroPoint = value.getSExtValue();
        if (zeroPoint < -128 || zeroPoint > 127)
          return emitError()
                 << "zero point must lie in [-128, 127], but element " << index
                 << " is " << zeroPoint;
      }
    }
    if (rank == 1)
      channels = sym::ConstantExprAttr::get(ctx, type.getDimSize(0));
  } else if (auto binding = dyn_cast<ParamBindingAttr>(attr)) {
    rank = static_cast<int64_t>(binding.getExtents().size());
    Type element = binding.getElementType();
    if (role == ParamRole::Scale && !element.isF32())
      return emitError()
             << "scale bindings must declare element type f32, but got "
             << element;
    if (role == ParamRole::ZeroPoint && !element.isSignlessInteger(32))
      return emitError()
             << "zero_point bindings must declare element type i32, but got "
             << element;
    if (rank == 1)
      channels = binding.getExtents()[0];
  } else {
    return emitError() << name
                       << " must be a dense constant or a #reloc.binding, but "
                          "got "
                       << attr;
  }

  if (!axis) {
    if (rank != 0)
      return emitError() << "per-tensor form (no axis) requires rank-0 "
                            "parameters, but "
                         << name << " has rank " << rank;
    return success();
  }
  if (role == ParamRole::Scale && rank != 1)
    return emitError()
           << "per-channel form requires a rank-1 scale, but scale has rank "
           << rank;
  if (rank == 1) {
    Attribute extent = shape[*axis];
    if (proveEqual(channels, extent) == Proof::Disproven)
      return emitError() << name
                         << (isa<DenseElementsAttr>(attr) ? " has "
                                                          : " declares ")
                         << describeExtent(channels)
                         << " channel entries, but axis " << *axis
                         << " has extent " << describeExtent(extent);
  }
  return success();
}

LogicalResult mlir::reloc::verifyQuantizationParameters(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<Attribute> shape,
    Attribute scale, Attribute zeroPoint, std::optional<int64_t> axis,
    NumericPolicy policy) {
  int64_t rank = static_cast<int64_t>(shape.size());
  if (axis && (*axis < 0 || *axis >= rank))
    return emitError() << "axis (" << *axis
                       << ") is out of range for operand rank " << rank;
  if (failed(verifyQuantParam(emitError, ParamRole::Scale, scale, shape, axis)))
    return failure();
  if (zeroPoint && failed(verifyQuantParam(emitError, ParamRole::ZeroPoint,
                                           zeroPoint, shape, axis)))
    return failure();
  auto scaleBinding = dyn_cast<ParamBindingAttr>(scale);
  auto zeroPointBinding = dyn_cast_or_null<ParamBindingAttr>(zeroPoint);
  if (scaleBinding && zeroPointBinding &&
      scaleBinding.getName() == zeroPointBinding.getName())
    return emitError() << "runtime parameters must use distinct binding "
                          "names, but scale and zero_point both bind \""
                       << scaleBinding.getName() << "\"";
  // symmetric_rne has no zero point: only the constant 0 is admitted, so the
  // runtime never needs a bind-time zero-point guard for this policy.
  if (policy == NumericPolicy::SymmetricRne && zeroPoint) {
    if (isa<ParamBindingAttr>(zeroPoint))
      return emitError() << "policy symmetric_rne admits only the constant "
                            "zero point 0, but zero_point is a runtime binding";
    for (auto [index, value] :
         llvm::enumerate(cast<DenseElementsAttr>(zeroPoint).getValues<APInt>()))
      if (!value.isZero())
        return emitError() << "policy symmetric_rne admits only the constant "
                              "zero point 0, but element "
                           << index << " is " << value.getSExtValue();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Fill folding through value stages (C2)
//===----------------------------------------------------------------------===//

/// The rank-0 dense constant of a per-tensor parameter, or null for a
/// per-channel constant or a runtime binding.
static DenseElementsAttr perTensorConstant(Attribute param) {
  auto dense = dyn_cast_or_null<DenseElementsAttr>(param);
  if (!dense || dense.getType().getRank() != 0)
    return {};
  return dense;
}

TypedAttr mlir::reloc::foldFillThroughStage(TypedAttr fill,
                                            ValueStageAttr stage) {
  if (!fill || !stage || fill.getType() != stage.getInputType())
    return {};
  const llvm::RoundingMode rne = APFloat::rmNearestTiesToEven;
  switch (stage.getTransform()) {
  case ValueTransform::Cast: {
    auto value = dyn_cast<FloatAttr>(fill);
    auto outType = dyn_cast<FloatType>(stage.getOutputType());
    if (!value || !outType)
      return {};
    APFloat converted = value.getValue();
    bool losesInfo = false;
    converted.convert(outType.getFloatSemantics(), rne, &losesInfo);
    return FloatAttr::get(outType, converted);
  }
  case ValueTransform::Quantize: {
    auto value = dyn_cast<FloatAttr>(fill);
    DenseElementsAttr scale = perTensorConstant(stage.getScale());
    if (!value || !scale || stage.getAxis() >= 0)
      return {};
    // docs/reloc-typed-semantics.md §3.3: inv = fl32(1/scale), t = fl32(x*inv),
    // clamp max-then-min (NaN -> -128), round to nearest even.
    APFloat inv(APFloat::IEEEsingle(), 1);
    inv.divide(scale.getSplatValue<APFloat>(), rne);
    APFloat t = value.getValue();
    t.multiply(inv, rne);
    int64_t q;
    if (t.isNaN()) {
      q = -128;
    } else {
      APFloat lo(APFloat::IEEEsingle(), "-128.0");
      APFloat hi(APFloat::IEEEsingle(), "127.0");
      if (t.compare(lo) == APFloat::cmpLessThan)
        t = lo;
      else if (t.compare(hi) == APFloat::cmpGreaterThan)
        t = hi;
      t.roundToIntegral(rne);
      q = static_cast<int64_t>(t.convertToFloat());
    }
    return IntegerAttr::get(stage.getOutputType(), q);
  }
  case ValueTransform::Dequantize: {
    auto value = dyn_cast<IntegerAttr>(fill);
    DenseElementsAttr scale = perTensorConstant(stage.getScale());
    auto outType = dyn_cast<FloatType>(stage.getOutputType());
    if (!value || !scale || !outType || stage.getAxis() >= 0)
      return {};
    int64_t zeroPoint = 0;
    if (Attribute zp = stage.getZeroPoint()) {
      DenseElementsAttr constant = perTensorConstant(zp);
      if (!constant)
        return {};
      zeroPoint = constant.getSplatValue<APInt>().getSExtValue();
    }
    // §3.4: d = q - zp exactly, y = fl32(d * scale).
    APFloat y(static_cast<float>(value.getValue().getSExtValue() - zeroPoint));
    y.multiply(scale.getSplatValue<APFloat>(), rne);
    return FloatAttr::get(outType, y);
  }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Verification proofs
//===----------------------------------------------------------------------===//

StringRef mlir::reloc::stringifyProof(Proof proof) {
  switch (proof) {
  case Proof::Proven:
    return "Proven";
  case Proof::Disproven:
    return "Disproven";
  case Proof::Unknown:
    return "Unknown";
  }
  llvm_unreachable("unknown Proof");
}

Proof mlir::reloc::proveEqual(Attribute lhs, Attribute rhs) {
  if (!lhs || !rhs)
    return Proof::Unknown;
  if (sym::UnificationSolver::areLogicallyEqual(lhs, rhs))
    return Proof::Proven;
  auto constLhs = dyn_cast<sym::ConstantExprAttr>(lhs);
  auto constRhs = dyn_cast<sym::ConstantExprAttr>(rhs);
  if (constLhs && constRhs)
    return constLhs.getValue() == constRhs.getValue() ? Proof::Proven
                                                      : Proof::Disproven;
  return Proof::Unknown;
}

Proof mlir::reloc::proveLessEqual(Attribute lhs, Attribute rhs) {
  if (!lhs || !rhs)
    return Proof::Unknown;
  auto constLhs = dyn_cast<sym::ConstantExprAttr>(lhs);
  auto constRhs = dyn_cast<sym::ConstantExprAttr>(rhs);
  if (constLhs && constRhs)
    return constLhs.getValue() <= constRhs.getValue() ? Proof::Proven
                                                      : Proof::Disproven;
  if (sym::UnificationSolver::areLogicallyEqual(lhs, rhs))
    return Proof::Proven; // x <= x
  return Proof::Unknown;
}

SmallVector<Attribute>
mlir::reloc::canonicalRowMajorStrides(ArrayRef<Attribute> extents,
                                      MLIRContext *ctx) {
  SmallVector<Attribute> strides(extents.size());
  Attribute running = sym::ConstantExprAttr::get(ctx, 1);
  for (int64_t k = static_cast<int64_t>(extents.size()) - 1; k >= 0; --k) {
    strides[k] = running;
    running = sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Mul,
                                           running, extents[k]);
  }
  return strides;
}

/// Combine proofs of conjoined relations: any Disproven disproves the
/// conjunction; otherwise any Unknown makes it Unknown.
static Proof combineProofs(ArrayRef<Proof> proofs) {
  Proof result = Proof::Proven;
  for (Proof proof : proofs) {
    if (proof == Proof::Disproven)
      return Proof::Disproven;
    if (proof == Proof::Unknown)
      result = Proof::Unknown;
  }
  return result;
}

Proof mlir::reloc::provePadRange(PadFillAttr pad, ArrayRef<AxisInfoAttr> axes,
                                 TensorDescAttr dst) {
  if (!pad || !dst)
    return Proof::Unknown;
  int64_t axis = pad.getDstAxis();
  if (axis < 0 || axis >= static_cast<int64_t>(axes.size()) ||
      axis >= static_cast<int64_t>(dst.getExtents().size()))
    return Proof::Disproven;
  MLIRContext *ctx = pad.getContext();
  Attribute zero = sym::ConstantExprAttr::get(ctx, 0);
  Attribute sum = sym::getSimplifiedBinaryExpr(
      ctx, sym::SymbolicExprOp::Add,
      sym::getSimplifiedBinaryExpr(ctx, sym::SymbolicExprOp::Add,
                                   axes[axis].getExtent(), pad.getLo()),
      pad.getHi());
  return combineProofs({proveLessEqual(zero, pad.getLo()),
                        proveLessEqual(zero, pad.getHi()),
                        proveEqual(sum, dst.getExtents()[axis])});
}
