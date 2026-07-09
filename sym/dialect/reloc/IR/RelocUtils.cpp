//===- RelocUtils.cpp - Reloc dialect utilities ---------------------------===//
//
// This file implements utilities for the Reloc dialect.
//
//===----------------------------------------------------------------------===//

#include "RelocUtils.h"
#include "SymDialect.h"
#include "SymUtils.h"

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
