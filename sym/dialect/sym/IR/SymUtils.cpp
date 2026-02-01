//===- SymUtils.cpp - Sym dialect utilities -------------------------------===//
//
// This file implements utility classes for the Sym dialect.
//
//===----------------------------------------------------------------------===//

#include "SymUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::sym;

//===----------------------------------------------------------------------===//
// UnificationSolver
//===----------------------------------------------------------------------===//

UnificationSolver::UnificationSolver(MLIRContext *context, Location loc)
    : context(context), loc(loc.getContext() ? loc : UnknownLoc::get(context)) {
}

bool UnificationSolver::areLogicallyEqual(Attribute a, Attribute b) {
  if (a == b)
    return true;

  // Check ConstantExprAttr equality by value
  if (auto constA = dyn_cast<ConstantExprAttr>(a)) {
    if (auto constB = dyn_cast<ConstantExprAttr>(b)) {
      return constA.getValue() == constB.getValue();
    }
    return false;
  }

  // Check SymbolExprAttr equality by name
  if (auto symA = dyn_cast<SymbolExprAttr>(a)) {
    if (auto symB = dyn_cast<SymbolExprAttr>(b)) {
      return symA.getName() == symB.getName();
    }
    return false;
  }

  // Check BinaryExprAttr - must have same opcode and operands
  // For commutative ops (Add, Mul), also check swapped operands
  if (auto binA = dyn_cast<BinaryExprAttr>(a)) {
    if (auto binB = dyn_cast<BinaryExprAttr>(b)) {
      if (binA.getOpcode() != binB.getOpcode())
        return false;

      // Direct match
      if (areLogicallyEqual(binA.getLhs(), binB.getLhs()) &&
          areLogicallyEqual(binA.getRhs(), binB.getRhs()))
        return true;

      // For commutative ops, check swapped operands
      if (binA.getOpcode() == SymbolicExprOp::Add ||
          binA.getOpcode() == SymbolicExprOp::Mul) {
        return areLogicallyEqual(binA.getLhs(), binB.getRhs()) &&
               areLogicallyEqual(binA.getRhs(), binB.getLhs());
      }
    }
    return false;
  }

  return false;
}

bool UnificationSolver::isConstantValue(Attribute attr, int64_t value) {
  if (auto constAttr = dyn_cast<ConstantExprAttr>(attr))
    return constAttr.getValue() == value;
  return false;
}

bool UnificationSolver::isConstantOne(Attribute attr) {
  return isConstantValue(attr, 1);
}

InFlightDiagnostic UnificationSolver::emitError(const Twine &message) {
  return mlir::emitError(loc, message);
}

std::string UnificationSolver::formatAttr(Attribute attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.print(os);
  return result;
}

FailureOr<SmallVector<Attribute>>
UnificationSolver::unify(ArrayRef<Attribute> shape1,
                         ArrayRef<Attribute> shape2) {
  size_t rank1 = shape1.size();
  size_t rank2 = shape2.size();
  size_t maxRank = std::max(rank1, rank2);

  SmallVector<Attribute> result;
  result.reserve(maxRank);

  // Create the constant 1 attribute for padding
  Attribute one = ConstantExprAttr::get(context, 1);

  // Iterate from the rightmost dimension to the left
  for (size_t i = 0; i < maxRank; ++i) {
    // Compute indices from the right (0 = rightmost)
    // For shape1: if i >= rank1, use 1; else use shape1[rank1 - 1 - i]
    // For shape2: if i >= rank2, use 1; else use shape2[rank2 - 1 - i]

    Attribute d1 = (i < rank1) ? shape1[rank1 - 1 - i] : one;
    Attribute d2 = (i < rank2) ? shape2[rank2 - 1 - i] : one;

    // Apply broadcasting rules
    if (areLogicallyEqual(d1, d2)) {
      // d1 == d2: result is d1
      result.push_back(d1);
    } else if (isConstantOne(d1)) {
      // d1 == 1: result is d2
      result.push_back(d2);
    } else if (isConstantOne(d2)) {
      // d2 == 1: result is d1
      result.push_back(d1);
    } else {
      // Incompatible dimensions
      size_t dimIndex = maxRank - 1 - i;
      return emitError("incompatible dimensions at index ")
             << dimIndex << ": " << formatAttr(d1) << " vs " << formatAttr(d2);
    }
  }

  // Reverse result since we built it from right to left
  std::reverse(result.begin(), result.end());
  return result;
}
