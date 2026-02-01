//===- SymUtils.h - Sym dialect utilities ---------------------------------===//
//
// This file declares utility classes for the Sym dialect, including the
// UnificationSolver for symbolic shape broadcasting.
//
//===----------------------------------------------------------------------===//

#ifndef SYM_UTILS_H
#define SYM_UTILS_H

#include "SymDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sym {

//===----------------------------------------------------------------------===//
// UnificationSolver
//===----------------------------------------------------------------------===//

/// UnificationSolver resolves symbolic shape broadcasting.
///
/// This class takes two symbolic shapes (represented as ArrayRef<Attribute>)
/// and determines if they are compatible under NumPy-style broadcasting rules.
/// If compatible, it returns the unified/broadcast shape. If not, it returns
/// failure with detailed diagnostics.
///
/// Broadcasting rules:
/// 1. Shapes are aligned from the right (trailing dimensions).
/// 2. Missing dimensions are treated as 1.
/// 3. For each dimension pair (d1, d2):
///    - If d1 == d2 (logically equal): result is d1
///    - If d1 == 1: result is d2
///    - If d2 == 1: result is d1
///    - Otherwise: incompatible (failure)
///
/// Example:
///   shape1 = [#sym.symbol<"batch">, #sym.constant<3>]
///   shape2 = [#sym.constant<1>, #sym.constant<3>]
///   result = [#sym.symbol<"batch">, #sym.constant<3>]
///
class UnificationSolver {
public:
  /// Construct a solver with the given context and optional location for
  /// diagnostics.
  explicit UnificationSolver(MLIRContext *context,
                             Location loc = UnknownLoc::get(nullptr));

  /// Unify two symbolic shapes using broadcasting rules.
  /// Returns the broadcast shape on success, or emits an error diagnostic
  /// and returns failure if the shapes are incompatible.
  FailureOr<SmallVector<Attribute>> unify(ArrayRef<Attribute> shape1,
                                          ArrayRef<Attribute> shape2);

  /// Check if two symbolic dimension attributes are logically equal.
  /// This handles:
  /// - Direct attribute equality
  /// - ConstantExprAttr equality by value
  /// - SymbolExprAttr equality by name
  /// - BinaryExprAttr equality with commutativity for Add/Mul
  static bool areLogicallyEqual(Attribute a, Attribute b);

  /// Check if an attribute represents the constant value 1.
  static bool isConstantOne(Attribute attr);

  /// Check if an attribute is a constant with a specific value.
  static bool isConstantValue(Attribute attr, int64_t value);

private:
  /// Emit an error diagnostic with the given message.
  InFlightDiagnostic emitError(const Twine &message);

  /// Format an attribute for diagnostic output.
  static std::string formatAttr(Attribute attr);

  MLIRContext *context;
  Location loc;
};

} // namespace sym
} // namespace mlir

#endif // SYM_UTILS_H
