//===- RelocUtils.h - Reloc dialect utilities -------------------*- C++ -*-===//
//
// This file declares utilities for the Reloc dialect: the compact symbolic
// expression syntax shared by all reloc attributes, the sym<->affine
// expression bridge, and plan-structure predicates.
// Conventions are documented in docs/reloc-design.md.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_UTILS_H
#define RELOC_UTILS_H

#include "RelocDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace reloc {

//===----------------------------------------------------------------------===//
// Compact expression syntax
//===----------------------------------------------------------------------===//
//
// Grammar (left-associative; * floordiv mod bind tighter than + -):
//   expr   := term  (('+' | '-') term)*
//   term   := factor (('*' | 'floordiv' | 'div' | 'mod') factor)*
//   factor := INTEGER | IDENT | STRING | '(' expr ')'
//
// Parsing builds through sym::getSimplifiedBinaryExpr, so expressions
// simplify at parse time exactly like #sym.binary.

/// Parse a compact symbolic expression. Returns the parsed sym expression
/// attribute, or null after emitting a located error.
Attribute parseSymExpr(AsmParser &parser);

/// Print a sym expression attribute in the compact syntax with minimal
/// parentheses. `expr` must satisfy isSymExpr().
void printSymExpr(AsmPrinter &printer, Attribute expr);

/// True if `attr` is a sym expression attribute (SymbolExprAttr,
/// ConstantExprAttr, or BinaryExprAttr).
bool isSymExpr(Attribute attr);

} // namespace reloc
} // namespace mlir

#endif // RELOC_UTILS_H
