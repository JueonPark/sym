//===- RelocUtils.h - Reloc dialect utilities -------------------*- C++ -*-===//
//
// This file declares utilities for the Reloc dialect: the compact symbolic
// expression syntax shared by all reloc attributes, the sym<->affine
// expression bridge, and plan-structure predicates.
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

//===----------------------------------------------------------------------===//
// sym <-> affine expression bridge
//===----------------------------------------------------------------------===//
//
// Sym symbols are named; affine symbols are positional. symToAffine assigns
// positions in order of first appearance, recording names in `symbolNames`
// (an already-recorded name reuses its position, so one call chain shares a
// binding across expressions). affineToSym maps positions back through the
// same list. Lossless for {symbol, constant, +, -, *, floordiv, mod}; Sub is
// encoded as lhs + rhs * -1 on the affine side; round-trip equality is judged
// modulo Add/Mul commutativity.

/// Convert a sym expression attribute to an AffineExpr over symbols.
/// Returns failure for attributes that are not sym expressions.
FailureOr<AffineExpr> symToAffine(Attribute expr,
                                  SmallVectorImpl<StringRef> &symbolNames,
                                  MLIRContext *ctx);

/// Convert an AffineExpr over symbols back to a sym expression attribute.
/// Returns null for expressions with no sym counterpart (ceildiv, dims,
/// out-of-range symbol positions).
Attribute affineToSym(AffineExpr expr, ArrayRef<StringRef> symbolNames,
                      MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Plan-structure predicates
//===----------------------------------------------------------------------===//
//
// Both predicates are sound but incomplete: `true` means provably so under
// sym simplification; `false` means "not proven".

/// True iff `outer.src_stride == inner.src_stride * inner.extent` is provable
/// via sym simplification — i.e. the two source axes can be treated as one
/// contiguous axis.
bool isContiguousCompatible(AxisInfoAttr outer, AxisInfoAttr inner);

/// True iff the plan provably performs no data movement: no pad_fill entries,
/// every axis has dst_stride == src_stride, and src/dst offsets are equal.
bool isPureView(PlanAttr plan);

//===----------------------------------------------------------------------===//
// Verification proofs
//===----------------------------------------------------------------------===//
//
// Three-valued proofs over sym expressions: Proven / Disproven answer
// definitively; Unknown means "not decidable with the current prover".
// The verifier rejects only on Disproven; Unknown never rejects.

enum class Proof { Proven, Disproven, Unknown };

/// Human-readable proof name ("Proven" / "Disproven" / "Unknown").
StringRef stringifyProof(Proof proof);

/// Prove or disprove `lhs == rhs`. Constants compare numerically;
/// logically-equal expressions are Proven; anything else is Unknown.
Proof proveEqual(Attribute lhs, Attribute rhs);

/// Prove or disprove `lhs <= rhs`. Constants compare numerically;
/// logically-equal expressions are Proven; anything else is Unknown.
Proof proveLessEqual(Attribute lhs, Attribute rhs);

/// Canonical row-major strides over `extents`: stride[rank-1] = 1,
/// stride[k] = stride[k+1] * extent[k+1], built with parse-style
/// simplification.
SmallVector<Attribute> canonicalRowMajorStrides(ArrayRef<Attribute> extents,
                                                MLIRContext *ctx);

/// Combined pad-widths proof for one pad_fill entry, with `a = dst_axis`:
///   0 <= lo,  0 <= hi,  axes[a].extent + lo + hi == dst.extents[a]
/// Returns Disproven if any relation is disproven or `a` is out of range
/// for `axes`/`dst`; Unknown if none is disproven but any is unknown;
/// Proven otherwise.
Proof provePadRange(PadFillAttr pad, ArrayRef<AxisInfoAttr> axes,
                    TensorDescAttr dst);

} // namespace reloc
} // namespace mlir

#endif // RELOC_UTILS_H
