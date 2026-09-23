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
#include "SymUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace reloc {

//===----------------------------------------------------------------------===//
// Compact expression syntax (moved to sym, issue #31)
//===----------------------------------------------------------------------===//
//
// The grammar is shared with the !sym.tensor type; see SymUtils.h. These
// using-declarations keep the reloc spellings working.

using sym::isSymExpr;
using sym::parseSymExpr;
using sym::printSymExpr;

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

/// True for the typed value transforms (reloc.cast, reloc.quantize,
/// reloc.dequantize; C1, issue #141). The fold pass folds them into a
/// #reloc.typed_plan (C2); the public exporter declares them unsupported
/// until C3 supplies the typed artifact encoding.
bool isTypedValueTransformOp(Operation *op);

//===----------------------------------------------------------------------===//
// Typed value transform legality (C1 rules, shared by ops and plan stages)
//===----------------------------------------------------------------------===//

/// Type pair and policy legality of one typed value transform
/// (docs/reloc-typed-semantics.md §1): cast f32 -> f16 under ieee_rne or
/// f16 -> f32 under exact, quantize f32 -> signless i8 under symmetric_rne,
/// dequantize signless i8 -> f32 under affine.
LogicalResult
verifyValueTransformSignature(function_ref<InFlightDiagnostic()> emitError,
                              ValueTransform transform, NumericPolicy policy,
                              Type input, Type output);

/// Quantization parameter legality over a logical operand `shape`
/// (docs/reloc-typed-semantics.md §4): dense constant or #reloc.binding,
/// role-specific element types, rank 0 (per tensor) or rank 1 (per channel,
/// `axis` present, length equal to the axis extent unless undecidable),
/// finite positive constant scales, zero points in [-128, 127], and under
/// symmetric_rne only the constant zero point 0. Unknown proofs never reject.
LogicalResult
verifyQuantizationParameters(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<Attribute> shape, Attribute scale,
                             Attribute zeroPoint, std::optional<int64_t> axis,
                             NumericPolicy policy);

/// Fold a constant pad fill through one value stage with the C1 reference
/// arithmetic (APFloat, binary32, round-to-nearest-even): ieee_rne / exact
/// casts always fold; quantize and dequantize fold only with per-tensor
/// constant parameters, because a per-channel or runtime parameter gives
/// every padded position its own code. Returns null when not foldable or
/// when `fill` does not have the stage's input type.
TypedAttr foldFillThroughStage(TypedAttr fill, ValueStageAttr stage);

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
