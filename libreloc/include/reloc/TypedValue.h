//===- TypedValue.h - runtime reference arithmetic for typed stages -*- C++
//-*-===//
//
// The MLIR-free side of docs/reloc-typed-semantics.md: element widths of
// wire types, and the reference fold of one constant value through one
// value stage (casts always; quantize/dequantize only with inline
// per-tensor parameters). The decoder uses it to re-verify fused pad fills
// (C3); R3's scalar reference paths reuse the same functions.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_TYPEDVALUE_H
#define RELOC_TYPEDVALUE_H

#include "reloc/Plan.h"

#include <cstdint>
#include <optional>

namespace reloc {
namespace typed {

/// Byte width of a wire element type, or 0 when the bitwidth is not a
/// positive multiple of 8.
uint32_t byteWidth(ElementType type);

/// True iff two wire types agree in kind and bitwidth.
bool sameType(ElementType lhs, ElementType rhs);

/// Fold the value `bits` (bit pattern of `type`, zero-extended) through
/// `stage` with the reference arithmetic. Returns nullopt when `type` is
/// not the stage's input type or when the stage's parameters are per
/// channel or runtime bindings (every padded position would then need its
/// own value). The result is the bit pattern of the stage's output type.
std::optional<uint64_t> foldFill(uint64_t bits, ElementType type,
                                 const ValueStage &stage);

/// Inline parameter values as host scalars. `scaleAt` reads element `index`
/// of an inline f32 parameter; `zeroPointAt` sign-extends an inline integer
/// parameter element from its bitwidth. Both require kind == Inline and an
/// in-range index (asserted).
float scaleAt(const StageParam &param, size_t index);
int64_t zeroPointAt(const StageParam &param, size_t index);

} // namespace typed
} // namespace reloc

#endif // RELOC_TYPEDVALUE_H
