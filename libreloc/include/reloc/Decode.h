//===- Decode.h - wire-format v0 decoder ----------------------*- C++ -*-===//
//
// The runtime's trust boundary: decodePlan validates everything
// (docs/reloc-plan-format.md) and never trusts a count before checking it
// against the remaining byte budget. Errors carry the byte offset of the
// violated item.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_DECODE_H
#define RELOC_DECODE_H

#include "reloc/Plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace reloc {

struct DecodeError {
  size_t offset = 0;
  std::string message;
};

using DecodeResult = std::variant<RelocationPlan, DecodeError>;

/// Decode a wire-format-v0 plan. On success the RelocationPlan satisfies
/// every structural invariant listed in the format doc (validated here,
/// not assumed): section arities, expression stack discipline, opcode
/// context rules, and index ranges. Rejects every other version (a v1
/// typed blob fails here exactly as it does on a pre-v1 runtime).
DecodeResult decodePlan(const uint8_t *data, size_t size);

/// The wire version of a blob with a valid header, or nullopt when the
/// magic is wrong or the header is truncated. Lets a consumer choose
/// between decodePlan (v0) and decodeTypedPlan (v1) without guessing.
std::optional<uint32_t> peekWireVersion(const uint8_t *data, size_t size);

using TypedDecodeResult = std::variant<TypedRelocationPlan, DecodeError>;

/// Decode a wire-format-v1 typed plan (C3, issue #143). Validates every
/// invariant of the "Decoder-enforced invariants" list in the format doc:
/// the v0 layout body, transform/policy/type pairs with signedness, the
/// stage type chain, parameter kinds/ranks/types/inline values, binding
/// declaration consistency, channel expressions over the result rank, and
/// fills that fold to the layout's fused fills. Rejects every other
/// version. A decoded plan is a validated representation, not an execution
/// capability.
TypedDecodeResult decodeTypedPlan(const uint8_t *data, size_t size);

} // namespace reloc

#endif // RELOC_DECODE_H
