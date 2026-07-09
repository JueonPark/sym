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
/// context rules, and index ranges.
DecodeResult decodePlan(const uint8_t *data, size_t size);

} // namespace reloc

#endif // RELOC_DECODE_H
