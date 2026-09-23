//===- RelocSerialization.h - Plan wire-format encoder ----------*- C++ -*-===//
//
// Compiler-side encoder for the MLIR-free plan wire format consumed by
// libreloc. The byte layout is normatively documented in
// docs/reloc-plan-format.md; this encoder implements format version 0.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_SERIALIZATION_H
#define RELOC_SERIALIZATION_H

#include "RelocDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace reloc {

/// Encode `plan` into wire format v0. The result is a pure function of the
/// attribute (deterministic across processes and builds). Inputs the format
/// cannot represent (complex element types, pad values wider than 64 bits,
/// affine ceildiv or symbols in the inverse map) produce an error emitted at
/// `loc` and a failure result.
/// If supplied, symbolNames receives the exact wire symbol-table order on
/// success.
FailureOr<std::vector<uint8_t>>
encodePlan(PlanAttr plan, Location loc,
           std::vector<std::string> *symbolNames = nullptr);

/// Encode a typed plan into wire format v1 (docs/reloc-plan-format.md, "Wire
/// Format v1"; C3, issue #143): logical source/result descriptors, the v0
/// layout body with its fused fills, the ordered stages (types with
/// signedness, parameters as exact inline bits or named bindings, channel
/// maps over the result coordinates through the wire symbol table) and the
/// original fills with entry stages. Deterministic; the symbol table is
/// first-use order over the sections. `symbolNames` receives that order.
FailureOr<std::vector<uint8_t>>
encodeTypedPlan(TypedPlanAttr plan, Location loc,
                std::vector<std::string> *symbolNames = nullptr);

} // namespace reloc
} // namespace mlir

#endif // RELOC_SERIALIZATION_H
