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
#include <vector>

namespace mlir {
namespace reloc {

/// Encode `plan` into wire format v0. The result is a pure function of the
/// attribute (deterministic across processes and builds). Inputs the format
/// cannot represent (complex element types, pad values wider than 64 bits,
/// affine ceildiv or symbols in the inverse map) produce an error emitted at
/// `loc` and a failure result.
FailureOr<std::vector<uint8_t>> encodePlan(PlanAttr plan, Location loc);

} // namespace reloc
} // namespace mlir

#endif // RELOC_SERIALIZATION_H
