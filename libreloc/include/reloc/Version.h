//===- Version.h - libreloc version and wire-format constants ---*- C++ -*-===//
//
// libreloc is the standalone runtime for #reloc.plan execution. It is
// MLIR-free and torch-free by contract: its only compiler-facing input is
// the serialized wire format (docs/reloc-plan-format.md, frozen v0).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_VERSION_H
#define RELOC_VERSION_H

#include <cstdint>

namespace reloc {

/// The layout-only wire-format version decodePlan decodes. Decoders must
/// reject any other version (spec: "Versioning").
inline constexpr uint32_t kWireFormatVersion = 0;

/// The typed wire-format version decodeTypedPlan decodes (C3, issue #143;
/// spec: "Wire Format v1"). v0 stays frozen and byte-identical.
inline constexpr uint32_t kTypedWireFormatVersion = 1;

/// Human-readable library identification.
const char *versionString();

} // namespace reloc

#endif // RELOC_VERSION_H
