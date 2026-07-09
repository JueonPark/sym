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

/// The wire-format version this runtime decodes. Decoders must reject any
/// other version (spec: "Versioning").
inline constexpr uint32_t kWireFormatVersion = 0;

/// Human-readable library identification.
const char *versionString();

} // namespace reloc

#endif // RELOC_VERSION_H
