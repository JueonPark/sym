//===- RelocPasses.h - Reloc dialect passes
//--------------------------------===//
//
// This file declares passes for the Reloc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_PASSES_H
#define RELOC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace reloc {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all Reloc dialect passes.
void registerRelocPasses();

//===----------------------------------------------------------------------===//
// Generated Pass Declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "RelocPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RelocPasses.h.inc"

} // namespace reloc
} // namespace mlir

#endif // RELOC_PASSES_H
