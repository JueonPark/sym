//===- SymPasses.h - Sym dialect passes -----------------------------------===//
//
// This file declares passes for the Sym dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SYM_PASSES_H
#define SYM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sym {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all Sym dialect passes.
void registerSymPasses();

//===----------------------------------------------------------------------===//
// Generated Pass Declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "SymPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "SymPasses.h.inc"

} // namespace sym
} // namespace mlir

#endif // SYM_PASSES_H
