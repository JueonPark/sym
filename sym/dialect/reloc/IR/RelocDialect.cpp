#include "RelocDialect.h"

using namespace mlir;
using namespace mlir::reloc;

// Include generated dialect definitions
#include "RelocDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Reloc Dialect Initialization
//===----------------------------------------------------------------------===//

void RelocDialect::initialize() {
  // No types, attributes, or operations registered yet (added in A2/A5).
}
