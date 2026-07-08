#include "RelocDialect.h"

using namespace mlir;
using namespace mlir::reloc;

// Include generated dialect definitions
#include "RelocDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Reloc Dialect Initialization
//===----------------------------------------------------------------------===//

void RelocDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "RelocOps.cpp.inc"
      >();
}
