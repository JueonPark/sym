#ifndef RELOC_DIALECT_H
#define RELOC_DIALECT_H

#include "SymDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

// Include TableGen outputs for Dialect
#include "RelocDialect.h.inc"

// Include TableGen outputs for Attributes
#define GET_ATTRDEF_CLASSES
#include "RelocAttributes.h.inc"

#endif // RELOC_DIALECT_H
