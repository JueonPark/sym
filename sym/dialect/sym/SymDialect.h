#ifndef SYM_DIALECT_H
#define SYM_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace sym {

// Define the OpCode Enum for symbolic binary operations
enum class SymbolicExprOp { Add, Sub, Mul, Div, Mod };

// Helper to convert Enum to String (for printing)
llvm::StringRef stringifySymbolicExprOp(SymbolicExprOp op);
// Helper to convert String to Enum (for parsing)
std::optional<SymbolicExprOp> symbolizeSymbolicExprOp(llvm::StringRef str);

} // namespace sym
} // namespace mlir

// Include TableGen outputs for Dialect
#include "SymDialect.h.inc"

// Include TableGen outputs for Attributes
#define GET_ATTRDEF_CLASSES
#include "SymAttributes.h.inc"

// Include TableGen outputs for Types
#define GET_TYPEDEF_CLASSES
#include "SymTypes.h.inc"

// Include TableGen outputs for Operations
#define GET_OP_CLASSES
#include "SymOps.h.inc"

#endif // SYM_DIALECT_H