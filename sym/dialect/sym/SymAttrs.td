// SymAttrs.td
#ifndef SYM_ATTRIBUTES_TD
#define SYM_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "SymDialect.td"

// 1. The Atomic Symbol Attribute (e.g., "a", "batch")
def SymbolExprAttr : Sym_Attr<"SymbolExpr", "symbol"> {
  let summary = "Represents a named symbolic variable";
  let parameters = (ins StringRefParameter<"symbol name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

// 2. The Constant Integer Attribute (e.g., 42)
// We wrap MLIR's built-in IntegerAttr or just store the int directly
def ConstantExprAttr : Sym_Attr<"ConstantExpr", "constant"> {
  let summary = "Represents a constant integer value";
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

// 3. The Binary Operation Attribute (e.g., Add, Mul)
// This is recursive: lhs and rhs are generic Attributes (which we verify are ExprAttrs)
// Uses the manually-defined C++ enum SymbolicExprOp from SymDialect.h
def BinaryExprAttr : Sym_Attr<"BinaryExpr", "op"> {
  let summary = "Represents a binary operation (add, mul, div, etc.)";
  
  let parameters = (ins 
    "SymbolicExprOp":$opcode,   // Use the C++ enum directly
    "Attribute":$lhs,           // Recursive structure
    "Attribute":$rhs
  );

  // Custom parser/printer since SymbolicExprOp is a custom C++ enum
  let hasCustomAssemblyFormat = 1;
  
  // Custom verification to ensure lhs/rhs are actually Sym attributes
  let genVerifyDecl = 1;
}

#endif // SYM_ATTRIBUTES_TD