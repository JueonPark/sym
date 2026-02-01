// SymTypes.td
#ifndef SYM_TYPES_TD
#define SYM_TYPES_TD

include "SymDialect.td"

def SymbolicTensorType : Sym_Type<"SymbolicTensor", "tensor"> {
  let summary = "Tensor with structured symbolic dimensions";
  
  let parameters = (ins
    // Now explicitly holds a list of Attributes (which must be SymbolExpr, ConstantExpr, etc.)
    ArrayRefParameter<"Attribute">:$shape,
    "Type":$elementType
  );

  // We keep custom assembly to make it readable: 
  // !sym.tensor<[16, #sym.op<add, "a", 1>], f32>
  let hasCustomAssemblyFormat = 1;

  // Enable verification to ensure shape attributes are valid
  let genVerifyDecl = 1;
}

#endif // SYM_TYPES_TD