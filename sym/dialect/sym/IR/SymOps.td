// SymOps.td - Operation definitions for the Sym dialect
#ifndef SYM_OPS_TD
#define SYM_OPS_TD

include "SymDialect.td"
include "SymTypes.td"
include "SymAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Sym Operations
//===----------------------------------------------------------------------===//

class Sym_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sym_Dialect, mnemonic, traits>;

// Placeholder operation for symbolic constants
def Sym_ConstantOp : Sym_Op<"constant", []> {
  let summary = "Create a symbolic constant";
  let description = [{
    Creates a constant with a symbolic tensor type.
    
    Example:
    ```mlir
    %0 = sym.constant : !sym.tensor<["batch", 32], f32>
    ```
  }];
  
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// Operation to convert a tensor to a symbolic tensor type
def Sym_ChangeTypeOp : Sym_Op<"change_type", [Pure]> {
  let summary = "Convert a tensor to a symbolic tensor type";
  let description = [{
    Converts an input tensor to a symbolic tensor type. This operation
    acts as a bridge between standard MLIR tensor types and symbolic
    tensor types.
    
    Example:
    ```mlir
    %0 = sym.change_type %input : tensor<?x?xf32> -> !sym.tensor<["batch", "seq"], f32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs SymbolicTensorType:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // SYM_OPS_TD
