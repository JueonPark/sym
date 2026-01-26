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

#endif // SYM_OPS_TD
