//===- SymPasses.td - Sym dialect passes ----------------------------------===//
//
// This file defines passes for the Sym dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SYM_PASSES_TD
#define SYM_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SymbolicShapeInferencePass
//===----------------------------------------------------------------------===//

def SymbolicShapeInferencePass : Pass<"sym-shape-inference", "::mlir::func::FuncOp"> {
  let summary = "Infer symbolic shapes for operations";
  let description = [{
    This pass walks through operations in a function and infers symbolic
    shapes for operations that implement the SymbolicShapeOpInterface.
    
    For each operation with the interface, it:
    1. Collects the symbolic shapes of the operands
    2. Calls inferSymbolicShapes() to compute result shapes
    3. Updates the operation's result types to use !sym.tensor<...>
    
    Operations without the interface are left unchanged.
    
    Example:
    ```mlir
    // Before:
    %0 = arith.addf %a, %b : tensor<?x?xf32>
    
    // After (if %a has shape [batch, 32] and %b has shape [1, 32]):
    %0 = arith.addf %a, %b : !sym.tensor<["batch", 32], f32>
    ```
  }];
  
  let dependentDialects = [
    "sym::SymDialect",
    "arith::ArithDialect"
  ];
}

#endif // SYM_PASSES_TD
