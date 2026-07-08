// RelocOps.td - Operation definitions for the Reloc dialect
#ifndef RELOC_OPS_TD
#define RELOC_OPS_TD

include "RelocDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Reloc Operations
//===----------------------------------------------------------------------===//

class Reloc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Reloc_Dialect, mnemonic, traits>;

// sym's symbolic tensor type, referenced across dialects via a predicate so
// no sym TableGen files need to be included here.
def Reloc_SymTensor : Type<
    CPred<"::mlir::isa<::mlir::sym::SymbolicTensorType>($_self)">,
    "sym symbolic tensor", "::mlir::sym::SymbolicTensorType">;

def Reloc_TransposeOp : Reloc_Op<"transpose", [Pure]> {
  let summary = "Permute the axes of a symbolic tensor";
  let description = [{
    Result dimension k equals operand dimension perm[k] — the same direction
    as #reloc.plan's perm (destination axis k <- source axis perm[k]).

    Example:
    ```mlir
    %0 = reloc.transpose %t perm [1, 0]
        : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, DenseI64ArrayAttr:$perm);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::llvm::ArrayRef<int64_t>":$perm)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // RELOC_OPS_TD
