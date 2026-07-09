// RelocOps.td - Operation definitions for the Reloc dialect
#ifndef RELOC_OPS_TD
#define RELOC_OPS_TD

include "RelocDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
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

def Reloc_ReshapeOp : Reloc_Op<"reshape", [Pure]> {
  let summary = "Reinterpret a symbolic tensor's shape (row-major)";
  let description = [{
    Row-major reinterpretation of the operand into target_shape, whose
    entries are sym expression attributes and must equal the result
    dimensions. The element count is preserved; the verifier rejects only
    provably-changed counts (symbolically undecidable counts pass — the
    folding pass emits the constraint).

    Example:
    ```mlir
    %0 = reloc.reshape %t to [N floordiv 64, 64]
        : !sym.tensor<["N"], f32> -> !sym.tensor<[(N floordiv 64), 64], f32>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, ArrayAttr:$targetShape);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   "::llvm::ArrayRef<::mlir::Attribute>":$targetShape)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Reloc_PadOp : Reloc_Op<"pad", [Pure]> {
  let summary = "Pad one axis of a symbolic tensor";
  let description = [{
    lo/hi are leading/trailing pad WIDTHS (the tensor.pad convention,
    identical to #reloc.plan's pad_fill): the result dimension at `axis`
    equals operand dimension + lo + hi; all other dimensions are unchanged.
    `value` fills the padded elements and must match the element type.

    Example:
    ```mlir
    %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32)
        : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, I64Attr:$axis, AnyAttr:$lo,
                       AnyAttr:$hi, TypedAttrInterface:$value);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis,
                   "::mlir::Attribute":$lo, "::mlir::Attribute":$hi,
                   "::mlir::TypedAttr":$value)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// #reloc.plan referenced via a predicate (like Reloc_SymTensor above) so no
// attribute TableGen files need to be included here.
def Reloc_PlanAttrParam : Attr<
    CPred<"::mlir::isa<::mlir::reloc::PlanAttr>($_self)">,
    "reloc plan attribute"> {
  let storageType = [{ ::mlir::reloc::PlanAttr }];
  let returnType = [{ ::mlir::reloc::PlanAttr }];
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def Reloc_PlanResultOp : Reloc_Op<"plan_result", [Pure]> {
  let summary = "Materialized relocation plan applied to a source tensor";
  let description = [{
    First-class result of the --reloc-fold pass: `input` relocated
    according to `plan`. Replaces a folded straight-line reloc.* chain so
    later passes can consume the #reloc.plan attribute directly.

    Example:
    ```mlir
    %0 = reloc.plan_result %t plan(#reloc.plan<...>)
        : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, Reloc_PlanAttrParam:$plan);
  let results = (outs Reloc_SymTensor:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // RELOC_OPS_TD
