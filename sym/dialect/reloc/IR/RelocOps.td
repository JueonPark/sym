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

//===----------------------------------------------------------------------===//
// Typed value transforms (C1, issue #141)
//===----------------------------------------------------------------------===//
//
// Value conversion is kept separate from address mapping: these ops preserve
// the logical shape (every result dimension equals the operand dimension) and
// change only the element type under a named numerical policy whose
// normative definition is docs/reloc-typed-semantics.md. Verifiers enforce
// static legality; value guards on runtime parameters and undecidable
// symbolic channel lengths are bind-time obligations handed to C3. A valid
// typed op is not yet an exportable artifact (C2/C3).

// #reloc.policy referenced via a predicate (like Reloc_PlanAttrParam) so no
// attribute TableGen files need to be included here.
def Reloc_NumericPolicyParam : Attr<
    CPred<"::mlir::isa<::mlir::reloc::NumericPolicyAttr>($_self)">,
    "reloc numerical policy attribute"> {
  let storageType = [{ ::mlir::reloc::NumericPolicyAttr }];
  let returnType = [{ ::mlir::reloc::NumericPolicy }];
  let convertFromStorage = "$_self.getValue()";
  let constBuilderCall =
      "::mlir::reloc::NumericPolicyAttr::get($_builder.getContext(), $0)";
}

def Reloc_CastOp : Reloc_Op<"cast", [Pure]> {
  let summary = "Convert element values between floating-point types";
  let description = [{
    Element-wise floating-point conversion with an explicit numerical
    policy. Supported pairs: f32 -> f16 under `ieee_rne` (round to nearest
    even, overflow to infinity, subnormals preserved) and f16 -> f32 under
    `exact`. Integer storage never appears in a cast: quantization and
    dequantization are distinct operations.

    Example:
    ```mlir
    %0 = reloc.cast %t policy ieee_rne
        : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], f16>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, Reloc_NumericPolicyParam:$policy);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Type":$elementType,
                   "::mlir::reloc::NumericPolicy":$policy)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Reloc_QuantizeOp : Reloc_Op<"quantize", [Pure]> {
  let summary = "Quantize f32 values to signed int8 with declared parameters";
  let description = [{
    f32 -> signed int8 (stored as signless `i8`; the signed interpretation is
    part of this operation). `scale` (f32, the size of one quantization
    step; the reciprocal is formed once in binary32) and the optional
    `zero_point` (signless integer in [-128, 127], default 0) are either
    dense constants or `#reloc.binding` runtime declarations. Without `axis`
    the form is per tensor (rank-0 parameters); with `axis` it is per
    channel along that operand axis (rank-1 scale whose length equals the
    axis extent; zero point rank 0 or 1). The only defined policy is
    `symmetric_rne`, which admits only the constant zero point 0.

    Example:
    ```mlir
    %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 1.0, 2.0]> : tensor<3xf32>)
        policy symmetric_rne
        : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, AnyAttr:$scale,
                       OptionalAttr<AnyAttr>:$zero_point,
                       OptionalAttr<I64Attr>:$axis,
                       Reloc_NumericPolicyParam:$policy);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Attribute":$scale,
                   "::mlir::Attribute":$zeroPoint,
                   "::std::optional<int64_t>":$axis,
                   "::mlir::reloc::NumericPolicy":$policy)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Reloc_DequantizeOp : Reloc_Op<"dequantize", [Pure]> {
  let summary = "Dequantize signed int8 values to f32 with declared parameters";
  let description = [{
    signed int8 (signless `i8` storage) -> f32 under the `affine` policy:
    fl32((q - zero_point) * scale), one binary32 rounding. Parameter forms
    and channel-axis rules are those of reloc.quantize; zero points may take
    any value in [-128, 127]. Dequantization is not the inverse of
    quantization and is never cancelled against it.

    Example:
    ```mlir
    %0 = reloc.dequantize %q scale(#reloc.binding<"s" : [], f32>) policy affine
        : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
    ```
  }];

  let arguments = (ins Reloc_SymTensor:$input, AnyAttr:$scale,
                       OptionalAttr<AnyAttr>:$zero_point,
                       OptionalAttr<I64Attr>:$axis,
                       Reloc_NumericPolicyParam:$policy);
  let results = (outs Reloc_SymTensor:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Attribute":$scale,
                   "::mlir::Attribute":$zeroPoint,
                   "::std::optional<int64_t>":$axis,
                   "::mlir::reloc::NumericPolicy":$policy)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // RELOC_OPS_TD
