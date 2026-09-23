// RelocAttrs.td
#ifndef RELOC_ATTRIBUTES_TD
#define RELOC_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "RelocDialect.td"

// 1. Symbolic tensor descriptor: per-axis extent/stride exprs, offset expr,
//    element type. Strides may be empty (canonical row-major); offset
//    defaults to constant 0. All exprs are sym expression attributes.
def Reloc_TensorDescAttr : Reloc_Attr<"TensorDesc", "tensor_desc"> {
  let summary = "Symbolic tensor descriptor (extents, strides, offset, element type)";
  let parameters = (ins
    ArrayRefParameter<"::mlir::Attribute">:$extents,
    ArrayRefParameter<"::mlir::Attribute">:$strides,
    "::mlir::Attribute":$offset,
    "::mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 2. Axis mapping entry: named axis with extent and src/dst strides.
def Reloc_AxisInfoAttr : Reloc_Attr<"AxisInfo", "axis_info"> {
  let summary = "Named plan axis with extent and src/dst strides";
  let parameters = (ins
    StringRefParameter<"axis name">:$name,
    "::mlir::Attribute":$extent,
    "::mlir::Attribute":$srcStride,
    "::mlir::Attribute":$dstStride
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 3. Padding specification for one destination axis.
def Reloc_PadFillAttr : Reloc_Attr<"PadFill", "pad_fill"> {
  let summary = "Pad-fill spec: dst axis, lo/hi pad extents, fill value";
  let parameters = (ins
    "int64_t":$dstAxis,
    "::mlir::Attribute":$lo,
    "::mlir::Attribute":$hi,
    "::mlir::TypedAttr":$value
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 4. Divisibility constraint: expr is divisible by divisor.
def Reloc_DivisibilityAttr : Reloc_Attr<"Divisibility", "divisibility"> {
  let summary = "Divisibility constraint (expr, divisor)";
  let parameters = (ins
    "::mlir::Attribute":$expr,
    "int64_t":$divisor
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 5. Alignment constraint: axis base address aligned to bytes.
def Reloc_AlignmentAttr : Reloc_Attr<"Alignment", "alignment"> {
  let summary = "Alignment constraint (axis, bytes)";
  let parameters = (ins
    "int64_t":$axis,
    "int64_t":$bytes
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 6. The relocation plan: a serializable, symbolic representation of a
//    folded layout-transform chain. Assembly format (one line; pad_fill and
//    constraints sections optional):
//      #reloc.plan<src = tensor<...>, dst = tensor<...>, perm = [ints],
//                  axes = [{...}, ...], pad_fill = [{...}, ...],
//                  constraints = {divisible(e, k), align(a, b),
//                                 contiguous = [bools], no_copy = bool},
//                  inverse = affine_map<...>>
def Reloc_PlanAttr : Reloc_Attr<"Plan", "plan"> {
  let summary = "Serializable symbolic relocation plan";
  let description = [{
    Captures a folded layout-transform chain: src/dst tensor descriptors,
    the dst-axis -> src-axis permutation, per-axis extent/stride mapping,
    pad-fill specs, constraint set (divisibility / alignment / contiguity),
    the no_copy flag, and the dst -> src inverse index map.
  }];
  let parameters = (ins
    "TensorDescAttr":$src,
    "TensorDescAttr":$dst,
    "::mlir::DenseI64ArrayAttr":$perm,
    ArrayRefParameter<"AxisInfoAttr">:$axes,
    ArrayRefParameter<"PadFillAttr">:$padFill,
    ArrayRefParameter<"DivisibilityAttr">:$divisibility,
    ArrayRefParameter<"AlignmentAttr">:$alignment,
    "::mlir::DenseBoolArrayAttr":$contiguity,
    "bool":$noCopy,
    "bool":$runtimePadCheck,
    "::mlir::AffineMapAttr":$inverse
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 7. Numerical policy of a typed value transform (C1, issue #141). Every
//    case names exactly one normative table in docs/reloc-typed-semantics.md;
//    a different rounding, saturation or NaN behavior is a new case with its
//    own name, never a reinterpretation of an existing one.
//      ieee_rne      f32 -> f16 narrowing: IEEE 754 round-to-nearest-even,
//                    overflow to +-inf, subnormals preserved, NaN -> NaN.
//      exact         f16 -> f32 widening: every value exactly representable.
//      symmetric_rne f32 -> s8: zero point 0, inv = fl32(1 / scale),
//                    t = fl32(x * inv), clamp max-then-min (NaN -> -128),
//                    round-to-nearest-even; the existing kernel contract.
//      affine        s8 -> f32: fl32((q - zero_point) * scale).
def Reloc_NumericPolicy : I32EnumAttr<"NumericPolicy",
    "declared numerical policy of a typed value transform", [
  I32EnumAttrCase<"IeeeRne", 0, "ieee_rne">,
  I32EnumAttrCase<"Exact", 1, "exact">,
  I32EnumAttrCase<"SymmetricRne", 2, "symmetric_rne">,
  I32EnumAttrCase<"Affine", 3, "affine">
]> {
  let cppNamespace = "::mlir::reloc";
  let genSpecializedAttr = 0;
}

def Reloc_NumericPolicyAttr
    : EnumAttr<Reloc_Dialect, Reloc_NumericPolicy, "policy"> {
  let assemblyFormat = "`<` $value `>`";
}

// 7b. Which typed value transform a plan stage applies (C2, issue #142).
def Reloc_ValueTransform : I32EnumAttr<"ValueTransform",
    "typed value transform applied by a plan stage", [
  I32EnumAttrCase<"Cast", 0, "cast">,
  I32EnumAttrCase<"Quantize", 1, "quantize">,
  I32EnumAttrCase<"Dequantize", 2, "dequantize">
]> {
  let cppNamespace = "::mlir::reloc";
  let genSpecializedAttr = 0;
}

// 8. Runtime quantization parameter declaration: a stable binding name (the
//    identity the runtime binds by, like a plan symbol; never an address),
//    the declared extents (rank 0 per tensor, rank 1 per channel; sym
//    expressions, so a symbolic channel length stays symbolic) and the
//    declared element type. Value and length guards on the bound tensor are
//    bind-time obligations (C3), not verifier facts.
//      #reloc.binding<"w_scale" : [C], f32>
def Reloc_ParamBindingAttr : Reloc_Attr<"ParamBinding", "binding"> {
  let summary = "Declared runtime quantization parameter (name, extents, element type)";
  let parameters = (ins
    StringRefParameter<"binding name">:$name,
    ArrayRefParameter<"::mlir::Attribute">:$extents,
    "::mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Typed plan (C2, issue #142): ordered value stages over one layout plan
//===----------------------------------------------------------------------===//

// 9. One value stage: the transform, its policy, input/output element types,
//    the logical shape of its operand (the shape the channel axis refers to),
//    the C1 parameters, and -- per channel only -- the channel-selection map
//    from the typed plan's logical RESULT coordinates (dims) and plan symbols
//    to the parameter index. Assembly:
//      #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C],
//                   scale = <dense|binding>, zero_point = <dense|binding>,
//                   axis = 1, channel = (d0, d1)[s0] -> (d1 mod s0)>
def Reloc_ValueStageAttr : Reloc_Attr<"ValueStage", "stage"> {
  let summary = "One typed value stage of a typed plan";
  let parameters = (ins
    "::mlir::reloc::ValueTransform":$transform,
    "::mlir::reloc::NumericPolicy":$policy,
    "::mlir::Type":$inputType,
    "::mlir::Type":$outputType,
    ArrayRefParameter<"::mlir::Attribute">:$shape,
    "::mlir::Attribute":$scale,
    "::mlir::Attribute":$zeroPoint,
    "int64_t":$axis,
    "::mlir::AffineMap":$channel
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 10. A pad fill as the program wrote it, with the stage index at which it
//     entered the computation (0 = before every value stage). The layout
//     plan's pad_fill carries the FUSED fill in the result dtype; this entry
//     keeps the original so the requested computation is reconstructible and
//     the fusion is verifiable. Body syntax (shared with the typed plan):
//       {dst_axis = 0, stage = 0, value = 1.0 : f32}
def Reloc_TypedFillAttr : Reloc_Attr<"TypedFill", "typed_fill"> {
  let summary = "Original pad fill and the value stage it entered at";
  let parameters = (ins
    "int64_t":$dstAxis,
    "int64_t":$stage,
    "::mlir::TypedAttr":$value
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

// 11. The typed plan: logical source/result descriptors, plan symbols bound
//     by name (affine symbol positions of the channel maps), ONE layout plan
//     whose src/dst element types are the source/result dtypes and whose pad
//     fills are fused into the result dtype, the ordered value stages, and
//     the original fills with their entry stages. Verified as a type chain.
//       #reloc.typed_plan<source = tensor<...>, result = tensor<...>,
//                         symbols = ["C"], layout = #reloc.plan<...>,
//                         stages = [#reloc.stage<...>, ...],
//                         fills = [{dst_axis = 0, stage = 0, value = ...}]>
def Reloc_TypedPlanAttr : Reloc_Attr<"TypedPlan", "typed_plan"> {
  let summary = "Ordered typed value stages over one folded layout plan";
  let parameters = (ins
    "TensorDescAttr":$source,
    "TensorDescAttr":$result,
    "::mlir::ArrayAttr":$symbols,
    "PlanAttr":$layout,
    ArrayRefParameter<"ValueStageAttr">:$stages,
    ArrayRefParameter<"TypedFillAttr">:$fills
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // RELOC_ATTRIBUTES_TD
