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

#endif // RELOC_ATTRIBUTES_TD
