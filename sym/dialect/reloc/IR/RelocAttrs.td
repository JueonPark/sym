// RelocAttrs.td
#ifndef RELOC_ATTRIBUTES_TD
#define RELOC_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
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

#endif // RELOC_ATTRIBUTES_TD
