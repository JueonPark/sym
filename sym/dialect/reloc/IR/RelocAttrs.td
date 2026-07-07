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

#endif // RELOC_ATTRIBUTES_TD
