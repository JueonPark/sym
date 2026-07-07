// RUN: sym-opt --allow-unregistered-dialect %s | sym-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// TensorDescAttr round-trips
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tensor_desc_static
func.func @tensor_desc_static() {
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[64, 32], f32, strides = [32, 1]>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[64, 32], f32, strides = [32, 1]>} : () -> ()
  return
}

// CHECK-LABEL: func.func @tensor_desc_symbolic
func.func @tensor_desc_symbolic() {
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[N, N], f32, strides = [s0, 1]>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[N, N], f32, strides = [s0, 1], offset = 0>} : () -> ()
  return
}

// CHECK-LABEL: func.func @tensor_desc_defaults
func.func @tensor_desc_defaults() {
  // No strides, no offset: both defaults omitted on print.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[8], i32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[8], i32>} : () -> ()
  // Offset without strides.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[N], f32, offset = N floordiv 2>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[N], f32, offset = N floordiv 2>} : () -> ()
  return
}

// CHECK-LABEL: func.func @tensor_desc_expressions
func.func @tensor_desc_expressions() {
  // Parse-time simplification: 64 * 64 * x folds the constant product.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[4096 * (N floordiv 64)], f32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[64 * 64 * (N floordiv 64)], f32>} : () -> ()
  // Precedence: parenthesized sum as mul operand keeps its parens.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[(N + 1) * 2], f32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[(N + 1) * 2], f32>} : () -> ()
  // RHS of subtraction keeps parens; left-assoc chain does not.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[N - (a + b), N - a - b], f32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[N - (a + b), N - a - b], f32>} : () -> ()
  // 'div' parses as an alias of 'floordiv'; prints as 'floordiv'.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[N floordiv 64, N mod 64], f32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[N div 64, N mod 64], f32>} : () -> ()
  // Quoted symbol with a non-identifier name stays quoted.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<["my dim" + 1], f32>}
  "test.use_attr"() {desc = #reloc.tensor_desc<["my dim" + 1], f32>} : () -> ()
  // Negative integer literals in extents and offsets.
  // CHECK: "test.use_attr"() {desc = #reloc.tensor_desc<[-7, N + -1], f32, offset = -4>}
  "test.use_attr"() {desc = #reloc.tensor_desc<[-7, N + -1], f32, offset = -4>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// AxisInfoAttr round-trips
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @axis_info
func.func @axis_info() {
  // CHECK: "test.use_attr"() {axis = #reloc.axis_info<{name = "n0", extent = N floordiv 64, src_stride = 64, dst_stride = 4096 * (N floordiv 64)}>}
  "test.use_attr"() {axis = #reloc.axis_info<{name = "n0", extent = N floordiv 64, src_stride = 64, dst_stride = 64 * 64 * (N floordiv 64)}>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// PadFillAttr round-trips
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @pad_fill
func.func @pad_fill() {
  // CHECK: "test.use_attr"() {pad = #reloc.pad_fill<{dst_axis = 1, lo = 0, hi = N mod 64, value = 0.000000e+00 : f32}>}
  "test.use_attr"() {pad = #reloc.pad_fill<{dst_axis = 1, lo = 0, hi = N mod 64, value = 0.0 : f32}>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// DivisibilityAttr / AlignmentAttr round-trips
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @constraint_attrs
func.func @constraint_attrs() {
  // CHECK: "test.use_attr"() {d = #reloc.divisibility<N, 64>}
  "test.use_attr"() {d = #reloc.divisibility<N, 64>} : () -> ()
  // CHECK: "test.use_attr"() {a = #reloc.alignment<2, 128>}
  "test.use_attr"() {a = #reloc.alignment<2, 128>} : () -> ()
  return
}
