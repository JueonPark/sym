// RUN: sym-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// Strides size must match extents size.
// expected-error @below {{strides size (1) must match extents size (2) or be empty}}
"test.use_attr"() {desc = #reloc.tensor_desc<[8, 8], f32, strides = [1]>} : () -> ()

// -----

// Element type must be a valid tensor element type.
// expected-error @below {{element type must be a valid tensor element type}}
"test.use_attr"() {desc = #reloc.tensor_desc<[8], memref<2xf32>>} : () -> ()

// -----

// Operator keyword cannot start an expression operand.
// expected-error @below {{unexpected operator keyword 'floordiv'; expected expression operand}}
"test.use_attr"() {desc = #reloc.tensor_desc<[floordiv], f32>} : () -> ()

// -----

// Dangling operator: nothing after '+'.
// expected-error @below {{expected integer, symbol, or '(' in expression}}
"test.use_attr"() {desc = #reloc.tensor_desc<[N + ], f32>} : () -> ()

// -----

// AxisInfo keys are ordered: name, extent, src_stride, dst_stride.
// expected-error @below {{expected 'name'}}
"test.use_attr"() {axis = #reloc.axis_info<{extent = 8, name = "x", src_stride = 1, dst_stride = 1}>} : () -> ()

// -----

// PadFill dst_axis must be non-negative.
// expected-error @below {{dst_axis must be non-negative}}
"test.use_attr"() {pad = #reloc.pad_fill<{dst_axis = -1, lo = 0, hi = 2, value = 0.0 : f32}>} : () -> ()

// -----

// Divisibility divisor must be positive.
// expected-error @below {{divisor must be positive}}
"test.use_attr"() {d = #reloc.divisibility<N, 0>} : () -> ()

// -----

// Alignment bytes must be positive.
// expected-error @below {{bytes must be positive}}
"test.use_attr"() {a = #reloc.alignment<0, 0>} : () -> ()
