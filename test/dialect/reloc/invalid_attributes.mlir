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
