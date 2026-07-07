// RUN: sym-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// Plans must start with the src descriptor.
// expected-error @below {{expected 'src'}}
"test.use_attr"() {plan = #reloc.plan<dst = tensor<[8], f32>>} : () -> ()

// -----

// Unknown constraint keyword.
// expected-error @below {{expected 'divisible', 'align', 'contiguous', or 'no_copy' in constraints}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], constraints = {bogus(1)}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// -----

// perm size must match the number of axes.
// expected-error @below {{perm size (2) must match number of axes (1)}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0, 1], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// -----

// contiguity size must match the number of axes (or be absent).
// expected-error @below {{contiguity size (2) must match number of axes (1) or be empty}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true, false]}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// -----

// Malformed expression inside a descriptor.
// expected-error @below {{expected integer, symbol, or '(' in expression}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[*], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
