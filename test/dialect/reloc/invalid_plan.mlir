// RUN: sym-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// Plans must start with the src descriptor.
// expected-error @below {{expected 'src'}}
"test.use_attr"() {plan = #reloc.plan<dst = tensor<[8], f32>>} : () -> ()

// -----

// Unknown constraint keyword.
// expected-error @below {{expected 'divisible', 'align', 'contiguous', 'no_copy', or 'runtime_pad_check' in constraints}}
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

// -----

// Pad widths provably inconsistent: 6 + 1 + 1 != 9.
// expected-error @below {{pad on dst_axis 0 is inconsistent: extent + lo + hi must equal the dst extent}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[9], f32>, perm = [0], axes = [{name = "x", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// -----

// Negative pad width.
// expected-error @below {{pad lo for dst_axis 0 is provably negative}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[7], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = -1, hi = 0, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// -----

// Pad dst_axis out of range.
// expected-error @below {{pad dst_axis (3) is out of range for 1 axes}}
"test.use_attr"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 3, lo = 0, hi = 0, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
