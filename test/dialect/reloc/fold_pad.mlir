// RUN: sym-opt --test-reloc-transfer --verify-diagnostics %s

// NOTE: symbolic (alignment) pads cannot be written here: the padded
// result type would need a binary-expr dim, which the !sym.tensor type
// grammar cannot express. Symbolic pads + runtime_pad_check are covered
// by the C++ oracle tests in unittest/reloc/PlanBuilderTest.cpp.

// Single static pad: the axis keeps its VALID extent (6); the dst grows
// to 8 and the pad_fill entry carries the widths and fill value.
func.func @pad_simple(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1.000000e+00 : f32}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>}}
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return %0 : !sym.tensor<[8], f32>
}

// Pad then transpose: the pad travels with its axis (dst_axis renumbers
// from 1 to 0), and dst strides are row-major over the PADDED extents.
func.func @pad_then_transpose(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[8, 4], f32> {
  %0 = reloc.pad %t axis 1 lo 0 hi 2 value (0.0 : f32) : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 8], f32>
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[4, 6], f32>, dst = tensor<[8, 4], f32>, perm = [1, 0], axes = [{name = "d1", extent = 6, src_stride = 1, dst_stride = 4}, {name = "d0", extent = 4, src_stride = 6, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.000000e+00 : f32}], constraints = {contiguous = [true, false], no_copy = false}, inverse = affine_map<(d0, d1) -> (d1, d0)>>}}
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[4, 8], f32> -> !sym.tensor<[8, 4], f32>
  return %1 : !sym.tensor<[8, 4], f32>
}

// Reshape keeping the padded axis 1:1 folds; the split applies to the
// unpadded axis only.
func.func @pad_keep_then_split(%t: !sym.tensor<[6, 4], f32>) -> !sym.tensor<[8, 2, 2], f32> {
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6, 4], f32> -> !sym.tensor<[8, 4], f32>
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6, 4], f32>, dst = tensor<[8, 2, 2], f32>, perm = [0, 1, 2], axes = [{name = "d0", extent = 6, src_stride = 4, dst_stride = 4}, {name = "d1", extent = 2, src_stride = 2, dst_stride = 2}, {name = "d2", extent = 2, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 0.000000e+00 : f32}], constraints = {contiguous = [false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d0, d1, d2)>>}}
  %1 = reloc.reshape %0 to [8, 2, 2] : !sym.tensor<[8, 4], f32> -> !sym.tensor<[8, 2, 2], f32>
  return %1 : !sym.tensor<[8, 2, 2], f32>
}
