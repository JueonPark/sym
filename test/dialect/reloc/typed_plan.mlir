// RUN: sym-opt --allow-unregistered-dialect %s | sym-opt --allow-unregistered-dialect | FileCheck %s

// C2 (issue #142) Task 1: the ordered typed-plan representation round-trips.
// A #reloc.typed_plan keeps the logical source/result descriptors, ONE folded
// layout plan (index mapping and fused pad fills in the result dtype), the
// ordered value stages with their parameters and channel-selection maps over
// the logical result coordinates, and the original fills with the stage at
// which each one entered. Plans that share final descriptors but differ in
// stage order, policy or parameter values are different attributes.

// MLIR hoists the layout plan's inverse map to a #map alias (printer-owned).
// CHECK: #map = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #map1 = affine_map<(d0) -> (d0)>

// Per-channel quantize on axis 1 of [2, 3], then transpose: the channel of a
// result element is its output axis 0 coordinate.
// CHECK-LABEL: func.func @quantize_then_transpose
func.func @quantize_then_transpose(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[3, 2], i8> {
  // CHECK: reloc.typed_plan_result %{{.*}} plan(#reloc.typed_plan<source = tensor<[2, 3], f32>, result = tensor<[3, 2], i8>, layout = #reloc.plan<src = tensor<[2, 3], f32>, dst = tensor<[3, 2], i8>, perm = [1, 0], axes = [{name = "d0", extent = 3, src_stride = 1, dst_stride = 2}, {name = "d1", extent = 2, src_stride = 3, dst_stride = 1}], inverse = #map>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[5.000000e-01, 2.500000e-01, 1.250000e-01]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d0)>]>) : !sym.tensor<[2, 3], f32> -> !sym.tensor<[3, 2], i8>
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[2, 3], f32>, result = tensor<[3, 2], i8>, layout = #reloc.plan<src = tensor<[2, 3], f32>, dst = tensor<[3, 2], i8>, perm = [1, 0], axes = [{name = "d0", extent = 3, src_stride = 1, dst_stride = 2}, {name = "d1", extent = 2, src_stride = 3, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d1, d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.25, 0.125]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d0)>]>) : !sym.tensor<[2, 3], f32> -> !sym.tensor<[3, 2], i8>
  return %0 : !sym.tensor<[3, 2], i8>
}

// Two lossy casts stay two stages: the type chain f32 -> f16 -> f32 is not an
// identity and the representation never collapses it.
// CHECK-LABEL: func.func @cast_round_trip_keeps_two_stages
func.func @cast_round_trip_keeps_two_stages(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f32> {
  // CHECK: stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<cast exact : f16 -> f32, shape = [4]>]
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<cast exact : f16 -> f32, shape = [4]>]>) : !sym.tensor<[4], f32> -> !sym.tensor<[4], f32>
  return %0 : !sym.tensor<[4], f32>
}

// Order witness (docs/reloc-typed-semantics.md, scale 0.5, zero point 0):
// padding f32 with 1.0 and then quantizing yields pad code 2, entered at
// stage 0; the fill is folded through the stage and the original is kept.
// CHECK-LABEL: func.func @pad_then_quantize_code_two
func.func @pad_then_quantize_code_two(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  // CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<5.000000e-01> : tensor<f32>>]
  // CHECK-SAME: fills = [{dst_axis = 0, stage = 0, value = 1.000000e+00 : f32}]
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 0, value = 1.0 : f32}]>) : !sym.tensor<[6], f32> -> !sym.tensor<[8], i8>
  return %0 : !sym.tensor<[8], i8>
}

// Quantizing first and then padding s8 with code 1 yields pad code 1, entered
// at stage 1: a different plan for the same descriptors.
// CHECK-LABEL: func.func @quantize_then_pad_code_one
func.func @quantize_then_pad_code_one(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  // CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6], scale = dense<5.000000e-01> : tensor<f32>>]
  // CHECK-SAME: fills = [{dst_axis = 0, stage = 1, value = 1 : i8}]
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 1, value = 1 : i8}]>) : !sym.tensor<[6], f32> -> !sym.tensor<[8], i8>
  return %0 : !sym.tensor<[8], i8>
}

// Symbolic channel: quantize [B, C] on axis 1 then flatten. The channel of
// flat index d0 is d0 mod C; C is a plan symbol bound by name, the scale is a
// runtime binding declared over [C], and the layout keeps the merged axis.
// CHECK-LABEL: func.func @quantize_then_flatten_symbolic
func.func @quantize_then_flatten_symbolic(%t: !sym.tensor<["B", "C"], f32>) -> !sym.tensor<["B" * "C"], i8> {
  // CHECK: #reloc.typed_plan<source = tensor<[B, C], f32>, result = tensor<[B * C], i8>, symbols = ["C"], layout = #reloc.plan<src = tensor<[B, C], f32>, dst = tensor<[B * C], i8>, perm = [0], axes = [{name = "d0", extent = B * C, src_stride = 1, dst_stride = 1}], inverse = #map1>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C], scale = #reloc.binding<"s" : [C], f32>, axis = 1, channel = (d0)[s0] -> (d0 mod s0)>]>
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[B, C], f32>, result = tensor<[B * C], i8>, symbols = ["C"], layout = #reloc.plan<src = tensor<[B, C], f32>, dst = tensor<[B * C], i8>, perm = [0], axes = [{name = "d0", extent = B * C, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C], scale = #reloc.binding<"s" : [C], f32>, axis = 1, channel = (d0)[s0] -> (d0 mod s0)>]>) : !sym.tensor<["B", "C"], f32> -> !sym.tensor<["B" * "C"], i8>
  return %0 : !sym.tensor<["B" * "C"], i8>
}

// Dequantize with a per-channel zero point, then a split of the other axis:
// the channel map is a plain output coordinate; parameter indexing stays
// meaningful after the reshape changed the rank.
// CHECK-LABEL: func.func @dequantize_then_split
func.func @dequantize_then_split(%q: !sym.tensor<[3, 4], i8>) -> !sym.tensor<[3, 2, 2], f32> {
  // CHECK: stages = [#reloc.stage<dequantize affine : i8 -> f32, shape = [3, 4], scale = dense<[1.000000e+00, 5.000000e-01, 2.500000e-01]> : tensor<3xf32>, zero_point = dense<[-128, 0, 127]> : tensor<3xi32>, axis = 0, channel = (d0, d1, d2) -> (d0)>]
  %0 = reloc.typed_plan_result %q plan(#reloc.typed_plan<source = tensor<[3, 4], i8>, result = tensor<[3, 2, 2], f32>, layout = #reloc.plan<src = tensor<[3, 4], i8>, dst = tensor<[3, 2, 2], f32>, perm = [0, 1, 2], axes = [{name = "d0", extent = 3, src_stride = 4, dst_stride = 4}, {name = "d1", extent = 2, src_stride = 2, dst_stride = 2}, {name = "d2", extent = 2, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0, d1, d2) -> (d0, d1, d2)>>, stages = [#reloc.stage<dequantize affine : i8 -> f32, shape = [3, 4], scale = dense<[1.0, 0.5, 0.25]> : tensor<3xf32>, zero_point = dense<[-128, 0, 127]> : tensor<3xi32>, axis = 0, channel = (d0, d1, d2) -> (d0)>]>) : !sym.tensor<[3, 4], i8> -> !sym.tensor<[3, 2, 2], f32>
  return %0 : !sym.tensor<[3, 2, 2], f32>
}

// The attributes are usable standalone.
// CHECK-LABEL: func.func @standalone_attributes
func.func @standalone_attributes() {
  // CHECK: "test.use_attr"() {stage = #reloc.stage<cast exact : f16 -> f32, shape = [N, 64]>}
  "test.use_attr"() {stage = #reloc.stage<cast exact : f16 -> f32, shape = [N, 64]>} : () -> ()
  // CHECK: "test.use_attr"() {fill = #reloc.typed_fill<dst_axis = 1, stage = 0, value = 0x7E00 : f16>}
  "test.use_attr"() {fill = #reloc.typed_fill<dst_axis = 1, stage = 0, value = 0x7E00 : f16>} : () -> ()
  return
}
