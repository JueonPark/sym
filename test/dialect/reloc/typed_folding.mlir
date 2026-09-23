// RUN: sym-opt --reloc-fold %s | FileCheck %s
// Idempotence: a second run must be a no-op on the first run's output.
// RUN: sym-opt --reloc-fold %s | sym-opt --reloc-fold | FileCheck %s

// C2 (issue #142) Task 2: typed value transforms fold with the layout chain
// into one #reloc.typed_plan. Transposes and reshapes are index maps and
// commute with element-wise value stages, so the layout folds into one
// #reloc.plan while every stage records how a result coordinate selects its
// channel parameter. Pads do not commute: the fill enters at a definite
// stage, is folded through the later stages, and the original is kept.
// Numerical order is the rules in docs/reloc-typed-semantics.md §5.

// Layout prefix, then a per-channel quantize on the transposed axis 0: the
// channel is the output axis 0 coordinate. Chain ops disappear.
// CHECK-LABEL: func.func @layout_then_quantize
func.func @layout_then_quantize(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], i8> {
  // CHECK-NOT: reloc.transpose
  // CHECK-NOT: reloc.quantize
  // CHECK: reloc.typed_plan_result %{{.*}} plan(#reloc.typed_plan<source = tensor<[4, 6], f32>, result = tensor<[6, 4], i8>, layout = #reloc.plan<src = tensor<[4, 6], f32>, dst = tensor<[6, 4], i8>, perm = [1, 0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 4}, {name = "d1", extent = 4, src_stride = 6, dst_stride = 1}], constraints = {contiguous = [true, false], no_copy = false}, inverse = #map{{[0-9]*}}>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6, 4], scale = #reloc.binding<"row_scale" : [6], f32>, axis = 0, channel = (d0, d1) -> (d0)>]>) : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], i8>
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.quantize %0 axis 0 scale(#reloc.binding<"row_scale" : [6], f32>) policy symmetric_rne : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], i8>
  return %1 : !sym.tensor<[6, 4], i8>
}

// [B, C] -> transpose [C, B]: original channel = output axis 0. Distinct
// scales per channel, so a wrong axis could not pass by accident.
// CHECK-LABEL: func.func @quantize_then_transpose
func.func @quantize_then_transpose(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[3, 2], i8> {
  // CHECK: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[5.000000e-01, 2.500000e-01, 1.250000e-01]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d0)>]
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 0.25, 0.125]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[2, 3], i8> -> !sym.tensor<[3, 2], i8>
  return %1 : !sym.tensor<[3, 2], i8>
}

// [B, 12] -> reshape [B, 4, 3]: original channel = 3 * c_outer + c_inner.
// The layout collapses to one contiguous axis, but the channel map and the
// result descriptor keep the logical rank 3.
// CHECK-LABEL: func.func @quantize_then_split
func.func @quantize_then_split(%t: !sym.tensor<[2, 12], f32>) -> !sym.tensor<[2, 4, 3], i8> {
  // CHECK: #reloc.typed_plan<source = tensor<[2, 12], f32>, result = tensor<[2, 4, 3], i8>, layout = #reloc.plan<src = tensor<[2, 12], f32>, dst = tensor<[24], i8>, perm = [0], axes = [{name = "d0", extent = 24, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = #map{{[0-9]*}}>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 12], scale = #reloc.binding<"s" : [12], f32>, axis = 1, channel = (d0, d1, d2) -> (d1 * 3 + d2)>]>
  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [12], f32>) policy symmetric_rne : !sym.tensor<[2, 12], f32> -> !sym.tensor<[2, 12], i8>
  %1 = reloc.reshape %0 to [2, 4, 3] : !sym.tensor<[2, 12], i8> -> !sym.tensor<[2, 4, 3], i8>
  return %1 : !sym.tensor<[2, 4, 3], i8>
}

// [B, C] -> flatten [B * C]: original channel = flat_index mod C.
// CHECK-LABEL: func.func @quantize_then_flatten
func.func @quantize_then_flatten(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[6], i8> {
  // CHECK: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[5.000000e-01, 2.500000e-01, 1.250000e-01]> : tensor<3xf32>, axis = 1, channel = (d0) -> (d0 mod 3)>]
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 0.25, 0.125]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  %1 = reloc.reshape %0 to [6] : !sym.tensor<[2, 3], i8> -> !sym.tensor<[6], i8>
  return %1 : !sym.tensor<[6], i8>
}

// Symbolic C split by 3: the fold retains divisible(C, 3), the stage keeps
// its logical shape [B, C] with the scale declared over [C] (the length
// guard parameter_length == C is C3's), and the channel is 3 * d1 + d2.
// CHECK-LABEL: func.func @quantize_then_symbolic_split
func.func @quantize_then_symbolic_split(%t: !sym.tensor<["B", "C"], f32>) -> !sym.tensor<["B", "C" floordiv 3, 3], i8> {
  // CHECK: constraints = {divisible(C, 3),
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C], scale = #reloc.binding<"s" : [C], f32>, axis = 1, channel = (d0, d1, d2) -> (d1 * 3 + d2)>]
  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [C], f32>) policy symmetric_rne : !sym.tensor<["B", "C"], f32> -> !sym.tensor<["B", "C"], i8>
  %1 = reloc.reshape %0 to [B, C floordiv 3, 3] : !sym.tensor<["B", "C"], i8> -> !sym.tensor<["B", "C" floordiv 3, 3], i8>
  return %1 : !sym.tensor<["B", "C" floordiv 3, 3], i8>
}

// Symbolic flatten: the channel needs the symbol itself (d0 mod C), bound
// by name through the plan's symbol list.
// CHECK-LABEL: func.func @quantize_then_symbolic_flatten
func.func @quantize_then_symbolic_flatten(%t: !sym.tensor<["B", "C"], f32>) -> !sym.tensor<["B" * "C"], i8> {
  // CHECK: symbols = ["C"]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C], scale = #reloc.binding<"s" : [C], f32>, axis = 1, channel = (d0)[s0] -> (d0 mod s0)>]
  %0 = reloc.quantize %t axis 1 scale(#reloc.binding<"s" : [C], f32>) policy symmetric_rne : !sym.tensor<["B", "C"], f32> -> !sym.tensor<["B", "C"], i8>
  %1 = reloc.reshape %0 to [B * C] : !sym.tensor<["B", "C"], i8> -> !sym.tensor<["B" * "C"], i8>
  return %1 : !sym.tensor<["B" * "C"], i8>
}

// A permutation followed by rank coalescing: quantize on axis 0 (channel
// B), move it last with a transpose, then merge the two leading axes. The
// channel is the last output coordinate, and the merged layout is not a
// standalone transpose.
// CHECK-LABEL: func.func @permutation_then_coalescing
func.func @permutation_then_coalescing(%t: !sym.tensor<[2, 3, 4], f32>) -> !sym.tensor<[12, 2], i8> {
  // CHECK: result = tensor<[12, 2], i8>
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3, 4], scale = dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>, axis = 0, channel = (d0, d1) -> (d1)>]
  %0 = reloc.quantize %t axis 0 scale(dense<[0.5, 0.25]> : tensor<2xf32>) policy symmetric_rne : !sym.tensor<[2, 3, 4], f32> -> !sym.tensor<[2, 3, 4], i8>
  %1 = reloc.transpose %0 perm [1, 2, 0] : !sym.tensor<[2, 3, 4], i8> -> !sym.tensor<[3, 4, 2], i8>
  %2 = reloc.reshape %1 to [12, 2] : !sym.tensor<[3, 4, 2], i8> -> !sym.tensor<[12, 2], i8>
  return %2 : !sym.tensor<[12, 2], i8>
}

// Order witness at scale 0.5, zero point 0: pad f32 with 1.0 then quantize
// gives pad code 2; the original fill and its entry stage are kept.
// CHECK-LABEL: func.func @pad_then_quantize_code_two
func.func @pad_then_quantize_code_two(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  // CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<5.000000e-01> : tensor<f32>>]
  // CHECK-SAME: fills = [{dst_axis = 0, stage = 0, value = 1.000000e+00 : f32}]
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %1 = reloc.quantize %0 scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>
  return %1 : !sym.tensor<[8], i8>
}

// Quantize first, then pad s8 with code 1: pad code 1, entered at stage 1.
// CHECK-LABEL: func.func @quantize_then_pad_code_one
func.func @quantize_then_pad_code_one(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  // CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6], scale = dense<5.000000e-01> : tensor<f32>>]
  // CHECK-SAME: fills = [{dst_axis = 0, stage = 1, value = 1 : i8}]
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], i8>
  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (1 : i8) : !sym.tensor<[6], i8> -> !sym.tensor<[8], i8>
  return %1 : !sym.tensor<[8], i8>
}

// A fill that precedes a cast converts under the cast's own table.
// CHECK-LABEL: func.func @pad_then_cast_folds_fill
func.func @pad_then_cast_folds_fill(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f16> {
  // CHECK: pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1.500000e+00 : f16}]
  // CHECK-SAME: fills = [{dst_axis = 0, stage = 0, value = 1.500000e+00 : f32}]
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.5 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], f16>
  return %1 : !sym.tensor<[8], f16>
}

// Channel-axis padding after a per-channel quantize: the new channels take
// the s8 fill and the channel map shifts by the leading width, so no padded
// position indexes a parameter.
// CHECK-LABEL: func.func @quantize_then_pad_channel_axis
func.func @quantize_then_pad_channel_axis(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[2, 4], i8> {
  // CHECK: pad_fill = [{dst_axis = 1, lo = 1, hi = 0, value = 0 : i8}]
  // CHECK-SAME: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[5.000000e-01, 2.500000e-01, 1.250000e-01]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d1 - 1)>]
  // CHECK-SAME: fills = [{dst_axis = 1, stage = 1, value = 0 : i8}]
  %0 = reloc.quantize %t axis 1 scale(dense<[0.5, 0.25, 0.125]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 3], i8>
  %1 = reloc.pad %0 axis 1 lo 1 hi 0 value (0 : i8) : !sym.tensor<[2, 3], i8> -> !sym.tensor<[2, 4], i8>
  return %1 : !sym.tensor<[2, 4], i8>
}

// Channel-axis padding before a per-channel quantize whose parameters cover
// the padded operand: the parameters index post-pad channels, but the fill
// would need a code per padded channel, so the fill program is absent and the
// fold bails (typed_fold_bail.mlir). Padding a NON-channel axis before a
// per-tensor quantize is the supported case above.

// Lossy pairs are never cancelled: two casts stay two stages over an
// identity layout that is nevertheless not a view.
// CHECK-LABEL: func.func @cast_round_trip_not_cancelled
func.func @cast_round_trip_not_cancelled(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f32> {
  // CHECK: constraints = {contiguous = [true], no_copy = false}
  // CHECK-SAME: stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<cast exact : f16 -> f32, shape = [4]>]
  %0 = reloc.cast %t policy ieee_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], f16>
  %1 = reloc.cast %0 policy exact : !sym.tensor<[4], f16> -> !sym.tensor<[4], f32>
  return %1 : !sym.tensor<[4], f32>
}

// CHECK-LABEL: func.func @quantize_dequantize_not_cancelled
func.func @quantize_dequantize_not_cancelled(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f32> {
  // CHECK: stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [4], scale = dense<5.000000e-01> : tensor<f32>>, #reloc.stage<dequantize affine : i8 -> f32, shape = [4], scale = dense<5.000000e-01> : tensor<f32>>]
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  %1 = reloc.dequantize %0 scale(dense<0.5> : tensor<f32>) policy affine : !sym.tensor<[4], i8> -> !sym.tensor<[4], f32>
  return %1 : !sym.tensor<[4], f32>
}

// A value stage between two layout segments folds with both: the transposes
// compose to the identity view, the stage records the channel over the
// final coordinates, and the plan is still not a view.
// CHECK-LABEL: func.func @layout_stage_layout
func.func @layout_stage_layout(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[2, 3], f16> {
  // CHECK: layout = #reloc.plan<src = tensor<[2, 3], f32>, dst = tensor<[6], f16>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}
  // CHECK-SAME: stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [3, 2]>]
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[2, 3], f32> -> !sym.tensor<[3, 2], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<[3, 2], f32> -> !sym.tensor<[3, 2], f16>
  %2 = reloc.transpose %1 perm [1, 0] : !sym.tensor<[3, 2], f16> -> !sym.tensor<[2, 3], f16>
  return %2 : !sym.tensor<[2, 3], f16>
}

// A single typed op is a complete chain over an identity layout.
// CHECK-LABEL: func.func @standalone_dequantize
func.func @standalone_dequantize(%q: !sym.tensor<["N"], i8>) -> !sym.tensor<["N"], f32> {
  // CHECK-NOT: reloc.dequantize %
  // CHECK: reloc.typed_plan_result %{{.*}} plan(#reloc.typed_plan<source = tensor<[N], i8>, result = tensor<[N], f32>, layout = #reloc.plan<src = tensor<[N], i8>, dst = tensor<[N], f32>, perm = [0], axes = [{name = "d0", extent = N, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = #map{{[0-9]*}}>, stages = [#reloc.stage<dequantize affine : i8 -> f32, shape = [N], scale = dense<2.500000e-01> : tensor<f32>, zero_point = dense<-3> : tensor<i32>>]>)
  %0 = reloc.dequantize %q scale(dense<0.25> : tensor<f32>) zero_point(dense<-3> : tensor<i32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
  return %0 : !sym.tensor<["N"], f32>
}

// Distinct parameter values give distinct plans for identical shapes.
// CHECK-LABEL: func.func @scale_half
func.func @scale_half(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], i8> {
  // CHECK: scale = dense<5.000000e-01> : tensor<f32>
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return %0 : !sym.tensor<[4], i8>
}
// CHECK-LABEL: func.func @scale_quarter
func.func @scale_quarter(%t: !sym.tensor<[4], f32>) -> !sym.tensor<[4], i8> {
  // CHECK: scale = dense<2.500000e-01> : tensor<f32>
  %0 = reloc.quantize %t scale(dense<0.25> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], i8>
  return %0 : !sym.tensor<[4], i8>
}

// Layout-only chains are untouched by C2: they still fold to #reloc.plan.
// CHECK-LABEL: func.func @layout_only_stays_v0
func.func @layout_only_stays_v0(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], f32> {
  // CHECK: reloc.plan_result %{{.*}} plan(#reloc.plan<
  // CHECK-NOT: reloc.typed_plan_result
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  return %0 : !sym.tensor<[6, 4], f32>
}
