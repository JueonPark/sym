// RUN: sym-opt --allow-unregistered-dialect --reloc-fold %s | FileCheck %s
// Idempotence: re-running over marked output changes nothing.
// RUN: sym-opt --allow-unregistered-dialect --reloc-fold %s | sym-opt --allow-unregistered-dialect --reloc-fold | FileCheck %s

// C2 (issue #142): a failed typed fold preserves the COMPLETE original chain
// and a stable reason. Every op of the chain carries reloc.fallback plus
// reloc.fallback_reason; nothing partially folded is ever published.
// Layout-only chains keep their v0 marking (reloc.fallback alone).

// A pad on a non-channel axis before a per-channel quantize: every padded
// position would take its own channel's code, so there is no single fused
// fill and the fill program is absent.
// CHECK-LABEL: func.func @pad_before_per_channel_quantize_bails
func.func @pad_before_per_channel_quantize_bails(%t: !sym.tensor<[6, 3], f32>) -> !sym.tensor<[8, 3], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.pad %{{.*}} axis 0 lo 1 hi 1 value (1.000000e+00 : f32) {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  // CHECK: reloc.quantize %{{.*}} axis 1 scale(dense<[5.000000e-01, 2.500000e-01, 1.250000e-01]> : tensor<3xf32>) policy symmetric_rne {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6, 3], f32> -> !sym.tensor<[8, 3], f32>
  %1 = reloc.quantize %0 axis 1 scale(dense<[0.5, 0.25, 0.125]> : tensor<3xf32>) policy symmetric_rne : !sym.tensor<[8, 3], f32> -> !sym.tensor<[8, 3], i8>
  return %1 : !sym.tensor<[8, 3], i8>
}

// Channel-axis padding before a per-channel quantize: the parameters cover
// the padded operand, but the padded channels would each need their own
// code. Rejected for the same reason.
// CHECK-LABEL: func.func @pad_channel_axis_before_per_channel_quantize_bails
func.func @pad_channel_axis_before_per_channel_quantize_bails(%t: !sym.tensor<[2, 3], f32>) -> !sym.tensor<[2, 4], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.pad {{.*}} {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  // CHECK: reloc.quantize {{.*}} {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  %0 = reloc.pad %t axis 1 lo 1 hi 0 value (0.0 : f32) : !sym.tensor<[2, 3], f32> -> !sym.tensor<[2, 4], f32>
  %1 = reloc.quantize %0 axis 1 scale(dense<[1.0, 0.5, 0.25, 0.125]> : tensor<4xf32>) policy symmetric_rne : !sym.tensor<[2, 4], f32> -> !sym.tensor<[2, 4], i8>
  return %1 : !sym.tensor<[2, 4], i8>
}

// A runtime (bound) scale cannot fold a fill at compile time.
// CHECK-LABEL: func.func @runtime_scale_after_pad_bails
func.func @runtime_scale_after_pad_bails(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.pad {{.*}} {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  // CHECK: reloc.quantize {{.*}} {reloc.fallback, reloc.fallback_reason = "fill_not_foldable"}
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %1 = reloc.quantize %0 scale(#reloc.binding<"s" : [], f32>) policy symmetric_rne : !sym.tensor<[8], f32> -> !sym.tensor<[8], i8>
  return %1 : !sym.tensor<[8], i8>
}

// A layout transfer function bails late (transposed merge is not
// contiguous): the earlier typed stage is rolled back with the chain, and the
// original quantize is exactly the op that was written.
// CHECK-LABEL: func.func @late_layout_bail_keeps_typed_chain
func.func @late_layout_bail_keeps_typed_chain(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.quantize %{{.*}} axis 0 scale(#reloc.binding<"s" : [4], f32>) policy symmetric_rne {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  // CHECK: reloc.reshape %{{.*}} to [24] {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  %0 = reloc.quantize %t axis 0 scale(#reloc.binding<"s" : [4], f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], i8>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[4, 6], i8> -> !sym.tensor<[6, 4], i8>
  %2 = reloc.reshape %1 to [24] : !sym.tensor<[6, 4], i8> -> !sym.tensor<[24], i8>
  return %2 : !sym.tensor<[24], i8>
}

// Pad-then-split on the padded axis bails after a quantize: the s8 pad is
// legal, the reshape is not (design decision 3), and the whole chain stays.
// CHECK-LABEL: func.func @pad_then_split_after_quantize_bails
func.func @pad_then_split_after_quantize_bails(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[2, 4], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.quantize %{{.*}} scale(dense<5.000000e-01> : tensor<f32>) policy symmetric_rne {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  // CHECK: reloc.pad %{{.*}} axis 0 lo 1 hi 1 value (1 : i8) {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  // CHECK: reloc.reshape %{{.*}} to [2, 4] {reloc.fallback, reloc.fallback_reason = "layout_bail"}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[6], f32> -> !sym.tensor<[6], i8>
  %1 = reloc.pad %0 axis 0 lo 1 hi 1 value (1 : i8) : !sym.tensor<[6], i8> -> !sym.tensor<[8], i8>
  %2 = reloc.reshape %1 to [2, 4] : !sym.tensor<[8], i8> -> !sym.tensor<[2, 4], i8>
  return %2 : !sym.tensor<[2, 4], i8>
}

// A non-reloc op interrupting the chain: the typed segment reports the
// structural reason; the layout-only segment keeps its v0 marking.
// CHECK-LABEL: func.func @interrupted_typed_chain
func.func @interrupted_typed_chain(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.quantize {{.*}} {reloc.fallback, reloc.fallback_reason = "structural"}
  // CHECK: "test.barrier"
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback} :
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], i8>
  %1 = "test.barrier"(%0) : (!sym.tensor<[4, 6], i8>) -> !sym.tensor<[4, 6], i8>
  %2 = reloc.transpose %1 perm [1, 0] : !sym.tensor<[4, 6], i8> -> !sym.tensor<[6, 4], i8>
  return %2 : !sym.tensor<[6, 4], i8>
}

// An escaping typed intermediate cannot be erased: structural bail.
// CHECK-LABEL: func.func @escaping_typed_intermediate
func.func @escaping_typed_intermediate(%t: !sym.tensor<[4, 6], f32>) -> (!sym.tensor<[6, 4], i8>, !sym.tensor<[4, 6], i8>) {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.quantize {{.*}} {reloc.fallback, reloc.fallback_reason = "structural"}
  // CHECK: reloc.transpose {{.*}} {reloc.fallback, reloc.fallback_reason = "structural"}
  %0 = reloc.quantize %t scale(dense<0.5> : tensor<f32>) policy symmetric_rne : !sym.tensor<[4, 6], f32> -> !sym.tensor<[4, 6], i8>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[4, 6], i8> -> !sym.tensor<[6, 4], i8>
  return %1, %0 : !sym.tensor<[6, 4], i8>, !sym.tensor<[4, 6], i8>
}

// A manual opt-out on a typed op marks the whole chain and records that the
// chain was pre-marked; a second run leaves the reason alone.
// CHECK-LABEL: func.func @manual_fallback_on_typed_op
func.func @manual_fallback_on_typed_op(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], i8> {
  // CHECK-NOT: reloc.typed_plan_result
  // CHECK: reloc.transpose {{.*}} {reloc.fallback, reloc.fallback_reason = "marked"}
  // CHECK: reloc.quantize {{.*}} {reloc.fallback, reloc.fallback_reason = "marked"}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.quantize %0 scale(dense<0.5> : tensor<f32>) policy symmetric_rne {reloc.fallback} : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], i8>
  return %1 : !sym.tensor<[6, 4], i8>
}
