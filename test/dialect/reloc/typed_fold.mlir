// RUN: sym-opt --reloc-fold %s | FileCheck %s

// C1 handoff to C2: a typed value transform is not a layout-chain member.
// The fold pass leaves it in place and treats it as a chain boundary; typed
// stages fold into the plan only once C2 defines how value transforms
// compose with transpose/reshape/pad (docs/reloc-typed-semantics.md, "Order
// and composition").

// A layout prefix folds to a plan on its own; the value transform consumes
// the materialized plan result and keeps its channel axis and parameters.
// CHECK-LABEL: func.func @layout_prefix_then_quantize
func.func @layout_prefix_then_quantize(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], i8> {
  // CHECK: %[[PLAN:.*]] = reloc.plan_result %{{.*}} plan(#reloc.plan<src = tensor<[4, 6], f32>, dst = tensor<[6, 4], f32>
  // CHECK: reloc.quantize %[[PLAN]] axis 0 scale(#reloc.binding<"row_scale" : [6], f32>) policy symmetric_rne : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], i8>
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.quantize %0 axis 0 scale(#reloc.binding<"row_scale" : [6], f32>) policy symmetric_rne : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], i8>
  return %1 : !sym.tensor<[6, 4], i8>
}

// A cast between two layout segments interrupts both (all-or-nothing per
// chain): each segment is marked for fallback and the cast is untouched.
// CHECK-LABEL: func.func @cast_between_layout_segments
func.func @cast_between_layout_segments(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], f16> {
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: reloc.cast %{{.*}} policy ieee_rne
  // CHECK: reloc.transpose %{{.*}} perm [0, 1] {reloc.fallback}
  // CHECK: return
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<[6, 4], f32> -> !sym.tensor<[6, 4], f16>
  %2 = reloc.transpose %1 perm [0, 1] : !sym.tensor<[6, 4], f16> -> !sym.tensor<[6, 4], f16>
  return %2 : !sym.tensor<[6, 4], f16>
}

// A standalone value transform has no layout chain to fold and is unchanged.
// CHECK-LABEL: func.func @standalone_dequantize
func.func @standalone_dequantize(%q: !sym.tensor<["N"], i8>) -> !sym.tensor<["N"], f32> {
  // CHECK: reloc.dequantize %{{.*}} scale(dense<2.500000e-01> : tensor<f32>) policy affine
  %0 = reloc.dequantize %q scale(dense<0.25> : tensor<f32>) policy affine : !sym.tensor<["N"], i8> -> !sym.tensor<["N"], f32>
  return %0 : !sym.tensor<["N"], f32>
}
