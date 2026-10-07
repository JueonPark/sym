// RUN: sym-opt --reloc-fold %s | FileCheck %s
// RUN: sym-opt --reloc-fold %s | sym-opt --reloc-fold | FileCheck %s

// Both SSA operands survive folding; selection is not an affine view.
// CHECK-LABEL: func.func @identity
func.func @identity(%x: !sym.tensor<[8, 5], f32>, %i: !sym.tensor<[3], i64>) -> !sym.tensor<[3, 5], f32> {
  // CHECK: reloc.indexed_plan_result %arg0, %arg1
  // CHECK-SAME: #reloc.indexed_plan<
  // CHECK-SAME: "exact"
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<[8, 5], f32>, !sym.tensor<[3], i64> -> !sym.tensor<[3, 5], f32>
  return %0 : !sym.tensor<[3, 5], f32>
}

// CHECK-LABEL: func.func @symbolic_cast
func.func @symbolic_cast(%x: !sym.tensor<["N", "D"], f32>, %i: !sym.tensor<["M"], i32>) -> !sym.tensor<["M", "D"], f16> {
  // CHECK: reloc.indexed_plan_result %arg0, %arg1
  // CHECK-SAME: "ieee_rne"
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<["N", "D"], f32>, !sym.tensor<["M"], i32> -> !sym.tensor<["M", "D"], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<["M", "D"], f32> -> !sym.tensor<["M", "D"], f16>
  return %1 : !sym.tensor<["M", "D"], f16>
}

// Unsupported compositions remain intact, including on a second pass.
// CHECK-LABEL: func.func @layout_after_select
func.func @layout_after_select(%x: !sym.tensor<[8, 5], f32>, %i: !sym.tensor<[3], i64>) -> !sym.tensor<[15], f32> {
  // CHECK: reloc.index_select
  // CHECK-SAME: reloc.fallback
  // CHECK: reloc.reshape
  // CHECK-SAME: reloc.fallback
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<[8, 5], f32>, !sym.tensor<[3], i64> -> !sym.tensor<[3, 5], f32>
  %1 = reloc.reshape %0 to [15] : !sym.tensor<[3, 5], f32> -> !sym.tensor<[15], f32>
  return %1 : !sym.tensor<[15], f32>
}

// CHECK-LABEL: func.func @escaping_select
func.func @escaping_select(%x: !sym.tensor<[8, 5], f32>, %i: !sym.tensor<[3], i64>) -> (!sym.tensor<[3, 5], f32>, !sym.tensor<[3, 5], f16>) {
  // CHECK: reloc.index_select
  // CHECK-SAME: reloc.fallback
  // CHECK: reloc.cast
  // CHECK-SAME: reloc.fallback_reason = "structural"
  %0 = reloc.index_select %x, %i axis 0 : !sym.tensor<[8, 5], f32>, !sym.tensor<[3], i64> -> !sym.tensor<[3, 5], f32>
  %1 = reloc.cast %0 policy ieee_rne : !sym.tensor<[3, 5], f32> -> !sym.tensor<[3, 5], f16>
  return %0, %1 : !sym.tensor<[3, 5], f32>, !sym.tensor<[3, 5], f16>
}
