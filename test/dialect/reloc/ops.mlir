// RUN: sym-opt %s | sym-opt | FileCheck %s

// CHECK-LABEL: func.func @transpose_static
func.func @transpose_static(%t: !sym.tensor<[64, 32], f32>) -> !sym.tensor<[32, 64], f32> {
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
  return %0 : !sym.tensor<[32, 64], f32>
}

// CHECK-LABEL: func.func @transpose_symbolic
func.func @transpose_symbolic(%t: !sym.tensor<["N", 64], f32>) -> !sym.tensor<[64, "N"], f32> {
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] : !sym.tensor<["N", 64], f32> -> !sym.tensor<[64, "N"], f32>
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<["N", 64], f32> -> !sym.tensor<[64, "N"], f32>
  return %0 : !sym.tensor<[64, "N"], f32>
}

// Non-self-inverse permutation: result dim k = operand dim perm[k].
// CHECK-LABEL: func.func @transpose_3d
func.func @transpose_3d(%t: !sym.tensor<[4, 5, 6], f32>) -> !sym.tensor<[6, 4, 5], f32> {
  // CHECK: reloc.transpose %{{.*}} perm [2, 0, 1] : !sym.tensor<[4, 5, 6], f32> -> !sym.tensor<[6, 4, 5], f32>
  %0 = reloc.transpose %t perm [2, 0, 1] : !sym.tensor<[4, 5, 6], f32> -> !sym.tensor<[6, 4, 5], f32>
  return %0 : !sym.tensor<[6, 4, 5], f32>
}

// CHECK-LABEL: func.func @reshape_static
func.func @reshape_static(%t: !sym.tensor<[4096], f32>) -> !sym.tensor<[64, 64], f32> {
  // CHECK: reloc.reshape %{{.*}} to [64, 64] : !sym.tensor<[4096], f32> -> !sym.tensor<[64, 64], f32>
  %0 = reloc.reshape %t to [64, 64] : !sym.tensor<[4096], f32> -> !sym.tensor<[64, 64], f32>
  return %0 : !sym.tensor<[64, 64], f32>
}

// Symbolic target with an unrelated symbol swapped in for one dim: the
// element count (N * 64 vs M * 64) is symbolically undecidable -> accepted.
// (sym's !sym.tensor type grammar cannot round-trip a binary-expression dim
// like "N floordiv 64" — a pre-existing sym limitation, out of scope here —
// so this exercises the same "undecidable count passes" path with symbol
// dims instead.)
// CHECK-LABEL: func.func @reshape_symbolic
func.func @reshape_symbolic(%t: !sym.tensor<["N", 64], f32>) -> !sym.tensor<["M", 64], f32> {
  // CHECK: reloc.reshape %{{.*}} to [M, 64] : !sym.tensor<["N", 64], f32> -> !sym.tensor<["M", 64], f32>
  %0 = reloc.reshape %t to [M, 64] : !sym.tensor<["N", 64], f32> -> !sym.tensor<["M", 64], f32>
  return %0 : !sym.tensor<["M", 64], f32>
}

// CHECK-LABEL: func.func @pad_static
func.func @pad_static(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f32> {
  // CHECK: reloc.pad %{{.*}} axis 0 lo 1 hi 1 value (1.000000e+00 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return %0 : !sym.tensor<[8], f32>
}

// The acceptance-criterion chain: transpose -> reshape -> pad in one function.
// CHECK-LABEL: func.func @chain
func.func @chain(%t: !sym.tensor<[64, 32], f32>) -> !sym.tensor<[8, 258], f32> {
  // CHECK: reloc.transpose
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
  // CHECK: reloc.reshape
  %1 = reloc.reshape %0 to [8, 256] : !sym.tensor<[32, 64], f32> -> !sym.tensor<[8, 256], f32>
  // CHECK: reloc.pad
  %2 = reloc.pad %1 axis 1 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[8, 256], f32> -> !sym.tensor<[8, 258], f32>
  return %2 : !sym.tensor<[8, 258], f32>
}

// plan_result: a folded chain materialized as a first-class value (#B4).
// CHECK-LABEL: func.func @plan_result_roundtrip
func.func @plan_result_roundtrip(%t: !sym.tensor<[8], i32>) -> !sym.tensor<[8], i32> {
  // CHECK: reloc.plan_result %{{.*}} plan(#reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = #map{{[0-9]*}}>) : !sym.tensor<[8], i32> -> !sym.tensor<[8], i32>
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[8], i32> -> !sym.tensor<[8], i32>
  return %0 : !sym.tensor<[8], i32>
}
