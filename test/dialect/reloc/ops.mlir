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
