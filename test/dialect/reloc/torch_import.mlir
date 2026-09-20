// RUN: sym-opt --reloc-fold %s | FileCheck %s

// The logical result keeps both axes independently of plan canonicalization.
// CHECK-LABEL: func.func @torch_split_transpose
// CHECK: -> !sym.tensor<[64, N floordiv 64], f32>
// CHECK-NOT: reloc.reshape
// CHECK: reloc.plan_result
// CHECK-SAME: divisible(N, 64)
// CHECK-NOT: reloc.plan_result
// CHECK: return {{.*}} : !sym.tensor<[64, N floordiv 64], f32>
func.func @torch_split_transpose(%x: !sym.tensor<["N"], f32>)
    -> !sym.tensor<[64, "N" floordiv 64], f32> {
  %r = reloc.reshape %x to [N floordiv 64, 64]
      : !sym.tensor<["N"], f32> -> !sym.tensor<["N" floordiv 64, 64], f32>
  %t = reloc.transpose %r perm [1, 0]
      : !sym.tensor<["N" floordiv 64, 64], f32> -> !sym.tensor<[64, "N" floordiv 64], f32>
  return %t : !sym.tensor<[64, "N" floordiv 64], f32>
}

// CHECK-LABEL: func.func @noncontiguous_merge
// CHECK-NOT: reloc.plan_result
// CHECK: reloc.transpose {{.*}} perm [1, 0] {reloc.fallback}
// CHECK: reloc.reshape {{.*}} to [24] {reloc.fallback}
// CHECK-NOT: reloc.plan_result
// CHECK: return
func.func @noncontiguous_merge(%x: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], f32> {
  %t = reloc.transpose %x perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %r = reloc.reshape %t to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  return %r : !sym.tensor<[24], f32>
}

// CHECK-LABEL: func.func @pad_split
// CHECK-NOT: reloc.plan_result
// CHECK: reloc.pad {{.*}} axis 0 lo 1 hi 1 value {{.*}} {reloc.fallback}
// CHECK: reloc.reshape {{.*}} to [2, 4] {reloc.fallback}
// CHECK-NOT: reloc.plan_result
// CHECK: return
func.func @pad_split(%x: !sym.tensor<[6], f32>) -> !sym.tensor<[2, 4], f32> {
  %p = reloc.pad %x axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %r = reloc.reshape %p to [2, 4] : !sym.tensor<[8], f32> -> !sym.tensor<[2, 4], f32>
  return %r : !sym.tensor<[2, 4], f32>
}

// CHECK-LABEL: func.func @conflicting_fills
// CHECK-NOT: reloc.plan_result
// CHECK: reloc.pad {{.*}} value (0.000000e+00 : f32) {reloc.fallback}
// CHECK: reloc.pad {{.*}} value (1.000000e+00 : f32) {reloc.fallback}
// CHECK-NOT: reloc.plan_result
// CHECK: return
func.func @conflicting_fills(%x: !sym.tensor<[4], f32>) -> !sym.tensor<[8], f32> {
  %a = reloc.pad %x axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[4], f32> -> !sym.tensor<[6], f32>
  %b = reloc.pad %a axis 0 lo 1 hi 1 value (1.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return %b : !sym.tensor<[8], f32>
}

// CHECK-LABEL: func.func @escaping_intermediate
// CHECK-NOT: reloc.plan_result
// CHECK: reloc.reshape {{.*}} {reloc.fallback}
// CHECK: reloc.transpose {{.*}} {reloc.fallback}
// CHECK-NOT: reloc.plan_result
// CHECK: return
func.func @escaping_intermediate(%x: !sym.tensor<[128], f32>) -> (!sym.tensor<[2, 64], f32>, !sym.tensor<[64, 2], f32>) {
  %r = reloc.reshape %x to [2, 64] : !sym.tensor<[128], f32> -> !sym.tensor<[2, 64], f32>
  %t = reloc.transpose %r perm [1, 0] : !sym.tensor<[2, 64], f32> -> !sym.tensor<[64, 2], f32>
  return %r, %t : !sym.tensor<[2, 64], f32>, !sym.tensor<[64, 2], f32>
}
