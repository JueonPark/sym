// RUN: sym-opt --allow-unregistered-dialect --reloc-fold %s | FileCheck %s
// Idempotence: a second run must be a no-op on the first run's output.
// RUN: sym-opt --allow-unregistered-dialect --reloc-fold %s | sym-opt --allow-unregistered-dialect --reloc-fold | FileCheck %s

// The static analogue of the build-doc reference chain: 2D -> 4D blocked
// view -> transpose folds to ONE plan_result; the chain ops disappear.
// #B5 canonicalization collapses the plan's rank-2 dst back down to the
// two axes that actually vary independently; the op's own result type
// stays the original (uncollapsed) [2, 2, 64, 64], which is legal per the
// plan_result rank relaxation.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-LABEL: func.func @full_chain
func.func @full_chain(%t: !sym.tensor<[128, 128], f32>) -> !sym.tensor<[2, 2, 64, 64], f32> {
  // CHECK-NOT: reloc.reshape
  // CHECK-NOT: reloc.transpose
  // CHECK: %[[R:.*]] = reloc.plan_result %{{.*}} plan(#reloc.plan<src = tensor<[128, 128], f32>, dst = tensor<[2, 128, 64], f32>, perm = [1, 0, 2], axes = [{name = "d0", extent = 2, src_stride = 64, dst_stride = 8192}, {name = "d1", extent = 128, src_stride = 128, dst_stride = 64}, {name = "d2", extent = 64, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, false, true], no_copy = false}, inverse = #map>) : !sym.tensor<[128, 128], f32> -> !sym.tensor<[2, 2, 64, 64], f32>
  // CHECK: return %[[R]]
  %0 = reloc.reshape %t to [2, 64, 2, 64] : !sym.tensor<[128, 128], f32> -> !sym.tensor<[2, 64, 2, 64], f32>
  %1 = reloc.transpose %0 perm [2, 0, 1, 3] : !sym.tensor<[2, 64, 2, 64], f32> -> !sym.tensor<[2, 2, 64, 64], f32>
  return %1 : !sym.tensor<[2, 2, 64, 64], f32>
}

// A single op is a complete chain.
// CHECK-LABEL: func.func @single_op_chain
func.func @single_op_chain(%t: !sym.tensor<[6], f32>) -> !sym.tensor<[8], f32> {
  // CHECK-NOT: reloc.pad
  // CHECK: reloc.plan_result %{{.*}} plan(#reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 0.000000e+00 : f32}], constraints = {contiguous = [true], no_copy = false}, inverse = #map{{[0-9]+}}>) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return %0 : !sym.tensor<[8], f32>
}

// Transfer-function bail (non-contiguous merge): all-or-nothing — BOTH ops
// stay and BOTH are marked reloc.fallback; no plan_result is created.
// CHECK-LABEL: func.func @transfer_bail
func.func @transfer_bail(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], f32> {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: reloc.reshape %{{.*}} to [24] {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.reshape %0 to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  return %1 : !sym.tensor<[24], f32>
}

// Multi-use INTERMEDIATE value: bail. The two transposes compose to the
// identity, so the fold itself would succeed -- the bail provably comes
// from the multi-use rule, and erasing %0 would otherwise be invalid.
// CHECK-LABEL: func.func @multiuse_intermediate_bails
func.func @multiuse_intermediate_bails(%t: !sym.tensor<[4, 6], f32>) -> (!sym.tensor<[6, 4], f32>, !sym.tensor<[4, 6], f32>) {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[4, 6], f32>
  return %0, %1 : !sym.tensor<[6, 4], f32>, !sym.tensor<[4, 6], f32>
}

// A tail whose result has several (non-reloc) uses still folds: every use
// is replaced by the plan_result value.
// CHECK-LABEL: func.func @tail_multiuse_folds
func.func @tail_multiuse_folds(%t: !sym.tensor<[4, 6], f32>) -> (!sym.tensor<[6, 4], f32>, !sym.tensor<[6, 4], f32>) {
  // CHECK: %[[P:.*]] = reloc.plan_result
  // CHECK: return %[[P]], %[[P]]
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  return %0, %0 : !sym.tensor<[6, 4], f32>, !sym.tensor<[6, 4], f32>
}

// Non-reloc op interrupting a reloc pipeline: both segments bail.
// CHECK-LABEL: func.func @interrupted_chain_bails
func.func @interrupted_chain_bails(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], f32> {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: "test.barrier"
  // CHECK: reloc.reshape %{{.*}} to [24] {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = "test.barrier"(%0) : (!sym.tensor<[6, 4], f32>) -> !sym.tensor<[6, 4], f32>
  %2 = reloc.reshape %1 to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  return %2 : !sym.tensor<[24], f32>
}

// A pre-existing reloc.fallback marker is a manual opt-out: the chain is
// left untouched even though it would fold.
// CHECK-LABEL: func.func @manual_fallback_respected
func.func @manual_fallback_respected(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[6, 4], f32> {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] {reloc.fallback} : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  return %0 : !sym.tensor<[6, 4], f32>
}

// Mark-on-skip: a diamond off a multi-use intermediate. The transpose ->
// reshape chain is processed first and bails on the multi-use rule, marking
// the transpose and the reshape. The transpose -> pad chain is processed
// second; its root (the transpose) is already marked, so mark-on-skip marks
// the whole second chain too, landing the pad. All three ops end up marked
// -- none are left in a third state that is neither folded nor marked.
// CHECK-LABEL: func.func @diamond_marks_all
func.func @diamond_marks_all(%t: !sym.tensor<[4, 6], f32>) -> (!sym.tensor<[24], f32>, !sym.tensor<[8, 4], f32>) {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: reloc.reshape %{{.*}} to [24] {reloc.fallback}
  // CHECK: reloc.pad %{{.*}} axis 0 lo 1 hi 1 value{{.*}} {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.reshape %0 to [24] : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  %2 = reloc.pad %0 axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6, 4], f32> -> !sym.tensor<[8, 4], f32>
  return %1, %2 : !sym.tensor<[24], f32>, !sym.tensor<[8, 4], f32>
}

// Mark-on-skip: a clean transpose feeds a reshape that already carries a
// manual reloc.fallback opt-out. The reshape is the chain tail (its user
// is the func.return, not a foldable op), so the chain is [transpose,
// reshape]; the pre-marked reshape triggers the skip path, and mark-on-skip
// marks the transpose too, even though the transpose itself was clean.
// CHECK-LABEL: func.func @marked_consumer_marks_chain
func.func @marked_consumer_marks_chain(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[24], f32> {
  // CHECK-NOT: reloc.plan_result
  // CHECK: reloc.transpose %{{.*}} perm [1, 0] {reloc.fallback}
  // CHECK: reloc.reshape %{{.*}} to [24] {reloc.fallback}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[6, 4], f32>
  %1 = reloc.reshape %0 to [24] {reloc.fallback} : !sym.tensor<[6, 4], f32> -> !sym.tensor<[24], f32>
  return %1 : !sym.tensor<[24], f32>
}
