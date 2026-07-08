// RUN: sym-opt --test-reloc-transfer --verify-diagnostics %s

// The pass folds each straight-line transpose chain into a #reloc.plan and
// reports it as a remark on the chain tail. finalize() emits per-axis
// contiguity flags (src_stride == 1 proven); divisibility comes from
// reshape folds (#B2) and no_copy detection lands in #B5.

// Single 3-D transpose: dst axis k <- src axis perm[k]; src row-major
// strides [8, 2, 1] follow the axes through the permutation; dst strides
// are row-major over the transposed extents [2, 6, 4].
func.func @single_transpose(%t: !sym.tensor<[6, 4, 2], f32>) -> !sym.tensor<[2, 6, 4], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6, 4, 2], f32>, dst = tensor<[2, 6, 4], f32>, perm = [2, 0, 1], axes = [{name = "d2", extent = 2, src_stride = 1, dst_stride = 24}, {name = "d0", extent = 6, src_stride = 8, dst_stride = 4}, {name = "d1", extent = 4, src_stride = 2, dst_stride = 1}], constraints = {contiguous = [true, false, false], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d1, d2, d0)>>}}
  %0 = reloc.transpose %t perm [2, 0, 1] : !sym.tensor<[6, 4, 2], f32> -> !sym.tensor<[2, 6, 4], f32>
  return %0 : !sym.tensor<[2, 6, 4], f32>
}

// transpose o transpose folds to ONE plan with the composed permutation
// [2, 1, 0] (= [2,0,1] then [0,2,1]); the remark lands on the chain tail.
func.func @composed_transposes(%t: !sym.tensor<[6, 4, 2], f32>) -> !sym.tensor<[2, 4, 6], f32> {
  %0 = reloc.transpose %t perm [2, 0, 1] : !sym.tensor<[6, 4, 2], f32> -> !sym.tensor<[2, 6, 4], f32>
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6, 4, 2], f32>, dst = tensor<[2, 4, 6], f32>, perm = [2, 1, 0], axes = [{name = "d2", extent = 2, src_stride = 1, dst_stride = 24}, {name = "d1", extent = 4, src_stride = 2, dst_stride = 6}, {name = "d0", extent = 6, src_stride = 8, dst_stride = 1}], constraints = {contiguous = [true, false, false], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d2, d1, d0)>>}}
  %1 = reloc.transpose %0 perm [0, 2, 1] : !sym.tensor<[2, 6, 4], f32> -> !sym.tensor<[2, 4, 6], f32>
  return %1 : !sym.tensor<[2, 4, 6], f32>
}

// Identity elision: transpose o transpose^-1 composes to the identity
// permutation and an identity inverse map (#B5 turns this into no_copy).
func.func @inverse_pair(%t: !sym.tensor<[6, 4, 2], f32>) -> !sym.tensor<[6, 4, 2], f32> {
  %0 = reloc.transpose %t perm [2, 0, 1] : !sym.tensor<[6, 4, 2], f32> -> !sym.tensor<[2, 6, 4], f32>
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6, 4, 2], f32>, dst = tensor<[6, 4, 2], f32>, perm = [0, 1, 2], axes = [{name = "d0", extent = 6, src_stride = 8, dst_stride = 8}, {name = "d1", extent = 4, src_stride = 2, dst_stride = 2}, {name = "d2", extent = 2, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d0, d1, d2)>>}}
  %1 = reloc.transpose %0 perm [1, 2, 0] : !sym.tensor<[2, 6, 4], f32> -> !sym.tensor<[6, 4, 2], f32>
  return %1 : !sym.tensor<[6, 4, 2], f32>
}

// Symbolic extent N is preserved verbatim through the fold: it moves with
// its axis and reappears in the dst extents and the symbolic stride 2 * N.
func.func @symbolic_extents(%t: !sym.tensor<[6, "N", 2], f32>) -> !sym.tensor<["N", 6, 2], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[6, N, 2], f32>, dst = tensor<[N, 6, 2], f32>, perm = [1, 0, 2], axes = [{name = "d1", extent = N, src_stride = 2, dst_stride = 12}, {name = "d0", extent = 6, src_stride = 2 * N, dst_stride = 2}, {name = "d2", extent = 2, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d1, d0, d2)>>}}
  %0 = reloc.transpose %t perm [1, 0, 2] : !sym.tensor<[6, "N", 2], f32> -> !sym.tensor<["N", 6, 2], f32>
  return %0 : !sym.tensor<["N", 6, 2], f32>
}

// A transpose feeding another reloc op is not a chain tail on its own: no
// remark on %0 (verify-diagnostics fails on any unexpected remark). Since
// #B2 the chain continues into the reshape tail, which bails here for the
// same non-contiguous-merge reason as fold_bail.mlir.
func.func @not_a_tail(%t: !sym.tensor<[4, 2], f32>) -> !sym.tensor<[8], f32> {
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[4, 2], f32> -> !sym.tensor<[2, 4], f32>
  // expected-remark @below {{fold bail: reloc.reshape}}
  %1 = reloc.reshape %0 to [8] : !sym.tensor<[2, 4], f32> -> !sym.tensor<[8], f32>
  return %1 : !sym.tensor<[8], f32>
}
