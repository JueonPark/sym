// RUN: sym-opt --test-reloc-transfer --verify-diagnostics %s

// NOTE: symbolic-split result types are writable since issue #31; the
// C++ oracle tests in unittest/reloc/PlanBuilderTest.cpp remain the
// semantic authority (they bind N and compare index tables).

// Static split: [4096] -> [64, 64]; strides peel right-to-left.
func.func @static_split(%t: !sym.tensor<[4096], f32>) -> !sym.tensor<[64, 64], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[4096], f32>, dst = tensor<[64, 64], f32>, perm = [0, 1], axes = [{name = "d0", extent = 64, src_stride = 64, dst_stride = 64}, {name = "d1", extent = 64, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, true], no_copy = false}, inverse = affine_map<(d0, d1) -> (d0, d1)>>}}
  %0 = reloc.reshape %t to [64, 64] : !sym.tensor<[4096], f32> -> !sym.tensor<[64, 64], f32>
  return %0 : !sym.tensor<[64, 64], f32>
}

// Static merge: row-major adjacent axes are contiguous-compatible.
func.func @static_merge(%t: !sym.tensor<[8, 128], f32>) -> !sym.tensor<[1024], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[1024], f32>, perm = [0], axes = [{name = "d0", extent = 1024, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>}}
  %0 = reloc.reshape %t to [1024] : !sym.tensor<[8, 128], f32> -> !sym.tensor<[1024], f32>
  return %0 : !sym.tensor<[1024], f32>
}

// m:n regroup: [4, 6] -> [2, 12] is merge-then-split in one group.
func.func @regroup(%t: !sym.tensor<[4, 6], f32>) -> !sym.tensor<[2, 12], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[4, 6], f32>, dst = tensor<[2, 12], f32>, perm = [0, 1], axes = [{name = "d0", extent = 2, src_stride = 12, dst_stride = 12}, {name = "d1", extent = 12, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, true], no_copy = false}, inverse = affine_map<(d0, d1) -> (d0, d1)>>}}
  %0 = reloc.reshape %t to [2, 12] : !sym.tensor<[4, 6], f32> -> !sym.tensor<[2, 12], f32>
  return %0 : !sym.tensor<[2, 12], f32>
}

// Reshape then transpose folds into ONE plan; the reshape re-baselines the
// view, the transpose composes onto it.
func.func @reshape_then_transpose(%t: !sym.tensor<[4096], f32>) -> !sym.tensor<[64, 64], f32> {
  %0 = reloc.reshape %t to [64, 64] : !sym.tensor<[4096], f32> -> !sym.tensor<[64, 64], f32>
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[4096], f32>, dst = tensor<[64, 64], f32>, perm = [1, 0], axes = [{name = "d1", extent = 64, src_stride = 1, dst_stride = 64}, {name = "d0", extent = 64, src_stride = 64, dst_stride = 1}], constraints = {contiguous = [true, false], no_copy = false}, inverse = affine_map<(d0, d1) -> (d1, d0)>>}}
  %1 = reloc.transpose %0 perm [1, 0] : !sym.tensor<[64, 64], f32> -> !sym.tensor<[64, 64], f32>
  return %1 : !sym.tensor<[64, 64], f32>
}

// Symbol passthrough (1:1 keep) plus a static merge in the same reshape.
func.func @symbolic_passthrough_merge(%t: !sym.tensor<["N", 4, 2], f32>) -> !sym.tensor<["N", 8], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[N, 4, 2], f32>, dst = tensor<[N, 8], f32>, perm = [0, 1], axes = [{name = "d0", extent = N, src_stride = 8, dst_stride = 8}, {name = "d1", extent = 8, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [false, true], no_copy = false}, inverse = affine_map<(d0, d1) -> (d0, d1)>>}}
  %0 = reloc.reshape %t to [N, 8] : !sym.tensor<["N", 4, 2], f32> -> !sym.tensor<["N", 8], f32>
  return %0 : !sym.tensor<["N", 8], f32>
}

// Symbolic split emits the divisibility constraint — in lit for the
// first time (issue #31 unblocked the result type).
func.func @symbolic_split(%t: !sym.tensor<["N", 4], f32>) -> !sym.tensor<["N" floordiv 8, 8, 4], f32> {
  // expected-remark @below {{folded plan: #reloc.plan<src = tensor<[N, 4], f32>, dst = tensor<[N floordiv 8, 8, 4], f32>, perm = [0, 1, 2], axes = [{name = "d0", extent = N floordiv 8, src_stride = 32, dst_stride = 32}, {name = "d1", extent = 8, src_stride = 4, dst_stride = 4}, {name = "d2", extent = 4, src_stride = 1, dst_stride = 1}], constraints = {divisible(N, 8), contiguous = [false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2) -> (d0, d1, d2)>>}}
  %0 = reloc.reshape %t to [N floordiv 8, 8, 4] : !sym.tensor<["N", 4], f32> -> !sym.tensor<["N" floordiv 8, 8, 4], f32>
  return %0 : !sym.tensor<["N" floordiv 8, 8, 4], f32>
}
