// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s

// The `canonicalize` unit attr routes a hand-written plan through
// canonicalizePlan and reports the result as a remark.

// Both-side contiguous + view-adjacent: axes merge, rank collapses, and
// the result is a pure view -> no_copy is set.
// expected-remark @below {{canonicalized: #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[1024], f32>, perm = [0], axes = [{name = "d0", extent = 1024, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = true}, inverse = affine_map<(d0) -> (d0)>>}}
"test.plan"() {canonicalize, plan = #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[8, 128], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = 128, dst_stride = 128}, {name = "i", extent = 128, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d0, d1)>>} : () -> ()

// Transposed pure view with EXPLICIT dst strides: the explicit strides put
// the plan outside fold-normal form, so the safe subset runs (no axis
// merging). Axes are view-adjacent under perm [1, 0] but never merge; axes
// are renamed d0/d1 and no_copy is recomputed to true (src/dst strides
// realign per axis -> it IS a pure view). The only case exercising the
// safe subset, view-adjacency, and no_copy recompute together.
// expected-remark @below {{canonicalized: #reloc.plan<src = tensor<[64, 32], f32, strides = [32, 1]>, dst = tensor<[32, 64], f32, strides = [1, 32]>, perm = [1, 0], axes = [{name = "d0", extent = 32, src_stride = 1, dst_stride = 1}, {name = "d1", extent = 64, src_stride = 32, dst_stride = 32}], constraints = {contiguous = [true, false], no_copy = true}, inverse = affine_map<(d0, d1) -> (d1, d0)>>}}
"test.plan"() {canonicalize, plan = #reloc.plan<src = tensor<[64, 32], f32, strides = [32, 1]>, dst = tensor<[32, 64], f32, strides = [1, 32]>, perm = [1, 0], axes = [{name = "c", extent = 32, src_stride = 1, dst_stride = 1}, {name = "r", extent = 64, src_stride = 32, dst_stride = 32}], inverse = affine_map<(d0, d1) -> (d1, d0)>>} : () -> ()

// A padded axis never merges, and a pad plan is never a pure view.
// expected-remark @below {{canonicalized: #reloc.plan<src = tensor<[6, 4], f32>, dst = tensor<[8, 4], f32>, perm = [0, 1], axes = [{name = "d0", extent = 6, src_stride = 4, dst_stride = 4}, {name = "d1", extent = 4, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 0.000000e+00 : f32}], constraints = {contiguous = [false, true], no_copy = false}, inverse = affine_map<(d0, d1) -> (d0, d1)>>}}
"test.plan"() {canonicalize, plan = #reloc.plan<src = tensor<[6, 4], f32>, dst = tensor<[8, 4], f32>, perm = [0, 1], axes = [{name = "a", extent = 6, src_stride = 4, dst_stride = 4}, {name = "b", extent = 4, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 0.0 : f32}], inverse = affine_map<(d0, d1) -> (d0, d1)>>} : () -> ()

// A wrong no_copy = true on a data-moving plan is cleared.
// expected-remark @below {{canonicalized: #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "d0", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.000000e+00 : f32}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>}}
"test.plan"() {canonicalize, plan = #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "x", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}], constraints = {no_copy = true}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()
