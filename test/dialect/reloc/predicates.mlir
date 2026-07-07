// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s

// isContiguousCompatible(outer, inner) is true iff
//   outer.src_stride == inner.src_stride * inner.extent
// provably under sym simplification. `pair = array<i64: i, j>` selects
// (outer, inner) = (axes[i], axes[j]).

// 1. Static, true: 128 == 1 * 128.
// expected-remark @below {{isContiguousCompatible(0, 1) = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[1024], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = 128, dst_stride = 128}, {name = "i", extent = 128, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0 floordiv 128, d0 mod 128)>>, pair = array<i64: 0, 1>} : () -> ()

// 2. Static, false: 64 != 1 * 32.
// expected-remark @below {{isContiguousCompatible(0, 1) = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8, 32], f32>, dst = tensor<[256], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = 64, dst_stride = 32}, {name = "i", extent = 32, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>>, pair = array<i64: 0, 1>} : () -> ()

// 3. Symbolic, true: N == 1 * N.
// expected-remark @below {{isContiguousCompatible(0, 1) = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[N, N], f32>, dst = tensor<[N * N], f32>, perm = [0, 1], axes = [{name = "o", extent = N, src_stride = N, dst_stride = N}, {name = "i", extent = N, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>>, pair = array<i64: 0, 1>} : () -> ()

// 4. Symbolic, true via commutativity: 64 * N == N * 64.
// expected-remark @below {{isContiguousCompatible(0, 1) = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[N, 64], f32>, dst = tensor<[N * 64], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = 64 * N, dst_stride = 1}, {name = "i", extent = 64, src_stride = N, dst_stride = 1}], inverse = affine_map<(d0) -> (d0, d0)>>, pair = array<i64: 0, 1>} : () -> ()

// 5. Symbolic, false: N != 1 * (N floordiv 64).
// expected-remark @below {{isContiguousCompatible(0, 1) = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[N, N], f32>, dst = tensor<[N], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = N, dst_stride = 1}, {name = "i", extent = N floordiv 64, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0, d0)>>, pair = array<i64: 0, 1>} : () -> ()

// 6. Symbolic, false: N + 1 != 1 * N.
// expected-remark @below {{isContiguousCompatible(0, 1) = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[N, N], f32>, dst = tensor<[N], f32>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = N + 1, dst_stride = 1}, {name = "i", extent = N, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0, d0)>>, pair = array<i64: 0, 1>} : () -> ()

// Out-of-range pair indices are reported.
// expected-remark @below {{isContiguousCompatible: pair index out of range}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, pair = array<i64: 0, 5>} : () -> ()

// isPureView: no pad_fill, every axis dst_stride == src_stride, offsets
// equal.

// Pure view: strided transpose relabeling.
// expected-remark @below {{isPureView = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[64, 32], f32>, dst = tensor<[32, 64], f32>, perm = [1, 0], axes = [{name = "r", extent = 64, src_stride = 32, dst_stride = 32}, {name = "c", extent = 32, src_stride = 1, dst_stride = 1}], constraints = {no_copy = true}, inverse = affine_map<(d0, d1) -> (d1, d0)>>} : () -> ()

// Not a pure view: axis strides differ (materializing transpose).
// expected-remark @below {{isPureView = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[64, 32], f32>, dst = tensor<[32, 64], f32>, perm = [1, 0], axes = [{name = "r", extent = 64, src_stride = 32, dst_stride = 1}, {name = "c", extent = 32, src_stride = 1, dst_stride = 64}], inverse = affine_map<(d0, d1) -> (d1, d0)>>} : () -> ()

// Not a pure view: padding present.
// expected-remark @below {{isPureView = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "x", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// Not a pure view: offsets differ.
// expected-remark @below {{isPureView = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8], f32, offset = 4>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
