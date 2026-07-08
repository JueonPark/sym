// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s | FileCheck %s

// A3 verifier positive cases. `rebuild_no_flag` makes the test pass
// reconstruct the plan programmatically with runtime_pad_check = false,
// exercising the C++-builder enforcement path.

// 1. Symbolically PROVABLE pad range: extent N, pads 0/2, dst extent N + 2.
//    The flag is NOT auto-set; the plan prints without it, and the
//    flagless rebuild succeeds.
// CHECK: pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.000000e+00 : f32}], constraints = {align(0, 64), no_copy = false}, inverse
// expected-remark @below {{rebuild without runtime_pad_check: ok}}
"test.plan"() {rebuild_no_flag, plan = #reloc.plan<src = tensor<[N], f32>, dst = tensor<[N + 2], f32>, perm = [0], axes = [{name = "x", extent = N, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}], constraints = {align(0, 64), no_copy = false}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// 2. UNDECIDABLE pad range: hi rounds N up to the next multiple of 64.
//    Parse auto-derives runtime_pad_check (degradation), and the
//    programmatic rebuild with the flag forced false is rejected.
// CHECK: constraints = {no_copy = false, runtime_pad_check}
// expected-error @below {{pad range for dst_axis 0 is not provable; set runtime_pad_check}}
"test.plan"() {rebuild_no_flag, plan = #reloc.plan<src = tensor<[N], f32>, dst = tensor<[64 * ((N + 63) floordiv 64)], f32>, perm = [0], axes = [{name = "x", extent = N, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = (64 * ((N + 63) floordiv 64)) - N, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// 3. An explicitly written runtime_pad_check keyword round-trips. No
//    rebuild_no_flag/pair attribute is set, so the pass falls through to
//    its default isPureView remark (this trivial 8 -> 8 plan is a pure
//    view: no pad_fill, matching strides, equal offsets).
// CHECK: constraints = {no_copy = true, runtime_pad_check}
// expected-remark @below {{isPureView = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], constraints = {no_copy = true, runtime_pad_check}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// 4. Non-self-inverse permutation: perm = [2, 0, 1] requires
//    inverse = inversePermutation((d0,d1,d2) -> (d2,d0,d1)) = (d0,d1,d2) -> (d1,d2,d0).
// CHECK: perm = [2, 0, 1]
// expected-remark @below {{isPureView = false}}
"test.plan"() {plan = #reloc.plan<src = tensor<[4, 5, 6], f32>, dst = tensor<[6, 4, 5], f32>, perm = [2, 0, 1], axes = [{name = "a", extent = 6, src_stride = 1, dst_stride = 20}, {name = "b", extent = 4, src_stride = 30, dst_stride = 5}, {name = "c", extent = 5, src_stride = 6, dst_stride = 1}], inverse = affine_map<(d0, d1, d2) -> (d1, d2, d0)>>} : () -> ()

// 5. Floordiv inverse: not a permutation map, so only arity is checked.
// The inverse map itself (a floordiv expression) is hoisted by the
// generic printer into a top-of-file #map alias, so it cannot be
// matched here in-order; check the op's own attributes instead.
// CHECK: perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = #map2
// expected-remark @below {{isPureView = true}}
"test.plan"() {plan = #reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0 floordiv 2 + d0 mod 2)>>} : () -> ()
