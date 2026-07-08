// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s | FileCheck %s

// Golden wire-format tests. The hex lines below are the committed golden
// blobs: any encoder change that alters the byte stream fails FileCheck,
// which is the cross-build determinism guarantee. Structural properties
// (size, in-process determinism, the E9 <4KB budget) are asserted via
// remarks. Golden hex is pinned in this file by Task 3.

// The worked example from docs/reloc-plan-format.md: 1-D identity plan.
// CHECK: plan_hex(identity): 52504c4e000000000000000001000000010000000108000000000000000000000001000000010000000000000000022000000001000000010000000108000000000000000000000001000000010000000000000000022000000001000000000000000100000001000000780100000001080000000000000001000000010100000000000000010000000101000000000000000000000000000000000000000000000000000100000001000000010000000700000000
// expected-remark @below {{encoded 181 bytes, deterministic = true, within_size_budget(4096) = true}}
"test.plan"() {serialize, name = "identity", plan = #reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// Reference plan (build doc §2.1): symbolic N, divisibility, contiguity.
// CHECK: plan_hex(reference): 52504c4e0000000001000000010000004e020000000100000000000000000100000000000000000200000001000000000000000001000000010100000000000000010000000100000000000000000020000000040000000300000000000000000140000000000000000501000000014000000000000000010000000140000000000000000300000000000000000140000000000000000500000000010000000100000000000000000020000000040000000100000000000000020000000300000004000000020000006e3003000000000000000001400000000000000005010000000140000000000000000500000001001000000000000000000000000140000000000000000504020000006230010000000140000000000000000300000001400000000000000000000000000405000000014000000000000000000000000001400000000000000005040200000062310100000001400000000000000001000000000000000003000000000000000001400000000000000005020000006e310300000000000000000140000000000000000501000000010100000000000000010000000101000000000000000000000001000000010000000000000000400000000000000000000000040000000000000100000400000004000000010000000701000000010000000700000000010000000702000000010000000703000000
// expected-remark @below {{encoded 512 bytes, deterministic = true, within_size_budget(4096) = true}}
"test.plan"() {serialize, name = "reference", plan = #reloc.plan<src = tensor<[N, N], f32, strides = [N, 1]>, dst = tensor<[N floordiv 64, 64, 64, N floordiv 64], f32>, perm = [1, 0, 2, 3], axes = [{name = "n0", extent = N floordiv 64, src_stride = 64, dst_stride = 4096 * (N floordiv 64)}, {name = "b0", extent = 64, src_stride = 64 * N, dst_stride = 64 * (N floordiv 64)}, {name = "b1", extent = 64, src_stride = N, dst_stride = N floordiv 64}, {name = "n1", extent = N floordiv 64, src_stride = 1, dst_stride = 1}], constraints = {divisible(N, 64), contiguous = [false, false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>>} : () -> ()

// Pad plan: exercises typed_value (f32 fill) and alignment.
// CHECK: plan_hex(pad): 52504c4e00000000000000000100000001000000011e0000000000000000000000010000000100000000000000000020000000010000000100000001200000000000000000000000010000000100000000000000000020000000010000000000000001000000010000007801000000011e000000000000000100000001010000000000000001000000010100000000000000010000000000000001000000010000000000000000010000000102000000000000000020000000000000000000000000000000010000000000000040000000000000000000000000000100000001000000010000000700000000
// expected-remark @below {{encoded 236 bytes, deterministic = true, within_size_budget(4096) = true}}
"test.plan"() {serialize, name = "pad", plan = #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "x", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}], constraints = {align(0, 64), no_copy = false}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// Degraded plan: runtime_pad_check auto-set at parse; symbols in pads.
// CHECK: plan_hex(degraded): 52504c4e0000000001000000010000004e010000000100000000000000000000000001000000010000000000000000002000000001000000070000000140000000000000000000000000013f000000000000000201400000000000000005040000000001000000010000000000000000002000000001000000000000000100000001000000780100000000000000000100000001010000000000000001000000010100000000000000010000000000000001000000010000000000000000090000000140000000000000000000000000013f000000000000000201400000000000000005040000000000030020000000000000000000000000000000000000000000000000010100000001000000010000000700000000
// expected-remark @below {{encoded 279 bytes, deterministic = true, within_size_budget(4096) = true}}
"test.plan"() {serialize, name = "degraded", plan = #reloc.plan<src = tensor<[N], f32>, dst = tensor<[64 * ((N + 63) floordiv 64)], f32>, perm = [0], axes = [{name = "x", extent = N, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = (64 * ((N + 63) floordiv 64)) - N, value = 0.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()

// Non-signless integer element types are rejected by the encoder (v0 is
// signless-only); the error fires during serialization, not verification.
// expected-error @below {{signed/unsigned integer element types are not representable in wire format v0 (signless only): 'si32'}}
"test.plan"() {serialize, name = "signed", plan = #reloc.plan<src = tensor<[8], si32>, dst = tensor<[8], si32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
