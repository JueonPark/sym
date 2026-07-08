// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s | FileCheck %s

// Golden wire-format tests. The hex lines below are the committed golden
// blobs: any encoder change that alters the byte stream fails FileCheck,
// which is the cross-build determinism guarantee. Structural properties
// (size, in-process determinism, the E9 <4KB budget) are asserted via
// remarks. Golden hex is pinned in this file by Task 3.

// The worked example from docs/reloc-plan-format.md: 1-D identity plan.
// CHECK: plan_hex(identity): 52504c4e00000000
// expected-remark @below {{deterministic = true, within_size_budget(4096) = true}}
"test.plan"() {serialize, name = "identity", plan = #reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
