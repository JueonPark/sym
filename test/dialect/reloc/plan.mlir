// RUN: sym-opt --allow-unregistered-dialect %s | sym-opt --allow-unregistered-dialect | FileCheck %s

// MLIR's generic AsmPrinter hoists AffineMapAttr values to file-level
// `#mapN` aliases (this is printer-owned, not PlanAttr's doing), so the
// CHECK lines below reference the aliases rather than inline `affine_map<>`.
// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #map2 = affine_map<(d0) -> (d0)>
// CHECK: #map3 = affine_map<(d0, d1) -> (d0 * 64 + d1)>

// Reference plan from build doc §2.1:
// 32768×32768 fp32 → 4D view (N/64, 64, 64, N/64) + transpose(0,1), N = 32768.
// CHECK-LABEL: func.func @reference_plan
func.func @reference_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[N, N], f32, strides = [N, 1]>, dst = tensor<[N floordiv 64, 64, 64, N floordiv 64], f32>, perm = [1, 0, 2, 3], axes = [{name = "n0", extent = N floordiv 64, src_stride = 64, dst_stride = 4096 * (N floordiv 64)}, {name = "b0", extent = 64, src_stride = 64 * N, dst_stride = 64 * (N floordiv 64)}, {name = "b1", extent = 64, src_stride = N, dst_stride = N floordiv 64}, {name = "n1", extent = N floordiv 64, src_stride = 1, dst_stride = 1}], constraints = {divisible(N, 64), contiguous = [false, false, false, true], no_copy = false}, inverse = #map>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[N, N], f32, strides = [N, 1], offset = 0>, dst = tensor<[N floordiv 64, 64, 64, N floordiv 64], f32>, perm = [1, 0, 2, 3], axes = [{name = "n0", extent = N floordiv 64, src_stride = 64, dst_stride = 64 * 64 * (N floordiv 64)}, {name = "b0", extent = 64, src_stride = 64 * N, dst_stride = 64 * (N floordiv 64)}, {name = "b1", extent = 64, src_stride = N, dst_stride = N floordiv 64}, {name = "n1", extent = N floordiv 64, src_stride = 1, dst_stride = 1}], pad_fill = [], constraints = {divisible(N, 64), contiguous = [false, false, false, true], no_copy = false}, inverse = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>>} : () -> ()
  return
}

// Static 2D transpose as a pure strided view (no data movement).
// CHECK-LABEL: func.func @transpose_view
func.func @transpose_view() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[64, 32], f32, strides = [32, 1]>, dst = tensor<[32, 64], f32, strides = [1, 32]>, perm = [1, 0], axes = [{name = "r", extent = 64, src_stride = 32, dst_stride = 32}, {name = "c", extent = 32, src_stride = 1, dst_stride = 1}], constraints = {no_copy = true}, inverse = #map1>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[64, 32], f32, strides = [32, 1]>, dst = tensor<[32, 64], f32, strides = [1, 32]>, perm = [1, 0], axes = [{name = "r", extent = 64, src_stride = 32, dst_stride = 32}, {name = "c", extent = 32, src_stride = 1, dst_stride = 1}], constraints = {no_copy = true}, inverse = affine_map<(d0, d1) -> (d1, d0)>>} : () -> ()
  return
}

// 1D pad with an f32 fill value and an alignment constraint.
// CHECK-LABEL: func.func @pad_plan
func.func @pad_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "x", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.000000e+00 : f32}], constraints = {align(0, 64), no_copy = false}, inverse = #map2>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[30], f32>, dst = tensor<[32], f32>, perm = [0], axes = [{name = "x", extent = 30, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 0, hi = 2, value = 0.0 : f32}], constraints = {align(0, 64), no_copy = false}, inverse = affine_map<(d0) -> (d0)>>} : () -> ()
  return
}

// Static reshape [4096] -> [64, 64] with contiguity flags.
// CHECK-LABEL: func.func @reshape_plan
func.func @reshape_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[4096], f32>, dst = tensor<[64, 64], f32>, perm = [0, 1], axes = [{name = "o", extent = 64, src_stride = 64, dst_stride = 64}, {name = "i", extent = 64, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true, true], no_copy = true}, inverse = #map3>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[4096], f32>, dst = tensor<[64, 64], f32>, perm = [0, 1], axes = [{name = "o", extent = 64, src_stride = 64, dst_stride = 64}, {name = "i", extent = 64, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true, true], no_copy = true}, inverse = affine_map<(d0, d1) -> (d0 * 64 + d1)>>} : () -> ()
  return
}

// Minimal plan: every optional section omitted.
// CHECK-LABEL: func.func @minimal_plan
func.func @minimal_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = #map2>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[8], i32>, dst = tensor<[8], i32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
  return
}

// Symbolic offset in the source descriptor.
// CHECK-LABEL: func.func @offset_plan
func.func @offset_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[N], f32, offset = N floordiv 2>, dst = tensor<[N], f32>, perm = [0], axes = [{name = "x", extent = N, src_stride = 1, dst_stride = 1}], inverse = #map2>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[N], f32, offset = N floordiv 2>, dst = tensor<[N], f32>, perm = [0], axes = [{name = "x", extent = N, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
  return
}

// Pad-only plan: pad_fill present, constraints block entirely absent.
// CHECK-LABEL: func.func @pad_only_plan
func.func @pad_only_plan() {
  // CHECK: "test.use_attr"() {plan = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1.000000e+00 : f32}], inverse = #map2>}
  "test.use_attr"() {plan = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1.0 : f32}], inverse = affine_map<(d0) -> (d0)>>} : () -> ()
  return
}
