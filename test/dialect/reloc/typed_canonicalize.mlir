// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s

// C2 (issue #142) Task 3: typed canonicalization touches the layout only
// through canonicalizePlan, keeps the logical result descriptor and channel
// maps at the logical rank, never removes a stage, and never lets a value
// program become a view. The `canonicalize` unit attr routes a hand-written
// #reloc.typed_plan through canonicalizeTypedPlan and reports the result.

// An identity layout over [8, 128] with a narrowing cast: the layout's axes
// merge (rank collapses to one axis of 1024) and isPureView would say true,
// yet no_copy stays false because the stage moves data.
// expected-remark @below {{canonicalized typed: #reloc.typed_plan<source = tensor<[8, 128], f32>, result = tensor<[8, 128], f16>, layout = #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[1024], f16>, perm = [0], axes = [{name = "d0", extent = 1024, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [8, 128]>]>}}
"test.plan"() {canonicalize, typed_plan = #reloc.typed_plan<source = tensor<[8, 128], f32>, result = tensor<[8, 128], f16>, layout = #reloc.plan<src = tensor<[8, 128], f32>, dst = tensor<[8, 128], f16>, perm = [0, 1], axes = [{name = "o", extent = 8, src_stride = 128, dst_stride = 128}, {name = "i", extent = 128, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [8, 128]>]>} : () -> ()

// Per-channel quantize over an identity layout: the layout collapses to one
// axis, but the channel map stays written over the logical rank-2 result,
// so parameter indexing survives the coalescing.
// expected-remark @below {{canonicalized typed: #reloc.typed_plan<source = tensor<[2, 12], f32>, result = tensor<[2, 12], i8>, layout = #reloc.plan<src = tensor<[2, 12], f32>, dst = tensor<[24], i8>, perm = [0], axes = [{name = "d0", extent = 24, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 12], scale = #reloc.binding<"s" : [12], f32>, axis = 1, channel = (d0, d1) -> (d1)>]>}}
"test.plan"() {canonicalize, typed_plan = #reloc.typed_plan<source = tensor<[2, 12], f32>, result = tensor<[2, 12], i8>, layout = #reloc.plan<src = tensor<[2, 12], f32>, dst = tensor<[2, 12], i8>, perm = [0, 1], axes = [{name = "a", extent = 2, src_stride = 12, dst_stride = 12}, {name = "b", extent = 12, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 12], scale = #reloc.binding<"s" : [12], f32>, axis = 1, channel = (d0, d1) -> (d1)>]>} : () -> ()

// Lossy sequences are not cancelled by canonicalization: f32 -> f16 -> f32
// keeps both stages, quantize -> dequantize keeps both stages, and the fills
// keep their exact bits (0x3EAAAAAB prints as its shortest round-trip
// decimal) with pads sorted by dst axis.
// expected-remark @below {{canonicalized typed: #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<cast exact : f16 -> f32, shape = [4]>]>}}
"test.plan"() {canonicalize, typed_plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "x", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<cast exact : f16 -> f32, shape = [4]>]>} : () -> ()

// expected-remark @below {{canonicalized typed: #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], constraints = {contiguous = [true], no_copy = false}, inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [4], scale = dense<0.333333343> : tensor<f32>>, #reloc.stage<dequantize affine : i8 -> f32, shape = [4], scale = dense<0.333333343> : tensor<f32>>]>}}
"test.plan"() {canonicalize, typed_plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "x", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [4], scale = dense<0x3EAAAAAB> : tensor<f32>>, #reloc.stage<dequantize affine : i8 -> f32, shape = [4], scale = dense<0x3EAAAAAB> : tensor<f32>>]>} : () -> ()

// Two pads entering at different stages on different axes: the fused fills
// and the originals survive, sorted by dst axis (pad on axis 1 was written
// first), and the padded axes never merge.
// expected-remark @below {{canonicalized typed: #reloc.typed_plan<source = tensor<[6, 4], f32>, result = tensor<[7, 6], i8>, layout = #reloc.plan<src = tensor<[6, 4], f32>, dst = tensor<[7, 6], i8>, perm = [0, 1], axes = [{name = "d0", extent = 6, src_stride = 4, dst_stride = 6}, {name = "d1", extent = 4, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 0, value = 2 : i8}, {dst_axis = 1, lo = 0, hi = 2, value = 7 : i8}], constraints = {contiguous = [false, true], no_copy = false}, inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [7, 4], scale = dense<5.000000e-01> : tensor<f32>>], fills = [{dst_axis = 0, stage = 0, value = 1.000000e+00 : f32}, {dst_axis = 1, stage = 1, value = 7 : i8}]>}}
"test.plan"() {canonicalize, typed_plan = #reloc.typed_plan<source = tensor<[6, 4], f32>, result = tensor<[7, 6], i8>, layout = #reloc.plan<src = tensor<[6, 4], f32>, dst = tensor<[7, 6], i8>, perm = [0, 1], axes = [{name = "a", extent = 6, src_stride = 4, dst_stride = 6}, {name = "b", extent = 4, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 1, lo = 0, hi = 2, value = 7 : i8}, {dst_axis = 0, lo = 1, hi = 0, value = 2 : i8}], inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [7, 4], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 1, stage = 1, value = 7 : i8}, {dst_axis = 0, stage = 0, value = 1.0 : f32}]>} : () -> ()
