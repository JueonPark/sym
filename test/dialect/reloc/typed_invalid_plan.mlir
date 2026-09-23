// RUN: sym-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

// C2 (issue #142) Task 1: type-chain verifiers of the typed plan. One stage's
// output must match the next stage's input, the layout's element types must
// match the plan's source/result, every fill resolves to a layout pad and
// carries the dtype of the stage it enters, fused fills equal the fold of the
// original fill, and a typed plan is never a pure view.

//===----------------------------------------------------------------------===//
// #reloc.stage
//===----------------------------------------------------------------------===//

func.func @stage_unsupported_cast_pair() {
  // expected-error @below {{cast from 'f32' to 'f32' is not a supported typed conversion (f32 -> f16, f16 -> f32)}}
  "test.use_attr"() {stage = #reloc.stage<cast ieee_rne : f32 -> f32, shape = [4]>} : () -> ()
  return
}

// -----

func.func @stage_wrong_policy() {
  // expected-error @below {{policy 'affine' is not defined for reloc.quantize (use 'symmetric_rne')}}
  "test.use_attr"() {stage = #reloc.stage<quantize affine : f32 -> i8, shape = [4], scale = dense<0.5> : tensor<f32>>} : () -> ()
  return
}

// -----

func.func @stage_cast_with_parameters() {
  // expected-error @below {{cast stages carry no parameters}}
  "test.use_attr"() {stage = #reloc.stage<cast ieee_rne : f32 -> f16, shape = [4], scale = dense<0.5> : tensor<f32>>} : () -> ()
  return
}

// -----

func.func @stage_quantize_without_scale() {
  // expected-error @below {{quantize stages require a scale}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [4]>} : () -> ()
  return
}

// -----

func.func @stage_axis_without_channel() {
  // expected-error @below {{per-channel stage requires a channel map}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.5, 0.5]> : tensor<3xf32>, axis = 1>} : () -> ()
  return
}

// -----

func.func @stage_channel_without_axis() {
  // expected-error @below {{channel map requires a channel axis}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<0.5> : tensor<f32>, channel = (d0, d1) -> (d1)>} : () -> ()
  return
}

// -----

func.func @stage_axis_out_of_range() {
  // expected-error @below {{axis (2) is out of range for operand rank 2}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.5, 0.5]> : tensor<3xf32>, axis = 2, channel = (d0, d1) -> (d1)>} : () -> ()
  return
}

// -----

// The C1 parameter rules apply to stages verbatim.
func.func @stage_channel_length_mismatch() {
  // expected-error @below {{scale has 4 channel entries, but axis 1 has extent 3}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.5, 0.5, 0.5]> : tensor<4xf32>, axis = 1, channel = (d0, d1) -> (d1)>} : () -> ()
  return
}

// -----

func.func @stage_channel_map_results() {
  // expected-error @below {{channel map must have exactly one result}}
  "test.use_attr"() {stage = #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.5, 0.5]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d0, d1)>} : () -> ()
  return
}

// -----

func.func @stage_empty_shape() {
  // expected-error @below {{stage shape must have rank at least 1}}
  "test.use_attr"() {stage = #reloc.stage<cast ieee_rne : f32 -> f16, shape = []>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// #reloc.typed_plan: chain, layout and fills
//===----------------------------------------------------------------------===//

// -----

func.func @plan_without_stages() {
  // expected-error @below {{typed plan needs at least one value stage; layout-only chains use #reloc.plan}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = []>} : () -> ()
  return
}

// -----

func.func @plan_layout_source_dtype() {
  // expected-error @below {{layout source element type ('f16') must match the typed plan source ('f32')}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f16>, dst = tensor<[4], f16>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>} : () -> ()
  return
}

// -----

func.func @plan_layout_result_dtype() {
  // expected-error @below {{layout destination element type ('f32') must match the typed plan result ('f16')}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>} : () -> ()
  return
}

// -----

func.func @plan_layout_source_rank() {
  // expected-error @below {{layout source rank (1) must match the typed plan source rank (2)}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[2, 2], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f16>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [2, 2]>]>} : () -> ()
  return
}

// -----

func.func @plan_first_stage_input() {
  // expected-error @below {{stage 0 consumes 'f16' but the typed plan source is 'f32'}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f32>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f32>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast exact : f16 -> f32, shape = [4]>]>} : () -> ()
  return
}

// -----

func.func @plan_chain_break() {
  // expected-error @below {{stage 1 consumes 'f32' but stage 0 produces 'f16'}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], i8>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], i8>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>, #reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [4], scale = dense<0.5> : tensor<f32>>]>} : () -> ()
  return
}

// -----

func.func @plan_last_stage_output() {
  // expected-error @below {{stage 0 produces 'f16' but the typed plan result is 'i8'}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], i8>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], i8>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>} : () -> ()
  return
}

// -----

func.func @plan_channel_dims() {
  // expected-error @below {{stage 0 channel map has 1 dims but the typed plan result has rank 2}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[2, 3], f32>, result = tensor<[3, 2], i8>, layout = #reloc.plan<src = tensor<[2, 3], f32>, dst = tensor<[3, 2], i8>, perm = [1, 0], axes = [{name = "d0", extent = 3, src_stride = 1, dst_stride = 2}, {name = "d1", extent = 2, src_stride = 3, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d1, d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [2, 3], scale = dense<[0.5, 0.25, 0.125]> : tensor<3xf32>, axis = 1, channel = (d0) -> (d0)>]>} : () -> ()
  return
}

// -----

func.func @plan_channel_symbols() {
  // expected-error @below {{stage 0 channel map has 1 symbols but the typed plan declares 0}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[B, C], f32>, result = tensor<[B * C], i8>, layout = #reloc.plan<src = tensor<[B, C], f32>, dst = tensor<[B * C], i8>, perm = [0], axes = [{name = "d0", extent = B * C, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [B, C], scale = #reloc.binding<"s" : [C], f32>, axis = 1, channel = (d0)[s0] -> (d0 mod s0)>]>} : () -> ()
  return
}

// -----

func.func @plan_fill_without_pad() {
  // expected-error @below {{fill on dst_axis 0 has no matching layout pad_fill entry}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[6], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[6], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 0, value = 1.0 : f32}]>} : () -> ()
  return
}

// -----

func.func @plan_pad_without_fill() {
  // expected-error @below {{layout pad_fill on dst_axis 0 has no typed fill entry}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<0.5> : tensor<f32>>]>} : () -> ()
  return
}

// -----

func.func @plan_fill_stage_out_of_range() {
  // expected-error @below {{fill on dst_axis 0 enters at stage 2 but the plan has 1 stages}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [6], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 2, value = 1 : i8}]>} : () -> ()
  return
}

// -----

// A fill entering before the quantize stage belongs to the f32 stage.
func.func @plan_fill_wrong_dtype() {
  // expected-error @below {{fill on dst_axis 0 enters at stage 0 as 'f32' but has type 'i8'}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 1 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 0, value = 1 : i8}]>} : () -> ()
  return
}

// -----

// The fused fill must be the C1 fold of the original: 1.0 at scale 0.5 is
// code 2, never 3.
func.func @plan_fused_fill_mismatch() {
  // expected-error @below {{fill on dst_axis 0 folds to 2 : i8 but the layout pad_fill carries 3 : i8}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6], f32>, result = tensor<[8], i8>, layout = #reloc.plan<src = tensor<[6], f32>, dst = tensor<[8], i8>, perm = [0], axes = [{name = "d0", extent = 6, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 3 : i8}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8], scale = dense<0.5> : tensor<f32>>], fills = [{dst_axis = 0, stage = 0, value = 1.0 : f32}]>} : () -> ()
  return
}

// -----

// A pad that precedes a per-channel stage has no single fused fill: every
// padded position would take its own channel's code. The fold must bail, so
// the representation rejects it.
func.func @plan_fill_not_foldable() {
  // expected-error @below {{fill on dst_axis 0 cannot be folded through stage 0 (per-channel or runtime parameters)}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[6, 3], f32>, result = tensor<[8, 3], i8>, layout = #reloc.plan<src = tensor<[6, 3], f32>, dst = tensor<[8, 3], i8>, perm = [0, 1], axes = [{name = "d0", extent = 6, src_stride = 3, dst_stride = 3}, {name = "d1", extent = 3, src_stride = 1, dst_stride = 1}], pad_fill = [{dst_axis = 0, lo = 1, hi = 1, value = 2 : i8}], inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<quantize symmetric_rne : f32 -> i8, shape = [8, 3], scale = dense<[0.5, 0.25, 0.125]> : tensor<3xf32>, axis = 1, channel = (d0, d1) -> (d1)>], fills = [{dst_axis = 0, stage = 0, value = 1.0 : f32}]>} : () -> ()
  return
}

// -----

// The layout part of a typed plan may be an identity view, but the plan as a
// whole moves data: a narrowing cast is never a view.
func.func @plan_no_copy() {
  // expected-error @below {{a typed plan is never a pure view: layout no_copy must be false}}
  "test.use_attr"() {plan = #reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f16>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], constraints = {no_copy = true}, inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>} : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// reloc.typed_plan_result
//===----------------------------------------------------------------------===//

// -----

func.func @result_input_dtype(%t: !sym.tensor<[4], f16>) {
  // expected-error @below {{input element type ('f16') must match the typed plan source element type ('f32')}}
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f16>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>) : !sym.tensor<[4], f16> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @result_rank(%t: !sym.tensor<[2, 2], f32>) {
  // expected-error @below {{result rank (1) must match the typed plan result rank (2)}}
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[2, 2], f32>, result = tensor<[2, 2], f16>, layout = #reloc.plan<src = tensor<[2, 2], f32>, dst = tensor<[2, 2], f16>, perm = [0, 1], axes = [{name = "d0", extent = 2, src_stride = 2, dst_stride = 2}, {name = "d1", extent = 2, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0, d1) -> (d0, d1)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [2, 2]>]>) : !sym.tensor<[2, 2], f32> -> !sym.tensor<[4], f16>
  return
}

// -----

func.func @result_input_extent(%t: !sym.tensor<[5], f32>) {
  // expected-error @below {{input dimension 0 provably disagrees with the typed plan source extent}}
  %0 = reloc.typed_plan_result %t plan(#reloc.typed_plan<source = tensor<[4], f32>, result = tensor<[4], f16>, layout = #reloc.plan<src = tensor<[4], f32>, dst = tensor<[4], f16>, perm = [0], axes = [{name = "d0", extent = 4, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>, stages = [#reloc.stage<cast ieee_rne : f32 -> f16, shape = [4]>]>) : !sym.tensor<[5], f32> -> !sym.tensor<[4], f16>
  return
}
