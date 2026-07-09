// RUN: sym-opt --split-input-file --verify-diagnostics %s

func.func @perm_size(%t: !sym.tensor<[64, 32], f32>) {
  // expected-error @below {{perm size (1) must match operand rank (2)}}
  %0 = reloc.transpose %t perm [0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
  return
}

// -----

func.func @perm_dup(%t: !sym.tensor<[64, 32], f32>) {
  // expected-error @below {{perm is not a permutation of [0, 2)}}
  %0 = reloc.transpose %t perm [0, 0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[32, 64], f32>
  return
}

// -----

func.func @transpose_bad_result(%t: !sym.tensor<[64, 32], f32>) {
  // expected-error @below {{result dimension 0 must equal operand dimension 1}}
  %0 = reloc.transpose %t perm [1, 0] : !sym.tensor<[64, 32], f32> -> !sym.tensor<[64, 32], f32>
  return
}

// -----

func.func @reshape_count(%t: !sym.tensor<[6], f32>) {
  // expected-error @below {{element count provably changes: operand has #sym.constant<6> elements, target has #sym.constant<8>}}
  %0 = reloc.reshape %t to [4, 2] : !sym.tensor<[6], f32> -> !sym.tensor<[4, 2], f32>
  return
}

// -----

func.func @reshape_result_mismatch(%t: !sym.tensor<[4096], f32>) {
  // expected-error @below {{result dimension 0 must equal target_shape entry 0}}
  %0 = reloc.reshape %t to [64, 64] : !sym.tensor<[4096], f32> -> !sym.tensor<[32, 128], f32>
  return
}

// -----

func.func @pad_axis(%t: !sym.tensor<[6], f32>) {
  // expected-error @below {{axis (3) is out of range for operand rank 1}}
  %0 = reloc.pad %t axis 3 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return
}

// -----

func.func @pad_value_type(%t: !sym.tensor<[6], f32>) {
  // expected-error @below {{pad value type ('f64') must match the element type ('f32')}}
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f64) : !sym.tensor<[6], f32> -> !sym.tensor<[8], f32>
  return
}

// -----

func.func @pad_result_dim(%t: !sym.tensor<[6], f32>) {
  // expected-error @below {{result dimension 0 must equal operand dimension + lo + hi}}
  %0 = reloc.pad %t axis 0 lo 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[9], f32>
  return
}

// -----

func.func @pad_negative_width(%t: !sym.tensor<[6], f32>) {
  // expected-error @below {{lo is provably negative: #sym.constant<-1>}}
  %0 = reloc.pad %t axis 0 lo 0 - 1 hi 1 value (0.0 : f32) : !sym.tensor<[6], f32> -> !sym.tensor<[6], f32>
  return
}

// -----

// plan_result: input rank must match the plan's src rank.
func.func @plan_result_bad_input_rank(%t: !sym.tensor<[2, 4], f32>) {
  // expected-error @below {{input rank (2) must match the plan src rank (1)}}
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[2, 4], f32> -> !sym.tensor<[8], f32>
  return
}

// -----

// plan_result: provably mismatched input extent.
func.func @plan_result_bad_input_extent(%t: !sym.tensor<[7], f32>) {
  // expected-error @below {{input dimension 0 provably disagrees with the plan src extent}}
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[7], f32> -> !sym.tensor<[8], f32>
  return
}

// -----

// plan_result: element types must match the plan.
func.func @plan_result_bad_elem_type(%t: !sym.tensor<[8], i32>) {
  // expected-error @below {{input element type ('i32') must match the plan src element type ('f32')}}
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[8], i32> -> !sym.tensor<[8], f32>
  return
}

// -----

// plan_result: provably mismatched result extent against the plan dst.
func.func @plan_result_bad_result_extent(%t: !sym.tensor<[8], f32>) {
  // expected-error @below {{result dimension 0 provably disagrees with the plan dst extent}}
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[8], f32> -> !sym.tensor<[9], f32>
  return
}

// -----

// plan_result: on a rank mismatch (result rank 2 vs plan dst rank 1), the
// element counts must not provably disagree (rank-collapsed canonical
// plans are legal; wrong sizes are not: 3*3 == 9 != 8).
func.func @plan_result_rank_mismatch_bad_count(%t: !sym.tensor<[8], f32>) {
  // expected-error @below {{result element count provably disagrees with the plan dst}}
  %0 = reloc.plan_result %t plan(#reloc.plan<src = tensor<[8], f32>, dst = tensor<[8], f32>, perm = [0], axes = [{name = "x", extent = 8, src_stride = 1, dst_stride = 1}], inverse = affine_map<(d0) -> (d0)>>) : !sym.tensor<[8], f32> -> !sym.tensor<[3, 3], f32>
  return
}
