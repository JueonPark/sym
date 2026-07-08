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
