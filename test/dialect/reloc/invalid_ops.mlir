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
