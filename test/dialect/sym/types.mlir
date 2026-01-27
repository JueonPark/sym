// RUN: sym-opt %s | sym-opt | FileCheck %s

// Test symbolic tensor type with constant dimensions
// CHECK-LABEL: func.func @test_constant_tensor
func.func @test_constant_tensor() -> !sym.tensor<[32, 64], f32> {
  // CHECK: sym.constant : !sym.tensor<[32, 64], f32>
  %0 = sym.constant : !sym.tensor<[32, 64], f32>
  return %0 : !sym.tensor<[32, 64], f32>
}

// Test symbolic dimensions with string symbols
// CHECK-LABEL: func.func @test_symbolic_tensor
func.func @test_symbolic_tensor() -> !sym.tensor<["batch", "seq_len"], f16> {
  // CHECK: sym.constant : !sym.tensor<["batch", "seq_len"], f16>
  %0 = sym.constant : !sym.tensor<["batch", "seq_len"], f16>
  return %0 : !sym.tensor<["batch", "seq_len"], f16>
}

// Test mixed constant and symbolic dimensions
// CHECK-LABEL: func.func @test_mixed_tensor
func.func @test_mixed_tensor() -> !sym.tensor<[16, "hidden_dim", 128], bf16> {
  // CHECK: sym.constant : !sym.tensor<[16, "hidden_dim", 128], bf16>
  %0 = sym.constant : !sym.tensor<[16, "hidden_dim", 128], bf16>
  return %0 : !sym.tensor<[16, "hidden_dim", 128], bf16>
}
