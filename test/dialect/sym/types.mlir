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

// Binary-expression dims (issue #31): writable in textual IR and
// round-trippable. The printer uses the compact infix syntax (bare
// identifiers inside expressions); plain symbol dims keep their quoted
// form.
// CHECK-LABEL: func.func @test_binary_dim_tensor
func.func @test_binary_dim_tensor() -> !sym.tensor<["N" floordiv 64, 64], f32> {
  // CHECK: sym.constant : !sym.tensor<[N floordiv 64, 64], f32>
  %0 = sym.constant : !sym.tensor<["N" floordiv 64, 64], f32>
  return %0 : !sym.tensor<["N" floordiv 64, 64], f32>
}

// Precedence and parentheses survive the round trip; parse-time
// simplification folds constants (64 * 2 becomes 128).
// CHECK-LABEL: func.func @test_binary_dim_precedence
func.func @test_binary_dim_precedence() -> !sym.tensor<[("N" + 1) * 8, 64 * 2], f32> {
  // CHECK: sym.constant : !sym.tensor<[(N + 1) * 8, 128], f32>
  %0 = sym.constant : !sym.tensor<[("N" + 1) * 8, 64 * 2], f32>
  return %0 : !sym.tensor<[("N" + 1) * 8, 64 * 2], f32>
}

// Bare identifiers are accepted on input (printer keeps plain symbol dims
// quoted, so the output form is stable).
// CHECK-LABEL: func.func @test_bare_ident_dim
func.func @test_bare_ident_dim() -> !sym.tensor<[N, 4], f32> {
  // CHECK: sym.constant : !sym.tensor<["N", 4], f32>
  %0 = sym.constant : !sym.tensor<[N, 4], f32>
  return %0 : !sym.tensor<["N", 4], f32>
}

// The legacy full-attribute dim spelling still parses (factor-level
// fallback) and prints compactly.
// CHECK-LABEL: func.func @test_full_attribute_dim
func.func @test_full_attribute_dim() -> !sym.tensor<[#sym.binary<#sym.symbol<"N"> + #sym.constant<1>>], f32> {
  // CHECK: sym.constant : !sym.tensor<[N + 1], f32>
  %0 = sym.constant : !sym.tensor<[#sym.binary<#sym.symbol<"N"> + #sym.constant<1>>], f32>
  return %0 : !sym.tensor<[N + 1], f32>
}
