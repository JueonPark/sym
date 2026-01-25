// RUN: sym-opt %s | sym-opt | FileCheck %s

// Test sym.constant operation
// CHECK-LABEL: func.func @test_constant_op
func.func @test_constant_op() -> !sym.tensor<[32], f32> {
  // CHECK: %[[C:.*]] = sym.constant : !sym.tensor<[32], f32>
  %0 = sym.constant : !sym.tensor<[32], f32>
  // CHECK: return %[[C]]
  return %0 : !sym.tensor<[32], f32>
}

// Test sym.constant with symbolic dimensions
// CHECK-LABEL: func.func @test_constant_symbolic
func.func @test_constant_symbolic() -> !sym.tensor<["n", "m"], f64> {
  // CHECK: %[[C:.*]] = sym.constant : !sym.tensor<["n", "m"], f64>
  %0 = sym.constant : !sym.tensor<["n", "m"], f64>
  // CHECK: return %[[C]]
  return %0 : !sym.tensor<["n", "m"], f64>
}

// Test multiple sym.constant operations
// CHECK-LABEL: func.func @test_multiple_constants
func.func @test_multiple_constants() -> (!sym.tensor<[8], f32>, !sym.tensor<["batch"], i32>) {
  // CHECK-DAG: %[[A:.*]] = sym.constant : !sym.tensor<[8], f32>
  %a = sym.constant : !sym.tensor<[8], f32>
  // CHECK-DAG: %[[B:.*]] = sym.constant : !sym.tensor<["batch"], i32>
  %b = sym.constant : !sym.tensor<["batch"], i32>
  return %a, %b : !sym.tensor<[8], f32>, !sym.tensor<["batch"], i32>
}
