// RUN: sym-opt %s 2>&1 | FileCheck %s

// Test sym.change_type operation

// CHECK-LABEL: func.func @test_change_type
func.func @test_change_type(%arg0: tensor<16x32xf32>) -> !sym.tensor<["batch", 32], f32> {
  // CHECK: sym.change_type %arg0 : tensor<16x32xf32> -> <["batch", 32], f32>
  %0 = sym.change_type %arg0 : tensor<16x32xf32> -> !sym.tensor<["batch", 32], f32>
  return %0 : !sym.tensor<["batch", 32], f32>
}

// CHECK-LABEL: func.func @test_change_type_dynamic
func.func @test_change_type_dynamic(%arg0: tensor<?x?xf32>) -> !sym.tensor<["batch", "seq"], f32> {
  // CHECK: sym.change_type %arg0 : tensor<?x?xf32> -> <["batch", "seq"], f32>
  %0 = sym.change_type %arg0 : tensor<?x?xf32> -> !sym.tensor<["batch", "seq"], f32>
  return %0 : !sym.tensor<["batch", "seq"], f32>
}

// CHECK-LABEL: func.func @test_change_type_mixed
func.func @test_change_type_mixed(%arg0: tensor<?x32xf32>) -> !sym.tensor<["batch", 32], f32> {
  // CHECK: sym.change_type %arg0 : tensor<?x32xf32> -> <["batch", 32], f32>
  %0 = sym.change_type %arg0 : tensor<?x32xf32> -> !sym.tensor<["batch", 32], f32>
  return %0 : !sym.tensor<["batch", 32], f32>
}
