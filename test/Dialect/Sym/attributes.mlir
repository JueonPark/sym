// RUN: sym-opt --allow-unregistered-dialect %s | sym-opt --allow-unregistered-dialect | FileCheck %s

// Test SymbolExprAttr - named symbolic variable
// CHECK-LABEL: func.func @test_symbol_attr
func.func @test_symbol_attr() {
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"batch">}
  "test.use_attr"() {sym_expr = #sym.symbol<"batch">} : () -> ()
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"sequence_length">}
  "test.use_attr"() {sym_expr = #sym.symbol<"sequence_length">} : () -> ()
  return
}

// Test ConstantExprAttr - constant integer value
// CHECK-LABEL: func.func @test_constant_attr
func.func @test_constant_attr() {
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<42>}
  "test.use_attr"() {sym_expr = #sym.constant<42>} : () -> ()
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.constant<0>} : () -> ()
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<-1>}
  "test.use_attr"() {sym_expr = #sym.constant<-1>} : () -> ()
  return
}

// Test BinaryExprAttr - binary operations
// CHECK-LABEL: func.func @test_binary_attr
func.func @test_binary_attr() {
  // add: batch + 1
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<add, #sym.symbol<"batch">, #sym.constant<1>>}
  "test.use_attr"() {sym_expr = #sym.op<add, #sym.symbol<"batch">, #sym.constant<1>>} : () -> ()
  
  // sub: a - b
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<sub, #sym.symbol<"a">, #sym.symbol<"b">>}
  "test.use_attr"() {sym_expr = #sym.op<sub, #sym.symbol<"a">, #sym.symbol<"b">>} : () -> ()
  
  // mul: x * 2
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<mul, #sym.symbol<"x">, #sym.constant<2>>}
  "test.use_attr"() {sym_expr = #sym.op<mul, #sym.symbol<"x">, #sym.constant<2>>} : () -> ()
  
  // div: total / batch_size
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<div, #sym.symbol<"total">, #sym.symbol<"batch_size">>}
  "test.use_attr"() {sym_expr = #sym.op<div, #sym.symbol<"total">, #sym.symbol<"batch_size">>} : () -> ()
  
  // mod: index % 8
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<mod, #sym.symbol<"index">, #sym.constant<8>>}
  "test.use_attr"() {sym_expr = #sym.op<mod, #sym.symbol<"index">, #sym.constant<8>>} : () -> ()
  return
}

// Test nested BinaryExprAttr - (a + b) * c
// CHECK-LABEL: func.func @test_nested_binary_attr
func.func @test_nested_binary_attr() {
  // (a + b) * c
  // CHECK: "test.use_attr"() {sym_expr = #sym.op<mul, #sym.op<add, #sym.symbol<"a">, #sym.symbol<"b">>, #sym.symbol<"c">>}
  "test.use_attr"() {sym_expr = #sym.op<mul, #sym.op<add, #sym.symbol<"a">, #sym.symbol<"b">>, #sym.symbol<"c">>} : () -> ()
  return
}
