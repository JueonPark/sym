// RUN: sym-opt --allow-unregistered-dialect %s | sym-opt --allow-unregistered-dialect | FileCheck %s

//===----------------------------------------------------------------------===//
// Constant Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_constant_folding
func.func @test_constant_folding() {
  // 3 + 4 -> 7
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<7>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<3> + #sym.constant<4>>} : () -> ()
  
  // 10 - 3 -> 7
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<7>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<10> - #sym.constant<3>>} : () -> ()
  
  // 6 * 7 -> 42
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<42>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<6> * #sym.constant<7>>} : () -> ()
  
  // 20 div 4 -> 5
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<5>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<20> div #sym.constant<4>>} : () -> ()
  
  // 17 mod 5 -> 2
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<2>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<17> mod #sym.constant<5>>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Identity Removal Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_identity_add
func.func @test_identity_add() {
  // n + 0 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> + #sym.constant<0>>} : () -> ()
  
  // 0 + n -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<0> + #sym.symbol<"n">>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_identity_sub
func.func @test_identity_sub() {
  // n - 0 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> - #sym.constant<0>>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_identity_mul
func.func @test_identity_mul() {
  // n * 1 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<1>>} : () -> ()
  
  // 1 * n -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<1> * #sym.symbol<"n">>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_identity_div
func.func @test_identity_div() {
  // n div 1 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> div #sym.constant<1>>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_identity_mod
func.func @test_identity_mod() {
  // n mod 1 -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> mod #sym.constant<1>>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Zero Propagation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_zero_propagation
func.func @test_zero_propagation() {
  // n * 0 -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<0>>} : () -> ()
  
  // 0 * n -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<0> * #sym.symbol<"n">>} : () -> ()
  
  // 0 div n -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<0> div #sym.symbol<"n">>} : () -> ()
  
  // 0 mod n -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.constant<0> mod #sym.symbol<"n">>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Self-Cancellation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_self_cancellation
func.func @test_self_cancellation() {
  // n - n -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> - #sym.symbol<"n">>} : () -> ()
  
  // (a + b) - (a + b) -> 0
  // CHECK: "test.use_attr"() {sym_expr = #sym.constant<0>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"a"> + #sym.symbol<"b">> - #sym.binary<#sym.symbol<"a"> + #sym.symbol<"b">>>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Associativity Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_associativity_add
func.func @test_associativity_add() {
  // (n + 1) + 2 -> n + 3
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> + #sym.constant<3>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<1>> + #sym.constant<2>>} : () -> ()
  
  // (n + 5) + 5 -> n + 10
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> + #sym.constant<10>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<5>> + #sym.constant<5>>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_associativity_sub
func.func @test_associativity_sub() {
  // (n + 1) - 1 -> n + 0 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<1>> - #sym.constant<1>>} : () -> ()
  
  // (n + 5) - 3 -> n + 2
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> + #sym.constant<2>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<5>> - #sym.constant<3>>} : () -> ()
  
  // (n - 3) + 5 -> n + 2
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> + #sym.constant<2>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> - #sym.constant<3>> + #sym.constant<5>>} : () -> ()
  
  // (n - 3) - 2 -> n - 5
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> - #sym.constant<5>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> - #sym.constant<3>> - #sym.constant<2>>} : () -> ()
  
  return
}

// CHECK-LABEL: func.func @test_associativity_mul
func.func @test_associativity_mul() {
  // (n * 2) * 3 -> n * 6
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<6>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> * #sym.constant<2>> * #sym.constant<3>>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Complex Nested Expression Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_complex_expressions
func.func @test_complex_expressions() {
  // ((n + 1) - 1) should simplify to n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<1>> - #sym.constant<1>>} : () -> ()
  
  // ((n * 2) * 1) should simplify to n * 2
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<2>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> * #sym.constant<2>> * #sym.constant<1>>} : () -> ()
  
  // (n + 0) * 1 -> n
  // CHECK: "test.use_attr"() {sym_expr = #sym.symbol<"n">}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.binary<#sym.symbol<"n"> + #sym.constant<0>> * #sym.constant<1>>} : () -> ()
  
  return
}

//===----------------------------------------------------------------------===//
// Non-simplifiable expressions (should remain unchanged)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_no_simplification
func.func @test_no_simplification() {
  // a + b should remain as-is
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"a"> + #sym.symbol<"b">>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"a"> + #sym.symbol<"b">>} : () -> ()
  
  // n * 2 should remain as-is
  // CHECK: "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<2>>}
  "test.use_attr"() {sym_expr = #sym.binary<#sym.symbol<"n"> * #sym.constant<2>>} : () -> ()
  
  return
}
