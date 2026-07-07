// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s

// Proof-kit protocol: ops with `lhs` and `rhs` sym-expr attributes get
// proveEqual / proveLessEqual remarks; ops with a `desc` tensor_desc
// attribute get a remark stating whether the canonical row-major strides
// of its extents provably equal its explicit strides.

// Constants: equal.
// expected-remark @below {{proveEqual = Proven, proveLessEqual = Proven}}
"test.prove"() {lhs = #sym.constant<4>, rhs = #sym.constant<4>} : () -> ()

// Constants: lhs < rhs.
// expected-remark @below {{proveEqual = Disproven, proveLessEqual = Proven}}
"test.prove"() {lhs = #sym.constant<3>, rhs = #sym.constant<4>} : () -> ()

// Constants: lhs > rhs.
// expected-remark @below {{proveEqual = Disproven, proveLessEqual = Disproven}}
"test.prove"() {lhs = #sym.constant<5>, rhs = #sym.constant<4>} : () -> ()

// Logically equal symbolic expressions (commutative Mul).
// expected-remark @below {{proveEqual = Proven, proveLessEqual = Proven}}
"test.prove"() {lhs = #sym.binary<#sym.constant<64> * #sym.symbol<"N">>, rhs = #sym.binary<#sym.symbol<"N"> * #sym.constant<64>>} : () -> ()

// Structurally different symbolic expressions: unknown.
// expected-remark @below {{proveEqual = Unknown, proveLessEqual = Unknown}}
"test.prove"() {lhs = #sym.symbol<"N">, rhs = #sym.binary<#sym.symbol<"N"> + #sym.constant<1>>} : () -> ()

// Symbol vs constant: unknown.
// expected-remark @below {{proveEqual = Unknown, proveLessEqual = Unknown}}
"test.prove"() {lhs = #sym.constant<0>, rhs = #sym.symbol<"N">} : () -> ()

// Row-major reconstruction, static: [8, 128] -> strides [128, 1].
// expected-remark @below {{row-major matches strides: true}}
"test.prove"() {desc = #reloc.tensor_desc<[8, 128], f32, strides = [128, 1]>} : () -> ()

// Row-major reconstruction, symbolic: matches the reference-plan pattern.
// expected-remark @below {{row-major matches strides: true}}
"test.prove"() {desc = #reloc.tensor_desc<[N floordiv 64, 64, 64, N floordiv 64], f32, strides = [4096 * (N floordiv 64), 64 * (N floordiv 64), N floordiv 64, 1]>} : () -> ()

// Row-major reconstruction, mismatch.
// expected-remark @below {{row-major matches strides: false}}
"test.prove"() {desc = #reloc.tensor_desc<[8, 128], f32, strides = [64, 1]>} : () -> ()
