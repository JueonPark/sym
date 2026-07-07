// RUN: sym-opt --allow-unregistered-dialect --test-reloc-utils --verify-diagnostics %s

// Bridge property tests: for each `expr` attribute, the pass runs
// symToAffine -> affineToSym and reports whether the result is logically
// equal to the input (areLogicallyEqual: structural equality modulo
// Add/Mul commutativity). Inputs are already sym-simplified because
// #sym.binary simplifies at parse time.

// expected-remark @below {{bridge round-trip ok: affine = s0}}
"test.expr"() {expr = #sym.symbol<"N">} : () -> ()

// expected-remark @below {{bridge round-trip ok: affine = 42}}
"test.expr"() {expr = #sym.constant<42>} : () -> ()

// expected-remark @below {{bridge round-trip ok: affine = s0 + 1}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"N"> + #sym.constant<1>>} : () -> ()

// Subtraction survives the round trip (affine stores it as s0 + -1).
// expected-remark @below {{bridge round-trip ok: affine = s0 - 1}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"N"> - #sym.constant<1>>} : () -> ()

// Symbol - symbol: affine stores a + b * -1 and prints it as a subtraction.
// expected-remark @below {{bridge round-trip ok: affine = s0 - s1}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"a"> - #sym.symbol<"b">>} : () -> ()

// div == floordiv.
// expected-remark @below {{bridge round-trip ok: affine = s0 floordiv 64}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"N"> div #sym.constant<64>>} : () -> ()

// expected-remark @below {{bridge round-trip ok: affine = s0 mod 8}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"N"> mod #sym.constant<8>>} : () -> ()

// Affine canonicalizes the constant to the RHS of mul; commutativity-aware
// equality accepts the round-tripped Mul(N, 64) for input Mul(64, N).
// expected-remark @below {{bridge round-trip ok: affine = s0 * 64}}
"test.expr"() {expr = #sym.binary<#sym.constant<64> * #sym.symbol<"N">>} : () -> ()

// Nested: (N floordiv 64) * 64 + r.
// expected-remark @below {{bridge round-trip ok: affine = (s0 floordiv 64) * 64 + s1}}
"test.expr"() {expr = #sym.binary<#sym.binary<#sym.binary<#sym.symbol<"N"> div #sym.constant<64>> * #sym.constant<64>> + #sym.symbol<"r">>} : () -> ()

// Symbol reuse: N appears twice, bound to one affine symbol.
// expected-remark @below {{bridge round-trip ok: affine = s0 * s0}}
"test.expr"() {expr = #sym.binary<#sym.symbol<"N"> * #sym.symbol<"N">>} : () -> ()

// Non-expression attributes are reported, not crashed on.
// expected-remark @below {{bridge: not a sym expression}}
"test.expr"() {expr = 42 : i64} : () -> ()
