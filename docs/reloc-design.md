# Reloc Dialect Design Decisions

Decisions resolved for P1a/A2 (issues #14, #17). Approved 2026-07-07.

## Decision 1 — Expression representation: sym at the boundary, affine inside

`#reloc.plan` and its helper attributes store **sym expression attributes**
(`SymbolExprAttr`, `ConstantExprAttr`, `BinaryExprAttr`) for every symbolic
scalar field (extents, strides, offsets, pad bounds, divisibility exprs).
MLIR `AffineExpr`/`AffineMap` is used **internally** for composition and
simplification, reached through a lossless bridge in `RelocUtils`
(`symToAffine` / `affineToSym`).

**Rationale:** composition correctness is the core of the folding approach
(P1b). MLIR's `AffineMap::compose`, `simplifyAffineExpr`, and `IntegerSet`
are battle-tested; reimplementing them over the sym AST is the larger risk.
Keeping sym attrs at the attribute boundary preserves dialect coherence and
gives P2's MLIR-free decoder a single expression AST to decode.

**Exception:** the `inverse` field is stored as a builtin `AffineMapAttr`
(matching the build doc §2.1 reference plan). Its dims `d0..d{n-1}` are the
**dst axes**; its results are **src indices**. Inverse maps that need free
symbols are deferred to P1b, which will define the symbol-binding convention
when it first constructs one.

### Bridge conventions

- Sym symbols are named; affine symbols are positional. `symToAffine`
  assigns positions in order of first appearance and records names in a
  caller-provided list; `affineToSym` maps positions back through that list.
- Sym `Sub` maps to affine `lhs + rhs * -1` (affine has no Sub).
  `affineToSym` recognizes `x + (y * -1)` and `x + (-c)` and rebuilds `Sub`,
  so subtraction survives the round trip.
- Affine `CeilDiv` and `AffineDimExpr` have no sym counterpart in A2;
  `affineToSym` returns null for them.
- Round-trip identity is judged by `UnificationSolver::areLogicallyEqual`
  (structural equality modulo Add/Mul commutativity), because affine
  canonicalizes constants to the RHS of `Mul` (`64 * N` → `s0 * 64`).
- `affineToSym` emits `Sub(x, c)` where affine holds `x + (-c)`; a sym
  expression written as `Add(x, -c)` maps to the same affine expression, so
  the bridge is not injective on such input — round-trip identity is
  guaranteed for `Sub`-form input.

## Decision 2 — `div` is floor division

`SymbolicExprOp::Div` is **floor division** (rounds toward negative
infinity) and `SymbolicExprOp::Mod` is the matching **floor modulo** (result
has the sign of the divisor). These are exactly MLIR affine `floordiv` /
`mod` semantics, making the bridge lossless. sym's constant folding is fixed
accordingly (`-7 div 2 = -4`, `-7 mod 2 = 1`); previously it used C++
truncating `/` and `%`.

The reloc compact syntax prints `floordiv` and accepts both `floordiv` and
`div` on parse. sym's own attribute syntax (`div`) is unchanged.

## Decision 3 — Constraints are plan-local

Divisibility (`(expr, k)` pairs), alignment (`(axis, bytes)` pairs),
per-axis contiguity flags, and the `no_copy` bool are parameters of
`#reloc.plan`, not extensions to sym. Keeps the sym change surface at zero
(beyond the decision-2 semantics pin).

## Compact expression syntax

All symbolic scalar fields in reloc attributes use this grammar:

```
expr   := term  (('+' | '-') term)*
term   := factor (('*' | 'floordiv' | 'div' | 'mod') factor)*
factor := INTEGER | IDENT | STRING | '(' expr ')'
```

- `INTEGER` → `ConstantExprAttr` (negative literals allowed).
- `IDENT` (bare identifier, not `floordiv`/`div`/`mod`) → `SymbolExprAttr`.
- `STRING` (quoted) → `SymbolExprAttr` with an arbitrary name.
- Left-associative; `* floordiv mod` bind tighter than `+ -`.
- Parsing builds through `sym::getSimplifiedBinaryExpr`, so expressions
  simplify at parse time exactly like `#sym.binary` (e.g. `64 * 64 * x`
  prints back as `4096 * x`). Round-trip stability is at the attribute
  level: parse → print → parse yields the identical attribute.
- The printer emits minimal parentheses: a child is parenthesized iff its
  precedence is lower than its parent's, or equal while in RHS position.
- Symbols print bare when identifier-shaped and unreserved, else quoted.

## Descriptor defaults

`tensor_desc` syntax: `tensor<[extents], elemType, strides = [..], offset = e>`.
`strides` may be omitted (stored empty = canonical row-major, layout defined
by the plan's axes); `offset` may be omitted (stored as constant 0). The
printer omits fields holding their default, so defaults are stable under
round-trip. The build doc §2.1 writes `tensor<?x? x f32, ...>`; the concrete
syntax uses bracketed extent lists with named symbols (`tensor<[N, N], f32,
...>`) — same information, unambiguous to parse (`x` separators collide with
symbol names).

## Verification split (A2 vs A3)

A2 attribute verifiers check only **local structure**: fields are sym
expressions of the right kind, sizes agree (`strides` empty-or-`extents`-sized,
`perm`/`contiguity` sized to `axes`), scalars in range (`divisor > 0`,
`bytes > 0`, `dst_axis >= 0`, `axis >= 0`). Cross-field semantics (perm is a permutation,
`inverse` dim count = rank, extent consistency between descriptors and axes,
`no_copy` ⇒ `isPureView`) belong to A3's plan verifier.

## Structure predicates

- `isContiguousCompatible(outer, inner)` ⟺
  `outer.src_stride == inner.src_stride * inner.extent`, proved via
  `getSimplifiedBinaryExpr` + `areLogicallyEqual`.
- `isPureView(plan)` ⟺ `pad_fill` empty ∧ every axis has
  `dst_stride == src_stride` ∧ `src.offset == dst.offset`.
- Both are **sound but incomplete**: `true` means provably so under sym
  simplification; `false` means "not proven". P1b may strengthen the prover.
