# Symbolic shapes in MLIR

The `sym` dialect represents tensor dimensions as algebraic expressions.
This lets a compiler describe a family of tensor shapes before concrete
input sizes are known. The `reloc` dialect uses these expressions to build
reusable tensor layout and value-transform plans.

## Tensor types and expressions

```mlir
// Named dimensions.
!sym.tensor<["batch", "features"], f32>

// Symbolic and fixed dimensions together.
!sym.tensor<["batch", 64], f16>

// A split dimension expressed without a concrete batch size.
!sym.tensor<[N floordiv 64, 64], f32>
```

Expressions support constants, symbols, arithmetic, floor division, and
modulo. Algebraic simplification removes identities such as `x + 0` and
`x * 1`. Symbolic tensor types can describe more than the subset accepted by
the relocation exporter; the [export contract](reloc-export.md) specifies
its supported types, expressions, and constraints.

`sym.constant` produces a symbolic tensor value. `sym.change_type` gives
a tensor an explicit symbolic type, as in this complete function:

```mlir
func.func @annotate(%input: tensor<?x32xf32>) -> !sym.tensor<["batch", 32], f32> {
  %result = sym.change_type %input : tensor<?x32xf32> -> !sym.tensor<["batch", 32], f32>
  return %result : !sym.tensor<["batch", 32], f32>
}
```

## Command-line tools

After [building Sym](getting-started.md#build-the-compiler-and-cpu-runtime),
use `sym-opt` to parse, verify, and transform MLIR:

```bash
"$SYM_BUILD/sym/tools/sym-opt" test/dialect/sym/types.mlir
"$SYM_BUILD/sym/tools/sym-opt" --sym-shape-inference test/dialect/sym/shape_inference.mlir
"$SYM_BUILD/sym/tools/sym-opt" --reloc-fold libreloc/examples/recipes/split_transpose.mlir
```

The shape-inference pass operates on functions and asks operations that
implement `SymbolicShapeOpInterface` to infer their result shapes. External
models attach the interface to supported MLIR `arith` operations, using
NumPy-style broadcasting unification for elementwise operations. Operations
without the interface are left unchanged.

`--reloc-fold` combines supported reshape/transpose/pad chains and explicit
cast/quantize/dequantize stages into relocation plans. Use
[`sym-reloc-export`](reloc-export.md) to produce binary artifacts for runtime
execution; `sym-opt`'s printed IR is useful for inspection.

## Compiler development

| Source | Purpose |
| --- | --- |
| [`sym/dialect/sym/IR`](../sym/dialect/sym/IR) | Symbolic types, expressions, operations, and shape interface |
| [`SymUtils.h`](../sym/dialect/sym/IR/SymUtils.h) | Expression helpers and `UnificationSolver` |
| [`SymExtensions.cpp`](../sym/dialect/sym/IR/SymExtensions.cpp) | External shape-inference models for `arith` |
| [`sym/dialect/sym/Transforms`](../sym/dialect/sym/Transforms) | Symbolic shape inference pass |
| [`sym/dialect/reloc`](../sym/dialect/reloc) | Relocation IR, folding, and serialization |
| [`sym/tools`](../sym/tools) | `sym-opt` and `sym-reloc-export` drivers |
| [`test/dialect/sym`](../test/dialect/sym) | Symbolic type and operation examples/tests |

Build and run the compiler tests with
`cmake --build "$SYM_BUILD" --target check-sym`. For relocation details,
see [typed semantics](reloc-typed-semantics.md),
[typed folding](reloc-typed-folding.md), and the
[wire format](reloc-plan-format.md).
