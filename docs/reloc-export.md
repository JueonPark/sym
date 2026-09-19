# Supported layout plan export (interface 1)

Build the compiler target `sym-reloc-export`, then invoke:

```sh
sym-reloc-export recipe.mlir --output plan.bin --manifest manifest.json
```

This compiler-only executable links MLIR/LLVM and has no Torch dependency.
`libreloc` remains MLIR/LLVM/Torch-free. `plan.bin` is exactly `encodePlan` wire
format v0, documented in [reloc-plan-format.md](reloc-plan-format.md). The exporter
does not bind symbols, execute transfers, or replace a frontend callable.

Both output paths must be distinct and absent. Existing files are rejected
without modification (including existing artifacts from an earlier invocation).
New files are exclusively created; failures remove files created by this
invocation. Consumers must wait for exit 0 before reading the pair; publication
is not an atomic two-file transaction. Use a fresh temporary directory per
invocation to avoid stale outputs and concurrent readers.

Exit codes:

- **0:** Both the complete plan and success manifest were written.
- **2:** Valid input is outside the supported compilation subset. Only an
  `unsupported` manifest is written, containing `reason` and explanatory `detail`.
- **1:** Invalid command line, missing/unparseable/invalid MLIR, file I/O error,
  compiler pass/verifier error, or encoding error. Diagnostics go to stderr.
  There is no successful artifact or manifest from this invocation.

The input module must contain exactly one defined, single-block `func.func`
with one `!sym.tensor` input and one `!sym.tensor` result. Its body must be one
connected chain of `reloc.transpose`, `reloc.reshape`, and/or `reloc.pad`, and
`func.return` must consume the final chain result. Every value has exactly its
next chain/return use. An identity recipe can use an identity transpose; an empty
chain is unsupported. No unrelated operations, branches, extra functions,
pre-folded `reloc.plan_result`, fallback markers, or escaping intermediates are
accepted. Tensor ranks must be positive, static extents positive, and element
types one of `f32`, `f16`, and signless `i8`, unchanged throughout the chain.
The current symbolic tensor parser rejects textual rank-zero types as malformed
input (exit 1); the exporter additionally checks rank before invoking the fold.
Symbolic extents are guarded by the frontend/runtime at binding time.

The tool runs the real `reloc-fold` pass, verifies its output, and requires one
complete `reloc.plan_result` with no fallback or residual chain operations.
A transfer-function bail rejects the entire recipe. For example,
transpose-then-noncontiguous-merge, pad-then-split, and conflicting pad fills
produce `fold_unsupported`, never a partial plan. Test passes and diagnostic
scraping are not part of this interface.

Reason codes are `invalid_function_count`, `unsupported_signature`,
`unsupported_operation`, `prefolded_input`, `unsupported_descriptor`,
`unsupported_dtype`, `unsupported_expression`, `disconnected_chain`,
`empty_chain`, and `fold_unsupported`. Consumers should retain unknown future
reason codes as fallback reasons. Syntactically or verifier-invalid input is
an error (exit 1), even if its intended operation would be unsupported.

## Manifest schema 1

All manifests contain `schema_version: 1`, `wire_version: 0`, `status`, and
`compiler`, whose fields are `name: "sym-reloc-export"`, `interface_version: 1`,
`llvm_version`, and `build_identity` (configured source revision/build type).
Successful manifests additionally contain:

- `plan_count: 1`.
- `symbols`: names in exactly the encoded wire symbol-table order, supplied by
  the encoder itself. Bind positional symbol values using this order.
- `logical_source` and `logical_destination`: original function descriptors,
  each with `shape`, dense row-major `strides`, zero `offset`, and `dtype`
  (`"float32"`, `"float16"`, or `"int8"`). They retain logical rank even if the
  encoded plan coalesces contiguous axes.
- `constraints: {"divisibility": [{"expr": Expr, "divisor": integer}, ...]}`:
  all compiler divisibility constraints in emission order.
- `plan_sha256`: lowercase hex SHA-256 of the exact plan bytes.
- `input_sha256`: lowercase hex SHA-256 of the exact input MLIR bytes.

`Expr` uses tagged JSON arrays:

```text
["const", signed_i64]
["symbol", name]
["add", Expr, Expr]
["mul", Expr, Expr]
["floordiv", Expr, positive_signed_i64]
["mod", Expr, positive_signed_i64]
```

Subtraction is represented by addition and multiplication by `-1`. Dense
strides are suffix products of logical extents; their expression trees can
retain products of constants. Compiler simplification may reassociate or
simplify equivalent expressions. A bridge must compare expression semantics
structurally/algebraically within this vocabulary, not parse printed strings.

The frontend must validate **both digests** against its submitted input and
loaded bytes, as well as versions, logical descriptors, symbols and constraints.
Identical shapes cannot associate an artifact with a recipe: square transpose
and identity have identical logical descriptors but different input digests.
Whitespace changes also change the input digest. Metadata is deterministic
across separate invocations of the same compiler build and identical input:
no timestamps, file paths, or random identifiers occur in success manifests.

The manifest is not a portable frontend Recipe. Transfer direction, original
callable and provenance remain frontend data. Pad scalar values live in the
v0 plan with their exact scalar bits; consumers must not reconstruct them by
round-tripping decimal text from diagnostics.
