# Supported plan export (interface 1)

Build the compiler target `sym-reloc-export`, then invoke:

```sh
# Layout-only chain: wire format v0 plan, manifest schema 1 (R1).
sym-reloc-export recipe.mlir --output plan.bin --manifest manifest.json
# Chain that may contain typed value transforms: wire format v1 typed plan,
# manifest schema 2 (C3, issue #143). Layout-only input is unchanged.
sym-reloc-export recipe.mlir --output plan.bin --manifest manifest.json --typed
```

This compiler-only executable links MLIR/LLVM and has no Torch dependency.
`libreloc` remains MLIR/LLVM/Torch-free. `plan.bin` is exactly `encodePlan`
wire format v0 for a layout-only chain and exactly `encodeTypedPlan` wire
format v1 for a typed chain, both documented in
[reloc-plan-format.md](reloc-plan-format.md). The exporter does not bind
symbols or parameters, execute transfers, or replace a frontend callable.

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
connected chain of `reloc.transpose`, `reloc.reshape`, and/or `reloc.pad`, plus,
under `--typed`, `reloc.cast`, `reloc.quantize` and/or `reloc.dequantize`, and
`func.return` must consume the final chain result. Every value has exactly its
next chain/return use. An identity recipe can use an identity transpose; an empty
chain is unsupported. No unrelated operations, branches, extra functions,
pre-folded `reloc.plan_result` / `reloc.typed_plan_result`, fallback markers, or
escaping intermediates are accepted. Tensor ranks must be positive, static
extents positive, and element types one of `f32`, `f16`, and signless `i8`,
unchanged throughout the chain except at a typed value transform, whose result
type is the transform's output type (C1 rules; the verifier rejects illegal
pairs). Stage parameters are `f32`/`f16`/`i8` scales and `i32` (or `i8`) zero
points. The current symbolic tensor parser rejects textual rank-zero types as
malformed input (exit 1); the exporter additionally checks rank before invoking
the fold. Symbolic extents and runtime parameters are guarded by the
frontend/runtime at binding time.

The tool runs the real `reloc-fold` pass, verifies its output, and requires one
complete `reloc.plan_result` (layout-only) or `reloc.typed_plan_result`
(typed, `--typed` only) with no fallback or residual chain operations.
A transfer-function bail rejects the entire recipe. For example,
transpose-then-noncontiguous-merge, pad-then-split, conflicting pad fills and
the typed bails of [reloc-typed-folding.md §3](reloc-typed-folding.md#3-rejection-boundaries)
(a pad crossing a per-channel or runtime-scaled quantize) produce
`fold_unsupported`, never a partial plan. Test passes and diagnostic scraping
are not part of this interface.

Reason codes are `invalid_function_count`, `unsupported_signature`,
`unsupported_operation`, `prefolded_input`, `unsupported_descriptor`,
`unsupported_dtype`, `unsupported_expression`, `disconnected_chain`,
`empty_chain`, `fold_unsupported`, and `typed_unsupported`. Consumers should
retain unknown future reason codes as fallback reasons. Syntactically or
verifier-invalid input is an error (exit 1), even if its intended operation
would be unsupported.

`typed_unsupported` marks a verifier-valid chain that contains a typed value
transform (`reloc.cast`, `reloc.quantize`, `reloc.dequantize`) submitted
**without** `--typed`: the layout-only interface (schema 1, wire format v0)
never encodes a value transform, so an R1 consumer that does not know the flag
keeps receiving exactly the manifests it always did. Their semantics are
defined in [reloc-typed-semantics.md](reloc-typed-semantics.md) (C1, issue
#141), their folded representation `#reloc.typed_plan` in
[reloc-typed-folding.md](reloc-typed-folding.md) (C2, issue #142), and the
artifact in the schema 2 section below (C3, issue #143). Under `--typed`,
`unsupported_expression` additionally covers parameter binding extents and
channel maps outside the manifest vocabularies.

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

## Manifest schema 2 (typed plans, `--typed`)

A successful typed export publishes `schema_version: 2`, `wire_version: 1`,
the same `status`, `compiler`, `plan_count`, `symbols`, `logical_source`,
`logical_destination`, `constraints`, `plan_sha256` and `input_sha256` fields
as schema 1 (the two logical descriptors now carry the chain's source and
result dtypes, which differ), plus three sections describing the typed plan:

- `stages`: the value stages in program order, each an object with
  `transform` (`"cast"`, `"quantize"`, `"dequantize"`), `policy` (`"ieee_rne"`,
  `"exact"`, `"symmetric_rne"`, `"affine"`), `input_dtype`, `output_dtype`,
  `shape` (the stage operand's logical shape, `Expr` list; the shape `axis`
  refers to), `scale` and `zero_point` (`Param` or `null`), `axis` (`-1` per
  tensor) and `channel` (`null` per tensor, otherwise
  `{"dims": result_rank, "expr": ChannelExpr}`).
- `fills`: the original pad fills, each `{"dst_axis", "stage", "dtype",
  "bits"}`, where `stage` is the boundary the pad entered at (0 = before the
  first stage) and `bits` the exact bit pattern of the original value.
- `parameters`: the runtime parameter declarations in first-declaration
  order, one per name, each `{"name", "dtype", "extents": [Expr, ...]}`. Every
  one must be supplied by name to `pyreloc.bind_typed`.

`Param` is either an inline constant or a binding:

```text
{"kind": "inline",  "dtype": D, "shape": [] | [channels], "bits": [hex, ...]}
{"kind": "binding", "name": name, "dtype": D, "extents": [Expr, ...]}
```

`D` is `"float32"`, `"float16"`, `"int8"` or `"int32"`. Every scalar (`bits`
of parameters and fills) is a **lowercase hex string of the exact bit
pattern** (two's complement for integers), never decimal text; the wire blob
carries the same bits.

`ChannelExpr` is a vocabulary of its own: channel maps are affine over the
logical **result** coordinates and the plan symbols, and their divisors may be
symbolic:

```text
["const", signed_i64]
["dim", index]            # 0 <= index < dims (logical result rank)
["symbol", name]          # a name from `symbols`
["add", ChannelExpr, ChannelExpr]
["mul", ChannelExpr, ChannelExpr]
["floordiv", ChannelExpr, ChannelExpr]
["mod", ChannelExpr, ChannelExpr]
```

A bridge validates the channel expression structurally (tags, dimension
range, symbol membership); evaluating it is the runtime's job. The frontend
admission for schema 2 checks, on top of schema 1's checks: the stage
sequence against the recipe's value transforms (transform, policy, dtypes,
operand shape, axis, parameters), the fills against the recipe's pads (count,
dtype and bits), the parameter declarations against the recipe's bindings,
and each of those against the plan the runtime decoded with
`pyreloc.load_typed_plan`. It never routes a typed plan through the v0
decoder, nor a v0 plan through the typed one.

Unsupported manifests always use `schema_version: 1` / `wire_version: 0`,
whichever flag produced them: they describe no plan, and every consumer of
either interface reads them.

## Compatibility matrix

| Input chain | Flag | Exit | Plan | Manifest |
|---|---|---|---|---|
| layout-only, foldable | none | 0 | wire v0 | schema 1 |
| layout-only, foldable | `--typed` | 0 | wire v0, **byte-identical** to the row above | schema 1, identical |
| contains a value transform | none | 2 | none | schema 1, `typed_unsupported` |
| contains a value transform, foldable | `--typed` | 0 | wire v1 | schema 2 |
| typed fold bail | `--typed` | 2 | none | schema 1, `fold_unsupported` |
| pre-folded `plan_result` / `typed_plan_result` | either | 2 | none | schema 1, `prefolded_input` |
| verifier-invalid | either | 1 | none | none |

Consumers: a runtime built before C3 (`libreloc 0.1`) rejects every wire v1
blob at byte offset 4 with `unsupported wire format version` and knows no
schema 2 (this was measured against the pre-change `pyreloc` before the
runtime was rebuilt); a C3 runtime decodes v0 with `load_plan` and v1 with
`load_typed_plan`, and `wire_version(bytes)` tells them apart. A frontend
compiled recipe is portable as `format_version` 1 (layout-only) or 2 (typed);
a C3 frontend loads both, an older frontend rejects format 2 by version.
