# T2: FX import and symbolic guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate eligible FX transfer/layout regions into verified symbolic
relocation plans with sufficient metadata and guards to execute or fall back safely.

**Architecture:** Normalize inventoried FX targets, extract a pure linear
region, and translate its index semantics into a small frontend recipe IR.
Emit symbolic MLIR and use R1's supported exporter; retain an untouched
callable for the original region and preserve logical tensor metadata apart
from the runtime's coalesced axes.

**Tech Stack:** CPython 3.14.7 (GIL enabled), PyTorch 2.14.0 FX/FakeTensor/SymInt, MLIR reloc/sym,
`sym-opt`, existing v0 `encodePlan`, pyreloc, pytest and lit.

**Spec:** [Project plan §1/T2](../project-finalization-plan.md),
[#131](https://github.com/JueonPark/sym/issues/131),
[shared contracts](README.md), [T1](t1-transfer-inventory.md).

## Global Constraints

- All [shared constraints](README.md#shared-constraints) apply.
- CPython `3.14.7` (GIL enabled) and PyTorch `2.14.0` CPU/cu126 are the
  qualification target; see [the version baseline](README.md#selected-version-baseline).
- "A failed correctness guard routes to the original graph before executing an invalid plan; the standalone runtime continues to report a bind error."
- "Existing compiler bail cases remain unsupported rather than being assumed foldable."
- "No premature conversion of symbolic sizes to Python integers."
- Compiler/runtime wire v0 remains sufficient for M1; typed import waits for C4/R3.

---

**Proposed child title:** `[Finalize][Torch][T2] Import FX relocation recipes with symbolic guards`

**Dependencies:** T1's metadata/classification contract. Recipe and folding
tests can begin immediately; public artifact acceptance requires R1 in
[#133](https://github.com/JueonPark/sym/issues/133). T3 consumes this output.

## File map

| Action | Path | Responsibility |
| --- | --- | --- |
| Create | `libreloc/python/reloc_torch/recipe.py` | Expression/operation/descriptor records and canonical recipe identity |
| Create | `libreloc/python/reloc_torch/fx_import.py` | Raw FX normalization, candidate extraction and original-region preservation |
| Create | `libreloc/python/reloc_torch/symbolic.py` | SymInt expression conversion, symbol provenance and correctness guards |
| Create | `libreloc/python/reloc_torch/mlir_emit.py` | Deterministic symbolic MLIR generation |
| Create | `libreloc/python/reloc_torch/compiler.py` | R1 exporter bridge and compiled recipe validation |
| Create | `libreloc/python/tests/torch_frontend/test_fx_import.py` | Region safety, op lowering and metadata semantics |
| Create | `libreloc/python/tests/torch_frontend/test_symbolic.py` | Symbol mapping and guarded bindings |
| Create | `libreloc/python/tests/torch_frontend/test_compiler.py` | Real compiler/export/load/bind/reference integration |
| Create | `test/dialect/reloc/torch_import.mlir` | Imported-recipe folding and bail regressions |
| Modify | `libreloc/python/reloc_torch/compat.py` | Pin-sensitive FX metadata/normalization access |
| Modify | `libreloc/python/tests/torch_frontend/conftest.py` | Small canonical recipes and compiler fixture |
| Modify | `docs/torch-support.md` | Import/fold support, separate from execution support |

Read existing `RelocOps.td`, `PlanBuilder.h/.cpp`, `RelocFold.cpp`,
`RelocSerialization.h`, `test/dialect/reloc/fold_bail.mlir`, and
`libreloc/python/tests/oracle.py` before implementation. R1 owns changes to
the compiler tool/export API; T2 does not add diagnostic scraping.

## Task 1: Select regions with explicit layout and alias semantics

**Interfaces:** Produce `normalize_graph`, `import_graph`, and `Recipe`.
`ImportReport` contains `candidates: tuple[Candidate, ...]` and
`exclusions: tuple[Exclusion, ...]`. `Exclusion` records FX node name and
reason. A `Candidate` owns its source/tail node names, all member names,
recipe, symbolic source bindings and an extracted original-region callable.

- [ ] Add tests for two successful regions:

  ```python
  def layout_before_transfer(x):
      return x.transpose(0, 1).contiguous().to("cuda")

  def layout_after_transfer(x):
      return x.to("cuda").transpose(0, 1).contiguous()
  ```

  Obtain their ATen graphs with symbolic/fake capture through the pinned
  compatibility helper. Audit the installed 2.14.0 `make_fx`, FakeTensor and
  symbolic-shape APIs first; consume T1's newly collected overload mappings.
  Test raw Dynamo graph normalization independently
  using T1's capture callback. Assert graph text and node users are unchanged
  after `import_graph`, with exactly one transfer in each accepted region.
- [ ] Run `test_fx_import.py` and confirm failure before adding the importer.
- [ ] Implement frozen recipe records: `TensorSpec(shape, strides, offset,
  dtype)`, `Transpose(perm)`, `Reshape(shape)`, `Pad(axis, lo, hi, fill)`, and
  `Recipe(source, operations, destination, direction)`. Shapes/strides/offset
  use Task 2's `Expr`; fill includes a dtype and exact scalar bit pattern.
  Canonical identity includes all operation order, descriptor and fill data.
- [ ] Normalize only the raw targets T1 observed. Handle `call_method` forms
  of `to`, `cpu`, `cuda`, `permute`, `transpose`, `view`, `reshape`, and
  `contiguous`, plus their recorded function/ATen equivalents. Resolve
  overload arguments through schemas; constants stay constants. Use fake
  metadata propagation, never actual cross-device execution to discover a
  graph. Unsupported normalization produces `unrecognized_fx_target`.
- [ ] Walk backward/forward from a transfer through allowed layout nodes.
  Require a dense zero-offset root, a fresh contiguous zero-offset region
  output, one transfer, no gradients and no mutation. Reject regions with
  escaping intermediate tensors, writes through a known alias, unknown
  side effects between members, or tensor subclasses. Scalar metadata users
  can be retained only if independent of erased tensor nodes; otherwise reject.
  Use schema alias information conservatively; do not assume single-use means
  no mutation. Multi-transfer chains are excluded as `multiple_transfers`.
- [ ] Encode supported operations with these exact rules:

  | Torch operation | Recipe treatment |
  | --- | --- |
  | `permute` / `transpose` | Normalize negative dimensions and construct a complete permutation; emit `Transpose`. |
  | `view` / `reshape` | Emit row-major `Reshape` only when alias/materialization analysis and compiler folding prove it. Constant-factor `-1` inference is handled in Task 2. |
  | `contiguous` / `clone(contiguous_format)` | A physical materialization boundary; preserve its effect in region semantics and demand contiguous output. Do not independently remove a visible view or alias. |
  | constant pad | Expand Torch's reversed pad-pair order to per-axis `Pad`; accept nonnegative compile-time widths and a representable constant fill of the requested dtype. |
  | transfer | Record direction/copy contract; layout MLIR contains the surrounding mapping, not a fake device operation. |

  A bare `x.permute(...).to('cuda')` may preserve noncontiguous strides;
  it is excluded if its metadata differs from a dense output. A view after
  an otherwise independent transfer can stay in the surrounding graph only
  if it is outside the selected region. Never relabel its stride as contiguous.
- [ ] Extract an original callable with arguments `(src, *symbol_values)`
  using `torch.fx.Graph.node_copy` and an explicit node mapping. It reproduces
  only the pure selected region, including its original transfer, options,
  and exceptions. Test it against the original function. Rejected regions
  remain wholly untouched. Commit as `feat(torch): import safe FX relocation regions`.

## Task 2: Preserve symbolic expressions and emit compiler-verifiable IR

**Interfaces:** Produce `Expr`, canonical symbol mapping, `emit_mlir(recipe)`,
and binding guards. Symbol names `s0`, `s1`, ... follow first independent
source-dimension occurrence, independent of Dynamo's generated symbol names.
`SymbolSource` identifies source axis and any equality with other axes.

- [ ] Add symbolic tests using a root of shape `[N]` and a constant split
  factor 64. Build one recipe and compare its canonical identity under fake
  captures with different hints. Assert symbol provenance is retained and
  no branch concretizes a `SymInt` through `int`, `bool`, `.item()`, or its hint.
  Run these tests in the new cp314 environment; earlier successful tracing
  on the old environment is not evidence for this contract. Keep the current
  `dynamic=True`/constant-factor design; adopting additional 2.14 dynamic-shape
  APIs is optional and cannot broaden the supported shape semantics implicitly.
- [ ] Define `Expr` as a tagged immutable tree with `Const(int)`, `Symbol(str)`,
  `Add(lhs, rhs)`, `Mul(lhs, rhs)`, `FloorDiv(lhs, positive_constant)`, and
  `Mod(lhs, positive_constant)`. Subtraction lowers to addition with a
  negative constant. Convert only this expression vocabulary through
  `compat.py`; reject unknown functions, data-dependent/unbacked symbols,
  symbolic divisors and noninteger expressions as `unsupported_symbolic_expr`.
  Use structural expression matching, not string `eval` or regex arithmetic.
- [ ] Derive dense stride expressions symbolically. Only accept incoming
  stride/offset expressions provably equal to the initial dense/zero-offset
  contract; preserve other layouts for fallback. For one inferred reshape
  dimension, divide the source element count by the constant product of the
  other dimensions and emit divisibility/equal-element-count guards. Reject
  multiple `-1` dimensions and dynamic split factors.
- [ ] Emit this concrete compiler fixture, with output logical shape
  `[64, N floordiv 64]` stored independently of folded/coalesced axes:

  ```mlir
  func.func @torch_split_transpose(%x: !sym.tensor<["N"], f32>)
      -> !sym.tensor<[64, "N" floordiv 64], f32> {
    %r = reloc.reshape %x to [N floordiv 64, 64]
        : !sym.tensor<["N"], f32>
        -> !sym.tensor<["N" floordiv 64, 64], f32>
    %t = reloc.transpose %r perm [1, 0]
        : !sym.tensor<["N" floordiv 64, 64], f32>
        -> !sym.tensor<[64, "N" floordiv 64], f32>
    return %t : !sym.tensor<[64, "N" floordiv 64], f32>
  }
  ```

  The emitter generates canonical frontend symbols (`s0`); this human-readable
  fixture uses `N`. For a transfer without layout changes emit an identity
  transpose so the existing folding path still produces one plan result.
- [ ] Add guard evaluation with exact integer arithmetic: positive extents,
  repeated-symbol equality, source descriptor match, element-count equality,
  divisibility and valid padding. At the real runtime boundary, validate
  signed-i64 range and overflow before calling the standalone binder. Alignment
  remains a runtime strategy downgrade, not a correctness exclusion.
- [ ] Add lit checks requiring one `reloc.plan_result` and the divisibility
  constraint for the fixture. Add noncontiguous transpose→merge and
  pad→split fixtures requiring `reloc.fallback` and unchanged operations;
  also cover conflicting repeated pad fills and escaping intermediate users.
- [ ] Run `sym-opt --reloc-fold` over the new fixture and the existing
  `fold_bail.mlir`; run `test_symbolic.py` and `ninja -C "$TORCH_BUILD" check-sym`.
  Commit as `feat(torch): preserve symbolic relocation recipes and guards`.

## Task 3: Consume R1 artifacts and prove compiler/runtime agreement

**Interfaces:** Produce `CompilerClient.compile`, `CompiledRecipe`,
`CompiledRecipe.bind_values`, and `UnsupportedRecipe(reason, detail)`.
`CompiledRecipe` owns canonical recipe, plan bytes, compiler identity, wire
version, ordered symbol names/provenance, constraints, and **logical**
source/destination descriptors. The original callable stays in `Candidate`,
not in a portable compiler artifact.

- [ ] Establish the R1 adapter test before implementation: compile the same
  recipe twice in separate processes and compare artifact bytes and manifest
  fields. Export must identify exactly one successfully folded plan and reject
  any `reloc.fallback` member. A zero process exit code alone cannot pass it.
- [ ] Implement `CompilerClient` over R1's agreed public API. If it is a tool,
  invoke an argument vector with `subprocess.run(..., check=False,
  capture_output=True)` and an owned temporary directory; validate return
  code, artifact/manifest presence and their consistency. Map declared
  unsupported cases to `UnsupportedRecipe`; preserve unexpected compiler
  crashes as actionable errors. Never parse `--test-reloc-utils` remarks.
- [ ] Cross-check manifest wire version, exact symbol set/order, element
  types, source/destination descriptor expressions and constraints against
  the frontend recipe before admitting the artifact. Reject stale/mismatched
  manifests. Test malformed export, missing symbols, and mismatched dtype.
- [ ] Add a real CPU reference test (the `compiler` fixture selects the
  configured R1 exporter; `split_transpose_recipe` constructs the Task 2
  recipe with its source axis bound to `s0`):

  ```python
  def test_one_artifact_multiple_bindings(compiler, split_transpose_recipe):
      import numpy as np
      import pyreloc
      from pyreloc.torch_interop import as_ptr
      compiled = compiler.compile(split_transpose_recipe)
      plan = pyreloc.load_plan(compiled.plan_bytes)
      for n in (64, 128, 192):
          x = torch.arange(n, dtype=torch.float32)
          symbols = compiled.bind_values(x)
          bound = pyreloc.bind(plan, symbols)
          actual = np.empty(n, dtype=np.float32)
          pyreloc.relocate(bound, *as_ptr(x), *as_ptr(actual))
          expected = x.reshape(n // 64, 64).t().contiguous().numpy()
          np.testing.assert_array_equal(actual.reshape(expected.shape), expected)
  ```

- [ ] Add direct invalid binds for `N=0`, `N=130`, missing/extra symbols,
  overflow, and a repeated-symbol equality violation. The standalone call
  must raise `pyreloc.BindError` where its contract applies. Test frontend
  guard rejection separately; T3 owns proof that no runtime launch follows.
  Include source/target shape rank restoration after bind-time coalescing.
- [ ] Cover identity transfer, 2-D transpose, static merge/split, symbolic
  split, constant pad, transpose+pad, and f32/f16/i8 layout-only recipes.
  Compare scalar values/bytes to NumPy or original Torch independently of
  compiler index logic. Include shape-varying cases and save/load in a fresh
  process. Commit as `feat(torch): consume verified compiler artifacts`.

## Acceptance and handoff

- [ ] Static and symbolic accepted FX chains produce verified plans through R1.
- [ ] Symbolic extents/strides are preserved until real binding; symbol sets,
  equality/divisibility constraints, and output metadata survive serialization.
- [ ] Compiler bail, unsupported nodes, noncontiguous roots, mutation, and
  escaping intermediate users preserve the original region with reasons.
- [ ] Layout semantics include materialization and constant-pad fill values;
  metadata-only views are not silently converted into physical relocation.
- [ ] T3 receives original-region callables, logical output descriptors,
  stable recipe keys, artifacts, guards, and declared unsupported errors.
- [ ] Missing R1 is reported as an unmet acceptance dependency, not hidden by
  a corpus-only test or a mock compiler.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend/test_fx_import.py libreloc/python/tests/torch_frontend/test_symbolic.py libreloc/python/tests/torch_frontend/test_compiler.py -m 'not gpu' -q
ninja -C "$TORCH_BUILD" check-sym
```

Configure the `compiler` fixture with the public executable/binding that R1
ships. That is a required integration dependency; no new exporter command is
claimed to exist by this plan. Update the fixture's exact invocation and the
support guide when R1's interface lands.
