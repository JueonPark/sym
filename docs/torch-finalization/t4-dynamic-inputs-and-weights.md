# T4: Dynamic inputs and weight lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate symbolic plan reuse and safe repeated parameter/buffer
relocation, with reproducible installation and usage examples for [#131](https://github.com/JueonPark/sym/issues/131).

**Architecture:** Exercise T3's backend across valid dynamic bindings and add
an explicit weight-preparation owner that resolves live module slots on every
use. Cache transformed host data only for explicitly stable inference weights,
validate freshness, and reuse existing quantized prefolding after its typed
contract is available. Keep model mutation and optimizer ownership in PyTorch.

**Tech Stack:** CPython 3.14.7 (GIL enabled), PyTorch 2.14.0, pytest, pybind11, existing
`reloc::prefold`, R1/R2 adapter and C3/C4/R3 typed support.

**Spec:** [Project plan §1/T4](../project-finalization-plan.md),
[#131](https://github.com/JueonPark/sym/issues/131),
[shared contracts](README.md), [T3](t3-custom-op-and-replacement.md).

## Global Constraints

- All [shared constraints](README.md#shared-constraints) apply.
- Qualify CPython `3.14.7` (GIL enabled), PyTorch `2.14.0` CPU/cu126, and
  pybind11 `3.0.4`; see [the version baseline](README.md#selected-version-baseline).
- "Reuse is guaranteed only within the same supported recipe, rank/layout/dtype family, and validated shape constraints."
- "This does not promise zero recompilations for an arbitrary enclosing PyTorch model."
- "Weight updates cannot return stale data."
- "Parameters are supplied by the program or an explicit configuration; this work does not add calibration/training or silently choose lower precision."

---

**Proposed child title:** `[Finalize][Torch][T4] Validate dynamic reuse and manage weight lifecycle`

**Dependencies:** T3 and R1/R2 for layout-only examples. Quantized prefold
acceptance also requires C3/C4 of [#132](https://github.com/JueonPark/sym/issues/132)
and R3 of [#133](https://github.com/JueonPark/sym/issues/133). R4 consumes the
combined evidence. Deliver Tasks 1/2 without waiting for the typed path;
do not mark the full T4 prefold acceptance complete on that basis alone.

## File map

| Action | Path | Responsibility |
| --- | --- | --- |
| Create | `libreloc/python/reloc_torch/weights.py` | Live slot resolution, stable preparation, invalidation and close |
| Create | `libreloc/python/reloc_torch/prefold.py` | Typed capability check and owned existing-prefolder integration |
| Create | `libreloc/python/PyPrefold.h` | `registerPrefoldBindings(pybind11::module_ &)` declaration |
| Create | `libreloc/python/PyPrefold.cpp` | Owned, validated Python bridge to existing CPU prefolding |
| Create | `libreloc/python/tests/torch_frontend/test_dynamic.py` | Symbolic reuse, guard fallback and compilation counters |
| Create | `libreloc/python/tests/torch_frontend/test_weights.py` | Parameter/buffer mutation, replacement, aliases and lifecycle |
| Create | `libreloc/python/tests/torch_frontend/test_prefold.py` | Typed prefold conformance and resource ownership |
| Create | `libreloc/python/examples/torch_dynamic_transfers.py` | Input H2D and independent output D2H examples |
| Create | `libreloc/python/examples/torch_weight_loading.py` | Repeated module/buffer/parameter loading and explicit preparation |
| Create | `docs/torch-integration.md` | Build/install, activation, support boundaries and reproduction |
| Modify | `libreloc/python/reloc_torch/__init__.py` | Export `prepare_weights` |
| Modify | `libreloc/python/reloc_torch/diagnostics.py` | Preparation/invalidation counters |
| Modify | `libreloc/python/pyreloc/__init__.py` | Export the torch-free prefold binding after typed validation |
| Modify | `libreloc/python/PyReloc.cpp` | Invoke the new prefold registration function |
| Modify | `libreloc/python/CMakeLists.txt` | Compile the separate binding file |
| Modify | `libreloc/python/tests/torch_frontend/conftest.py` | Real compiler/runtime and supplied typed-parameter fixtures |
| Modify | `.github/workflows/build.yml` | Run the dynamic/weight/prefold tests in T1's cp314 job and publish versioned evidence |
| Modify | `libreloc/README.md`, `README.md`, `docs/torch-support.md` | Link the supported integration and evidence |

## Task 1: Prove dynamic rebinding without conflating compile counters

**Interfaces:** Consume T3 backend/cache counters and T2 symbolic recipes;
produce tests and `torch_dynamic_transfers.py`. No new plan format is needed.

- [ ] Add a GPU test that feeds one backend three valid shapes:

  ```python
  def test_dynamic_plan_reuse(backend):
      def fn(x):
          return x.reshape(x.shape[0] // 64, 64).t().contiguous().to("cuda")
      compiled_fn = torch.compile(fn, backend=backend, dynamic=True)
      start = backend.stats()
      with torch.no_grad():
          for n in (128, 192, 256):
              x = torch.arange(n, dtype=torch.float32)
              actual = compiled_fn(x)
              torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
      end = backend.stats()
      assert end["plan_compiles"] - start["plan_compiles"] == 1
      assert end["symbol_binds"] - start["symbol_binds"] == 3
      assert end["runtime_executions"] - start["runtime_executions"] == 3
      assert end["dynamo_compiles"] > start["dynamo_compiles"]
  ```

  Inspect the captured graph to verify dimensions really are symbolic. Record
  the actual Dynamo callback count without asserting that every enclosing
  model compiles once. If Dynamo emits extra graphs for the same family,
  artifact alpha-renaming/cache identity must still preserve one plan compile.
- [ ] Run the test before changing cache integration; a CPU variant uses
  T2's real compiler/binder/reference path and validates one artifact at the
  same bindings. Do not use a fake CUDA success to satisfy the GPU assertion.
- [ ] Correct symbolic canonicalization and binding only where these tests
  expose a concrete gap. The key must include rank, dtype/layout family,
  constant split factors and recipe order; it must not include sample hints.
  Bind on each real invocation in this first implementation so the counters
  have a simple reproducible meaning.
- [ ] Add an invalid-divisibility invocation (130 elements): original PyTorch
  may itself reject its reshape, and fallback must preserve that exception.
  Add a noncontiguous input for which PyTorch succeeds while the frontend
  falls back. Confirm zero invalid-plan launches. New dtype/rank/recipe is
  a distinct family or explicit exclusion; empty/scalar input is excluded.
- [ ] Implement the example with `--direction h2d|d2h`, `--sizes 128 192 256`
  and `--device cuda:0`. D2H compiles its requested forward computation from
  a CUDA root, independently of H2D. Print input/output metadata and separate
  plan/bind/Dynamo/execution/fallback counters. Commit as
  `test(torch): demonstrate guarded symbolic plan reuse`.

## Task 2: Prepare inference weights with conservative freshness checks

**Interfaces:** Implement `prepare_weights(module, recipes, *, backend,
stable=False) -> PreparedWeights`, `.get(name, *, device)`,
`.invalidate(name=None)`, `.close()` and context-manager methods.
`recipes` maps fully qualified module parameter/buffer names to T2 recipes.
The wrapper returns relocated tensors for explicit inference use; it does
not rewrite model slots, change `Parameter` identity or alter `requires_grad`.

- [ ] Add a mutation/replacement test, with `transpose_weight_recipe` defined
  in conftest as f32 shape `(4, 6)` → `(6, 4)` using one transpose:

  ```python
  def test_prepared_weight_follows_live_slot(backend, transpose_weight_recipe):
      module = torch.nn.Module()
      module.register_buffer("weight", torch.arange(24.).reshape(4, 6))
      with torch.no_grad(), prepare_weights(
              module, {"weight": transpose_weight_recipe}, backend=backend,
              stable=True) as prepared:
          first = prepared.get("weight", device="cuda")
          module.weight.add_(1)
          second = prepared.get("weight", device="cuda")
          torch.testing.assert_close(second, module.weight.t().contiguous().cuda(),
                                     rtol=0, atol=0)
          assert not torch.equal(first, second)
          module.weight = torch.full_like(module.weight, 9)
          third = prepared.get("weight", device="cuda")
          torch.testing.assert_close(third, torch.full_like(third, 9), rtol=0, atol=0)
  ```

  Mark the test GPU; CPU tests use the real host reference adapter for lifecycle
  mechanics. Add parameter tests as well as buffers and nested module names.
  Rerun slot replacement, tied-parameter and mutation-version behavior on
  PyTorch 2.14.0; these internals are part of the version audit. Test close,
  reference release and exception cleanup under CPython 3.14's regular GIL
  build without enabling free-threaded or subinterpreter support.
- [ ] Resolve live slots at every `get`, using a weak module reference plus
  fully qualified names. Check object identity, shape, stride, offset, dtype,
  device, storage identity and an available mutation version. Hold strong
  references for the duration of preparation/transfer. A dead module, removed
  slot or closed handle raises a clear error; it never returns an old tensor.
- [ ] Define explicit preparation as an inference API: under disabled grad,
  a normal Parameter can be read through a detached snapshot without altering
  the module's Parameter or its flag. With grad enabled and a gradient-requiring
  source, replay the supplied recipe through PyTorch and record `requires_grad`;
  do not detach it silently. `stable=False` always uses current values and caches
  plans only. `stable=True` may retain transformed host data after validation.
- [ ] Implement stable freshness conservatively. A version counter alone
  misses `.data` or NumPy writes. For eligible CPU weights, retain a byte
  snapshot and compare current source bytes on every reuse, as well as the
  identity/descriptor/version checks. Byte comparison also avoids treating
  unchanged NaNs as mutations. Inference tensors lacking version counters use
  the snapshot path; unsupported external storage/devices use no data caching.

  ```python
  current_bytes = source.detach().contiguous().view(torch.uint8)
  unchanged = metadata_matches and torch.equal(current_bytes, saved_bytes)
  if not unchanged:
      self.invalidate(name)
      saved_bytes = current_bytes.clone()
  ```

  `metadata_matches` includes live object identity and descriptor fields.
  Build a new materialization from an owned snapshot; compare version/bytes
  before publication and retry or use current PyTorch if changed. Concurrent
  external writes during a read are outside ordinary supported tensor usage;
  serialize wrapper `get/invalidate/close` calls with its own lock.
  This full scan is deliberate until a stronger mutation contract exists.
- [ ] Materialize layout-only stable weights into owned host storage with the
  existing CPU relocation executor, then send that completed layout using an
  identity transfer artifact through R2. Derive the transfer artifact from
  the **prepared** logical descriptor; never apply the transform twice.
  Return fresh destination storage on every `get`, so a caller's output
  mutation cannot poison the preparation cache. Track `weight_preparations`
  and `weight_invalidations` separately from plan compilation.
- [ ] Add cases for in-place mutation through an alias, `.data` mutation,
  `load_state_dict` copy/update and replacement, changed dtype/shape,
  registered buffers, tied parameters, slot deletion, explicit invalidation,
  repeated close and garbage collection. Preserve ties in the original module;
  skip automatic `Module.to` interception where PyTorch's parameter/subclass
  behavior cannot be preserved. Eager observation still accounts for those transfers.
  Commit as `feat(torch): manage prepared inference weight freshness`.

## Task 3: Reuse existing quantized prefolding behind the typed capability gate

**Interfaces:** Add `pyreloc.prefold_s8(bound, src_ptr, src_bytes, inv_scales_ptr,
inv_scales_bytes, *, output_spec, gather_threads=1) -> PrefoldHandle`.
`PrefoldHandle` has `.nbytes`, `.copy_to(dst_ptr, dst_bytes)`, `.close()` and
context-manager methods. These interfaces are new and contain no Torch types.
`output_spec` is exactly `s8_quant_pack` or `s8_gather_quant`.

This bridge wraps the existing `Prefold.h` API; it does not define new value
semantics. C3/C4/R3 must first establish the typed artifact/parameter contract.
C4 extends the frontend recipe with ordered typed transforms and declared
parameter bindings; this task consumes that extension and does not add a
second recipe format or infer parameter bindings from raw tensor values.
If R3 already provides an equivalent owned Python surface, use it and update
this file map/interface instead of adding a duplicate binding.

- [ ] Add a CPU conformance test using the C4 scalar reference and supplied
  quantization parameters. A valid typed recipe must produce byte-identical
  s8 data through normal typed execution and existing prefolding. Include
  clipping, ties, signed extremes and the declared non-finite behavior.
- [ ] Add the typed-to-prefold eligibility check in `prefold.py`: only match
  an implemented f32→s8 recipe with declared scale/zero-point/channel-axis
  semantics equivalent to the existing kernel. Existing kernels use inverse
  scales, zero-point zero and a particular channel layout; arbitrary zero
  points, reordered channels, pad fills or incompatible rounding/non-finite
  semantics cannot be passed through this bridge. Unsupported combinations
  use normal typed execution or original PyTorch with a reason.
- [ ] Implement `PyPrefold.cpp` with an owner whose member declaration order
  is backend, gather pool, then `PrefoldArtifact`, so destruction frees the
  artifact before its allocating backend. Use `HostBackend` initially;
  R2 can copy the prepared bytes into its own pinned staging. Validate source
  footprint, exact scale count, positive finite scales, bound dtype/rank,
  pad/stride constraints, byte overflow and output spec before calling a
  kernel with asserted preconditions. Release the GIL only after validation.

  ```cpp
  struct OwnedPrefold {
    reloc::HostBackend backend;
    reloc::GatherPool pool;
    reloc::prefold::PrefoldArtifact artifact;
  };
  ```

  Supply an explicit constructor using the existing backend/pool constructors
  and call `prefoldArtifact` only after the checks above. `copy_to` validates
  destination capacity and copies host bytes while holding the owner alive;
  `close` invalidates the artifact first and joins/releases pool resources.
  A failed or closed handle rejects copy access. Wire
  `registerPrefoldBindings` from `PyReloc.cpp` and stage exports via CMake.
- [ ] Copy the owned s8 image into an owned Torch host tensor for R2 transport
  under the prepared destination descriptor. Its dtype is s8 because the
  **user requested** quantization. Never enable this for ordinary float weight
  loading. Integrate with Task 2 freshness and include all supplied scale and
  zero-point tensors, axis/order, rounding settings and artifact identity in
  the preparation key. Snapshot parameter bytes just like weight bytes.
- [ ] Test scale/zero-point replacement and in-place mutation, channel-axis
  change, artifact eviction while a transfer is live, failed allocation,
  invalid scale lengths, and cleanup after a copy failure. Demonstrate one
  preparation across repeated unchanged uses and a rebuild after each real
  weight/parameter change. Missing typed capability must keep the layout-only
  examples runnable. Commit as `feat(torch): reuse validated quantized weight prefolding`.

## Task 4: Publish supported usage and final integration evidence

**Interfaces:** Ship the two runnable examples, `docs/torch-integration.md`,
and the final support matrix. Share these commands/results with R4; [#131](https://github.com/JueonPark/sym/issues/131)
completion requires all T1–T4 acceptance, not just documentation presence.

- [ ] Make `torch_weight_loading.py` demonstrate a parameter, a buffer, and
  repeated `load_state_dict` outside `forward`; print observer/fallback counters
  for ordinary module APIs. Add an explicit stable preparation example with
  repeated uses and a weight update, and a separately selected supplied-parameter
  quantized example after Task 3. Assert the updated numerical output in each.
- [ ] Write exact installation instructions for the pinned CPU and cu126
  environments, compiler/MLIR build, pybind11 and the optional package. Preserve
  the repository LLVM/MLIR `21.1.8` pin. Use an installed `uv` with support
  for the selected Python release, following its
  [managed Python instructions](https://docs.astral.sh/uv/guides/install-python/).
  Provision the regular CPython build explicitly, inside the container when
  running CI; a host interpreter path is not a container installation.
  These commands belong to T1's environment bootstrap as well as this guide:

  ```bash
  uv python install 3.14.7
  uv venv --python 3.14.7 /tmp/sym-torch-cpu
  uv pip install --python /tmp/sym-torch-cpu/bin/python 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/sym-torch-cpu/bin/python -r libreloc/python/requirements-torch-test.txt
  uv venv --python 3.14.7 /tmp/sym-torch-cuda
  uv pip install --python /tmp/sym-torch-cuda/bin/python 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python /tmp/sym-torch-cuda/bin/python -r libreloc/python/requirements-torch-test.txt
  ```

  T1's requirements file pins `pybind11==3.0.4` and uses Python-3.14-capable
  pytest/NumPy ranges. Verify the actual wheel variant and interpreter after
  both installs; write a resolved package/version record for each environment.
  On Linux the official wheels require glibc 2.28 or later; the existing
  Ubuntu 22.04 image meets that platform floor but its system Python does
  not supply the selected interpreter.
- [ ] Rebuild the runtime/Python extension in fresh CPU and CUDA build
  directories with the corresponding interpreter and pybind11 headers. Once
  `MLIR_DIR` and `LLVM_DIR` point to the existing LLVM/MLIR 21.1.8 installation,
  configure the CPU build as follows:

  ```bash
  export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
  export TORCH_BUILD="$PWD/build/torch-cpu"
  cmake -G Ninja -S . -B "$TORCH_BUILD" \
    -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" \
    -DPython_EXECUTABLE="$TORCH_PYTHON" \
    -DPYBIND11_FINDPYTHON=ON \
    -Dpybind11_DIR="$("$TORCH_PYTHON" -m pybind11 --cmakedir)" \
    -DRELOC_ENABLE_CUDA=OFF
  cmake --build "$TORCH_BUILD"
  ```

  Configure `build/torch-cuda` with `/tmp/sym-torch-cuda/bin/python`,
  `RELOC_ENABLE_CUDA=ON`, and `CUDAToolkit_ROOT`/`CMAKE_CUDA_COMPILER`
  pointing to the installed CUDA 12.6.3 toolkit. Retain libreloc's `75;89`
  CUDA architectures. Record the actual driver/toolkit versions and validate
  on both Turing and Ada; the PyTorch wheel does not install NVCC. Include
  the concrete R1 exporter configuration, R2-enabled invocation and lit
  location that actually passed in the final guide. R1/R2 own their APIs
  and must be complete by this task. Never mix the old cp310 extension,
  new cp314 headers, or CPU/CUDA build artifacts.
- [ ] Document callable backend activation, eager context scope, stable weight
  ownership/close, fallback counters, blocking behavior, dynamic-shape bounds,
  explicit quantization, and process-local compiled graph handles. Explain
  that eager observation cannot undo already executed layout operations.
  Show raw PyTorch usage outside the opt-in scope and torch-free core import.
- [ ] Run from a clean configured build: compiler/lit checks, CPU Torch and
  existing pyreloc suites, dynamic H2D/forward-D2H examples, repeated weight
  loading, typed prefold and CUDA stream/resource tests. Record actual
  versions, device, seed, pass/skip counts, executed paths, and artifact
  save/load evidence. Any unrun GPU/typed gate remains explicitly open.
- [ ] Update support rows from evidence; link this guide from both READMEs.
  Attach the evidence locations to the child issue drafts for later filing.
  Keep performance measurements descriptive and refer historical issue
  disposition to the project-level R4 handoff. Commit as
  `docs(torch): document supported dynamic transfers and weight loading`.

## Acceptance and handoff

- [ ] All qualification results use CPython 3.14.7, PyTorch 2.14.0, the
  selected CPU/cu126 build and pybind11 3.0.4; the guide records exact resolved
  test dependencies and fresh cp314 extension builds.
- [ ] Valid bindings share one symbolic plan; invalid guards never launch it.
- [ ] Plan compilation, symbol binding, Dynamo compilation, execution and
  weight preparation are separate counters.
- [ ] Input H2D, independent forward output D2H, and repeated parameter/buffer
  loading execute correctly with documented fallback boundaries.
- [ ] Stable data is never reused after observed weight/parameter mutation or
  replacement; byte snapshots cover writes missed by mutation versions.
- [ ] Existing prefolding is exercised for an explicitly requested, jointly
  validated quantized recipe; normal float transfers retain dtype and values.
- [ ] Backend/preparation handles, staging, pools and tensor references have
  deterministic cleanup and error tests.
- [ ] A fresh checkout can follow the install guide and reproduce all enabled
  support rows. R4 receives the final evidence and unresolved exclusions.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend/test_dynamic.py libreloc/python/tests/torch_frontend/test_weights.py libreloc/python/tests/torch_frontend/test_prefold.py -m 'not gpu' -q
# On the pinned CUDA environment with R1/R2 and typed prerequisites installed:
export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m gpu -q
"$TORCH_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py --direction h2d --sizes 128 192 256 --device cuda:0
"$TORCH_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py --direction d2h --sizes 128 192 256 --device cuda:0
"$TORCH_PYTHON" libreloc/python/examples/torch_weight_loading.py --device cuda:0
```
