# T3: Guarded custom transfer and graph replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute eligible compiled and eager CPU–GPU transfers through one
custom-op/runtime adapter, preserving metadata, fallback and lifetime semantics.

**Architecture:** Register a functional Torch custom op with explicit symbolic
output metadata, then replace only the pure regions accepted by T2. A common
executor preflights each invocation and runs its saved original region when
an expected guard or capability check fails. R2 owns tensor allocation,
forward H2D/D2H transport, stream ordering and completion.

**Tech Stack:** CPython 3.14.7 (GIL enabled), PyTorch 2.14.0 `torch.library`/FX/Dynamo,
pyreloc, R1 artifacts, R2 storage/stream adapter, pytest and CUDA validation.

**Spec:** [Project plan §1/T3](../project-finalization-plan.md),
[#131](https://github.com/JueonPark/sym/issues/131),
[shared contracts](README.md), [T2](t2-fx-import-and-guards.md).

## Global Constraints

- All [shared constraints](README.md#shared-constraints) apply.
- Use CPython `3.14.7` with the GIL, PyTorch `2.14.0` CPU/cu126, and the
  rebuilt cp314 extension from [T1](t1-transfer-inventory.md).
- "Fake kernels must match the real output metadata, and opcheck does not replace numerical testing."
- "copy_ requires mutation semantics; an out-of-place custom op cannot silently replace it."
- "A same-layout CPU→GPU transfer still requires data movement; a layout no_copy flag cannot turn it into a cross-device view."
- "The documented blocking path works first; nonblocking calls are offloaded only when completion and lifetime guarantees are implemented."

---

**Proposed child title:** `[Finalize][Torch][T3] Replace eligible transfers with a guarded custom op`

**Dependencies:** T1/T2 and R1/R2 in [#133](https://github.com/JueonPark/sym/issues/133).
Mocks can unblock control-flow tests but cannot close this issue. Typed
custom ops, autograd formulas and Inductor layout assignment are outside M1.

## File map

| Action | Path | Responsibility |
| --- | --- | --- |
| Create | `libreloc/python/reloc_torch/runtime.py` | R2 adapter bridge and common guarded execution |
| Create | `libreloc/python/reloc_torch/cache.py` | Bounded artifact cache and live graph registration ownership |
| Create | `libreloc/python/reloc_torch/diagnostics.py` | Compilation/binding/execution/fallback counters |
| Create | `libreloc/python/reloc_torch/ops.py` | Functional custom op and fake implementation |
| Create | `libreloc/python/reloc_torch/backend.py` | Backend object and safe graph replacement |
| Create | `libreloc/python/reloc_torch/eager.py` | Scoped eager transfer replacement and reentrancy guard |
| Create | `libreloc/python/tests/torch_frontend/test_runtime.py` | Preflight and fallback behavior |
| Create | `libreloc/python/tests/torch_frontend/test_custom_op.py` | Registration/fake metadata checks |
| Create | `libreloc/python/tests/torch_frontend/test_backend.py` | Graph rewriting and eager safety |
| Create | `libreloc/python/tests/torch_frontend/test_cache.py` | Keys, eviction, ownership and counters |
| Create | `libreloc/python/tests/torch_frontend/test_transfers_gpu.py` | Actual transfers, stream ordering and allocations |
| Modify | `libreloc/python/reloc_torch/__init__.py` | Export explicit backend and eager activation APIs |
| Modify | `libreloc/python/tests/torch_frontend/conftest.py` | Counting adapters and real runtime fixture |
| Modify | `docs/torch-support.md` | Enable only jointly validated rows |

R2 owns changes to storage validation, stream bridges, `PyReloc.cpp`, and
CUDA transport. In particular it must not expose inverse scatter as ordinary
forward D2H. No Torch headers/dependencies enter `reloc_runtime`.

## Task 1: Preflight execution, own caches, and make fallback observable

**Interfaces:** Implement the shared `RuntimeAdapter.preflight/execute`,
`PreparedCall`, `RelocBackend.stats`, and
`execute_or_fallback(entry, src, symbols, device, *, non_blocking=False)`.
Here `symbols` is the ordered list of concrete values supplied by the op;
preflight creates the exact name-to-value map expected by `pyreloc.bind`.
An execution entry contains its compiled recipe, saved original callable,
adapter and diagnostics. `PreparedCall` retains src, validated bindings,
bound plan, destination descriptor and R2's validated request until completion.

- [ ] Add a counting-adapter test proving guard failure occurs before a
  launch and runs the saved callable once:

  ```python
  def test_guard_failure_never_launches(entry, counting_runtime):
      x = torch.arange(12, dtype=torch.float32)[::2]
      actual = execute_or_fallback(entry, x, [], torch.device("cpu"))
      assert torch.equal(actual, x.clone())
      assert counting_runtime.executions == 0
      assert entry.diagnostics.fallbacks["unsupported_layout"] == 1
      assert entry.fallback_calls == 1
  ```

  Define `entry` in conftest as a dense-input identity recipe with a counting
  `src.clone()` fallback and a CPU-only test adapter. This tests control flow;
  same-device copies remain ineligible in the production frontend. The test
  adapter must fail preflight for stride `(2,)`, not simulate CUDA success.
- [ ] Run the new test and confirm failure before implementing common execution.
- [ ] Implement the expected-error boundary as follows; define
  `entry.fallback(src, *symbols)` using T2's ordered symbol list and suspend
  eager interception around both branches:

  ```python
  try:
      call = entry.runtime.preflight(
          entry.compiled, src, device, non_blocking=non_blocking)
  except UnsupportedRecipe as exc:
      entry.diagnostics.fallbacks[exc.reason] += 1
      return entry.fallback(src, *symbols)
  return entry.runtime.execute(call)
  ```

  Translate expected descriptor/correctness `BindError` into a reason before
  execution. Do not catch arbitrary execution errors and rerun a region after
  work has launched. Surface execution failures with recipe/direction context.
  Keep direct `pyreloc.bind` failure behavior unchanged. Preflight must reconcile
  explicit op symbols with values read from real source metadata.
- [ ] Implement a bounded LRU (default 128 entries) for artifacts keyed by
  canonical recipe, frontend/compiler identity, wire version, source/output
  dtype/layout family, transfer semantics and runtime capability identity.
  Exclude concrete values of supported dynamic dimensions; include constant
  split factors and requested value transforms. Never cache by pointer.
  Unsupported-recipe caching uses the same semantic/version identity and
  bounded capacity; a new runtime/compiler capability cannot reuse a stale rejection.
- [ ] Keep graph execution handles separate from artifact keys. A thread-safe
  process-local registry maps each live handle to its region entry. The
  returned graph callable owns registrations strongly and releases them when
  destroyed; cache eviction cannot invalidate a live graph. Backend `.close()`
  invalidates use explicitly, releases resources, and is idempotent. Handles
  are not a portable serialized model format; only R1 artifacts are portable.
- [ ] Count `dynamo_compiles` at backend callback entry, `plan_compiles` only
  for real compiler calls, `symbol_binds` for actual binder calls,
  `cache_hits`, `runtime_executions`, and `fallbacks` by reason. Increment
  execution only on actual dispatch. Snapshots must not hold tensor references.
  Test eviction, concurrent same-key compilation, graph lifetime, close,
  and dtype/device/recipe/version key separation. Commit as
  `feat(torch): add guarded runtime execution and owned plan cache`.

## Task 2: Register a functional transfer op with exact fake metadata

**Interfaces:** Register this layout-only schema once at module initialization:

```text
reloc_torch::transfer(Tensor src, str handle, SymInt[] symbols,
                     SymInt[] out_shape, SymInt[] out_strides,
                     Device device) -> Tensor
```

Output dtype equals `src.dtype`, offset is zero, and output never aliases
input. Explicit `SymInt[]` arguments preserve dynamic shape dependencies
without a fake kernel consulting process-local runtime objects.

- [ ] Add schema/fake tests on the CPU test adapter and numerical GPU tests
  on R2. Assert actual and fake shape, stride, dtype, device index and storage
  offset for identity and transpose/split output. Test that identity H2D
  allocates CUDA storage even when `compiled.no_copy` is true.
- [ ] Define the custom op with `torch.library.custom_op`, `mutates_args=()`,
  the explicit schema above and a real body that retrieves the live handle,
  validates declared output metadata against the compiled descriptor, and
  calls Task 1's common executor. Do not cache output tensors. Guard unsupported
  gradient-requiring direct calls; T3's public entry points fall back before
  reaching the custom op in that case.
  Use the [2.14 registration contract](https://docs.pytorch.org/docs/2.14/library.html)
  and test registration/import in CPython 3.14, including its deferred
  annotations. Keep the explicit schema as the source of symbolic argument
  types; do not infer 2.14 behavior from the previous local Torch installation.
- [ ] Register a fake kernel containing only metadata operations:

  ```python
  @transfer.register_fake
  def transfer_fake(src, handle, symbols, out_shape, out_strides, device):
      return torch.empty_strided(
          out_shape, out_strides, dtype=src.dtype, device=device)
  ```

  Never call `data_ptr`, compiler export, runtime binding, GPU initialization
  or cache lookup in this kernel. R2 real outputs must have precisely these
  strides and zero offset. Reject an imported region if its original fallback
  cannot provide the same output metadata over the accepted shape family.
- [ ] Use `torch.library.opcheck(torch.ops.reloc_torch.transfer.default, args)`
  with inference tensors on both supported source device types. Run all default
  registration checks and document results; use separate PyTorch comparisons
  for numerical correctness. Test dynamic fake outputs with at least two symbol
  bindings and ensure fake execution changes no runtime counters.
  Record the actual default check set/results from 2.14.0. Run CPU and CUDA
  checks in their corresponding cp314 builds; full registration evidence
  cannot consist of syntax checks or old-version opcheck results.
- [ ] Verify calls with a closed/unknown execution handle fail clearly and do
  not fabricate an output. Test reload/duplicate import registration behavior.
  Commit as `feat(torch): register relocation custom op and fake kernel`.

## Task 3: Replace accepted FX regions and route eligible eager transfers

**Interfaces:** Implement callable `RelocBackend` and
`eager_transfers(*, backend)`. `RelocBackend(compiler=..., runtime=...)`
accepts the R1/R2 bridge objects; optional default construction resolves their
documented configuration lazily. Preserve the original GraphModule.

- [ ] Add an end-to-end test using a backend and original function:

  ```python
  def test_compiled_layout_transfer(backend):
      def fn(x):
          return x.to("cuda").transpose(0, 1).contiguous()
      compiled_fn = torch.compile(fn, backend=backend, dynamic=True)
      with torch.no_grad():
          x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
          actual = compiled_fn(x)
          expected = fn(x)
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)
      assert actual.stride() == expected.stride()
      assert actual.storage_offset() == expected.storage_offset()
      assert backend.stats()["runtime_executions"] >= 1
  ```

  Mark this test `gpu`. Add CPU graph-structure equivalents with the test
  adapter, and tests that inspect retained original nodes for rejected regions.
- [ ] For each T2 candidate, create symbolic output size/stride nodes and
  ordered symbol inputs, register its original fallback, insert the custom
  op, and redirect tail uses. Erase only verified member nodes in reverse
  topological order after use checks. Run `gm.graph.lint()` and `gm.recompile()`.
  Return a callable that owns the rewritten graph and its handles. If input
  grad state requires autograd, execute the original graph before any rewritten
  work; leave training graphs unmodified at capture as well.
- [ ] Add adversarial graphs: returned intermediate, branch/shared user,
  `copy_`, mutation through a view of the source, unknown side effect between
  layout nodes, two transfers, a noncontiguous transfer result, and a view-only
  function. Assert result values, aliases, destination identity and version
  changes against original PyTorch and zero replacement for excluded regions.
  Include an unrelated counter mutation before a pure candidate: a runtime
  guard miss must replay only that region, never the enclosing graph's effects.
- [ ] Implement the eager mode around T1's overload inventory. Only intercept
  supported real CPU/CUDA out-of-place transfers, unchanged dtype and blocking
  semantics. Compile an identity recipe from the **current** source descriptor;
  prior eager transpose/reshape/materialization work is already done. Reuse
  the common cache/adapter. Preserve the exact original operator/arguments as
  fallback. `copy_`, subclasses, FakeTensor/compiler tracing, same-device
  operations, unsupported memory formats and gradient execution redispatch.
- [ ] Implement scoped reentrancy suspension so allocation, producer ordering,
  runtime-internal Torch calls, custom-op execution and fallback are not
  re-intercepted. Test nested scopes, exceptions, concurrent threads and an
  eager mode surrounding a compiled function for no duplicate offloading.
  Commit as `feat(torch): replace guarded FX and eager transfer boundaries`.

## Task 4: Close actual transfer, stream and resource acceptance

**Interfaces:** Consume real R1/R2 implementations; produce execution evidence
and enabled rows in `docs/torch-support.md`. No runtime mocks in this gate.

- [ ] Add independent H2D and **forward** D2H tests for identity, transpose,
  reshape+transpose, and padding. D2H must compare to the requested CUDA-source
  PyTorch function, not to inverse H2D round-tripping. Test f32/f16/i8 only where
  R2 advertises actual execution support. Include pinned/pageable host input.
- [ ] Add a side-stream D2H producer and immediate CPU consumer without any
  test-side global synchronize:

  ```python
  stream = torch.cuda.Stream()
  with torch.cuda.stream(stream), torch.no_grad(), eager_transfers(backend=backend):
      src = torch.arange(4096, device="cuda", dtype=torch.float32)
      src.add_(7)
      out = src.cpu()
  torch.testing.assert_close(out, torch.arange(4096) + 7, rtol=0, atol=0,
                             check_dtype=False)
  ```

  Also produce H2D output on a nondefault caller stream and immediately consume
  it there and on the default stream after the blocking return. Tests expose
  missing producer/consumer ordering; R2 supplies the synchronization. Work
  produced on another stream still follows PyTorch's normal caller dependency
  contract; do not claim to discover unrecorded producer streams automatically.
- [ ] Repeat transfers while dropping inputs and allocating same-size tensors
  immediately. Verify no corruption, live operation references after completion,
  or unbounded cache/pool growth. Test exception cleanup and idempotent close.
  Use bounded stress loops and explicit pool counters where available rather
  than assuming PyTorch allocator reserved bytes must return to zero.
- [ ] Test `non_blocking=True` uses original PyTorch with
  `nonblocking_unavailable` until R2 async support has its own acceptance tests.
  Test missing CUDA runtime/compiler capability, invalid bindings, empty/rank-0
  input and unsupported dtype/layout with a reason and zero runtime launches.
- [ ] Run the CPU frontend/regression suite and the real CUDA suite; update
  support rows with exact commands, versions and counts. Commit as
  `test(torch): validate guarded transfers and stream ownership`.

## Acceptance and handoff

- [ ] Supported FX regions and eligible eager transfers invoke the same adapter.
- [ ] Real/fake metadata and functional alias promises agree; opcheck and
  independent numerical comparisons both pass.
- [ ] Failed guards use the original pure region before launch; mutation and
  escaping intermediate users are not erased or replayed incorrectly.
- [ ] Caller streams, completion, tensor/staging ownership and cleanup are
  validated for H2D and forward D2H on real CUDA.
- [ ] Compiler calls, binds, Dynamo callbacks, execution paths and fallback
  reasons are distinguishable; live graphs survive artifact-cache eviction.
- [ ] Core runtime remains MLIR/LLVM/torch-free and the layout-only path works
  without typed operations or calibration data.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m 'not gpu' -q
"$TORCH_PYTHON" -m pytest libreloc/python/tests -m 'not gpu' -q
ctest --test-dir "$TORCH_BUILD" -R 'reloc-runtime' --output-on-failure
# In an R2-enabled CUDA build:
export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend/test_custom_op.py libreloc/python/tests/torch_frontend/test_transfers_gpu.py libreloc/python/tests/torch_frontend/test_backend.py -m gpu -q
```
