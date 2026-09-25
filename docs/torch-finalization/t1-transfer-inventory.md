# T1: Transfer inventory and eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Account for the eager and FX CPU–GPU transfers relevant to [#131](https://github.com/JueonPark/sym/issues/131),
and publish a tested, reason-coded eligibility contract.

**Architecture:** Add an optional sibling `reloc_torch` package containing
version isolation, immutable metadata records and pure classification.
Observe eager dispatch without changing its results, and inspect FX graphs
without running them. Share classification between both paths.

**Tech Stack:** CPython 3.14.7 (GIL enabled), PyTorch 2.14.0, pytest, existing CMake staging.

**Spec:** [Project plan §1/T1](../project-finalization-plan.md),
[#131](https://github.com/JueonPark/sym/issues/131),
[shared contracts and dependency map](README.md).

## Global Constraints

- All [shared constraints](README.md#shared-constraints) apply.
- Qualify CPython `3.14.7` (GIL enabled) and PyTorch `2.14.0` CPU/cu126
  using [the selected baseline](README.md#selected-version-baseline).
- "Unsupported cases execute through the original PyTorch path with a recorded reason."
- "Keep torch dependencies outside the core runtime."
- Observation neither replaces operators nor executes them twice. An
  eligibility candidate is not yet an enabled execution path.

---

**Proposed child title:** `[Finalize][Torch][T1] Inventory transfers and define eligibility`

**Dependencies:** None. T2 can use the record contract while CUDA evidence is
being collected. T3 cannot enable a candidate without T2 and R2 capability checks.

## File map

| Action | Path | Responsibility |
| --- | --- | --- |
| Create | `libreloc/python/reloc_torch/__init__.py` | Explicit public activation; no global mode installation |
| Create | `libreloc/python/reloc_torch/compat.py` | Version check and pinned Torch internal access |
| Create | `libreloc/python/reloc_torch/records.py` | Immutable tensor, transfer and graph records |
| Create | `libreloc/python/reloc_torch/eligibility.py` | Pure categories and stable exclusion reasons |
| Create | `libreloc/python/reloc_torch/observer.py` | Scoped eager dispatch and phase labels |
| Create | `libreloc/python/reloc_torch/graph_inventory.py` | Raw FX operator and adjacency inventory |
| Create | `libreloc/python/requirements-torch-test.txt` | `torch==2.14.0`, `pybind11==3.0.4`, `pytest>=9,<10`, `numpy>=2.3.3,<3` |
| Create | `libreloc/python/tests/torch_frontend/conftest.py` | Pinned-version check and GPU fixtures |
| Create | `libreloc/python/tests/torch_frontend/test_inventory.py` | Observation, categories and FX inventory |
| Create | `libreloc/python/examples/torch_transfer_inventory.py` | Input/output/weight-loading inventory runner |
| Create | `docs/torch-support.md` | Version/direction/operation eligibility table and evidence |
| Modify | `libreloc/python/CMakeLists.txt` | Stage the sibling package and explicitly discover the selected interpreter/development module with FindPython |
| Modify | `.github/workflows/build.yml` | CPython 3.14.7 Torch CPU job, its own extension build, and ABI-specific artifacts |

## Task 1: Establish the version and record contract

**Interfaces:** Produce `check_version()`, `TensorMetadata`, `TransferRecord`,
`GraphRecord`, `Eligibility`, and `classify(record)` for T2/T3. Keep Torch
internal APIs behind `compat.py`; serialized records contain plain values.

- [ ] Create the CPU/cu126 environments from T4's installation commands;
  use them for all T1 work. Rebuild `_pyreloc` with CPython 3.14.7 and
  pybind11 3.0.4 in `build/torch-cpu` or `build/torch-cuda`; existing cp310
  artifacts are not reusable. Record Python implementation, full version,
  `sysconfig.get_config_var('SOABI')`, GIL status, `torch.__version__`,
  `torch.version.cuda`, pybind11/NumPy/pytest versions and compiler identity.
- [ ] Add the requirements file with exactly these entries:

  ```text
  torch==2.14.0
  pybind11==3.0.4
  pytest>=9,<10
  numpy>=2.3.3,<3
  ```

  Install Torch first from the explicit CPU/cu126 index, then this file from
  the normal package index. Test that `compat.check_version()` accepts the
  qualified CPython 3.14.7 regular GIL build and Torch 2.14.0 CPU/cu126 pair.
  Reject unknown Python builds with `unsupported_python_version` or
  `unsupported_python_build`, prerelease/untested Torch with
  `unsupported_torch_version`, and unqualified CUDA wheels with
  `unsupported_torch_build`. Check the parsed base version and actual CUDA
  runtime metadata as well as the suffix. Keep acceptance pins explicit;
  upgrading them is a conformance change, not just removing the guard.
  Core `import pyreloc` must not import Torch; assert this in a subprocess.
- [ ] Before setting operator allowlists, rerun the no-op/copy/cast and
  symbolic FX probes on 2.14.0. Treat snippets below as semantic test recipes
  whose overload names must be confirmed by this run. Do not use results
  collected under the old local environment as release qualification.
- [ ] Define these dataclasses in `records.py` (all frozen). Metadata snapshots
  store `shape`, `strides`, `storage_offset`, `dtype`, device type/index,
  `requires_grad`, layout, pinning, subclass status, and optional storage
  capacity. Graph dimensions use expression strings and are never forced to int.

  ```python
  @dataclass(frozen=True)
  class Eligibility:
      category: str  # transfer, layout, same_device_copy, cast, mutation, other
      candidate: bool
      reason: str

  @dataclass(frozen=True)
  class TransferRecord:
      operator: str
      phase: str
      source: TensorMetadata
      destination: TensorMetadata
      non_blocking: bool
      mutates: bool
      aliases_source: bool
      layout_history: tuple[str, ...]
  ```

  Import `dataclass` and define `TensorMetadata` first using the fields above.
  `GraphRecord` contains node name, node kind, target string, input-node names,
  user-node names and optional tensor metadata. No record owns a Tensor.
- [ ] Implement category ordering before enablement: mutation first;
  different CPU/CUDA devices distinguish H2D/D2H; same-device dtype change
  is `cast`; same-device `_to_copy` is a forced copy; view operators are
  `layout`. Candidate rejection reasons include `requires_grad`,
  `unsupported_device`, `unsupported_dtype`, `unsupported_layout`,
  `unsupported_rank`, `empty_tensor`, `storage_offset`, `tensor_subclass`,
  `mutation`, `nonblocking_unavailable`, and `typed_transform_unavailable`.
  Successful layout-transfer candidates use `needs_compile_and_runtime_check`.
- [ ] Test representative `dataclasses.replace` variants of one record:

  ```python
  def test_mutation_cannot_be_a_functional_candidate(h2d_record):
      changed = dataclasses.replace(h2d_record, mutates=True)
      decision = classify(changed)
      assert decision.category == "mutation"
      assert not decision.candidate
      assert decision.reason == "mutation"
  ```

  Define `h2d_record` in conftest from plain metadata: shape `(4, 6)`, strides
  `(6, 1)`, f32, source CPU, destination CUDA index 0, offset 0,
  `requires_grad=False`, capacity 96 bytes. It needs no GPU allocation.
- [ ] Run `"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend/test_inventory.py -k 'version or classify or mutation' -q`
  first with the new interfaces absent (failure), then with the implementation
  (pass). Stage the exact changed package/test/requirements files and commit
  as `feat(torch): define transfer eligibility records`.

## Task 2: Observe eager calls and weight-loading boundaries

**Interfaces:** Consume the records; produce
`observe_transfers() -> TransferObserver`, `.records`, and `.phase(name)`.
The returned object implements a context manager over a TorchDispatchMode.

- [ ] Write a CPU test for the no-op/forced-copy/cast distinction:

  ```python
  def test_observer_preserves_copy_and_noop():
      x = torch.arange(6, dtype=torch.float32)
      with observe_transfers() as inventory:
          same = x.to("cpu")
          copied = x.to("cpu", copy=True)
          casted = x.to(torch.float16)
      assert same is x
      assert copied.data_ptr() != x.data_ptr()
      assert torch.equal(copied, x)
      assert casted.dtype == torch.float16
      copies = [r for r in inventory.records
                if r.operator == "aten._to_copy.default"]
      assert len(copies) == 2
  ```

  Confirm whether the no-op has a dispatcher event on 2.14.0. The inventory
  runner records named API scenarios separately so a no-op with no event is
  accounted for, without claiming every Python method call is observable.
  Freeze the two `_to_copy` assertions only after the new inventory confirms
  their overload names and counts.
- [ ] Run the test and confirm failure before adding `observer.py`.
- [ ] Implement `__torch_dispatch__` with one redispatch and metadata capture
  before/after it. Match overload objects through `compat.py`, using the
  schema to locate mutable/out arguments. Record exceptions as failed
  scenarios while preserving their type/message. Never call `.cpu()`,
  `.item()`, or synchronize to inspect contents.

  ```python
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      kwargs = {} if kwargs is None else kwargs
      before = self.snapshot_inputs(func, args, kwargs)
      result = func(*args, **kwargs)
      self.record_result(func, before, result)
      return result
  ```

  Implement `snapshot_inputs` and `record_result` in this task. Use weak tensor
  references with identity checks for bounded layout provenance; never use a
  recycled pointer as identity. Snapshot metadata outside provenance ownership.
  Nested scopes and exceptions must restore the dispatch stack. Layout
  history describes already observed operations, not a promise to fuse them.
- [ ] Add GPU scenarios for `.to(device)`, `.cuda()`, `.cpu()`, forced-copy
  overloads, pinned/pageable sources, D2H, cross-device `copy_`, and
  `nn.Module.to` with both a parameter and a registered buffer. Include
  `load_state_dict` into a CUDA module. Record actual low-level targets;
  inspect source/destination independently for in-place `copy_`.
- [ ] Verify aliases, destination object identity for mutation, and unchanged
  exceptions against calls made outside the observer. Test nested modes,
  cleanup after an exception, and that dropping tensors permits collection.
- [ ] Run the CPU tests, then the marked GPU inventory tests on a CUDA host.
  CPU pass plus GPU skips is provisional T1 progress, not completed CUDA
  evidence. Commit as `feat(torch): observe eager relocation boundaries`.

## Task 3: Inventory FX graphs and publish the eligibility evidence

**Interfaces:** Produce `graph_inventory(gm)` and a documented table mapping
Python scenarios, raw Dynamo FX targets, ATen targets, direction, metadata
restrictions, candidate state, and reason. T2 uses this mapping to normalize.

- [ ] Add a CPU symbolic FX test with a concrete expected target set:

  ```python
  def test_symbolic_layout_inventory():
      from torch.fx.experimental.proxy_tensor import make_fx
      def recipe(x):
          return x.reshape(x.shape[0] // 4, 4).transpose(0, 1).contiguous()
      gm = make_fx(recipe, tracing_mode="symbolic")(torch.ones(16))
      records = graph_inventory(gm)
      targets = {r.target for r in records}
      assert "aten.view.default" in targets
      assert "aten.transpose.int" in targets
      assert "aten.clone.default" in targets
      assert "aten.sym_size.int" in targets
  ```

  Keep tests of raw `torch.compile` backend graphs as well: they can contain
  `call_method('to')` or Python functions instead of the ATen forms above.
  The capture callback saves the graph and returns `gm.forward`; it does not
  attempt graph optimization. Include constant pad, multiple users, a
  same-device cast, and an intervening mutation. Record the actual 2.14.0
  targets in these expectations before making them the compatibility contract.
- [ ] Implement graph traversal as immutable node records, reading
  `node.meta['val']` or `example_value` through `compat.py`. Preserve symbolic
  expressions; record `metadata_unavailable` for absent metadata. Save graph
  text before/after and assert inventory makes no mutation.
- [ ] Create the runnable inventory example with `--device cpu|cuda` and
  `--output PATH`. Use phase labels `input`, `output`, `module_to`, and
  `load_state_dict`; emit JSON containing versions, named API scenarios,
  operator records, classification totals, and exclusions. Every scenario
  must account for transfers or explicitly state that no transfer occurred.
- [ ] On each CUDA qualification machine, record GPU name/capability,
  `torch.cuda.get_arch_list()`, driver version and NVCC version, and execute a
  small allocation/kernel/copy smoke test before inventory. The cu126 wheel
  must run on the actual Turing/Ada target; do not require that its cubin list
  literally include `sm_89` when compatible kernels can execute on Ada.
- [ ] Add sibling-package CMake staging independently of the pybind11 early
  return, so observation works without the runtime extension. Keep
  `reloc_torch.__init__` runtime imports lazy. Use FindPython with the selected
  `Python_EXECUTABLE`, Interpreter and Development.Module components, and
  `PYBIND11_FINDPYTHON=ON`; verify the resulting module has the cp314 ABI.
  The optional frontend job installs CPython 3.14.7 inside its container and
  builds its own extension. It must not consume a cp310 `.so` from the existing
  Ubuntu 22.04 build job. Extend artifacts/cache keys with `cp314` and the
  CPU/cu126 variant, include the new package, and require explicit imports.
  Keep the existing runtime test environment free of a Torch requirement.
- [ ] Fill `docs/torch-support.md` from the runner's observed results. Include
  metadata-only views, same-device conversions and mutation rows; no row may
  be labeled enabled merely because the observer saw it. Link each supported
  row to its eventual T2/T3 test and keep unimplemented rows excluded.
  Give every observed transfer a current supported/excluded execution status:
  a candidate awaiting compiler/runtime evidence is excluded with
  `needs_compile_and_runtime_check`. Keep the separate candidate flag so T2
  can work on that row without implying execution support.
- [ ] Run the complete CPU inventory suite and the example, then capture
  CUDA evidence on the cu126 test environment. Commit as
  `docs(torch): publish pinned transfer inventory and eligibility`.

## Acceptance and handoff

- [ ] CPU and CUDA evidence names CPython 3.14.7, PyTorch 2.14.0, the exact
  wheel/ABI, and a freshly built extension. No old-version snapshot qualifies.
- [ ] `.to()`, `.cpu()`, `.cuda()`, cross-device `copy_`, parameter transfers,
  buffer transfers, and weight loading have named scenarios and observed
  outcomes; every CPU–GPU event has a classification/reason.
- [ ] Shapes, strides, offsets, dtype, device index, pinning, mutation,
  alias semantics, nearby layout operations, and the actual operator are recorded.
- [ ] Same-device no-op/copy/cast and metadata-only views are distinct.
- [ ] Observation preserves values, aliases, mutation results and exceptions.
- [ ] T2 receives the tested raw-FX-to-ATen mapping; T3 receives stable reasons.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m 'not gpu' -q
"$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cpu --output /tmp/torch-inventory-cpu.json
# On the pinned CUDA environment:
export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m gpu -q
"$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cuda --output /tmp/torch-inventory-cuda.json
```
