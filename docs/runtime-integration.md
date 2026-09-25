# Compiler-to-runtime integration: entry point and handoff (R4, issue [#148](https://github.com/JueonPark/sym/issues/148))

This is the one place to start. It says how to build the stack from a fresh
checkout, how to drive it from C++ and Python, what is supported, how to
reproduce the evidence, and where each historical issue stands. The detailed
support rows stay with their owners and are linked, not copied:

| Topic | Authoritative document |
| --- | --- |
| Wire format v0/v1 | [reloc-plan-format.md](reloc-plan-format.md) |
| Exporter interface, manifests, compatibility matrix | [reloc-export.md](reloc-export.md) |
| Typed value semantics (C1) and folding (C2) | [reloc-typed-semantics.md](reloc-typed-semantics.md), [reloc-typed-folding.md](reloc-typed-folding.md) |
| Typed dispatch rows, policies, byte accounting (R3) | [runtime-dispatch.md](runtime-dispatch.md) |
| Typed conformance and frontend import matrix (C4) | [typed-relocation-support.md](typed-relocation-support.md) |
| Torch installation, activation, boundaries (T1–T4) | [torch-integration.md](torch-integration.md) |
| Torch support rows and evidence (T1–T4, R2) | [torch-support.md](torch-support.md) |
| Runtime library surface | [libreloc/README.md](../libreloc/README.md) |
| Research claims and their standing | [claim-ledger.md](claim-ledger.md) |

## 1. Build from a fresh checkout

Three separately built pieces, one configured tree each. The compiler links
LLVM/MLIR 21.1.8; the runtime `libreloc` and its C++ consumer link neither
(CTest checks both); the Python extension targets regular-GIL CPython 3.14.7.

```bash
# Toolchains (versions are the qualified baseline; see torch-finalization/README.md).
pip install uv==0.12.11 && uv python install 3.14.7
uv venv --python 3.14.7 /tmp/sym-torch-cpu        # CPU wheel
uv pip install --python /tmp/sym-torch-cpu/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/sym-torch-cpu/bin/python -r libreloc/python/requirements-torch-test.txt
# For CUDA rows, a second venv with torch==2.14.0 from .../whl/cu126 and the CUDA 12.6 toolkit.

# CPU tree (compiler + runtime + extension + examples), fresh directories.
cmake -G Ninja -S . -B build/torch-cpu -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$LLVM/lib/cmake/mlir -DLLVM_DIR=$LLVM/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$LLVM/bin/llvm-lit \
  -DPython_EXECUTABLE=/tmp/sym-torch-cpu/bin/python -DPYBIND11_FINDPYTHON=ON \
  -Dpybind11_DIR=$(/tmp/sym-torch-cpu/bin/python -m pybind11 --cmakedir)
cmake --build build/torch-cpu -j8

# CUDA tree: the same, with -DRELOC_ENABLE_CUDA=ON, the cu126 venv's python,
# and nvcc 12.6 on PATH. CUDA_ARCHITECTURES are 75;89 (Turing, Ada).
```

Select the matching tree for every command below (the built package must
precede the source tree on `PYTHONPATH`):

```bash
export BUILD="$PWD/build/torch-cpu"          # or build/torch-cuda
export PYTHONPATH="$BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$BUILD/sym/tools/sym-reloc-export"
```

A fresh copy of the committed tree (tracked files only, new build and
artifact directories) was configured, built and validated with the `cmake`
and `export` commands above against an already installed CPU venv (the
toolchain lines were not rerun); section 6 records the result. A tree
exported without `.git` should set `SYM_SOURCE_REVISION` so the runner can
record which revision it validated.

## 2. Public paths

**C++ (layout, wire v0).** `reloc-run-artifact` is the whole public path in
one program: decode, bind, validate, transfer through a `CopyBackend`.

```bash
sym-reloc-export libreloc/examples/recipes/split_transpose.mlir --output plan.bin --manifest m.json
python3 -c "import numpy as np; np.arange(128, dtype=np.float32).tofile('in.bin')"
$BUILD/libreloc/examples/reloc-run-artifact plan.bin --symbols s0=128 \
    --input in.bin --output out.bin --direction host        # or h2d / d2h on a CUDA tree
python3 -c "import numpy as np; x=np.fromfile('in.bin', np.float32); \
  assert np.fromfile('out.bin', np.float32).tobytes() == x.reshape(2, 64).T.copy().tobytes()"
```

It prints a one-line JSON report (`source_bytes`, `destination_bytes`,
`requested_transfer_bytes`, and for H2D the separate
`verification_readback_bytes`) and exits nonzero with a diagnostic for
unknown or missing symbols, violated constraints, short inputs, malformed
plans, and typed v1 artifacts (which go through the Python dispatch path).

**Python, Torch-free.** `pyreloc.load_plan` / `bind` / `validate_transfer_source`
/ `make_transfer` / `execute_transfer` for layout plans; `load_typed_plan` /
`bind_typed` / `query_capability` / `prepare_dispatch` / `execute_dispatch`
for typed plans ([libreloc/README.md](../libreloc/README.md)). Force the
original CPU pipeline and inspect the selected path without any benchmark
infrastructure:

```python
request = pyreloc.prepare_dispatch(bound, src_view, dst_view, "h2d", policy="original_cpu")
report = pyreloc.execute_dispatch(request)       # report["implementation"], ["wire_bytes"], ...
pyreloc.query_capability(bound, "h2d", "cuda")   # every eligible row and every exclusion reason
```

**Torch.** `torch.compile(fn, backend=RelocBackend())` replaces supported
regions (layout, and C4's proved casts/dequantize) with the guarded custom
ops; `eager_transfers(backend=...)` routes eligible eager transfers;
`prepare_weights` prepares stable inference weights;
`reloc_torch.dispatch.prepare_typed_transfer(..., policy="original_cpu")`
forces R3's CPU baseline for a typed artifact. `backend.stats()` reports
plan compilations, symbol binds, Dynamo callbacks, runtime executions,
typed dispatches by implementation and bytes, and reason-coded
fallbacks/exclusions ([torch-integration.md](torch-integration.md)).

## 3. Named examples

| Scenario | Command | Expected result |
| --- | --- | --- |
| C++ artifact consumer | `reloc-run-artifact PLAN --symbols ... --direction host\|h2d\|d2h` | bytes equal the numpy replay; one artifact rebound at 64/128/192 |
| Input transfer (H2D), symbolic reuse | `python libreloc/python/examples/torch_dynamic_transfers.py --direction h2d --sizes 128 192 256` | `[64, n/64]` float32 on `cuda:0`, strides `(n/64, 1)`, offset 0; exact vs PyTorch; `plan_compiles=1`, `symbol_binds=3`, `runtime_executions=3`, no fallback |
| Output transfer (forward D2H) | same with `--direction d2h` | the forward program on a CUDA source (not an inverse scatter); same counters |
| Weight loading | `python libreloc/python/examples/torch_weight_loading.py --device cuda:0` (or `cpu`) | observed transfers classified; `module.to` routed; prepared weights reused, invalidated by in-place update and `load_state_dict`, results fresh; quantized preparation reports `typed_artifacts_unavailable` |
| Typed relocation | `python libreloc/python/examples/torch_typed_relocation.py [--cuda]` | layout+cast, layout+quantize, dequantize+layout match the independent reference at two batch sizes; path and bytes printed |
| Everything above, as evidence | `python libreloc/python/examples/compiler_runtime_handoff.py --device cpu\|cuda --output PATH` | JSON with every scenario `passed`; exit 0 |

## 4. Combined behavior matrix

Each row names its evidence; "runner" rows are scenarios of
`compiler_runtime_handoff.py` recorded in `docs/runtime-evidence/{cpu,cuda}.json`.

| Behavior | Recipe / direction / dtypes | Required implementation or fallback | Evidence |
| --- | --- | --- | --- |
| Layout identity, transpose, merge/split, constant pad, exact f32/f16/i8 bytes | layout, H2D and D2H | R2 forward transfer | `test_compiler.py` (CPU), `test_transfers_gpu.py::test_h2d_layouts_match_pytorch_exactly`, `::test_forward_d2h_layouts_match_the_cuda_source_function` (CUDA) |
| Non-self-inverse permutation + non-zero pad through the C++ path | `transpose_pad.mlir`, host/H2D/D2H | `reloc-run-artifact` | runner `cpp_host_layout`, `cpp_cuda_layout` |
| Same witness through the T3 entry point | `torch.compile`, H2D and D2H | one runtime execution, exact metadata | `test_runtime_integration.py::test_permutation_and_pad_witness_through_torch_compile_both_directions` |
| One symbolic artifact rebound | split+transpose at 64/128/192 (C++, fresh process) and 128/192/256 (Torch) | one plan compile, one bind per size | runner `cpp_host_layout`, `portable_artifact_fresh_process`, `dynamic_transfers` |
| Extent-one boundary | first capture at `s0 // 64 == 1` | that capture keeps the original region (`conditional_materialization`), later sizes run on the runtime, all exact | runner `dynamic_extent_one_boundary` |
| Guard-invalid shapes, invalid bindings | divisibility, extents, repeated symbols | original region once, zero launches | `test_transfers_gpu.py::test_invalid_bindings_and_unsupported_sources_fall_back_with_zero_launches`, `test_backend.py::test_cpu_region_executes_through_the_adapter_and_replays_only_itself_on_guard_miss` |
| Default and non-default streams, immediate consumers, allocation reuse | layout H2D/D2H | R2 stream ordering | `test_transfers_gpu.py::test_side_stream_d2h_producer_and_immediate_cpu_consumer`, `::test_h2d_on_nondefault_caller_stream_is_consumed_there_and_on_the_default_stream`, `::test_repeated_transfers_with_dropped_inputs_and_reallocation`, `test_transport.py::test_d2h_orders_after_a_delayed_producer_on_a_nondefault_stream` |
| Execution failure propagation | injected copy failure | error, no second execution, resources released | `test_transfers_gpu.py::test_execution_errors_clean_up_and_do_not_retry`, `test_runtime.py`, `Dispatch.BackendFailuresPropagateWithoutASecondPath` |
| Invalid, malformed, stale artifacts | truncated plans, mutated manifests, v0/v1 swaps | rejection before execution | runner `cpp_rejections`, `wire_version_compatibility`; `test_compiler.py` (mutating exporter), `test_typed_artifact.py::test_stale_or_foreign_typed_artifacts_are_rejected` |
| Artifact compatibility | v0 layout, v1 typed; portable format 1/2 | each loader accepts only its version; the pre-C3 runtime rejects v1 at byte offset 4 (recorded baseline) | runner `wire_version_compatibility`, `portable_artifact_fresh_process`; [reloc-export.md](reloc-export.md) |
| Typed mixed programs | layout+cast, layout+quantize/dequantize; H2D and D2H | R3 rows; forced `original_cpu` and `auto` with recorded reasons; exact source/wire/destination/parameter bytes | runner `typed_example`, `typed_corpus_fresh`, `gpu_test_selection` (`test_typed_conformance_gpu.py`, `test_dispatch.py`); [typed-relocation-support.md](typed-relocation-support.md) |
| Typed D2H cast via T3 | `x.t().contiguous().to("cpu", f16)` | forward program, `cpu_reference` (no GPU narrowing kernel) | `test_runtime_integration.py::test_typed_d2h_cast_runs_the_forward_program_through_torch_compile` |
| Unsupported optimized kernel | nonzero-zero-point dequantize, f32→f16 on the GPU | the qualified CPU reference row, never a substitute | `Dispatch.CapabilityListsOnlyImplementedRows`, `test_typed_dispatch.py` |
| Missing calibration | `auto` without a model, several eligible rows | CPU reference, `no_calibration` | `test_dispatch.py`, runner `typed_example` (CUDA; the host-mode example has one row) |
| Non-blocking, empty, rank-0, offset, overlapping sources | any | original PyTorch with reason | runner `expected_exclusion_nonblocking`; `test_transfers_gpu.py::test_nonblocking_transfers_use_original_pytorch_with_a_reason`; `test_transport.py` |
| Quantize import | `quantized_decomposed.quantize_*` | original PyTorch, `quantize_semantics_unproved` | runner `expected_exclusion_quantize_import`; `test_typed_import.py` |
| Weight lifecycle | parameters/buffers, `load_state_dict`, in-place mutation, storage replacement, ties, close, collection | fresh results, reprepare or fallback, no stale data | runner `weight_loading`; `test_weights.py` (T4) |
| Typed parameter changes | runtime scales/zero points mutated between calls | stale request refused; re-preparation gives fresh results; invalid values fail preflight | `test_dispatch.py::test_parameter_values_are_snapshotted_and_rechecked`, `test_typed_import.py` |
| Prefold ownership | S8 prefolder | artifact keeps its backend alive until staging is freed; typed capability `typed_prefold_spec` | `PrefoldArtifactTest.SharedBackendOutlivesTheCallerReference`, `test_typed_dispatch.py` |
| Non-current CUDA target (multi-GPU host) | H2D to `cuda:1` while `cuda:0` is current, then D2H back | the transfer runs on its target device and leaves the current device unchanged | `test_transport.py::test_transfers_target_a_noncurrent_device_when_available`, executed on the four-device evidence host inside `gpu_test_selection`; a one-device host skips it with its reason |

**Semantics.** Transfers are blocking: every call returns after its own work
completed and orders after the caller's current CUDA stream; `non_blocking`
requests use PyTorch. Runtime parameters are validated owned snapshots bound
by declared name; device-resident parameters are excluded
(`device_parameters_unavailable`). A D2H program is the forward computation
from the device source; no inverse is inferred. Everything outside the
supported surface runs the original PyTorch region once with a recorded
reason.

## 5. Evidence

| File | Produced by | Device | Result |
| --- | --- | --- | --- |
| [runtime-evidence/cpu.json](runtime-evidence/cpu.json) | `compiler_runtime_handoff.py --device cpu` on a clean tree at `e337d53` (`environment.source_revision`) | CPU (torch 2.14.0+cpu, CPython 3.14.7) | 9/9 scenarios passed |
| [runtime-evidence/cuda.json](runtime-evidence/cuda.json) | `compiler_runtime_handoff.py --device cuda` on the same clean tree | RTX 2080 Ti, driver 595.71.05, torch 2.14.0+cu126, CUDA 12.6 | 15/15 passed; 140 of 142 gpu-marked pytest cases and 18 CUDA gtests (12 suites) executed, 0 failed; 2 by-design skips (float32-only witness parametrization) |
| [typed-evidence/c4-cuda-run.txt](typed-evidence/c4-cuda-run.txt) | C4 GPU conformance | same device | 17 passed |
| CI | `.github/workflows/build.yml` | CPU | `build`/`Run Tests` (lit, CTest, Torch-free pytest, typed corpus check, typed example), `Torch inventory CPU cp314` (CTest, full CPU pytest, examples, the runner) |

The earlier T1 inventory evidence ([torch-evidence](torch-evidence)) is
observation only and is not execution evidence. The CUDA evidence comes from
one Turing model (RTX 2080 Ti) on a four-device host, where the
non-current-device test above uses a second device; Ada (sm_89) is compiled
but not measured here, and nothing is claimed for other wheels,
architectures, or multi-GPU use beyond that test.

**Descriptive latency** (`cuda.json`, `descriptive_latency`; median of 15
samples after 3 warmup calls, each sample ending in
`torch.cuda.synchronize()`; compilation excluded):

| Scenario | Source bytes | Wire bytes | libreloc path | PyTorch | Path |
| --- | --- | --- | --- | --- | --- |
| layout H2D split+transpose, 2^16 elements | 262,144 | 262,144 | 4.27 ms | 0.158 ms | R2 forward H2D via the compiled region |
| same, 2^20 | 4,194,304 | 4,194,304 | 31.2 ms | 1.95 ms | same |
| same, 2^22 | 16,777,216 | 16,777,216 | 74.7 ms | 8.82 ms | same |
| typed H2D transpose+cast to f16, 2^20 (preparation included) | 4,194,304 | 2,097,152 | 39.5 ms | 3.99 ms | R3 `cpu_reference` (forced by `original_cpu`) |

The supported path is correct but slower than PyTorch's own copy at every
measured size in this configuration. Every blocking call binds the symbols
(`backend_counters`: one plan compile, 54 binds for 54 executions),
re-validates the request, constructs a `CudaBackend` and allocates its
pinned staging ring; none of this was profiled further, and no optimization
or threshold is part of this handoff. The research measurements of the
underlying transfer methods remain those of the claim ledger.

## 6. Reproduction

```bash
# CPU leg (also what CI runs)
cmake --build "$BUILD" --target check-sym
ctest --test-dir "$BUILD" --output-on-failure
/tmp/sym-torch-cpu/bin/python -m pytest libreloc/python/tests -m 'not gpu' -q
/tmp/sym-torch-cpu/bin/python libreloc/python/examples/compiler_runtime_handoff.py --device cpu --output runtime-cpu.json
# CUDA leg (qualified cu126 venv, CUDA-enabled tree, a real GPU)
/tmp/sym-torch-cuda/bin/python -m pytest libreloc/python/tests -m gpu -q
/tmp/sym-torch-cuda/bin/python libreloc/python/examples/compiler_runtime_handoff.py --device cuda --output runtime-cuda.json
```

Fresh-checkout run (tracked files of `e337d53` extracted with `git archive`
into a new directory, new build tree, the CPU commands above with
`SYM_SOURCE_REVISION` set): configure and build succeeded, check-sym 39/39,
CTest 11/11, pytest 599 passed / 1 skipped / 142 deselected (the gpu
selection), runner 9/9 passed with exit 0. Its JSON records the declared
revision and `source_dirty: null`, because an exported tree has no `.git`
to inspect. Later commits change only documentation and the evidence files.

## 7. Project disposition

**Parent issues.** [#131](https://github.com/JueonPark/sym/issues/131)
(Torch capture) is closed. [#132](https://github.com/JueonPark/sym/issues/132)
(typed transforms: C1–C4) and [#133](https://github.com/JueonPark/sym/issues/133)
(compiler-to-runtime execution: R1–R4) meet their functional acceptance with
this handoff: every child is merged or delivered here, the documented
examples run from a fresh checkout, CPU CI requires compilation, binding and
reference execution, and actual CUDA evidence covers transfers, stream
ordering, typed kernels and lifetimes. Remaining exclusions are the rows
marked excluded above and in the linked matrices.

**Historical issues.**

| Issue | Disposition | Evidence |
| --- | --- | --- |
| [#55](https://github.com/JueonPark/sym/issues/55) decoder rank checks | **Done here.** The layout decoder (v0 and v1) rejects an axis count of zero and a destination rank different from the axis count, at their byte offsets; bind() keeps its guards | `Decode.RejectsRankZeroPlanAtTheAxesSection`, `.RejectsDestinationRankDifferentFromTheAxisCount` |
| [#56](https://github.com/JueonPark/sym/issues/56) non-zero fill golden | **Done here.** `pad_nonzero` (fill bits 0x11223344) is pinned in `serialize.mlir` and driven decode → bind → execute | `Execute.NonZeroFillGoldenRoundTripsEndToEnd` |
| [#57](https://github.com/JueonPark/sym/issues/57) shared padded-extent helper | **Open, deferred.** A cosmetic refactor with no behavior change; not needed for any acceptance above | — |
| [#71](https://github.com/JueonPark/sym/issues/71) ASan/TSan CI | **Open, deferred.** Not implemented; the runtime's tests would be the primary target. No sanitizer result is claimed | — |
| [#88](https://github.com/JueonPark/sym/issues/88) R7 end-to-end overlap | **Closed earlier; unchanged.** Its research result stays `narrowed` in the claim ledger. The weight-loading scenario reused here is functional (correctness, lifecycle), not an overlap measurement | [claim-ledger.md](claim-ledger.md), [r7-e2e-overlap.md](r7-e2e-overlap.md) |
| [#63](https://github.com/JueonPark/sym/issues/63) optimized transfer experiments | **Research track concluded; its performance criterion was not met** (recorded in the issue). The engineering completion above does not change that | [issue comment](https://github.com/JueonPark/sym/issues/63#issuecomment-5267443274), [claim-ledger.md](claim-ledger.md) |
| [#73](https://github.com/JueonPark/sym/issues/73) regime-crossover plan | **Closed earlier; unchanged.** Its failed and withdrawn claims stay in the ledger as recorded | [claim-ledger.md](claim-ledger.md) |

The historical source records under `human_history/` stay local and
unpublished.
