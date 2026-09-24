# Torch transfer inventory, compiler artifacts, guarded replacement, transport and weight lifecycle (T1–T4/R2, issues #134–#137/#146)

The combined entry point, examples and CPU/CUDA integration evidence are
[runtime-integration.md](runtime-integration.md) (R4, issue #148); typed rows
are [typed-relocation-support.md](typed-relocation-support.md).

T1 observes eager dispatch and inventories FX graphs. T2 now imports conservative
layout/transfer regions, preserves symbolic guards, emits reloc IR, and accepts
only artifacts verified through the public `sym-reloc-export` interface. The
accepted artifact is portable: it includes the canonical recipe, plan bytes,
compiler identity, ordered symbols and provenance, constraints, and logical
source/destination rank. It can be rebound from concrete source metadata without
retaining the FX capture or its original callable.

**Execution rows are enabled for blocking, dense, dtype-preserving CPU<->CUDA
transfers** (float32/float16/int8, rank >= 1, zero storage offset, plain
tensors, inference only) captured by `RelocBackend` or intercepted by
`eager_transfers`. R2's storage/stream adapter
([#146](https://github.com/JueonPark/sym/issues/146), `reloc_torch.transport`)
validates tensor storage, orders after the caller's CUDA stream, and executes
the forward relocation through libreloc; the T3 CUDA acceptance suite below
passes against it on real hardware. Everything outside that surface still runs
the original PyTorch operation with a recorded reason.

T2's real CPU tests cover identity, transpose, static merge/split, symbolic
split, constant pad, transpose+pad, float32/float16/int8 exact bytes, compiler
bail, invalid guards, stale/malformed manifests, and fresh-process artifact
reload. An actual captured symbolic FX candidate reuses one artifact at multiple
shapes. Unsupported nodes, noncontiguous roots, mutation, escaping users,
training inputs, and nonblocking transfers retain their reason-coded original
PyTorch path.

Compile and persist a verified artifact explicitly, then bind its symbols from
the concrete source tensor before calling the standalone runtime:

```python
import pyreloc
from reloc_torch import CompilerClient, CompiledRecipe

compiled = CompilerClient("/path/to/sym-reloc-export").compile(recipe)
compiled.save("recipe.reloc.json")
compiled = CompiledRecipe.load("recipe.reloc.json")
plan = pyreloc.load_plan(compiled.plan_bytes)
bound = pyreloc.bind(plan, compiled.bind_values(source_tensor))
```

This surface prepares and binds a host relocation plan only; the sections
below describe execution.

## R2: tensor, stream and lifetime adapter (#146)

`reloc_torch.transport.prepare_transfer(compiled, source, device, *,
non_blocking=False)` validates without allocating or launching: frontend
metadata guards, exact symbol binding, the standalone binder, the destination
descriptor, the direction implied by the recipe against the real source/target
devices, storage identity read from `tensor.untyped_storage()` (allocation
base, capacity, offset), CUDA ownership through `cudaPointerGetAttributes`
(`device_mismatch`), and the native span/plan-fit proof
(`pyreloc.validate_transfer_source`). Expected exclusions raise
`UnsupportedRecipe` with the reasons above plus `direction_mismatch`,
`cuda_unavailable`, `nonblocking_unavailable`, `insufficient_capacity`,
`unsupported_layout`, `integer_overflow` and `plan_mismatch`.
`execute_transfer(request)` rechecks the source (stale requests are errors,
not fallbacks), allocates the fresh dense destination on the target device,
orders libreloc's private streams after the caller's current CUDA stream on
the transfer device, runs the blocking forward transfer, and returns the
destination once that request completed. Requests are single-use.

Forward D2H applies the requested recipe to a CUDA source (staging copy plus
forward host gather); it never routes through the inverse-scatter `d2h`. The
`(2, 3, 4) -> permute(1, 2, 0)` witness and the transpose/pad recipes compare
against independently constructed CPU results, not round trips.

## T3: guarded custom op, graph replacement and eager routing (#136)

`reloc_torch.RelocBackend(compiler=..., runtime=...)` is a callable
`torch.compile` backend and `reloc_torch.eager_transfers(backend=...)` is a
scoped dispatch mode. Both route through one executor,
`reloc_torch.runtime.execute_or_fallback`, which runs every expected guard
(metadata, exact symbol binding, symbol reconciliation, extent family,
declared output metadata, adapter preflight) before a destination is allocated
or work is launched, and runs the saved original region exactly once on an
expected rejection. Errors after the launch decision surface as
`ExecutionError` with recipe/direction context and are never retried.

```python
import torch
from reloc_torch import RelocBackend, eager_transfers

backend = RelocBackend()  # SYM_RELOC_EXPORT / SYM_OPT select the R1 exporter
compiled = torch.compile(fn, backend=backend, dynamic=True)
with torch.no_grad():
    y = compiled(x)                 # accepted regions call reloc_torch::transfer
    with eager_transfers(backend=backend):
        w = weight.to("cuda")       # eligible eager transfers use the same adapter
print(backend.stats())              # dynamo_compiles, plan_compiles, symbol_binds,
                                    # cache_hits, runtime_executions, fallbacks, ...
backend.close()
```

Implemented and tested on the qualified baseline (observed 2026-09-22):

- `reloc_torch::transfer(Tensor src, str handle, SymInt[] symbols, SymInt[] out_shape,
  SymInt[] out_strides, Device device) -> Tensor`, registered once through
  `torch.library.custom_op` with `mutates_args=()`. The fake kernel is
  `torch.empty_strided(out_shape, out_strides, dtype=src.dtype, device=device)`
  and consults no registry, binder, compiler or runtime. Real results are
  verified to be fresh, zero-offset tensors with exactly the declared metadata.
  `torch.library.opcheck` with inference tensors ran the 2.14.0 default set
  `test_schema`, `test_autograd_registration`, `test_faketensor`,
  `test_aot_dispatch_dynamic`: all `SUCCESS` on a CPU source (CPU build) and on
  a CUDA source (cu126 build, result produced by the recorded fallback). Direct
  gradient-requiring calls fail at call time; entry points fall back before the op.
  A module reload or duplicate import reuses the live registration: re-registering
  would replace the dispatcher entry and invalidate `OpOverload` objects already
  captured as FX node targets.
- Every result handed back by either entry point, from the adapter or from the
  original region once the promised metadata is known, passes
  `reloc_torch.runtime.verify_result`: fresh storage, declared shape, dtype,
  zero offset and device, and declared strides on every extent larger than one.
- Graph replacement preserves the captured `GraphModule`, inserts `sym_size`
  symbols and symbolic output shape/stride nodes, erases only verified member
  nodes, and keeps every rejected region's original nodes. Live graphs own their
  handle registrations; artifact-cache eviction cannot invalidate them; a
  guard miss replays only the region, never the enclosing graph's effects.
- Eager interception admits only what T1's `classify` admits (real blocking
  CPU/CUDA `aten._to_copy` of plain dense tensors with unchanged dtype); it
  compiles one symbolic identity artifact per rank/dtype/direction and reuses
  the common cache. Nested scopes, exceptions, other threads and a compiled
  function inside a scope never offload twice.
- Artifacts are cached in a bounded LRU (default 128) keyed by canonical recipe,
  frontend/compiler identity, wire version, dtype/layout family, transfer
  semantics and runtime capability identity; rejections share that identity and
  bound; concurrent same-key requests compile once.
- `requires_grad` excludes a source only while grad mode is enabled, when
  execution would be gradient-requiring; under `torch.no_grad()` parameters and
  buffers are ordinary dense inputs for both the eager and the graph path
  (Dynamo guards the captured grad mode; the graph callable re-checks per call).
- Dynamo does not trace a frame while the eager dispatch mode is active: a
  `torch.compile` call first executed inside `eager_transfers` runs eagerly and
  its transfers are offloaded there, once each. Compile outside the scope for
  graph replacement; a compiled function run inside the scope executes its
  custom op with interception suspended and never offloads twice.
- Importer refinement: `Tensor.contiguous()` after a non-dense layout is accepted
  when the ShapeEnv proves every extent is at least two (Dynamo's 0/1
  specialization), which is exactly when the materialization is unconditional.
  Those extents are recorded as candidate guards and enforced at bind time
  (`singleton_extent` fallback), so the original region and the recipe agree on
  output metadata over the accepted family. Derived extents such as `s0 // 64`
  remain `conditional_materialization`.

CUDA evidence for T3 comes from the second environment below: an NVIDIA GeForce
RTX 2080 Ti (capability 7.5, four devices, driver 595.71.05), CUDA toolkit
12.6.3 (`nvcc` V12.6.85) at `/tmp/sym-cuda-toolkit-12.6.3`, PyTorch
2.14.0+cu126 listing sm_50–sm_90, and a cp314 extension built with
`RELOC_ENABLE_CUDA=ON`. Real Dynamo captures of `x.to("cuda").transpose(0, 1).contiguous()`
are rewritten (one artifact reused across shapes) and produce PyTorch-exact
values, strides, offsets and devices through the recorded fallback; excluded
regions (`copy_`, mutation through a view, returned/shared intermediates, two
transfers, noncontiguous results, view-only functions) show zero replacement and
identical values, aliases and version counters.

| Gate (T3 #136, R2 #146, T4 #137) | Status on 2026-09-23 |
| --- | --- |
| Registration, fake metadata, opcheck (CPU and CUDA sources) | Passed |
| Graph safety, fallback before launch, handle lifetime, close | Passed |
| Eager routing, reentrancy, threads, nonblocking exclusion | Passed |
| Missing runtime/compiler capability, invalid bindings, empty/rank-0, dtype/layout: reason and zero launches on CUDA tensors | Passed |
| Real H2D and forward D2H through compiled graphs (identity, transpose, reshape+transpose, pad; f32/f16/i8; pinned/pageable) | Passed with R2 (`runtime_executions >= 1`, exact values/strides/offsets) |
| Direct adapter H2D/D2H (identity, transpose, pad, permutation witness; f32/f16/i8; sizes 64/128/192; no_copy identity still copies) | Passed |
| Side-stream D2H after a delayed producer, nondefault-stream H2D consumed there and on the default stream | Passed (three repetitions, 40 dependent matmuls of delay) |
| Repeated allocate/transfer/free/reuse, exception cleanup, idempotent close, stale/consumed requests | Passed (64 iterations, prepared requests collected, < 1 MiB retained) |
| Transfer to a non-current CUDA device (multi-GPU host) and four concurrent threads | Passed on a four-device host |
| Native validation before any copy (one byte short, overflow, pad-only regions), host-to-host forward path, single use, backend failure | Passed (`libreloc-test` `Transfer.*`, both builds) |
| Symbolic reuse: one artifact for sizes 128/192/256 (1 plan compile, 3 binds, 3 executions), symbolic capture verified, divisibility miss surfaces PyTorch's error with zero launches, strided input falls back, float16 is a distinct family | Passed (CUDA) plus a CPU host-adapter variant |
| Prepared weights: live-slot resolution, in-place/`.data`/NumPy/`load_state_dict`/replacement/tie invalidation, fresh outputs, close/invalidate/GC cleanup, grad-mode replay | Passed (CPU host adapter; the issue's GPU live-slot test on CUDA) |
| Prefold bridge conformance (declared int8 semantics through `s8_gather_quant`), validation, lifecycle | Passed; quantized weight preparation itself stays gated on C4 (`typed_artifacts_unavailable`): C3's typed artifact and binding exist ([reloc-export.md](reloc-export.md), `pyreloc.bind_typed`) and R3 executes typed plans and publishes the prefold capability (`pyreloc.typed_prefold_spec`, [runtime-dispatch.md](runtime-dispatch.md)); T4 wires them behind C4's conformance gate |

Only the "R2 absent" regression test now skips, by design. Still excluded:
`non_blocking=True` (falls back to PyTorch with `nonblocking_unavailable`),
mutation (`copy_`), gradients, subclasses, nonzero offsets, non-dense sources,
casts, and regions T2 does not import (for example a Dynamo `contiguous()`
after a derived extent such as `s0 // 2`, which is `conditional_materialization`
under dynamic shapes and needs static shapes).

Exact counts for this revision. CPU environment: CPython 3.14.7
(`cpython-314-x86_64-linux-gnu`, GIL enabled), PyTorch 2.14.0+cpu, NumPy 2.5.3,
pytest 9.1.1, pybind11 3.0.4, extension
`build/torch-cpu/python/pyreloc/_pyreloc.cpython-314-x86_64-linux-gnu.so`.
CUDA environment: the same interpreter and dependency versions with PyTorch
2.14.0+cu126 (CUDA 12.6) and the `build/torch-cuda` extension.

| Command | Environment | Result |
| --- | --- | --- |
| `pytest libreloc/python/tests -m 'not gpu' -q` | CPU | 493 passed, 1 skipped (R2 present), 123 deselected |
| `ctest --test-dir build/torch-cpu -R 'libreloc-test\|reloc-runtime'` | CPU | 3 of 3 passed (`Transfer.*` included) |
| `pytest libreloc/python/tests/torch_frontend -q` (all marks) | CUDA | 401 passed, 3 skipped (1 R2-absent regression, 2 float32-only witness variants) |
| `pytest libreloc/python/tests/torch_frontend/test_transport.py -m gpu -q` | CUDA | R2 acceptance, all passed (part of the row above) |
| `ctest --test-dir build/torch-cuda -R 'libreloc-test\|reloc-runtime'` | CUDA | 3 of 3 passed (`CudaPipeline` and `Transfer.*` included) |

The pre-R2 counts (244/454 CPU, 272 passed + 44 R2 skips CUDA) and the pre-T4
counts (471 CPU, 373 CUDA) are recorded in the git history of this file.

## T4: dynamic inputs and weight lifecycle (#137)

The installation, activation and boundary guide is
[Torch integration](torch-integration.md). T4 adds:

- **Symbolic reuse evidence.** `torch_dynamic_transfers.py` compiles
  `x.reshape(x.shape[0] // 64, 64).t().contiguous()` followed by a transfer
  once and runs sizes 128/192/256: one plan compile, three symbol binds, three
  runtime executions, one Dynamo callback, exact results, in both directions
  (D2H compiles the forward computation from a CUDA root). Running the test
  first exposed two importer gaps, now fixed: `Tensor.t()` joins the
  normalization vocabulary, and a derived extent such as `s0 // 64` is proven
  at least two from Dynamo's own inequality guards (`Ne(s0 // 64, 1)`,
  `Ne(s0 // 64, 0)` plus non-negativity), so the `contiguous()` region is
  accepted with that extent as a bind-time guard. Declaring a minimum with
  `mark_dynamic` is not an alternative: Dynamo rejects the reshape guard.
- **Prepared inference weights.** `prepare_weights(module, recipes, *,
  backend, stable=False)` resolves fully qualified parameter/buffer slots on
  the live module at every `get`, returns fresh tensors, never rewrites slots
  or `requires_grad`, replays through PyTorch when autograd is live, and with
  `stable=True` keeps the transformed host layout only while identity,
  descriptor, storage, mutation version and a byte snapshot all match.
  Counters `weight_preparations` and `weight_invalidations` are separate from
  plan compilation. `torch_weight_loading.py` on CUDA: observation classifies
  9 events (3 `module_to` candidates, 6 `load_state_dict` mutations); eager
  `module.to("cuda")` under `no_grad` executes 3 relocations with no fallback;
  stable preparation reports 4 preparations and 2 invalidations across
  repeated gets, an in-place update and a `load_state_dict`.
- **Prefold bridge, gated.** `pyreloc.prefold_s8` (Torch-free, owned,
  validated) reproduces the declared int8 semantics byte for byte through the
  fused gather path; `reloc_torch.prefold` reports
  `typed_artifacts_unavailable` until C4 qualifies the typed recipes C3
  defines and R3 executes (`reloc_torch.recipe.Quantize` compiled with
  `--typed`, dispatched by `reloc_torch.dispatch`; the S8 prefold capability
  is `pyreloc.typed_prefold_spec`), so float weights are never quantized. Handoff fact: a compiled identity artifact
  coalesces to one axis, so `s8_quant_pack` needs a channel-preserving plan.

Observed on 2026-09-09: regular-GIL CPython 3.14.7,
`cpython-314-x86_64-linux-gnu`, PyTorch 2.14.0+cpu and 2.14.0+cu126
(CUDA 12.6), NumPy 2.5.3, pytest 9.1.1, pybind11 3.0.4. Each environment
uses its separately built cp314 extension. Exact resolved versions, extension
paths, records, exclusions and named scenario outcomes are in the compact
[CPU evidence](torch-evidence/cpu.json) and [CUDA evidence](torch-evidence/cuda.json).

The CUDA host is an NVIDIA GeForce RTX 4070 Ti SUPER, capability 8.9,
driver 595.79, NVCC 12.6.85. Its cu126 wheel lists sm_50, sm_60, sm_70,
sm_75, sm_80, sm_86, sm_90; allocation, kernel and CPU round-trip succeeded
on actual Ada hardware. A literal sm_89 cubin is not required for compatible
kernel execution. Turing qualification remains pending hardware.

| Python scenario | Raw Dynamo target | ATen observation | Direction / semantics | Candidate and current exclusion |
| --- | --- | --- | --- | --- |
| `.to("cuda")`, `.cuda()` | `to`, `cuda` | `aten._to_copy.default` | H2D | **Supported** (opt-in): blocking dense inference float32/float16/int8, zero offset, plain tensors, via `RelocBackend` regions or `eager_transfers`; validated on RTX 2080 Ti with R2 |
| `.cpu()` | `cpu` | `aten._to_copy.default` | D2H | **Supported** (opt-in) with the same restrictions; forward relocation of the CUDA source |
| `copy_` | `copy_` | `aten.copy_.default` | H2D/D2H mutation | No; `mutation` |
| Frozen module parameters and buffers `.to()` | Not captured by this raw recipe | `aten._to_copy.default` | H2D, phase `module_to` | **Supported** (opt-in) inside `eager_transfers` under the same restrictions; weight lifecycle management is T4 |
| `load_state_dict` | Not captured by this raw recipe | `aten.copy_.default` | H2D mutation into resident weights/buffers | No; `mutation` |
| Same-device `.to()` | `to` (when retained) | No eager dispatch for no-op | Identity/alias; no transfer | No; proven identity is `same_device_noop` |
| `.to(copy=True)` | Not captured by this raw recipe | `aten._to_copy.default` | Same-device allocation | No; `same_device_copy` |
| `.to(float16)` | `to` | `aten._to_copy.default` | Same-device cast | Same device alone: no, `same_device_transfer` (nothing to relocate; eager `.to(dtype)` stays `typed_transform_unavailable`). Riding a transfer (`.to("cuda", float16)`, or a cast next to the transferring region): **yes** since C4, as a typed recipe (`Cast ieee_rne` / `exact`) compiled with `--typed` and dispatched by R3 ([typed-relocation-support.md](typed-relocation-support.md)) |
| `reshape`, `transpose` | `reshape`, `transpose` | `aten.view.default`, `aten.transpose.int` | Metadata-only views in tested recipe | No; `layout_only` |
| `contiguous` | `contiguous` | `aten.clone.default` | Materializes tested transposed view | No as an eager transfer; `unsupported_operator`. Inside a captured region T3 accepts it when Dynamo proves every extent >= 2 (guarded `singleton_extent`), otherwise `conditional_materialization` |
| Constant pad | `torch._C._nn.pad` | `aten.constant_pad_nd.default` | Padding recipe | No; compiler/runtime evidence absent |
| Symbolic size and division | `operator.floordiv` with shape input | `aten.sym_size.int`, `operator.floordiv` | Shape expressions | No; not a transfer |
| Intervening `add_` | `add_` | `aten.add_.Tensor` | Mutation; provenance cleared | No; `mutation` |
| Nonblocking `.to()` | `to` | `aten._to_copy.default` | H2D/D2H | No; `nonblocking_unavailable` (T3 redispatches to PyTorch before any adapter call) |

Current tests: [raw Dynamo / symbolic ATen / real CUDA graph tests](../libreloc/python/tests/torch_frontend/test_graph_inventory.py),
[eager CPU/CUDA observations](../libreloc/python/tests/torch_frontend/test_inventory.py),
[pure eligibility contract](../libreloc/python/tests/torch_frontend/test_contract.py),
and [CLI accounting/failure tests](../libreloc/python/tests/torch_frontend/test_inventory_cli.py).
R2 tests: [transport adapter](../libreloc/python/tests/torch_frontend/test_transport.py)
and native [`TransferTest.cpp`](../libreloc/test/TransferTest.cpp).
T3 tests: [preflight and fallback](../libreloc/python/tests/torch_frontend/test_runtime.py),
[registration and fake metadata](../libreloc/python/tests/torch_frontend/test_custom_op.py),
[graph rewriting and eager safety](../libreloc/python/tests/torch_frontend/test_backend.py),
[keys, eviction, ownership and counters](../libreloc/python/tests/torch_frontend/test_cache.py),
and [actual transfers, streams and allocations](../libreloc/python/tests/torch_frontend/test_transfers_gpu.py).

Metadata snapshots contain shapes, strides, offset, dtype, device index, pinning,
layout, subclass status and capacity. Symbolic expressions are strings, never
forced to integers. Actual FX FakeTensor is accepted as metadata, without retaining
Torch objects in records. Absent tensor metadata is `metadata_unavailable`.
Graph inventory is read-only, never executes a graph, and uses the same pure
`classify` function as eager observations. Raw targets remain unchanged; a
boundary description is not authorization to rewrite them.

Eager `requires_grad` is captured at dispatch time below autograd, not from a
post-autograd output. A source requiring gradients excludes training. Nonzero
offsets, empty tensors, unsupported rank/dtype/device/layout, user subclasses,
nonblocking requests and mutation retain explicit exclusion reasons. Nearby layout
history is conservative and clears on mutation or unrelated arithmetic.

Reproduce from the repository root after installing
[requirements](../libreloc/python/requirements-torch-test.txt), selecting the CPU
index or cu126 index explicitly. Use current uv (qualified with 0.12.11) to install
Python 3.14.7; older catalogs may not contain this release. Configure each fresh
build with its exact `Python_EXECUTABLE`, `PYBIND11_FINDPYTHON=ON` and environment's
`pybind11_DIR`; CUDA additionally selects the CUDA 12.6 toolkit and
`RELOC_ENABLE_CUDA=ON`.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m 'not gpu' -q
"$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cpu --output /tmp/inventory-cpu.json
export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -m gpu -q
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend/test_custom_op.py libreloc/python/tests/torch_frontend/test_transfers_gpu.py libreloc/python/tests/torch_frontend/test_backend.py -m gpu -q
PATH=/tmp/sym-cuda-toolkit-12.6.3/bin:$PATH "$TORCH_PYTHON" libreloc/python/examples/torch_transfer_inventory.py --device cuda --output /tmp/inventory-cuda.json
```

The CUDA build used for T3 was configured with `-DRELOC_ENABLE_CUDA=ON
-DCUDAToolkit_ROOT=/tmp/sym-cuda-toolkit-12.6.3
-DCMAKE_CUDA_COMPILER=/tmp/sym-cuda-toolkit-12.6.3/bin/nvcc
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` (the toolkit runfile installs without
root through `--toolkitpath`/`--defaultroot`). The `configure_file` copies of
`reloc_torch` in `$TORCH_BUILD/python` precede the source tree on `PYTHONPATH`;
rebuild `pyreloc_ext` after editing the package so tests see current sources.

The CLI exits nonzero on scenario failure or unavailable requested CUDA. CPU
runs explicitly mark `.cuda()` as not requested. Every scenario reports event
and cross-device transfer counts, including no-ops. Optional observation also
works from a CMake build configured with pybind11 disabled; importing
`reloc_torch` remains Torch-free until an observation entry point is called.
The legacy runtime CI job keeps Torch absent, while the sibling CPU job installs
3.14.7 and builds its own ABI-specific artifacts.
