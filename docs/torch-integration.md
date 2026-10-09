# Torch integration: install, activate, boundaries and reproduction

For a complete installed environment, see [Installation](installation.md).
The commands below describe developer source builds. A complete installed wheel
includes its exporter and discovers it automatically; use `SYM_RELOC_EXPORT` or
`SYM_OPT` only when intentionally selecting a different build. CPU/CUDA package
variants retain the exact qualification contract below.

Status: T1–T4 of [#131](https://github.com/JueonPark/sym/issues/131) and
R1/R2 of [#133](https://github.com/JueonPark/sym/issues/133) are implemented.
The supported surface is opt-in inference relocation of dense CPU↔CUDA
transfers and their adjacent layout operations (transpose/permute,
reshape/view, squeeze/unsqueeze, integer-dimension flatten, materialization,
constant pad), with dynamic shapes, guarded
fallback to PyTorch, and explicit inference weight preparation. Typed value
transforms execute through the typed compiler/runtime path; automatic capture
accepts the casts and dequantization forms listed in the
[typed support matrix](typed-relocation-support.md). Quantize import and
typed weight preparation remain gated. Evidence, counts and the support matrix
live in [Torch support](torch-support.md); this guide records how to reproduce them.
CPU dimension-0 `index_select` can also fuse with a blocking H2D transfer and
optional FP32/FP16 cast; see the row-selection scope and example in
[Torch support](torch-support.md).

The end-to-end handoff (build from a fresh checkout, every named example,
CPU and CUDA evidence) is [runtime-integration.md](runtime-integration.md).

## 1. Qualified environments

One baseline is qualified: CPython 3.14.7 (regular GIL build,
`cpython-314-x86_64-linux-gnu`), PyTorch 2.14.0 (`+cpu` and `+cu126`),
pybind11 3.0.4, NumPy 2.5.3, pytest 9.1.1, LLVM/MLIR 21.1.8 (repository pin),
CUDA toolkit 12.6.3 for the CUDA build. Install the interpreter with a `uv`
that carries 3.14.7 (0.12.11 qualified; older catalogs stop at 3.14.6):

```bash
uv python install 3.14.7
uv venv --python 3.14.7 /tmp/sym-torch-cpu
uv pip install --python /tmp/sym-torch-cpu/bin/python 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/sym-torch-cpu/bin/python -r libreloc/python/requirements-torch-test.txt
uv venv --python 3.14.7 /tmp/sym-torch-cuda
uv pip install --python /tmp/sym-torch-cuda/bin/python 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python /tmp/sym-torch-cuda/bin/python -r libreloc/python/requirements-torch-test.txt
```

Verify each environment before building (`reloc_torch.check_version` enforces
the same facts at activation):

```bash
/tmp/sym-torch-cpu/bin/python -c 'import sys, sysconfig, torch; print(sys.version.split()[0], sysconfig.get_config_var("SOABI"), sys._is_gil_enabled(), torch.__version__, torch.version.cuda)'
# 3.14.7 cpython-314-x86_64-linux-gnu True 2.14.0+cpu None
/tmp/sym-torch-cuda/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
# 2.14.0+cu126 12.6 True
```

The PyTorch wheel does not install NVCC. The CUDA build needs the CUDA 12.6.3
toolkit; the runfile installs without root:

```bash
sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit \
  --toolkitpath=/tmp/sym-cuda-toolkit-12.6.3 --defaultroot=/tmp/sym-cuda-toolkit-12.6.3 \
  --tmpdir=/tmp/cuda-tmp --no-man-page --no-drm --override
/tmp/sym-cuda-toolkit-12.6.3/bin/nvcc --version   # release 12.6, V12.6.85
```

## 2. Build the compiler tools and the cp314 extension

Configure one build directory per environment against the existing LLVM/MLIR
21.1.8 installation (`MLIR_DIR`/`LLVM_DIR`). Never mix the cp310 extension in
`build/sym`, cp314 headers, or CPU/CUDA artifacts.

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
cmake -G Ninja -S . -B "$TORCH_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" -DLLVM_EXTERNAL_LIT="$LLVM_LIT" \
  -DPython_EXECUTABLE="$TORCH_PYTHON" -DPYBIND11_FINDPYTHON=ON \
  -Dpybind11_DIR="$("$TORCH_PYTHON" -m pybind11 --cmakedir)" -DRELOC_ENABLE_CUDA=OFF
ninja -C "$TORCH_BUILD" sym-opt sym-reloc-export reloc_runtime libreloc-test pyreloc_ext

export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
cmake -G Ninja -S . -B "$TORCH_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$MLIR_DIR" -DLLVM_DIR="$LLVM_DIR" -DLLVM_EXTERNAL_LIT="$LLVM_LIT" \
  -DPython_EXECUTABLE="$TORCH_PYTHON" -DPYBIND11_FINDPYTHON=ON \
  -Dpybind11_DIR="$("$TORCH_PYTHON" -m pybind11 --cmakedir)" \
  -DRELOC_ENABLE_CUDA=ON -DCUDAToolkit_ROOT=/tmp/sym-cuda-toolkit-12.6.3 \
  -DCMAKE_CUDA_COMPILER=/tmp/sym-cuda-toolkit-12.6.3/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
ninja -C "$TORCH_BUILD" sym-opt sym-reloc-export reloc_runtime libreloc-test pyreloc_ext
```

libreloc keeps its `75;89` CUDA architectures. The build stages `pyreloc` and
`reloc_torch` into `$TORCH_BUILD/python`; that directory must precede the
source tree on `PYTHONPATH` (the source `pyreloc` has no extension), and the
staged copies are refreshed by the build step, so rebuild `pyreloc_ext` after
editing the package.

## 3. Run the validation

```bash
export TORCH_PYTHON=/tmp/sym-torch-cpu/bin/python
export TORCH_BUILD="$PWD/build/torch-cpu"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"$TORCH_PYTHON" -c "import torch, reloc_torch, pyreloc, pyreloc._pyreloc as ext; import reloc_torch.transport, reloc_torch.backend, reloc_torch.weights; reloc_torch.check_version(); assert 'cpython-314' in ext.__file__"
"$TORCH_PYTHON" -m pytest libreloc/python/tests -m 'not gpu' -q
ctest --test-dir "$TORCH_BUILD" -R 'libreloc-test|reloc-runtime' --output-on-failure
"$TORCH_PYTHON" libreloc/python/examples/torch_weight_loading.py --device cpu

export TORCH_PYTHON=/tmp/sym-torch-cuda/bin/python
export TORCH_BUILD="$PWD/build/torch-cuda"
export PYTHONPATH="$TORCH_BUILD/python:$PWD/libreloc/python"
export SYM_OPT="$TORCH_BUILD/sym/tools/sym-opt"
export SYM_RELOC_EXPORT="$TORCH_BUILD/sym/tools/sym-reloc-export"
"$TORCH_PYTHON" -m pytest libreloc/python/tests/torch_frontend -q          # CPU and GPU marks
ctest --test-dir "$TORCH_BUILD" -R 'libreloc-test|reloc-runtime' --output-on-failure
"$TORCH_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py --direction h2d --sizes 128 192 256 --device cuda:0
"$TORCH_PYTHON" libreloc/python/examples/torch_dynamic_transfers.py --direction d2h --sizes 128 192 256 --device cuda:0
"$TORCH_PYTHON" libreloc/python/examples/torch_weight_loading.py --device cuda:0
```

GPU tests carry the `gpu` marker and skip without CUDA; a CPU run therefore
never certifies a transfer row. The R1 exporter is `sym-reloc-export` from the
same build (`SYM_RELOC_EXPORT`, or the sibling of `SYM_OPT`); lit checks use
`LLVM_EXTERNAL_LIT` from the LLVM installation.

## 4. Activation

Everything is opt-in; importing `reloc_torch` or `pyreloc` never imports Torch
or changes PyTorch behavior.

```python
import torch
from reloc_torch import RelocBackend, eager_transfers, prepare_weights

backend = RelocBackend()                      # bundled exporter (or explicit override), R2 transport
compiled = torch.compile(fn, backend=backend, dynamic=True)
with torch.no_grad():
    y = compiled(x)                           # accepted regions run reloc_torch::transfer
    with eager_transfers(backend=backend):
        model.to("cuda")                      # eligible eager transfers use the same adapter
    with prepare_weights(model, {"layer.weight": recipe}, backend=backend, stable=True) as prepared:
        w = prepared.get("layer.weight", device="cuda")
print(backend.stats())
backend.close()
```

- **Callable backend.** `RelocBackend(compiler=..., runtime=..., cache_capacity=128)`
  is a `torch.compile` backend. It never modifies the captured graph: accepted
  regions are rewritten in a copy, rejected regions keep their nodes, and the
  returned callable owns process-local execution handles (not a portable
  model format; only compiled artifacts are portable). `stats()` snapshots
  `dynamo_compiles`, `plan_compiles`, `symbol_binds`, `cache_hits`,
  `runtime_executions`, `weight_preparations`, `weight_invalidations` and
  reason-coded `fallbacks`/`exclusions`/`redispatches`; `close()` invalidates
  every handle and later use.
- **Eager scope.** `eager_transfers(backend=...)` intercepts only real,
  blocking, dtype-preserving CPU↔CUDA `aten._to_copy` calls on plain dense
  tensors; everything else redispatches with a recorded reason. Observation
  cannot undo layout operations that already executed, so eager offload
  compiles an identity recipe from the current source. Dynamo does not trace
  a frame while the dispatch mode is active: compile outside the scope; a
  compiled function run inside it executes its custom op with interception
  suspended and never offloads twice.
- **Weights.** `prepare_weights(module, recipes, *, backend, stable=False)`
  resolves the named parameter/buffer on the live module at every `get`,
  returns a fresh tensor, and never rewrites slots, `Parameter` identity or
  `requires_grad`. `stable=True` retains the transformed host layout only
  after freshness checks (identity, descriptor, storage, mutation version and
  an owned byte snapshot compared on every reuse); a CUDA target moves the
  retained layout with an identity artifact, a CPU target copies it directly;
  `invalidate()` and `close()` release it. With grad mode on and a
  gradient-requiring source, the recipe is replayed through PyTorch and
  recorded.

### Compose transfers with Inductor (#225)

```python
backend = RelocBackend(compute_backend="inductor")
compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
with torch.no_grad():
    result = compiled(inputs)
print(backend.stats()["inductor_compiles"])
backend.close()  # invalidates compiled callables and their transfer handles
```

The default `compute_backend="eager"` executes the rewritten FX graph as before.
The opt-in `"inductor"` mode sends a copy of that graph through the qualified
Torch 2.14 Inductor/AOT pipeline, including graphs with no accepted transfer.
Sym's layout, typed and indexed custom ops remain opaque calls with their
existing fake metadata, runtime guards and fresh output contract. Inductor
compiles the surrounding tensor computation. No graph-level eager fallback is
installed: Inductor compilation errors propagate and release new handles.
An unsupported transfer region stays in the graph for Inductor; an expected
runtime guard miss replays only that region through its original Torch code,
with the existing reason counter. Execution errors after launch still surface.
Leave Dynamo's global `suppress_errors` disabled to retain this error contract.

Composition is **inference only**: capture and every invocation require
`torch.no_grad()` or `torch.inference_mode()`, including inputs without gradients.
Training should use Inductor alone; default RelocBackend retains its existing
autograd fallback. Transfer ops retain process-local handles and are not a
portable compiled model. CUDA graph capture is unsupported: this mode forces
`triton.cudagraphs=False` and rejects an explicit true value. Optional
`inductor_options={...}` passes other qualified compiler settings; use this
argument rather than `torch.compile(mode=..., options=...)` with this backend.

`inductor_compiles`, `inductor_compile_failures` and `inductor_executions` expose
the handoff separately from Dynamo callbacks and Sym artifact compilation.
The tests also verify actual generated kernels, changing symbolic sizes,
nondefault-stream ordering, exact output strides and live parameter updates.

An asynchronous `PreparedWireWeights.prefetch()` window is a second supported
boundary: keep queue ownership and iteration in Python and compile the tensor
compute function consuming each delivered group. This preserves bounded
lookahead and overlaps submitted work where hardware/scheduling allow it.
Queue methods are not Dynamo graph operations. The
[model benchmark](../bench/issue225/README.md) uses this boundary for GPT/MoE and
an integrated transfer-plus-compute graph for DLRM/GraphSAGE, with matched
Inductor controls and explicit capture exclusions. The implementation follows
PyTorch's [FX backend contract](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)
and [opaque custom-op/fake-kernel integration](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html).

### Explicit resources for direct transfers

Direct `prepare_transfer` / `execute_transfer` callers can share a bounded
resource owner across layout recipes and shapes:

```python
from reloc_torch import TransferResources
from reloc_torch.transport import prepare_transfer, execute_transfer

# compiled_recipe is the result of CompilerClient.compile(recipe).
with TransferResources(max_retained_bytes=256 << 20,
                       max_live_staging_bytes=512 << 20) as resources:
    for source in inputs:
        request = prepare_transfer(compiled_recipe, source, "cuda:0")
        output = execute_transfer(request, resources=resources, gather_threads=8)
        consume(output)
    print(resources.stats())
```

Construction and inspection do not initialize CUDA, and importing the facade
does not load Torch or the optional native runtime. Each execution rechecks the
source, creates an independently owned output, captures the caller's current
CUDA stream, and completes the existing chunk pipeline. Earlier outputs remain
valid when staging is reused. Forward D2H retains its download-then-gather
schedule. Passing `resources=None` or omitting it keeps per-call allocations.

The defaults retain at most 256 MiB of staging, four contexts, two contexts per
backend/device, and 64 owned background gather workers. Optional
`max_live_staging_bytes` and `acquire_timeout_ms` bound live capacity and admission
waiting. Limits include reservations and quarantined resources; the timeout
does not interrupt execution. `stats()` reports allocated and additionally
reserved capacity, hits/growths/misses, resource-creation counters and devices.
A supplied `gather_pool` wins over `gather_threads` and remains caller-owned.

Use distinct prepared requests for concurrent calls. Reusing the same request
while it is allocating output or executing is rejected. A failure before the
handoff to native execution leaves the prepared request reusable; admission and
execution errors after handoff consume it and never trigger Torch replay.
Callers must not mutate or resize participating storage concurrently.

`clear()` retires idle resources and makes older active leases retire on return.
`close()` stops admission and waits for normal active work while releasing the
GIL. Successful calls retain no tensor owners through the cache. Unknown
completion retains both tensors and native resources for the process lifetime;
execution and close report `completion_unknown` rather than freeing memory
that a copy might still access. Handles cannot be serialized or reused after
fork. Prefer context managers or explicit close before interpreter shutdown.

### Shared resources for compiled and eager calls

`RelocBackend()` now selects `AUTO` when it creates its own transport adapter.
The adapter creates one owned resource cache on its first layout execution and
shares it across compiled and eager layout calls. Explicit `AUTO` selects the
same behavior; `transfer_resources=None` opts out and allocates resources per
call. A standalone `TransportAdapter()` still defaults to per-call resources;
configure it explicitly when injecting it into a backend. Import,
backend construction, compilation, preflight, fake execution, and stats do not
materialize that cache. Typed transfers use their existing dispatch path and
do not populate these resource counters.

```python
import torch
from reloc_torch import RelocBackend, eager_transfers

backend = RelocBackend(transfer_options={"gather_threads": 8})
transpose_to_gpu = torch.compile(
    lambda x: x.t().contiguous().to("cuda"), backend=backend, dynamic=True,
)
try:
    for source in inputs:
        output = transpose_to_gpu(source)
        with eager_transfers(backend=backend):
            identity_output = source.to("cuda")
        consume(output, identity_output)
    print(backend.stats()["transfer_resources"])
finally:
    backend.close()
```

One adapter shares its cache across graph entries, layout recipes and shapes;
compatible requests reuse native contexts. Artifact eviction and individual
entry closure do not close the cache. Fresh outputs, current-stream ordering,
blocking completion and the CPU-gather/PCIe chunk pipeline remain unchanged.
Leaving `eager_transfers` stops interception but leaves the backend and its
resources alive. Keep the same backend for calls that should share resources,
and close it when those calls are finished.

`transfer_options` is a copied mapping accepting `n_buffers` (default 4),
`n_streams` (2), `gather_threads` (1; 0 means all cores), and `gather_pool`
(None). A supplied pool remains borrowed and overrides the gather count.
Invalid keys/types/ranges fail at configuration time. When injecting a custom
`runtime`, omit the backend's resource/options arguments (explicit None is also
accepted) and configure the injected runtime itself. Explicit AUTO or a resource
owner alongside a custom runtime is rejected as ambiguous; omission never
overrides the injected runtime's policy.

Passing an explicit `TransferResources` borrows it, allowing multiple backends
and direct transfers to share one budget. Backend close stops new entry
execution, then closes its owned adapter outside its Python lock. The adapter
drains only its own AUTO cache; it never closes a borrowed owner. An injected
adapter is also borrowed and remains usable by other backends. Close that
adapter or shared owner explicitly when all borrowers are finished. Already
leased requests finish; racing requests that lose admission fail without Torch
replay. Inherited backend/adapter use after fork is rejected before their locks;
construct new owners in the child process.

For shared budgets or `clear()` without closing the backend, supply an owner:

```python
from reloc_torch import RelocBackend, TransferResources

with TransferResources(max_retained_bytes=64 << 20,
                       max_live_staging_bytes=128 << 20,
                       acquire_timeout_ms=1000) as resources:
    backend = RelocBackend(transfer_resources=resources)
    try:
        compiled = torch.compile(fn, backend=backend)
        output = compiled(source)
        resources.clear()  # earlier outputs stay valid; active leases retire on return
        output = compiled(next_source)
    finally:
        backend.close()  # the enclosing owner remains usable until its own close
```

Limits are per owner, not process-wide. AUTO uses the same defaults as
`TransferResources`: 256 MiB retained staging, four contexts total/two per device,
and 64 owned background workers. These are admission limits, not allocations at
construction. The default one-participant gather creates no background workers.
The retained-byte limit includes cacheable construction and active leases as
well as idle capacity. A request larger than that budget can use an ephemeral
context; context/worker limits and any `max_live_staging_bytes` still apply.
There is **no hard live-byte cap by default**. Supply an explicit owner when an
aggregate live staging cap or admission timeout is needed. The live-byte limit
includes allocations, additional reservations and quarantine, but not Torch
input/output storage. Admission waits do not time out by default; a configured
timeout does not cancel a transfer that has already started.

`backend.stats()["transfer_resources"]` is a native cache snapshot when its
adapter has an owner, otherwise None (including other custom adapters).
Inspection never creates resources. Native counters remain inspectable after
closure. For successful repeated compatible calls, `staging_allocations`,
`stream_creations` and `worker_creations` stop increasing after warmup;
`outstanding_events` returns to zero at completion. Shape/configuration changes
can require growth or another context. Explicit `None` has no cache counters;
None is not a measurement of zero allocations.

The [enablement record](transfer-resource-enablement.md) maps the acceptance
checks to completed performance and lifecycle evidence, including TSan's
unavailable coverage. It also records why the existing default limits remain.
Native consumers and external `CopyBackend` implementations must be rebuilt
together for the added `quiesce()` interface; plan wire formats are unchanged.

## 5. Semantics and boundaries

- **Blocking.** Every transfer completes before returning; `non_blocking=True`
  falls back to PyTorch (`nonblocking_unavailable`).
- **Dynamic shapes.** Reuse is guaranteed within one recipe, rank/dtype/layout
  family and its validated constraints (divisibility, positive extents, and
  the extent family a `contiguous()` materialization was proven for). A guard
  miss runs the original region once; an invalid shape surfaces PyTorch's own
  error. This does not promise one Dynamo compile per enclosing model.
- **Gradients.** `requires_grad` excludes a source only while grad mode is
  enabled; under `torch.no_grad()` parameters and buffers are ordinary inputs.
- **Exclusions with reasons.** Mutation (`copy_`, `load_state_dict`),
  subclasses, nonzero offsets, non-dense sources, unsupported dtype changes,
  unsupported memory formats, empty or rank-0 tensors, other devices, and regions the importer
  does not accept (for example a `contiguous()` after a derived extent that no
  guard bounds) run on PyTorch and appear in the counters.
- **Quantization.** Only explicitly requested. The prefold bridge
  (`pyreloc.prefold_s8`) is validated and Torch-free. C3 (issue [#143](https://github.com/JueonPark/sym/issues/143)) defines
  the typed recipe and parameter contract: `reloc_torch.recipe.Cast`,
  `Quantize` and `Dequantize` with `InlineParam` / `BindingParam` parameters
  compile through `sym-reloc-export --typed` into a wire v1 typed plan with a
  schema-2 manifest ([reloc-export.md](reloc-export.md)), portable as
  `format_version` 2, loaded with `pyreloc.load_typed_plan` and bound with
  `pyreloc.bind_typed` ([libreloc/README.md](../libreloc/README.md#typed-plans-c3-issue-143)).
  R3 (issue [#147](https://github.com/JueonPark/sym/issues/147)) executes them: `reloc_torch.dispatch.prepare_typed_transfer`
  / `execute_typed_transfer` run a typed recipe through the qualified rows of
  [runtime-dispatch.md](runtime-dispatch.md) (`original_cpu` forces the CPU
  reference pipeline; `auto` consults an optional calibration) and return the
  tensor with a scalar report. C4 ([typed-relocation-support.md](typed-relocation-support.md))
  imports the proved Torch forms (f32 <-> f16 casts on `_to_copy`,
  `quantized_decomposed.dequantize_*`) into typed recipes that the guarded
  custom-op route executes through R3; `quantize_*` captures and every other
  dtype change keep the original region with a reason, and float weight
  loading always keeps dtype and values.
- **Core stays Torch-free.** `import pyreloc` and the transfer/prefold bindings
  load without Torch; the runtime library is MLIR/LLVM/Torch-free by CTest.

Outside the opt-in scopes plain PyTorch runs unchanged:

```python
y = x.to("cuda")          # no backend, no scope: ordinary PyTorch transfer
```

## 6. Deferred and handoff

- Chunked forward D2H (the first path stages the whole dense source).
- `s8_quant_pack` through compiled artifacts: the v0 binder coalesces an
  identity relocation to one axis and the prefolder needs a distinct channel
  axis; the fused `s8_gather_quant` path is reachable.
- Ada validation of the CUDA build (this host is Turing); multi-GPU tested on
  four devices here.
- R4 collects the combined evidence and the disposition of historical issues.
