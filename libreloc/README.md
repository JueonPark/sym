# libreloc

The standalone runtime for `#reloc.plan` execution (project phase P2).
Consumes wire-format-v0 plans (`docs/reloc-plan-format.md`) and executes
them; it is the runtime half of the compiler → runtime handoff.

## Linkage contract

- `reloc_runtime` (this library, `libreloc_runtime.so`) is **MLIR-free,
  LLVM-free, and torch-free**. Its only compiler-facing contract is the
  frozen wire format. This is asserted, not aspirational: the
  `reloc-runtime-mlir-free` CTest test (and a CI step) fails if any
  MLIR/LLVM symbol appears in the shared object's dynamic symbol table.
- The **test binary** (`libreloc-test`) links `llvm_gtest` as shared test
  infrastructure — the contract binds the library, not its tests.
- Include paths: the repository's root CMake globally injects MLIR include
  directories; this is include-path-only pollution, tolerated for v0
  (issue #41). Do not include MLIR headers from libreloc sources.
- `reloc_runtime` is a plain `add_library`, so it does not go through
  `llvm_update_compile_flags`: only the repository's global
  warning/codegen flags are inherited, and **exceptions and RTTI stay ON**
  for this target (LLVM's tree-wide `-fno-exceptions -fno-rtti` is not
  applied here). This is exactly what pybind11 (#C6) needs; #C6 should
  still re-verify this when the bindings land.

## Surface

- `reloc::decodePlan` (`reloc/Decode.h`) — wire-format v0 in
  (`docs/reloc-plan-format.md`), `RelocationPlan` out; strict validation
  with byte-offset diagnostics. The decoder is the trust boundary: every
  count is sanity-capped against the remaining byte budget before it gates
  an allocation, and hostile inputs are rejected by construction and
  fuzz-tested (`libreloc/test/DecodeTest.cpp`).
- `reloc::bind` (`reloc/Bind.h`) — `RelocationPlan` plus a caller-supplied
  `{symbol -> value}` map in, `BoundPlan` out. Requires an exact symbol-map
  match, enforces the two-class constraint contract (divisibility and
  runtime pad-range violations are hard bind errors; alignment is recorded
  on the bound plan for execute-time downgrade, never a bind failure),
  coalesces adjacent contiguous axes to a fixpoint, and picks an execution
  strategy unless the caller forces one (`libreloc/test/BindTest.cpp`).
- `reloc::executeView` / `executeH2D` / `executeH2DThreaded` / `gatherChunk` /
  `executeD2H` (`reloc/Execute.h`) — CPU relocation executors over a
  `BoundPlan`; `no_copy` view publish, single- and multi-thread strided copy
  with an AVX2 inner run, and the CPU D2H scatter; the gather primitive is
  #C5's per-chunk form (`libreloc/test/ExecuteTest.cpp`).
- `reloc::executeH2DPipelined` / `executeD2HPipelined` (`reloc/Pipeline.h`) —
  Strategy 4: chunked, pinned-staged, event-recycled H2D and its symmetric D2H
  inverse. Written once against the `reloc::CopyBackend` interface
  (`reloc/Backend.h`): `HostBackend` (`reloc/HostBackend.h`, worker threads +
  condition variables) makes the ring-buffer / chunk / synchronization logic
  byte-exact-testable in CPU-only CI; `CudaBackend` (`reloc/CudaBackend.h`,
  `cudaHostAlloc` / `cudaMemcpyAsync` / `cudaEvent`) runs it on the GPU under
  `RELOC_ENABLE_CUDA`. `PinnedBufferPool` (`reloc/PinnedBufferPool.h`) gates
  buffer reuse on event completion; `ChunkSchedule` (`reloc/ChunkSchedule.h`)
  cuts the outermost coalesced axis with a fixed byte heuristic plus an
  override (design decision 4). Output is bit-identical to `executeH2D` /
  `executeD2H` (`libreloc/test/PipelineTest.cpp`, `CudaPipelineTest.cpp`).
- `reloc::validateTransferSource` / `validateTransfer` / `executeTransfer`
  (`reloc/Transfer.h`) — R2's validated forward transfer requests (issue
  #146). A `BufferView` declares a framework buffer as its allocation (base,
  capacity) plus the logical view (byte offset, extents, element strides,
  element size, host/CUDA kind, device ordinal). Validation proves, with
  overflow-checked arithmetic, that the view is nonempty, non-negative-stride
  and injective (mixed-radix test), that the plan's maximal source read and
  destination write fit the declared capacities, that the destination is dense
  row-major of exactly `totalBytes`, and that kinds match the direction.
  `executeTransfer` runs the plan's *forward* relocation through any
  `CopyBackend` and blocks until that request's events complete: H2D reuses
  the pinned/stream pipeline; D2H copies the dense device source into owned
  pinned staging, waits for exactly that copy, then applies the forward host
  gather into the destination. Requests are single-use; failures are reported
  by value (`TransferError{code, message}`), never thrown
  (`libreloc/test/TransferTest.cpp`). The `CopyBackend` contract gained
  `waitStream(externalStream)` (order every private queue after a caller's
  producer stream), a sticky `failed()`/`error()` state, and `device()`;
  `CudaBackend` checks every CUDA status and works under its own device.
- `reloc::GatherPool` (`reloc/GatherPool.h`) — D1's persistent worker pool
  (issue #65): the pipeline partitions each chunk's valid outer rows across
  the pool's threads (`gatherThreads` argument or a caller-owned pool), with
  a per-worker byte floor (`kMinGatherBytesPerWorker`) so tiny chunks stay
  inline, and a counting barrier before `copyAsync` / staging reuse.
  Conservative safety guards fall back to inline gather/scatter — serialized
  (non-row-disjoint-dst) schedules for H2D, non-injective src layouts for
  D2H — so output stays bit-identical to `executeH2D`/`executeD2H`, and
  `gatherThreads == 1` never constructs a pool. Explicit `close()` lifecycle
  for pybind, dispatches and `close()` are serialized internally, so
  concurrent use from multiple threads is safe (`libreloc/test/GatherPoolTest.cpp`).
- `reloc::quant` (`reloc/Quant.h`) — R0.1's CPU transform kernels
  (issue #74): contiguous per-channel int8 quantize
  (`quantizePackF32S8`), the fused strided-gather + quantize Case-1a
  kernel over a `BoundPlan` (`gatherQuantizeF32S8`, chunk form mirroring
  `gatherChunk`), int4 nibble pack (`packS8S4`), and fp32→fp16 convert
  (`convertF32F16`). Every kernel has a scalar reference variant plus
  SIMD tiers (AVX2 and/or AVX-512, per issue #74's variant table) behind
  runtime dispatch (`Variant`, `cpuSupports`, `resolveFor`), bit-identical
  across variants by contract, and a
  `*Parallel` wrapper that partitions over a caller-owned `GatherPool`
  with the pipeline's per-worker byte floor
  (`libreloc/test/QuantTest.cpp`; bandwidth: `bench/quant_bw.cpp`,
  pinning via `taskset` documented in that driver's header).
- `reloc::cuda` (`reloc/CudaKernels.h`) — R0.2's GPU kernels (issue #75),
  compiled for sm_75 + sm_89 under `RELOC_ENABLE_CUDA`: the JustCopy
  ceiling (`copyF32`), plan-driven strided relocate in naive
  (`relocateNaiveF32`) and SMEM-tiled 32×32 forms (`relocateF32`, tiled
  when the coalesced plan is a 2-D transpose, naive fallback otherwise,
  bit-identical either way), GPU-side per-channel quantize
  (`quantizeF32S8`, bit-identical to the CPU scalar contract), the
  Method-A receive paths (`dequantS8F32`, `unpackS4S8`,
  `dequantRelocateS8F32`), and the EXP-4 pathological scatter
  (`scatterRandomF32`). Streams are type-erased to `void *`; launches are
  async, caller synchronizes (`libreloc/test/CudaKernelsTest.cpp`, local
  GPU only, never CI).

## Python bindings (pyreloc)

The optional Torch frontend (`reloc_torch`: `torch.compile` backend, eager
transfer scope, prepared inference weights) is documented in
[docs/torch-integration.md](../docs/torch-integration.md) with its evidence in
[docs/torch-support.md](../docs/torch-support.md).

`libreloc/python/` builds a pybind11 extension exposing the runtime to
Python (issue #46): `load_plan(bytes) -> PlanHandle`,
`bind(plan, {symbol: value}, strategy="auto") -> BoundPlan`,
`relocate` / `relocate_inverse` (host CPU strategies), and `h2d` / `d2h`
(the C5 pinned/stream pipeline, `RELOC_ENABLE_CUDA` builds only;
`pyreloc.cuda_enabled` reports which you have).
`relocate`/`h2d`/`d2h` accept `gather_threads=` (0 = all cores) or a
reusable `gather_pool=pyreloc.GatherPool(threads)` — a context manager
whose `close()` joins its workers deterministically, so no pool threads
outlive the interpreter (issue #65).
Buffers are passed as
`(pointer, nbytes)` integer pairs — design decision 2;
`pyreloc.torch_interop.as_ptr` maps torch tensors / numpy arrays without
any C++ torch dependency. Decode/bind failures raise
`pyreloc.DecodeError` / `pyreloc.BindError` carrying the C++ diagnostic.
Note the CUDA stream-interop contract of the legacy `h2d`/`d2h` entry
points: they write through libreloc's own non-blocking streams and host-block
on completion before returning, but they are NOT ordered against work the
caller has queued on other streams — synchronize first (e.g.
`torch.cuda.synchronize()`) when the device buffer was just produced by an
async fill or kernel. The validated transfer API below carries that ordering
itself.

### Validated forward transfers (R2, issue #146)

`pyreloc.BufferView(base, capacity_bytes, offset_bytes, extents, strides,
element_size, kind, device=-1)` describes a buffer by its allocation and
logical view (`kind` is `"host"` or `"cuda"`). Torch callers read `base`,
`capacity_bytes` and the offset from `tensor.untyped_storage()`, never from
`data_ptr()`/`numel()`, and prove a CUDA view's device with
`pyreloc.cuda_pointer_device(base)`.

- `validate_transfer_source(bound, view, direction) -> int` is the
  allocation-free preflight: it returns the source span in bytes or raises
  `pyreloc.TransferError("<code>: <detail>")`.
- `make_transfer(bound, src_view, dst_view, direction) -> TransferRequest`
  adds the dense destination and returns the single-use request (properties
  `direction`, `source_span_bytes`, `destination_bytes`, `consumed`).
- `execute_transfer(request, *, caller_stream=None, n_buffers=4, n_streams=2,
  gather_threads=1, gather_pool=None)` runs the forward relocation and
  returns only after this request's work completed (no device-wide
  synchronization). `caller_stream` is the raw `cudaStream_t` handle of the
  caller's current stream on the transfer device (`0` is the legacy default
  stream; `None` means nothing to order after): every private stream waits on
  an event recorded there before touching the source or destination, which
  also covers destinations the caller's allocator may still be recycling.
  Dependencies on *other* producer streams are the caller's obligation, as in
  PyTorch's own stream contract; unrecorded producers cannot be inferred from
  a tensor. A consumed request raises `already_executed`; a backend failure
  raises `backend_failure` and is never retried by the runtime.
- Direction `"d2h"` is the plan's forward relocation applied to a device
  source. The existing `d2h`/`relocate_inverse` functions remain the separate
  inverse-scatter contract.
- Admitted descriptors: rank >= 1, every extent >= 1, non-negative strides
  that address each element once (broadcast, negative and overlapping views
  are `unsupported_layout`), element size equal to the plan's, capacities
  covering the checked spans (`insufficient_capacity`), no arithmetic
  overflow (`integer_overflow`), plan/view agreement (`plan_mismatch`,
  `direction_mismatch`), and at execution a CUDA view on the backend's own
  device (`device_mismatch`). A host view is admitted on either end so the
  whole path runs under `HostBackend` in CI. The validator proves declared
  views against declared allocations; it cannot verify raw addresses, so the
  Torch adapter derives both from the tensor's storage and proves the CUDA
  ordinal with `cuda_pointer_device`.
- Blocking only: `non_blocking=True` is not offered by this API; the Torch
  adapter (`reloc_torch.transport`) reports it as `nonblocking_unavailable`.

The Python caller retains every owner (source, destination, request) for the
duration of the blocking call; only the GIL is released around native work.

Wheel-less install (packaging is out of scope for v0): build with
pybind11 discoverable, then point `PYTHONPATH` at the build tree —

    uv venv --python /usr/bin/python3.10 .venv
    uv pip install --python .venv/bin/python pybind11 pytest numpy
    cmake -B build/sym \
      -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) \
      -DPython_EXECUTABLE=$PWD/.venv/bin/python
    ninja -C build/sym pyreloc_ext
    export PYTHONPATH=$PWD/build/sym/python

Without pybind11 the target is skipped with a notice and everything else
still builds.

### pytest oracle harness

`libreloc/python/tests/` byte-compares `relocate` against a numpy
(`transpose`/`reshape`/`pad`) replay of each plan's generating op chain:
≥100 randomized, seeded cases per run (`RELOC_SEED` env, default
20260710); failures print the recipe MLIR, the binding, the strategy, and
the first differing byte offset. Plans come from the compiler side: the
committed corpus under `libreloc/test/corpus/` (`.bin` wire blobs +
`.json` recipes) is regenerated by `generate_corpus.py`, which folds
generated `reloc.*` chains through `sym-opt --reloc-fold` and serializes
via `--test-reloc-utils`. GPU tests are marked `gpu` and auto-skip
without a CUDA build + GPU + torch (run locally, never in CI).

## Building

Built as part of the normal repository build; no extra steps:

    cmake -S . -B build/sym -DMLIR_DIR=... && cmake --build build/sym

### CUDA (optional, default OFF)

    cmake ... -DRELOC_ENABLE_CUDA=ON

Gates the CUDA toolkit dependency and the `cuda/CudaBackend.cu` translation
unit (the `CudaBackend` CopyBackend). With the option OFF the library builds
and every non-GPU test passes on a CUDA-less machine — CPU-only CI builds this
configuration.

## Tests

- `libreloc-test` (gtest): runs under `ctest`, and under `check-sym` via
  the `test/unit/libreloc.test` lit wrapper.
- GPU tests (from #C5 onward) run locally on a CUDA machine, never in CI;
  anything algorithmic must also be exercisable through the CPU
  `HostBackend` (P2 tracking issue, test conventions).
