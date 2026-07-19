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
  AVX2/AVX-512 tiers behind runtime dispatch (`Variant`, `cpuSupports`,
  `resolveFor`), bit-identical across variants by contract, and a
  `*Parallel` wrapper that partitions over a caller-owned `GatherPool`
  with the pipeline's per-worker byte floor
  (`libreloc/test/QuantTest.cpp`; bandwidth: `bench/quant_bw.cpp`,
  pinning via `taskset` documented in that driver's header).

## Python bindings (pyreloc)

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
Note the CUDA stream-interop contract: `h2d`/`d2h` write through libreloc's
own non-blocking streams and host-block on completion before returning, but
they are NOT ordered against work the caller has queued on other streams —
synchronize first (e.g. `torch.cuda.synchronize()`) when the device buffer
was just produced by an async fill or kernel.

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
