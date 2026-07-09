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

## Building

Built as part of the normal repository build; no extra steps:

    cmake -S . -B build/sym -DMLIR_DIR=... && cmake --build build/sym

### CUDA (optional, default OFF)

    cmake ... -DRELOC_ENABLE_CUDA=ON

Gates the CUDA toolkit dependency and (from #C5) the `cuda/*.cu`
translation units. With the option OFF the library builds and every
non-GPU test passes on a CUDA-less machine — CPU-only CI builds this
configuration.

## Tests

- `libreloc-test` (gtest): runs under `ctest`, and under `check-sym` via
  the `test/unit/libreloc.test` lit wrapper.
- GPU tests (from #C5 onward) run locally on a CUDA machine, never in CI;
  anything algorithmic must also be exercisable through the CPU
  `HostBackend` (P2 tracking issue, test conventions).
