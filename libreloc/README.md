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
- Compile flags are inherited from the repository's LLVM configuration
  (HandleLLVMOptions). Revisit per-target (exceptions/RTTI) when the
  pybind11 bindings land (#C6).

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
