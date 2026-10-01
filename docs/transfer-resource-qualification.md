# Transfer resource lifecycle qualification (#172)

The supported resource-lifecycle matrix passes on the reference host, including
real CUDA streams, two GPUs, changing shapes, concurrent callers, failure
ownership and close/clear races. Focused host ASan/UBSan checks pass with leak
detection enabled. **ThreadSanitizer could not start on this host**, so there is
no passing TSan result or claim that sanitizer coverage proves the absence of
data races. The [R8 enablement record](transfer-resource-enablement.md) carries
this limitation forward and records the default-policy decision.

This report accompanies the R7 tests at
`54ec674951c35d8b2b0722755c60fe289dd18476`, based on main after #186
(`64a8387`). The change adds tests and build-only fault modes; production runtime
and frontend behavior were unchanged in R7. Automatic reuse was opt-in at that
revision; R8 (#173) enables it after review of R6 and R7. Counts below describe
the R7 revision, not the later default-policy regression run.

The [R6 performance report](../bench/results/transfer-resource-reuse-171/README.md)
separately establishes completed-transfer speedups and CPU–PCIe overlap. These
stress tests do not measure performance or introduce a new D2H pipeline.

## Results and environment

AMD EPYC 7351, four RTX 2080 Ti GPUs; the device-sharing test uses GPUs 0 and 1.
CUDA 12.6, driver 595.71.05, Torch 2.14.0+cu126 and regular-GIL CPython 3.14.7.
Normal builds use GCC 11.4.0; sanitizer builds use Clang 18.1.8.
[Environment, test-source hashes and runtime hashes](qualification/transfer-resources-172/environment.json)
identify the tested configurations. The final extended GPU run uses a clean
worktree. Full suites were followed by that run with the final, stricter retained
byte and per-device count assertions.

| Validation | Result | Evidence |
| --- | --- | --- |
| Full CUDA frontend | 556 passed, 3 existing skips | [log](qualification/transfer-resources-172/frontend-cuda.log) |
| CPU-only frontend, Python reference-count GIL assertions enabled | 384 passed, 1 existing skip, 174 GPU cases deselected | [log](qualification/transfer-resources-172/frontend-cpu.log) |
| Native Python resource ownership/failure suite, CUDA build | 24 passed | [log](qualification/transfer-resources-172/native-python.log) |
| Same native Python suite, CPU-only build | 24 passed | [log](qualification/transfer-resources-172/native-python-cpu.log) |
| Extended CUDA qualification, 100 rounds | 6 passed, no hardware skips; 4,000 transfers plus 2 borrower-survival calls | [log](qualification/transfer-resources-172/gpu-stress-final.log) |
| Release CUDA native suite | 315 passed, 2 existing SIMD skips; both dependency guards passed | [CTest](qualification/transfer-resources-172/release-cuda.log), [full native log](qualification/transfer-resources-172/release-cuda-native-full.log.gz) |
| Debug host native suite | 293 passed, 2 existing SIMD skips; both dependency guards passed | [CTest](qualification/transfer-resources-172/debug.log), [full native log](qualification/transfer-resources-172/debug-native-full.log.gz) |
| Host ASan + UBSan, focused lifecycle/pipeline suites | 87 passed; leak detection enabled; no sanitizer diagnostic | [log](qualification/transfer-resources-172/asan-ubsan.log) |
| Host TSan | Unavailable: runtime exits 66 before tests with `unexpected memory mapping`; same result with per-process ASLR disabled | [normal](qualification/transfer-resources-172/tsan.log), [ASLR-disabled attempt](qualification/transfer-resources-172/tsan-noaslr.log) |

The frontend skips cover one test for an absent runtime and two non-FP32
permutation witnesses. Native skips require SIMD capabilities absent on this
CPU. None is counted as passing coverage. TSan's startup failure is also not a
test pass. The full frontend suites include the earlier ownership, serialization,
PID mismatch, typed/fake execution and backend-shutdown tests.

## What the integrated tests establish

The compiled stress case uses AUTO and borrowed owners in both H2D and forward
D2H. Each performs 100 rounds across 65×67, 1024×4096 and 1009×4093 FP32 inputs,
with transpose and padded transpose recipes. Three caller streams alternate,
including the default stream. A returned output is consumed immediately on a
different stream without adding a transfer-completion event; the consumer's own
copy back runs on that consumer stream. D2H updates its source on the current
caller stream before transfer. Native CUDA tests additionally hold producer
streams behind explicit host callback gates to test ordering deterministically.

Each call checks exact output bytes and frontend execution counts. Prior outputs
remain independently live while later calls allocate and fill other tensors.
The compiled loops perform 2,400 transfers without warmed Dynamo recompilation.
Artifact capacity is one while both compiled recipes keep using the shared
resource owner. Borrower close leaves the owner usable by a new adapter.

The concurrent cases add 1,600 transfers from four callers, first on one GPU and
then across two GPUs. They alternate small and growing shapes, use both transfer
directions and default/nondefault streams, retain first/last outputs, and clear
the shared cache while other callers may be active. Device and output placement
are checked on each call. Stats describe live cache entries; a device that has
finished can disappear from the snapshot after `clear()`.

Checks run throughout execution, rather than just after the workload:

- Compiled cases: at most two contexts, four owned background workers and 32 MiB
  allocated-plus-reserved staging in completed snapshots, with zero outstanding
  events. The explicit owner permits 32 MiB retained and 48 MiB live staging.
- Concurrent cases: at most four contexts total/two per GPU, eight background
  workers, 64 MiB retained and 96 MiB live staging including reservations, and
  eight outstanding event records. Peak live staging is checked as well.
- Successful close leaves no contexts, staging, reservations, streams, workers
  or event records. Allocation/free, stream creation/destruction, worker
  creation/join and event creation/retirement counters balance.

These are checked bounds for a finite workload, not a multi-hour soak or a
universal upper bound on Torch's allocator. Allocation pressure means repeated
independent tensor allocation/fill, not deliberately exhausting GPU memory.

## Failure, race and ownership coverage

The new native tests first warm a context, then inject a copy failure, an
exception after submission, event-record failure or wait failure. A real
HostBackend copy is held behind a gate: source/output owners remain alive and
staging cannot be freed until that copy drains. The failed call submits once,
the context retires, and a subsequent call can obtain a fresh healthy context.
Another test pauses staging growth while `clear()` or `close()` advances the
cache lifecycle, checking reservation charges and retirement/publication rules.

Build-only Python fault modes extend the same boundaries through the GIL-released
binding and frontend. Failures propagate without replaying the original Torch
region. Bounded subprocesses test normal owner-token destruction and discarded
exceptions; unknown completion keeps source/output owners and native resources
alive through close, cache destruction and interpreter shutdown. No fatal CUDA
device fault is injected into the shared hardware.

Earlier suites remain part of qualification:

| Requirement | Coverage |
| --- | --- |
| Exclusive leases, FIFO admission, timeouts and impossible budgets | `TransferResourceCache` native tests; Python lifecycle subprocesses |
| Construction/growth rollback, clear and close races, retirement charges | Native cache tests, including the two additions; backend lifecycle subprocess |
| Event-record failure after enqueue; no premature overwrite or free | `TransferExecution`, `TransferResources`, cached failure tests and gated Python faults |
| Cleanup cannot establish completion | Quarantine tests retain ownership, keep capacity charged, disable affected admission and make close report `completion_unknown` |
| Borrowed pools/owners and shutdown races | Native borrowed-pool close/recursive-call tests; Python shared-owner and concurrent backend-close tests |
| Fresh producer ordering and completed outputs | Gated `CudaTransferResources` tests plus compiled/default/nondefault-stream and multi-device stress |
| Checked capacity in Release, generic layouts and padding | `TransferExecution`, native transfer/pipeline matrices and full frontend suites |
| PID mismatch and serialization | Native fork tests and bounded Python fork/serialization checks; rejection precedes inherited locks |

Unknown completion deliberately has no automatic recovery: quarantined owners
remain charged for the process lifetime even if a test later lets its simulated
copy finish. A cleanup result of unknown is not treated as successful release.
Fault injection covers the backend's supported submission/event/quiescence
boundaries; hardware reset, fatal driver failure and arbitrary allocator teardown
failure are not exercised on real GPUs.

## Reproduction

Set `PYTHONPATH` to the desired build's `python/`, and `SYM_RELOC_EXPORT` /
`SYM_OPT` to built compiler tools. Run native-binding and frontend pytest roots
separately because their existing `conftest` module imports collide when combined.

```sh
python -m pytest -q -rs libreloc/python/tests/torch_frontend
python -m pytest -q libreloc/python/tests/test_transfer_resources.py
SYM_RESOURCE_STRESS_ROUNDS=100 python -m pytest -q -rs \
  libreloc/python/tests/torch_frontend/test_resource_qualification.py
ctest --test-dir build/torch-cuda --output-on-failure \
  -R '^libreloc-test$|^reloc-runtime-'
```

For the CPU-only extension, add `-m 'not gpu'` to the frontend command. Build it
with `RELOC_ENABLE_CUDA=OFF` and
`-DCMAKE_CXX_FLAGS=-DPYBIND11_ASSERT_GIL_HELD_INCREF_DECREF`. Use a separate Debug
host tree with CUDA/Python off for the native Debug run.

The focused sanitizer configuration uses the existing LLVM CMake support and
explicit compiler flags, without adding general sanitizer CI (#71):

```sh
cmake -G Ninja -S . -B build/resource-asan-clang \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=/usr/bin/clang-18 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DMLIR_DIR="$PWD/build/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/build/llvm-project/build/lib/cmake/llvm" \
  -DLLVM_EXTERNAL_LIT="$PWD/build/llvm-project/build/bin/llvm-lit" \
  -DSYM_BUILD_PYTHON=OFF -DSYM_BUILD_BENCHMARKS=OFF -DSYM_BUILD_EXAMPLES=OFF \
  -DRELOC_ENABLE_CUDA=OFF '-DLLVM_USE_SANITIZER=Address;Undefined' \
  -DCMAKE_CXX_FLAGS='-O1 -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/resource-asan-clang --target libreloc-test -j 6
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  build/resource-asan-clang/libreloc/test/libreloc-test \
  --gtest_filter='HostBackend.*:Backend.*:Pool.*:GatherPool.*:Pipeline.*:Transfer*'
```

Both sanitizer references were verified in the runtime binary. Explicit UBSan
flags matter: the imported LLVM module's `Address;Undefined` option alone did
not populate its UBSan flags in this project. An initial GCC attempt also linked
two ASan runtime versions through the imported test-library search paths; it
failed at startup and is excluded. The reported combined result uses Clang.
The runtime and native tests are instrumented; prebuilt LLVM test infrastructure,
Python and CUDA/driver code are not.

The TSan tree uses Clang 18, `LLVM_USE_SANITIZER=Thread`,
`-fsanitize=thread -fno-omit-frame-pointer -O1` and CUDA/Python off. Both ordinary
execution and `setarch x86_64 -R` fail before any test runs. #71 continues to own
portable sanitizer build/CI integration; this report does not close that issue.

Raw logs and compressed native logs are under
[`qualification/transfer-resources-172/`](qualification/transfer-resources-172/).
Run `sha256sum -c SHA256SUMS` in that directory to verify them. No production
correctness failure remains from the exercised matrix; TSan and untested
hardware-failure behavior remain explicit limits on this evidence.
