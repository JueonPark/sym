# D1: Persistent Gather Worker Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #65 — a persistent `GatherPool` of `T` worker threads that the Strategy-4 pipeline uses to partition each chunk's `gatherChunk`/`scatterChunk` across workers, exposed through `gather_threads` / a reusable pool object in C++ and pybind, plus a gather-bandwidth microbench with committed JSON.

**Architecture:** A new `reloc::GatherPool` (long-lived threads, per-dispatch counting barrier) is the only new concurrency primitive. `executeH2DPipelined` / `executeD2HPipelined` gain a `gatherThreads` count (transient pool per call) and caller-owned-`GatherPool` overloads; per chunk, the valid outer-row range is partitioned into ≤ T sub-ranges with a per-worker byte floor, and the barrier completes before `copyAsync` (H2D) / before staging-buffer reuse (D2H). Conservative safety guards fall back to inline (single-thread) gather/scatter whenever row-disjointness is not provable, so `gather_threads=1` and all fallback paths are bit-identical to the current pipeline.

**Tech Stack:** C++17 (`std::thread`/`std::mutex`/`std::condition_variable`), gtest (`llvm_gtest`), pybind11, pytest + numpy, `bench/protocol.h`, CMake/Ninja.

## Global Constraints

- **MLIR-free contract:** no `#include <mlir/...>` or `<llvm/...>` anywhere under `libreloc/src`, `libreloc/include`, `libreloc/cuda`, `libreloc/python` — the `reloc-runtime-no-mlir-includes` ctest fails the build otherwise.
- **C++17** (`target_compile_features(reloc_runtime PUBLIC cxx_std_17)`); exceptions/RTTI are ON for `reloc_runtime` (plain `add_library`, not under LLVM flags).
- **No exceptions cross the `CopyBackend` interface**; `gatherChunk`/`scatterChunk` do not throw.
- **`gather_threads == 1` must be bit-identical in behavior to the current pipeline** (no pool touched, no threads spawned on that path) — issue #65 regression guard.
- **`threads == 0` resolves to `std::thread::hardware_concurrency()`** (the codebase's existing convention from `executeH2DThreaded`; the issue's "physical cores" maps to this).
- **File banners:** every new C++ file starts with the repo's `//===- name - purpose ---*- C++ -*-===//` banner style; comments state constraints, not narration.
- **Commit style:** `update(libreloc): D1 <what> (#65)` / `update(bench): ...` (see `git log`), each ending with the Claude co-author trailer.
- **Anything algorithmic must be exercisable through `HostBackend` in CPU-only CI** (P2 test convention); GPU paths are local-only.
- All builds/tests run from repo root `/home/jueonpark/sym`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `libreloc/include/reloc/GatherPool.h` | Create | Pool class: lifecycle, `parallelFor` contract |
| `libreloc/src/GatherPool.cpp` | Create | Worker loop, partitioning, counting barrier |
| `libreloc/test/GatherPoolTest.cpp` | Create | Pool unit tests (coverage, floor, reuse, close) |
| `libreloc/include/reloc/Pipeline.h` | Modify | `gatherThreads`/`GatherPool` overloads, byte-floor constant |
| `libreloc/src/Pipeline.cpp` | Modify | Per-chunk dispatch, disjointness guards |
| `libreloc/test/PipelineTest.cpp` | Modify | Threads dimension in matrices, guard tests, pool reuse |
| `libreloc/CMakeLists.txt` | Modify | Add `src/GatherPool.cpp` |
| `libreloc/test/CMakeLists.txt` | Modify | Add `GatherPoolTest.cpp` |
| `libreloc/python/PyReloc.cpp` | Modify | `GatherPool` py class; `gather_threads`/`gather_pool` args |
| `libreloc/python/pyreloc/__init__.py` | Modify | Export `GatherPool` |
| `libreloc/python/tests/test_gather_pool.py` | Create | Teardown (thread-count-asserted), parity, closed-pool errors |
| `bench/gather_bw.cpp` | Create | Gather-BW microbench driver (single vs T threads) |
| `bench/CMakeLists.txt` | Modify | Build target + ctest smoke |
| `bench/results/gather_bw_n4096.json` | Create | Committed measurement (E2 seed data) |
| `libreloc/README.md` | Modify | Surface + pyreloc doc bullets |

**Interpretation note (from the issue's acceptance criteria):** "Byte-exact ... in CI" is delivered by extending `PipelineTest.cpp`'s matrices (they run under `ctest` in the existing CI job). "TSan clean" is verified locally with a TSan-instrumented build (Task 6) and recorded in the PR description — CI has no sanitizer job today and adding one is out of scope for D1.

---

### Task 0: One-time local toolchain + baseline build

This machine has no `cmake`, `pybind11`, `pytest`, or LLVM build. Everything below is one-time setup; nothing is committed (except nothing — verify `git status` stays clean).

**Files:** none (environment only).

- [ ] **Step 1: Create a Python venv with the build/test tools**

```bash
cd /home/jueonpark/sym
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python cmake pybind11 pytest numpy
export PATH="$PWD/.venv/bin:$PATH"
cmake --version
```

Expected: `cmake version 3.2x` (or newer; must be ≥ 3.20). `.venv/` is already untracked (verify with `git status --porcelain` → if `.venv` shows up, add it to `.gitignore` — check first, the repo may already ignore it).

**IMPORTANT:** `export PATH="$PWD/.venv/bin:$PATH"` (run from the repo root) must be re-run in **every new shell** before any `cmake` / `ctest` / `python3 -m pytest` command in this plan — the venv provides cmake, pybind11, pytest, and numpy.

- [ ] **Step 2: Build LLVM/MLIR at the pinned commit** (heavy: ~30–60 min on this 32-core box, ~40 GB disk; skip if `build/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake` already exists)

```bash
LLVM_COMMIT=$(cat build_tools/llvm_version.txt | tr -d '[:space:]')
git clone https://github.com/llvm/llvm-project.git build/llvm-project/src
git -C build/llvm-project/src checkout "$LLVM_COMMIT"
cmake -G Ninja -S build/llvm-project/src/llvm -B build/llvm-project/build \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON
ninja -C build/llvm-project/build
```

This mirrors `build_tools/build_mlir.sh` minus its `check-mlir` run (not needed to build sym).

- [ ] **Step 3: Configure and build sym; run the baseline test suite**

```bash
cmake -G Ninja -S . -B build/sym \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$PWD/build/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$PWD/build/llvm-project/build/bin/llvm-lit \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DPython_EXECUTABLE=$PWD/.venv/bin/python
ninja -C build/sym libreloc-test pyreloc_ext bench-bind-cost bench-protocol-test
ctest --test-dir build/sym -R 'libreloc|reloc-runtime|bench' --output-on-failure
PYTHONPATH=$PWD/build/sym/python python3 -m pytest libreloc/python/tests -q
```

Expected: all ctest tests PASS; pytest passes (GPU tests skip). If the baseline fails, STOP and report — do not build D1 on a broken baseline.

*(No commit — environment setup only.)*

---

### Task 1: `GatherPool` — persistent workers + counting barrier

**Files:**
- Create: `libreloc/include/reloc/GatherPool.h`
- Create: `libreloc/src/GatherPool.cpp`
- Create: `libreloc/test/GatherPoolTest.cpp`
- Modify: `libreloc/CMakeLists.txt` (add source)
- Modify: `libreloc/test/CMakeLists.txt` (add test)

**Interfaces:**
- Consumes: nothing from this plan (std-only).
- Produces (used by Tasks 2–5):
  - `reloc::GatherPool::GatherPool(unsigned threads = 0)` — `0` → `hardware_concurrency()`; spawns `threadCount() - 1` OS threads (the caller is the T-th worker of every dispatch).
  - `int GatherPool::threadCount() const` — resolved T, ≥ 1.
  - `bool GatherPool::closed() const`
  - `void GatherPool::close()` — joins all workers; idempotent; destructor calls it.
  - `void GatherPool::parallelFor(int64_t begin, int64_t end, int64_t minPerWorker, const std::function<void(int64_t, int64_t)> &fn)` — partitions `[begin,end)` into ≤ `threadCount()` contiguous sub-ranges of ≥ `minPerWorker` rows (`parts = min(threadCount(), max(1, n / minPerWorker))`), runs one sub-range inline on the caller, blocks until all complete. When the partition collapses to one sub-range, runs `fn(begin,end)` inline without touching workers. Not reentrant; `fn` must not throw; must not be called after `close()`.

- [ ] **Step 1: Write the failing tests**

Create `libreloc/test/GatherPoolTest.cpp`:

```cpp
//===- GatherPoolTest.cpp - D1 worker-pool unit tests ---------------------===//
//
// The pool contract (issue #65): parallelFor covers [begin, end) exactly
// once whatever the thread/floor combination, the min-rows floor stops tiny
// ranges from shredding into per-row tasks, one pool is reusable across many
// dispatches, threads == 1 never leaves the calling thread, and close() is
// an idempotent, observable teardown.
//
//===----------------------------------------------------------------------===//

#include "reloc/GatherPool.h"
#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {

using reloc::GatherPool;

TEST(GatherPool, ParallelForCoversRangeExactlyOnce) {
  for (unsigned threads : {1u, 2u, 4u, 16u})
    for (int64_t n : {int64_t(0), int64_t(1), int64_t(7), int64_t(64),
                      int64_t(1000)})
      for (int64_t minPer : {int64_t(1), int64_t(4), int64_t(100)}) {
        GatherPool pool(threads);
        std::vector<std::atomic<int>> hits(static_cast<size_t>(n) + 1);
        for (auto &h : hits)
          h.store(0);
        pool.parallelFor(0, n, minPer, [&](int64_t b, int64_t e) {
          for (int64_t i = b; i < e; ++i)
            hits[static_cast<size_t>(i)].fetch_add(1);
        });
        for (int64_t i = 0; i < n; ++i)
          EXPECT_EQ(hits[static_cast<size_t>(i)].load(), 1)
              << "i=" << i << " threads=" << threads << " n=" << n
              << " minPer=" << minPer;
      }
}

TEST(GatherPool, MinRowsFloorLimitsPartitionCount) {
  GatherPool pool(8);
  // 10 rows / 4-row floor: at most 2 sub-ranges despite 8 threads.
  std::atomic<int> calls{0};
  pool.parallelFor(0, 10, 4, [&](int64_t, int64_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 2);
  // Entirely below the floor: exactly one inline call.
  calls = 0;
  pool.parallelFor(0, 3, 4, [&](int64_t, int64_t) { calls.fetch_add(1); });
  EXPECT_EQ(calls.load(), 1);
}

TEST(GatherPool, SingleThreadRunsInlineOnCaller) {
  GatherPool pool(1);
  EXPECT_EQ(pool.threadCount(), 1);
  std::mutex mu;
  std::set<std::thread::id> ids;
  pool.parallelFor(0, 100, 1, [&](int64_t, int64_t) {
    std::lock_guard<std::mutex> lk(mu);
    ids.insert(std::this_thread::get_id());
  });
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(*ids.begin(), std::this_thread::get_id());
}

TEST(GatherPool, ReusableAcrossDispatches) {
  GatherPool pool(4);
  for (int round = 0; round < 50; ++round) {
    std::atomic<int64_t> sum{0};
    pool.parallelFor(0, 128, 1, [&](int64_t b, int64_t e) {
      int64_t s = 0;
      for (int64_t i = b; i < e; ++i)
        s += i;
      sum.fetch_add(s);
    });
    EXPECT_EQ(sum.load(), 128 * 127 / 2) << "round " << round;
  }
}

TEST(GatherPool, CloseIsIdempotentAndObservable) {
  GatherPool pool(4);
  EXPECT_FALSE(pool.closed());
  pool.close();
  EXPECT_TRUE(pool.closed());
  pool.close(); // second close must be a no-op, not a crash
  EXPECT_TRUE(pool.closed());
}

TEST(GatherPool, ZeroThreadsResolvesToHardwareConcurrency) {
  GatherPool pool(0);
  EXPECT_GE(pool.threadCount(), 1);
}

} // namespace
```

Add to `libreloc/test/CMakeLists.txt` — in the `add_executable(libreloc-test ...)` list, insert `GatherPoolTest.cpp` after `PoolTest.cpp`:

```cmake
  PoolTest.cpp
  GatherPoolTest.cpp
  ChunkScheduleTest.cpp
```

- [ ] **Step 2: Run to verify it fails**

Run: `ninja -C build/sym libreloc-test`
Expected: FAIL — `fatal error: 'reloc/GatherPool.h' file not found`.

- [ ] **Step 3: Write the implementation**

Create `libreloc/include/reloc/GatherPool.h`:

```cpp
//===- GatherPool.h - persistent gather/scatter worker pool -----*- C++ -*-===//
//
// D1 (issue #65): a long-lived pool of worker threads. parallelFor runs a
// row-range callable over [begin, end) partitioned into <= threadCount()
// contiguous sub-ranges (one runs inline on the calling thread) and returns
// only after every sub-range completed (counting barrier) -- the pipeline
// relies on that barrier before copyAsync reads the staging buffer. Explicit
// close() lifecycle so pybind callers can tear the pool down
// deterministically: no threads outlive the interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_GATHERPOOL_H
#define RELOC_GATHERPOOL_H

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace reloc {

class GatherPool {
public:
  /// threads == 0 resolves to std::thread::hardware_concurrency() (>= 1).
  /// Spawns threadCount() - 1 OS workers; the calling thread acts as the
  /// last worker of every dispatch, so no worker idles while the caller
  /// blocks.
  explicit GatherPool(unsigned threads = 0);
  ~GatherPool(); // close()s if still open

  GatherPool(const GatherPool &) = delete;
  GatherPool &operator=(const GatherPool &) = delete;

  int threadCount() const { return threads_; }
  bool closed() const { return closed_; }

  /// Join every worker. Idempotent. Must not be called while a parallelFor
  /// is in flight (single-driver contract, asserted).
  void close();

  /// Partition [begin, end) into <= threadCount() contiguous sub-ranges of
  /// at least minPerWorker rows each and run fn(subBegin, subEnd) across the
  /// workers; blocks until every sub-range completed. Collapses to a plain
  /// inline fn(begin, end) -- no locks, no worker wakeup -- when only one
  /// sub-range results, so the threads==1 path is bit-identical in behavior
  /// to calling fn directly. Not reentrant (one dispatch at a time); fn must
  /// not throw; must not be called after close().
  void parallelFor(int64_t begin, int64_t end, int64_t minPerWorker,
                   const std::function<void(int64_t, int64_t)> &fn);

private:
  struct Range {
    int64_t begin, end;
  };

  void workerLoop();

  int threads_ = 1;
  bool closed_ = false;
  std::vector<std::thread> workers_; // threads_ - 1 entries

  std::mutex mu_;                // guards everything below
  std::condition_variable cv_;   // wakes workers (new work or stop)
  std::condition_variable done_; // wakes the parallelFor barrier
  const std::function<void(int64_t, int64_t)> *fn_ = nullptr; // live dispatch
  std::vector<Range> pending_;   // sub-ranges not yet claimed
  int outstanding_ = 0;          // handed to workers, not yet finished
  bool stop_ = false;
};

} // namespace reloc

#endif // RELOC_GATHERPOOL_H
```

Create `libreloc/src/GatherPool.cpp`:

```cpp
//===- GatherPool.cpp - persistent gather/scatter worker pool -------------===//

#include "reloc/GatherPool.h"

#include <algorithm>
#include <cassert>

namespace reloc {

GatherPool::GatherPool(unsigned threads) {
  if (threads == 0)
    threads = std::thread::hardware_concurrency();
  if (threads == 0)
    threads = 1;
  threads_ = static_cast<int>(threads);
  workers_.reserve(static_cast<size_t>(threads_ - 1));
  for (int w = 0; w + 1 < threads_; ++w) {
    // Same rationale as executeH2DThreaded: if a std::thread constructor
    // throws partway through, join the already-spawned workers before
    // rethrowing so unwinding never destroys a joinable std::thread.
    try {
      workers_.emplace_back([this] { workerLoop(); });
    } catch (...) {
      close();
      throw;
    }
  }
}

GatherPool::~GatherPool() { close(); }

void GatherPool::close() {
  if (closed_)
    return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    assert(pending_.empty() && outstanding_ == 0 &&
           "close() during an in-flight parallelFor");
    stop_ = true;
  }
  cv_.notify_all();
  for (std::thread &t : workers_)
    if (t.joinable())
      t.join();
  workers_.clear();
  closed_ = true;
}

void GatherPool::workerLoop() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
    if (pending_.empty())
      return; // stop_ set and nothing left to claim
    Range r = pending_.back();
    pending_.pop_back();
    const auto *fn = fn_;
    lk.unlock();
    (*fn)(r.begin, r.end);
    lk.lock();
    if (--outstanding_ == 0)
      done_.notify_one();
  }
}

void GatherPool::parallelFor(int64_t begin, int64_t end, int64_t minPerWorker,
                             const std::function<void(int64_t, int64_t)> &fn) {
  assert(!closed_ && "parallelFor on a closed GatherPool");
  const int64_t n = end - begin;
  if (n <= 0)
    return;
  if (minPerWorker < 1)
    minPerWorker = 1;
  const int64_t parts =
      std::min<int64_t>(threads_, std::max<int64_t>(1, n / minPerWorker));
  const int64_t per = (n + parts - 1) / parts; // ceil
  // Sub-ranges 1..parts-1 go to the workers; range 0 runs inline below.
  // (ceil rounding can make trailing ranges empty; skip them.)
  std::vector<Range> rest;
  for (int64_t p = 1; p < parts; ++p) {
    int64_t b = begin + p * per;
    int64_t e = std::min(begin + (p + 1) * per, end);
    if (b < e)
      rest.push_back({b, e});
  }
  if (rest.empty()) {
    fn(begin, end); // single sub-range: bit-identical to a direct call
    return;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    pending_ = rest;
    outstanding_ = static_cast<int>(rest.size());
  }
  cv_.notify_all();
  fn(begin, std::min(begin + per, end));
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return outstanding_ == 0; });
  fn_ = nullptr; // still under mu_: workers only read fn_ under the lock
}

} // namespace reloc
```

Add to `libreloc/CMakeLists.txt` — in the `add_library(reloc_runtime SHARED ...)` list, insert after `src/ChunkSchedule.cpp`:

```cmake
  src/ChunkSchedule.cpp
  src/GatherPool.cpp
  src/Pipeline.cpp
```

- [ ] **Step 4: Run to verify it passes**

Run: `ninja -C build/sym libreloc-test && ./build/sym/libreloc/test/libreloc-test --gtest_filter='GatherPool.*'`
Expected: PASS, 6 tests. Also run the full binary once (`./build/sym/libreloc/test/libreloc-test`) — all previously passing tests still pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/GatherPool.h libreloc/src/GatherPool.cpp \
        libreloc/test/GatherPoolTest.cpp libreloc/CMakeLists.txt \
        libreloc/test/CMakeLists.txt
git commit -m "update(libreloc): D1 GatherPool - persistent workers + counting barrier (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: H2D pipeline integration — per-chunk parallel gather

**Files:**
- Modify: `libreloc/include/reloc/Pipeline.h`
- Modify: `libreloc/src/Pipeline.cpp` (H2D half)
- Test: `libreloc/test/PipelineTest.cpp`

**Interfaces:**
- Consumes: `GatherPool` (Task 1), `ChunkSchedule::serialized` / `rowBytes` (existing), `gatherChunk` (existing).
- Produces (used by Tasks 3–5):
  - `constexpr size_t reloc::kMinGatherBytesPerWorker = 1ull << 20;` in `Pipeline.h` — per-worker byte floor; the pipeline converts it to rows via the chunk row size.
  - `void executeH2DPipelined(const BoundPlan&, const void *srcBase, void *deviceDst, CopyBackend&, int nBuffers, size_t chunkSizeOverride = 0, unsigned gatherThreads = 1)` — `gatherThreads == 1` takes the existing code path untouched (no pool constructed); `!= 1` builds a transient `GatherPool(gatherThreads)` for the call.
  - `void executeH2DPipelined(const BoundPlan&, const void *srcBase, void *deviceDst, CopyBackend&, int nBuffers, size_t chunkSizeOverride, GatherPool &gather)` — caller-owned pool, reused across calls.
  - `void executeH2DPipelined(const BoundPlan&, const void *srcBase, void *deviceDst, CopyBackend&, PinnedBufferPool &pool, size_t chunkSizeOverride = 0, GatherPool *gather = nullptr)` — the core; `gather == nullptr` means inline gather (exact current behavior).
- Safety rule implemented here: parallel gather only when `!sched.serialized` (a serialized schedule means outer rows are not provably disjoint in dst, so partitioning them would race — same fallback logic as `executeH2DThreaded`).

- [ ] **Step 1: Write the failing tests**

In `libreloc/test/PipelineTest.cpp`, replace the helper `expectPipelineExactH2D` (currently lines 69–83) with a threads-aware version:

```cpp
// Run executeH2DPipelined over HostBackend and assert byte-exact vs executeH2D.
// gatherThreads: 0 = hardware concurrency, 1 = the inline regression path.
void expectPipelineExactH2D(const BoundPlan &b, int nBuffers, int nStreams,
                            size_t chunkOverride, unsigned gatherThreads = 1) {
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(nStreams);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
  reloc::executeH2DPipelined(b, src.data(), device.data(), backend, nBuffers,
                             chunkOverride, gatherThreads);
  EXPECT_EQ(device, reference)
      << "nBuffers=" << nBuffers << " nStreams=" << nStreams
      << " override=" << chunkOverride << " gatherThreads=" << gatherThreads;
}
```

Replace `TEST(Pipeline, H2DByteExactMatrix)` with the issue's acceptance matrix (threads {1, 2, hw}):

```cpp
TEST(Pipeline, H2DByteExactMatrix) {
  for (const BoundPlan &b : plans())
    for (int nBuffers : {1, 2, 4})
      for (int nStreams : {1, 2})
        for (size_t override : {size_t(0), size_t(16), size_t(64)})
          for (unsigned threads : {1u, 2u, 0u}) // 0 == hardware concurrency
            expectPipelineExactH2D(b, nBuffers, nStreams, override, threads);
}
```

Append these new tests after `TEST(Pipeline, H2DSingleBufferSerializes)` (keep `#include "reloc/GatherPool.h"` with the other includes at the top of the file):

```cpp
TEST(Pipeline, H2DParallelGatherEngagesWorkers) {
  // 4096 rows x 4 KiB = 16 MiB. A 2 MiB chunk override gives 512-row chunks,
  // and the 1 MiB/worker byte floor yields 2 workers per chunk -- unlike the
  // tiny matrix plans (which collapse to the inline path via the floor), this
  // actually exercises concurrent gatherChunk sub-ranges. Meaningful under
  // TSan.
  BoundPlan b = makeBound({4096, 1024}, {1024, 1}, {1024, 1}, 4);
  for (unsigned threads : {2u, 0u})
    expectPipelineExactH2D(b, /*nBuffers=*/2, /*nStreams=*/2,
                           /*override=*/size_t(2) << 20, threads);
}

TEST(Pipeline, H2DNonDisjointDstStaysExactWithThreads) {
  // Column-major dst: outer rows interleave in dst byte space, planChunks
  // serializes to one whole-tensor chunk, and the gather guard must keep
  // that chunk single-threaded (partitioning it would race). Exactness with
  // threads requested is the assertion.
  BoundPlan b = makeBound({16, 4}, {4, 1}, {1, 16}, 4);
  expectPipelineExactH2D(b, /*nBuffers=*/2, /*nStreams=*/2, /*override=*/0,
                         /*gatherThreads=*/8);
}

TEST(Pipeline, CallerOwnedGatherPoolReusedAcrossCalls) {
  // One GatherPool across several pipeline calls: byte-exact every time,
  // close() observable afterwards (the pybind context object relies on
  // exactly this reuse pattern).
  BoundPlan b = makeBound({4096, 1024}, {1024, 1}, {1024, 1}, 4);
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> reference(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), reference.data());

  HostBackend backend(2);
  reloc::GatherPool gather(4);
  for (int call = 0; call < 3; ++call) {
    std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xCD);
    reloc::executeH2DPipelined(b, src.data(), device.data(), backend,
                               /*nBuffers=*/2, /*override=*/size_t(2) << 20,
                               gather);
    EXPECT_EQ(device, reference) << "call " << call;
  }
  gather.close();
  EXPECT_TRUE(gather.closed());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ninja -C build/sym libreloc-test`
Expected: FAIL — no overload of `executeH2DPipelined` takes 7 arguments / a `GatherPool`.

- [ ] **Step 3: Implement**

In `libreloc/include/reloc/Pipeline.h`: after `class PinnedBufferPool;` add `class GatherPool;`, add the floor constant, and replace the two H2D declarations:

```cpp
class PinnedBufferPool;
class GatherPool;

/// Per-worker byte floor for parallel per-chunk gather/scatter (issue #65):
/// a worker's sub-range must cover at least this many staging bytes
/// (converted to rows via the chunk row size), so tiny chunks run inline
/// instead of shredding into per-row tasks. E2 retunes via this constant.
constexpr size_t kMinGatherBytesPerWorker = 1ull << 20; // 1 MiB

/// Strategy 4 (H2D). `deviceDst` is sized bound.totalBytes (a device pointer
/// for CudaBackend, plain host for HostBackend). `nBuffers` in {1,2,4};
/// number of streams = backend.numQueues(). chunkSizeOverride == 0 uses the
/// size heuristic. gatherThreads partitions each chunk's gather across a
/// per-call GatherPool (0 = hardware concurrency); 1 is bit-identical to
/// the pre-D1 pipeline (no pool, no threads). Byte-identical to executeH2D.
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0,
                         unsigned gatherThreads = 1);

/// As above with a CALLER-OWNED GatherPool so long-lived callers (pybind)
/// amortize worker startup across calls.
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather);

/// Strategy 4 (H2D) with a CALLER-OWNED staging pool: buffers are reused
/// across calls instead of allocated per call, so steady-state latency is
/// measurable (issue #47's benchmark) and long-lived callers amortize
/// pinned allocation. Buffer count comes from pool.nBuffers().
/// Precondition (asserted): pool.bufferBytes() >=
/// planChunks(bound, pool.nBuffers(), chunkSizeOverride).maxChunkBytes,
/// and `pool` was created against this `backend`. gather == nullptr gathers
/// inline (single thread). Byte-identical to the pool-per-call overload.
/// (No D2H twin yet -- add when a caller needs it.)
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend,
                         PinnedBufferPool &pool, size_t chunkSizeOverride = 0,
                         GatherPool *gather = nullptr);
```

In `libreloc/src/Pipeline.cpp`: add `#include "reloc/GatherPool.h"`, `#include <algorithm>`, `#include <functional>`, and `#include <vector>` to the includes; add to the anonymous namespace (after `rebase`):

```cpp
// Run op over rows [begin, end) through `gather` when parallelism is safe
// and the pool is real; inline otherwise. The per-worker floor is
// kMinGatherBytesPerWorker expressed in rows of this chunk schedule's
// rowBytes, so a worker never receives less than ~1 MiB of gather work.
void dispatchRows(GatherPool *gather, bool parallelSafe, int64_t rowBytes,
                  int64_t begin, int64_t end,
                  const std::function<void(int64_t, int64_t)> &op) {
  if (!gather || !parallelSafe || gather->threadCount() <= 1) {
    op(begin, end);
    return;
  }
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowBytes));
  gather->parallelFor(begin, end, minRows, op);
}
```

Replace the pool-overload body of `executeH2DPipelined` (the one taking `PinnedBufferPool &`) with:

```cpp
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend,
                         PinnedBufferPool &pool, size_t chunkSizeOverride,
                         GatherPool *gather) {
  ChunkSchedule sched = planChunks(bound, pool.nBuffers(), chunkSizeOverride);
  assert(pool.bufferBytes() >= sched.maxChunkBytes &&
         "caller-owned staging pool too small for this plan's chunks");
  const int nStreams = backend.numQueues();
  // A serialized schedule means outer rows are NOT provably disjoint in dst
  // (see planChunks): partitioning the whole-tensor chunk across workers
  // would race, so gather stays inline -- same fallback executeH2DThreaded
  // makes.
  const bool parallelSafe = !sched.serialized;

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(); // blocks on this buffer's prior copy event
    void *staging = pool.buffer(i);

    fillStagingWindow(bound, staging, c.bytes);
    if (c.validEnd > c.validBegin) {
      uint8_t *dstBase = rebase(staging, bound, c.paddedBegin);
      // The counting barrier inside dispatchRows completes before copyAsync
      // may read the staging bytes.
      dispatchRows(gather, parallelSafe, sched.rowBytes, c.validBegin,
                   c.validEnd, [&](int64_t rb, int64_t re) {
                     gatherChunk(bound, srcBase, dstBase, rb, re);
                   });
    }

    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    backend.copyAsync(q, static_cast<uint8_t *>(deviceDst) + c.byteOffset,
                      staging, c.bytes, CopyDir::HostToDevice);
    pool.setEvent(i, backend.recordEvent(q));
  }
  pool.drain();
}
```

Replace the `int nBuffers` overload and add the `GatherPool &` overload:

```cpp
void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, unsigned gatherThreads) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  if (gatherThreads == 1) {
    // Regression guard (issue #65): threads == 1 must not even construct a
    // GatherPool -- bit-identical behavior to the pre-D1 pipeline.
    executeH2DPipelined(bound, srcBase, deviceDst, backend, pool,
                        chunkSizeOverride, /*gather=*/nullptr);
    return;
  }
  GatherPool gather(gatherThreads);
  executeH2DPipelined(bound, srcBase, deviceDst, backend, pool,
                      chunkSizeOverride, &gather);
}

void executeH2DPipelined(const BoundPlan &bound, const void *srcBase,
                         void *deviceDst, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  executeH2DPipelined(bound, srcBase, deviceDst, backend, pool,
                      chunkSizeOverride, &gather);
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `ninja -C build/sym libreloc-test && ./build/sym/libreloc/test/libreloc-test --gtest_filter='Pipeline.*'`
Expected: PASS (all Pipeline tests, including the new three). Then the full binary: PASS.

- [ ] **Step 5: Commit**

```bash
git add libreloc/include/reloc/Pipeline.h libreloc/src/Pipeline.cpp \
        libreloc/test/PipelineTest.cpp
git commit -m "update(libreloc): D1 parallel per-chunk gather in the H2D pipeline (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: D2H parity — parallel per-chunk scatter

**Files:**
- Modify: `libreloc/include/reloc/Pipeline.h` (D2H declarations)
- Modify: `libreloc/src/Pipeline.cpp` (D2H half)
- Test: `libreloc/test/PipelineTest.cpp`

**Interfaces:**
- Consumes: `GatherPool`, `dispatchRows` (Task 2), `scatterChunk` (existing).
- Produces (used by Tasks 4–5):
  - `void executeD2HPipelined(const BoundPlan&, const void *deviceSrc, void *srcBaseV, CopyBackend&, int nBuffers, size_t chunkSizeOverride = 0, unsigned gatherThreads = 1)`
  - `void executeD2HPipelined(const BoundPlan&, const void *deviceSrc, void *srcBaseV, CopyBackend&, int nBuffers, size_t chunkSizeOverride, GatherPool &gather)`
- Safety rule implemented here: parallel scatter only when the src layout is provably injective (`srcRowsWriteDisjoint`) — workers with disjoint outer ranges then write disjoint (possibly interleaved, e.g. transposed src) byte sets. Non-injective (aliasing/broadcast) layouts scatter inline.

- [ ] **Step 1: Write the failing tests**

In `libreloc/test/PipelineTest.cpp`, replace `expectPipelineExactD2H` with a threads-aware version:

```cpp
// Run executeD2HPipelined over HostBackend and assert it reconstructs src
// byte-exact (== executeD2H). `dst` is a known-good dst-layout buffer.
void expectPipelineExactD2H(const BoundPlan &b, int nBuffers, int nStreams,
                            size_t chunkOverride, unsigned gatherThreads = 1) {
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), device.data()); // build the dst layout

  HostBackend backend(nStreams);
  std::vector<uint8_t> back(static_cast<size_t>(srcElems) * b.elementSize,
                            0xCD);
  reloc::executeD2HPipelined(b, device.data(), back.data(), backend, nBuffers,
                             chunkOverride, gatherThreads);
  EXPECT_EQ(back, src) << "nBuffers=" << nBuffers << " nStreams=" << nStreams
                       << " override=" << chunkOverride
                       << " gatherThreads=" << gatherThreads;
}
```

Replace `TEST(Pipeline, D2HByteExactMatrix)` with:

```cpp
TEST(Pipeline, D2HByteExactMatrix) {
  for (const BoundPlan &b : plans())
    for (int nBuffers : {1, 2, 4})
      for (int nStreams : {1, 2})
        for (size_t override : {size_t(0), size_t(16), size_t(64)})
          for (unsigned threads : {1u, 2u, 0u}) // 0 == hardware concurrency
            expectPipelineExactD2H(b, nBuffers, nStreams, override, threads);
}
```

Append after `TEST(Pipeline, RoundTripH2DThenD2H)`:

```cpp
TEST(Pipeline, D2HParallelScatterEngagesWorkers) {
  // Transposed src (srcStrides {1, 4096}): outer rows INTERLEAVE in src byte
  // space but the layout is injective, so the reverse disjointness argument
  // holds and parallel scatter must engage AND stay exact. Same sizing logic
  // as H2DParallelGatherEngagesWorkers so the byte floor yields 2 workers
  // per chunk. Meaningful under TSan.
  BoundPlan b = makeBound({4096, 1024}, {1, 4096}, {1024, 1}, 4);
  for (unsigned threads : {2u, 0u})
    expectPipelineExactD2H(b, /*nBuffers=*/2, /*nStreams=*/2,
                           /*override=*/size_t(2) << 20, threads);
}

TEST(Pipeline, D2HAliasedSrcSerializesScatter) {
  // srcStrides {1, 1}: indices (i, j) and (i+1, j-1) hit the SAME src
  // element, so partitioned scatter would race. The injectivity guard must
  // serialize the scatter; the result must equal executeD2H exactly
  // (identical row order => identical last-writer on aliased cells).
  BoundPlan b = makeBound({16, 4}, {1, 1}, {4, 1}, 4);
  int64_t srcElems = product(b.extents); // oversized; aliased reads stay inside
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), device.data());

  std::vector<uint8_t> want(src.size(), 0xEE), got(src.size(), 0xEE);
  reloc::executeD2H(b, device.data(), want.data());
  HostBackend backend(2);
  reloc::executeD2HPipelined(b, device.data(), got.data(), backend,
                             /*nBuffers=*/2, /*override=*/16,
                             /*gatherThreads=*/8);
  EXPECT_EQ(got, want);
}

TEST(Pipeline, D2HCallerOwnedGatherPool) {
  // The caller-owned-pool overload must match the per-call overload.
  BoundPlan b = makeBound({4096, 1024}, {1, 4096}, {1024, 1}, 4);
  int64_t srcElems = product(b.extents);
  std::vector<uint8_t> src = iotaBytes(srcElems, b.elementSize);
  std::vector<uint8_t> device(static_cast<size_t>(b.totalBytes), 0xAB);
  reloc::executeH2D(b, src.data(), device.data());

  HostBackend backend(2);
  reloc::GatherPool gather(4);
  for (int call = 0; call < 2; ++call) {
    std::vector<uint8_t> back(static_cast<size_t>(srcElems) * b.elementSize,
                              0xCD);
    reloc::executeD2HPipelined(b, device.data(), back.data(), backend,
                               /*nBuffers=*/2, /*override=*/size_t(2) << 20,
                               gather);
    EXPECT_EQ(back, src) << "call " << call;
  }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ninja -C build/sym libreloc-test`
Expected: FAIL — no overload of `executeD2HPipelined` takes 7 arguments / a `GatherPool`.

- [ ] **Step 3: Implement**

In `libreloc/include/reloc/Pipeline.h`, replace the D2H declaration with:

```cpp
/// Strategy 4 inverse (D2H). gatherThreads parallelizes each chunk's
/// scatter (0 = hardware concurrency; 1 = inline, the pre-D1 behavior).
/// Non-injective src layouts (aliasing/broadcast strides) scatter inline
/// regardless, so the result is always bit-identical to executeD2H.
void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride = 0,
                         unsigned gatherThreads = 1);

/// As above with a CALLER-OWNED GatherPool (reused across calls).
void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather);
```

In `libreloc/src/Pipeline.cpp`, add to the anonymous namespace (after `dispatchRows`; also add `#include <utility>` if not present):

```cpp
// Sufficient condition for parallel scatter: valid index tuples map
// injectively to src element offsets, so workers holding disjoint outer
// ranges write disjoint src bytes -- interleaved (e.g. transposed src) is
// fine, colliding is not. Standard mixed-radix test over stride-sorted
// axes; broadcast/degenerate strides (<= 0 with extent > 1) fail.
// Conservative: false only costs parallelism (scatter runs inline), never
// correctness.
bool srcRowsWriteDisjoint(const BoundPlan &b) {
  std::vector<std::pair<int64_t, int64_t>> ax; // (stride, extent), extent > 1
  for (size_t k = 0; k < b.extents.size(); ++k) {
    if (b.extents[k] <= 1)
      continue;
    if (b.srcStrides[k] <= 0)
      return false;
    ax.emplace_back(b.srcStrides[k], b.extents[k]);
  }
  std::sort(ax.begin(), ax.end());
  int64_t span = 0; // max element offset reachable by smaller-stride axes
  for (const auto &se : ax) {
    if (se.first <= span)
      return false;
    span += (se.second - 1) * se.first;
  }
  return true;
}
```

Rename the existing `executeD2HPipelined` body into a file-local impl taking `GatherPool *gather` and route both public overloads through it:

```cpp
namespace {

void d2hPipelinedImpl(const BoundPlan &bound, const void *deviceSrc,
                      void *srcBaseV, CopyBackend &backend, int nBuffers,
                      size_t chunkSizeOverride, GatherPool *gather) {
  assert(nBuffers >= 1 && "nBuffers must be >= 1");
  ChunkSchedule sched = planChunks(bound, nBuffers, chunkSizeOverride);
  PinnedBufferPool pool(backend, nBuffers, sched.maxChunkBytes);
  const int nStreams = backend.numQueues();
  const bool parallelSafe = srcRowsWriteDisjoint(bound);

  struct InFlight {
    int buf;
    EventHandle ev;
    size_t chunk;
  };
  std::deque<InFlight> inflight;

  // Wait for a chunk's D2H copy to land, then scatter its valid cells from the
  // staging buffer into srcBaseV. Frees the staging buffer for reuse. The
  // dispatchRows barrier returns only after every sub-range completed, so the
  // buffer-reuse reasoning below is unchanged by parallel scatter.
  auto scatterOne = [&](const InFlight &f) {
    const Chunk &c = sched.chunks[f.chunk];
    backend.waitEvent(f.ev);
    if (c.validEnd > c.validBegin) {
      const uint8_t *dstBase = rebase(pool.buffer(f.buf), bound, c.paddedBegin);
      dispatchRows(gather, parallelSafe, sched.rowBytes, c.validBegin,
                   c.validEnd, [&](int64_t rb, int64_t re) {
                     scatterChunk(bound, dstBase, srcBaseV, rb, re);
                   });
    }
  };

  for (size_t k = 0; k < sched.chunks.size(); ++k) {
    const Chunk &c = sched.chunks[k];
    int i = pool.acquire(); // blocks on this buffer's prior copy event
    int q = static_cast<int>(k % static_cast<size_t>(nStreams));
    backend.copyAsync(q, pool.buffer(i),
                      static_cast<const uint8_t *>(deviceSrc) + c.byteOffset,
                      c.bytes, CopyDir::DeviceToHost);
    EventHandle ev = backend.recordEvent(q);
    pool.setEvent(i, ev);
    inflight.push_back({i, ev, k});
    // Keep at most pool.nBuffers() copies outstanding; drain the oldest
    // (which uses the buffer we are about to reuse next) before it is
    // overwritten. The deferred scatterOne(front) below (waitEvent + scatter)
    // runs synchronously on this single driver thread strictly before the
    // next acquire() reuses that same buffer, so the buffer is fully drained
    // and scattered before any new D2H copy overwrites it.
    if (static_cast<int>(inflight.size()) == pool.nBuffers()) {
      scatterOne(inflight.front());
      inflight.pop_front();
    }
  }
  for (const InFlight &f : inflight)
    scatterOne(f);
  pool.drain();
}

} // namespace

void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, unsigned gatherThreads) {
  if (gatherThreads == 1) {
    d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                     chunkSizeOverride, /*gather=*/nullptr);
    return;
  }
  GatherPool gather(gatherThreads);
  d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                   chunkSizeOverride, &gather);
}

void executeD2HPipelined(const BoundPlan &bound, const void *deviceSrc,
                         void *srcBaseV, CopyBackend &backend, int nBuffers,
                         size_t chunkSizeOverride, GatherPool &gather) {
  d2hPipelinedImpl(bound, deviceSrc, srcBaseV, backend, nBuffers,
                   chunkSizeOverride, &gather);
}
```

Note: the anonymous-namespace block for `d2hPipelinedImpl` must appear after `dispatchRows`/`srcRowsWriteDisjoint` definitions; keep a single `namespace { ... }` region per location rather than scattering many.

- [ ] **Step 4: Run to verify it passes**

Run: `ninja -C build/sym libreloc-test && ./build/sym/libreloc/test/libreloc-test`
Expected: PASS — full suite, including the four new D2H tests and the existing round-trip/pool tests.

- [ ] **Step 5: Commit**

```bash
git add libreloc/include/reloc/Pipeline.h libreloc/src/Pipeline.cpp \
        libreloc/test/PipelineTest.cpp
git commit -m "update(libreloc): D1 parallel per-chunk scatter in the D2H pipeline (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: pybind surface — `GatherPool` object + `gather_threads`

**Files:**
- Modify: `libreloc/python/PyReloc.cpp`
- Modify: `libreloc/python/pyreloc/__init__.py`
- Test: `libreloc/python/tests/test_gather_pool.py` (create)

**Interfaces:**
- Consumes: `GatherPool` (Task 1), threaded pipeline overloads (Tasks 2–3).
- Produces (Python API):
  - `pyreloc.GatherPool(threads=0)` — `.threads` (resolved count), `.closed`, `.close()` (idempotent), context manager (`__exit__` closes). Owns `threads-1` OS threads.
  - `pyreloc.relocate(bound, src_ptr, src_nbytes, dst_ptr, dst_nbytes, gather_threads=1, gather_pool=None)` — the two new kwargs affect only the `chunked_pipeline` strategy; `gather_pool` wins over `gather_threads`; a closed pool or negative `gather_threads` raises `ValueError`.
  - `pyreloc.h2d(..., n_buffers=4, n_streams=2, gather_threads=1, gather_pool=None)` and `pyreloc.d2h(...)` likewise (CUDA builds).

- [ ] **Step 1: Write the failing tests**

Create `libreloc/python/tests/test_gather_pool.py`:

```python
"""D1 (issue #65): GatherPool lifecycle + parallel-gather parity through the
pybind surface. Teardown is ASSERTED via the process thread count (/proc,
Linux-only -- which is what CI runs), not assumed."""
import pathlib
import sys

import numpy as np
import pytest

from conftest import golden_hex

pyreloc = pytest.importorskip("pyreloc")


def _thread_count():
    status = pathlib.Path("/proc/self/status").read_text()
    return int(status.split("Threads:")[1].split()[0])


def _bind_reference(n=256, strategy="chunked_pipeline"):
    plan = pyreloc.load_plan(bytes.fromhex(golden_hex("reference")))
    return pyreloc.bind(plan, {"N": n}, strategy=strategy)


def _ptr(arr):
    assert arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data, arr.nbytes


def test_gather_pool_thread_count_resolves():
    pool = pyreloc.GatherPool()  # threads=0 -> hardware concurrency
    assert pool.threads >= 1
    pool.close()
    pool = pyreloc.GatherPool(threads=3)
    assert pool.threads == 3
    pool.close()


@pytest.mark.skipif(sys.platform != "linux", reason="/proc thread count")
def test_gather_pool_close_joins_all_threads():
    base = _thread_count()
    pool = pyreloc.GatherPool(threads=4)
    # The pool owns T-1 OS workers; the caller is the T-th worker.
    assert _thread_count() == base + 3
    pool.close()
    assert pool.closed
    assert _thread_count() == base
    pool.close()  # idempotent
    assert _thread_count() == base


@pytest.mark.skipif(sys.platform != "linux", reason="/proc thread count")
def test_gather_pool_context_manager_tears_down():
    base = _thread_count()
    with pyreloc.GatherPool(threads=2) as pool:
        assert not pool.closed
        assert _thread_count() == base + 1
    assert pool.closed
    assert _thread_count() == base


def test_relocate_gather_threads_parity():
    # N=2048 (16 MiB): big enough that the 1 MiB/worker byte floor lets the
    # parallel path actually engage (N=256 would collapse to inline).
    bound = _bind_reference(n=2048)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 255, bound.min_src_bytes, dtype=np.uint8)
    outs = []
    for threads in (1, 2, 0):  # 0 -> hardware concurrency
        dst = np.zeros(bound.total_bytes, dtype=np.uint8)
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst),
                         gather_threads=threads)
        outs.append(dst.tobytes())
    assert outs[0] == outs[1] == outs[2]
    # And byte-identical to the single-thread reference strategy.
    ref_bound = _bind_reference(n=2048, strategy="single_thread_simd")
    ref = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(ref_bound, *_ptr(src), *_ptr(ref))
    assert outs[0] == ref.tobytes()


def test_relocate_with_reused_gather_pool():
    bound = _bind_reference(n=2048)  # see parity test: engages the pool
    rng = np.random.default_rng(1)
    src = rng.integers(0, 255, bound.min_src_bytes, dtype=np.uint8)
    ref = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(bound, *_ptr(src), *_ptr(ref))
    with pyreloc.GatherPool(threads=4) as pool:
        for _ in range(3):  # the SAME pool serves several calls
            dst = np.zeros(bound.total_bytes, dtype=np.uint8)
            pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_pool=pool)
            assert dst.tobytes() == ref.tobytes()


def test_closed_pool_is_rejected():
    bound = _bind_reference()
    src = np.zeros(bound.min_src_bytes, dtype=np.uint8)
    dst = np.zeros(bound.total_bytes, dtype=np.uint8)
    pool = pyreloc.GatherPool(threads=2)
    pool.close()
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_pool=pool)


def test_invalid_thread_counts_rejected():
    bound = _bind_reference()
    src = np.zeros(bound.min_src_bytes, dtype=np.uint8)
    dst = np.zeros(bound.total_bytes, dtype=np.uint8)
    with pytest.raises(ValueError):
        pyreloc.relocate(bound, *_ptr(src), *_ptr(dst), gather_threads=-1)
    with pytest.raises(ValueError):
        pyreloc.GatherPool(threads=-2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD/build/sym/python python3 -m pytest libreloc/python/tests/test_gather_pool.py -q`
Expected: FAIL — `AttributeError: module 'pyreloc' has no attribute 'GatherPool'` for the pool tests, `TypeError: ... unexpected keyword argument 'gather_threads'` for the relocate tests.

- [ ] **Step 3: Implement the bindings**

In `libreloc/python/PyReloc.cpp`:

Add `#include "reloc/GatherPool.h"` after the `reloc/Execute.h` include and `#include <memory>` to the std includes.

Add a helper in the anonymous namespace (after `checkBuffer`):

```cpp
// Shared validation for the gather-parallelism kwargs. Runs with the GIL
// held (before any release), so py::value_error is safe. gather_pool wins
// over gather_threads when both are given.
void checkGatherArgs(int gatherThreads,
                     const std::shared_ptr<reloc::GatherPool> &pool) {
  if (gatherThreads < 0)
    throw py::value_error("gather_threads must be >= 0 (0 = all cores)");
  if (pool && pool->closed())
    throw py::value_error("gather_pool is closed");
}
```

Replace `relocateHost` with:

```cpp
void relocateHost(const reloc::BoundPlan &b, uintptr_t srcPtr, size_t srcBytes,
                  uintptr_t dstPtr, size_t dstBytes, int gatherThreads,
                  std::shared_ptr<reloc::GatherPool> gatherPool) {
  checkBuffer("src", srcPtr, srcBytes, minSrcBytes(b));
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *src = reinterpret_cast<const void *>(srcPtr);
  void *dst = reinterpret_cast<void *>(dstPtr);
  py::gil_scoped_release release;
  switch (b.strategy) {
  case reloc::Strategy::MultiThreadTiled:
    reloc::executeH2DThreaded(b, src, dst);
    break;
  case reloc::Strategy::ChunkedPipeline: {
    reloc::HostBackend backend(2);
    if (gatherPool)
      reloc::executeH2DPipelined(b, src, dst, backend, /*nBuffers=*/2,
                                 /*chunkSizeOverride=*/0, *gatherPool);
    else
      reloc::executeH2DPipelined(b, src, dst, backend, /*nBuffers=*/2,
                                 /*chunkSizeOverride=*/0,
                                 static_cast<unsigned>(gatherThreads));
    break;
  }
  default:
    // Auto / ViewNoCopy / SingleThreadSimd all materialize via the
    // single-thread copy (a no_copy view has nothing to publish across a
    // language boundary that handed us a destination buffer).
    reloc::executeH2D(b, src, dst);
    break;
  }
}
```

Replace `h2dCuda` / `d2hCuda` with gather-aware versions (same pattern; the `#else` arm gains `(void)gatherThreads, (void)gatherPool;`):

```cpp
void h2dCuda(const reloc::BoundPlan &b, uintptr_t srcPtr, size_t srcBytes,
             uintptr_t dstPtr, size_t dstBytes, int nBuffers, int nStreams,
             int gatherThreads, std::shared_ptr<reloc::GatherPool> gatherPool) {
#ifdef RELOC_ENABLE_CUDA
  checkBuffer("src", srcPtr, srcBytes, minSrcBytes(b));
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *src = reinterpret_cast<const void *>(srcPtr);
  void *dst = reinterpret_cast<void *>(dstPtr);
  py::gil_scoped_release release;
  reloc::CudaBackend backend(nStreams);
  if (gatherPool)
    reloc::executeH2DPipelined(b, src, dst, backend, nBuffers,
                               /*chunkSizeOverride=*/0, *gatherPool);
  else
    reloc::executeH2DPipelined(b, src, dst, backend, nBuffers,
                               /*chunkSizeOverride=*/0,
                               static_cast<unsigned>(gatherThreads));
#else
  (void)b, (void)srcPtr, (void)srcBytes, (void)dstPtr, (void)dstBytes;
  (void)nBuffers, (void)nStreams, (void)gatherThreads, (void)gatherPool;
  throw std::runtime_error("pyreloc was built without RELOC_ENABLE_CUDA");
#endif
}

void d2hCuda(const reloc::BoundPlan &b, uintptr_t dstPtr, size_t dstBytes,
             uintptr_t srcPtr, size_t srcBytes, int nBuffers, int nStreams,
             int gatherThreads, std::shared_ptr<reloc::GatherPool> gatherPool) {
#ifdef RELOC_ENABLE_CUDA
  checkBuffer("dst", dstPtr, dstBytes, static_cast<size_t>(b.totalBytes));
  checkBuffer("src(out)", srcPtr, srcBytes, minSrcBytes(b));
  checkGatherArgs(gatherThreads, gatherPool);
  const void *dst = reinterpret_cast<const void *>(dstPtr);
  void *src = reinterpret_cast<void *>(srcPtr);
  py::gil_scoped_release release;
  reloc::CudaBackend backend(nStreams);
  if (gatherPool)
    reloc::executeD2HPipelined(b, dst, src, backend, nBuffers,
                               /*chunkSizeOverride=*/0, *gatherPool);
  else
    reloc::executeD2HPipelined(b, dst, src, backend, nBuffers,
                               /*chunkSizeOverride=*/0,
                               static_cast<unsigned>(gatherThreads));
#else
  (void)b, (void)dstPtr, (void)dstBytes, (void)srcPtr, (void)srcBytes;
  (void)nBuffers, (void)nStreams, (void)gatherThreads, (void)gatherPool;
  throw std::runtime_error("pyreloc was built without RELOC_ENABLE_CUDA");
#endif
}
```

In `PYBIND11_MODULE`, add the class binding before `m.def("relocate", ...)`:

```cpp
  py::class_<reloc::GatherPool, std::shared_ptr<reloc::GatherPool>>(
      m, "GatherPool",
      "Persistent gather/scatter worker pool (issue #65). threads == 0 "
      "resolves to the hardware thread count; the pool owns threads-1 OS "
      "workers (the calling thread is the last worker of each dispatch). "
      "close() joins every worker deterministically -- also usable as a "
      "context manager. A closed pool cannot be passed to relocate/h2d/d2h.")
      .def(py::init([](int threads) {
             if (threads < 0)
               throw py::value_error("threads must be >= 0 (0 = all cores)");
             return std::make_shared<reloc::GatherPool>(
                 static_cast<unsigned>(threads));
           }),
           py::arg("threads") = 0)
      .def_property_readonly("threads", &reloc::GatherPool::threadCount)
      .def_property_readonly("closed", &reloc::GatherPool::closed)
      .def("close", &reloc::GatherPool::close,
           py::call_guard<py::gil_scoped_release>(),
           "Join all workers. Idempotent.")
      .def("__enter__",
           [](const std::shared_ptr<reloc::GatherPool> &p) { return p; })
      .def("__exit__", [](reloc::GatherPool &p, const py::object &,
                          const py::object &, const py::object &) {
        p.close();
        return false;
      });
```

Extend the four `m.def` argument lists (defaults keep every existing call site working):

```cpp
  m.def("relocate", &relocateHost, py::arg("bound"), py::arg("src_ptr"),
        py::arg("src_nbytes"), py::arg("dst_ptr"), py::arg("dst_nbytes"),
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
        "Host relocation (CPU strategies): src -> dst-layout buffer. "
        "gather_threads / gather_pool parallelize the chunked_pipeline "
        "strategy's per-chunk gather (0 = all cores; a given pool wins).");
```

and for `h2d` / `d2h`, after `py::arg("n_streams") = 2`:

```cpp
        py::arg("gather_threads") = 1, py::arg("gather_pool") = nullptr,
```

(keep their existing docstrings, appending: `"gather_threads/gather_pool parallelize per-chunk gather (scatter for d2h)."`).

In `libreloc/python/pyreloc/__init__.py`, add `GatherPool` to the import list (alphabetical, after `DecodeError`):

```python
from ._pyreloc import (  # noqa: F401
    BindError,
    BoundPlan,
    DecodeError,
    GatherPool,
    PlanHandle,
    bind,
    cuda_enabled,
    d2h,
    h2d,
    load_plan,
    relocate,
    relocate_inverse,
)
```

- [ ] **Step 4: Run to verify it passes**

```bash
ninja -C build/sym pyreloc_ext
PYTHONPATH=$PWD/build/sym/python python3 -m pytest libreloc/python/tests -q
```

Expected: PASS — the whole pytest suite including the 8 new tests (GPU tests skip). If a thread-count assertion is flaky because an unrelated library spawned a thread mid-test, re-measure `base` immediately before pool construction (the tests already do) and re-run; persistent failure = real leak, investigate.

- [ ] **Step 5: Commit**

```bash
git add libreloc/python/PyReloc.cpp libreloc/python/pyreloc/__init__.py \
        libreloc/python/tests/test_gather_pool.py
git commit -m "update(libreloc): D1 pybind GatherPool + gather_threads on relocate/h2d/d2h (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Gather-BW microbench + committed JSON

**Files:**
- Create: `bench/gather_bw.cpp`
- Modify: `bench/CMakeLists.txt`
- Create: `bench/results/gather_bw_n4096.json` (generated by running the bench)

**Interfaces:**
- Consumes: `bench/protocol.h` (`runOnce`, `summarize`, `analyzeReruns`, `seriesToJson`, `jsonNumber`), `bench/reference_plan.h` (`referencePlanBytes`), `reloc::decodePlan`, `reloc::bind`, `reloc::gatherChunk`, `reloc::GatherPool`, `reloc::kMinGatherBytesPerWorker`.
- Produces: `bench-gather-bw` executable — `--n N --threads T --json PATH|- --warmup W --iters I --reruns R`; JSON with `methods.gather_1thread` and `methods.gather_multithread`, each carrying a `wall_ms` series and per-rerun `gb_per_s`. This number seeds E2.

- [ ] **Step 1: Write the driver**

Create `bench/gather_bw.cpp`:

```cpp
//===- gather_bw.cpp - D1 gather-bandwidth micro-benchmark ----------------===//
//
// Issue #65's benchmark (and E2's seed data point): gatherChunk bandwidth on
// the golden reference plan, single-thread vs a GatherPool of T workers,
// through bench/protocol.h. Pure CPU gather into a malloc'd dst-layout
// buffer -- no CopyBackend -- so the number isolates exactly the primitive
// the pipeline parallelizes, dispatched with the same per-worker byte floor
// the pipeline applies.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace {

// The pipeline's per-worker floor (Pipeline.h) in rows of this plan, so the
// bench measures the exact dispatch executeH2DPipelined performs per chunk.
int64_t minRowsPerWorker(const reloc::BoundPlan &b) {
  int64_t rowBytes = b.dstStrides[0] * static_cast<int64_t>(b.elementSize);
  return std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowBytes));
}

struct Measurement {
  bench::Series wall;
  std::vector<double> gbPerS; // per rerun, from that rerun's median
};

Measurement measureGather(const reloc::BoundPlan &b, const uint8_t *src,
                          uint8_t *dst, unsigned threads, int warmup,
                          int iters, int reruns) {
  reloc::GatherPool pool(threads);
  const int64_t outer = b.extents[0];
  const int64_t minRows = minRowsPerWorker(b);
  std::vector<std::vector<double>> wallPerRerun;
  for (int r = 0; r < reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(
        [&] {
          pool.parallelFor(0, outer, minRows, [&](int64_t rb, int64_t re) {
            reloc::gatherChunk(b, src, dst, rb, re);
          });
        },
        warmup, iters);
    wallPerRerun.push_back(std::move(s.wall_ms));
  }
  Measurement m;
  m.wall = bench::analyzeReruns(wallPerRerun);
  for (const bench::Stats &st : m.wall.reruns)
    m.gbPerS.push_back(st.median > 0
                           ? static_cast<double>(b.totalBytes) /
                                 (st.median * 1e-3) / 1e9
                           : 0.0);
  return m;
}

std::string gbToJson(const std::vector<double> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += bench::jsonNumber(v[i]);
  }
  return out + "]";
}

int run(int64_t n, unsigned threads, const char *jsonPath, int warmup,
        int iters, int reruns) {
  std::vector<uint8_t> bytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan) {
    std::fprintf(stderr, "error: golden reference plan failed to decode\n");
    return 1;
  }
  auto boundResult = reloc::bind(*plan, {{"N", n}});
  auto *b = std::get_if<reloc::BoundPlan>(&boundResult);
  if (!b) {
    std::fprintf(stderr, "error: bind failed for N=%lld\n",
                 static_cast<long long>(n));
    return 1;
  }

  // Source sized by the max element offset reachable via srcStrides.
  int64_t maxOff = 0;
  for (size_t k = 0; k < b->extents.size(); ++k)
    maxOff += (b->extents[k] - 1) * b->srcStrides[k];
  std::vector<uint8_t> src(static_cast<size_t>(maxOff + 1) * b->elementSize);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>((i * 131) & 0xff);
  std::vector<uint8_t> dst(static_cast<size_t>(b->totalBytes), 0);

  // Correctness gate before timing: the pool-dispatched gather must match
  // executeH2D byte-for-byte (a wrong benchmark is worse than none).
  {
    std::vector<uint8_t> ref(static_cast<size_t>(b->totalBytes), 0);
    reloc::executeH2D(*b, src.data(), ref.data());
    reloc::GatherPool pool(threads);
    pool.parallelFor(0, b->extents[0], minRowsPerWorker(*b),
                     [&](int64_t rb, int64_t re) {
                       reloc::gatherChunk(*b, src.data(), dst.data(), rb, re);
                     });
    if (std::memcmp(ref.data(), dst.data(), ref.size()) != 0) {
      std::fprintf(stderr, "error: parallel gather mismatch vs executeH2D\n");
      return 1;
    }
  }

  Measurement single =
      measureGather(*b, src.data(), dst.data(), 1, warmup, iters, reruns);
  reloc::GatherPool probe(threads); // resolve 0 -> hw for reporting
  const int threadsResolved = probe.threadCount();
  probe.close();
  Measurement multi = measureGather(*b, src.data(), dst.data(), threads,
                                    warmup, iters, reruns);

  std::string doc =
      "{\n  \"config\": {\"benchmark\": \"gather_bw\", \"plan\": "
      "\"reference\", \"N\": " +
      std::to_string(n) +
      ", \"total_bytes\": " + std::to_string(b->totalBytes) +
      ", \"threads_multi\": " + std::to_string(threadsResolved) +
      ", \"min_rows_per_worker\": " + std::to_string(minRowsPerWorker(*b)) +
      ", \"warmup\": " + std::to_string(warmup) +
      ", \"iters\": " + std::to_string(iters) +
      ", \"reruns\": " + std::to_string(reruns) +
      "},\n  \"methods\": {\n    \"gather_1thread\": {\"wall_ms\": " +
      bench::seriesToJson(single.wall) +
      ", \"gb_per_s\": " + gbToJson(single.gbPerS) +
      "},\n    \"gather_multithread\": {\"wall_ms\": " +
      bench::seriesToJson(multi.wall) +
      ", \"gb_per_s\": " + gbToJson(multi.gbPerS) + "}\n  }\n}\n";
  if (std::strcmp(jsonPath, "-") == 0) {
    std::fputs(doc.c_str(), stdout);
  } else {
    std::FILE *f = std::fopen(jsonPath, "w");
    if (!f) {
      std::fprintf(stderr, "error: cannot write %s\n", jsonPath);
      return 1;
    }
    std::fputs(doc.c_str(), f);
    std::fclose(f);
  }
  std::fprintf(stderr,
               "gather_bw: N=%lld 1T %.2f GB/s, %dT %.2f GB/s (%.2fx), "
               "rerun spread %.2f%% / %.2f%%\n",
               static_cast<long long>(n), single.gbPerS.front(),
               threadsResolved, multi.gbPerS.front(),
               multi.gbPerS.front() / single.gbPerS.front(),
               single.wall.medianSpreadPct, multi.wall.medianSpreadPct);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  int64_t n = 4096;
  unsigned threads = 0; // 0 = hardware concurrency
  const char *jsonPath = "-";
  int warmup = bench::kWarmupIters, iters = bench::kTimedIters,
      reruns = bench::kReruns;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--n")
      n = std::atoll(next());
    else if (a == "--threads")
      threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--json")
      jsonPath = next();
    else if (a == "--warmup")
      warmup = std::atoi(next());
    else if (a == "--iters")
      iters = std::atoi(next());
    else if (a == "--reruns")
      reruns = std::atoi(next());
    else {
      std::fprintf(stderr,
                   "usage: bench-gather-bw [--n N] [--threads T] "
                   "[--json PATH|-] [--warmup W] [--iters I] [--reruns R]\n");
      return 2;
    }
  }
  if (n <= 0 || n % 64 != 0) {
    std::fprintf(stderr,
                 "error: N must be positive and divisible by 64 (got %lld)\n",
                 static_cast<long long>(n));
    return 2;
  }
  if (warmup < 0 || iters < 1 || reruns < 1) {
    std::fprintf(stderr,
                 "error: warmup must be >= 0 and iters/reruns must be >= 1 "
                 "(got warmup=%d, iters=%d, reruns=%d)\n",
                 warmup, iters, reruns);
    return 2;
  }
  return run(n, threads, jsonPath, warmup, iters, reruns);
}
```

Add to `bench/CMakeLists.txt` after the `bench-bind-cost-smoke` block:

```cmake
add_executable(bench-gather-bw gather_bw.cpp)
target_include_directories(bench-gather-bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-gather-bw PRIVATE cxx_std_17)
target_link_libraries(bench-gather-bw PRIVATE reloc_runtime)
# Smoke-run under ctest (same rationale as bench-bind-cost-smoke): proves the
# D1 gather-BW driver runs end-to-end -- including its parallel-vs-executeH2D
# byte-exactness gate -- on every CI run. Tiny counts; not a measurement.
add_test(NAME bench-gather-bw-smoke
         COMMAND bench-gather-bw --n 256 --threads 2 --json - --warmup 1
                 --iters 3 --reruns 2)
```

- [ ] **Step 2: Build and smoke-run**

```bash
ninja -C build/sym bench-gather-bw
ctest --test-dir build/sym -R bench-gather-bw-smoke --output-on-failure
```

Expected: test PASSES; the JSON document prints to stdout with both `gather_1thread` and `gather_multithread` methods and a `gather_bw: N=256 ...` stderr summary.

- [ ] **Step 3: Produce and commit the real measurement**

```bash
./build/sym/bench/bench-gather-bw --n 4096 --threads 0 \
  --json bench/results/gather_bw_n4096.json
cat bench/results/gather_bw_n4096.json
```

Expected: stderr line like `gather_bw: N=4096 1T <x> GB/s, 32T <y> GB/s (<s>x), rerun spread <a>% / <b>%`. Sanity-check the JSON: both methods present, `gb_per_s` values > 0, multithread ≥ single-thread (if it is not, that is still an honest committable result — the repo committed an honest 0.36× before — but mention it in the PR). If `median_spread_pct` exceeds ~5%, close other workloads and re-run once.

- [ ] **Step 4: Commit**

```bash
git add bench/gather_bw.cpp bench/CMakeLists.txt \
        bench/results/gather_bw_n4096.json
git commit -m "update(bench): D1 gather-BW microbench, 1T vs multi-T JSON (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: TSan verification, docs, final sweep

**Files:**
- Modify: `libreloc/README.md`
- No other source changes expected (fixes only if TSan finds races).

- [ ] **Step 1: TSan build and run** (local verification of the issue's "TSan clean" acceptance; CI has no sanitizer job)

```bash
cmake -G Ninja -S . -B build/sym-tsan \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread" \
  -DMLIR_DIR=$PWD/build/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$PWD/build/llvm-project/build/bin/llvm-lit
ninja -C build/sym-tsan libreloc-test
TSAN_OPTIONS=halt_on_error=1 \
  ./build/sym-tsan/libreloc/test/libreloc-test \
  --gtest_filter='GatherPool.*:Pipeline.*:Backend.*:Pool.*'
```

Expected: all filtered tests PASS with **zero** `WARNING: ThreadSanitizer` reports. (Caveat to note in the PR: `llvm_gtest` links uninstrumented from the LLVM build tree; the pool/pipeline code itself is fully instrumented.) If TSan reports a race in `GatherPool`/`Pipeline`, fix it and re-run Tasks 1–3's test steps before proceeding — do not suppress.

- [ ] **Step 2: Document the new surface**

In `libreloc/README.md`, insert a new bullet in the `## Surface` section, directly after the `executeH2DPipelined` / `executeD2HPipelined` bullet:

```markdown
- `reloc::GatherPool` (`reloc/GatherPool.h`) — D1's persistent worker pool
  (issue #65): the pipeline partitions each chunk's valid outer rows across
  the pool's threads (`gatherThreads` argument or a caller-owned pool), with
  a per-worker byte floor (`kMinGatherBytesPerWorker`) so tiny chunks stay
  inline, and a counting barrier before `copyAsync` / staging reuse.
  Conservative safety guards fall back to inline gather/scatter — serialized
  (non-row-disjoint-dst) schedules for H2D, non-injective src layouts for
  D2H — so output stays bit-identical to `executeH2D`/`executeD2H`, and
  `gatherThreads == 1` never constructs a pool. Explicit `close()` lifecycle
  for pybind (`libreloc/test/GatherPoolTest.cpp`).
```

In the `## Python bindings (pyreloc)` section's opening paragraph, extend the function list sentence: after ``and `h2d` / `d2h` (the C5 pinned/stream pipeline, `RELOC_ENABLE_CUDA` builds only; `pyreloc.cuda_enabled` reports which you have)`` append:

```markdown
`relocate`/`h2d`/`d2h` accept `gather_threads=` (0 = all cores) or a
reusable `gather_pool=pyreloc.GatherPool(threads)` — a context manager
whose `close()` joins its workers deterministically, so no pool threads
outlive the interpreter (issue #65).
```

- [ ] **Step 3: Full local sweep (everything CI will run for this area)**

```bash
ninja -C build/sym libreloc-test pyreloc_ext bench-gather-bw
ctest --test-dir build/sym -R 'libreloc|reloc-runtime|bench' --output-on-failure
PYTHONPATH=$PWD/build/sym/python python3 -m pytest libreloc/python/tests -q
```

Expected: everything PASSES, including `reloc-runtime-no-mlir-includes` (proves no MLIR header crept in) and `reloc-runtime-mlir-free`.

- [ ] **Step 4: Commit and open the PR**

```bash
git add libreloc/README.md
git commit -m "update(libreloc): D1 document GatherPool surface + pyreloc kwargs (#65)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin HEAD
gh pr create --repo JueonPark/sym \
  --title "update(libreloc, bench): D1 persistent gather worker pool + parallel per-chunk gather/scatter (#65)" \
  --body "$(cat <<'EOF'
Implements #65.

- `reloc::GatherPool`: persistent workers, counting-barrier `parallelFor`,
  explicit `close()` lifecycle.
- Pipeline: per-chunk gather/scatter partitioned across the pool
  (`gatherThreads` / caller-owned pool); per-worker 1 MiB byte floor;
  guards fall back inline for serialized schedules (H2D) and non-injective
  src layouts (D2H). `gather_threads == 1` never constructs a pool.
- pybind: `pyreloc.GatherPool` (context manager, thread teardown asserted
  via /proc in tests), `gather_threads`/`gather_pool` on relocate/h2d/d2h.
- bench: `bench-gather-bw` (protocol.h), smoke-tested in ctest;
  `bench/results/gather_bw_n4096.json` committed (1T <X> GB/s vs 32T <Y>
  GB/s -- E2 seed data).
- TSan (local, `-fsanitize=thread` build of libreloc-test,
  GatherPool/Pipeline/Backend/Pool filters): clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(Fill `<X>`/`<Y>` from the committed JSON before running. Branch/worktree: create via the `superpowers:using-git-worktrees` skill at execution start, e.g. branch `d1-gather-pool`.)

---

## Acceptance-criteria traceability (issue #65)

| Criterion | Where |
|---|---|
| Byte-exact vs `executeH2D` across plan corpus × buffers {1,2,4} × threads {1,2,hw} on `HostBackend` in CI | Task 2/3: `H2DByteExactMatrix` + `D2HByteExactMatrix` threads dimension (runs under existing CI ctest) |
| TSan clean | Task 6 Step 1 (local TSan build; noted in PR) |
| `gather_threads = 1` bit-identical to current pipeline | Tasks 2/3: `gatherThreads == 1` path constructs no pool (code); matrices pin exactness; `dispatchRows` inline fast path |
| Teardown: no leaked threads after pybind context close (asserted) | Task 4: `test_gather_pool_close_joins_all_threads` (`/proc/self/status` thread count) |
| Gather-BW microbench JSON committed, single- + multi-thread | Task 5: `bench/results/gather_bw_n4096.json` |
| Worker-pool context object, explicit lifecycle | Task 1 (`GatherPool`), Task 4 (pybind class) |
| Per-chunk dispatch, ≤ T sub-ranges, rows floor, counting barrier before `copyAsync` | Task 2 (`dispatchRows` + `parallelFor` barrier) |
| D2H parity | Task 3 |
| `gather_threads` through both executors + pybind, 0 → cores | Tasks 2–4 |
