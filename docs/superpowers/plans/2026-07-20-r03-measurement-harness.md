# R0.3 Pipeline & Measurement Harness Implementation Plan (issue #76)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `bench/rtrack/` — the Method-A vs Method-B end-to-end (transform+transfer) latency harness over the merged R0.1 CPU quant kernels and R0.2 GPU kernels, with chunk-size sweep, per-stage timing, session calibration, CSV output, and the Figure-1 matplotlib script.

**Architecture:** All plans are hand-authored `BoundPlan`s (the `CudaKernelsTest::transposePlan` convention) verified against independent index-math oracles — the harness must NOT decode `bench/reference_plan.h`, whose golden blob on main still carries the non-injective pre-#63-fix strides. CPU-testable pieces (plans, chunking, stats, CSV, workload table) are header-only and run under ctest in CI; the pipeline driver `rtrack_bench.cu` is GPU-only and CUDA-gated like `bench-poc-transpose`. Python scripts (calibration, sweep runner, Figure 1) orchestrate around the C++ driver.

**Tech Stack:** C++17 (std + libreloc only), CUDA runtime API, gtest (llvm_gtest in-tree), Python 3 stdlib + matplotlib.

## Global Constraints

- Issue #76 protocol: **5 warmup + 30 timed** iterations; report **median, min, p95**; flag any config with **IQR/median > 5%**.
- Chunk sweep default **C ∈ {4, 16, 64, 256} MiB**; best-C is per method (never fixed globally) — selection happens in `figure1.py`, the driver reports every C.
- Isolated metric: end-to-end (transform+transfer) latency for one tensor, **source in pageable host DRAM**, staging = **2 pinned buffers of chunk size**, destination = final layout in GPU global memory.
- Full-pipeline timing via CUDA events (start recorded before the first stage, stop after the last enqueued op, then `cudaStreamSynchronize`); CPU stages via `steady_clock` with compiler fences.
- CSV columns exactly: `machine,gpu,method,transform,N,dtype_out,r,threads,chunk_req_mib,staging_bytes,n_chunks,median_ms,min_ms,p95_ms,iqr_over_median_pct,unstable,effective_input_GBps,gpu_pipeline_ms,cpu_stage_ms,h2d_ms,gpu_kernel_ms,verified`.
- Every timed config is verified against a CPU reference first (repo rule: a wrong benchmark is worse than none). Verification oracles must be independent of the plan being executed (issue #63 lesson).
- bench code: no MLIR/LLVM headers; LLVM style, 80 col; includes as `"rtrack/foo.h"` (include dir = `bench/`).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Local builds: CPU tests via existing `build/sym` Ninja tree; GPU driver via standalone `nvcc -ccbin g++` (plain gcc has a broken cc1plus on this box); nvcc at `/usr/local/cuda-12.5/bin/nvcc`, `-arch=sm_75`.

---

### Task 0: Branch setup

**Files:** none

- [ ] **Step 1:** `git -C /home/jueonpark/sym switch -c rtrack-r03-harness main` (work in the main checkout — the configured `build/sym` tree is bound to this path; status is clean except untracked `.claude/` and `docs/superpowers/`).
- [ ] **Step 2:** Confirm CI expectations: read `.github/workflows/build.yml`; the new `bench-rtrack-test` must build without CUDA and run under ctest.

### Task 1: Plans + index-math oracle tests

**Files:**
- Create: `bench/rtrack/plans.h`
- Create: `bench/rtrack/RtrackTest.cpp`
- Modify: `bench/CMakeLists.txt` (add `bench-rtrack-test` target + ctest)

**Interfaces:**
- Produces: `bench::rtrack::identityPlan(int64_t n)`, `transposePlan(n)`, `blockedTransposePlan(n)`, `nchwToNhwcPlan(n)` → `reloc::BoundPlan`; `maxSrcOffset(const reloc::BoundPlan &)` → `int64_t`.
- Plan shapes (all fp32 source, N² elements, packed dst, `n % 64 == 0`):
  - identity: extents `{n, n}`, src `{n, 1}`, dst `{n, 1}`.
  - transpose: extents `{n, n}`, src `{1, n}`, dst `{n, 1}`.
  - blocked transpose (the sym#63 anchor, CORRECTED axes, authored pre-coalesced rank-3 with the b1+n1 merge bind() would produce): m = n/64; extents `{64, m, n}`, src `{n, 64*n, 1}`, dst `{n*m, n, 1}`.
  - NCHW→NHWC with (B,C,H,W) = (n/64, 64, 64, n/64), coalesced dst order (b,h,w,c): extents `{n/64, 64, n/64, 64}`, src `{64*n, n/64, 1, n}`, dst `{64*n, n, 64, 1}`.
- Before setting `BoundPlan::L`, grep `libreloc/src` for consumers of `.L`; set it to the true innermost contiguous run of the authored axes (identity: n; transpose: 1; blocked: n; nchw: 1) if consumed, else leave default with a comment.

- [ ] **Step 1: Write the failing tests** (`bench/rtrack/RtrackTest.cpp`):

```cpp
//===- RtrackTest.cpp - R0.3 harness CPU-side tests (CI) ------------------===//
//
// Everything here runs without a GPU: plan builders vs independent
// index-math oracles (the issue-#63 lesson: never verify a plan against
// its own executor), chunk math, the 5+30 stats summary, CSV formatting,
// the workload table, and the quant round-trip error bound.
//
//===----------------------------------------------------------------------===//

#include "rtrack/plans.h"

#include "reloc/Execute.h"
#include "reloc/Quant.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using bench::rtrack::blockedTransposePlan;
using bench::rtrack::identityPlan;
using bench::rtrack::maxSrcOffset;
using bench::rtrack::nchwToNhwcPlan;
using bench::rtrack::transposePlan;

// Walk the plan's full dst index space; call check(dstOff, srcOff).
template <class Fn>
void forEachCell(const reloc::BoundPlan &b, Fn check) {
  const size_t r = b.extents.size();
  std::vector<int64_t> idx(r, 0);
  while (true) {
    int64_t so = 0, dso = 0;
    for (size_t k = 0; k < r; ++k) {
      so += idx[k] * b.srcStrides[k];
      dso += idx[k] * b.dstStrides[k];
    }
    check(idx, dso, so);
    size_t k = r;
    for (;;) {
      if (k == 0)
        return;
      --k;
      if (++idx[k] < b.extents[k])
        break;
      idx[k] = 0;
    }
  }
}

// src offsets must be a bijection onto [0, n^2) -- the exact property the
// pre-fix golden reference plan violated for N > 4096.
void expectBijective(const reloc::BoundPlan &b, int64_t total) {
  std::vector<int64_t> offs;
  offs.reserve(static_cast<size_t>(total));
  forEachCell(b, [&](const std::vector<int64_t> &, int64_t, int64_t so) {
    offs.push_back(so);
  });
  ASSERT_EQ(static_cast<int64_t>(offs.size()), total);
  std::sort(offs.begin(), offs.end());
  for (int64_t i = 0; i < total; ++i)
    ASSERT_EQ(offs[static_cast<size_t>(i)], i);
}

TEST(RtrackPlans, IdentityIsIdentity) {
  const int64_t n = 64;
  auto b = identityPlan(n);
  forEachCell(b, [&](const std::vector<int64_t> &, int64_t dso, int64_t so) {
    ASSERT_EQ(dso, so);
  });
  expectBijective(b, n * n);
  EXPECT_EQ(maxSrcOffset(b), n * n - 1);
}

TEST(RtrackPlans, TransposeMatchesIndexMath) {
  const int64_t n = 64;
  auto b = transposePlan(n);
  // dst (i, j) holds src (j, i): srcOff = j * n + i.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t dso, int64_t so) {
    ASSERT_EQ(dso, idx[0] * n + idx[1]);
    ASSERT_EQ(so, idx[1] * n + idx[0]);
  });
  expectBijective(b, n * n);
}

TEST(RtrackPlans, BlockedTransposeMatchesViewTransposeOracle) {
  const int64_t n = 128, m = n / 64;
  auto b = blockedTransposePlan(n);
  // out = x.view(N/64, 64, 64, N/64).transpose(0, 1). The plan is authored
  // rank-3 (a, bq, j) with j the merged (c, d) inner pair: c = j / m,
  // d = j % m. x_view strides (row-major): (64*n, n, m, 1), so
  // src = bq*64n + a*n + c*m + d, computed here WITHOUT the plan's strides.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t, int64_t so) {
    const int64_t a = idx[0], bq = idx[1], j = idx[2];
    const int64_t c = j / m, d = j % m;
    ASSERT_EQ(so, bq * 64 * n + a * n + c * m + d);
  });
  expectBijective(b, n * n);
}

TEST(RtrackPlans, NchwToNhwcMatchesIndexMath) {
  const int64_t n = 128;
  const int64_t B = n / 64, C = 64, H = 64, W = n / 64;
  auto b = nchwToNhwcPlan(n);
  // dst (b, h, w, c) packed NHWC; src NCHW: b*CHW + c*HW + h*W + w.
  forEachCell(b, [&](const std::vector<int64_t> &idx, int64_t dso, int64_t so) {
    const int64_t bb = idx[0], h = idx[1], w = idx[2], c = idx[3];
    ASSERT_EQ(dso, ((bb * H + h) * W + w) * C + c);
    ASSERT_EQ(so, ((bb * C + c) * H + h) * W + w);
  });
  expectBijective(b, n * n);
}

// The blocked plan must also round-trip through the library executor --
// authored strides and executeH2D agree on a real buffer.
TEST(RtrackPlans, BlockedTransposeExecutesBijectively) {
  const int64_t n = 128;
  auto b = blockedTransposePlan(n);
  std::vector<float> src(static_cast<size_t>(n * n));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<float>(i);
  std::vector<float> dst(src.size(), -1.0f);
  reloc::executeH2D(b, src.data(), dst.data());
  std::vector<float> sorted = dst;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i)
    ASSERT_EQ(sorted[i], static_cast<float>(i));
}

} // namespace
```

- [ ] **Step 2:** Add to `bench/CMakeLists.txt` (after the bench-quant-bw block):

```cmake
# R0.3 (issue #76): the rtrack pipeline & measurement harness. The CPU-side
# pieces (plan builders, chunk math, stats, CSV, workload table) run under
# ctest on every CI run; the pipeline driver below is GPU-only and
# CUDA-gated like bench-poc-transpose.
add_executable(bench-rtrack-test rtrack/RtrackTest.cpp)
target_include_directories(bench-rtrack-test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-rtrack-test PRIVATE cxx_std_17)
target_link_libraries(bench-rtrack-test PRIVATE reloc_runtime llvm_gtest
  llvm_gtest_main)
add_test(NAME bench-rtrack-test COMMAND bench-rtrack-test)
```

- [ ] **Step 3:** Run: `cmake --build /home/jueonpark/sym/build/sym --target bench-rtrack-test`. Expected: FAIL (`rtrack/plans.h` not found).
- [ ] **Step 4: Implement `bench/rtrack/plans.h`:**

```cpp
//===- plans.h - hand-authored BoundPlans for the R-track workloads -------===//
//
// R0.3 (issue #76). Every plan is authored directly as a BoundPlan (the
// CudaKernelsTest convention) instead of decoding bench/reference_plan.h:
// the frozen golden blob on main does not encode the intended blocked
// transpose for N > 4096 (issue #63), and RtrackTest verifies each builder
// here against independent index math -- which a decode-then-execute
// self-check cannot do. Axes are authored in coalesced dst-descending
// order, packed dst, elementSize 4 (fp32 source).
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_PLANS_H
#define BENCH_RTRACK_PLANS_H

#include "reloc/Bind.h"

#include <cassert>
#include <cstdint>

namespace bench {
namespace rtrack {

/// [N, N] contiguous copy (T3/T5's host-read pattern; no relocation).
inline reloc::BoundPlan identityPlan(int64_t n) {
  reloc::BoundPlan b;
  b.extents = {n, n};
  b.srcStrides = {n, 1};
  b.dstStrides = {n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = n;
  return b;
}

/// Plain 2-D transpose: dst row i = src column i (T1/T2; the tiled-SMEM
/// shape for relocateF32).
inline reloc::BoundPlan transposePlan(int64_t n) {
  reloc::BoundPlan b;
  b.extents = {n, n};
  b.srcStrides = {1, n};
  b.dstStrides = {n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = 1;
  return b;
}

/// The sym#63 anchor: x.view(N/64, 64, 64, N/64).transpose(0, 1), with the
/// CORRECTED axes from the issue-#63 fix, pre-coalesced to rank 3 exactly
/// as bind() would merge them (b1+n1 -> one src- and dst-contiguous run of
/// N elements).
inline reloc::BoundPlan blockedTransposePlan(int64_t n) {
  assert(n % 64 == 0);
  const int64_t m = n / 64;
  reloc::BoundPlan b;
  b.extents = {64, m, n};
  b.srcStrides = {n, 64 * n, 1};
  b.dstStrides = {n * m, n, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = n;
  return b;
}

/// T4: NCHW -> NHWC with (B, C, H, W) = (N/64, 64, 64, N/64) so the total
/// stays N^2. Coalesced dst order (b, h, w, c); src strides index NCHW.
inline reloc::BoundPlan nchwToNhwcPlan(int64_t n) {
  assert(n % 64 == 0);
  const int64_t B = n / 64, C = 64, H = 64, W = n / 64;
  (void)B;
  reloc::BoundPlan b;
  b.extents = {n / 64, H, W, C};
  b.srcStrides = {C * H * W, W, 1, H * W};
  b.dstStrides = {H * W * C, W * C, C, 1};
  b.elementSize = 4;
  b.totalBytes = n * n * 4;
  b.L = 1;
  return b;
}

/// Largest source element offset reachable through the plan (buffer
/// sizing; equals N^2 - 1 for every bijective builder above).
inline int64_t maxSrcOffset(const reloc::BoundPlan &b) {
  int64_t off = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    off += (b.extents[k] - 1) * b.srcStrides[k];
  return off;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_PLANS_H
```

(Adjust the `L` assignments per the Step-4 grep of `.L` consumers; document the finding in the header comment if it differs.)

- [ ] **Step 5:** Build + run: `cmake --build build/sym --target bench-rtrack-test && ./build/sym/bin/bench-rtrack-test` (binary location may be `build/sym/bench/...` — find with `find build/sym -name 'bench-rtrack-test'`). Expected: PASS.
- [ ] **Step 6:** Commit: `git add bench/rtrack/plans.h bench/rtrack/RtrackTest.cpp bench/CMakeLists.txt && git commit` — message `update(bench): R0.3 rtrack plan builders + index-math oracles (#76)`.

### Task 2: Chunk math

**Files:**
- Create: `bench/rtrack/chunking.h`
- Modify: `bench/rtrack/RtrackTest.cpp` (append tests)

**Interfaces:**
- Produces: `bench::rtrack::RowChunks {rowsPerChunk, nChunks, rowBytes, stagingBytes}`, `planRowChunks(rows, rowBytes, chunkBytes)`; `ByteChunks {bytesPerChunk, nChunks}`, `planByteChunks(totalBytes, chunkBytes)`.
- Semantics: Method A chunks on the dst outer axis (staging holds transformed output rows; `stagingBytes = rowsPerChunk * rowBytes`, which EXCEEDS the requested chunk when one row is bigger than C — recorded honestly in the CSV). Method B chunks the contiguous fp32 source by bytes; the last chunk may be short.

- [ ] **Step 1: Append failing tests:**

```cpp
#include "rtrack/chunking.h"

TEST(RtrackChunks, RowChunksCoverExactly) {
  auto c = bench::rtrack::planRowChunks(/*rows=*/100, /*rowBytes=*/1000,
                                        /*chunkBytes=*/4096);
  EXPECT_EQ(c.rowsPerChunk, 4);
  EXPECT_EQ(c.nChunks, 25);
  EXPECT_EQ(c.stagingBytes, 4000);
}

TEST(RtrackChunks, OversizedRowGetsOwnChunk) {
  auto c = bench::rtrack::planRowChunks(8, /*rowBytes=*/1 << 24,
                                        /*chunkBytes=*/1 << 22);
  EXPECT_EQ(c.rowsPerChunk, 1);
  EXPECT_EQ(c.nChunks, 8);
  EXPECT_EQ(c.stagingBytes, 1 << 24); // staging grows past the request
}

TEST(RtrackChunks, ChunkLargerThanTensorIsOneChunk) {
  auto c = bench::rtrack::planRowChunks(16, 64, 1 << 20);
  EXPECT_EQ(c.rowsPerChunk, 16);
  EXPECT_EQ(c.nChunks, 1);
  auto bc = bench::rtrack::planByteChunks(1024, 1 << 20);
  EXPECT_EQ(bc.bytesPerChunk, 1024);
  EXPECT_EQ(bc.nChunks, 1);
}

TEST(RtrackChunks, ByteChunksLastShort) {
  auto bc = bench::rtrack::planByteChunks(10 << 20, 4 << 20);
  EXPECT_EQ(bc.bytesPerChunk, 4 << 20);
  EXPECT_EQ(bc.nChunks, 3); // 4 + 4 + 2 MiB
}
```

- [ ] **Step 2:** Build/run → FAIL (missing header).
- [ ] **Step 3: Implement `bench/rtrack/chunking.h`:**

```cpp
//===- chunking.h - rtrack chunk planning ------------------------*- C++ -*-===//
//
// Method A chunks on the plan's dst outer axis (the staging buffer holds
// TRANSFORMED output rows, so the chunk size is measured in output bytes).
// Method B chunks the contiguous fp32 source by plain bytes. Double
// buffering means 2 x stagingBytes of pinned memory per config.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_CHUNKING_H
#define BENCH_RTRACK_CHUNKING_H

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace bench {
namespace rtrack {

struct RowChunks {
  int64_t rowsPerChunk;
  int64_t nChunks;
  int64_t rowBytes;
  int64_t stagingBytes; // rowsPerChunk * rowBytes; > chunkBytes when one
                        // row alone exceeds the requested chunk
};

inline RowChunks planRowChunks(int64_t rows, int64_t rowBytes,
                               int64_t chunkBytes) {
  assert(rows >= 1 && rowBytes >= 1 && chunkBytes >= 1);
  RowChunks c;
  c.rowBytes = rowBytes;
  c.rowsPerChunk =
      std::min<int64_t>(rows, std::max<int64_t>(1, chunkBytes / rowBytes));
  c.nChunks = (rows + c.rowsPerChunk - 1) / c.rowsPerChunk;
  c.stagingBytes = c.rowsPerChunk * rowBytes;
  return c;
}

struct ByteChunks {
  int64_t bytesPerChunk; // last chunk may be short
  int64_t nChunks;
};

inline ByteChunks planByteChunks(int64_t totalBytes, int64_t chunkBytes) {
  assert(totalBytes >= 1 && chunkBytes >= 1);
  ByteChunks c;
  c.bytesPerChunk = std::min(totalBytes, chunkBytes);
  c.nChunks = (totalBytes + c.bytesPerChunk - 1) / c.bytesPerChunk;
  return c;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_CHUNKING_H
```

- [ ] **Step 4:** Build/run → PASS.
- [ ] **Step 5:** Commit `update(bench): R0.3 rtrack chunk planning (#76)`.

### Task 3: Stats (5+30 protocol) + fenced clock

**Files:**
- Create: `bench/rtrack/rstats.h`
- Modify: `bench/rtrack/RtrackTest.cpp`

**Interfaces:**
- Produces: `bench::rtrack::kWarmup = 5`, `kIters = 30`, `kIqrFlagPct = 5.0`; `RStats {median, min, p95, iqrOverMedianPct, unstable, n}`; `summarizeSamples(std::vector<double>)`; `nowMs()` (fenced steady_clock, milliseconds).
- Reuses `bench::percentileSorted` from `bench/protocol.h`.

- [ ] **Step 1: Append failing tests:**

```cpp
#include "rtrack/rstats.h"

TEST(RtrackStats, ProtocolConstantsMatchIssue76) {
  EXPECT_EQ(bench::rtrack::kWarmup, 5);
  EXPECT_EQ(bench::rtrack::kIters, 30);
  EXPECT_DOUBLE_EQ(bench::rtrack::kIqrFlagPct, 5.0);
}

TEST(RtrackStats, MedianMinP95) {
  std::vector<double> s;
  for (int i = 20; i >= 1; --i)
    s.push_back(i); // 1..20
  auto r = bench::rtrack::summarizeSamples(s);
  EXPECT_DOUBLE_EQ(r.median, 10.5);
  EXPECT_DOUBLE_EQ(r.min, 1.0);
  EXPECT_DOUBLE_EQ(r.p95, 19.05); // numpy linear percentile
  EXPECT_EQ(r.n, 20u);
}

TEST(RtrackStats, UnstableFlagAtFivePercent) {
  // median 100, q1 99, q3 104 -> IQR 5 -> exactly 5% is NOT flagged (> only)
  std::vector<double> tight = {99, 99, 100, 100, 104, 104};
  auto t = bench::rtrack::summarizeSamples(tight);
  EXPECT_FALSE(t.unstable);
  std::vector<double> wide = {90, 95, 100, 100, 105, 111};
  auto w = bench::rtrack::summarizeSamples(wide);
  EXPECT_TRUE(w.unstable);
}

TEST(RtrackStats, NowMsMonotonic) {
  double a = bench::rtrack::nowMs();
  double b = bench::rtrack::nowMs();
  EXPECT_GE(b, a);
}
```

Check the `tight` fixture by hand with numpy-linear quartiles before committing; adjust values so IQR/median lands exactly at or just under 5% (and `wide` clearly above). If the arithmetic doesn't land exactly, use fixtures with obvious margins instead of boundary-exact ones.

- [ ] **Step 2:** Build/run → FAIL.
- [ ] **Step 3: Implement `bench/rtrack/rstats.h`:**

```cpp
//===- rstats.h - issue #76 measurement protocol ----------------*- C++ -*-===//
//
// 5 warmup + 30 timed iterations; report median, min, p95; flag any config
// with IQR/median > 5%. Percentiles reuse bench/protocol.h's numpy-linear
// interpolation. nowMs() is steady_clock bracketed by compiler fences so a
// stage boundary cannot be reordered around the read.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_RSTATS_H
#define BENCH_RTRACK_RSTATS_H

#include "protocol.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace bench {
namespace rtrack {

inline constexpr int kWarmup = 5;
inline constexpr int kIters = 30;
inline constexpr double kIqrFlagPct = 5.0;

struct RStats {
  double median = 0;
  double min = 0;
  double p95 = 0;
  double iqrOverMedianPct = 0;
  bool unstable = false;
  size_t n = 0;
};

inline RStats summarizeSamples(std::vector<double> samples) {
  RStats r;
  if (samples.empty())
    return r;
  std::sort(samples.begin(), samples.end());
  r.n = samples.size();
  r.min = samples.front();
  r.median = bench::percentileSorted(samples, 0.50);
  r.p95 = bench::percentileSorted(samples, 0.95);
  const double iqr = bench::percentileSorted(samples, 0.75) -
                     bench::percentileSorted(samples, 0.25);
  r.iqrOverMedianPct = r.median > 0 ? 100.0 * iqr / r.median : 0.0;
  r.unstable = r.iqrOverMedianPct > kIqrFlagPct;
  return r;
}

inline double nowMs() {
  asm volatile("" ::: "memory");
  const double t = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  asm volatile("" ::: "memory");
  return t;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_RSTATS_H
```

- [ ] **Step 4:** Build/run → PASS. Commit `update(bench): R0.3 rtrack measurement protocol stats (#76)`.

### Task 4: CSV emission

**Files:**
- Create: `bench/rtrack/csv.h`
- Modify: `bench/rtrack/RtrackTest.cpp`

**Interfaces:**
- Produces: `bench::rtrack::CsvRow` (fields listed in Global Constraints, stage columns are medians), `csvHeaderLine()`, `csvRowLine(const CsvRow &)`.

- [ ] **Step 1: Append failing tests:**

```cpp
#include "rtrack/csv.h"

#include <sstream>

namespace {
size_t commaCount(const std::string &s) {
  return static_cast<size_t>(std::count(s.begin(), s.end(), ','));
}
} // namespace

TEST(RtrackCsv, HeaderAndRowFieldCountsMatch) {
  bench::rtrack::CsvRow row;
  EXPECT_EQ(commaCount(bench::rtrack::csvHeaderLine()),
            commaCount(bench::rtrack::csvRowLine(row)));
}

TEST(RtrackCsv, RowGolden) {
  bench::rtrack::CsvRow r;
  r.machine = "epyc-2080ti";
  r.gpu = "NVIDIA GeForce RTX 2080 Ti";
  r.method = "a";
  r.transform = "transpose";
  r.n = 8192;
  r.r = 0.25;
  r.dtypeOut = "s8";
  r.threads = 8;
  r.chunkReqBytes = 4ll << 20;
  r.stagingBytes = 4ll << 20;
  r.nChunks = 16;
  r.wall = {12.5, 12.0, 13.75, 2.4, false, 30};
  r.gpuPipe = {12.4, 0, 0, 0, false, 30};
  r.cpuStage = {8.0, 0, 0, 0, false, 30};
  r.h2d = {2.75, 0, 0, 0, false, 30};
  r.gpuKernel = {0.0, 0, 0, 0, false, 30};
  r.effectiveInputGbps = 21.47;
  r.verified = true;
  EXPECT_EQ(bench::rtrack::csvRowLine(r),
            "epyc-2080ti,NVIDIA GeForce RTX 2080 Ti,a,transpose,8192,s8,0.25,"
            "8,4,4194304,16,12.5,12,13.75,2.4,0,21.47,12.4,8,2.75,0,1");
}
```

- [ ] **Step 2:** Build/run → FAIL.
- [ ] **Step 3: Implement `bench/rtrack/csv.h`:**

```cpp
//===- csv.h - rtrack CSV rows (issue #76 output format) --------*- C++ -*-===//
//
// One row per (workload, method, chunk) config. Session metadata (machine
// calibration, versions, environment controls) travels as '#'-prefixed
// header comment lines written by run_rtrack.py, not here. Stage columns
// are medians over the 30 timed iterations. No quoting: field values must
// not contain commas (asserted).
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_CSV_H
#define BENCH_RTRACK_CSV_H

#include "protocol.h"
#include "rtrack/rstats.h"

#include <cassert>
#include <cstdint>
#include <string>

namespace bench {
namespace rtrack {

struct CsvRow {
  std::string machine, gpu, method, transform, dtypeOut;
  int64_t n = 0;
  double r = 0;
  unsigned threads = 1;
  int64_t chunkReqBytes = 0;
  int64_t stagingBytes = 0;
  int64_t nChunks = 0;
  RStats wall, gpuPipe, cpuStage, h2d, gpuKernel;
  double effectiveInputGbps = 0;
  bool verified = false;
};

inline std::string csvHeaderLine() {
  return "machine,gpu,method,transform,N,dtype_out,r,threads,chunk_req_mib,"
         "staging_bytes,n_chunks,median_ms,min_ms,p95_ms,iqr_over_median_pct,"
         "unstable,effective_input_GBps,gpu_pipeline_ms,cpu_stage_ms,h2d_ms,"
         "gpu_kernel_ms,verified";
}

inline std::string csvRowLine(const CsvRow &r) {
  assert(r.machine.find(',') == std::string::npos &&
         r.gpu.find(',') == std::string::npos);
  auto num = [](double v) { return bench::jsonNumber(v); };
  std::string out;
  out += r.machine + ',' + r.gpu + ',' + r.method + ',' + r.transform + ',';
  out += std::to_string(r.n) + ',' + r.dtypeOut + ',' + num(r.r) + ',';
  out += std::to_string(r.threads) + ',';
  out += num(static_cast<double>(r.chunkReqBytes) / (1 << 20)) + ',';
  out += std::to_string(r.stagingBytes) + ',' + std::to_string(r.nChunks);
  out += ',' + num(r.wall.median) + ',' + num(r.wall.min) + ',' +
         num(r.wall.p95) + ',' + num(r.wall.iqrOverMedianPct) + ',' +
         (r.wall.unstable ? "1" : "0");
  out += ',' + num(r.effectiveInputGbps);
  out += ',' + num(r.gpuPipe.median) + ',' + num(r.cpuStage.median) + ',' +
         num(r.h2d.median) + ',' + num(r.gpuKernel.median);
  out += r.verified ? ",1" : ",0";
  return out;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_CSV_H
```

- [ ] **Step 4:** Build/run → PASS (fix the golden string if `jsonNumber` formatting differs — derive the expected string from the actual `%.6g` behavior, verifying each field by eye once).
- [ ] **Step 5:** Commit `update(bench): R0.3 rtrack CSV output format (#76)`.

### Task 5: Workload table + quant round-trip bound

**Files:**
- Create: `bench/rtrack/workloads.h`
- Modify: `bench/rtrack/RtrackTest.cpp`

**Interfaces:**
- Produces: `DtypeOut {F32, F16, S8}`, `dtypeName()`, `dtypeBytes()`; `CpuStage {GatherF32, GatherQuant, QuantPack, ConvertF16}`; `GpuStage {Relocate, RelocateQuant, Quantize, ConvertF16}`; `struct Workload {id, transform, dtypeOut, r, makePlan, cpuStage, gpuStage}`; `allWorkloads()` (T1, T1b, T2, T3, T4, T5), `findWorkload(id)`.

- [ ] **Step 1: Append failing tests:**

```cpp
#include "rtrack/workloads.h"

TEST(RtrackWorkloads, TableConsistent) {
  const auto &ws = bench::rtrack::allWorkloads();
  ASSERT_EQ(ws.size(), 6u);
  for (const auto &w : ws) {
    SCOPED_TRACE(w.id);
    auto b = w.makePlan(128);
    EXPECT_EQ(b.totalBytes, 128 * 128 * 4);
    EXPECT_EQ(bench::rtrack::maxSrcOffset(b), 128 * 128 - 1);
    // r is exactly the dtype width ratio (fp32 in).
    EXPECT_DOUBLE_EQ(w.r, bench::rtrack::dtypeBytes(w.dtypeOut) / 4.0);
    // Packed dst rows: outer stride == product of inner extents (the
    // chunked-staging rebase and the B-side per-channel quantize rely on
    // this).
    int64_t inner = 1;
    for (size_t k = 1; k < b.extents.size(); ++k)
      inner *= b.extents[k];
    EXPECT_EQ(b.dstStrides[0], inner);
    EXPECT_NE(bench::rtrack::findWorkload(w.id), nullptr);
  }
  EXPECT_EQ(bench::rtrack::findWorkload("nope"), nullptr);
}

TEST(RtrackWorkloads, GatherQuantizeOnIdentityEqualsQuantizePack) {
  // T3's Method-A kernel is quantizePackF32S8; the driver's s8 reference
  // for plan-strided workloads is gatherQuantizeF32S8. On the identity
  // plan they must agree bit-exactly.
  const int64_t n = 128;
  auto b = bench::rtrack::identityPlan(n);
  std::vector<float> src(static_cast<size_t>(n * n));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  std::vector<float> inv(static_cast<size_t>(n), 1.0f / 3.0f);
  std::vector<int8_t> a(src.size()), g(src.size());
  reloc::quant::quantizePackF32S8(src.data(), a.data(), n, n, inv.data(),
                                  reloc::quant::Variant::Scalar);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), g.data(), inv.data(), 0, n,
                                    reloc::quant::Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(a.data(), g.data(), a.size()));
}

TEST(RtrackWorkloads, QuantRoundTripMaxAbsErrBound) {
  // R0 exit criterion: |x - dequant(quant(x))| <= scale/2 for unsaturated
  // inputs when invScale = 127 / maxAbs.
  const int64_t n = 4096;
  std::vector<float> src(static_cast<size_t>(n));
  float maxAbs = 0;
  for (int64_t i = 0; i < n; ++i) {
    src[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.37f) * 100.0f;
    maxAbs = std::max(maxAbs, std::fabs(src[static_cast<size_t>(i)]));
  }
  const float invScale = 127.0f / maxAbs, scale = maxAbs / 127.0f;
  std::vector<int8_t> q(src.size());
  reloc::quant::quantizePackF32S8(src.data(), q.data(), 1, n, &invScale,
                                  reloc::quant::Variant::Scalar);
  double worst = 0;
  for (size_t i = 0; i < src.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(src[i]) -
                                      static_cast<double>(q[i]) * scale));
  EXPECT_LE(worst, 0.5 * scale * 1.0001);
}
```

- [ ] **Step 2:** Build/run → FAIL.
- [ ] **Step 3: Implement `bench/rtrack/workloads.h`:**

```cpp
//===- workloads.h - the R1 workload matrix (T1-T5 + anchor) ----*- C++ -*-===//
//
// Each workload names its plan, its Method-A per-chunk CPU transform
// (R0.1 kernels / gatherChunk) and its Method-B post-transfer GPU stage
// (R0.2 kernels). r = output bytes / input bytes; the final artifact of
// both methods is identical (dtypeOut in the plan's dst layout) and
// bit-exact comparable, per the R0.1/R0.2 CPU==GPU quantize contract.
// R2's dequant/unpack receive variants slot in as new GpuStage values.
//
//===----------------------------------------------------------------------===//

#ifndef BENCH_RTRACK_WORKLOADS_H
#define BENCH_RTRACK_WORKLOADS_H

#include "rtrack/plans.h"

#include <string>
#include <vector>

namespace bench {
namespace rtrack {

enum class DtypeOut { F32, F16, S8 };

inline const char *dtypeName(DtypeOut d) {
  switch (d) {
  case DtypeOut::F32:
    return "f32";
  case DtypeOut::F16:
    return "f16";
  case DtypeOut::S8:
    return "s8";
  }
  return "?";
}

inline int dtypeBytes(DtypeOut d) {
  switch (d) {
  case DtypeOut::F32:
    return 4;
  case DtypeOut::F16:
    return 2;
  case DtypeOut::S8:
    return 1;
  }
  return 0;
}

/// Method A's per-chunk CPU transform into pinned staging.
enum class CpuStage { GatherF32, GatherQuant, QuantPack, ConvertF16 };

/// Method B's post-transfer GPU kernel sequence.
enum class GpuStage { Relocate, RelocateQuant, Quantize, ConvertF16 };

struct Workload {
  const char *id;        // CLI name
  const char *transform; // CSV transform column
  DtypeOut dtypeOut;
  double r; // output bytes / input bytes
  reloc::BoundPlan (*makePlan)(int64_t n);
  CpuStage cpuStage;
  GpuStage gpuStage;
};

inline const std::vector<Workload> &allWorkloads() {
  static const std::vector<Workload> ws = {
      {"T1", "transpose", DtypeOut::F32, 1.0, &transposePlan,
       CpuStage::GatherF32, GpuStage::Relocate},
      {"T1b", "blocked_transpose", DtypeOut::F32, 1.0, &blockedTransposePlan,
       CpuStage::GatherF32, GpuStage::Relocate},
      {"T2", "transpose_quant", DtypeOut::S8, 0.25, &transposePlan,
       CpuStage::GatherQuant, GpuStage::RelocateQuant},
      {"T3", "quant", DtypeOut::S8, 0.25, &identityPlan, CpuStage::QuantPack,
       GpuStage::Quantize},
      {"T4", "nchw_nhwc_quant", DtypeOut::S8, 0.25, &nchwToNhwcPlan,
       CpuStage::GatherQuant, GpuStage::RelocateQuant},
      {"T5", "convert_f16", DtypeOut::F16, 0.5, &identityPlan,
       CpuStage::ConvertF16, GpuStage::ConvertF16},
  };
  return ws;
}

inline const Workload *findWorkload(const std::string &id) {
  for (const Workload &w : allWorkloads())
    if (id == w.id)
      return &w;
  return nullptr;
}

} // namespace rtrack
} // namespace bench

#endif // BENCH_RTRACK_WORKLOADS_H
```

(Test file needs `#include "reloc/Quant.h"`, `<cstring>`, `<cmath>` — already partly there.)

- [ ] **Step 4:** Build/run → PASS. Commit `update(bench): R0.3 rtrack workload table + quant round-trip bound (#76)`.

### Task 6: The pipeline driver

**Files:**
- Create: `bench/rtrack/rtrack_bench.cu`
- Modify: `bench/CMakeLists.txt` (CUDA-gated `bench-rtrack` target inside the existing `if(RELOC_ENABLE_CUDA)` block)

**Interfaces:**
- Consumes: everything above + `reloc::quant::*`, `reloc::cuda::*`, `reloc::GatherPool`, `reloc::executeH2D`, `reloc::gatherChunk`, `reloc::kMinGatherBytesPerWorker`.
- CLI: `bench-rtrack [--transform all|T1,T3,...] [--method both|a|b] [--n N] [--chunk-mib 4,16,64,256] [--threads T] [--variant auto|scalar|avx2|avx512] [--warmup 5] [--iters 30] [--machine NAME] [--csv PATH|-] [--csv-header] [--no-verify]`. Rows append to the CSV target; stderr gets one human-readable summary per row. Exit 1 on any verify failure.

- [ ] **Step 1: Write `bench/rtrack/rtrack_bench.cu`** (no CI test possible — the correctness gates ARE the test, run in Step 3):

```cpp
//===- rtrack_bench.cu - R0.3 pipeline & measurement harness (#76) --------===//
//
// Method A vs Method B end-to-end (transform + transfer) latency for one
// tensor. Source: pageable host DRAM. Staging: 2 pinned buffers of chunk
// size (double-buffered, event-gated). Destination: final layout in GPU
// global memory.
//   A: per-chunk CPU transform (R0.1 kernels / gatherChunk, parallelized
//      over a GatherPool) into pinned staging -> cudaMemcpyAsync of r*S
//      bytes total. (R2's dequant/unpack receive kernels slot in later.)
//   B: per-chunk pageable->pinned memcpy (parallelized over the same pool,
//      so both methods get T threads) -> cudaMemcpyAsync of S bytes ->
//      R0.2 GPU transform kernels into the final layout.
// Timing: full pipeline via CUDA events (start recorded before the first
// stage, stop after the last enqueued op, then cudaStreamSynchronize),
// fenced steady_clock wall time, and per-stage sums (CPU transform ms,
// summed per-chunk H2D event ms, GPU kernel event ms). Protocol: 5 warmup
// + 30 timed; median/min/p95; IQR/median > 5% flags the row.
// Every (workload, method, chunk) config is verified bit-exact against a
// CPU reference before it is timed; plans are hand-authored (rtrack/plans.h)
// and oracle-checked in RtrackTest, NOT decoded from the frozen golden blob
// (issue #63).
//
//===----------------------------------------------------------------------===//

#include "rtrack/chunking.h"
#include "rtrack/csv.h"
#include "rtrack/rstats.h"
#include "rtrack/workloads.h"

#include "reloc/CudaKernels.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err_), #x);                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

using namespace bench::rtrack;

// GPU side of T5. Bench-local (the poc_transpose precedent): issue #75's
// kernel set has no f32->f16 convert. __float2half_rn is round-to-nearest-
// even, matching the CPU convertF32F16 contract for non-NaN inputs (the
// generated data has no NaNs).
__global__ void convertF32F16Kernel(const float *src, __half *dst,
                                    int64_t count) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < count)
    dst[i] = __float2half_rn(src[i]);
}

void launchConvertF32F16(const float *dSrc, void *dDst, int64_t count,
                         cudaStream_t stream) {
  const int block = 256;
  const int64_t grid = (count + block - 1) / block;
  convertF32F16Kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(
      dSrc, static_cast<__half *>(dDst), count);
  CUDA_CHECK(cudaGetLastError());
}

struct Options {
  std::vector<std::string> transforms; // resolved workload ids
  std::string method = "both";
  int64_t n = 8192;
  std::vector<int64_t> chunkBytes = {4ll << 20, 16ll << 20, 64ll << 20,
                                     256ll << 20};
  unsigned threads = 1;
  reloc::quant::Variant variant = reloc::quant::Variant::Auto;
  int warmup = kWarmup;
  int iters = kIters;
  std::string machine;
  const char *csvPath = "-";
  bool csvHeader = false;
  bool verify = true;
};

std::string defaultMachine() {
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0)
    return "unknown";
  return host;
}

// Per-(workload, N) fixture: pageable source, per-channel scales, CPU
// reference of the final artifact, device buffers.
struct Fixture {
  const Workload *w = nullptr;
  reloc::BoundPlan bound;
  std::vector<float> hostSrc;   // pageable; N^2 elements
  std::vector<float> invScales; // extents[0] entries (quant workloads)
  std::vector<uint8_t> ref;     // expected final artifact, outBytes
  int64_t totalElems = 0, inBytes = 0, outBytes = 0;
  int64_t rows = 0, rowOutBytes = 0, channels = 0, channelSize = 0;
  // Device:
  void *dOut = nullptr;   // final artifact (A's DMA target, B's kernel dst)
  float *dLin = nullptr;  // B: linear fp32 source copy
  float *dTmp = nullptr;  // B RelocateQuant: relocated fp32 before quantize
  float *dInv = nullptr;  // per-channel invScales
};

bool needsQuant(const Workload &w) {
  return w.cpuStage == CpuStage::GatherQuant ||
         w.cpuStage == CpuStage::QuantPack;
}

void buildFixture(Fixture &f, const Workload &w, int64_t n, bool methodB) {
  f.w = &w;
  f.bound = w.makePlan(n);
  f.totalElems = n * n;
  f.inBytes = f.totalElems * 4;
  f.outBytes = f.totalElems * dtypeBytes(w.dtypeOut);
  f.rows = f.bound.extents[0];
  f.rowOutBytes = f.bound.dstStrides[0] * dtypeBytes(w.dtypeOut);
  f.channels = f.bound.extents[0];
  f.channelSize = f.totalElems / f.channels;

  f.hostSrc.resize(static_cast<size_t>(f.totalElems));
  for (int64_t i = 0; i < f.totalElems; ++i)
    f.hostSrc[static_cast<size_t>(i)] =
        (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;

  if (needsQuant(w)) {
    // Honest per-channel scales: channel = the plan's coalesced outer axis.
    f.invScales.assign(static_cast<size_t>(f.channels), 0.0f);
    const size_t rank = f.bound.extents.size();
    for (int64_t c = 0; c < f.channels; ++c) {
      float maxAbs = 0.0f;
      std::vector<int64_t> idx(rank, 0);
      idx[0] = c;
      while (true) {
        int64_t so = 0;
        for (size_t k = 0; k < rank; ++k)
          so += idx[k] * f.bound.srcStrides[k];
        maxAbs = std::max(maxAbs,
                          std::fabs(f.hostSrc[static_cast<size_t>(so)]));
        size_t k = rank;
        for (;;) {
          if (k <= 1)
            goto channelDone;
          --k;
          if (++idx[k] < f.bound.extents[k])
            break;
          idx[k] = 0;
        }
      }
    channelDone:
      f.invScales[static_cast<size_t>(c)] =
          maxAbs > 0 ? 127.0f / maxAbs : 1.0f;
    }
  }

  // CPU reference of the final artifact (scalar kernels; the plan itself
  // is oracle-verified in RtrackTest).
  f.ref.resize(static_cast<size_t>(f.outBytes));
  switch (w.dtypeOut) {
  case DtypeOut::F32:
    reloc::executeH2D(f.bound, f.hostSrc.data(), f.ref.data());
    break;
  case DtypeOut::S8:
    reloc::quant::gatherQuantizeF32S8(
        f.bound, f.hostSrc.data(), reinterpret_cast<int8_t *>(f.ref.data()),
        f.invScales.data(), 0, f.channels, reloc::quant::Variant::Scalar);
    break;
  case DtypeOut::F16:
    reloc::quant::convertF32F16(f.hostSrc.data(),
                                reinterpret_cast<uint16_t *>(f.ref.data()),
                                f.totalElems, reloc::quant::Variant::Scalar);
    break;
  }

  CUDA_CHECK(cudaMalloc(&f.dOut, static_cast<size_t>(f.outBytes)));
  if (methodB) {
    CUDA_CHECK(cudaMalloc(&f.dLin, static_cast<size_t>(f.inBytes)));
    if (w.gpuStage == GpuStage::RelocateQuant)
      CUDA_CHECK(cudaMalloc(&f.dTmp, static_cast<size_t>(f.inBytes)));
  }
  if (needsQuant(w)) {
    CUDA_CHECK(cudaMalloc(&f.dInv, static_cast<size_t>(f.channels) * 4));
    CUDA_CHECK(cudaMemcpy(f.dInv, f.invScales.data(),
                          static_cast<size_t>(f.channels) * 4,
                          cudaMemcpyHostToDevice));
  }
}

void freeFixture(Fixture &f) {
  cudaFree(f.dOut);
  cudaFree(f.dLin);
  cudaFree(f.dTmp);
  cudaFree(f.dInv);
  f.dOut = nullptr;
  f.dLin = f.dTmp = f.dInv = nullptr;
}

struct StageTimes {
  double wall = 0, gpu = 0, cpu = 0, h2d = 0, kern = 0;
};

// Shared pipeline scaffolding: stream, 2 pinned staging buffers, per-chunk
// H2D event pairs, pipeline start/stop + kernel events.
struct Pipeline {
  cudaStream_t stream = nullptr;
  void *staging[2] = {nullptr, nullptr};
  int64_t nChunks = 0;
  std::vector<cudaEvent_t> h2dBeg, h2dEnd;
  cudaEvent_t evStart = nullptr, evStop = nullptr, kBeg = nullptr,
              kEnd = nullptr;

  void init(int64_t stagingBytes, int64_t chunks) {
    nChunks = chunks;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (void *&p : staging)
      CUDA_CHECK(cudaHostAlloc(&p, static_cast<size_t>(stagingBytes),
                               cudaHostAllocDefault));
    h2dBeg.resize(static_cast<size_t>(chunks));
    h2dEnd.resize(static_cast<size_t>(chunks));
    for (int64_t c = 0; c < chunks; ++c) {
      CUDA_CHECK(cudaEventCreate(&h2dBeg[static_cast<size_t>(c)]));
      CUDA_CHECK(cudaEventCreate(&h2dEnd[static_cast<size_t>(c)]));
    }
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventCreate(&kBeg));
    CUDA_CHECK(cudaEventCreate(&kEnd));
  }

  void destroy() {
    for (auto &e : h2dBeg)
      cudaEventDestroy(e);
    for (auto &e : h2dEnd)
      cudaEventDestroy(e);
    h2dBeg.clear();
    h2dEnd.clear();
    if (evStart)
      cudaEventDestroy(evStart);
    if (evStop)
      cudaEventDestroy(evStop);
    if (kBeg)
      cudaEventDestroy(kBeg);
    if (kEnd)
      cudaEventDestroy(kEnd);
    evStart = evStop = kBeg = kEnd = nullptr;
    for (void *&p : staging) {
      cudaFreeHost(p);
      p = nullptr;
    }
    if (stream)
      cudaStreamDestroy(stream);
    stream = nullptr;
  }

  double sumH2dMs() const {
    double total = 0;
    for (int64_t c = 0; c < nChunks; ++c) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, h2dBeg[static_cast<size_t>(c)],
                                      h2dEnd[static_cast<size_t>(c)]));
      total += ms;
    }
    return total;
  }
};

// Method A: per-chunk CPU transform into staging, then DMA into the final
// artifact. The rebase pointer trick follows Execute.h's gatherChunk
// contract ("dstBase is the address at which dst element offset 0 would
// land -- rebase it for a staging buffer").
StageTimes runMethodA(const Fixture &f, const RowChunks &ck,
                      reloc::GatherPool &pool,
                      reloc::quant::Variant variant) {
  const Workload &w = *f.w;
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, ck.rowBytes));
  Pipeline &pl = *f.pl; // see note below: Pipeline stored on Fixture

  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int buf = static_cast<int>(c & 1);
    if (c >= 2)
      CUDA_CHECK(cudaEventSynchronize(pl.h2dEnd[static_cast<size_t>(c - 2)]));
    const int64_t rb = c * ck.rowsPerChunk;
    const int64_t re = std::min(f.rows, rb + ck.rowsPerChunk);
    char *stage = static_cast<char *>(pl.staging[buf]);
    const double t0 = nowMs();
    switch (w.cpuStage) {
    case CpuStage::GatherF32: {
      void *rebased = stage - rb * ck.rowBytes;
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::gatherChunk(f.bound, f.hostSrc.data(), rebased, sb, se);
      });
      break;
    }
    case CpuStage::GatherQuant: {
      int8_t *rebased =
          reinterpret_cast<int8_t *>(stage) - rb * f.bound.dstStrides[0];
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::gatherQuantizeF32S8(f.bound, f.hostSrc.data(), rebased,
                                          f.invScales.data(), sb, se, variant);
      });
      break;
    }
    case CpuStage::QuantPack:
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::quantizePackF32S8(
            f.hostSrc.data() + sb * f.channelSize,
            reinterpret_cast<int8_t *>(stage) + (sb - rb) * f.channelSize,
            se - sb, f.channelSize, f.invScales.data() + sb, variant);
      });
      break;
    case CpuStage::ConvertF16:
      pool.parallelFor(rb, re, minRows, [&](int64_t sb, int64_t se) {
        reloc::quant::convertF32F16(
            f.hostSrc.data() + sb * f.channelSize,
            reinterpret_cast<uint16_t *>(stage) + (sb - rb) * f.channelSize,
            (se - sb) * f.channelSize, variant);
      });
      break;
    }
    t.cpu += nowMs() - t0;
    const int64_t dstOff = rb * ck.rowBytes;
    const int64_t bytes = (re - rb) * ck.rowBytes;
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char *>(f.dOut) + dstOff, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  return t;
}

// Method B: per-chunk pageable->pinned memcpy + DMA of the raw fp32
// tensor, then the R0.2 transform kernels into the final layout.
StageTimes runMethodB(const Fixture &f, const ByteChunks &ck,
                      reloc::GatherPool &pool) {
  const Workload &w = *f.w;
  Pipeline &pl = *f.pl;
  const char *src = reinterpret_cast<const char *>(f.hostSrc.data());

  StageTimes t;
  const double w0 = nowMs();
  CUDA_CHECK(cudaEventRecord(pl.evStart, pl.stream));
  for (int64_t c = 0; c < ck.nChunks; ++c) {
    const int buf = static_cast<int>(c & 1);
    if (c >= 2)
      CUDA_CHECK(cudaEventSynchronize(pl.h2dEnd[static_cast<size_t>(c - 2)]));
    const int64_t off = c * ck.bytesPerChunk;
    const int64_t bytes = std::min(ck.bytesPerChunk, f.inBytes - off);
    char *stage = static_cast<char *>(pl.staging[buf]);
    const double t0 = nowMs();
    // Same thread budget as Method A's transform: split the staging copy.
    pool.parallelFor(0, bytes, 1 << 20, [&](int64_t bb, int64_t be) {
      std::memcpy(stage + bb, src + off + bb, static_cast<size_t>(be - bb));
    });
    t.cpu += nowMs() - t0;
    CUDA_CHECK(cudaEventRecord(pl.h2dBeg[static_cast<size_t>(c)], pl.stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(f.dLin) + off, stage,
                               static_cast<size_t>(bytes),
                               cudaMemcpyHostToDevice, pl.stream));
    CUDA_CHECK(cudaEventRecord(pl.h2dEnd[static_cast<size_t>(c)], pl.stream));
  }
  CUDA_CHECK(cudaEventRecord(pl.kBeg, pl.stream));
  switch (w.gpuStage) {
  case GpuStage::Relocate:
    reloc::cuda::relocateF32(f.bound, f.dLin, static_cast<float *>(f.dOut),
                             pl.stream);
    break;
  case GpuStage::RelocateQuant:
    reloc::cuda::relocateF32(f.bound, f.dLin, f.dTmp, pl.stream);
    reloc::cuda::quantizeF32S8(f.dTmp, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, pl.stream);
    break;
  case GpuStage::Quantize:
    reloc::cuda::quantizeF32S8(f.dLin, static_cast<int8_t *>(f.dOut),
                               f.channels, f.channelSize, f.dInv, pl.stream);
    break;
  case GpuStage::ConvertF16:
    launchConvertF32F16(f.dLin, f.dOut, f.totalElems, pl.stream);
    break;
  }
  CUDA_CHECK(cudaEventRecord(pl.kEnd, pl.stream));
  CUDA_CHECK(cudaEventRecord(pl.evStop, pl.stream));
  CUDA_CHECK(cudaStreamSynchronize(pl.stream));
  t.wall = nowMs() - w0;
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.evStart, pl.evStop));
  t.gpu = ms;
  t.h2d = pl.sumH2dMs();
  CUDA_CHECK(cudaEventElapsedTime(&ms, pl.kBeg, pl.kEnd));
  t.kern = ms;
  return t;
}

} // namespace

int main(int argc, char **argv) { /* CLI parse + the run loop; see Step 2 */ }
```

Notes locked in for implementation:
  - Store `Pipeline *pl` on `Fixture` (non-owning) or pass it as a parameter — pick passing as a parameter (`runMethodA(f, ck, pl, pool, variant)`), which avoids the mutable-fixture hack shown above. Adjust signatures accordingly.
  - The run loop per (workload, method, chunkReq): build `RowChunks`/`ByteChunks`, `Pipeline::init(stagingBytes or bytesPerChunk, nChunks)`, verify (memset `dOut` to 0xAB, run once, download, memcmp vs `f.ref`, exit 1 with a clear message on mismatch, skip if `--no-verify`), warmup loop, 30 timed iterations collecting `StageTimes`, summarize each field with `summarizeSamples`, fill `CsvRow` (`effectiveInputGbps = inBytes / (wall.median * 1e-3) / 1e9`), append `csvRowLine` to the CSV target, print a stderr summary (`rtrack: T3 a chunk=16MiB T=8 wall 41.2 ms (14.5 GB/s in, iqr 1.2%) [verified]`), `Pipeline::destroy()`.
  - Method A's `Pipeline::init` uses `ck.stagingBytes`; Method B's uses `ck.bytesPerChunk`.
  - Validate `--n`: positive, `n % 64 == 0`.
  - `--csv-header` prints `csvHeaderLine()` to the CSV target first.
  - GPU name via `cudaGetDeviceProperties(&prop, 0)`; machine defaults to `gethostname`.
  - One `reloc::GatherPool pool(threads)` for the whole run; `pool.close()` before exit.
  - CSV file opened in append mode (`"a"`).

- [ ] **Step 2:** Add to `bench/CMakeLists.txt` inside the existing `if(RELOC_ENABLE_CUDA)` block:

```cmake
  # R0.3 pipeline driver (issue #76). GPU-only, never in CI; run locally
  # per bench/rtrack/README.md.
  add_executable(bench-rtrack rtrack/rtrack_bench.cu)
  target_include_directories(bench-rtrack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(bench-rtrack PRIVATE cxx_std_17)
  target_link_libraries(bench-rtrack PRIVATE reloc_runtime CUDA::cudart)
```

- [ ] **Step 3: Standalone build + gated smoke on the 2080 Ti** (the CMake tree has CUDA OFF; use the proven standalone recipe):

```bash
cd /home/jueonpark/sym
/usr/local/cuda-12.5/bin/nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -Ilibreloc/include -Ibench \
  bench/rtrack/rtrack_bench.cu libreloc/src/*.cpp libreloc/quant/Quant.cpp \
  libreloc/cuda/CudaBackend.cu libreloc/cuda/CudaKernels.cu \
  -o /tmp/claude-2017/-home-jueonpark-sym/*/scratchpad/bench-rtrack \
  -Xcompiler -pthread
```

(Scalar-only quant build is fine for the smoke — SIMD TUs need per-TU flags; if AVX paths are wanted, add `-DRELOC_QUANT_HAVE_X86_SIMD=1` plus the two TUs with `-Xcompiler -mavx2,...` — but variant=auto must then resolve; simplest is to omit the SIMD TUs and pass `--variant scalar`... NO: `Variant::Auto` with no SIMD TUs compiled resolves to Scalar automatically, so plain build + default flags works.)

Run: `./bench-rtrack --transform all --method both --n 1024 --chunk-mib 4 --threads 2 --warmup 1 --iters 3 --csv-header --csv -`
Expected: header + 12 rows (6 workloads × 2 methods), all `verified` = 1, exit 0.

- [ ] **Step 4:** Fix until the gate passes for all 12 configs; then a second smoke at `--n 4096 --chunk-mib 4,16` (24 rows).
- [ ] **Step 5:** Commit `update(bench): R0.3 rtrack pipeline driver — Method A/B, staged timing, verify gates (#76)`.

### Task 7: Measurement sanity on this box + committed sample CSV

**Files:**
- Create: `bench/results/rtrack_smoke_n8192_epyc_2080ti.csv`

- [ ] **Step 1:** Full-protocol run: `--transform T1,T1b,T3 --method both --n 8192 --chunk-mib 4,16,64,256 --threads 8 --machine epyc7351-2080ti --csv-header --csv bench/results/rtrack_smoke_n8192_epyc_2080ti.csv`.
- [ ] **Step 2:** Sanity-check against known numbers for this box (bare-metal EPYC 7351 + 2080 Ti, PCIe gen3): H2D stage ≈ 11–13 GB/s; T1b Method-A gather at T=8 should exceed the 1-thread ~7.9 GB/s figure; Method B's `gpu_kernel_ms` small relative to H2D (the "flat" claim). Investigate anything wildly off before committing (systematic-debugging skill).
- [ ] **Step 3:** Commit the CSV: `update(bench): R0.3 smoke measurement on bare-metal EPYC/2080 Ti (#76)`.

### Task 8: Session calibration script

**Files:**
- Create: `bench/rtrack/calibrate.py`

**Interfaces:**
- Produces: `python3 bench/rtrack/calibrate.py [--out calibration.json] [--load-bin PATH_TO_bench-rtrack]` → JSON with: timestamp, hostname, kernel, cpu_model, governors (unique set), thp, numactl_available, driver_version, cuda_toolkit, gpus[] (name, pcie gen/width current+max, idle), pcie_under_load (sampled while `--load-bin` runs a short T1 method-b loop in the background; null with a note if no binary given), triad_gbps (compiles a tiny non-STREAM triad with `cc -O3 -march=native`, best of 5), nvbandwidth (h2d/d2h if the binary exists in PATH, else null + note).
- Every probe is best-effort: a missing tool records `null` plus a `notes` entry, never a crash. stdlib only.

- [ ] **Step 1:** Write the script (~180 lines): subprocess helpers with timeouts; triad C source written to a tempdir:

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// Simple schoolbook triad a[i] = b[i] + s*c[i] over 2^27 doubles x 5 reps;
// prints best GB/s (3 arrays * 8 bytes moved per element).
```

(Full triad source + the python code are written at implementation time; keep the JSON schema above stable — run_rtrack.py consumes it.)

- [ ] **Step 2:** Run it on this box: `python3 bench/rtrack/calibrate.py --out <scratch>/calibration.json`; eyeball JSON validity and that missing tools (numactl, nvbandwidth) degrade to notes.
- [ ] **Step 3:** Commit `update(bench): R0.3 per-session calibration recorder (#76)`.

### Task 9: Sweep runner (JSON config → CSV)

**Files:**
- Create: `bench/rtrack/run_rtrack.py`
- Create: `bench/rtrack/configs/example.json`

**Interfaces:**
- Config schema:

```json
{
  "machine": "epyc7351-2080ti",
  "bin": "./bench-rtrack",
  "numactl": "",
  "transforms": ["T1", "T1b", "T2", "T3", "T4", "T5"],
  "methods": "both",
  "n": [2048, 4096, 8192, 16384],
  "chunk_mib": [4, 16, 64, 256],
  "threads": [1, 2, 4, 8],
  "warmup": 5,
  "iters": 30,
  "calibration": "calibration.json",
  "out_csv": "rtrack.csv"
}
```

- Behavior: loads calibration JSON (runs calibrate.py first if the file is missing), writes `# key: value` comment lines (flattened calibration + config echo) then the header line (captured from `bin --csv-header --transform T1 --method a --n 64 ...`? No — simpler: the runner asks the driver to emit the header by launching the FIRST point with `--csv-header`), then loops (n × threads) invoking the driver once per pair with the full transform/method/chunk sweep, streaming driver stdout rows into the CSV, prefixing `numactl <args> --` when configured and available (warn when configured but missing). Nonzero driver exit aborts the run with the failing command echoed.

- [ ] **Step 1:** Write the script (~130 lines, stdlib only).
- [ ] **Step 2:** Smoke: tiny config (`n: [1024]`, `threads: [2]`, `chunk_mib: [4]`, `warmup: 1`, `iters: 3`, T1+T3) against the scratch driver binary → CSV exists, has `#` header lines + header + 4 rows.
- [ ] **Step 3:** Commit `update(bench): R0.3 sweep runner — JSON config to CSV (#76)` (include the example config).

### Task 10: Figure 1 script

**Files:**
- Create: `bench/rtrack/figure1.py`

**Interfaces:**
- `python3 bench/rtrack/figure1.py --csv gen3.csv [gen4.csv ...] [--n N] [--out figure1.png]`: skips `#` comments; for each (machine, transform): filter to `--n` (default: largest N present per machine), take **best C per method** = min `median_ms` over chunk rows (at the max thread count present unless `--threads` given), speedup = `median_ms(B) / median_ms(A)`; grouped bar chart (x = transform in T1..T5 order, one bar per machine), horizontal line at 1.0, best-C annotated on each bar, warn on any contributing `unstable` row (hatch the bar). matplotlib only.

- [ ] **Step 1:** Write the script (~140 lines, csv + matplotlib).
- [ ] **Step 2:** Smoke against the Task-7 CSV: `python3 bench/rtrack/figure1.py --csv bench/results/rtrack_smoke_n8192_epyc_2080ti.csv --out <scratch>/fig1.png` (matplotlib from `.venv` if not system-installed). Inspect the PNG renders.
- [ ] **Step 3:** Commit `update(bench): R0.3 Figure 1 plot script (#76)`.

### Task 11: README + environment controls

**Files:**
- Create: `bench/rtrack/README.md`

Content: what the harness measures (the issue #76 isolated-metric definition verbatim), build (CMake CUDA path + the standalone nvcc recipe), the session ritual (governor → performance, `nvidia-smi -lgc` where the driver allows, `numactl --membind=0`, note THP, run calibrate.py, then run_rtrack.py), CSV column glossary, the sym#63 regression-anchor procedure on the Gen4 box (gather ~14 / H2D ~24 / Method-B flat, ±10%) **with the caveat that those numbers predate the issue-#63 reference-plan fix and the anchor must be re-baselined against post-fix Gen4 runs**, and why plans are hand-authored (issue #63).

- [ ] **Step 1:** Write it. Commit `update(bench): R0.3 harness README + environment-control checklist (#76)`.

### Task 12: Verification, review, PR

- [ ] **Step 1:** Rebuild + run the full CPU suite: `cmake --build build/sym --target bench-rtrack-test bench-protocol-test bench-quant-bw bench-bind-cost bench-gather-bw && ctest --test-dir build/sym -R 'bench-' --output-on-failure`. All pass.
- [ ] **Step 2:** Re-run the standalone GPU driver smoke (`--n 4096`, all transforms, both methods) — all rows verified.
- [ ] **Step 3:** superpowers:verification-before-completion, then the code-review skill on the diff; fix findings.
- [ ] **Step 4:** Push `rtrack-r03-harness`; `gh pr create` targeting main. PR body: what/why, workload↔kernel map, local (Gen3, bare-metal) smoke numbers table, exit-criteria status (Gen4 anchor pending on the WSL2 box; anchor-number caveat re issue #63), `Closes #76`, generated-with footer.

## Self-Review Notes

- Spec coverage: double-buffered 2×chunk staging (Pipeline struct, Tasks 2/6); chunk sweep + per-method best-C (driver sweeps, figure1 selects); CUDA-event full-pipeline + fenced steady_clock CPU stages (rstats/driver); 5+30/median/min/p95/IQR-flag (rstats); calibration once per session (Task 8); JSON config → CSV rows with the issue's columns (Tasks 4/9); matplotlib Figure 1 (Task 10); env controls recorded + documented (Tasks 8/11); correctness bit-exact for relocate + quant round-trip bound (Tasks 5/6).
- Known deliberate scope choices: no dequant/unpack receive-path workloads (R2's r-sweep adds them as GpuStage values); Method B's GPU stage runs after the full transfer (matches the sym#63 baseline; stage_breakdown exposes kernel cost); T5's GPU convert kernel is bench-local (absent from issue #75's set, poc_transpose precedent).
- `Fixture::pl` note in Task 6 Step 1 is resolved by passing `Pipeline &` as a parameter.
