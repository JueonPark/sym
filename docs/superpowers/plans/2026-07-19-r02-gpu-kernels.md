# R0.2 GPU Kernels (`libreloc/cuda/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #75 — the eight R0.2 GPU kernels (`copy_f32`, `relocate_naive_f32`, `relocate_f32`, `quantize_f32_s8`, `dequant_s8_f32`, `unpack_s4_s8`, `dequant_relocate_s8_f32`, `scatter_random_f32`) compiled for sm_75 (Turing) and sm_89 (Ada), with correctness tests against the CPU reference implementations.

**Architecture:** New `reloc::cuda` namespace: host-callable launch wrappers declared in `libreloc/include/reloc/CudaKernels.h` (CUDA-handle-free — streams type-erased to `void *`, following `CudaBackend.h`), kernels defined in `libreloc/cuda/CudaKernels.cu`, compiled into `reloc_runtime` only under `RELOC_ENABLE_CUDA` with `CUDA_ARCHITECTURES "75;89"`. Plan-driven kernels (`relocate_*`, `dequant_relocate`) consume a `BoundPlan` host-side and pack extents/strides into a fixed-rank `Axes` struct (the `bench/poc_transpose.cu` pattern). Tests live in `libreloc/test/CudaKernelsTest.cpp`, `#ifdef RELOC_ENABLE_CUDA`-guarded, run locally on the GPU, never in CI (the `CudaPipelineTest.cpp` convention).

**Tech Stack:** CUDA C++ (nvcc, CUDA ≥ 13), C++17 host code, CMake + Ninja, googletest.

## Global Constraints

- **Scope boundary:** correctness kernels + tests ONLY. Bandwidth/pipeline measurement is R0.3 (issue #76). Do not add benchmark drivers here.
- **CUDA gating (repo convention):** all CUDA code compiles only under `RELOC_ENABLE_CUDA`; the test file is `#ifdef`-guarded and compiles to an empty TU in CI builds; GPU tests run locally only (dev box: RTX 4070 Ti SUPER, sm_89, WSL2).
- **Multi-arch:** `reloc_runtime` gets `CUDA_ARCHITECTURES "75;89"` — sm_75 is compile-proof for the future 2080 Ti box (M0 bring-up is separate); execution here is sm_89 only. Verify both ELFs with `cuobjdump`.
- **Header hygiene:** `CudaKernels.h` includes NO CUDA headers; streams are `void *` (`nullptr` = default stream) — the `CudaBackend.h` precedent. It may include `reloc/Bind.h` (host-only STL struct). The MLIR-free include scan already covers `libreloc/cuda/`.
- **GPU quantize is BIT-IDENTICAL to the CPU scalar contract:** `y = x * invScale; y = fmaxf(y, -128.0f); y = fminf(y, 127.0f); q = (int8_t)__float2int_rn(y)`. CUDA `fmaxf(NaN, -128.0f)` returns −128 (IEEE: non-NaN operand) and `__float2int_rn` is round-to-nearest-even — both match the CPU `quantOne` (`std::fmax`/`std::fmin`/`nearbyintf`). Tests enforce `memcmp` equality against `reloc::quant::quantizePackF32S8(..., Variant::Scalar)`.
- **Relocate kernels are byte-identical to CPU `executeH2D`** for the same `BoundPlan` (fp32, no pads). `relocate_f32`'s tiled and fallback paths are both bit-identical to `relocate_naive_f32`.
- **`unpack_s4_s8` is the exact inverse of the CPU `packS8S4`** on saturated values: `unpack(pack(x)) == clamp(x, -8, 7)`; low nibble = even index.
- **Per-channel scale conventions match the CPU side:** `quantize_f32_s8` / `dequant_s8_f32` use `channels × channelSize` contiguous layout, scale per channel; `dequant_relocate_s8_f32` uses channel = coalesced outer axis (`extents[0]` scales).
- **Plan-driven kernel preconditions (asserted host-side):** `elementSize == 4`, `padRegions.empty()`, `1 <= rank <= 8` (`kMaxRank`), and rank ≥ 2 for `dequant_relocate` (per-channel outer axis).
- **clang-format gate:** CI runs `clang-format-21 --dry-run --Werror` on every `*.cpp`/`*.h`. Before EVERY commit, run `/home/jueonpark/sym/.venv/bin/clang-format -i` on all touched `.h`, `.cpp`, AND `.cu` files (`.cu` isn't CI-gated but stays consistent), then verify with `--dry-run --Werror`.
- **Build trees (worktree-local):** CPU/CI-parity tree `build/sym` (CUDA OFF); CUDA tree `build/cuda` (`-DRELOC_ENABLE_CUDA=ON`). Tests for this plan build and run via `build/cuda`; run `build/sym` once at the end to prove the CUDA-off build stays green.
- **Commit style:** `update(libreloc): R0.2 <what> (#75)` + the Claude co-author trailer.
- **Error handling convention:** launch wrappers are `void` and asynchronous (caller syncs), matching `CudaBackend`'s no-throw style; tests `ASSERT_EQ(cudaSuccess, ...)` on every runtime call and `cudaDeviceSynchronize()` before reading results.

## File Structure

| File | Responsibility |
|---|---|
| `libreloc/include/reloc/CudaKernels.h` (create) | Public launch-wrapper declarations for all 8 kernels (`namespace reloc::cuda`), `#ifdef RELOC_ENABLE_CUDA` |
| `libreloc/cuda/CudaKernels.cu` (create) | `Axes` packing, all `__global__` kernels, launch wrappers |
| `libreloc/test/CudaKernelsTest.cpp` (create) | All GPU correctness tests (device helpers + 8 test groups) |
| `libreloc/CMakeLists.txt` (modify) | Add `cuda/CudaKernels.cu` to the CUDA block; set `CUDA_ARCHITECTURES "75;89"` |
| `libreloc/test/CMakeLists.txt` (modify) | Add `CudaKernelsTest.cpp` to `libreloc-test` |
| `libreloc/README.md` (modify) | Surface entry for `reloc::cuda` (final task) |

---

### Task 1: Scaffolding + `copy_f32` + multi-arch build

**Files:**
- Create: `libreloc/include/reloc/CudaKernels.h`
- Create: `libreloc/cuda/CudaKernels.cu`
- Create: `libreloc/test/CudaKernelsTest.cpp`
- Modify: `libreloc/CMakeLists.txt`, `libreloc/test/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing new (CUDA toolchain, existing CMake CUDA block).
- Produces:
  - `void reloc::cuda::copyF32(const float *dSrc, float *dDst, int64_t count, void *stream = nullptr)` — device→device copy, the JustCopy/EXP-4 ceiling kernel. Pointers from `cudaMalloc` (16-byte alignment assumed for the vectorized body).
  - Test-side device helpers `uploadFloats` / `downloadFloats` / `DeviceBuffer` reused by every later task.

- [ ] **Step 1: Write the failing test**

Create `libreloc/test/CudaKernelsTest.cpp`:

```cpp
//===- CudaKernelsTest.cpp - R0.2 GPU kernel correctness (local only) -----===//
//
// Compiled and run only under RELOC_ENABLE_CUDA on a machine with a GPU;
// never in CI (the CudaPipelineTest convention). Every kernel is checked
// against its CPU reference: relocate vs executeH2D byte-exact, quantize vs
// reloc::quant scalar bit-exact, dequant/unpack exact by construction.
//
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Bind.h"
#include "reloc/CudaKernels.h"
#include "reloc/Execute.h"
#include "reloc/Quant.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

// RAII device buffer; ASSERT-friendly (constructor cannot assert, so
// callers check valid()).
struct DeviceBuffer {
  void *p = nullptr;
  explicit DeviceBuffer(size_t bytes) { cudaMalloc(&p, bytes); }
  ~DeviceBuffer() { cudaFree(p); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  bool valid() const { return p != nullptr; }
  template <typename T> T *as() const { return static_cast<T *>(p); }
};

template <typename T>
void upload(const DeviceBuffer &d, const std::vector<T> &h) {
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d.p, h.data(), h.size() * sizeof(T),
                                    cudaMemcpyHostToDevice));
}

template <typename T> std::vector<T> download(const DeviceBuffer &d, size_t n) {
  std::vector<T> h(n);
  EXPECT_EQ(cudaSuccess,
            cudaMemcpy(h.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

std::vector<float> randomFloats(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float &x : v)
    x = dist(rng);
  return v;
}

TEST(CudaCopy, RoundTripOddCount) {
  const int64_t n = (1 << 20) + 13; // odd tail: exercises the non-vector path
  std::vector<float> src = randomFloats(static_cast<size_t>(n), 1, -1e6f, 1e6f);
  DeviceBuffer dSrc(n * sizeof(float)), dDst(n * sizeof(float));
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  ASSERT_EQ(cudaSuccess, cudaMemset(dDst.p, 0xAB, n * sizeof(float)));
  reloc::cuda::copyF32(dSrc.as<float>(), dDst.as<float>(), n);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> out = download<float>(dDst, static_cast<size_t>(n));
  ASSERT_EQ(0, std::memcmp(src.data(), out.data(), n * sizeof(float)));
}

} // namespace

#endif // RELOC_ENABLE_CUDA
```

- [ ] **Step 2: Create the header**

Create `libreloc/include/reloc/CudaKernels.h`:

```cpp
//===- CudaKernels.h - R0.2 GPU transform kernels ---------------*- C++ -*-===//
//
// Host-callable launch wrappers for issue #75's kernel set (Method-B
// transforms, Method-A receive paths, and the EXP-4 calibration kernels).
// Streams are type-erased to void* (nullptr = default stream) so this
// header pulls in no CUDA headers — the CudaBackend.h convention. All
// launches are asynchronous; the caller synchronizes. All device pointers
// come from cudaMalloc (16-byte aligned). Compiled only under
// RELOC_ENABLE_CUDA, for sm_75 (Turing) and sm_89 (Ada).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_CUDAKERNELS_H
#define RELOC_CUDAKERNELS_H

#ifdef RELOC_ENABLE_CUDA

#include "reloc/Bind.h"

#include <cstdint>

namespace reloc {
namespace cuda {

/// `copy_f32`: device->device element copy, the JustCopy bandwidth ceiling
/// (EXP-4 calibration, Colfax-comparable). Vectorized float4 body + scalar
/// tail.
void copyF32(const float *dSrc, float *dDst, int64_t count,
             void *stream = nullptr);

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
#endif // RELOC_CUDAKERNELS_H
```

- [ ] **Step 3: Wire CMake and observe the link failure**

In `libreloc/CMakeLists.txt`, inside the existing `if(RELOC_ENABLE_CUDA)` block, extend it to:

```cmake
if(RELOC_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(reloc_runtime PUBLIC RELOC_ENABLE_CUDA=1)
  target_sources(reloc_runtime PRIVATE
    cuda/CudaBackend.cu
    cuda/CudaKernels.cu
  )
  set_source_files_properties(cuda/CudaBackend.cu PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cuda/CudaKernels.cu PROPERTIES LANGUAGE CUDA)
  # R0.2 (issue #75): fatbin for Turing (the 2080 Ti box, M0) and Ada (the
  # dev box). sm_75 is compile-proof until M0 lands.
  set_target_properties(reloc_runtime PROPERTIES CUDA_ARCHITECTURES "75;89")
  target_link_libraries(reloc_runtime PRIVATE CUDA::cudart)
endif()
```

In `libreloc/test/CMakeLists.txt`, add `CudaKernelsTest.cpp` to the `add_executable(libreloc-test ...)` list (after `CudaPipelineTest.cpp`).

Create `libreloc/cuda/CudaKernels.cu` as a stub (file header only, no `copyF32` yet):

```cpp
//===- CudaKernels.cu - R0.2 GPU transform kernels ------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaKernels.h"

#include <cuda_runtime.h>

namespace reloc {
namespace cuda {

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
```

Build: `ninja -C build/cuda libreloc-test`
Expected: FAIL — **undefined reference to `reloc::cuda::copyF32`** (the RED signal).

- [ ] **Step 4: Implement `copyF32`**

Fill `libreloc/cuda/CudaKernels.cu`:

```cpp
//===- CudaKernels.cu - R0.2 GPU transform kernels ------------------------===//
//
// Issue #75's kernel set. Launch geometry: 256-thread blocks, grid-stride
// where the body is trivially divisible. Kernels are correctness-first
// (R0.2); bandwidth work belongs to the R0.3 harness (issue #76).
//
//===----------------------------------------------------------------------===//

#ifdef RELOC_ENABLE_CUDA

#include "reloc/CudaKernels.h"

#include <cassert>
#include <cuda_runtime.h>

namespace reloc {
namespace cuda {
namespace {

constexpr int kThreads = 256;

cudaStream_t asStream(void *p) { return static_cast<cudaStream_t>(p); }

int64_t gridFor(int64_t work) { return (work + kThreads - 1) / kThreads; }

__global__ void copyF32Vec4Kernel(const float4 *src, float4 *dst,
                                  int64_t count4) {
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < count4; i += stride)
    dst[i] = src[i];
}

__global__ void copyF32TailKernel(const float *src, float *dst, int64_t begin,
                                  int64_t count) {
  int64_t i = begin + threadIdx.x;
  if (i < count)
    dst[i] = src[i];
}

} // namespace

void copyF32(const float *dSrc, float *dDst, int64_t count, void *stream) {
  const int64_t count4 = count / 4;
  if (count4 > 0) {
    int64_t blocks = std::min<int64_t>(gridFor(count4), 65535);
    copyF32Vec4Kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                        asStream(stream)>>>(
        reinterpret_cast<const float4 *>(dSrc),
        reinterpret_cast<float4 *>(dDst), count4);
  }
  if (count % 4 != 0)
    copyF32TailKernel<<<1, 4, 0, asStream(stream)>>>(dSrc, dDst, count4 * 4,
                                                     count);
}

} // namespace cuda
} // namespace reloc

#endif // RELOC_ENABLE_CUDA
```

Add `#include <algorithm>` to the include block (for `std::min`).

- [ ] **Step 5: Build, run, verify both ELF archs**

```bash
ninja -C build/cuda libreloc-test
build/cuda/libreloc/test/libreloc-test --gtest_filter='CudaCopy.*'
cuobjdump --list-elf build/cuda/libreloc/libreloc_runtime.so | grep -oE 'sm_[0-9]+' | sort -u
```

Expected: test PASS; cuobjdump prints exactly `sm_75` and `sm_89`.

- [ ] **Step 6: clang-format + commit**

```bash
CF=/home/jueonpark/sym/.venv/bin/clang-format
for f in libreloc/include/reloc/CudaKernels.h libreloc/cuda/CudaKernels.cu \
         libreloc/test/CudaKernelsTest.cpp; do "$CF" -i "$f"; \
         "$CF" --dry-run --Werror "$f"; done
git add libreloc/include/reloc/CudaKernels.h libreloc/cuda/CudaKernels.cu \
        libreloc/test/CudaKernelsTest.cpp libreloc/CMakeLists.txt \
        libreloc/test/CMakeLists.txt
git commit -m "update(libreloc): R0.2 CUDA kernel scaffolding + copy_f32, sm_75+sm_89 (#75)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `relocate_naive_f32`

**Files:** Modify: `libreloc/include/reloc/CudaKernels.h`, `libreloc/cuda/CudaKernels.cu`, `libreloc/test/CudaKernelsTest.cpp`

**Interfaces:**
- Consumes: `BoundPlan` (`extents`/`srcStrides`/`dstStrides`/`elementSize`/`padRegions`), test helpers from Task 1.
- Produces:
  - `void reloc::cuda::relocateNaiveF32(const BoundPlan &bound, const float *dSrc, float *dDst, void *stream = nullptr)`
  - Internal `Axes` struct + `packAxes(const BoundPlan &)` + `relocateNaiveKernel` — reused by Tasks 3 and 6.
  - Test helpers `transposePlan(rows, cols)` and `maxSrcOffset(bound)` (same shapes as QuantTest.cpp's) — reused by Tasks 3 and 6.

- [ ] **Step 1: Write the failing tests**

Append inside the anonymous namespace of `CudaKernelsTest.cpp`:

```cpp
reloc::BoundPlan transposePlan(int64_t rows, int64_t cols) {
  reloc::BoundPlan b;
  b.extents = {rows, cols};
  b.srcStrides = {1, rows};
  b.dstStrides = {cols, 1};
  b.elementSize = 4;
  b.totalBytes = rows * cols * 4;
  return b;
}

int64_t maxSrcOffset(const reloc::BoundPlan &b) {
  int64_t off = 0;
  for (size_t k = 0; k < b.extents.size(); ++k)
    off += (b.extents[k] - 1) * b.srcStrides[k];
  return off;
}

// CPU oracle: executeH2D over the same plan, on the same host data.
std::vector<float> cpuRelocate(const reloc::BoundPlan &b,
                               const std::vector<float> &src) {
  std::vector<float> dst(static_cast<size_t>(b.totalBytes / 4), 0.0f);
  reloc::executeH2D(b, src.data(), dst.data());
  return dst;
}

void expectRelocateMatchesCpu(const reloc::BoundPlan &b, uint32_t seed,
                              bool tiled) {
  std::vector<float> src = randomFloats(
      static_cast<size_t>(maxSrcOffset(b) + 1), seed, -1e6f, 1e6f);
  std::vector<float> want = cpuRelocate(b, src);
  DeviceBuffer dSrc(src.size() * 4), dDst(want.size() * 4);
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  ASSERT_EQ(cudaSuccess, cudaMemset(dDst.p, 0xCD, want.size() * 4));
  if (tiled)
    reloc::cuda::relocateF32(b, dSrc.as<float>(), dDst.as<float>());
  else
    reloc::cuda::relocateNaiveF32(b, dSrc.as<float>(), dDst.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> got = download<float>(dDst, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * 4));
}

TEST(CudaRelocateNaive, TransposeMatchesCpu) {
  expectRelocateMatchesCpu(transposePlan(129, 517), 3, /*tiled=*/false);
  expectRelocateMatchesCpu(transposePlan(512, 512), 5, /*tiled=*/false);
}

TEST(CudaRelocateNaive, Rank3MatchesCpu) {
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100};
  b.dstStrides = {198, 33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  expectRelocateMatchesCpu(b, 7, /*tiled=*/false);
}
```

Note: `expectRelocateMatchesCpu` references `relocateF32` (Task 3). For THIS task, declare only `relocateNaiveF32` in the header and add a forward declaration comment — the compile will fail on `relocateF32`; to keep the task self-contained, add BOTH declarations to the header now (Task 3 implements the second), but only define `relocateNaiveF32` in the `.cu`. The link error on `relocateF32` will not trigger because no test calls it yet with `tiled=true` — however the reference in `expectRelocateMatchesCpu` still needs the symbol at link time IF the compiler cannot elide it: it cannot. Therefore in THIS task give `relocateF32` a temporary definition that simply forwards to `relocateNaiveF32`, marked with a `// Task 3 replaces this forward` comment. Task 3 replaces it with the real tiled dispatch. (This keeps every task independently green.)

- [ ] **Step 2: Run to verify failure** — `ninja -C build/cuda libreloc-test` → undefined references (RED).

- [ ] **Step 3: Implement**

Header (append after `copyF32`):

```cpp
/// `relocate_naive_f32`: GMEM->GMEM strided permute over the BoundPlan's
/// stride sets, one thread per valid element (the hiding-ratio floor for
/// EXP-4). Preconditions (asserted host-side): elementSize == 4, no pads,
/// 1 <= rank <= 8.
void relocateNaiveF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                      void *stream = nullptr);

/// `relocate_f32`: Method B's transform. SMEM-tiled + padded 32x32
/// transpose when the coalesced plan is 2-D-transpose-shaped; falls back
/// to the naive kernel otherwise. Output bit-identical to
/// relocateNaiveF32 either way. Same preconditions.
void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream = nullptr);
```

`.cu` (inside the anonymous namespace):

```cpp
constexpr int kMaxRank = 8;

struct Axes {
  int64_t ext[kMaxRank];
  int64_t srcStride[kMaxRank];
  int64_t dstStride[kMaxRank];
  int rank;
};

Axes packAxes(const reloc::BoundPlan &b) {
  assert(b.elementSize == 4 && "fp32 kernels only");
  assert(b.padRegions.empty() && "pads unsupported in R0.2 kernels");
  assert(!b.extents.empty() &&
         b.extents.size() <= static_cast<size_t>(kMaxRank) &&
         "rank out of kernel range");
  Axes a;
  a.rank = static_cast<int>(b.extents.size());
  for (int k = 0; k < a.rank; ++k) {
    a.ext[k] = b.extents[k];
    a.srcStride[k] = b.srcStrides[k];
    a.dstStride[k] = b.dstStrides[k];
  }
  return a;
}

int64_t totalElements(const reloc::BoundPlan &b) {
  int64_t total = 1;
  for (int64_t e : b.extents)
    total *= e;
  return total;
}

// One thread per valid element: decompose the linear index over the
// extents (row-major), gather via srcStrides, scatter via dstStrides —
// the bench/poc_transpose.cu baseline kernel, now a library citizen.
__global__ void relocateNaiveKernel(const float *src, float *dst, Axes a,
                                    int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  int64_t rem = i, srcOff = 0, dstOff = 0;
  for (int k = a.rank - 1; k >= 0; --k) {
    int64_t c = rem % a.ext[k];
    rem /= a.ext[k];
    srcOff += c * a.srcStride[k];
    dstOff += c * a.dstStride[k];
  }
  dst[dstOff] = src[srcOff];
}
```

and the public wrappers (in `namespace cuda`, after `copyF32`):

```cpp
void relocateNaiveF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                      void *stream) {
  Axes a = packAxes(bound);
  int64_t total = totalElements(bound);
  relocateNaiveKernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                        asStream(stream)>>>(dSrc, dDst, a, total);
}

// Task 3 replaces this forward with the tiled dispatch.
void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream) {
  relocateNaiveF32(bound, dSrc, dDst, stream);
}
```

- [ ] **Step 4: Build + run** — `--gtest_filter='CudaRelocateNaive.*'`: 2 tests PASS.

- [ ] **Step 5: clang-format (all three touched files, same commands as Task 1) + commit**

```bash
git add -u
git commit -m "update(libreloc): R0.2 relocate_naive_f32 plan-driven GPU kernel (#75)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `relocate_f32` — SMEM-tiled + padded 32×32 transpose

**Files:** Modify: `libreloc/cuda/CudaKernels.cu`, `libreloc/test/CudaKernelsTest.cpp` (header already declares `relocateF32`)

**Interfaces:**
- Consumes: `Axes`/`packAxes`/`relocateNaiveKernel`, `expectRelocateMatchesCpu`.
- Produces: the real `relocateF32` dispatch — tiled kernel for 2-D-transpose-shaped plans, naive fallback otherwise. Bit-identical to naive in both paths.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(CudaRelocateTiled, TransposeShapesMatchCpu) {
  // multiples of 32, remainder tiles both axes, tiny, tall/wide
  expectRelocateMatchesCpu(transposePlan(1024, 1024), 11, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(129, 517), 13, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(32, 32), 17, /*tiled=*/true);
  expectRelocateMatchesCpu(transposePlan(1, 4096), 19, /*tiled=*/true);
}

TEST(CudaRelocateTiled, NonTransposePlanFallsBackBitExact) {
  // 3-D plan: not 2-D-transpose-shaped -> must take the naive fallback and
  // still match the CPU oracle.
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100};
  b.dstStrides = {198, 33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  expectRelocateMatchesCpu(b, 23, /*tiled=*/true);
}

TEST(CudaRelocateTiled, TiledIdenticalToNaive) {
  auto b = transposePlan(801, 333); // remainder tiles, non-square
  std::vector<float> src = randomFloats(
      static_cast<size_t>(maxSrcOffset(b) + 1), 29, -1e6f, 1e6f);
  DeviceBuffer dSrc(src.size() * 4), dA(static_cast<size_t>(b.totalBytes)),
      dB(static_cast<size_t>(b.totalBytes));
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dA.valid());
  ASSERT_TRUE(dB.valid());
  upload(dSrc, src);
  reloc::cuda::relocateF32(b, dSrc.as<float>(), dA.as<float>());
  reloc::cuda::relocateNaiveF32(b, dSrc.as<float>(), dB.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  size_t n = static_cast<size_t>(b.totalBytes / 4);
  std::vector<float> a = download<float>(dA, n), c = download<float>(dB, n);
  ASSERT_EQ(0, std::memcmp(a.data(), c.data(), n * 4));
}
```

Note the current forward (Task 2) makes these PASS vacuously except that the tiled path doesn't exist — the RED signal for this task is behavioral only after implementation, so instead: implement first the shape test `isTranspose2D` returning true and dispatching to a `transposeTiledKernel` that is NOT yet written → compile error is the RED. Follow the brief order: write tests (they pass against the forward — note that in the report), then replace the forward with the dispatch calling the not-yet-written kernel (compile RED), then implement the kernel (GREEN).

- [ ] **Step 2: Implement**

`.cu` anonymous namespace:

```cpp
constexpr int kTile = 32;
constexpr int kBlockRows = 8;

// Coalesced 2-D transpose: read a 32x32 tile of `in` coalesced, write it
// back transposed and coalesced; +1 padding kills SMEM bank conflicts.
// `in` is inRows x inCols row-major; out[c][r] = in[r][c].
__global__ void transposeTiledKernel(const float *in, float *out,
                                     int64_t inRows, int64_t inCols) {
  __shared__ float tile[kTile][kTile + 1];
  int64_t x = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.x;
  int64_t y = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.y;
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inCols && y + j < inRows)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * inCols + x];
  __syncthreads();
  x = blockIdx.y * static_cast<int64_t>(kTile) + threadIdx.x; // out col
  y = blockIdx.x * static_cast<int64_t>(kTile) + threadIdx.y; // out row
  for (int j = 0; j < kTile; j += kBlockRows)
    if (x < inRows && y + j < inCols)
      out[(y + j) * inRows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// relocate_f32's fast path applies when the coalesced plan is exactly a
// 2-D transpose: dst [R,C] dense row-major, src read column-major.
bool isTranspose2D(const reloc::BoundPlan &b) {
  return b.extents.size() == 2 && b.srcStrides[0] == 1 &&
         b.srcStrides[1] == b.extents[0] && b.dstStrides[1] == 1 &&
         b.dstStrides[0] == b.extents[1];
}
```

Replace the Task-2 forward:

```cpp
void relocateF32(const BoundPlan &bound, const float *dSrc, float *dDst,
                 void *stream) {
  if (!isTranspose2D(bound)) {
    relocateNaiveF32(bound, dSrc, dDst, stream);
    return;
  }
  // Plan dst[r][c] = src[c*R + r]: view src as (C x R) row-major `in`,
  // dst as out with out[r][c] = in[c][r] -> launch with inRows=C, inCols=R.
  const int64_t R = bound.extents[0], C = bound.extents[1];
  dim3 block(kTile, kBlockRows);
  dim3 grid(static_cast<unsigned>((R + kTile - 1) / kTile),
            static_cast<unsigned>((C + kTile - 1) / kTile));
  transposeTiledKernel<<<grid, block, 0, asStream(stream)>>>(dSrc, dDst,
                                                             /*inRows=*/C,
                                                             /*inCols=*/R);
}
```

Derivation note for the implementer (do not skip): with `in = src` viewed as C×R row-major, `in[c][r] = src[c*R + r]`, and the kernel writes `out[(y+j)*inRows + x] = out[c*R + r]`… the kernel's `out` is inCols×inRows = R×C **column-of-in-major**; work through one element: kernel guarantees `out[a][b] = in[b][a]` for out dims (inCols=R rows? No —) **the kernel as written produces out as inCols×inRows row-major with out[c][r] = in[r][c]** where its `in` is inRows×inCols. Setting kernel-in = src-as-CxR (inRows=C, inCols=R): kernel-out is R×C row-major with out[r][c] = in[c][r] = src[c*R + r] = plan's dst[r][c]. ✓ The `TiledIdenticalToNaive` test is the executable proof — if the mapping is transposed the memcmp fails immediately.

- [ ] **Step 3: Build + run** — `--gtest_filter='CudaRelocate*'`: all PASS (naive 2 + tiled 3).

- [ ] **Step 4: clang-format touched files + commit**

```bash
git add -u
git commit -m "update(libreloc): R0.2 relocate_f32 SMEM-tiled 32x32 transpose + fallback (#75)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `quantize_f32_s8` (GPU) — bit-exact vs CPU

**Files:** Modify: `libreloc/include/reloc/CudaKernels.h`, `libreloc/cuda/CudaKernels.cu`, `libreloc/test/CudaKernelsTest.cpp`

**Interfaces:**
- Consumes: CPU oracle `reloc::quant::quantizePackF32S8(..., Variant::Scalar)` (`reloc/Quant.h`).
- Produces: `void reloc::cuda::quantizeF32S8(const float *dSrc, int8_t *dDst, int64_t channels, int64_t channelSize, const float *dInvScales, void *stream = nullptr)` — `dInvScales` is a DEVICE array of `channels` floats.

- [ ] **Step 1: Failing test**

```cpp
TEST(CudaQuantize, BitExactVsCpuScalar) {
  const int64_t channels = 5, chSize = 1031;
  std::vector<float> src =
      randomFloats(static_cast<size_t>(channels * chSize), 31, -300.f, 300.f);
  // Poke the clamp/NaN/tie lanes.
  src[0] = NAN;
  src[1] = HUGE_VALF;
  src[2] = -HUGE_VALF;
  src[3] = 0.5f;
  src[4] = 2.5f;
  src[5] = -2.5f;
  src[6] = 200.0f;
  src[7] = -200.0f;
  std::vector<float> inv(channels);
  for (int64_t c = 0; c < channels; ++c)
    inv[c] = 0.05f + 0.9f * static_cast<float>(c);
  std::vector<int8_t> want(static_cast<size_t>(channels * chSize), 0);
  reloc::quant::quantizePackF32S8(src.data(), want.data(), channels, chSize,
                                  inv.data(), reloc::quant::Variant::Scalar);
  DeviceBuffer dSrc(src.size() * 4), dInv(inv.size() * 4), dDst(want.size());
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dInv.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  upload(dInv, inv);
  reloc::cuda::quantizeF32S8(dSrc.as<float>(), dDst.as<int8_t>(), channels,
                             chSize, dInv.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<int8_t> got = download<int8_t>(dDst, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size()));
}
```

- [ ] **Step 2: Implement**

Header:

```cpp
/// `quantize_f32_s8` (GPU side, Method B for EXP-2's quantize workloads):
/// contiguous per-channel int8 quantize, BIT-IDENTICAL to the CPU scalar
/// contract (fmaxf-then-fminf clamp so NaN -> -128; __float2int_rn = RNE).
/// dInvScales: device array of `channels` floats.
void quantizeF32S8(const float *dSrc, int8_t *dDst, int64_t channels,
                   int64_t channelSize, const float *dInvScales,
                   void *stream = nullptr);
```

`.cu` kernel + wrapper:

```cpp
__global__ void quantizeF32S8Kernel(const float *src, int8_t *dst,
                                    int64_t channelSize,
                                    const float *invScales, int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  float y = src[i] * invScales[i / channelSize];
  y = fmaxf(y, -128.0f); // NaN -> -128, matching the CPU quantOne contract
  y = fminf(y, 127.0f);
  dst[i] = static_cast<int8_t>(__float2int_rn(y));
}
```

```cpp
void quantizeF32S8(const float *dSrc, int8_t *dDst, int64_t channels,
                   int64_t channelSize, const float *dInvScales,
                   void *stream) {
  int64_t total = channels * channelSize;
  quantizeF32S8Kernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                        asStream(stream)>>>(dSrc, dDst, channelSize,
                                            dInvScales, total);
}
```

- [ ] **Step 3: Build + run** — `--gtest_filter='CudaQuantize.*'` PASS. If the memcmp fails, diff the first mismatching index and its input float on the host before touching kernel code — the contract requires investigation, not tweaking.

- [ ] **Step 4: clang-format + commit** — `update(libreloc): R0.2 GPU quantize_f32_s8 bit-exact vs CPU (#75)` + trailer.

---

### Task 5: `dequant_s8_f32` + `unpack_s4_s8`

**Files:** Modify: header, `.cu`, test file (same three).

**Interfaces:**
- Consumes: CPU `reloc::quant::packS8S4` as the inverse-pair oracle.
- Produces:
  - `void reloc::cuda::dequantS8F32(const int8_t *dSrc, float *dDst, int64_t channels, int64_t channelSize, const float *dScales, void *stream = nullptr)` — `out[i] = (float)in[i] * scales[i / channelSize]` (exact fp32 arithmetic).
  - `void reloc::cuda::unpackS4S8(const uint8_t *dSrc, int8_t *dDst, int64_t pairs, void *stream = nullptr)` — byte i → `dDst[2i]` (sign-extended low nibble), `dDst[2i+1]` (high nibble); exact inverse of `packS8S4` on `[-8,7]`.

- [ ] **Step 1: Failing tests**

```cpp
TEST(CudaDequant, ExactVsHostReference) {
  const int64_t channels = 7, chSize = 517;
  std::mt19937 rng(37);
  std::vector<int8_t> src(static_cast<size_t>(channels * chSize));
  for (int8_t &v : src)
    v = static_cast<int8_t>(rng());
  std::vector<float> scales(channels);
  for (int64_t c = 0; c < channels; ++c)
    scales[c] = 0.013f * static_cast<float>(c + 1);
  std::vector<float> want(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    want[i] = static_cast<float>(src[i]) *
              scales[static_cast<int64_t>(i) / chSize];
  DeviceBuffer dSrc(src.size()), dScales(scales.size() * 4),
      dDst(want.size() * 4);
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dScales.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  upload(dScales, scales);
  reloc::cuda::dequantS8F32(dSrc.as<int8_t>(), dDst.as<float>(), channels,
                            chSize, dScales.as<float>());
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> got = download<float>(dDst, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * 4));
}

TEST(CudaUnpack, InverseOfCpuPack) {
  const int64_t pairs = 100003;
  std::mt19937 rng(41);
  std::vector<int8_t> orig(static_cast<size_t>(2 * pairs));
  for (int8_t &v : orig)
    v = static_cast<int8_t>(rng()); // full range: saturation exercised
  std::vector<uint8_t> packed(static_cast<size_t>(pairs));
  reloc::quant::packS8S4(orig.data(), packed.data(), pairs);
  // Expected after round-trip: clamp(orig, -8, 7).
  std::vector<int8_t> want(orig.size());
  for (size_t i = 0; i < orig.size(); ++i)
    want[i] = static_cast<int8_t>(orig[i] < -8 ? -8
                                  : orig[i] > 7 ? 7
                                                : orig[i]);
  DeviceBuffer dPacked(packed.size()), dOut(want.size());
  ASSERT_TRUE(dPacked.valid());
  ASSERT_TRUE(dOut.valid());
  upload(dPacked, packed);
  reloc::cuda::unpackS4S8(dPacked.as<uint8_t>(), dOut.as<int8_t>(), pairs);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<int8_t> got = download<int8_t>(dOut, want.size());
  ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size()));
}
```

- [ ] **Step 2: Implement**

Header:

```cpp
/// `dequant_s8_f32` (Method A dtype-only receive path): contiguous
/// per-channel int8 -> fp32, out = (float)s8 * scale[channel]. Exact fp32
/// arithmetic. dScales: device array of `channels` floats.
void dequantS8F32(const int8_t *dSrc, float *dDst, int64_t channels,
                  int64_t channelSize, const float *dScales,
                  void *stream = nullptr);

/// `unpack_s4_s8` (Method A r=0.125 receive path): int4 nibble unpack,
/// exact inverse of the CPU packS8S4 on saturated values. Byte i ->
/// dDst[2i] = sign-extended low nibble, dDst[2i+1] = high nibble.
void unpackS4S8(const uint8_t *dSrc, int8_t *dDst, int64_t pairs,
                void *stream = nullptr);
```

`.cu`:

```cpp
__global__ void dequantS8F32Kernel(const int8_t *src, float *dst,
                                   int64_t channelSize, const float *scales,
                                   int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  dst[i] = static_cast<float>(src[i]) * scales[i / channelSize];
}

__global__ void unpackS4S8Kernel(const uint8_t *src, int8_t *dst,
                                 int64_t pairs) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= pairs)
    return;
  const uint8_t b = src[i];
  dst[2 * i] = static_cast<int8_t>(static_cast<uint8_t>(b << 4)) >> 4;
  dst[2 * i + 1] = static_cast<int8_t>(b) >> 4;
}
```

Wrappers follow the `quantizeF32S8` launch pattern (`total = channels * channelSize` and `pairs` respectively).

- [ ] **Step 3: Build + run** — `--gtest_filter='CudaDequant.*:CudaUnpack.*'` PASS.

- [ ] **Step 4: clang-format + commit** — `update(libreloc): R0.2 GPU dequant_s8_f32 + unpack_s4_s8 (#75)` + trailer.

---

### Task 6: `dequant_relocate_s8_f32`

**Files:** Modify: header, `.cu`, test file.

**Interfaces:**
- Consumes: `Axes`/`packAxes`/`totalElements`, `transposePlan`/`maxSrcOffset` test helpers.
- Produces: `void reloc::cuda::dequantRelocateS8F32(const BoundPlan &bound, const int8_t *dSrc, float *dDst, const float *dScales, void *stream = nullptr)` — Method A's fused receive path: int8 in the plan's SRC layout in, fp32 in the plan's DST layout out, `dScales` = device array of `extents[0]` per-outer-channel floats. Preconditions: `elementSize == 4` (fp32 OUT; plan offsets are element-indexed on both sides), no pads, `2 <= rank <= 8`.

- [ ] **Step 1: Failing test**

```cpp
// Host oracle: odometer over the full index space (independent of the
// kernel's index decomposition).
std::vector<float> cpuDequantRelocate(const reloc::BoundPlan &b,
                                      const std::vector<int8_t> &src,
                                      const std::vector<float> &scales) {
  const size_t r = b.extents.size();
  int64_t total = 1;
  for (int64_t e : b.extents)
    total *= e;
  std::vector<float> dst(static_cast<size_t>(total), 0.0f);
  std::vector<int64_t> idx(r, 0);
  while (true) {
    int64_t so = 0, dso = 0;
    for (size_t k = 0; k < r; ++k) {
      so += idx[k] * b.srcStrides[k];
      dso += idx[k] * b.dstStrides[k];
    }
    dst[dso] = static_cast<float>(src[so]) * scales[idx[0]];
    size_t k = r;
    for (;;) {
      if (k == 0)
        return dst;
      --k;
      if (++idx[k] < b.extents[k])
        break;
      idx[k] = 0;
    }
  }
}

TEST(CudaDequantRelocate, MatchesHostOracle) {
  for (auto b : {transposePlan(129, 517), transposePlan(64, 64)}) {
    std::mt19937 rng(43);
    std::vector<int8_t> src(static_cast<size_t>(maxSrcOffset(b) + 1));
    for (int8_t &v : src)
      v = static_cast<int8_t>(rng());
    std::vector<float> scales(static_cast<size_t>(b.extents[0]));
    for (size_t c = 0; c < scales.size(); ++c)
      scales[c] = 0.007f * static_cast<float>(c + 1);
    std::vector<float> want = cpuDequantRelocate(b, src, scales);
    DeviceBuffer dSrc(src.size()), dScales(scales.size() * 4),
        dDst(want.size() * 4);
    ASSERT_TRUE(dSrc.valid());
    ASSERT_TRUE(dScales.valid());
    ASSERT_TRUE(dDst.valid());
    upload(dSrc, src);
    upload(dScales, scales);
    ASSERT_EQ(cudaSuccess, cudaMemset(dDst.p, 0, want.size() * 4));
    reloc::cuda::dequantRelocateS8F32(b, dSrc.as<int8_t>(), dDst.as<float>(),
                                      dScales.as<float>());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::vector<float> got = download<float>(dDst, want.size());
    ASSERT_EQ(0, std::memcmp(want.data(), got.data(), want.size() * 4));
  }
}
```

- [ ] **Step 2: Implement**

Header:

```cpp
/// `dequant_relocate_s8_f32` (Method A's fused receive path; the R0 exit
/// test's PCIe-hiding candidate on Turing): int8 in the plan's SRC layout
/// -> fp32 in the plan's DST layout, scaled per coalesced outer channel
/// (dScales: device array of extents[0] floats). Preconditions (asserted
/// host-side): elementSize == 4, no pads, 2 <= rank <= 8.
void dequantRelocateS8F32(const BoundPlan &bound, const int8_t *dSrc,
                          float *dDst, const float *dScales,
                          void *stream = nullptr);
```

`.cu`:

```cpp
__global__ void dequantRelocateKernel(const int8_t *src, float *dst, Axes a,
                                      const float *scales, int64_t total) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= total)
    return;
  int64_t rem = i, srcOff = 0, dstOff = 0, c0 = 0;
  for (int k = a.rank - 1; k >= 0; --k) {
    int64_t c = rem % a.ext[k];
    rem /= a.ext[k];
    srcOff += c * a.srcStride[k];
    dstOff += c * a.dstStride[k];
    if (k == 0)
      c0 = c;
  }
  dst[dstOff] = static_cast<float>(src[srcOff]) * scales[c0];
}
```

```cpp
void dequantRelocateS8F32(const BoundPlan &bound, const int8_t *dSrc,
                          float *dDst, const float *dScales, void *stream) {
  assert(bound.extents.size() >= 2 &&
         "per-channel scale needs a distinct outer axis");
  Axes a = packAxes(bound);
  int64_t total = totalElements(bound);
  dequantRelocateKernel<<<static_cast<unsigned>(gridFor(total)), kThreads, 0,
                          asStream(stream)>>>(dSrc, dDst, a, dScales, total);
}
```

- [ ] **Step 3: Build + run** — `--gtest_filter='CudaDequantRelocate.*'` PASS.

- [ ] **Step 4: clang-format + commit** — `update(libreloc): R0.2 GPU dequant_relocate_s8_f32 fused receive path (#75)` + trailer.

---

### Task 7: `scatter_random_f32` + README + full verification

**Files:** Modify: header, `.cu`, test file, `libreloc/README.md`.

**Interfaces:**
- Consumes: everything above.
- Produces: `void reloc::cuda::scatterRandomF32(const float *dSrc, const int64_t *dIdx, float *dDst, int64_t count, void *stream = nullptr)` — `dDst[dIdx[i]] = dSrc[i]` (EXP-4's data-dependent pathological case; `dIdx` must be a permutation for a race-free deterministic result — caller's contract, documented).

- [ ] **Step 1: Failing test**

```cpp
TEST(CudaScatterRandom, PermutationRoundTrips) {
  const int64_t n = (1 << 20) + 7;
  std::vector<float> src = randomFloats(static_cast<size_t>(n), 47, -1e6f,
                                        1e6f);
  std::vector<int64_t> idx(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    idx[static_cast<size_t>(i)] = i;
  std::mt19937_64 rng(53);
  std::shuffle(idx.begin(), idx.end(), rng);
  DeviceBuffer dSrc(src.size() * 4), dIdx(idx.size() * 8),
      dDst(src.size() * 4);
  ASSERT_TRUE(dSrc.valid());
  ASSERT_TRUE(dIdx.valid());
  ASSERT_TRUE(dDst.valid());
  upload(dSrc, src);
  upload(dIdx, idx);
  reloc::cuda::scatterRandomF32(dSrc.as<float>(), dIdx.as<int64_t>(),
                                dDst.as<float>(), n);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  std::vector<float> got = download<float>(dDst, src.size());
  for (int64_t i = 0; i < n; ++i)
    ASSERT_EQ(got[static_cast<size_t>(idx[static_cast<size_t>(i)])],
              src[static_cast<size_t>(i)])
        << "i=" << i;
}
```

(Add `#include <algorithm>` to the test's include block for `std::shuffle`.)

- [ ] **Step 2: Implement**

Header:

```cpp
/// `scatter_random_f32` (EXP-4's pathological data-dependent case):
/// dDst[dIdx[i]] = dSrc[i]. dIdx: device array of `count` int64 indices;
/// must be a permutation of [0, count) for a deterministic, race-free
/// result (caller's contract).
void scatterRandomF32(const float *dSrc, const int64_t *dIdx, float *dDst,
                      int64_t count, void *stream = nullptr);
```

`.cu`:

```cpp
__global__ void scatterRandomKernel(const float *src, const int64_t *idx,
                                    float *dst, int64_t count) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= count)
    return;
  dst[idx[i]] = src[i];
}
```

Wrapper follows the standard launch pattern.

- [ ] **Step 3: README Surface entry**

Append to `libreloc/README.md`'s `## Surface` list, after the `reloc::quant` entry:

```markdown
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
```

- [ ] **Step 4: Full verification**

```bash
ninja -C build/cuda && ctest --test-dir build/cuda --output-on-failure
build/cuda/libreloc/test/libreloc-test   # full gtest incl. all Cuda* suites
ninja -C build/sym && ctest --test-dir build/sym --output-on-failure  # CUDA-off stays green
cuobjdump --list-elf build/cuda/libreloc/libreloc_runtime.so | grep -oE 'sm_[0-9]+' | sort -u
```

Expected: all pass in both trees; `sm_75` + `sm_89` both present.

- [ ] **Step 5: clang-format + commit** — `update(libreloc): R0.2 scatter_random_f32 + reloc::cuda surface docs (#75)` + trailer.

---

### Task 8: Final review + finish branch

- [ ] Full-suite verification fresh at HEAD (both build trees, per Task 7 Step 4).
- [ ] Dispatch the final whole-branch code review (most capable model), including accumulated Minor findings for triage; fix wave if needed.
- [ ] clang-format `--dry-run --Werror` over every touched `.cpp`/`.h` as a CI-parity gate before pushing.
- [ ] Use superpowers:verification-before-completion, then superpowers:finishing-a-development-branch — target: PR titled `update(libreloc): R0.2 GPU kernels for sm_75/sm_89 (#75)`, body with the correctness-test matrix (kernel × oracle), the sm_75/sm_89 cuobjdump proof, `Closes #75`, and a note that bandwidth measurement follows in R0.3 (#76). Draft (do not post) a #73 status comment.

---

## Self-Review (completed at plan time)

1. **Spec coverage vs issue #75:** all 7 table rows → 8 kernels: `relocate_f32` (Task 3, SMEM-tiled+padded 32×32), `relocate_naive_f32` (Task 2), `copy_f32` (Task 1), `dequant_relocate_s8_f32` (Task 6), `dequant_s8_f32` + `unpack_s4_s8` (Task 5), `quantize_f32_s8` (Task 4), `scatter_random_f32` (Task 7). sm_75+sm_89 → Task 1 CMake + cuobjdump verification. #76's R0 exit criterion ("bit-exact vs reference for relocate; bounded quant round-trip") → the relocate memcmp oracles and the bit-exact quantize test (stronger than a bound).
2. **Placeholder scan:** every code step carries complete code; Task 2's temporary `relocateF32` forward is explicit and replaced in Task 3.
3. **Type consistency:** `Axes`/`packAxes`/`totalElements` defined once (Task 2), consumed in Tasks 3/6; test helpers (`DeviceBuffer`, `upload`/`download`, `randomFloats`, `transposePlan`, `maxSrcOffset`, `expectRelocateMatchesCpu`) defined once and reused; all launch wrappers share the `void *stream = nullptr` convention.
