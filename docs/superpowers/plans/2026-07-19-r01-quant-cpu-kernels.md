# R0.1 CPU Quant Kernels (`libreloc/quant/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #74 — the R0.1 CPU transform kernels (`quantize_pack_f32_s8`, `gather_quantize_f32_s8`, `pack_s8_s4`, `convert_f32_f16`) with scalar + SIMD variants, GatherPool multi-thread wrappers, tests, and a bandwidth micro-benchmark — as the first slice of the #73 R-track campaign.

**Architecture:** New `reloc::quant` namespace inside the existing `reloc_runtime` shared library. Public API in `libreloc/include/reloc/Quant.h`; sources in `libreloc/quant/` (sibling of `src/`, mirroring how `cuda/` is laid out). SIMD variants live in per-ISA translation units compiled with per-file `-m` flags and selected at runtime via `__builtin_cpu_supports` — so one binary runs on the AVX-512 Zen4 box (7800X3D) and a possibly AVX2-only 2080 Ti host. The fused gather+quantize kernel consumes a `BoundPlan` exactly like the existing `gatherChunk` primitive (chunked outer-axis form), so the D-track `GatherPool` parallelizes it unchanged.

**Tech Stack:** C++17, x86 intrinsics (AVX2/FMA/F16C, AVX-512 F+BW), CMake + Ninja, googletest (`llvm_gtest` in the `libreloc-test` binary), `bench/protocol.h` measurement protocol.

## Global Constraints

- **MLIR-free contract:** nothing under `libreloc/src`, `libreloc/include`, `libreloc/quant` may include an `mlir/` or `llvm/` header. Enforced by the existing `reloc-runtime-no-mlir-includes` ctest — note it scans `../src ../include ../cuda ../python`; Task 1 must add `../quant` to that scan.
- **Naming:** repo code style is camelCase functions in `namespace reloc` (`gatherChunk`, `executeH2D`). The issue's snake_case kernel names (`quantize_pack_f32_s8`, …) are the *reporting* names, used verbatim only in the bench JSON and comments.
- **Quantization semantics (all variants must implement exactly this):** `q = (int8) rne(clamp(x * invScale, -128.0f, 127.0f))` where the multiply is fp32, clamp order is `max(y, -128.0f)` then `min(y, 127.0f)` (so NaN → −128, matching x86 `MAXPS(v, lo)` NaN rules), and `rne` is round-to-nearest-even (default FP environment; `nearbyintf` scalar ≡ `cvtps2dq` vector).
- **fp16 semantics:** IEEE 754 binary16, round-to-nearest-even, subnormals preserved, overflow (≥ 65520) → ±inf, NaN → some quiet half NaN. Scalar must be bit-identical to F16C/`VCVTPS2PH` hardware output for all non-NaN inputs.
- **int4 pack semantics:** each int8 saturated to `[-8, 7]`, then low nibble = even input index, high nibble = odd input index. API counts output bytes (`pairs`); input length is `2 * pairs`.
- **Bit-identical variants:** every SIMD variant must produce byte-identical output to the scalar variant for the same input, including remainder tails. Tests enforce this with `memcmp`.
- **Runtime-skip on unsupported hosts:** tests and bench must never execute an unsupported ISA path; tests use `GTEST_SKIP()` / silent skip loops, kernels `assert` on explicit unsupported variants.
- **Build dirs:** dev/test cycle uses `/home/jueonpark/sym/build/sym` (Ninja, CUDA OFF, asserts on). Real bandwidth numbers only from `/home/jueonpark/sym/build/cuda-release` (Release). Compiler is GCC 11.4 — supports all flags used here.
- **Test binary:** `/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test`; run `ninja -C /home/jueonpark/sym/build/sym libreloc-test` to build it.
- **Commit style (from git log):** `update(libreloc): <what> (#74)` / `update(bench): … (#74)`, ending with the Claude co-author trailer.
- **Scope guard:** this plan is issue #74 only. The pipeline/PCIe measurement harness (R0.3), GPU kernels (R0.2), and workload matrix (EXP-1) are separate follow-up plans under #73.

## File Structure

| File | Responsibility |
|---|---|
| `libreloc/include/reloc/Quant.h` (create) | Public API: `Variant`/`Kernel` enums, `cpuSupports`/`kernelHasVariant`/`resolveFor`, the four kernels, four `*Parallel` wrappers |
| `libreloc/quant/QuantKernels.h` (create) | Internal per-variant function declarations + the shared inline `quantOne` scalar helper |
| `libreloc/quant/Quant.cpp` (create) | CPU dispatch, scalar variants, `f32ToF16Scalar`, plan walk for the fused kernel, parallel wrappers |
| `libreloc/quant/QuantAVX2.cpp` (create) | AVX2 quantize + F16C convert; compiled with `-mavx2 -mfma -mf16c` |
| `libreloc/quant/QuantAVX512.cpp` (create) | AVX-512 quantize / gather+quantize (plain + prefetch) / int4 pack / convert; compiled with `-mavx512f -mavx512bw` |
| `libreloc/test/QuantTest.cpp` (create) | All kernel unit tests (added to `libreloc-test`) |
| `libreloc/CMakeLists.txt` (modify) | Add quant sources, per-TU SIMD flags, `RELOC_QUANT_HAVE_X86_SIMD` define |
| `libreloc/test/CMakeLists.txt` (modify) | Add `QuantTest.cpp`; extend the no-mlir-includes grep to `../quant` |
| `bench/quant_bw.cpp` (create) | Per-kernel × variant × thread-count bandwidth driver on `bench/protocol.h` |
| `bench/CMakeLists.txt` (modify) | Add `bench-quant-bw` target + smoke ctest |
| `libreloc/README.md` (modify) | Surface entry for `reloc::quant` |

---

### Task 1: Variant dispatch scaffolding

Create the public header (dispatch surface only), the core TU with `cpuSupports`/`kernelHasVariant`/`resolveFor`, CMake wiring, and the extended MLIR-free source scan.

**Files:**
- Create: `libreloc/include/reloc/Quant.h`
- Create: `libreloc/quant/Quant.cpp`
- Create: `libreloc/test/QuantTest.cpp`
- Modify: `libreloc/CMakeLists.txt` (add `quant/Quant.cpp` to `add_library`)
- Modify: `libreloc/test/CMakeLists.txt` (add `QuantTest.cpp`; add `../quant` to the include-scan test)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by every later task):
  - `enum class reloc::quant::Variant { Auto, Scalar, AVX2, AVX512, AVX512Pf }`
  - `enum class reloc::quant::Kernel { QuantizePack, GatherQuantize, PackS8S4, ConvertF32F16 }`
  - `bool reloc::quant::cpuSupports(Variant)`
  - `bool reloc::quant::kernelHasVariant(Kernel, Variant)`
  - `Variant reloc::quant::resolveFor(Kernel, Variant)`

- [ ] **Step 1: Write the failing test**

Create `libreloc/test/QuantTest.cpp`:

```cpp
//===- QuantTest.cpp - R0.1 CPU quant kernels (issue #74) -----------------===//

#include "reloc/Quant.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

using reloc::quant::Kernel;
using reloc::quant::Variant;

TEST(QuantDispatch, ScalarAndAutoAlwaysSupported) {
  EXPECT_TRUE(reloc::quant::cpuSupports(Variant::Scalar));
  EXPECT_TRUE(reloc::quant::cpuSupports(Variant::Auto));
}

TEST(QuantDispatch, KernelVariantTableMatchesIssue74) {
  // quantize_pack: scalar / AVX2 / AVX-512
  EXPECT_TRUE(kernelHasVariant(Kernel::QuantizePack, Variant::AVX2));
  EXPECT_TRUE(kernelHasVariant(Kernel::QuantizePack, Variant::AVX512));
  EXPECT_FALSE(kernelHasVariant(Kernel::QuantizePack, Variant::AVX512Pf));
  // gather_quantize: scalar / AVX-512 gather / prefetch+tiled
  EXPECT_FALSE(kernelHasVariant(Kernel::GatherQuantize, Variant::AVX2));
  EXPECT_TRUE(kernelHasVariant(Kernel::GatherQuantize, Variant::AVX512));
  EXPECT_TRUE(kernelHasVariant(Kernel::GatherQuantize, Variant::AVX512Pf));
  // pack_s8_s4: AVX-512 (+ scalar reference)
  EXPECT_FALSE(kernelHasVariant(Kernel::PackS8S4, Variant::AVX2));
  EXPECT_TRUE(kernelHasVariant(Kernel::PackS8S4, Variant::AVX512));
  // convert_f32_f16: F16C (the AVX2 tier) / AVX-512
  EXPECT_TRUE(kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX2));
  EXPECT_TRUE(kernelHasVariant(Kernel::ConvertF32F16, Variant::AVX512));
}

TEST(QuantDispatch, ResolveAutoPicksAnImplementedSupportedTier) {
  for (Kernel k : {Kernel::QuantizePack, Kernel::GatherQuantize,
                   Kernel::PackS8S4, Kernel::ConvertF32F16}) {
    Variant r = reloc::quant::resolveFor(k, Variant::Auto);
    EXPECT_NE(r, Variant::Auto);
    EXPECT_NE(r, Variant::AVX512Pf) << "prefetch tier is opt-in only";
    EXPECT_TRUE(reloc::quant::cpuSupports(r));
    EXPECT_TRUE(kernelHasVariant(k, r));
  }
}

TEST(QuantDispatch, ResolveExplicitIsIdentity) {
  EXPECT_EQ(reloc::quant::resolveFor(Kernel::QuantizePack, Variant::Scalar),
            Variant::Scalar);
  if (reloc::quant::cpuSupports(Variant::AVX512)) {
    EXPECT_EQ(
        reloc::quant::resolveFor(Kernel::GatherQuantize, Variant::AVX512Pf),
        Variant::AVX512Pf);
  }
}

} // namespace
```

Note: `kernelHasVariant` is found via ADL-free qualified use inside `namespace {}` — call it as `reloc::quant::kernelHasVariant(...)`; the test above relies on the `using` declarations only for the enum names. Write the calls fully qualified if the compiler complains.

- [ ] **Step 2: Create the header so the test compiles against the intended API**

Create `libreloc/include/reloc/Quant.h`:

```cpp
//===- Quant.h - R0.1 CPU transform kernels (issue #74) ---------*- C++ -*-===//
//
// The R-track's CPU-side transform kernels: contiguous per-channel int8
// quantize, fused strided-gather + quantize over a BoundPlan (the Case-1a
// kernel), int4 nibble pack, and fp32->fp16 convert. Every kernel has a
// scalar reference variant plus SIMD tiers selected at runtime, and every
// SIMD tier is bit-identical to scalar (tests enforce memcmp equality).
//
// Quantize semantics (all variants): q = (int8) rne(clamp(x * invScale,
// -128.0f, 127.0f)); clamp is max-then-min so NaN maps to -128 (the x86
// MAXPS(v, lo) rule); rne is round-to-nearest-even (default FP env).
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_QUANT_H
#define RELOC_QUANT_H

#include "reloc/Bind.h"

#include <cstdint>

namespace reloc {

class GatherPool;

namespace quant {

/// SIMD tier. AVX2 also implies FMA + F16C (co-resident on every AVX2 CPU
/// since Haswell; checked at runtime). AVX512 means F + BW. AVX512Pf is the
/// software-prefetch gather variant of gatherQuantizeF32S8 only, never
/// chosen by Auto -- the bench decides whether it earns its keep.
enum class Variant : int { Auto = 0, Scalar, AVX2, AVX512, AVX512Pf };

/// The four issue-#74 kernels, for per-kernel variant queries.
enum class Kernel : int { QuantizePack, GatherQuantize, PackS8S4,
                          ConvertF32F16 };

/// True when this host can execute `v`. Auto and Scalar are always true.
bool cpuSupports(Variant v);

/// True when `k` implements tier `v` (issue #74's variant table). Auto and
/// Scalar are true for every kernel.
bool kernelHasVariant(Kernel k, Variant v);

/// Auto -> the best implemented+supported plain tier (never AVX512Pf);
/// explicit variants return themselves. Asserts kernelHasVariant(k, v) and
/// cpuSupports(v).
Variant resolveFor(Kernel k, Variant v);

} // namespace quant
} // namespace reloc

#endif // RELOC_QUANT_H
```

- [ ] **Step 3: Run the test to verify it fails to link**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test
```

Expected: FAIL — `QuantTest.cpp` is not yet in the test CMake, so first add the CMake edits (Step 4), rebuild, and expect **undefined reference to `reloc::quant::cpuSupports(...)`** (header exists, no implementation).

- [ ] **Step 4: Wire CMake**

In `libreloc/CMakeLists.txt`, extend the `add_library` source list:

```cmake
add_library(reloc_runtime SHARED
  src/Version.cpp
  src/Decode.cpp
  src/Bind.cpp
  src/Execute.cpp
  src/CopyRun.cpp
  src/HostBackend.cpp
  src/PinnedBufferPool.cpp
  src/ChunkSchedule.cpp
  src/GatherPool.cpp
  src/Pipeline.cpp
  quant/Quant.cpp
)
```

In `libreloc/test/CMakeLists.txt`, add `QuantTest.cpp` to the `add_executable(libreloc-test ...)` list (after `PipelineTest.cpp`), and extend the include-scan test's directory list with `${CMAKE_CURRENT_SOURCE_DIR}/../quant`:

```cmake
add_test(NAME reloc-runtime-no-mlir-includes
         COMMAND bash -c "grep -rE '#include *[\"<](mlir|llvm)[/\"]' ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../cuda ${CMAKE_CURRENT_SOURCE_DIR}/../python ${CMAKE_CURRENT_SOURCE_DIR}/../quant; test \$? -eq 1")
```

- [ ] **Step 5: Write the minimal implementation**

Create `libreloc/quant/Quant.cpp`:

```cpp
//===- Quant.cpp - R0.1 quant kernel dispatch + scalar variants -----------===//

#include "reloc/Quant.h"

#include <cassert>

namespace reloc {
namespace quant {

bool cpuSupports(Variant v) {
  switch (v) {
  case Variant::Auto:
  case Variant::Scalar:
    return true;
  case Variant::AVX2:
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") &&
           __builtin_cpu_supports("fma");
#else
    return false;
#endif
  case Variant::AVX512:
  case Variant::AVX512Pf:
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
  }
  return false;
}

bool kernelHasVariant(Kernel k, Variant v) {
  if (v == Variant::Auto || v == Variant::Scalar)
    return true;
  switch (k) {
  case Kernel::QuantizePack:
    return v == Variant::AVX2 || v == Variant::AVX512;
  case Kernel::GatherQuantize:
    return v == Variant::AVX512 || v == Variant::AVX512Pf;
  case Kernel::PackS8S4:
    return v == Variant::AVX512;
  case Kernel::ConvertF32F16:
    return v == Variant::AVX2 || v == Variant::AVX512;
  }
  return false;
}

Variant resolveFor(Kernel k, Variant v) {
  assert(kernelHasVariant(k, v) && "variant not implemented for this kernel");
  assert(cpuSupports(v) && "variant unsupported on this host");
  if (v != Variant::Auto)
    return v;
  if (kernelHasVariant(k, Variant::AVX512) && cpuSupports(Variant::AVX512))
    return Variant::AVX512;
  if (kernelHasVariant(k, Variant::AVX2) && cpuSupports(Variant::AVX2))
    return Variant::AVX2;
  return Variant::Scalar;
}

} // namespace quant
} // namespace reloc
```

(`RELOC_QUANT_HAVE_X86_SIMD` is defined by CMake in Task 3; until then every SIMD tier reports unsupported, which is the correct conservative answer.)

- [ ] **Step 6: Build and run the tests**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test
/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test --gtest_filter='QuantDispatch.*'
```

Expected: 4 tests PASS. Also run the contract tests:

```bash
ctest --test-dir /home/jueonpark/sym/build/sym -R 'reloc-runtime-(mlir-free|no-mlir-includes)' --output-on-failure
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant/Quant.cpp \
        libreloc/test/QuantTest.cpp libreloc/CMakeLists.txt \
        libreloc/test/CMakeLists.txt
git commit -m "update(libreloc): R0.1 quant variant dispatch scaffolding (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `quantize_pack_f32_s8` — scalar variant

**Files:**
- Create: `libreloc/quant/QuantKernels.h`
- Modify: `libreloc/include/reloc/Quant.h` (add the public kernel)
- Modify: `libreloc/quant/Quant.cpp` (scalar impl + dispatch)
- Modify: `libreloc/test/QuantTest.cpp` (append tests)

**Interfaces:**
- Consumes: `Variant`, `Kernel`, `resolveFor` from Task 1.
- Produces:
  - Public: `void reloc::quant::quantizePackF32S8(const float *src, int8_t *dst, int64_t channels, int64_t channelSize, const float *invScales, Variant v = Variant::Auto)` — `src`/`dst` are `channels * channelSize` contiguous elements; `invScales[c]` applies to channel `c`.
  - Internal (`reloc::quant::detail`): `inline int8_t quantOne(float x, float invScale)` and `void quantizePackScalar(const float *src, int8_t *dst, int64_t n, float invScale)` — Tasks 3/6/7 reuse both.

- [ ] **Step 1: Append the failing tests to `libreloc/test/QuantTest.cpp`**

Add inside the anonymous namespace (test helpers first — Tasks 3–8 reuse them):

```cpp
// Independent reformulation of the quant contract (Global Constraints).
int8_t refQuantOne(float x, float invScale) {
  float y = x * invScale;
  if (std::isnan(y))
    return -128;
  if (y < -128.0f)
    y = -128.0f;
  if (y > 127.0f)
    y = 127.0f;
  return static_cast<int8_t>(std::lrintf(y)); // FE_TONEAREST default = RNE
}

std::vector<float> randomFloats(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float &x : v)
    x = d(rng);
  return v;
}

TEST(QuantizePack, ScalarMatchesReferencePerChannel) {
  const int64_t channels = 5, chSize = 67; // 67 = 4*16 + 3: remainder tail
  std::vector<float> src = randomFloats(channels * chSize, 1, -300.f, 300.f);
  std::vector<float> inv(channels);
  for (int64_t c = 0; c < channels; ++c)
    inv[c] = 0.05f + 0.9f * static_cast<float>(c);
  std::vector<int8_t> dst(channels * chSize, 42);
  reloc::quant::quantizePackF32S8(src.data(), dst.data(), channels, chSize,
                                  inv.data(), Variant::Scalar);
  for (int64_t c = 0; c < channels; ++c)
    for (int64_t i = 0; i < chSize; ++i)
      ASSERT_EQ(dst[c * chSize + i], refQuantOne(src[c * chSize + i], inv[c]))
          << "c=" << c << " i=" << i;
}

TEST(QuantizePack, ScalarSpecialValues) {
  // invScale = 1: RNE ties go to even; saturation clamps; NaN -> -128.
  const float src[] = {0.0f,   -0.0f,   0.5f, -0.5f, 1.5f,      2.5f, -2.5f,
                       200.0f, -200.0f, HUGE_VALF, -HUGE_VALF, NAN,  126.6f};
  const int8_t want[] = {0, 0, 0, 0, 2, 2, -2, 127, -128, 127, -128, -128, 127};
  const int64_t n = sizeof(src) / sizeof(src[0]);
  std::vector<int8_t> dst(n, 42);
  const float inv = 1.0f;
  reloc::quant::quantizePackF32S8(src, dst.data(), 1, n, &inv,
                                  Variant::Scalar);
  for (int64_t i = 0; i < n; ++i)
    EXPECT_EQ(dst[i], want[i]) << "i=" << i;
}
```

- [ ] **Step 2: Run to verify failure**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test
```

Expected: FAIL — `error: 'quantizePackF32S8' is not a member of 'reloc::quant'`.

- [ ] **Step 3: Implement**

Append to `libreloc/include/reloc/Quant.h` inside `namespace quant`, after `resolveFor`:

```cpp
/// K1, issue #74's `quantize_pack_f32_s8`: contiguous fp32 -> contiguous
/// int8 with a per-channel scale. `src`/`dst` hold channels * channelSize
/// elements; channel c covers [c * channelSize, (c+1) * channelSize) and
/// is quantized with invScales[c] (q = rne(clamp(x * invScale))).
void quantizePackF32S8(const float *src, int8_t *dst, int64_t channels,
                       int64_t channelSize, const float *invScales,
                       Variant v = Variant::Auto);
```

Create `libreloc/quant/QuantKernels.h`:

```cpp
//===- QuantKernels.h - internal per-variant kernel decls -------*- C++ -*-===//
//
// Shared between the dispatch TU (Quant.cpp) and the per-ISA TUs
// (QuantAVX2.cpp, QuantAVX512.cpp). SIMD symbols are only DEFINED when the
// build adds those TUs (RELOC_QUANT_HAVE_X86_SIMD); dispatch code must
// guard references accordingly.
//
//===----------------------------------------------------------------------===//

#ifndef RELOC_QUANT_QUANTKERNELS_H
#define RELOC_QUANT_QUANTKERNELS_H

#include <cmath>
#include <cstdint>

namespace reloc {
namespace quant {
namespace detail {

/// The scalar quant contract. fmax first so NaN -> -128, matching the
/// vector MAXPS(v, lo) operand order; nearbyintf under the default FP env
/// is round-to-nearest-even, matching CVTPS2DQ.
inline int8_t quantOne(float x, float invScale) {
  float y = x * invScale;
  y = std::fmax(y, -128.0f);
  y = std::fmin(y, 127.0f);
  return static_cast<int8_t>(std::nearbyintf(y));
}

void quantizePackScalar(const float *src, int8_t *dst, int64_t n,
                        float invScale);

} // namespace detail
} // namespace quant
} // namespace reloc

#endif // RELOC_QUANT_QUANTKERNELS_H
```

In `libreloc/quant/Quant.cpp`, add `#include "QuantKernels.h"` (after `reloc/Quant.h`), and append:

```cpp
namespace detail {

void quantizePackScalar(const float *src, int8_t *dst, int64_t n,
                        float invScale) {
  for (int64_t i = 0; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

} // namespace detail

void quantizePackF32S8(const float *src, int8_t *dst, int64_t channels,
                       int64_t channelSize, const float *invScales,
                       Variant v) {
  const Variant r = resolveFor(Kernel::QuantizePack, v);
  (void)r; // scalar-only until the SIMD TUs land (Task 3)
  for (int64_t c = 0; c < channels; ++c)
    detail::quantizePackScalar(src + c * channelSize, dst + c * channelSize,
                               channelSize, invScales[c]);
}
```

(Placement note: the `detail` block and the public function go inside the existing `namespace reloc { namespace quant {` scope, before the closing braces.)

- [ ] **Step 4: Build and run**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test
/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test --gtest_filter='QuantizePack.*'
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant/Quant.cpp \
        libreloc/quant/QuantKernels.h libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 quantize_pack_f32_s8 scalar kernel (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `quantize_pack_f32_s8` — AVX2 and AVX-512 variants

**Files:**
- Create: `libreloc/quant/QuantAVX2.cpp`
- Create: `libreloc/quant/QuantAVX512.cpp`
- Modify: `libreloc/quant/QuantKernels.h` (declare SIMD runs)
- Modify: `libreloc/quant/Quant.cpp` (dispatch to SIMD)
- Modify: `libreloc/CMakeLists.txt` (SIMD TUs + flags + define)
- Modify: `libreloc/test/QuantTest.cpp` (bit-exactness tests)

**Interfaces:**
- Consumes: `detail::quantOne`, `detail::quantizePackScalar`, dispatch from Tasks 1–2.
- Produces (`reloc::quant::detail`, defined only under `RELOC_QUANT_HAVE_X86_SIMD`):
  - `void quantizePackAVX2(const float *src, int8_t *dst, int64_t n, float invScale)`
  - `void quantizePackAVX512(const float *src, int8_t *dst, int64_t n, float invScale)` — Task 7's stride-1 fast path reuses this.

- [ ] **Step 1: Append the failing test**

```cpp
TEST(QuantizePack, SimdVariantsBitExactVsScalar) {
  bool ranAny = false;
  for (Variant v : {Variant::AVX2, Variant::AVX512}) {
    if (!reloc::quant::cpuSupports(v))
      continue;
    ranAny = true;
    for (int64_t n : {1, 15, 16, 17, 31, 32, 33, 64, 1000, 4099}) {
      std::vector<float> src =
          randomFloats(n, static_cast<uint32_t>(7 + n), -300.f, 300.f);
      // Poke the clamp/NaN lanes inside a full vector when room allows.
      if (n >= 33) {
        src[3] = NAN;
        src[17] = HUGE_VALF;
        src[32] = -HUGE_VALF;
      }
      std::vector<int8_t> a(n, 0), b(n, 0);
      const float inv = 0.37f;
      reloc::quant::quantizePackF32S8(src.data(), a.data(), 1, n, &inv,
                                      Variant::Scalar);
      reloc::quant::quantizePackF32S8(src.data(), b.data(), 1, n, &inv, v);
      ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n))
          << "variant=" << static_cast<int>(v) << " n=" << n;
    }
  }
  if (!ranAny)
    GTEST_SKIP() << "no SIMD tier supported on this host";
}
```

- [ ] **Step 2: Run to verify failure**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test && \
/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test --gtest_filter='QuantizePack.SimdVariantsBitExactVsScalar'
```

Expected on this machine (7800X3D): the test **runs but currently SKIPs** (`cpuSupports` returns false because `RELOC_QUANT_HAVE_X86_SIMD` is not yet defined). That skip is the failure signal here — after Step 3–5 it must run both tiers and pass.

- [ ] **Step 3: CMake — add the SIMD TUs**

In `libreloc/CMakeLists.txt`, after the `target_include_directories(reloc_runtime ...)` block:

```cmake
# R0.1 SIMD variants (issue #74): per-TU ISA flags + runtime dispatch via
# __builtin_cpu_supports, so the shared object runs on any x86-64 host
# (the 2080 Ti box's CPU may be AVX2-only). Non-x86 builds get scalar only.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_sources(reloc_runtime PRIVATE
    quant/QuantAVX2.cpp
    quant/QuantAVX512.cpp
  )
  set_source_files_properties(quant/QuantAVX2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(quant/QuantAVX512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw")
  target_compile_definitions(reloc_runtime PRIVATE RELOC_QUANT_HAVE_X86_SIMD=1)
endif()
```

- [ ] **Step 4: Declare + implement the SIMD kernels**

Append to `libreloc/quant/QuantKernels.h` inside `namespace detail` (after `quantizePackScalar`):

```cpp
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
// Defined in QuantAVX2.cpp / QuantAVX512.cpp (per-TU -m flags).
void quantizePackAVX2(const float *src, int8_t *dst, int64_t n,
                      float invScale);
void quantizePackAVX512(const float *src, int8_t *dst, int64_t n,
                        float invScale);
#endif
```

Create `libreloc/quant/QuantAVX2.cpp`:

```cpp
//===- QuantAVX2.cpp - AVX2/FMA/F16C quant kernel variants ----------------===//
// Compiled with -mavx2 -mfma -mf16c (see libreloc/CMakeLists.txt); callers
// must gate on cpuSupports(Variant::AVX2).

#include "QuantKernels.h"

#include <immintrin.h>

namespace reloc {
namespace quant {
namespace detail {

void quantizePackAVX2(const float *src, int8_t *dst, int64_t n,
                      float invScale) {
  const __m256 vinv = _mm256_set1_ps(invScale);
  const __m256 vlo = _mm256_set1_ps(-128.0f);
  const __m256 vhi = _mm256_set1_ps(127.0f);
  // packs_epi32 + packs_epi16 interleave 128-bit lanes; this permutation
  // restores source order (dword k of the result = elements 4k..4k+3).
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i q[4];
    for (int k = 0; k < 4; ++k) {
      __m256 v = _mm256_loadu_ps(src + i + 8 * k);
      v = _mm256_mul_ps(v, vinv);
      v = _mm256_max_ps(v, vlo); // NaN -> vlo (MAXPS second-operand rule)
      v = _mm256_min_ps(v, vhi);
      q[k] = _mm256_cvtps_epi32(v); // RNE
    }
    __m256i p01 = _mm256_packs_epi32(q[0], q[1]);
    __m256i p23 = _mm256_packs_epi32(q[2], q[3]);
    __m256i p = _mm256_packs_epi16(p01, p23);
    p = _mm256_permutevar8x32_epi32(p, order);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), p);
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

} // namespace detail
} // namespace quant
} // namespace reloc
```

Create `libreloc/quant/QuantAVX512.cpp`:

```cpp
//===- QuantAVX512.cpp - AVX-512 (F+BW) quant kernel variants -------------===//
// Compiled with -mavx512f -mavx512bw (see libreloc/CMakeLists.txt); callers
// must gate on cpuSupports(Variant::AVX512).

#include "QuantKernels.h"

#include <cassert>
#include <immintrin.h>

namespace reloc {
namespace quant {
namespace detail {

void quantizePackAVX512(const float *src, int8_t *dst, int64_t n,
                        float invScale) {
  const __m512 vinv = _mm512_set1_ps(invScale);
  const __m512 vlo = _mm512_set1_ps(-128.0f);
  const __m512 vhi = _mm512_set1_ps(127.0f);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    v = _mm512_mul_ps(v, vinv);
    v = _mm512_max_ps(v, vlo); // NaN -> vlo (MAXPS second-operand rule)
    v = _mm512_min_ps(v, vhi);
    __m512i q = _mm512_cvtps_epi32(v); // RNE
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i),
                     _mm512_cvtsepi32_epi8(q));
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i], invScale);
}

} // namespace detail
} // namespace quant
} // namespace reloc
```

In `libreloc/quant/Quant.cpp`, replace the body of `quantizePackF32S8` with the real dispatch:

```cpp
void quantizePackF32S8(const float *src, int8_t *dst, int64_t channels,
                       int64_t channelSize, const float *invScales,
                       Variant v) {
  const Variant r = resolveFor(Kernel::QuantizePack, v);
  for (int64_t c = 0; c < channels; ++c) {
    const float *s = src + c * channelSize;
    int8_t *d = dst + c * channelSize;
    switch (r) {
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
    case Variant::AVX512:
      detail::quantizePackAVX512(s, d, channelSize, invScales[c]);
      break;
    case Variant::AVX2:
      detail::quantizePackAVX2(s, d, channelSize, invScales[c]);
      break;
#endif
    default:
      detail::quantizePackScalar(s, d, channelSize, invScales[c]);
      break;
    }
  }
}
```

- [ ] **Step 5: Reconfigure, build, run**

```bash
cmake -S /home/jueonpark/sym -B /home/jueonpark/sym/build/sym >/dev/null && \
ninja -C /home/jueonpark/sym/build/sym libreloc-test && \
/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test --gtest_filter='Quant*'
```

Expected: all Quant tests PASS; `SimdVariantsBitExactVsScalar` exercises AVX2 **and** AVX512 on this machine (no skip). Re-run the two `reloc-runtime-*` ctests too (new TUs must not violate the contract).

- [ ] **Step 6: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/quant/QuantAVX2.cpp libreloc/quant/QuantAVX512.cpp \
        libreloc/quant/QuantKernels.h libreloc/quant/Quant.cpp \
        libreloc/CMakeLists.txt libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 quantize_pack AVX2 + AVX-512 variants (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `convert_f32_f16` — scalar, F16C, AVX-512

**Files:**
- Modify: `libreloc/include/reloc/Quant.h`, `libreloc/quant/QuantKernels.h`, `libreloc/quant/Quant.cpp`, `libreloc/quant/QuantAVX2.cpp`, `libreloc/quant/QuantAVX512.cpp`, `libreloc/test/QuantTest.cpp`

**Interfaces:**
- Consumes: dispatch machinery (Task 1), SIMD TU layout (Task 3).
- Produces:
  - Public: `void reloc::quant::convertF32F16(const float *src, uint16_t *dst, int64_t count, Variant v = Variant::Auto)`
  - Internal: `uint16_t detail::f32ToF16Scalar(float f)`, `void detail::convertF32F16Scalar(const float *, uint16_t *, int64_t)`, and under the SIMD define `convertF32F16F16C` / `convertF32F16AVX512`.

- [ ] **Step 1: Append the failing tests**

```cpp
TEST(ConvertF32F16, ScalarSpecials) {
  struct Case {
    float in;
    uint16_t want;
  } cases[] = {
      {0.0f, 0x0000},          {-0.0f, 0x8000},
      {1.0f, 0x3C00},          {-2.0f, 0xC000},
      {65504.0f, 0x7BFF},      {65519.0f, 0x7BFF}, // below RNE-to-inf cut
      {65520.0f, 0x7C00},      {HUGE_VALF, 0x7C00},
      {-HUGE_VALF, 0xFC00},
      {5.9604645e-8f, 0x0001}, // 2^-24, smallest half subnormal
      {2.9802322e-8f, 0x0000}, // 2^-25: tie, rounds to even (zero)
      {4.4703484e-8f, 0x0001}, // 1.5 * 2^-25: rounds up
      {6.1035156e-5f, 0x0400}, // 2^-14, smallest half normal
      {0.1f, 0x2E66},          {3.14159265f, 0x4248},
  };
  for (const Case &c : cases) {
    uint16_t out = 0;
    reloc::quant::convertF32F16(&c.in, &out, 1, Variant::Scalar);
    EXPECT_EQ(out, c.want) << "in=" << c.in;
  }
  float nan = NAN;
  uint16_t h = 0;
  reloc::quant::convertF32F16(&nan, &h, 1, Variant::Scalar);
  EXPECT_EQ(h & 0x7C00u, 0x7C00u);
  EXPECT_NE(h & 0x3FFu, 0u); // still a NaN, not inf
}

TEST(ConvertF32F16, SimdVariantsBitExactVsScalar) {
  bool ranAny = false;
  for (Variant v : {Variant::AVX2, Variant::AVX512}) {
    if (!reloc::quant::cpuSupports(v))
      continue;
    ranAny = true;
    std::mt19937 rng(11);
    for (int64_t n : {1, 7, 8, 9, 16, 33, 100000}) {
      // Random BIT PATTERNS: covers normals, subnormals, inf, both signs,
      // every exponent. This empirically pins the scalar converter to the
      // hardware VCVTPS2PH result. NaNs are tested semantically below.
      std::vector<float> src(n);
      for (float &f : src) {
        uint32_t bits = rng();
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        f = std::isnan(x) ? 1.0f : x;
      }
      std::vector<uint16_t> a(n, 0xDEAD), b(n, 0xBEEF);
      reloc::quant::convertF32F16(src.data(), a.data(), n, Variant::Scalar);
      reloc::quant::convertF32F16(src.data(), b.data(), n, v);
      ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n * sizeof(uint16_t)))
          << "variant=" << static_cast<int>(v) << " n=" << n;
    }
    float nan = NAN;
    uint16_t h = 0;
    reloc::quant::convertF32F16(&nan, &h, 1, v);
    EXPECT_EQ(h & 0x7C00u, 0x7C00u);
    EXPECT_NE(h & 0x3FFu, 0u);
  }
  if (!ranAny)
    GTEST_SKIP() << "no SIMD tier supported on this host";
}
```

- [ ] **Step 2: Run to verify failure**

`ninja -C /home/jueonpark/sym/build/sym libreloc-test` — expected: FAIL, `'convertF32F16' is not a member of 'reloc::quant'`.

- [ ] **Step 3: Implement**

`libreloc/include/reloc/Quant.h`, after `quantizePackF32S8`:

```cpp
/// K4, issue #74's `convert_f32_f16`: contiguous fp32 -> IEEE binary16
/// (round-to-nearest-even, subnormals preserved, overflow -> inf). Scalar
/// output is bit-identical to F16C/AVX-512 VCVTPS2PH for non-NaN inputs.
void convertF32F16(const float *src, uint16_t *dst, int64_t count,
                   Variant v = Variant::Auto);
```

`libreloc/quant/QuantKernels.h`, inside `namespace detail` (the scalar decls go with `quantizePackScalar`; add `#include <cstring>` at the top for `memcpy` if placing the converter inline — here it is declared only):

```cpp
uint16_t f32ToF16Scalar(float f);
void convertF32F16Scalar(const float *src, uint16_t *dst, int64_t n);
```

and inside the existing `#if defined(RELOC_QUANT_HAVE_X86_SIMD)` block:

```cpp
void convertF32F16F16C(const float *src, uint16_t *dst, int64_t n);
void convertF32F16AVX512(const float *src, uint16_t *dst, int64_t n);
```

`libreloc/quant/Quant.cpp` — add `#include <cstring>` at the top, then inside `namespace detail`:

```cpp
uint16_t f32ToF16Scalar(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  x &= 0x7fffffffu;
  if (x > 0x7f800000u) // NaN: quiet bit + truncated payload (VCVTPS2PH)
    return static_cast<uint16_t>(sign | 0x7e00u | ((x >> 13) & 0x3ffu));
  if (x >= 0x477ff000u) // inf, or >= 65520 which RNE-rounds to inf
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (x >= 0x38800000u) {
    // Normal half. Add 0xfff + (bit 13) to RNE the 13 dropped mantissa
    // bits; the carry ripples into the exponent field correctly.
    const uint32_t r = x + 0xfffu + ((x >> 13) & 1u);
    return static_cast<uint16_t>(sign | ((r - 0x38000000u) >> 13));
  }
  if (x <= 0x33000000u) // <= 2^-25: ties-to-even lands on zero
    return sign;
  // Subnormal half: value = mant24 * 2^(exp - 150), result ulp = 2^-24.
  const uint32_t shift = 126u - (x >> 23); // in [14, 24] here
  const uint32_t mant = (x & 0x7fffffu) | 0x800000u;
  uint32_t r = mant >> shift;
  const uint32_t rem = mant & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (r & 1u)))
    ++r; // may carry into 0x400: that IS the smallest normal -- correct
  return static_cast<uint16_t>(sign | r);
}

void convertF32F16Scalar(const float *src, uint16_t *dst, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    dst[i] = f32ToF16Scalar(src[i]);
}
```

and the public dispatch after `quantizePackF32S8`:

```cpp
void convertF32F16(const float *src, uint16_t *dst, int64_t count,
                   Variant v) {
  const Variant r = resolveFor(Kernel::ConvertF32F16, v);
  switch (r) {
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  case Variant::AVX512:
    detail::convertF32F16AVX512(src, dst, count);
    break;
  case Variant::AVX2:
    detail::convertF32F16F16C(src, dst, count);
    break;
#endif
  default:
    detail::convertF32F16Scalar(src, dst, count);
    break;
  }
}
```

`libreloc/quant/QuantAVX2.cpp`, append inside `namespace detail`:

```cpp
void convertF32F16F16C(const float *src, uint16_t *dst, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(dst + i),
        _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  for (; i < n; ++i)
    dst[i] = f32ToF16Scalar(src[i]);
}
```

`libreloc/quant/QuantAVX512.cpp`, append inside `namespace detail`:

```cpp
void convertF32F16AVX512(const float *src, uint16_t *dst, int64_t n) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(dst + i),
        _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  for (; i < n; ++i)
    dst[i] = f32ToF16Scalar(src[i]);
}
```

- [ ] **Step 4: Build and run**

```bash
ninja -C /home/jueonpark/sym/build/sym libreloc-test && \
/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test --gtest_filter='ConvertF32F16.*'
```

Expected: PASS (the 100k-random-pattern case pins scalar to hardware output bit-for-bit).

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 convert_f32_f16 scalar + F16C + AVX-512 (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `pack_s8_s4` — scalar and AVX-512

**Files:**
- Modify: `libreloc/include/reloc/Quant.h`, `libreloc/quant/QuantKernels.h`, `libreloc/quant/Quant.cpp`, `libreloc/quant/QuantAVX512.cpp`, `libreloc/test/QuantTest.cpp`

**Interfaces:**
- Consumes: dispatch machinery.
- Produces:
  - Public: `void reloc::quant::packS8S4(const int8_t *src, uint8_t *dst, int64_t pairs, Variant v = Variant::Auto)` — reads `2 * pairs` int8, writes `pairs` bytes.
  - Internal: `detail::packS8S4Scalar`, `detail::packS8S4AVX512`.

- [ ] **Step 1: Append the failing tests**

```cpp
uint8_t refNibble(int8_t v) {
  int x = v < -8 ? -8 : (v > 7 ? 7 : v);
  return static_cast<uint8_t>(x & 0xF);
}

TEST(PackS8S4, ScalarPacksAndSaturates) {
  const std::vector<int8_t> src = {0,   1,    -1, 7, -8, 8,
                                   -9,  127,  -128, 3, 5,  -6};
  const int64_t pairs = 6;
  std::vector<uint8_t> dst(pairs, 0xAA);
  reloc::quant::packS8S4(src.data(), dst.data(), pairs, Variant::Scalar);
  for (int64_t i = 0; i < pairs; ++i) {
    const uint8_t want = static_cast<uint8_t>(
        refNibble(src[2 * i]) | (refNibble(src[2 * i + 1]) << 4));
    EXPECT_EQ(dst[i], want) << "i=" << i;
  }
}

TEST(PackS8S4, Avx512BitExactVsScalar) {
  if (!reloc::quant::cpuSupports(Variant::AVX512))
    GTEST_SKIP() << "AVX-512 unsupported on this host";
  std::mt19937 rng(13);
  for (int64_t pairs : {1, 31, 32, 33, 100, 100003}) {
    std::vector<int8_t> src(2 * pairs);
    for (int8_t &b : src)
      b = static_cast<int8_t>(rng()); // full int8 range incl. saturating
    std::vector<uint8_t> a(pairs, 0), b(pairs, 1);
    reloc::quant::packS8S4(src.data(), a.data(), pairs, Variant::Scalar);
    reloc::quant::packS8S4(src.data(), b.data(), pairs, Variant::AVX512);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), pairs)) << "pairs=" << pairs;
  }
}
```

- [ ] **Step 2: Run to verify failure** — `'packS8S4' is not a member of 'reloc::quant'`.

- [ ] **Step 3: Implement**

Header, after `convertF32F16`:

```cpp
/// K3, issue #74's `pack_s8_s4`: int8 -> int4 nibble pack for the r=0.125
/// sweep. Each input is saturated to [-8, 7]; output byte i = low nibble
/// from src[2i], high nibble from src[2i+1]. Reads 2*pairs, writes pairs.
void packS8S4(const int8_t *src, uint8_t *dst, int64_t pairs,
              Variant v = Variant::Auto);
```

`QuantKernels.h` (`detail`, scalar section + a shared inline):

```cpp
inline uint8_t nibbleSat(int8_t v) {
  int x = v < -8 ? -8 : (v > 7 ? 7 : v);
  return static_cast<uint8_t>(x & 0xF);
}

void packS8S4Scalar(const int8_t *src, uint8_t *dst, int64_t pairs);
```

and in the SIMD block: `void packS8S4AVX512(const int8_t *src, uint8_t *dst, int64_t pairs);`

`Quant.cpp` (`detail` + public dispatch):

```cpp
void packS8S4Scalar(const int8_t *src, uint8_t *dst, int64_t pairs) {
  for (int64_t i = 0; i < pairs; ++i)
    dst[i] = static_cast<uint8_t>(nibbleSat(src[2 * i]) |
                                  (nibbleSat(src[2 * i + 1]) << 4));
}
```

```cpp
void packS8S4(const int8_t *src, uint8_t *dst, int64_t pairs, Variant v) {
  const Variant r = resolveFor(Kernel::PackS8S4, v);
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  if (r == Variant::AVX512) {
    detail::packS8S4AVX512(src, dst, pairs);
    return;
  }
#endif
  (void)r;
  detail::packS8S4Scalar(src, dst, pairs);
}
```

`QuantAVX512.cpp`:

```cpp
void packS8S4AVX512(const int8_t *src, uint8_t *dst, int64_t pairs) {
  const __m512i lo8 = _mm512_set1_epi8(-8);
  const __m512i hi7 = _mm512_set1_epi8(7);
  const __m512i mask = _mm512_set1_epi16(0x0F0F);
  int64_t i = 0;
  for (; i + 32 <= pairs; i += 32) { // 64 int8 in -> 32 packed bytes out
    __m512i v = _mm512_loadu_si512(src + 2 * i);
    v = _mm512_max_epi8(v, lo8);
    v = _mm512_min_epi8(v, hi7);
    v = _mm512_and_si512(v, mask);
    // Per epi16 lane: bits 0-3 = even nibble, bits 8-11 = odd nibble.
    // v | (v >> 4) puts the odd nibble at bits 4-7; VPMOVWB truncates each
    // lane to its low byte = the packed result.
    const __m512i t = _mm512_or_si512(v, _mm512_srli_epi16(v, 4));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i),
                        _mm512_cvtepi16_epi8(t));
  }
  for (; i < pairs; ++i)
    dst[i] = static_cast<uint8_t>(nibbleSat(src[2 * i]) |
                                  (nibbleSat(src[2 * i + 1]) << 4));
}
```

- [ ] **Step 4: Build and run** — `--gtest_filter='PackS8S4.*'`, expected PASS (both tests, AVX-512 live on this machine).

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 pack_s8_s4 scalar + AVX-512 (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `gather_quantize_f32_s8` — plan-driven scalar variant

The Case-1a kernel: strided fp32 reads through a `BoundPlan`'s stride sets, contiguous int8 writes, per-outer-channel scale, chunked outer range exactly like `gatherChunk` (`libreloc/src/Execute.cpp:115`).

**Files:**
- Modify: `libreloc/include/reloc/Quant.h`, `libreloc/quant/QuantKernels.h`, `libreloc/quant/Quant.cpp`, `libreloc/test/QuantTest.cpp`

**Interfaces:**
- Consumes: `reloc::BoundPlan` (`reloc/Bind.h`): `extents`, `srcStrides`, `dstStrides` (element units), `elementSize`, `padRegions`. `detail::quantOne`, `detail::quantizePackScalar`.
- Produces:
  - Public: `void reloc::quant::gatherQuantizeF32S8(const BoundPlan &bound, const float *srcBase, int8_t *dstBase, const float *invScales, int64_t outerBegin, int64_t outerEnd, Variant v = Variant::Auto)` — `invScales` has `bound.extents[0]` entries (channel = coalesced outer axis); `dstBase` is an int8 buffer addressed by the plan's dst element offsets (1 byte per element).
  - Internal: `using QuantRunFn = void (*)(const float *src, int64_t srcStride, int8_t *dst, int64_t n, float invScale);` and `void quantRunScalar(...)` with that signature — Task 7 plugs SIMD runs into the same walk.
  - Preconditions (asserted): `elementSize == 4`, `padRegions.empty()`, `extents.size() >= 2`, `dstStrides.back() == 1`.

- [ ] **Step 1: Append the failing tests**

```cpp
// Hand-built 2-D transpose-style plan: dst is rows x cols dense row-major,
// src is read with swapped strides (mirrors what bind() produces for
// reloc.transpose; ExecuteTest builds plans the same way).
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

// Naive full-index-space walk: the independent oracle for the fused kernel.
void refGatherQuantize(const reloc::BoundPlan &b, const float *src,
                       int8_t *dst, const float *invScales) {
  const size_t r = b.extents.size();
  std::vector<int64_t> idx(r, 0);
  while (true) {
    int64_t so = 0, dso = 0;
    for (size_t k = 0; k < r; ++k) {
      so += idx[k] * b.srcStrides[k];
      dso += idx[k] * b.dstStrides[k];
    }
    dst[dso] = refQuantOne(src[so], invScales[idx[0]]);
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

TEST(GatherQuantize, ScalarMatchesNaiveWalk2D) {
  auto b = transposePlan(5, 67); // strided inner reads + remainder tail
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 3, -300.f, 300.f);
  std::vector<float> inv = {0.9f, 0.1f, 1.7f, 0.03f, 2.5f};
  const size_t total = 5 * 67;
  std::vector<int8_t> got(total, 42), want(total, 24);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), total));
}

TEST(GatherQuantize, ContiguousInnerFastPath) {
  // srcStrides.back() == 1: rows with a gap between them (row-major copy
  // out of a larger parent buffer) -- exercises the stride-1 fast path.
  reloc::BoundPlan b;
  b.extents = {4, 33};
  b.srcStrides = {40, 1};
  b.dstStrides = {33, 1};
  b.elementSize = 4;
  b.totalBytes = 4 * 33 * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 5, -300.f, 300.f);
  std::vector<float> inv = {1.0f, 0.5f, 0.25f, 2.0f};
  std::vector<int8_t> got(4 * 33, 0), want(4 * 33, 1);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size()));
}

TEST(GatherQuantize, ScalarMatchesNaiveWalk3D) {
  reloc::BoundPlan b;
  b.extents = {4, 6, 33};
  b.srcStrides = {2, 9, 100}; // arbitrary positive, strided innermost
  b.dstStrides = {198, 33, 1}; // dense row-major dst
  b.elementSize = 4;
  b.totalBytes = 4 * 6 * 33 * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 9, -300.f, 300.f);
  std::vector<float> inv = {0.4f, 1.1f, 0.7f, 3.0f};
  std::vector<int8_t> got(4 * 6 * 33, 0), want(4 * 6 * 33, 1);
  refGatherQuantize(b, src.data(), want.data(), inv.data());
  reloc::quant::gatherQuantizeF32S8(b, src.data(), got.data(), inv.data(), 0,
                                    b.extents[0], Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(got.data(), want.data(), got.size()));
}

TEST(GatherQuantize, ChunkedEqualsWholeRange) {
  auto b = transposePlan(5, 67);
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 3, -300.f, 300.f);
  std::vector<float> inv = {0.9f, 0.1f, 1.7f, 0.03f, 2.5f};
  const size_t total = 5 * 67;
  std::vector<int8_t> whole(total, 0), chunked(total, 1);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), whole.data(), inv.data(),
                                    0, 5, Variant::Scalar);
  for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2}, {2, 4}, {4, 5}})
    reloc::quant::gatherQuantizeF32S8(b, src.data(), chunked.data(),
                                      inv.data(), lo, hi, Variant::Scalar);
  EXPECT_EQ(0, std::memcmp(whole.data(), chunked.data(), total));
}
```

- [ ] **Step 2: Run to verify failure** — `'gatherQuantizeF32S8' is not a member of 'reloc::quant'`.

- [ ] **Step 3: Implement**

Header, after `packS8S4`:

```cpp
/// K2, issue #74's `gather_quantize_f32_s8` (the Case-1a kernel): strided
/// fp32 reads through the BoundPlan's stride sets, fused per-channel int8
/// quantize, contiguous int8 writes at the plan's dst element offsets
/// (1 byte per element -- an int8 image of the dst layout). Chunked outer
/// form mirroring gatherChunk: writes ONLY outer indices [outerBegin,
/// outerEnd). Channel = coalesced outer axis: invScales has extents[0]
/// entries. v0 preconditions (asserted): elementSize == 4, no padRegions,
/// rank >= 2, dstStrides.back() == 1.
void gatherQuantizeF32S8(const BoundPlan &bound, const float *srcBase,
                         int8_t *dstBase, const float *invScales,
                         int64_t outerBegin, int64_t outerEnd,
                         Variant v = Variant::Auto);
```

`QuantKernels.h` (`detail`):

```cpp
/// One innermost run of the fused kernel: n elements read at srcStride,
/// written contiguously. Implementations must be bit-identical.
using QuantRunFn = void (*)(const float *src, int64_t srcStride, int8_t *dst,
                            int64_t n, float invScale);

void quantRunScalar(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale);
```

`Quant.cpp` — inside `namespace detail`:

```cpp
void quantRunScalar(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale) {
  if (srcStride == 1) {
    quantizePackScalar(src, dst, n, invScale);
    return;
  }
  for (int64_t i = 0; i < n; ++i)
    dst[i] = quantOne(src[i * srcStride], invScale);
}
```

then, in the anonymous namespace of `Quant.cpp` (add one above the `quant` namespace or nested — mirror Execute.cpp's file-local style):

```cpp
namespace {

// Mirror of Execute.cpp's walk() for the fused kernel: only the top axis is
// ranged to the chunk; deeper axes cover their full extent. The per-channel
// invScale is picked at depth 0 and constant below it.
void quantWalk(const reloc::BoundPlan &b, const float *src, int8_t *dst,
               const float *invScales, size_t depth, int64_t iBegin,
               int64_t iEnd, int64_t srcOff, int64_t dstOff, float invScale,
               reloc::quant::detail::QuantRunFn run) {
  const size_t r = b.extents.size();
  if (depth == r - 1) {
    run(src + srcOff + iBegin * b.srcStrides[depth], b.srcStrides[depth],
        dst + dstOff + iBegin, iEnd - iBegin, invScale);
    return;
  }
  for (int64_t i = iBegin; i < iEnd; ++i)
    quantWalk(b, src, dst, invScales, depth + 1, 0, b.extents[depth + 1],
              srcOff + i * b.srcStrides[depth],
              dstOff + i * b.dstStrides[depth],
              depth == 0 ? invScales[i] : invScale, run);
}

} // namespace
```

and the public function:

```cpp
void gatherQuantizeF32S8(const BoundPlan &bound, const float *srcBase,
                         int8_t *dstBase, const float *invScales,
                         int64_t outerBegin, int64_t outerEnd, Variant v) {
  assert(bound.elementSize == 4 && "fused quantize is fp32-only");
  assert(bound.padRegions.empty() && "pads unsupported in gatherQuantize v0");
  assert(bound.extents.size() >= 2 &&
         "per-channel scale needs a distinct outer axis");
  assert(bound.dstStrides.back() == 1 &&
         "dst innermost axis must be contiguous");
  const Variant r = resolveFor(Kernel::GatherQuantize, v);
  detail::QuantRunFn run = detail::quantRunScalar;
  (void)r; // SIMD runs land in Task 7
  quantWalk(bound, srcBase, dstBase, invScales, /*depth=*/0, outerBegin,
            outerEnd, /*srcOff=*/0, /*dstOff=*/0, /*invScale=*/0.0f, run);
}
```

(`quantWalk` is declared before the `quant` namespace's function definitions; add `#include <cassert>` if not already present.)

- [ ] **Step 4: Build and run** — `--gtest_filter='GatherQuantize.*'`, expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 gather_quantize_f32_s8 plan-driven scalar (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `gather_quantize_f32_s8` — AVX-512 gather and prefetch variants

**Files:**
- Modify: `libreloc/quant/QuantKernels.h`, `libreloc/quant/QuantAVX512.cpp`, `libreloc/quant/Quant.cpp` (run-fn selection), `libreloc/test/QuantTest.cpp`

**Interfaces:**
- Consumes: `QuantRunFn`, `quantWalk`, `quantizePackAVX512`.
- Produces (`detail`, SIMD block): `void quantRunAVX512(const float *, int64_t, int8_t *, int64_t, float)` and `void quantRunAVX512Pf(...)` (same signature).

- [ ] **Step 1: Append the failing test**

```cpp
TEST(GatherQuantize, SimdVariantsBitExactVsScalar) {
  if (!reloc::quant::cpuSupports(Variant::AVX512))
    GTEST_SKIP() << "AVX-512 unsupported on this host";
  struct PlanCase {
    const char *name;
    reloc::BoundPlan b;
  };
  std::vector<PlanCase> plans;
  plans.push_back({"transpose 129x517", transposePlan(129, 517)});
  plans.push_back({"transpose 16x16", transposePlan(16, 16)});
  {
    reloc::BoundPlan b; // contiguous inner fast path
    b.extents = {7, 133};
    b.srcStrides = {140, 1};
    b.dstStrides = {133, 1};
    b.elementSize = 4;
    b.totalBytes = 7 * 133 * 4;
    plans.push_back({"contiguous inner", b});
  }
  {
    reloc::BoundPlan b; // large stride: distinct cache line per gather lane
    b.extents = {3, 65};
    b.srcStrides = {1, 8192};
    b.dstStrides = {65, 1};
    b.elementSize = 4;
    b.totalBytes = 3 * 65 * 4;
    plans.push_back({"stride 8192", b});
  }
  for (auto &pc : plans) {
    std::vector<float> src =
        randomFloats(maxSrcOffset(pc.b) + 1, 21, -300.f, 300.f);
    std::vector<float> inv(pc.b.extents[0]);
    for (size_t c = 0; c < inv.size(); ++c)
      inv[c] = 0.03f + 0.11f * static_cast<float>(c);
    const size_t total = static_cast<size_t>(pc.b.totalBytes / 4);
    std::vector<int8_t> ref(total, 0);
    reloc::quant::gatherQuantizeF32S8(pc.b, src.data(), ref.data(),
                                      inv.data(), 0, pc.b.extents[0],
                                      Variant::Scalar);
    for (Variant v : {Variant::AVX512, Variant::AVX512Pf}) {
      std::vector<int8_t> got(total, 1);
      reloc::quant::gatherQuantizeF32S8(pc.b, src.data(), got.data(),
                                        inv.data(), 0, pc.b.extents[0], v);
      ASSERT_EQ(0, std::memcmp(ref.data(), got.data(), total))
          << pc.name << " variant=" << static_cast<int>(v);
    }
  }
}
```

- [ ] **Step 2: Run to verify failure**

Expected: the test **aborts on the debug assert** inside `resolveFor`? No — `AVX512` is implemented for `GatherQuantize` in the Task 1 table and supported on this host, but `gatherQuantizeF32S8` still selects `quantRunScalar` for every variant, so the test PASSES vacuously... To make it fail first, note Step 3 changes behavior; the honest failing signal is a **link error**: reference `detail::quantRunAVX512` in the dispatch *before* implementing it. So: apply the `Quant.cpp` dispatch edit (below) first, build → expected **undefined reference to `reloc::quant::detail::quantRunAVX512`**, then implement in `QuantAVX512.cpp`.

`Quant.cpp` — replace the run-fn selection in `gatherQuantizeF32S8`:

```cpp
  const Variant r = resolveFor(Kernel::GatherQuantize, v);
  detail::QuantRunFn run = detail::quantRunScalar;
#if defined(RELOC_QUANT_HAVE_X86_SIMD)
  if (r == Variant::AVX512)
    run = detail::quantRunAVX512;
  else if (r == Variant::AVX512Pf)
    run = detail::quantRunAVX512Pf;
#endif
```

`QuantKernels.h` — add to the SIMD block:

```cpp
void quantRunAVX512(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale);
void quantRunAVX512Pf(const float *src, int64_t srcStride, int8_t *dst,
                      int64_t n, float invScale);
```

- [ ] **Step 3: Implement in `libreloc/quant/QuantAVX512.cpp`**

```cpp
namespace {

// Shared body: quantize 16 gathered floats and store. `Prefetch` pulls the
// next-but-one iteration's lines (each strided element usually sits on its
// own cache line, so one prefetch per lane).
template <bool Prefetch>
void quantRunAVX512Impl(const float *src, int64_t srcStride, int8_t *dst,
                        int64_t n, float invScale) {
  using reloc::quant::detail::quantOne;
  if (srcStride == 1) {
    reloc::quant::detail::quantizePackAVX512(src, dst, n, invScale);
    return;
  }
  // i32gather indices are signed 32-bit, scaled by 4 bytes.
  assert(srcStride > 0 && srcStride <= (INT32_MAX / 16) &&
         "gather index would overflow i32");
  const __m512 vinv = _mm512_set1_ps(invScale);
  const __m512 vlo = _mm512_set1_ps(-128.0f);
  const __m512 vhi = _mm512_set1_ps(127.0f);
  const __m512i lane =
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i vidx =
      _mm512_mullo_epi32(lane, _mm512_set1_epi32(static_cast<int>(srcStride)));
  constexpr int64_t kPfDist = 32; // elements ahead = 2 vector iterations
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    if (Prefetch && i + kPfDist + 16 <= n) {
      const float *pf = src + (i + kPfDist) * srcStride;
      for (int k = 0; k < 16; ++k)
        _mm_prefetch(reinterpret_cast<const char *>(pf + k * srcStride),
                     _MM_HINT_T0);
    }
    __m512 v = _mm512_i32gather_ps(vidx, src + i * srcStride, 4);
    v = _mm512_mul_ps(v, vinv);
    v = _mm512_max_ps(v, vlo);
    v = _mm512_min_ps(v, vhi);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i),
                     _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(v)));
  }
  for (; i < n; ++i)
    dst[i] = quantOne(src[i * srcStride], invScale);
}

} // namespace
```

(place the anonymous namespace before the `reloc` namespace block, and add `#include <cstdint>` / `#include <climits>` if `INT32_MAX` needs it), then inside `namespace detail`:

```cpp
void quantRunAVX512(const float *src, int64_t srcStride, int8_t *dst,
                    int64_t n, float invScale) {
  quantRunAVX512Impl<false>(src, srcStride, dst, n, invScale);
}

void quantRunAVX512Pf(const float *src, int64_t srcStride, int8_t *dst,
                      int64_t n, float invScale) {
  quantRunAVX512Impl<true>(src, srcStride, dst, n, invScale);
}
```

Note the anonymous-namespace template refers to `detail::quantizePackAVX512` — since the anonymous namespace sits *outside* `reloc::quant`, qualify fully as shown.

- [ ] **Step 4: Build and run** — `--gtest_filter='GatherQuantize.*'`, expected: all PASS including the new SIMD case (4 plans × 2 variants).

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/quant libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 gather_quantize AVX-512 gather + prefetch variants (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: GatherPool multi-thread wrappers

Issue #74's "multi-thread wrappers via the existing D-track parallel-producer pipeline": partition over a caller-owned `GatherPool` with the pipeline's per-worker byte floor. Thread *pinning* is external (`taskset`/`numactl`) and documented in the bench driver (Task 9).

**Files:**
- Modify: `libreloc/include/reloc/Quant.h`, `libreloc/quant/Quant.cpp`, `libreloc/test/QuantTest.cpp`

**Interfaces:**
- Consumes: `reloc::GatherPool::parallelFor(begin, end, minPerWorker, fn)` (`reloc/GatherPool.h`), `reloc::kMinGatherBytesPerWorker` (`reloc/Pipeline.h`, 1 MiB), all four kernels.
- Produces (public):
  - `void quantizePackF32S8Parallel(GatherPool &pool, const float *src, int8_t *dst, int64_t channels, int64_t channelSize, const float *invScales, Variant v = Variant::Auto)`
  - `void gatherQuantizeF32S8Parallel(GatherPool &pool, const BoundPlan &bound, const float *srcBase, int8_t *dstBase, const float *invScales, Variant v = Variant::Auto)`
  - `void packS8S4Parallel(GatherPool &pool, const int8_t *src, uint8_t *dst, int64_t pairs, Variant v = Variant::Auto)`
  - `void convertF32F16Parallel(GatherPool &pool, const float *src, uint16_t *dst, int64_t count, Variant v = Variant::Auto)`

- [ ] **Step 1: Append the failing tests**

```cpp
#include "reloc/GatherPool.h"   // add to QuantTest.cpp's include block

TEST(QuantParallel, AllWrappersMatchSerial) {
  reloc::GatherPool pool(4);
  // quantize_pack: 13 channels x 1031 elements (odd split boundaries)
  {
    const int64_t ch = 13, cs = 1031;
    std::vector<float> src = randomFloats(ch * cs, 31, -300.f, 300.f);
    std::vector<float> inv(ch, 0.21f);
    std::vector<int8_t> a(ch * cs, 0), b(ch * cs, 1);
    reloc::quant::quantizePackF32S8(src.data(), a.data(), ch, cs, inv.data());
    reloc::quant::quantizePackF32S8Parallel(pool, src.data(), b.data(), ch,
                                            cs, inv.data());
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
  }
  // gather_quantize on a transpose plan
  {
    auto p = transposePlan(517, 263);
    std::vector<float> src =
        randomFloats(maxSrcOffset(p) + 1, 33, -300.f, 300.f);
    std::vector<float> inv(517);
    for (size_t c = 0; c < inv.size(); ++c)
      inv[c] = 0.02f + 0.001f * static_cast<float>(c);
    std::vector<int8_t> a(517 * 263, 0), b(517 * 263, 1);
    reloc::quant::gatherQuantizeF32S8(p, src.data(), a.data(), inv.data(), 0,
                                      p.extents[0]);
    reloc::quant::gatherQuantizeF32S8Parallel(pool, p, src.data(), b.data(),
                                              inv.data());
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), a.size()));
  }
  // pack_s8_s4
  {
    const int64_t pairs = 100003;
    std::mt19937 rng(35);
    std::vector<int8_t> src(2 * pairs);
    for (int8_t &x : src)
      x = static_cast<int8_t>(rng());
    std::vector<uint8_t> a(pairs, 0), b(pairs, 1);
    reloc::quant::packS8S4(src.data(), a.data(), pairs);
    reloc::quant::packS8S4Parallel(pool, src.data(), b.data(), pairs);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), pairs));
  }
  // convert_f32_f16
  {
    const int64_t n = (1 << 20) + 37;
    std::vector<float> src = randomFloats(n, 37, -70000.f, 70000.f);
    std::vector<uint16_t> a(n, 0), b(n, 1);
    reloc::quant::convertF32F16(src.data(), a.data(), n);
    reloc::quant::convertF32F16Parallel(pool, src.data(), b.data(), n);
    ASSERT_EQ(0, std::memcmp(a.data(), b.data(), n * sizeof(uint16_t)));
  }
  pool.close();
}

TEST(QuantParallel, NonDisjointDstRowsFallBackInline) {
  // dstStrides[0] < inner span: outer rows alias in dst, so the parallel
  // wrapper must serialize (same guard as executeH2DThreaded) and still
  // produce exactly the serial result.
  reloc::BoundPlan b;
  b.extents = {6, 8};
  b.srcStrides = {8, 1};
  b.dstStrides = {4, 1}; // rows overlap: span 7 >= stride 4
  b.elementSize = 4;
  b.totalBytes = (5 * 4 + 7 + 1) * 4;
  std::vector<float> src = randomFloats(maxSrcOffset(b) + 1, 41, -10.f, 10.f);
  std::vector<float> inv(6, 1.0f);
  const size_t total = 5 * 4 + 7 + 1;
  std::vector<int8_t> a(total, 0), c(total, 1);
  reloc::quant::gatherQuantizeF32S8(b, src.data(), a.data(), inv.data(), 0, 6,
                                    Variant::Scalar);
  reloc::GatherPool pool(4);
  reloc::quant::gatherQuantizeF32S8Parallel(pool, b, src.data(), c.data(),
                                            inv.data(), Variant::Scalar);
  pool.close();
  EXPECT_EQ(0, std::memcmp(a.data(), c.data(), total));
}
```

- [ ] **Step 2: Run to verify failure** — `'quantizePackF32S8Parallel' is not a member of 'reloc::quant'`.

- [ ] **Step 3: Implement**

Header — after the four kernels:

```cpp
/// Multi-thread wrappers over a caller-owned GatherPool (D-track parallel
/// producer, issue #65): partition with the pipeline's per-worker byte
/// floor (kMinGatherBytesPerWorker) so small inputs stay inline. Output is
/// byte-identical to the serial kernel. Thread PINNING is the caller's job
/// (taskset/numactl); the pool does not pin.
void quantizePackF32S8Parallel(GatherPool &pool, const float *src,
                               int8_t *dst, int64_t channels,
                               int64_t channelSize, const float *invScales,
                               Variant v = Variant::Auto);
void gatherQuantizeF32S8Parallel(GatherPool &pool, const BoundPlan &bound,
                                 const float *srcBase, int8_t *dstBase,
                                 const float *invScales,
                                 Variant v = Variant::Auto);
void packS8S4Parallel(GatherPool &pool, const int8_t *src, uint8_t *dst,
                      int64_t pairs, Variant v = Variant::Auto);
void convertF32F16Parallel(GatherPool &pool, const float *src, uint16_t *dst,
                           int64_t count, Variant v = Variant::Auto);
```

`Quant.cpp` — add includes `"reloc/GatherPool.h"` and `"reloc/Pipeline.h"` and `<algorithm>`, then:

```cpp
namespace {

// Per-worker floor in outer units, from the pipeline's byte floor.
int64_t minPerWorker(int64_t bytesPerUnit) {
  return std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, bytesPerUnit));
}

} // namespace
```

(merge into the existing anonymous namespace holding `quantWalk`), and the wrappers inside `namespace quant`:

```cpp
void quantizePackF32S8Parallel(GatherPool &pool, const float *src,
                               int8_t *dst, int64_t channels,
                               int64_t channelSize, const float *invScales,
                               Variant v) {
  pool.parallelFor(0, channels, minPerWorker(channelSize * 4),
                   [&](int64_t cb, int64_t ce) {
                     quantizePackF32S8(src + cb * channelSize,
                                       dst + cb * channelSize, ce - cb,
                                       channelSize, invScales + cb, v);
                   });
}

void gatherQuantizeF32S8Parallel(GatherPool &pool, const BoundPlan &bound,
                                 const float *srcBase, int8_t *dstBase,
                                 const float *invScales, Variant v) {
  // Outer rows must write disjoint dst bytes to split across workers --
  // the same conservative guard as executeH2DThreaded (Execute.cpp).
  int64_t innerSpan = 0;
  for (size_t k = 1; k < bound.dstStrides.size(); ++k)
    innerSpan += (bound.extents[k] - 1) * bound.dstStrides[k];
  if (bound.dstStrides[0] < innerSpan + 1) {
    gatherQuantizeF32S8(bound, srcBase, dstBase, invScales, 0,
                        bound.extents[0], v);
    return;
  }
  int64_t rowElems = 1;
  for (size_t k = 1; k < bound.extents.size(); ++k)
    rowElems *= bound.extents[k];
  pool.parallelFor(0, bound.extents[0], minPerWorker(rowElems * 4),
                   [&](int64_t rb, int64_t re) {
                     gatherQuantizeF32S8(bound, srcBase, dstBase, invScales,
                                         rb, re, v);
                   });
}

void packS8S4Parallel(GatherPool &pool, const int8_t *src, uint8_t *dst,
                      int64_t pairs, Variant v) {
  pool.parallelFor(0, pairs, minPerWorker(2), [&](int64_t pb, int64_t pe) {
    packS8S4(src + 2 * pb, dst + pb, pe - pb, v);
  });
}

void convertF32F16Parallel(GatherPool &pool, const float *src, uint16_t *dst,
                           int64_t count, Variant v) {
  pool.parallelFor(0, count, minPerWorker(4), [&](int64_t b, int64_t e) {
    convertF32F16(src + b, dst + b, e - b, v);
  });
}
```

- [ ] **Step 4: Build and run** — `--gtest_filter='QuantParallel.*'`, expected: 2 tests PASS. Then run the whole suite once: `/home/jueonpark/sym/build/sym/libreloc/test/libreloc-test` — expected: **all** tests PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
cd /home/jueonpark/sym
git add libreloc/include/reloc/Quant.h libreloc/quant/Quant.cpp libreloc/test/QuantTest.cpp
git commit -m "update(libreloc): R0.1 GatherPool parallel wrappers for quant kernels (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: `bench-quant-bw` driver, smoke test, README surface entry

Per-kernel bandwidth measurement through `bench/protocol.h` — kernel × variant × thread count per invocation, plus the `gather_f32` baseline (existing `gatherChunk`, "keep as-is" per issue #74). This produces the "measure, don't assume" numbers for `gather_quantize` and feeds R1's stage rooflines.

**Files:**
- Create: `bench/quant_bw.cpp`
- Modify: `bench/CMakeLists.txt`
- Modify: `libreloc/README.md`

**Interfaces:**
- Consumes: `bench/protocol.h` (`bench::runOnce`, `bench::analyzeReruns`, `bench::Series`, `bench::seriesToJson`, `bench::jsonNumber`, `kWarmupIters`/`kTimedIters`/`kReruns`), `bench/reference_plan.h` (`bench::referencePlanBytes`), `reloc::decodePlan`, `reloc::bind`, all Task 1–8 APIs, `reloc::gatherChunk`, `reloc::executeH2D`, `reloc::GatherPool`, `reloc::kMinGatherBytesPerWorker`.
- Produces: `bench-quant-bw` executable; JSON schema `{"config": {...}, "kernels": {"<name>": {"variant", "in_bytes", "out_bytes", "wall_ms", "in_gb_per_s", "out_gb_per_s"}}}`; ctest `bench-quant-bw-smoke`.

- [ ] **Step 1: Write the driver**

Create `bench/quant_bw.cpp`:

```cpp
//===- quant_bw.cpp - R0.1 quant-kernel bandwidth micro-benchmark ---------===//
//
// Issue #74's exit measurement and the R1 stage-roofline feeder: bandwidth
// of each libreloc/quant kernel (snake_case reporting names from the issue)
// plus the gather_f32 baseline (existing gatherChunk), at a chosen SIMD
// variant and GatherPool thread count, through bench/protocol.h. Every
// timed configuration is verified byte-exact against the serial scalar
// path first (a wrong benchmark is worse than none).
//
// Thread pinning is external by design (the pool does not pin):
//   taskset -c 0-7 ./bench-quant-bw --kernel all --threads 8 ...
// Pin to distinct physical cores; avoid SMT sibling pairs for T <= 4.
//
//===----------------------------------------------------------------------===//

#include "protocol.h"
#include "reference_plan.h"

#include "reloc/Bind.h"
#include "reloc/Decode.h"
#include "reloc/Execute.h"
#include "reloc/GatherPool.h"
#include "reloc/Pipeline.h"
#include "reloc/Quant.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

using reloc::quant::Kernel;
using reloc::quant::Variant;

const char *variantName(Variant v) {
  switch (v) {
  case Variant::Auto:
    return "auto";
  case Variant::Scalar:
    return "scalar";
  case Variant::AVX2:
    return "avx2";
  case Variant::AVX512:
    return "avx512";
  case Variant::AVX512Pf:
    return "avx512pf";
  }
  return "?";
}

bool parseVariant(const std::string &s, Variant &out) {
  for (Variant v : {Variant::Auto, Variant::Scalar, Variant::AVX2,
                    Variant::AVX512, Variant::AVX512Pf})
    if (s == variantName(v)) {
      out = v;
      return true;
    }
  return false;
}

struct Timing {
  bench::Series wall;
  std::vector<double> inGbps, outGbps;
};

template <typename Fn>
Timing timeIt(Fn &&fn, int64_t inBytes, int64_t outBytes, int warmup,
              int iters, int reruns) {
  std::vector<std::vector<double>> per;
  for (int r = 0; r < reruns; ++r) {
    bench::RerunSamples s = bench::runOnce(fn, warmup, iters);
    per.push_back(std::move(s.wall_ms));
  }
  Timing t;
  t.wall = bench::analyzeReruns(per);
  for (const bench::Stats &st : t.wall.reruns) {
    const double sec = st.median * 1e-3;
    t.inGbps.push_back(sec > 0 ? static_cast<double>(inBytes) / sec / 1e9
                               : 0.0);
    t.outGbps.push_back(sec > 0 ? static_cast<double>(outBytes) / sec / 1e9
                                : 0.0);
  }
  return t;
}

std::string vecToJson(const std::vector<double> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += bench::jsonNumber(v[i]);
  }
  return out + "]";
}

std::string entryJson(const std::string &kernel, Variant v, int64_t inB,
                      int64_t outB, const Timing &t) {
  return "    \"" + kernel + "\": {\"variant\": \"" + variantName(v) +
         "\", \"in_bytes\": " + std::to_string(inB) +
         ", \"out_bytes\": " + std::to_string(outB) +
         ", \"wall_ms\": " + bench::seriesToJson(t.wall) +
         ", \"in_gb_per_s\": " + vecToJson(t.inGbps) +
         ", \"out_gb_per_s\": " + vecToJson(t.outGbps) + "}";
}

std::vector<float> makeFloats(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (static_cast<float>((i * 131) & 0xff) - 128.0f) * 0.9f;
  return v;
}

struct Config {
  int64_t n = 4096;
  unsigned threads = 1;
  Variant variant = Variant::Auto;
  int warmup = bench::kWarmupIters;
  int iters = bench::kTimedIters;
  int reruns = bench::kReruns;
};

// Returns std::nullopt on a correctness-gate failure (caller exits 1).
std::optional<Timing> runQuantizePack(reloc::GatherPool &pool,
                                      const Config &c, int64_t &inB,
                                      int64_t &outB) {
  const int64_t ch = c.n, cs = c.n; // n x n, per-row scale
  std::vector<float> src = makeFloats(static_cast<size_t>(ch * cs));
  std::vector<float> inv(static_cast<size_t>(ch), 1.0f / 127.0f);
  std::vector<int8_t> dst(static_cast<size_t>(ch * cs), 0);
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::quantizePackF32S8(src.data(), ref.data(), ch, cs,
                                    inv.data(), Variant::Scalar);
    reloc::quant::quantizePackF32S8Parallel(pool, src.data(), dst.data(), ch,
                                            cs, inv.data(), c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = ch * cs * 4;
  outB = ch * cs;
  return timeIt(
      [&] {
        reloc::quant::quantizePackF32S8Parallel(pool, src.data(), dst.data(),
                                                ch, cs, inv.data(), c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runConvert(reloc::GatherPool &pool, const Config &c,
                                 int64_t &inB, int64_t &outB) {
  const int64_t n = c.n * c.n;
  std::vector<float> src = makeFloats(static_cast<size_t>(n));
  std::vector<uint16_t> dst(static_cast<size_t>(n), 0);
  {
    std::vector<uint16_t> ref(dst.size(), 1);
    reloc::quant::convertF32F16(src.data(), ref.data(), n, Variant::Scalar);
    reloc::quant::convertF32F16Parallel(pool, src.data(), dst.data(), n,
                                        c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size() * 2) != 0)
      return std::nullopt;
  }
  inB = n * 4;
  outB = n * 2;
  return timeIt(
      [&] {
        reloc::quant::convertF32F16Parallel(pool, src.data(), dst.data(), n,
                                            c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runPack(reloc::GatherPool &pool, const Config &c,
                              int64_t &inB, int64_t &outB) {
  const int64_t pairs = c.n * c.n / 2;
  std::vector<int8_t> src(static_cast<size_t>(2 * pairs));
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<int8_t>((i * 37) & 0xff);
  std::vector<uint8_t> dst(static_cast<size_t>(pairs), 0);
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::quant::packS8S4(src.data(), ref.data(), pairs, Variant::Scalar);
    reloc::quant::packS8S4Parallel(pool, src.data(), dst.data(), pairs,
                                   c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = 2 * pairs;
  outB = pairs;
  return timeIt(
      [&] {
        reloc::quant::packS8S4Parallel(pool, src.data(), dst.data(), pairs,
                                       c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

// The two plan-driven kernels share the golden reference plan (gather_bw's
// setup) bound at N = c.n.
struct PlanFixture {
  reloc::BoundPlan bound;
  std::vector<float> src; // fp32 elements, sized by max reachable offset
};

std::optional<PlanFixture> makePlanFixture(int64_t n) {
  std::vector<uint8_t> bytes = bench::referencePlanBytes();
  auto decoded = reloc::decodePlan(bytes.data(), bytes.size());
  auto *plan = std::get_if<reloc::RelocationPlan>(&decoded);
  if (!plan)
    return std::nullopt;
  auto boundResult = reloc::bind(*plan, {{"N", n}});
  auto *b = std::get_if<reloc::BoundPlan>(&boundResult);
  if (!b)
    return std::nullopt;
  PlanFixture f;
  f.bound = *b;
  int64_t maxOff = 0;
  for (size_t k = 0; k < f.bound.extents.size(); ++k)
    maxOff += (f.bound.extents[k] - 1) * f.bound.srcStrides[k];
  f.src = makeFloats(static_cast<size_t>(maxOff + 1));
  return f;
}

std::optional<Timing> runGatherQuantize(reloc::GatherPool &pool,
                                        const PlanFixture &f, const Config &c,
                                        int64_t &inB, int64_t &outB) {
  const reloc::BoundPlan &b = f.bound;
  std::vector<int8_t> dst(static_cast<size_t>(b.totalBytes / 4), 0);
  std::vector<float> inv(static_cast<size_t>(b.extents[0]), 1.0f / 127.0f);
  {
    std::vector<int8_t> ref(dst.size(), 1);
    reloc::quant::gatherQuantizeF32S8(b, f.src.data(), ref.data(), inv.data(),
                                      0, b.extents[0], Variant::Scalar);
    reloc::quant::gatherQuantizeF32S8Parallel(pool, b, f.src.data(),
                                              dst.data(), inv.data(),
                                              c.variant);
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = b.totalBytes; // fp32 read side
  outB = b.totalBytes / 4;
  return timeIt(
      [&] {
        reloc::quant::gatherQuantizeF32S8Parallel(pool, b, f.src.data(),
                                                  dst.data(), inv.data(),
                                                  c.variant);
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

std::optional<Timing> runGatherF32(reloc::GatherPool &pool,
                                   const PlanFixture &f, const Config &c,
                                   int64_t &inB, int64_t &outB) {
  const reloc::BoundPlan &b = f.bound;
  const auto *srcBytes = reinterpret_cast<const uint8_t *>(f.src.data());
  std::vector<uint8_t> dst(static_cast<size_t>(b.totalBytes), 0);
  const int64_t rowBytes = b.dstStrides[0] * 4;
  const int64_t minRows = std::max<int64_t>(
      1, static_cast<int64_t>(reloc::kMinGatherBytesPerWorker) /
             std::max<int64_t>(1, rowBytes));
  {
    std::vector<uint8_t> ref(dst.size(), 1);
    reloc::executeH2D(b, srcBytes, ref.data());
    pool.parallelFor(0, b.extents[0], minRows, [&](int64_t rb, int64_t re) {
      reloc::gatherChunk(b, srcBytes, dst.data(), rb, re);
    });
    if (std::memcmp(ref.data(), dst.data(), dst.size()) != 0)
      return std::nullopt;
  }
  inB = b.totalBytes;
  outB = b.totalBytes;
  return timeIt(
      [&] {
        pool.parallelFor(0, b.extents[0], minRows,
                         [&](int64_t rb, int64_t re) {
                           reloc::gatherChunk(b, srcBytes, dst.data(), rb, re);
                         });
      },
      inB, outB, c.warmup, c.iters, c.reruns);
}

const char *const kAllKernels[] = {"quantize_pack_f32_s8",
                                   "gather_quantize_f32_s8", "pack_s8_s4",
                                   "convert_f32_f16", "gather_f32"};

std::optional<Kernel> quantKernelFor(const std::string &name) {
  if (name == "quantize_pack_f32_s8")
    return Kernel::QuantizePack;
  if (name == "gather_quantize_f32_s8")
    return Kernel::GatherQuantize;
  if (name == "pack_s8_s4")
    return Kernel::PackS8S4;
  if (name == "convert_f32_f16")
    return Kernel::ConvertF32F16;
  return std::nullopt; // gather_f32: not a quant kernel, variant ignored
}

int run(const std::string &kernelArg, const Config &c, const char *jsonPath) {
  std::vector<std::string> kernels;
  if (kernelArg == "all")
    kernels.assign(std::begin(kAllKernels), std::end(kAllKernels));
  else
    kernels.push_back(kernelArg);

  // Validate the variant per requested quant kernel BEFORE any setup.
  for (const std::string &k : kernels) {
    if (auto qk = quantKernelFor(k)) {
      if (!reloc::quant::kernelHasVariant(*qk, c.variant) ||
          !reloc::quant::cpuSupports(c.variant)) {
        std::fprintf(stderr,
                     "error: variant %s not available for kernel %s on this "
                     "host\n",
                     variantName(c.variant), k.c_str());
        return 3;
      }
    }
  }

  std::optional<PlanFixture> plan;
  for (const std::string &k : kernels)
    if (k == "gather_quantize_f32_s8" || k == "gather_f32") {
      plan = makePlanFixture(c.n);
      if (!plan) {
        std::fprintf(stderr, "error: reference plan decode/bind failed\n");
        return 1;
      }
      break;
    }

  reloc::GatherPool pool(c.threads);
  std::string body;
  for (const std::string &k : kernels) {
    int64_t inB = 0, outB = 0;
    std::optional<Timing> t;
    if (k == "quantize_pack_f32_s8")
      t = runQuantizePack(pool, c, inB, outB);
    else if (k == "convert_f32_f16")
      t = runConvert(pool, c, inB, outB);
    else if (k == "pack_s8_s4")
      t = runPack(pool, c, inB, outB);
    else if (k == "gather_quantize_f32_s8")
      t = runGatherQuantize(pool, *plan, c, inB, outB);
    else if (k == "gather_f32")
      t = runGatherF32(pool, *plan, c, inB, outB);
    else {
      std::fprintf(stderr, "error: unknown kernel %s\n", k.c_str());
      return 2;
    }
    if (!t) {
      std::fprintf(stderr, "error: %s mismatch vs serial scalar reference\n",
                   k.c_str());
      return 1;
    }
    if (!body.empty())
      body += ",\n";
    body += entryJson(k, c.variant, inB, outB, *t);
    std::fprintf(stderr,
                 "quant_bw: %-24s %8s T=%d in %.2f GB/s out %.2f GB/s "
                 "(spread %.2f%%)\n",
                 k.c_str(), variantName(c.variant), pool.threadCount(),
                 t->inGbps.front(), t->outGbps.front(),
                 t->wall.medianSpreadPct);
  }

  const std::string doc =
      "{\n  \"config\": {\"benchmark\": \"quant_bw\", \"N\": " +
      std::to_string(c.n) +
      ", \"threads\": " + std::to_string(pool.threadCount()) +
      ", \"variant\": \"" + variantName(c.variant) +
      "\", \"warmup\": " + std::to_string(c.warmup) +
      ", \"iters\": " + std::to_string(c.iters) +
      ", \"reruns\": " + std::to_string(c.reruns) +
      "},\n  \"kernels\": {\n" + body + "\n  }\n}\n";
  pool.close();
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
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string kernel = "all";
  Config c;
  const char *jsonPath = "-";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--kernel")
      kernel = next();
    else if (a == "--n")
      c.n = std::atoll(next());
    else if (a == "--threads")
      c.threads = static_cast<unsigned>(std::atoi(next()));
    else if (a == "--variant") {
      if (!parseVariant(next(), c.variant)) {
        std::fprintf(stderr, "error: bad --variant (auto|scalar|avx2|avx512|"
                             "avx512pf)\n");
        return 2;
      }
    } else if (a == "--json")
      jsonPath = next();
    else if (a == "--warmup")
      c.warmup = std::atoi(next());
    else if (a == "--iters")
      c.iters = std::atoi(next());
    else if (a == "--reruns")
      c.reruns = std::atoi(next());
    else {
      std::fprintf(stderr,
                   "usage: bench-quant-bw [--kernel NAME|all] [--n N] "
                   "[--threads T] [--variant V] [--json PATH|-] [--warmup W] "
                   "[--iters I] [--reruns R]\n");
      return 2;
    }
  }
  if (c.n <= 0 || c.n % 64 != 0) {
    std::fprintf(stderr,
                 "error: N must be positive and divisible by 64 (got %lld)\n",
                 static_cast<long long>(c.n));
    return 2;
  }
  if (c.warmup < 0 || c.iters < 1 || c.reruns < 1) {
    std::fprintf(stderr, "error: bad warmup/iters/reruns\n");
    return 2;
  }
  return run(kernel, c, jsonPath);
}
```

- [ ] **Step 2: CMake target + smoke test**

Append to `bench/CMakeLists.txt` after the `bench-gather-bw` block:

```cmake
add_executable(bench-quant-bw quant_bw.cpp)
target_include_directories(bench-quant-bw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(bench-quant-bw PRIVATE cxx_std_17)
target_link_libraries(bench-quant-bw PRIVATE reloc_runtime)
# Smoke-run under ctest (same rationale as the other bench smokes): proves
# every R0.1 kernel runs end-to-end -- including the byte-exactness gate
# against the serial scalar path -- on every CI run, at whatever SIMD tier
# the CI host supports (variant=auto). Tiny counts; not a measurement.
add_test(NAME bench-quant-bw-smoke
         COMMAND bench-quant-bw --kernel all --n 256 --threads 2
                 --variant auto --json - --warmup 1 --iters 3 --reruns 2)
```

- [ ] **Step 3: Build and run the smoke test**

```bash
ninja -C /home/jueonpark/sym/build/sym bench-quant-bw && \
ctest --test-dir /home/jueonpark/sym/build/sym -R bench-quant-bw-smoke --output-on-failure
```

Expected: PASS, with five stderr summary lines (one per kernel) and a JSON doc on stdout.

- [ ] **Step 4: Sanity measurement on this machine (Release build)**

```bash
cmake -S /home/jueonpark/sym -B /home/jueonpark/sym/build/cuda-release >/dev/null && \
ninja -C /home/jueonpark/sym/build/cuda-release bench-quant-bw && \
taskset -c 0-7 /home/jueonpark/sym/build/cuda-release/bench/bench-quant-bw \
  --kernel all --n 8192 --threads 8 --variant auto \
  --json /home/jueonpark/sym/bench/results/quant_bw_n8192_t8.json
```

Expected: exit 0; `gather_quantize_f32_s8` input-side GB/s in the same ballpark as `gather_f32` (the issue's ~8.6–14 GB/s known range — if it is wildly below `gather_f32`, flag it in the PR, don't tune silently). Also run `--variant scalar` and `--variant avx512` at `--threads 1` for the variant spread, and save those JSONs too (`quant_bw_n8192_t1_scalar.json`, `quant_bw_n8192_t1_avx512.json`).

- [ ] **Step 5: README surface entry**

In `libreloc/README.md`, append a bullet to the `## Surface` list (after the `reloc::GatherPool` entry):

```markdown
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
```

- [ ] **Step 6: Commit**

```bash
cd /home/jueonpark/sym
git add bench/quant_bw.cpp bench/CMakeLists.txt libreloc/README.md \
        bench/results/quant_bw_n8192_t8.json \
        bench/results/quant_bw_n8192_t1_scalar.json \
        bench/results/quant_bw_n8192_t1_avx512.json
git commit -m "update(bench, libreloc): R0.1 quant-kernel bandwidth driver + smoke (#74)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Full verification and branch finish

**Files:** none new.

**Interfaces:** consumes everything above.

- [ ] **Step 1: Full test suite**

```bash
ninja -C /home/jueonpark/sym/build/sym && \
ctest --test-dir /home/jueonpark/sym/build/sym --output-on-failure
```

Expected: **all** tests PASS, including `libreloc-test`, `reloc-runtime-mlir-free`, `reloc-runtime-no-mlir-includes`, `bench-quant-bw-smoke`, and every pre-existing test.

- [ ] **Step 2: Verify the runtime still exports no MLIR/LLVM symbols and loads standalone**

```bash
ctest --test-dir /home/jueonpark/sym/build/sym -R reloc-runtime --output-on-failure
```

Expected: PASS ×2.

- [ ] **Step 3: Use superpowers:verification-before-completion**, then **superpowers:finishing-a-development-branch** — target: PR titled `update(libreloc, bench): R0.1 CPU quant kernels (#74)` whose body includes the measured `quant_bw` numbers from Task 9 Step 4 (honest-reporting style, like `docs/poc-reproduction.md`) and `Closes #74`, plus a comment on issue #73 noting R0.1 is done and linking the PR.

---

## Self-Review (completed at plan time)

1. **Spec coverage vs issue #74:** `quantize_pack_f32_s8` scalar/AVX2/AVX-512 → Tasks 2–3. `gather_quantize_f32_s8` scalar/AVX-512-gather/prefetch+tiled → Tasks 6–7. `pack_s8_s4` AVX-512 (+ scalar reference) → Task 5. `convert_f32_f16` F16C/AVX-512 (+ scalar) → Task 4. `gather_f32` "keep as-is" → no kernel work; baseline entry in the bench (Task 9). Multi-thread wrappers via D-track pool → Task 8. Pinning/SMT guidance → documented in the driver header + README (Task 9), external by design since `GatherPool` does not pin. AVX2-only-host caveat → runtime dispatch (Task 1/3). "~14 GB/s bound — measure, don't assume" → Task 9 Step 4. VNNI: listed in the issue as a CPU-capability parenthetical; no quant kernel here computes dot products, so no VNNI intrinsics are used — noted so a reviewer doesn't look for them.
2. **Placeholder scan:** every code step carries full code; no TBDs.
3. **Type consistency:** `Variant`/`Kernel`/`resolveFor`/`kernelHasVariant` (Task 1) used identically in Tasks 2–9; `QuantRunFn` signature identical in Tasks 6–7; parallel wrapper signatures identical between Task 8 header/impl/tests and Task 9 bench calls.
