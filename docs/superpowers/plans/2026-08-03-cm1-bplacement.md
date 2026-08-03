# CM1 — BPlacement Term + recv-kernel m Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #109 (CM1): a `BPlacement` (Serial/Overlapped) parameter on `reloc::costmodel`'s B-path formula that kills the r=1 slope-tie by construction, plus `recv.m.*` calibration keys derived R4-style, delivered as a draft PR (Gen4 measurement deferred to a runbook).

**Architecture:** The spec is `docs/superpowers/specs/2026-07-31-cm1-bplacement-design.md` — read it first. Three layers: (1) C++ core — `BPlacement` enum in `MethodDecision.h`, trailing default-`Overlapped` parameter on `pathCosts`/`decide`, Serial = sum of link+HBM slopes, Overlapped = existing max plus a slope-folded fill/drain term gated on a calibration key; (2) pyreloc `b_placement` kwarg; (3) calibration — recv kernels added to `bench/rtrack/hiding_ratio.cu`, one Gen3 run on this box, `make_calibration.py` emit blocks, both `.cal` files regenerated additively.

**Tech Stack:** C++17 (gtest), CUDA (sm_75, standalone nvcc build on this box), pybind11 (pyreloc), Python 3 (make_calibration.py, pytest).

## Global Constraints

- Every cost term must stay affine in S (intercept + slope·S) — the single-boundary `thresholdBytes` model depends on it (`CostModel.cpp:251-256` documents this invariant).
- All existing tests must keep passing with **zero fixture edits** — the new parameter defaults to `Overlapped`, and the fill/drain term is 0 when `pipeline.chunks_per_buffer` is absent (issue test iii).
- Existing `.cal` lines must not change — new keys are pure insertions ("regenerate byte-identically except the new keys"). Never touch `bench/results/r4_hiding_ratio_epyc_2080ti.json` or `bench/results/v3_prediction_report.json`.
- Method A's cost form does not change (out of scope per #107). `recv.m.*` keys are landed but NOT consumed by `pathCosts` (deferred to CM5).
- CPU tests build: `ninja -C build/sym libreloc-test` (Release, RELOC_ENABLE_CUDA=OFF); test binary `build/sym/libreloc/test/libreloc-test`. Python: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q`.
- GPU tool builds are standalone nvcc on this box (no CUDA cmake tree): `/usr/local/cuda-12.5/bin/nvcc -ccbin g++` (plain gcc has a broken cc1plus), `-arch=sm_75`.
- clang-format clean on touched C++ files (`pip install clang-format` in a venv if the binary is missing — no system clang-format on this box).
- Branch: `cm1-bplacement` (already exists, spec committed). Final deliverable is a **draft** PR.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: BPlacement enum + Serial formula in pathCosts/decide

**Files:**
- Modify: `libreloc/include/reloc/MethodDecision.h` (enum + field + name fn)
- Modify: `libreloc/include/reloc/CostModel.h` (signatures + header comment)
- Modify: `libreloc/src/CostModel.cpp` (formula + name fn + forwarding)
- Test: `libreloc/test/CostModelTest.cpp`

**Interfaces:**
- Consumes: existing `pathCosts`/`decide` (`CostModel.h:71-79`), `PathCosts` struct, `kSynth` fixture (`CostModelTest.cpp:102-118`).
- Produces: `enum class BPlacement { Serial, Overlapped }` and `const char *placementName(BPlacement)` in `reloc::costmodel` (MethodDecision.h); `MethodDecision::bPlacement` field (default `Overlapped`); new trailing parameters `pathCosts(..., bool broadcast = false, BPlacement bPlace = BPlacement::Overlapped)` and `decide(..., bool broadcast = false, BPlacement bPlace = BPlacement::Overlapped)`. Tasks 2–4 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Append to `libreloc/test/CostModelTest.cpp` (after `CostModelPathCosts.AffineFormsAndK`; add `using reloc::costmodel::BPlacement;` next to the other usings at line 76-79):

```cpp
TEST(CostModelPathCosts, SerialPlacementSumsLinkAndHbm) {
  // Issue #109: Serial B (b_fair) pays transfer THEN kernel -- slopes
  // add. kSynth contiguous r=0.25, S=1e9: overlapped B slope
  // max(1/10, 1/100) -> 100 ms; serial 1/10 + 1/100 -> 110 ms.
  CostModel m = mustParse(kSynth);
  const int64_t S = 1000000000;
  auto ov = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 1, false,
                      BPlacement::Overlapped);
  auto se = pathCosts(m, Pattern::Contiguous, S, 0.25, 8, 1, false,
                      BPlacement::Serial);
  ASSERT_TRUE(ov.has_value());
  ASSERT_TRUE(se.has_value());
  EXPECT_NEAR(ov->tBMs, 0.1 + 100.0, 1e-9);
  EXPECT_NEAR(se->tBMs, 0.1 + 110.0, 1e-9);
  // Placement touches only the B side.
  EXPECT_DOUBLE_EQ(se->tAMs, ov->tAMs);
  EXPECT_DOUBLE_EQ(se->aSlopeMsPerByte, ov->aSlopeMsPerByte);
}

TEST(CostModelDecide, SerialKillsRoneSlopeTie) {
  // Issue #109 test (i) / docs/v3-costmodel.md S4: at r=1.0 the
  // overlapped model's A and B DMA slopes are algebraically identical
  // whenever A is DMA-bound, so the decision collapses to intercept
  // noise. Serial's slope a+b can never tie A's for any K/broadcast.
  CostModel m = mustParse(kSynth);
  // The canonical degenerate cell: K=1 contiguous r=1 (A DMA-bound:
  // dma 1e-7 > cpu 5e-8). Overlapped ties exactly...
  auto ov = pathCosts(m, Pattern::Contiguous, 1 << 20, 1.0, 8, 1, false,
                      BPlacement::Overlapped);
  ASSERT_TRUE(ov.has_value());
  EXPECT_DOUBLE_EQ(ov->bSlopeMsPerByte, ov->aSlopeMsPerByte);
  auto dOv = decide(m, Pattern::Contiguous, 1 << 20, 1.0, 8, 1, -1, false,
                    BPlacement::Overlapped);
  ASSERT_TRUE(dOv.has_value());
  EXPECT_DOUBLE_EQ(dOv->thresholdBytes, -1); // parallel lines: no boundary
  // ...and Serial breaks the tie with a real, finite boundary:
  // bSlope = 1/10+1/100 = 1.1e-7 > aSlope 1e-7;
  // S* = (0.1-0.5)/(1e-7-1.1e-7) = 4e7.
  auto dSe = decide(m, Pattern::Contiguous, 1 << 20, 1.0, 8, 1, -1, false,
                    BPlacement::Serial);
  ASSERT_TRUE(dSe.has_value());
  EXPECT_NEAR(dSe->thresholdBytes, 4e7, 1.0);
  // No (K, broadcast) combination ties under Serial at r=1.
  for (int K : {1, 4}) {
    for (bool bc : {false, true}) {
      auto pc = pathCosts(m, Pattern::Contiguous, 1 << 20, 1.0, 8, K, bc,
                          BPlacement::Serial);
      ASSERT_TRUE(pc.has_value()) << "K=" << K << " bc=" << bc;
      EXPECT_NE(pc->bSlopeMsPerByte, pc->aSlopeMsPerByte)
          << "K=" << K << " bc=" << bc;
    }
  }
}

TEST(CostModelDecide, PlacementRecordedOnDecision) {
  CostModel m = mustParse(kSynth);
  auto dDefault = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8);
  ASSERT_TRUE(dDefault.has_value());
  EXPECT_EQ(dDefault->bPlacement, BPlacement::Overlapped);
  auto dSe = decide(m, Pattern::Contiguous, 1 << 20, 0.25, 8, 1, -1, false,
                    BPlacement::Serial);
  ASSERT_TRUE(dSe.has_value());
  EXPECT_EQ(dSe->bPlacement, BPlacement::Serial);
  EXPECT_STREQ(placementName(BPlacement::Serial), "serial");
  EXPECT_STREQ(placementName(BPlacement::Overlapped), "overlapped");
}
```

Also add `using reloc::costmodel::placementName;` beside the `methodName` using at line 167-169.

- [ ] **Step 2: Run tests to verify they fail**

Run: `ninja -C build/sym libreloc-test 2>&1 | tail -5`
Expected: compile FAILURE — `BPlacement` not declared. (A compile failure is this step's "red".)

- [ ] **Step 3: Implement**

`libreloc/include/reloc/MethodDecision.h` — after the `Pattern` block (line 27), add:

```cpp
/// Which B implementation the decision prices (issue #109/CM1).
/// Serial (b_fair): the receive kernel runs after the whole transfer,
/// so link and HBM slopes ADD. Overlapped (b_pipelined): chunked
/// double-buffering hides the smaller stage under the larger -- max,
/// plus a fill/drain term when the calibration carries
/// pipeline.chunks_per_buffer.
enum class BPlacement { Serial, Overlapped };

const char *placementName(BPlacement p);
```

In `struct MethodDecision` (after `nReuse`, line 37) add:

```cpp
  BPlacement bPlacement = BPlacement::Overlapped;
```

`libreloc/include/reloc/CostModel.h` — change the two signatures (lines 71-79) to:

```cpp
std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p,
                                   int64_t srcBytes, double r, int threads,
                                   int K = 1, bool broadcast = false,
                                   BPlacement bPlace = BPlacement::Overlapped);

std::optional<MethodDecision>
decide(const CostModel &m, Pattern p, int64_t srcBytes, double r,
       int threads = 8, int K = 1, int64_t nReuse = -1, bool broadcast = false,
       BPlacement bPlace = BPlacement::Overlapped);
```

Update the header-top formula comment (lines 6-7) to:

```
//   T_A = overhead.a_ms + S * max(1/BW_cpu(pattern), wireBytes/S/BW_link)
//   T_B(Overlapped) = overhead.b_ms + S * (max(1/BW_link, m/BW_hbm)
//                     + min(1/BW_link, m/BW_hbm)/n)   [n = chunks/buffer, 0 if uncalibrated]
//   T_B(Serial)     = overhead.b_ms + S * (1/BW_link + m/BW_hbm)
```

`libreloc/src/CostModel.cpp`:

1. Add `placementName` next to `patternName` (line 98):

```cpp
const char *placementName(BPlacement p) {
  switch (p) {
  case BPlacement::Serial:
    return "serial";
  case BPlacement::Overlapped:
    return "overlapped";
  }
  return "?";
}
```

2. `pathCosts` — add the `BPlacement bPlace` parameter and replace the B block (lines 189-193) with:

```cpp
  // B: DMA of kMult*S vs GPU transform m*kMult*S over HBM (issue #109).
  const double bDmaSlope = kMult * msPerByteAt(*bwDel);
  const double bHbmSlope = kMult * mm * msPerByteAt(bwHbm);
  if (bPlace == BPlacement::Serial) {
    // b_fair: the kernel starts only after the whole transfer landed --
    // the stages ADD, so B's slope strictly exceeds the bare DMA slope
    // for any m > 0 and the r=1 A/B slope tie (v3-costmodel.md S4) is
    // dead by construction.
    pc.bSlopeMsPerByte = bDmaSlope + bHbmSlope;
  } else {
    // b_pipelined: chunks overlap; the hidden stage still pays one
    // chunk of fill/drain. chunk = S/n in ChunkSchedule's mid-range,
    // so the term folds into the slope (min/n) and stays affine in S.
    // n absent or <= 0 -> term 0 -> exactly the V3 formula.
    const double nChunks = m.get("pipeline.chunks_per_buffer", 0.0);
    const double fillDrain =
        nChunks > 0 ? std::min(bDmaSlope, bHbmSlope) / nChunks : 0.0;
    pc.bSlopeMsPerByte = std::max(bDmaSlope, bHbmSlope) + fillDrain;
  }
  pc.bInterceptMs = m.get("overhead.b_ms", 0.0);
```

3. `decide` — add the trailing `BPlacement bPlace` parameter, forward it (`pathCosts(m, p, srcBytes, r, threads, K, broadcast, bPlace)` at line 214), and set `d.bPlacement = bPlace;` next to `d.k = K;` (line 220).

- [ ] **Step 4: Run tests to verify they pass**

Run: `ninja -C build/sym libreloc-test && build/sym/libreloc/test/libreloc-test --gtest_filter='CostModel*'`
Expected: ALL PASS, including every pre-existing `CostModel*` test (the default-parameter path is byte-identical V3 behavior).

- [ ] **Step 5: Run the full suite**

Run: `build/sym/libreloc/test/libreloc-test 2>&1 | tail -3`
Expected: all tests pass (152+ plus the 3 new; 2 skips are normal).

- [ ] **Step 6: Commit**

```bash
git add libreloc/include/reloc/MethodDecision.h libreloc/include/reloc/CostModel.h \
        libreloc/src/CostModel.cpp libreloc/test/CostModelTest.cpp
git commit -m "feat(libreloc): BPlacement Serial/Overlapped term in pathCosts/decide (#109)"
```

---

### Task 2: Overlapped fill/drain term test (pipeline.chunks_per_buffer)

**Files:**
- Test: `libreloc/test/CostModelTest.cpp`
- (Implementation already landed in Task 1's B block — this task pins its arithmetic and the absent-key reduction with a dedicated test. If Task 1 was done correctly this test passes immediately; it still gates the exact numbers.)

**Interfaces:**
- Consumes: `pathCosts(..., BPlacement)` from Task 1; `kSynth`.
- Produces: nothing new — a regression test for issue test (iii)'s fill/drain half.

- [ ] **Step 1: Write the test**

Append to `libreloc/test/CostModelTest.cpp`:

```cpp
TEST(CostModelPathCosts, FillDrainTermFromChunksKey) {
  // Issue #109 test (iii): with pipeline.chunks_per_buffer the
  // Overlapped slope gains exactly min(a,b)/n; without the key the
  // formula is byte-for-byte V3's max -- which is also the R4 hiding
  // condition's limit (m/BW_hbm <= 1/BW_link => max picks DMA).
  CostModel base = mustParse(kSynth);
  CostModel withN =
      mustParse(std::string(kSynth) + "pipeline.chunks_per_buffer 8\n");
  const int64_t S = 1000000000;
  auto pcBase = pathCosts(base, Pattern::Contiguous, S, 0.25, 8, 1, false,
                          BPlacement::Overlapped);
  auto pcN = pathCosts(withN, Pattern::Contiguous, S, 0.25, 8, 1, false,
                       BPlacement::Overlapped);
  ASSERT_TRUE(pcBase.has_value());
  ASSERT_TRUE(pcN.has_value());
  // kSynth contiguous: a = 1e-7 (link 10), b = 1e-8 (m=1, HBM 100).
  EXPECT_DOUBLE_EQ(pcBase->bSlopeMsPerByte, 1e-7); // V3 exactly
  EXPECT_NEAR(pcN->bSlopeMsPerByte, 1e-7 + 1e-8 / 8.0, 1e-18);
  EXPECT_NEAR(pcN->tBMs, 0.1 + 101.25, 1e-9);
  // Serial ignores the chunks key (no pipeline to fill/drain).
  auto seN = pathCosts(withN, Pattern::Contiguous, S, 0.25, 8, 1, false,
                       BPlacement::Serial);
  ASSERT_TRUE(seN.has_value());
  EXPECT_NEAR(seN->tBMs, 0.1 + 110.0, 1e-9);
}
```

- [ ] **Step 2: Run it**

Run: `ninja -C build/sym libreloc-test && build/sym/libreloc/test/libreloc-test --gtest_filter='*FillDrain*'`
Expected: PASS (Task 1 implemented the term). If it FAILS, the Task 1 fill/drain arithmetic is wrong — fix `CostModel.cpp`, not the test.

- [ ] **Step 3: Commit**

```bash
git add libreloc/test/CostModelTest.cpp
git commit -m "test(libreloc): pin Overlapped fill/drain arithmetic + absent-key V3 reduction (#109)"
```

---

### Task 3: Brute-force agreement over both placements + bind() passes Overlapped explicitly

**Files:**
- Modify: `libreloc/test/CostModelTest.cpp:324-368` (`ThresholdAgreesWithBruteForce`)
- Modify: `libreloc/src/Bind.cpp:342-348` (step 8)
- Test: `libreloc/test/BindTest.cpp:361-380`

**Interfaces:**
- Consumes: `decide(..., BPlacement)`, `pathCosts(..., BPlacement)`, `MethodDecision::bPlacement`, `BPlacement` (Task 1).
- Produces: `bind()` records `decision->bPlacement == Overlapped`. No signature changes.

- [ ] **Step 1: Parameterize the brute-force test (issue test ii)**

In `CostModelTest.cpp`, wrap the body of `ThresholdAgreesWithBruteForce` in a placement loop. The existing `for (double r : ...)` / `for (Pattern p : ...)` nest gains an outer loop, and every `decide`/`pathCosts` call gains the trailing args. The changed test in full:

```cpp
TEST(CostModelDecide, ThresholdAgreesWithBruteForce) {
  // Issue #97 acceptance: threshold precompute vs brute-force agreement.
  // Issue #109 test (ii): retained for BOTH placements.
  CostModel m = mustParse(kSynth);
  for (BPlacement bp : {BPlacement::Overlapped, BPlacement::Serial}) {
    for (double r : {1.0, 0.5, 0.25, 0.125}) {
      for (Pattern p : {Pattern::Contiguous, Pattern::Blocked}) {
        auto probe = decide(m, p, 1 << 20, r, 8, 1, -1, false, bp);
        if (!probe.has_value())
          continue;
        const double thr = probe->thresholdBytes;
        // (comment block unchanged)
        MethodDecision::Method prevMethod = MethodDecision::Method::B;
        int64_t prevS = 0;
        bool havePrev = false;
        int flips = 0;
        for (int64_t S = 1 << 12; S <= (1ll << 34); S <<= 1) {
          auto d = decide(m, p, S, r, 8, 1, -1, false, bp);
          ASSERT_TRUE(d.has_value());
          auto pc = pathCosts(m, p, S, r, 8, 1, false, bp);
          ASSERT_TRUE(pc.has_value());
          EXPECT_EQ(d->method == MethodDecision::Method::A,
                    pc->tAMs <= pc->tBMs)
              << placementName(bp) << " " << patternName(p) << " r=" << r
              << " S=" << S;
          if (havePrev && d->method != prevMethod) {
            ++flips;
            EXPECT_LE(static_cast<double>(prevS), thr)
                << placementName(bp) << " " << patternName(p) << " r=" << r
                << " prevS=" << prevS << " thr=" << thr;
            EXPECT_LE(thr, static_cast<double>(S))
                << placementName(bp) << " " << patternName(p) << " r=" << r
                << " S=" << S << " thr=" << thr;
          }
          prevMethod = d->method;
          prevS = S;
          havePrev = true;
        }
        EXPECT_EQ(flips, thr > 0 ? 1 : 0)
            << placementName(bp) << " " << patternName(p) << " r=" << r
            << " thr=" << thr;
      }
    }
  }
}
```

Keep the existing explanatory comment block (lines 333-340) verbatim inside the loop.

- [ ] **Step 2: Run it**

Run: `ninja -C build/sym libreloc-test && build/sym/libreloc/test/libreloc-test --gtest_filter='*BruteForce*'`
Expected: PASS. Notably the Serial r=1.0 contiguous cell now has a real boundary (4e7) and flips exactly once — the Overlapped one keeps thr=-1/0 flips.

- [ ] **Step 3: Make bind() explicit + write the failing BindTest assertion**

`libreloc/src/Bind.cpp` step 8 (line 345-346) becomes:

```cpp
    if (auto d = costmodel::decide(*model, pat, bound.totalBytes, wireRatio,
                                   /*threads=*/8, K, nReuse,
                                   /*broadcast=*/false,
                                   // The library's real B is the double-
                                   // buffered pipeline (Pipeline.cpp).
                                   costmodel::BPlacement::Overlapped))
      bound.decision = *d;
```

In `BindTest.cpp`, `Bind.CostModelDecisionPopulatedWhenModelPassed` (line 374-379), add after the pattern expectation:

```cpp
  EXPECT_EQ(bound->decision->bPlacement,
            reloc::costmodel::BPlacement::Overlapped);
```

- [ ] **Step 4: Run the full suite**

Run: `build/sym/libreloc/test/libreloc-test 2>&1 | tail -3`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add libreloc/test/CostModelTest.cpp libreloc/src/Bind.cpp libreloc/test/BindTest.cpp
git commit -m "test(libreloc): brute-force threshold agreement x placement; bind passes Overlapped explicitly (#109)"
```

---

### Task 4: pyreloc b_placement kwarg

**Files:**
- Modify: `libreloc/python/PyReloc.cpp:314-349` (`predict`)
- Test: `libreloc/python/tests/test_costmodel.py`

**Interfaces:**
- Consumes: `decide(..., BPlacement)`, `placementName` (Task 1).
- Produces: `pyreloc.predict(..., b_placement="overlapped"|"serial")` kwarg (default `"overlapped"`); result dict gains `"b_placement"`. `test_prediction.py` and `test_wire_row_decision.py` call `predict` without the kwarg → outputs unchanged → the frozen V3 report stays valid.

- [ ] **Step 1: Write the failing tests**

Append to `libreloc/python/tests/test_costmodel.py`:

```python
def test_predict_b_placement_serial(cal):
    # CM1 (#109): serial B adds link+HBM slopes. Mirrors the C++
    # SerialPlacementSumsLinkAndHbm numbers: overlapped 100.1, serial 110.1.
    ov = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25)
    se = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25,
                         b_placement="serial")
    assert ov["b_placement"] == "overlapped"
    assert se["b_placement"] == "serial"
    assert ov["t_b_ms"] == pytest.approx(100.1)
    assert se["t_b_ms"] == pytest.approx(110.1)
    assert se["t_a_ms"] == pytest.approx(ov["t_a_ms"])


def test_predict_rejects_unknown_placement(cal):
    with pytest.raises(ValueError):
        pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25,
                        b_placement="pipelined")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/test_costmodel.py -q`
Expected: 2 FAIL with `TypeError: ... unexpected keyword argument 'b_placement'`.

- [ ] **Step 3: Implement**

In `PyReloc.cpp`'s `predict` lambda: add a `const std::string &bPlacement` parameter after `broadcast`; map it before the `decide` call:

```cpp
        cmns::BPlacement bp;
        if (bPlacement == "overlapped")
          bp = cmns::BPlacement::Overlapped;
        else if (bPlacement == "serial")
          bp = cmns::BPlacement::Serial;
        else
          throw py::value_error("unknown b_placement '" + bPlacement +
                                "' (expected 'overlapped' or 'serial')");
        auto d = cmns::decide(cal, p, srcBytes, r, threads, k, nReuse,
                              broadcast, bp);
```

Add to the output dict (after `out["pattern"]`):

```cpp
        out["b_placement"] = std::string(cmns::placementName(d->bPlacement));
```

Add the kwarg after `py::arg("broadcast") = false`:

```cpp
      py::arg("b_placement") = "overlapped",
```

and extend the docstring to `"... threshold_bytes, pattern, b_placement}."`.

- [ ] **Step 4: Rebuild + run all python tests**

Run: `ninja -C build/sym && PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q`
Expected: ALL PASS (171+2 skip baseline plus the 2 new). `test_prediction.py` passing here is the proof the frozen report is untouched.

- [ ] **Step 5: Commit**

```bash
git add libreloc/python/PyReloc.cpp libreloc/python/tests/test_costmodel.py
git commit -m "feat(pyreloc): b_placement kwarg on predict (#109)"
```

---

### Task 5: recv kernels in hiding_ratio.cu + Gen3 targeted run

**Files:**
- Modify: `bench/rtrack/hiding_ratio.cu` (three verified-then-timed recv entries, inserted after the `transpose_smem_padded` block at line 289, before the scatter sweep)
- Create (measured artifact): `bench/results/cm1_recv_kernel_bw_epyc_2080ti.json`

**Interfaces:**
- Consumes: `reloc::cuda::convertF16F32/dequantS8F32/unpackS4S8` (`CudaKernels.h:56-71`), the file's existing `DeviceBuf`/`timeKernel`/`timingJson`/`emit`/`download`/`clearDst` helpers.
- Produces: JSON kernel entries named `convert_f16_f32`, `dequant_s8_f32`, `unpack_dequant_s4` in the existing `by_n{kernel{...gb_per_s}}` schema. Task 6's `make_calibration.py` reads exactly these names.

- [ ] **Step 1: Add a host f16→f32 decoder** (file-local, above `runN`; exact for all finite inputs):

```cpp
// Exact host IEEE binary16 -> binary32 (finite inputs only; the bench
// forces finite bit patterns below, so the inf/NaN branch is untaken).
float f16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign; // signed zero
    } else { // subnormal: renormalize into f32
      int shift = 0;
      while (!(man & 0x400u)) {
        man <<= 1;
        ++shift;
      }
      man &= 0x3ffu;
      bits = sign | ((113u - static_cast<uint32_t>(shift)) << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (man << 13); // inf/NaN (untaken)
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
```

(112 = 127 − 15, so a normal's f32 exponent field is `exp + 112`. A subnormal renormalized with `shift` left-shifts has value `(1+frac)·2^(1−15−shift)`, so its field is `113 − shift`. **Verify against known values in Step 3's smoke test**: 0x0001 → 5.9604645e-08, 0x03ff → 6.097555e-05, 0x3c00 → 1.0, 0x7bff → 65504.0.)

- [ ] **Step 2: Add the three measured blocks** inside `runN` after the `transpose_smem_padded` emit (line 289), before the scatter section:

```cpp
  // --- Method-A receive kernels (issue #109/CM1) --------------------------
  // R4-style: isolated kernel BW on a read+write traffic basis, later
  // divided into the same run's copy_f32 ceiling by make_calibration.py.
  {
    // convert_f16_f32: read 2B + write 4B per element = 1.5*S traffic.
    const int64_t halfBytes = total * 2;
    DeviceBuf dHalf(static_cast<size_t>(halfBytes));
    std::vector<uint16_t> hHalf(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      uint16_t h = static_cast<uint16_t>((i * 2654435761ull) & 0xffff);
      if ((h & 0x7c00) == 0x7c00)
        h = static_cast<uint16_t>(h ^ 0x0400); // force finite
      hHalf[static_cast<size_t>(i)] = h;
    }
    CUDA_CHECK(cudaMemcpyAsync(dHalf.p, hHalf.data(),
                               static_cast<size_t>(halfBytes),
                               cudaMemcpyHostToDevice, stream));
    std::vector<float> ref(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i)
      ref[static_cast<size_t>(i)] = f16ToF32(hHalf[static_cast<size_t>(i)]);
    clearDst();
    reloc::cuda::convertF16F32(dHalf.as<uint16_t>(), dDst.as<float>(), total,
                               stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: convert_f16_f32 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trConv = halfBytes + S;
    emit(timingJson("convert_f16_f32", trConv,
                    timeKernel(
                        [&] {
                          reloc::cuda::convertF16F32(dHalf.as<uint16_t>(),
                                                     dDst.as<float>(), total,
                                                     stream);
                        },
                        trConv, opt.warmup, opt.iters, stream)));
  }
  {
    // dequant_s8_f32: read 1B + write 4B per element = 1.25*S traffic
    // (per-channel scales excluded by definition: n floats, negligible).
    DeviceBuf dS8(static_cast<size_t>(total));
    DeviceBuf dScales(static_cast<size_t>(n) * 4);
    std::vector<int8_t> hS8(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i)
      hS8[static_cast<size_t>(i)] = static_cast<int8_t>((i * 131) & 0xff);
    std::vector<float> hScales(static_cast<size_t>(n));
    for (int64_t c = 0; c < n; ++c)
      hScales[static_cast<size_t>(c)] = 0.25f * static_cast<float>((c & 7) + 1);
    CUDA_CHECK(cudaMemcpyAsync(dS8.p, hS8.data(), static_cast<size_t>(total),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dScales.p, hScales.data(),
                               static_cast<size_t>(n) * 4,
                               cudaMemcpyHostToDevice, stream));
    std::vector<float> ref(static_cast<size_t>(total));
    for (int64_t c = 0; c < n; ++c)
      for (int64_t j = 0; j < n; ++j)
        ref[static_cast<size_t>(c * n + j)] =
            static_cast<float>(hS8[static_cast<size_t>(c * n + j)]) *
            hScales[static_cast<size_t>(c)];
    clearDst();
    reloc::cuda::dequantS8F32(dS8.as<int8_t>(), dDst.as<float>(), n, n,
                              dScales.as<float>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: dequant_s8_f32 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trDeq = total + S;
    emit(timingJson("dequant_s8_f32", trDeq,
                    timeKernel(
                        [&] {
                          reloc::cuda::dequantS8F32(dS8.as<int8_t>(),
                                                    dDst.as<float>(), n, n,
                                                    dScales.as<float>(),
                                                    stream);
                        },
                        trDeq, opt.warmup, opt.iters, stream)));

    // unpack_dequant_s4: the r=0.125 receive CHAIN (unpackS4S8 then
    // dequantS8F32, two launches -- how rtrack_bench.cu:716-722 runs it).
    // Traffic: 0.5B read + 1B write (unpack) + 1B read + 4B write
    // (dequant) per element = 1.625*S.
    const int64_t pairs = total / 2;
    DeviceBuf dPacked(static_cast<size_t>(pairs));
    DeviceBuf dS8mid(static_cast<size_t>(total));
    std::vector<uint8_t> hPacked(static_cast<size_t>(pairs));
    for (int64_t i = 0; i < pairs; ++i)
      hPacked[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 37) & 0xff);
    CUDA_CHECK(cudaMemcpyAsync(dPacked.p, hPacked.data(),
                               static_cast<size_t>(pairs),
                               cudaMemcpyHostToDevice, stream));
    for (int64_t i = 0; i < pairs; ++i) {
      const uint8_t b = hPacked[static_cast<size_t>(i)];
      const int8_t lo =
          static_cast<int8_t>(static_cast<int8_t>(b << 4) >> 4);
      const int8_t hi = static_cast<int8_t>(static_cast<int8_t>(b) >> 4);
      const int64_t e0 = 2 * i, e1 = 2 * i + 1;
      ref[static_cast<size_t>(e0)] =
          static_cast<float>(lo) * hScales[static_cast<size_t>(e0 / n)];
      ref[static_cast<size_t>(e1)] =
          static_cast<float>(hi) * hScales[static_cast<size_t>(e1 / n)];
    }
    clearDst();
    reloc::cuda::unpackS4S8(dPacked.as<uint8_t>(), dS8mid.as<int8_t>(), pairs,
                            stream);
    reloc::cuda::dequantS8F32(dS8mid.as<int8_t>(), dDst.as<float>(), n, n,
                              dScales.as<float>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(download(dDst).data(), ref.data(),
                    static_cast<size_t>(S)) != 0) {
      std::fprintf(stderr, "VERIFY FAILED: unpack_dequant_s4 N=%lld\n",
                   static_cast<long long>(n));
      return 1;
    }
    const int64_t trS4 = pairs + total + total + S;
    emit(timingJson("unpack_dequant_s4", trS4,
                    timeKernel(
                        [&] {
                          reloc::cuda::unpackS4S8(dPacked.as<uint8_t>(),
                                                  dS8mid.as<int8_t>(), pairs,
                                                  stream);
                          reloc::cuda::dequantS8F32(dS8mid.as<int8_t>(),
                                                    dDst.as<float>(), n, n,
                                                    dScales.as<float>(),
                                                    stream);
                        },
                        trS4, opt.warmup, opt.iters, stream)));
  }
```

Also update the file-top comment's kernel list (lines 11-16) to mention the three CM1 receive entries, and add `#include <cstring>` if not present (it is — line 31).

- [ ] **Step 3: Build standalone (nvcc; no CUDA cmake tree on this box) + smoke test**

```bash
SCRATCH=build/cm1-tools && mkdir -p "$SCRATCH"   # build/ is untracked on this box
/usr/local/cuda-12.5/bin/nvcc -ccbin g++ -O3 -DNDEBUG -std=c++17 -arch=sm_75 \
  -DRELOC_ENABLE_CUDA=1 -Ilibreloc/include -Ibench \
  bench/rtrack/hiding_ratio.cu libreloc/src/*.cpp libreloc/cuda/*.cu \
  -o "$SCRATCH/bench-hiding-ratio" -Xcompiler -pthread
"$SCRATCH/bench-hiding-ratio" --n 256 --warmup 1 --iters 3 --json -
```

Expected: JSON on stdout containing all three new kernel entries with `gb_per_s > 0`, and **no `VERIFY FAILED`** — the memcmp verifies gate correctness (including the f16 decoder). If nvcc errors on unrelated quant TUs, that recipe matched `bench-poc-transpose`'s documented build; adapt by listing only needed `libreloc/src` files (Bind.cpp, Coalesce.cpp, Execute.cpp, ChunkSchedule.cpp, etc. per link errors).

- [ ] **Step 4: The Gen3 measured run** (2080 Ti, GPU0, affinity cores per M0: 4-7,20-23):

```bash
nvidia-smi -pm 1 2>/dev/null || echo "persistence mode not settable (non-root) — proceed; kernel-only timings tolerate it"
taskset -c 4-7,20-23 "$SCRATCH/bench-hiding-ratio" \
  --json bench/results/cm1_recv_kernel_bw_epyc_2080ti.json
```

Sanity-check the artifact before committing:
- `copy_f32` `gb_per_s` within ~3% of 544 (R4's ceiling) at both N.
- All three recv kernels in the 400–560 GB/s range with `iqr_over_median_pct < 5`.
- If wildly off, re-run once; if still off, STOP and report — do not commit a suspect artifact.

- [ ] **Step 5: Commit (code + artifact together)**

```bash
git add bench/rtrack/hiding_ratio.cu bench/results/cm1_recv_kernel_bw_epyc_2080ti.json
git commit -m "bench(rtrack): recv-kernel BW entries in hiding_ratio + Gen3 targeted run (#109)"
```

---

### Task 6: make_calibration.py recv.m.* + pipeline.chunks_per_buffer; regenerate both .cal

**Files:**
- Modify: `bench/rtrack/make_calibration.py` (SOURCES docstring, `build_epyc`, `build_gen4`, two new shared helpers)
- Modify (regenerated): `calibration/epyc7351-2080ti.cal`, `calibration/7800x3d-4070tis.cal`

**Interfaces:**
- Consumes: Task 5's `bench/results/cm1_recv_kernel_bw_epyc_2080ti.json`; committed `bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv` (columns `method`, `transform`, `N`, `r`, `gpu_recv_ms`); existing helpers `load_csv_rows`, `read_json_optional`, `source_bytes`, `Emitter.emit`.
- Produces: keys `recv.m.convert_f16_f32`, `recv.m.dequant_s8_f32`, `recv.m.unpack_dequant_s4`, `pipeline.chunks_per_buffer` (epyc: all four; gen4: only `pipeline.chunks_per_buffer` until the Gen4 artifact exists). Nothing in C++ reads `recv.m.*` (by design, CM5's job); `pipeline.chunks_per_buffer` is read by Task 1's code.

- [ ] **Step 1: Add the two shared helpers** (after `emit_strategy_defaults`, line 495):

```python
def emit_pipeline_defaults(e):
    """pipeline.chunks_per_buffer: the Overlapped fill/drain n (issue
    #109/CM1). Value = ChunkSchedule.h's kChunksPerBuffer; chunk = S/n in
    the clamp's mid-range, so the model's fill/drain term min(a,b)/n
    stays affine in S."""
    e.emit("pipeline.chunks_per_buffer", 8,
           "libreloc/include/reloc/ChunkSchedule.h",
           note="kChunksPerBuffer -- Overlapped fill/drain n (issue #109)")


def emit_recv_from_cm1_run(e, path, kernels):
    """recv.m.* from a CM1 targeted isolated-kernel run (issue #109),
    R4-style: m = copy_f32 / kernel BW at the largest measured N, both
    from the SAME run (session self-consistency: a different session's
    ceiling would bias every m derived against it). Artifact absent ->
    keys omitted (loud omission, the pack_s8_s4 precedent)."""
    doc = read_json_optional(path)
    if doc is None:
        return
    by_n = doc["by_n"]
    n_ref = str(max(int(k) for k in by_n))
    copy = by_n[n_ref]["copy_f32"]["gb_per_s"]
    for kern in kernels:
        bw = by_n[n_ref][kern]["gb_per_s"]
        e.emit(f"recv.m.{kern}", round(copy / bw, 2), path,
               note=f"copy_f32/{kern} at N={n_ref}, same-run ceiling")
```

- [ ] **Step 2: Wire into `build_epyc`** — insert immediately after the `hiding.ratio` emit (line 264), keeping every existing emit call untouched (byte-identical prefix):

```python
    emit_pipeline_defaults(e)

    # Recv-kernel multipliers (issue #109/CM1): Method A's post-DMA GPU
    # decompress kernels. f16/s8 derive from the committed V2 rsweep's
    # per-chunk gpu_recv_ms ("from committed artifacts where possible"):
    # traffic = r*S read + S written; min gpu_recv_ms over the chunk
    # sweep (largest chunk = least launch-diluted, closest to isolated);
    # divided into copy_f32 at the m-table N (same rule as hbm.m.*).
    rsweep_path = f"{RESULTS}/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv"
    rrows = [row for row in load_csv_rows(rsweep_path, e.machine)
             if row["method"] == "a" and row["transform"] == "quant"]
    for key, r_wire in (("recv.m.convert_f16_f32", 0.5),
                        ("recv.m.dequant_s8_f32", 0.25)):
        cells = [row for row in rrows if float(row["r"]) == r_wire]
        if not cells:
            sys.exit(f"error: no method=a quant r={r_wire} rows in "
                     f"{rsweep_path}")
        best = min(cells, key=lambda row: float(row["gpu_recv_ms"]))
        s = source_bytes(int(best["N"]))
        bw = (1.0 + r_wire) * s / (float(best["gpu_recv_ms"]) * 1e-3) / 1e9
        e.emit(key, round(copy_m / bw, 2), rsweep_path,
               note=f"copy_f32(N={n_m})/recv BW; traffic=(1+{r_wire})*S, "
                    f"min gpu_recv_ms over chunks, N={best['N']} (issue #109)")
    # s4 has no committed Gen3 measurement -> the CM1 targeted run.
    emit_recv_from_cm1_run(
        e, f"{RESULTS}/cm1_recv_kernel_bw_epyc_2080ti.json",
        ["unpack_dequant_s4"])
```

(`copy_m` and `n_m` are already in scope from the R4 block above.)

- [ ] **Step 3: Wire into `build_gen4`** — insert after the `hbm.m.tiled` proxy emit (line 392), before the nsweep overhead fit:

```python
    emit_pipeline_defaults(e)

    # Recv-kernel multipliers (issue #109/CM1): the committed Gen4
    # pipeline CSVs are too noisy for an R4-style derivation (in-pipeline
    # event timings collapse to 23-52 GB/s on some cells) and no Gen4
    # copy_f32 ceiling exists at all -- ALL THREE keys wait for the
    # targeted run (runbook: bench/rtrack/README.md, issue #109).
    emit_recv_from_cm1_run(
        e, f"{RESULTS}/cm1_recv_kernel_bw_7800x3d_4070tis.json",
        ["convert_f16_f32", "dequant_s8_f32", "unpack_dequant_s4"])
```

- [ ] **Step 4: Extend the SOURCES docstring** (module top): under `epyc7351-2080ti:` add

```
    - pipeline.chunks_per_buffer     <- libreloc/include/reloc/ChunkSchedule.h
      (kChunksPerBuffer constant; issue #109 fill/drain n)
    - recv.m.{convert_f16_f32,dequant_s8_f32}
                                     <- bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv
      (gpu_recv_ms, method=a transform=quant, min over chunk sweep, vs
      the R4 copy_f32 ceiling at the m-table N)
    - recv.m.unpack_dequant_s4       <- bench/results/cm1_recv_kernel_bw_epyc_2080ti.json
      (CM1 targeted isolated run; same-run copy_f32 ceiling)
```

and under `7800x3d-4070tis:` add

```
    - pipeline.chunks_per_buffer     <- ChunkSchedule.h (as epyc)
    - recv.m.*                       <- bench/results/cm1_recv_kernel_bw_7800x3d_4070tis.json
      when it exists (issue #109 runbook); omitted until then. The
      "relocate/transpose recv" m values from the issue's list are the
      existing hbm.m.{pattern} keys -- not duplicated under recv.m.*.
```

- [ ] **Step 5: Regenerate both files + verify additive-only**

```bash
python3 bench/rtrack/make_calibration.py --machine epyc7351-2080ti --out calibration/epyc7351-2080ti.cal
python3 bench/rtrack/make_calibration.py --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal
git diff --stat calibration/
git diff -U0 calibration/ | grep '^-' | grep -v '^---'   # MUST print nothing
```

Expected: epyc gains exactly 4 lines (`pipeline.chunks_per_buffer 8`, `recv.m.convert_f16_f32 1.05`, `recv.m.dequant_s8_f32 1.07`, `recv.m.unpack_dequant_s4 <measured>`); gen4 gains exactly 1 (`pipeline.chunks_per_buffer 8`). Zero deleted/modified lines. (1.05/1.07 = 543.01 ceiling / {519.2, 507.7} GB/s from the committed CSV's min `gpu_recv_ms` {3.10206, 2.64382} at N=16384 — if a different value appears, the derivation rule was implemented differently than pre-computed here: STOP and reconcile.)

- [ ] **Step 6: Cross-check the CSV-derived values against the targeted run** (report-only):

```bash
python3 - <<'EOF'
import json
d = json.load(open("bench/results/cm1_recv_kernel_bw_epyc_2080ti.json"))
n = str(max(int(k) for k in d["by_n"]))
copy = d["by_n"][n]["copy_f32"]["gb_per_s"]
for kern, csv_m in [("convert_f16_f32", 1.05), ("dequant_s8_f32", 1.07)]:
    run_m = copy / d["by_n"][n][kern]["gb_per_s"]
    div = abs(run_m - csv_m) / csv_m * 100
    print(f"{kern}: csv m={csv_m}  targeted-run m={run_m:.3f}  divergence={div:.1f}%")
EOF
```

Record the output for the PR notes. >5% divergence: report it prominently (keys stay CSV-sourced per the spec) — do not change the emitted values.

- [ ] **Step 7: Run the integrity gate + full test suites**

```bash
python3 bench/rtrack/v3_gate.py 2>&1 | grep -A3 'CALIBRATION-REGEN\|REPORT-REGEN'
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
```

Expected: `CALIBRATION-REGEN epyc7351-2080ti: PASS`, `CALIBRATION-REGEN 7800x3d-4070tis: PASS`, `REPORT-REGEN: PASS`; both suites green. (If `v3_gate.py` needs different CLI args, check its argparse `main()` — the two checks are `calibration_regen_check()` / `report_regen_check()`.)

- [ ] **Step 8: Commit**

```bash
git add bench/rtrack/make_calibration.py calibration/
git commit -m "feat(bench): recv.m.* + pipeline.chunks_per_buffer calibration keys (#109)"
```

---

### Task 7: Gen4 runbook, clang-format, draft PR

**Files:**
- Modify: `bench/rtrack/README.md` (runbook section)
- No other code changes.

**Interfaces:**
- Consumes: everything above.
- Produces: the draft PR.

- [ ] **Step 1: Append a runbook section to `bench/rtrack/README.md`** (near the existing make_calibration/bench-hiding-ratio docs; match the file's heading style):

```markdown
### CM1 Gen4 recv-kernel run (issue #109 runbook)

The Gen4 box has no `copy_f32` ceiling measurement and no usable committed
recv-kernel data (`gpu_recv_ms` in the pipeline CSVs is chunked/overlapped
event time and collapses on several cells), so all three `recv.m.*` keys
are omitted from `calibration/7800x3d-4070tis.cal` until this run lands.
On the 7800X3D/4070 Ti SUPER box:

1. Build `bench-hiding-ratio` (CUDA cmake tree, or the standalone recipe in
   `docs/m0-2080ti-bringup.md` with `-arch=sm_89`).
2. `bench-hiding-ratio --json bench/results/cm1_recv_kernel_bw_7800x3d_4070tis.json`
   — this also produces the first real Gen4 `copy_f32` ceiling.
3. Commit the JSON, then regenerate:
   `python3 bench/rtrack/make_calibration.py --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal`
   (the three `recv.m.*` keys appear automatically; every pre-existing
   line must stay byte-identical) and commit the `.cal`.
4. `python3 bench/rtrack/v3_gate.py` — CALIBRATION-REGEN must PASS.

Deliberately out of scope here: re-baselining the existing Gen4 `hbm.*`
proxy keys onto the new run's ceiling. That would change committed keys
("byte-identical except the new keys" would fail) and re-open V3's
as-measured verdicts — a separate decision for CM5's re-run, recorded
here so it isn't lost.
```

- [ ] **Step 2: clang-format the touched C++ files**

```bash
clang-format -i libreloc/include/reloc/MethodDecision.h libreloc/include/reloc/CostModel.h \
  libreloc/src/CostModel.cpp libreloc/src/Bind.cpp libreloc/test/CostModelTest.cpp \
  libreloc/test/BindTest.cpp libreloc/python/PyReloc.cpp bench/rtrack/hiding_ratio.cu
git diff --stat   # if formatting changed anything, re-run both test suites before amending
```

(No system clang-format on this box: `SCRATCH=build/cm1-tools && python3 -m venv "$SCRATCH/fmt" && "$SCRATCH/fmt/bin/pip" install clang-format` per the documented recipe, then use `$SCRATCH/fmt/bin/clang-format`.)

- [ ] **Step 3: Final full verification**

```bash
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
python3 bench/rtrack/v3_gate.py 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
```

Expected: all green / all PASS. Commit the README (+ any format-only diffs):

```bash
git add bench/rtrack/README.md
git commit -m "docs(bench): CM1 Gen4 recv-kernel runbook (#109)"
```

- [ ] **Step 4: Push and open the draft PR**

```bash
git push -u origin cm1-bplacement
gh pr create --draft --title "feat(libreloc, bench): CM1 — BPlacement term + recv-kernel m calibration (#109)" --body "<body>"
```

PR body must contain, in this order: (1) verdict-first summary — slope-tie dead by construction (link the new `SerialKillsRoneSlopeTie` test), both placements brute-force-verified, calibrations regenerated additively (CALIBRATION-REGEN PASS ×2, REPORT-REGEN PASS); (2) the Task 6 Step 6 cross-check numbers verbatim; (3) **what's deferred and why**: Gen4 `recv.m.*` awaiting the home-box run, with the README runbook linked — this is WHY the PR is a draft; (4) the `recv.m.*`-not-consumed-by-`pathCosts` decision (CM5's job) with the spec link; (5) file-adjacency note: #57 also touches `Bind.cpp`, whichever lands second rebases (#107 rule); (6) `Refs #109, #107`. End with the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. `build/sym/libreloc/test/libreloc-test` — all pass (baseline 152+2skip + ~5 new).
2. `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q` — all pass (baseline 171+2skip + 2 new); `test_prediction.py` green proves the frozen V3 report is untouched.
3. `python3 bench/rtrack/v3_gate.py` — CALIBRATION-REGEN PASS for both machines, REPORT-REGEN PASS.
4. `git diff main -- calibration/ | grep '^-' | grep -v '^---'` prints nothing (pure insertions).
5. Issue #109 acceptance mapping: test (i) = `SerialKillsRoneSlopeTie`; test (ii) = placement-parameterized `ThresholdAgreesWithBruteForce`; test (iii) = untouched V3 tests passing under the `Overlapped` default + `FillDrainTermFromChunksKey`; "calibrations regenerate byte-identically except the new keys" = item 3+4; "slope-tie regression test in place" = test (i).
6. Draft PR open with runbook + cross-check + deferral notes.
