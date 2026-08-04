# CM1 — B-path kernel-placement term + recv-kernel `m` values

**Issue**: #109 (CM1), parent tracking #107. **Date**: 2026-07-31.
**Deliverable shape**: draft PR on branch `cm1-bplacement` — code + tests complete, Gen3
calibration keys landed, Gen4 keys left as loud omissions with a runbook (the Gen4 box is
not reachable from this server; the run happens later on the home machine).

## Context

V3 (#97/#106) shipped `reloc::costmodel` with a single B-path form,
`T_B = ovh.b + S·max(1/BW_link, m/BW_hbm)` — the Overlapped placement, hard-coded.
`docs/v3-costmodel.md` §3–§4 root-caused the RSTAR FAIL (|Δr*| = 0.363) and 5 of the 8
misclassified cells to one structural mechanism: at r = 1 the A and B DMA slopes
(`K·r/BW_link` vs `K/BW_link`) are algebraically identical, so wherever A is DMA-bound the
predicted gap collapses to the 0.044 ms intercept difference — an intercept-sign coin flip,
not physics. V1 (#95/#104) measured the missing physics: B's kernel placement is a
first-class cost term — serial-receive B (`b_fair`) costs the *sum* of transfer and kernel
time, overlapped B costs their *max*. #107 confirmed both forms (Build Doc v3 §2.5) and
scoped CM1 to: implement the placement term, calibrate recv-kernel `m` multipliers, and pin
the slope-tie fix with regression tests. Method A's cost form is explicitly out of scope.

Decisions taken during brainstorming (with the user, 2026-07-31):

1. **recv-`m` keys land as calibration keys only.** `pathCosts` keeps reading
   `hbm.m.{pattern}` for B's kernel multiplier. Composing recv passes into the formula
   (`m_eff = m_pattern + m_recv(r)`) is deferred to CM5, when BP data exists to validate
   it. Rationale: the issue's confirmed formula uses a single `m`; under Overlapped the
   composition is a no-op at current values (m_eff ≈ 4 ≪ hiding ratio 41.7); under Serial
   it would inject an unvalidated ~2.5% correction. Precedent: `hiding.ratio` is already a
   landed-but-unconsumed key.
2. **Fill/drain is slope-folded**: `min(a,b)/n` added to the Overlapped slope, with `n`
   from a calibration key; absent key → term is exactly 0. Keeps every cost affine in S
   (the documented single-threshold invariant, `CostModel.cpp:251-256`) and makes issue
   test (iii) hold with zero fixture edits.
3. **Gen4 measurement deferred**: harness + runbook now, targeted run later on the Gen4
   box. Gen3's targeted run happens in this PR (2080 Ti idle on this box).

## §1 API and formulas (C++ core)

**Enum** — `libreloc/include/reloc/MethodDecision.h`:

```cpp
enum class BPlacement { Serial, Overlapped };
```

**Signatures** — both gain a trailing `BPlacement bPlace = BPlacement::Overlapped`
parameter, so every existing caller compiles unchanged and V3 behavior is the default:

```cpp
std::optional<PathCosts> pathCosts(const CostModel &m, Pattern p, int64_t srcBytes,
                                   double r, int threads, int K = 1,
                                   bool broadcast = false,
                                   BPlacement bPlace = BPlacement::Overlapped);
std::optional<MethodDecision> decide(..., BPlacement bPlace = BPlacement::Overlapped);
```

**Formulas** — with `a = msPerByteAt(BW_link)`, `b = m·msPerByteAt(BW_hbm)`,
`m = hbm.m.{pattern}` (unchanged), and the existing `kMult` broadcast/K handling:

```
Serial:     bSlope = kMult · (a + b)
Overlapped: bSlope = kMult · (max(a,b) + min(a,b)/n)
```

`n` is read as `m.get("pipeline.chunks_per_buffer", 0.0)`; `n <= 0` → the fill/drain term
is 0 and Overlapped is byte-for-byte the V3 formula. The fill/drain term is the standard
pipeline approximation (total = dominant stage over all chunks + one chunk of the hidden
stage); with `ChunkSchedule.h`'s `chunk = clamp(S/8, 4MiB, 64MiB)` the "one chunk" is S/8
in the mid-range, so the term folds into the slope and stays affine everywhere. The clamp's
extremes are a knowingly accepted approximation error (documented in a code comment).

**`decide()` / threshold**: mechanically unchanged — the new `bSlope` flows through the
existing `sStar = (bIntercept − aIntercept)/(aSlope − bSlope)` precompute. Under Serial,
the algebraic A/B DMA-slope identity cannot occur: when A is DMA-bound (the §4 tie case),
`bSlope = a + b > a = aSlope` strictly; in CPU-bound regimes the slopes differ generically
(the regression test asserts inequality across all K/broadcast combos), so `dSlope ≠ 0`
is guaranteed and the §4 slope-tie is dead by construction.
`MethodDecision` gains a `BPlacement bPlacement` field recording which placement the
decision priced against.

**Call sites**:
- `bind()` (`libreloc/src/Bind.cpp`, step 8) passes `BPlacement::Overlapped` explicitly —
  the library's real B implementation is the double-buffered pipeline (`Pipeline.cpp`).
- `pyreloc.predict` (`libreloc/python/PyReloc.cpp`) gains kwarg
  `b_placement: str = "overlapped"` (accepts `"serial"`; unknown value →
  `py::value_error`), and its result dict reports the placement back.
- No other surfaces change; consolidating `figure_rstar.py` vs `pyreloc.predict` is CM3.

## §2 Calibration and measurement

New keys (namespaces chosen so no existing key changes — both committed `.cal` files stay
byte-identical except for additions):

| Key | Meaning | Gen3 source | Gen4 source |
|---|---|---|---|
| `pipeline.chunks_per_buffer` | fill/drain `n` = 8 | `ChunkSchedule.h` constant (provenance note names the header) | same |
| `recv.m.convert_f16_f32` | m of the f16→f32 recv kernel | committed `bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv` (`gpu_recv_ms`, best chunk per cell; ≈1.05) | omitted |
| `recv.m.dequant_s8_f32` | m of the s8 dequant recv kernel | same CSV (≈1.07) | omitted |
| `recv.m.unpack_dequant_s4` | m of the s4 unpack+dequant two-kernel chain | targeted run on this box (no committed r=0.125 data exists for Gen3) | omitted |

Derivation is R4-style throughout: kernel BW on a read+write traffic basis
(`r·S` read + `S` written for convert/dequant; the s4 chain counts both kernels' traffic),
divided into the box's `copy_f32` ceiling. "relocate/transpose recv" from the issue's key
list is *already calibrated* as `hbm.m.{pattern}` — no duplicate keys; the SOURCES
docstring states the mapping.

**Targeted run tool**: extend `bench/rtrack/hiding_ratio.cu` with `convert_f16_f32`,
`dequant_s8_f32`, and the `unpack_dequant_s4` chain, following the file's existing
verify-against-CPU-reference-then-time pattern and its JSON schema
(`by_n{kernel{traffic_bytes, median_ms, min_ms, p95_ms, iqr_over_median_pct, gb_per_s}}`).
The Gen3 run is committed as `bench/results/cm1_recv_kernel_bw_epyc_2080ti.json`.

**Session self-consistency rule**: a new run's `copy_f32` ceiling may differ slightly from
R4's 544.6 GB/s, so `recv.m.unpack_dequant_s4` is derived against *its own run's* ceiling.
Existing keys keep their R4 artifact source untouched. The Gen3 f16/s8 values from the
targeted run serve as a cross-check only: if they diverge >5% from the committed-CSV
derivation, that is reported in the PR notes (the key values themselves stay
committed-sourced, per the issue's "from committed artifacts where possible").

**Gen4 runbook** (recorded in the PR description and `bench/rtrack/README.md`): the gen4
builder's recv-m block uses the existing `read_json_optional` omission path until the
artifact exists. On the Gen4 box, later: ① build and run
`bench-hiding-ratio --json bench/results/cm1_recv_kernel_bw_7800x3d_4070tis.json`
(this also produces the first real Gen4 `copy_f32` ceiling); ② commit the JSON; ③ re-run
`make_calibration.py --machine 7800x3d-4070tis`; ④ `v3_gate.py` regen check. Gen4 recv-m
divides by that run's own ceiling. **Re-baselining the existing Gen4 `hbm.*` proxy keys is
out of CM1's scope** — it would break "byte-identical except the new keys" and V3's
verdicts stand as measured; the runbook records it as a future decision.

**`make_calibration.py` changes**: recv-m emit blocks in both builders + SOURCES docstring
entries; `pipeline.chunks_per_buffer` emitted for both machines. Missing required Gen3
sources still `sys.exit`; the Gen4 recv artifact is optional-by-design.

## §3 Tests, error handling, deliverable

Unit tests (in `libreloc/test/CostModelTest.cpp`, reusing `kSynth`):

1. **Slope-tie regression** (issue test i): at r = 1.0 under Serial, assert
   `bSlopeMsPerByte > aSlopeMsPerByte` strictly and `thresholdBytes` is a finite positive
   boundary, for K ∈ {1, 4} × broadcast ∈ {false, true} — pinning that the §4 mechanism
   cannot recur.
2. **Brute-force agreement × placement** (issue test ii): parameterize the existing
   `ThresholdAgreesWithBruteForce` scan (S ∈ [2^12, 2^34], winner-vs-direct-comparison,
   exactly one flip, threshold brackets the flip) over both placements.
3. **Overlapped ≡ V3** (issue test iii): all existing V3 tests pass unmodified under the
   Overlapped default (the absent `pipeline.chunks_per_buffer` key zeroes the fill/drain
   term on `kSynth`); plus a fill/drain-specific case: adding the key raises the Overlapped
   slope by exactly `min(a,b)/8` and reduces to the V3 value when the key is absent —
   this also covers the R4 hiding condition (`m/BW_hbm ≤ 1/BW_link` ⇒ max picks the DMA
   term, matching V3).
4. Supporting: `MethodDecision.bPlacement` recorded; `pyreloc.predict(b_placement=...)`
   matches the C++ arithmetic (`test_costmodel.py`); `BindTest` covers the explicit
   Overlapped pass-through.

**Error handling**: no new failure modes. Missing BW keys → `nullopt` (existing), missing
optional keys → `get` fallback (fill/drain 0). `v3_gate.py`'s CALIBRATION-REGEN compares
against the updated committed `.cal` files and must PASS; REPORT-REGEN (the frozen V3
prediction report) must stay untouched — regenerating it is CM5's job.

**Acceptance mapping** (issue #109): both placements tested (tests 1–3); calibrations
regenerate byte-identically except the new keys (regen check + additive-only key policy);
slope-tie regression test in place (test 1).

**Verification, end-to-end**: `libreloc-test` and `pytest` fully green; `v3_gate.py`
CALIBRATION-REGEN + REPORT-REGEN PASS; clang-format clean; Gen3 cross-check reported.
File-adjacency note: #57 also touches `Bind.cpp` — per #107, whichever lands second
rebases; stated in the PR notes.

## Amendment (2026-08-04, during implementation)

§2's table above listed `pipeline.chunks_per_buffer` as a CM1 key for both machines.
During implementation this was found to activate the `Overlapped` fill/drain term on the
frozen V3 prediction cells and silently re-score `bench/results/v3_prediction_report.json`
(RSTAR 0.363→0.078 without pre-registration) — violating #107's "V3 verdicts stand as
measured" and V3 §5's pre-registration obligation. Ruling: the key is deferred to CM5;
CM1 ships the term implemented + tested against synthetic calibrations only.

The final review additionally found §1/§2's "chunk = S/8" mapping wrong by the buffer
count: `ChunkSchedule.h` divides by `kChunksPerBuffer * nBuffers`, and the pipeline's
`nBuffers = 2`, so the mid-range chunk is actually `S/16`, not `S/8`. When CM5 emits
`pipeline.chunks_per_buffer`, its value must be the total chunk count (16 at 2 buffers) —
`kChunksPerBuffer` alone (8) would understate the divisor and overstate the fill/drain
term by 2x. (The key name is now slightly misleading given it must hold the total, not the
per-buffer, count — CM5 may prefer a name like `pipeline.total_chunks`; the C++ key string
itself is left unchanged in CM1.)
