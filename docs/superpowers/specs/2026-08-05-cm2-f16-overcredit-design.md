# CM2 — f16-stage over-credit: attribution + correction

**Issue**: #110 (CM2), parent tracking #107. **Date**: 2026-08-05.
**Deliverable shape**: regular (non-draft) PR on branch `cm2-f16-overcredit` — both boxes'
corrections derive from committed artifacts; no new measurement is needed.

## Context

V3's one genuine roofline error (`docs/v3-costmodel.md` §4): on Gen3, the r=0.5 quant
prediction says speedup 1.332 against a measured 1.060 (~26% over-credit to Method A).
Issue #110 requires attribution before patching, then a unit test pinning the corrected
Gen3 r=0.5 quant prediction within ±0.10 of the committed measurement (1.0601163292018307).
Binding constraints: #107 forbids changing Method A's cost *form* (`max(cpu, dma)` stays)
and the committed V3 verdicts (the frozen `bench/results/v3_prediction_report.json` stands
as measured); the model may be corrected freely before #CM4's registration, not after.

## Attribution (evidence-backed; the issue's two named candidates are both refuted)

- **Candidate (a) — "wrong bandwidth key (contig_read 37 vs convert 17)": refuted by
  code.** `cpuBw` at r=0.5 contiguous reads only `cpu.t8.contiguous.convert_f32_f16_gbps`
  (17.4191 on Gen3); `contig_read` is consulted only at r=1.0. Substituting contig_read
  moves the prediction to 1.997 — away from 1.060.
- **Candidate (b) — "A's convert cannot overlap its own DMA": refuted by code and data.**
  The harness is genuinely double-buffered (`rtrack_bench.cu` waits on `h2dEnd[c-2]`), and
  the measured stage decomposition shows overlap: cpu_stage 74.17 ms + h2d 48.63 ms =
  122.8 ms against a 77.47 ms wall. The serialized (sum) counterfactual predicts 0.80 —
  wrong in the opposite direction.
- **Confirmed attribution: in-pipeline DRAM-contention derate.** The model credits the CPU
  convert stage with its *isolated* roofline; in the live pipeline both overlapped stages
  compete for the same host DRAM (EPYC 7351: 4 of 8 channels, triad 44 GB/s) and both
  derate ~16% in lockstep — CPU 17.42 → 14.48 GB/s, DMA 13.07 → 11.04 GB/s. Prior art for
  exactly this effect: `docs/poc-reproduction-v2.md:126-131` (in-pipeline gather 9.5 vs
  isolated 14.3 GB/s under concurrent H2D).
- Corollary findings, recorded for CM4/CM5 (out of CM2's scope): the under-prediction of A
  is monotone in concurrent CPU work, not f16-specific (Gen3: +12.7% at r=1.0 — DMA-bound,
  unfixable by a CPU key — +18.8% at r=0.25, +25.5% at r=0.5); Gen4's sign is *reversed*
  (A measured faster than its convert roofline), so the correction must be a per-box
  calibrated value, never a code constant; A's post-DMA recv-kernel drain (~3.3 ms here,
  `gpu_recv_ms`) remains unmodelled — adding a term would change A's cost form (#107 Out).
  The attribution summary is posted as a comment on issue #110 (acceptance: "attribution
  documented").

## §1 Correction

**New calibration key** (namespace parallel to the existing `cpu.t{t}.{pattern}.{kernel}_gbps`):

```
cpu_pipe.t8.contiguous.convert_f32_f16_gbps
```

meaning: the *in-pipeline effective* bandwidth of the CPU convert stage, measured with
concurrent DMA on the same host — derived by `make_calibration.py` from committed rsweep
CSVs as `srcBytes / cpu_stage_ms` of the best-chunk (min `median_ms`) method=a,
transform=quant, r=0.5, N=16384, t8 row:

- **Gen3** (`bench/results/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv`):
  1073741824 B / 74.1677 ms ⇒ **≈14.48 GB/s** (round(…, 2), provenance-commented).
- **Gen4**: same rule against the committed Gen4 rsweep CSV carrying `cpu_stage_ms` for
  quant r=0.5 N=16384. If no usable row exists, the key is **omitted** (loud omission —
  `cpuBw` falls back to the roofline key, preserving current behavior).
- Existing `.cal` lines unchanged — pure insertions; CALIBRATION-REGEN stays byte-identical.

**Model change** (`libreloc/src/CostModel.cpp`, `cpuBw` r=0.5 contiguous branch only): if
the `cpu_pipe.` key is present, return it; else fall back to `convert_f32_f16` as today.
The non-contiguous harmonic composition and every other r tier are untouched (A1 scope).
`pathCosts`' `max(aCpuSlope, aDmaSlope)` form is untouched (#107 compliance).

**Expected numbers** (Gen3, S = 16384²·4): t_a = 0.068 + S/14.48 GB/s ≈ 74.2 ms →
predicted speedup 82.177/74.2 ≈ **1.107**, |Δ| vs 1.060 ≈ 0.047 ≤ 0.10. The ~4% residual
is the unmodelled recv drain, documented above.

## §2 Tests, frozen-report integrity, deliverable

**Acceptance pin** (new self-contained pytest, `test_wire_row_decision.py`'s shape): load
the real `calibration/epyc7351-2080ti.cal`, compute speedup(0.5) =
`predict(r=1.0).t_b_ms / predict(r=0.5).t_a_ms` (pattern=contiguous, S=16384²·4, t8), and
assert `abs(speedup − 1.0601163292018307) <= 0.10` where the measured constant is read
from `bench/results/v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json`
(`families.quant.speedup_measured["0.5"]`), not hard-coded.

**C++ unit test** (`CostModelTest.cpp`, synthetic calibration): `cpuBw(contiguous, 0.5, 8)`
prefers the `cpu_pipe.` key when present and falls back to `convert_f32_f16` when absent;
`pathCosts` reflects the preferred value in `aSlopeMsPerByte`.

**Frozen-report integrity**: `test_prediction.py` is the report *generator* — after this
model change, running it would rewrite the frozen `v3_prediction_report.json` with moved
predictions and break REPORT-REGEN. Fix: redirect its report output to pytest `tmp_path`
(all structural assertions kept; the committed report is never rewritten). A file comment
records the rationale: the committed report is V3's as-measured record against the model
that produced it (#107); re-registration against the corrected model is #CM4's job.

**Out of scope** (recorded, not patched): r=1.0 and r=0.25 residuals (same mechanism;
r=1.0 is DMA-bound and needs an in-pipeline *link* derate that would also move B — a CM4
registration-time decision); Gen4's reversed-sign misses beyond what its own `cpu_pipe.`
key corrects; any A-side recv term (cost-form change, #107 Out).

**Verification, end-to-end**: full C++ suite + full pytest green with a clean working tree
afterward (proves the frozen report was not rewritten); `v3_gate.py` CALIBRATION-REGEN
PASS ×2 and REPORT-REGEN PASS; clang-format clean; attribution comment posted on #110.
