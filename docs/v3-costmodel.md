# V3 — P3b cost-model component (issue #97)

**Verdicts: V3-MISCLASS 8/60 = 0.133 PASS (bar ≤ 0.15, cushion is 2 cells,
not the 1.7-point gap it looks like — 9/60 = 0.150 would still PASS, only
10/60 = 0.167 fails); V3-RSTAR max|Δ| = 0.363 FAIL (bar ≤ 0.15); V3-REGRET-P90
0.184 PASS (bar ≤ 0.20, margin 0.016 — and this value IS one of the eight
misclassified cells, not an independent check); CALIBRATION-REGEN PASS
(byte-identical regeneration, both boxes).** Two of three quantitative bars
pass, with thin margins on both; the model is directionally right but not
comfortably clear of the line, and the RSTAR FAIL and the REGRET/MISCLASS
near-misses share a single root mechanism (below), not four unrelated
findings.

Bars (`MISCLASS_BAR=0.15`, `RSTAR_ABS_BAR=0.15`, `REGRET_P90_BAR=0.20`) were
fixed in `bench/rtrack/v3_gate.py` at commit `5b9572e`, committed standalone
**before** the prediction test (`libreloc/python/tests/test_prediction.py`)
first ran against data (commit `9721cdb`). No bar was touched afterward.
Measured on: bare-metal EPYC 7351 + 4x RTX 2080 Ti, CUDA 12.5, sm_75
(`epyc7351-2080ti`), and WSL2 7800X3D + 4070 Ti SUPER (`7800x3d-4070tis`).

## 1. Components

`reloc::costmodel` (`libreloc/include/reloc/CostModel.h`,
`libreloc/src/CostModel.cpp`) consolidates the R-track's scattered analytic
scripts into one MLIR-free libreloc component:

- **Calibration parser** — `CostModel::parse`/`load`: flat-text
  `# costmodel calibration v0` format, dotted `key value` lines, `#`
  comments, required version header, optional `# machine:` line. Rejects
  missing/wrong version, malformed lines, and duplicate keys, each naming
  the offending line number. Read via `has`/`at`/`get`/`machine`.
- **Gather-pattern classifier** — `classify(BoundPlan)`: maps a bound
  plan's coalesced run length to `Contiguous` / `SingleElement` / `Blocked`
  / `Tiled` (`L == totalElems` / `L == 1` / `L >= 64 elems` / else),
  unit-tested via a synthetic `planWithL` truth table over all four
  pattern boundaries (`CostModelTest.cpp`'s `CostModelClassify.MapsLToPattern`),
  a bound-fixture pattern assertion in `BindTest.cpp`'s
  `CostModelDecisionPopulatedWhenModelPassed` (asserts `Contiguous` on the
  N=64 degraded fixture), and the wire row's bind-time classification
  (§6) — not against the six T-workload plans directly, which the
  classifier is never exercised against.
- **Two-path affine cost model** — `pathCosts`: `T_A = overhead.a_ms + S ·
  max(1/BW_cpu(pattern), wireBytes/S/BW_link)`, `T_B = overhead.b_ms + S ·
  max(1/BW_link, m/BW_hbm)`, both affine in source bytes `S`; multi-GPU
  scatter/broadcast scale the DMA term by `K` and (for broadcast) drop the
  `r` reduction on B's side. The prefold arm reuses V4's validated
  `reloc::prefold::prefoldWins` rule verbatim — no new arithmetic.
- **`decide()` + threshold precompute** — `MethodDecision decide(...)`
  returns the chosen method, both predicted times, and (for single-free-
  symbol plans) a precomputed `thresholdBytes` boundary so `bind()` only
  has to compare the bound size against a stored constant, matching the
  "compile-time folded" story. Multi-symbol plans fall back to
  evaluate-at-bind (still microseconds); brute-force-vs-threshold agreement
  is unit-tested.
- **`bind()` hook** — `bind(plan, symbolMap, strategy, const CostModel
  *model = nullptr)`: default `nullptr` keeps every existing caller
  source-compatible; when a model is supplied, `BoundPlan::decision` is
  populated (`libreloc/include/reloc/Bind.h:80`, wired at
  `libreloc/src/Bind.cpp:342-347`). The P2 fixed-heuristic constants
  (`kL2Bytes`, `kMultiThreadMaxBytes`) become calibration-overridable via
  `strategy.single_thread_max_bytes`/`strategy.multi_thread_max_bytes` when
  a model carrying those keys is present — the override path itself is
  unit-tested (a calibration override does change the selected `Strategy`
  for an otherwise-identical plan/N) — retained as the no-model fallback
  otherwise. Both committed calibrations currently just seed the P2
  defaults verbatim rather than overriding them: no measured small-size
  sweep exists yet to derive different threshold values (§7).
- **pyreloc surface** — `pyreloc.load_calibration(path)` →
  `Calibration` handle; `pyreloc.predict(calibration, pattern=..., ...)` →
  dict. Both call the identical C++ arithmetic (`libreloc/python/PyReloc.cpp`);
  the prediction test and any future notebook share one implementation, no
  mirrored Python model.

## 2. Calibration provenance

`bench/rtrack/make_calibration.py` assembles `calibration/epyc7351-2080ti.cal`
(38 `key value` lines) and `calibration/7800x3d-4070tis.cal` (42 lines)
from committed `bench/results` artifacts only. **Both** files carry the
same `strategy.single_thread_max_bytes`/`strategy.multi_thread_max_bytes`
seed pair — it is not gen4-exclusive; gen4's 42 vs epyc's 38 nets +8
`pack_s8_s4` keys (gen4's `r=0.125` tier is modelable, unlike epyc's —
see below) against -4 keys epyc has that gen4 lacks (`hiding.ratio`,
`multigpu.delivery_gbps.k2`/`k4`, `prefold.alloc_ms_per_gib` — R3/R4/V4
never ran on the gen4 box). Every emitted line carries a trailing
provenance comment naming its source file, and regeneration is byte-for-byte
deterministic (`git diff --exit-code calibration/` after two independent
regenerations).

Honest proxies, stated rather than hidden:

- **Gen4 HBM keys are a proxy, not a Gen4 measurement.** No Gen4 HBM sweep
  exists, so `hbm.bw_gbps`/`hbm.m.*` on `7800x3d-4070tis` are copied through
  from the EPYC box's numbers (544 GB/s, m={1, 2.97, 1.43, 2.97}); every
  emitted line carries the proxy note verbatim.
- **`hbm.m.tiled` reuses `hbm.m.blocked`'s value (2.97).** No separate
  tiled GPU-receive kernel was ever measured; the calibration note says so
  explicitly rather than inventing a distinct number.
- **`overhead.{a,b}_ms` are two-point affine fits**, not measured
  intercepts: EPYC fits the `quant` (T3) best-chunk rows at N=2048 and
  N=16384 (`overhead.a_ms=0.068`, `overhead.b_ms=0.0242`). Gen4's fit was
  corrected during review (Task 6) to merge the original 224-row nsweep
  CSV with its rerun companion, taking the stabler-IQR best-chunk row per
  (method, N) — the pre-declared rule from
  `docs/r2-exp2-gen4-crossover.md:48-58` — because the rerun file alone
  only spans quant N∈{2048,4096} and both its own endpoints are
  `unstable=1`. The merged fit spans N=2048..16384 like EPYC's and flips
  the a/b intercept ordering (`overhead.a_ms: 0.068→0.0000` clamped at the
  floor, `overhead.b_ms: 0.0731→0.0927`) — this ordering flip is exactly
  the mechanism behind the RSTAR FAIL's structural read (§4).
- **`pack_s8_s4` is omitted entirely from the EPYC calibration.** The one
  candidate source (`quant_bw_n8192_t{8,1}*.json`) was found on inspection
  to be mislabeled Gen4 data (its kernels resolve to `avx512`, and the
  EPYC 7351 has no AVX-512F), so it was not used for either box; EPYC's
  r=0.125 tier is genuinely unmodelable rather than backed by invented
  numbers — no cell in the frozen inventory needs it.

## 3. Prediction table, ablation, and the misclassified cells

60 modelable cells (0 unmodelable): 24 gen3 single-GPU + 24 gen4 single-GPU
(best-chunk / merged-stabler-chunk per (transform, N)) + 12 multi-GPU
scatter/broadcast (winner `a` vs `bxk` per scenario/K). r* is reported
separately (§4) — it is not a per-cell winner/regret quantity.

**Ablation** (mean regret vs. the measured oracle, n=60):

| policy | mean regret |
|---|---|
| model | **0.046** |
| always-A | 3.69 |
| always-B | 0.161 |

The model beats always-B (the "safer-looking" fixed choice) by >3x and
always-A by ~80x, confirming it does real, non-trivial work despite the 8
misses below.

### All 8 misclassified cells

| cell_id | winner_meas | winner_pred | t_a_meas | t_b_meas | t_a_pred | t_b_pred |
|---|---|---|---|---|---|---|
| gen3:convert_f16:N=2048 | b | a | 1.3670 | 1.3508 | 1.0312 | 1.3078 |
| gen4:convert_f16:N=8192 | a | b | 8.8267 | 10.7866 | 10.2438 | 10.0866 |
| gen4:convert_f16:N=16384 | a | b | 33.9244 | 42.6932 | 40.9750 | 40.0682 |
| multigpu:scatter:N=8192:K=1 | a | b | 17.7255 | 21.3585 | 20.6063 | 20.5625 |
| multigpu:scatter:N=16384:K=1 | a | b | 71.5663 | 84.7105 | 82.2212 | 82.1774 |
| multigpu:broadcast_contig:N=8192:K=1 | a | b | 16.9038 | 21.2886 | 20.6063 | 20.5625 |
| multigpu:broadcast_contig:N=8192:K=2 | a | b | 19.6097 | 31.1882 | 25.6090 | 25.5652 |
| multigpu:broadcast_contig:N=8192:K=4 | a | b | 22.3165 | 45.0738 | 25.6577 | 25.6139 |

**Predicted margins, corrected.** The model's predicted margin is 0.05–2.3%
in 7 of the 8 misses (`gen4:convert_f16` 1.56%/2.26%; the five multi-GPU
degenerate ties 0.05–0.21%) — near-ties the model was never confident
about. `gen3:convert_f16:N=2048` is the sole exception: the model predicted
a **26.8%** margin the wrong way (b at 1.3078 vs a at 1.0312) against a
measured margin of only 1.19% (1.367 vs 1.351 ms) — confidently wrong, not
a near-tie. That measured 1.19% gap is larger than either contributing
row's own noise band (`iqr_over_median_pct` 0.714% for the `a` row, 0.070%
for `b_fair`), so it is a real, if small, measured difference, not
measurement noise the model could reasonably have called either way.

#### The multi-GPU cluster: a structural r=1 slope-tie, not scattered noise

At `r=1.0`, `CostModel.cpp`'s `aDmaSlope = K·r·msPerByteAt(bwDel)` and
`bDmaSlope = K·msPerByteAt(bwDel)` are **algebraically identical for any K
and any broadcast flag** — `r=1` cancels the only term that could
distinguish them. Recomputing `t_b_pred − t_a_pred` for all 12 multi-GPU
cells shows this degeneracy holds in **9 of the 12**:

| cell | K | Δ (b−a) pred | degenerate? | winner_meas | winner_pred | hit/miss |
|---|---|---|---|---|---|---|
| scatter N=8192 | 1 | −0.0438 | yes | a | b | **miss** |
| scatter N=8192 | 2 | −0.0438 | yes | b | b | hit (luck) |
| scatter N=8192 | 4 | −0.8728 | no | b | b | hit |
| scatter N=16384 | 1 | −0.0438 | yes | a | b | **miss** |
| scatter N=16384 | 2 | −0.0438 | yes | b | b | hit (luck) |
| scatter N=16384 | 4 | −3.3599 | no | b | b | hit |
| broadcast N=8192 | 1 | −2.9615 | no | b | b | hit |
| broadcast N=8192 | 2 | −0.0438 | yes | b | b | hit (luck) |
| broadcast N=8192 | 4 | −0.0438 | yes | b | b | hit (luck) |
| broadcast_contig N=8192 | 1 | −0.0438 | yes | a | b | **miss** |
| broadcast_contig N=8192 | 2 | −0.0438 | yes | a | b | **miss** |
| broadcast_contig N=8192 | 4 | −0.0438 | yes | a | b | **miss** |

Whenever the DMA term is also the term `pathCosts` picks for A's
`max(cpuSlope, dmaSlope)`, both slopes tie exactly and the predicted gap
collapses to the fixed-intercept difference alone: `overhead.a_ms −
overhead.b_ms = 0.068 − 0.0242 = 0.0438 ms` against 20–82 ms transfers (a
0.05–0.2% margin) — a coin flip that **always** lands "b" on this box
since `a_ms > b_ms` here. **5 of the 9 degenerate cells are misses; the
other 4 landed on "b" only because B happened to be the real winner there
for unrelated reasons** — those 4 hits are luck, not model correctness,
and both passing bars (MISCLASS and REGRET-P90) are set by cells from this
same degenerate cluster:

- `V3-REGRET-P90 = 0.1837` **is** the regret of
  `multigpu:scatter:N=16384:K=1`, one of the five degenerate-tie misses —
  not an independent data point. The next-largest regret in the sorted
  list is `0.205` (`multigpu:scatter:N=8192:K=1`, also a degenerate-tie
  miss); had the percentile landed there instead, REGRET-P90 would
  **FAIL** (0.205 > 0.20).
- `MISCLASS`'s cushion against its `≤0.15` bar is **2 cells**
  (9/60 = 0.150 still PASSes on the inclusive boundary; 10/60 = 0.167
  fails), not the visually larger 1.7-point gap between 13.3% and 15%.
  Note the pre-registered gate implements this as an *inclusive* `<=`
  comparison (`rate <= MISCLASS_BAR` in `v3_gate.py`), while issue #97's
  own prose states the bar as strict ("winner misclassification **< 15
  %**"). The two readings agree on the actual result (0.133 clears both),
  but the 9/60 = 0.150 boundary case above is exactly where they would
  diverge: the gate as coded would PASS it, the issue's literal prose
  would not. Stated for the record, not changed — the bars are frozen
  (see the top of this document).

**Named follow-up (not an excuse, not attempted here — never refit):** the
model lacks a per-receiver/staging overhead term for method B at K>1.
`multigpu.delivery_gbps.k{2,4}` was measured on a pure-DMA micro-benchmark
(`m0_multigpu_h2d_*.json`) that never exercises `bxk`'s real per-device
kernel-launch/sync/staging path, so the model has no key representing why
measured `bxk` wall time grows faster with K (21.3 → 31.2 → 45.1 ms,
K=1→2→4, `broadcast_contig`) than the model's K-scaled DMA-only prediction
(20.6 → 25.6 → 25.6 ms).

## 4. RSTAR FAIL — corrected attribution

**This is not a diffuse family/box roofline weakness — it is the same
r=1 slope-tie mechanism from §3, applied to the r*-crossing search
instead of the winner decision.**

On **gen3**, predicted speedup (`t_b_ms(r=1.0)/t_a_ms(r)`, `quant`,
N=16384, `epyc7351-2080ti`) across the measured grid:

| r | 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| predicted speedup | 1.7587 | 1.3317 | **0.99947** |

The r=1.0 point sits **0.053% below 1.0** for exactly the §3 reason: at
r=1, A's and B's slopes tie (both DMA-bound), so the predicted speedup at
r=1 reduces to `t_b_intercept/t_a_intercept`-scale noise around 1.0, not a
genuine transform-vs-link comparison. `crossing()` (inlined verbatim from
`figure_rstar.py`) finds the sign change between r=0.5 (above 1) and r=1.0
(just below 1) and reports `r*≈0.9989` against a measured `r*=0.6356` —
`|Δ|=0.363`, the value that drives the FAIL.

**The contrast that proves it's structural, not box-specific:** on
**gen4**, the intercepts are *reversed* post-Task-6-fix
(`overhead.a_ms=0`, `overhead.b_ms=0.0927` — B is more expensive at the
margin here, not less), and the same r=1 slope-tie evaluates to speedup
**1.0023** — just *above* 1.0 instead of just below. No spurious r≈1
crossing fires on gen4; the sign of the near-1.0 value is fully explained
by which box's `overhead.b_ms` is larger, an intercept-sign coincidence,
not a property of the `quant` transform's roofline physics on either box.

**The genuine roofline error worth naming instead**: at r=0.5 on gen3, the
model predicts speedup **1.332** against a measured **1.060** — a real
~26% over-credit for the model's dtype-reduction (f16) stage on this box,
independent of any intercept artifact. That is the actual "roofline is
wrong here" finding this dataset supports; the headline r*=0.9989-vs-0.6356
number is not. (**Correction**: `0.9994673` is the r=1.0 *speedup* value
from the table above, not r* itself — those are different quantities. The
crossing-interpolated `rstar_predicted` the gate actually compares against
the measured `r*=0.6356` is `0.9988891`, distinct from but close to the
r=1 speedup because the interpolated crossing between r=0.5 and r=1.0
lands very near the r=1 endpoint; `|Δ|=0.363` either way.)

**Two-sided implementation note.** The committed
`v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json` already stores
`rstar_predicted: 1.0` under `figure_rstar.py`'s own (independently
implemented, same-formula) model for this family/box, so `pyreloc.predict`
reproduces a pre-existing property of the roofline formula, not a new port
bug. But that evidence is one-sided: `v1_gen4_rstar_bfair.json`'s own
stored `rstar_predicted` values (0.1807 `blocked_transpose`, 0.1640
`quant`) diverge **1.6–2.9×** from what `pyreloc.predict` gives on the
same family/box (0.2914, 0.4806) — the two implementations are
demonstrably *not* numerically equivalent in general. "Not a new port
bug" rests on the shared r=1 slope-tie mechanism (both implementations
follow the identical `aDmaSlope`/`bDmaSlope` formula), not on the two
models agreeing overall. **V3's `pyreloc.predict`/`CostModel.cpp` is the
maintained implementation** going forward; `figure_rstar.py`'s standalone
model is the R2-era reference it was checked against, not a co-equal
second implementation to keep in sync.

## 5. Gate-structure limitation (documented, not patched)

`rstar` rows classified `mismatch_one_sided` (one side has a crossing, the
other doesn't) are **excluded entirely** from `RSTAR_ABS_BAR`'s `diffs`
list in `v3_gate.py`'s `gate_rstar` — only `both_exist` rows feed the max.
A **more wrong** prediction — one where gen3 `quant`'s r=1 speedup had
landed on the *other* side of 1.0 and produced no crossing at all — would
have been reclassified as `mismatch_one_sided` and **escaped the bar
entirely**, showing only as a qualitative mismatch line instead of driving
the 0.363 FAIL. The gen3 `quant` predicted r=1 speedup (0.99947) sits
**0.053% below 1.0** — a hair's-breadth from that reclassification
boundary; a value 0.06% higher (well within plausible measurement noise)
would make this exact same underlying weakness vanish from the bar
entirely rather than failing it. No gaming occurred here — the committed
report was generated once, honestly, before this was noticed — but the
bar's robustness to this near-miss reclassification is fragile. **Any
future re-run of this gate (new calibration, new box, a repeat after a
fix) must pre-register a tightened rule before that run happens** — e.g.
treating `mismatch_one_sided` as an automatic FAIL contribution rather
than an exclusion — before, not after, seeing which families would land
in which bucket.

## 6. The wire row (title-gap closure)

A symbolic blocked-transpose plan travels the full compiler path — MLIR
chain → `sym-opt --reloc-fold` → `encodePlan` (`generate_corpus.py`,
committed as `libreloc/test/corpus/blocked_transpose_sym.{bin,json}`) —
into `bench-rtrack`'s new `--plan-wire PATH --transform TW` mode:
**fold → wire → decode → bind → measure**. Bijective CLI validation:
`--plan-wire` requires exactly `--transform TW` and vice versa.

**Bind sanity check (pre-measurement gate)**: decoding + binding the
corpus `.bin` at N=8192 and diffing against `plans.h`'s hand-authored
`blockedTransposePlan(8192)` is **bit-exact**: `extents={64, 128, 8192}`,
`srcStrides={8192, 524288, 1}`, `dstStrides={1048576, 8192, 1}`, `L=8192`
match exactly. All 12 (method × chunk) measured configs print `[verified]`
(bit-exact against the scalar CPU reference).

**Acceptance vs. the committed T1b row** (N=8192, T=8, best-chunk cells):

| method | wire median (ms) | T1b committed [min, p95] (ms) | verdict |
|---|---|---|---|
| a | 32.9088 | [32.1341, 33.7097] | PASS |
| b | 24.4351 | [23.0837, 26.2996] | PASS |
| b_fair | 21.6768 | [21.6812, 21.6889] | **FAIL** (0.02% BELOW the window) |

**Overall acceptance: FAIL by the letter of the bar** (1 of 3 methods
outside the window), but this is a near-total resolution: `a` is a clean
PASS, `b` is a clean PASS, and `b_fair`'s miss is 0.02% — an order of
magnitude smaller than the 0.2%-over-p95 miss from the round-0 recipe
(§below) and on the *opposite* side of the window (the wire run measured
marginally faster than the committed row, not slower). `b_fair`'s own
per-config `[min, p95]` at its best chunk (`[21.6754, 21.6855]`) overlaps
almost entirely with the T1b window (`[21.6812, 21.6889]`). **Correction**:
the `0.0044 ms` gap is the wire row's best-chunk **median** (21.6768)
against the T1b window's **minimum** (21.6812), not a minima-to-minima
comparison — the two runs' actual minima (21.6754 vs 21.6812) differ by
`0.0058 ms`, marginally larger but still consistent with ordinary
session-to-session DMA-floor jitter on this box — `b_fair` is a pure
DMA-floor measurement with no CPU-stage or plan-shape sensitivity left
once `a` and `b` already agree with the coalesced-identical plan. This is
presented as the measured, as-reported result — no rerun was performed to
try to close the gap.

**Model decision for this row (spec §6 acceptance)**:
`predict(calibration/epyc7351-2080ti.cal, pattern="blocked",
src_bytes=8192*8192*4, r=1.0)` returns `method="b"`; the measured winner
is `b` (`b_fair` 21.6768 ms vs `a` 32.9088 ms, both best-chunk) — decision
matches. Unit-tested standalone in
`libreloc/python/tests/test_wire_row_decision.py`, which loads only the
epyc calibration and this row's CSV and never touches
`test_prediction.py`'s report writer.

**The L=64 variant, kept as a decomposition-sensitivity finding.** The
first corpus recipe attempt used `blocked_reference`'s own decomposition
(`reshape([N/64, 64, N/64, 64])` + `transpose([2,0,1,3])`) rather than
`plans.h`'s hand-authored `blockedTransposePlan` decomposition
(`x.view(N/64, 64, 64, N/64).transpose(0, 1)`) — both are individually
correct, oracle-verified blocked-transpose access patterns, just with
different axis orderings. That version binds to `L=64` (256 B innermost
coalesced run) instead of `L=8192` (32 KiB run), and Method A's CPU
gather stage is L-sensitive: its best-chunk median (45.4727 ms, chunk=16)
came in **+41.5% over the T1b window's min** (32.1341 ms), **+34.9% over
its p95** (33.7097 ms), and **+38.2% over the corrected primary wire
row's own best-chunk median** (32.9088 ms) — three baselines for the same
method-a discrepancy between two valid blocked transposes, driven purely
by decomposition, not by any fold/bind bug. That measurement is preserved
as its own row
(`bench/results/v3_wire_row_l64variant_epyc_2080ti.csv`) — **the fold
path faithfully preserves whichever decomposition the MLIR source
expresses**; L=64 vs L=8192 is a property of the source program, not of
the fold/bind/measure machinery, and the corpus recipe was subsequently
corrected to match `plans.h`'s exact decomposition for the primary row
above.

## 7. Caveats

- **21/48 single-GPU cells use unstable (IQR/median > 5%) best-chunk
  rows.** Verdicts are robust to this: comparing each cell's measured
  margin against the larger contributing row's IQR band, **exactly 1 of
  48** cells has a measured margin smaller than its own noise band
  (`gen3:blocked_transpose:N=4096`, margin 0.778 ms vs. a 0.931 ms IQR
  band) — and it is **not** one of the 8 misclassified cells. The 44%
  unstable-row rate should be stated rather than left unexamined, but it
  does not appear to be quietly driving any reported misclassification.
- **Gen4's predicted speedup curve is non-monotonic in r — r* is not
  always single-valued.** The gen4 `quant` curve across r = 0.125, 0.25,
  0.5, 1.0 is 1.187, 1.366, 0.978, 1.002 — it crosses 1.0 *twice* (down
  between 0.25 and 0.5, back up between 0.5 and 1.0). `crossing()` returns
  the **first** sign change scanning upward from small r and stops, so the
  reported gen4 `quant` r*=0.4806 reflects only that first crossing; the
  second, near r=1, is silently dropped. Read "r*" here as "the first
  crossing found scanning upward from small r," not "the unique point
  where A starts winning."
- **Overhead intercepts come from two-point affine fits**, not multi-point
  regressions — see §2's provenance detail on which two N values were used
  per box and why (min/max N present for the `quant` transform in each
  source file, not a hardcoded N pair).
- **The r grid is restricted to measured points** ({0.125, 0.25, 0.5, 1.0}
  where present); no interpolation beyond those wire-dtype tiers is
  performed or implied.
- **Strategy thresholds (`strategy.single_thread_max_bytes`,
  `strategy.multi_thread_max_bytes`) are seeded with P2's existing
  defaults**, not derived from a small-size sweep — no such sweep exists
  yet.
- **Threshold precompute is valid because `allocMs` is linear in `S`**:
  the single-free-symbol threshold boundary relies on every cost term
  (including the prefold arm's allocation cost) being affine in source
  bytes; this holds for every plan in the current tree, and multi-symbol
  plans fall back to evaluate-at-bind rather than attempting a
  (currently out-of-scope) piecewise-linear multi-symbol region.

## 8. Spec amendment

See `docs/superpowers/specs/2026-07-29-v3-costmodel-design.md`'s threshold
paragraph: the "breakpoints where the min/max arms switch" sentence has
been replaced with the overhead-intercept refinement discovered during
planning (cited to `docs/superpowers/plans/2026-07-29-v3-costmodel.md`'s
header note).

## 9. bench/rtrack additions

`bench/rtrack/make_calibration.py`, `bench/rtrack/v3_gate.py`, and
`bench-rtrack`'s `--plan-wire` flag are documented in
`bench/rtrack/README.md`.

## v1 — CM5 gate re-run on the BP dataset (issue #113)

**Verdicts (bars fixed pre-data; rule v1 per CM4):**

| gate | universe | b_fair (Serial) | b_pipelined (Overlapped) |
|---|---|---|---|
| MISCLASS ≤ 0.15 | all cells (48/40) | 2/48 = 0.0417 PASS | 2/40 = 0.0500 PASS |
| MISCLASS ≤ 0.15 | held-out test-N (24/20) | 1/24 = 0.0417 PASS | 1/20 = 0.0500 PASS |
| REGRET-p90 ≤ 0.20 | all cells | 0.0000 PASS | 0.0000 PASS |
| REGRET-p90 ≤ 0.20 | held-out test-N | 0.0000 PASS | 0.0000 PASS |
| RSTAR ≤ 0.15 (rule v1) | 8 rows/placement (16 measurable of 24 registered) | max\|Δ\|=0.0709, 2 one-sided FAIL | max\|Δ\|=0.2942, 1 one-sided FAIL |

Data: BP3 (#116/#124), stabler-preference merged (merge_audit in the report).
Predictions: cm4_registered_predictions.json only. Held-out caveat re-quoted:
"the committed calibrations' overhead.{a,b}_ms intercepts were two-point-fit
on N in {2048, 16384} endpoints, so test-N data touched calibration inputs;
the split stratifies evaluation (per #87's reconciliation in #107), not
calibration -- recorded, not fixed; bars unchanged". No held-out RSTAR
(r-sweep is N=16384-only — already all-test-N). v0 verdicts above stand as
recorded.

### Ablation (mean regret; p90 in parentheses)

| policy | b_fair all | b_fair held-out | b_pipelined all | b_pipelined held-out |
|---|---|---|---|---|
| model | 0.0021 (0.0000) | 0.0030 (0.0000) | 0.0036 (0.0000) | 0.0049 (0.0000) |
| always-A | 4.2493 (18.0291) | 4.8301 (18.3654) | 2.8690 (6.3474) | 3.2046 (6.3474) |
| always-B | 0.1502 (0.5073) | 0.1597 (0.5073) | 0.1640 (0.4817) | 0.1706 (0.4230) |
| oracle | 0 | 0 | 0 | 0 |

### Misses, per family (findings, not refit targets)

**convert_f16 (epyc7351-2080ti only).** All 4 registered MISCLASS misses
are this family and box: b_fair at N=2048 (train split, `t_a_meas=1.38842`
ms vs `t_b_meas=1.34893` ms) and N=4096 (test split, `t_a_meas=5.71905` ms
vs `t_b_meas=5.33385` ms), plus the matching b_pipelined cells at the same
two N (`t_b_meas=1.32661` ms and `5.21035` ms respectively) — the model
calls `a`, B measures faster in every one of the four. §3 above documents
the same family+box direction error at N=2048 (`gen3:convert_f16:N=2048`),
characterized there as **confidently wrong, not a near-tie** (a 26.8%
predicted margin the wrong way against a 1.19% measured gap) — the
intercept-scale coin-flip pattern §3 describes belongs to the *gen4*
convert_f16 misses instead, not gen3/epyc. These four BP cells reproduce
that same gen3/epyc direction error at N=2048 and extend it to N=4096; the
report carries no field decomposing *why* A is confidently favored here
beyond that shared attribution — cause not further determined by this
evaluation.

**quant (both boxes).** Three of the four registered RSTAR rows for this
family are `mismatch_one_sided` (7800x3d-4070tis serial and overlapped,
epyc7351-2080ti serial): the model's predicted speedup curve never crosses
1.0 on the measured grid while the measured curve does (`rstar_meas` 0.6107
and 0.6150 on 7800x3d-4070tis, 0.7025 on epyc7351-2080ti) — rule v1 (CM4)
counts any one-sided mismatch as an automatic FAIL contribution regardless
of a numeric delta. The fourth row, epyc7351-2080ti/overlapped, is
`both_exist` and carries this evaluation's largest delta: predicted
r*=0.9966 against measured r*=0.7024 (`|Δ|=0.2942`). That predicted value
sits within 0.35% of the r=1.0 grid edge — the same
`t_b_intercept/t_a_intercept`-scale-noise mechanism §4 above documents for
this exact family/box/N (where the analogous r=1 speedup landed at
0.99947, just below 1.0, and drove V0's own RSTAR FAIL): a prediction
sitting a fraction of a percent from the grid boundary can manufacture or
miss a "crossing" that has nothing to do with the transform's actual
roofline behavior. (Correction to an earlier working note: `rstar_rows`
excludes `convert_f16` entirely — no BP r-sweep measurement, per
`excluded_cells` — but the registration does carry `convert_f16` rstar
predictions, including one numerically identical to `quant`'s at this
box/placement, since both families share `pattern="contiguous"` in
`cm4_register.py`'s `FAMILY_MAP`. Only `quant` has a measured counterpart
in this evaluation, so the 0.294 delta and the 0.9966 near-edge prediction
this section describes are `quant`'s alone.)

**blocked_transpose (7800x3d-4070tis) — not a gate miss, noted for
context.** This family's overlapped RSTAR row is `both_exist` with
`|Δ|=0.0691` (predicted r*=0.2914, measured r*=0.3605), under the 0.15 bar
on its own and not counted among the misses above. The predicted value
(0.2914) is the same `pyreloc.predict` r*-prediction §4's "Two-sided
implementation note" already flagged as diverging 1.6–2.9× from
`figure_rstar.py`'s own stored value (0.1807) for this exact family/box — a
known V0-era cross-implementation gap, not a new finding here.

### Deferred-item disposition

`pipeline.chunks_per_buffer` emission, `recv.m.*` m_eff composition
(measurement-basis decision included), and Gen4 `hbm.*` re-baselining were
deferred out of CM5 (evaluation-only scope, user decision 2026-08-11):
emitting any of them mid-evaluation would re-score the model against its own
pre-registered predictions and break CM4-REGEN. CM1's "the key lands in CM5"
is superseded; all three land in the CM6 follow-up issue (filed with this
PR), motivated by the miss table above.
