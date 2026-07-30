# V3 — P3b cost-model component (issue #97) — design

**Status**: approved 2026-07-29.
**Scope**: sub-issue #V3 of #94. Consolidates the scattered analytic
scripts into `reloc::costmodel`, a libreloc component consumed by
`bind`, with a pre-registered prediction test over the committed R-track
data and two additions from the 2026-07-28 project review: the decision
covers the prefold arm via `reloc::prefold::prefoldWins`, and one
measured validation row is driven by a compiler-emitted plan.

## Why

Every calibration input the model needs is already measured and
committed (V1's admissible `BW_B` on both boxes, R1+V2 per-pattern CPU
rooflines, R4's `m` table and hiding ratio, M0/R3 multi-GPU delivery
rates), and the r\* equations exist in `bench/rtrack/figure_rstar.py`.
What is missing is the component: something that takes a plan plus a
binding, predicts both paths, emits a decision, and precomputes the
decision as a symbolic threshold so runtime `bind` only compares.
Without it the title claim — compile-time folded symbolic planning —
has no code path connecting the folding machinery to the measured wins.

## Architecture

One C++ component, one pybind surface, one calibration assembler, one
pre-registered test, one bench row. Single source of truth: the model
arithmetic exists only in C++; Python calls it through pyreloc.

### 1. Calibration files — `calibration/*.cal`

Dependency-free flat text format (libreloc has no JSON parser and the
schema is small and frozen):

```
# costmodel calibration v0
# machine: epyc7351-2080ti
# provenance: bench/results/v1_calibration_epyc_2080ti.json
pcie.h2d_gbps 13.07
cpu.t8.contiguous.quantize_pack_gbps 22.76   # v2_rooflines/identity_avx2_t8_...
cpu.t8.blocked.gather_f32_gbps 11.39
cpu.t8.single_element.gather_f32_gbps 0.53   # r1 rooflines
cpu.t8.tiled.gather_quantize_gbps 2.14
cpu.t1.contiguous.quantize_pack_gbps 13.13   # T=1 tier, same key shape
hbm.bw_gbps 544
hbm.m.transpose_smem_padded 1.43
hbm.m.relocate_naive 2.97
hiding.ratio 41.7
multigpu.delivery_gbps.k2 21.02
multigpu.delivery_gbps.k4 41.96
```

- Format: `key value` per line, dotted keys, `#` comments (full-line and
  trailing), version header line required. Parser rejects unknown
  version, malformed lines, duplicate keys.
- **Producer**: `bench/rtrack/make_calibration.py` assembles a `.cal`
  from the committed `bench/results` artifacts; every emitted line
  carries a trailing provenance comment naming its source file. Two
  files committed: `calibration/epyc7351-2080ti.cal`,
  `calibration/7800x3d-4070tis.cal`. Regeneration is deterministic from
  the repo tree.

### 2. Component — `libreloc/include/reloc/CostModel.h`, `src/CostModel.cpp`

**Loader**: `CostModel::load(path) -> std::optional<CostModel>` (or an
error-string variant matching `BindResult` style). Unconditional
validation, no asserts (-DNDEBUG builds).

**Gather-pattern classifier** (plan properties only — misclassifying
the pattern dominates every other modelling error, R1's >20× spread):

```
classify(BoundPlan):
  L == totalElems                 -> Contiguous
  L == 1                          -> SingleElement
  L >= kBlockedRunFloor (64 elems)-> Blocked
  else                            -> Tiled
```

Exact enum + mapping to roofline keys frozen in the header; classifier
is a pure function, unit-tested against the six T-workload plans (T1 ->
SingleElement, T1b -> Blocked, T3/T5 -> Contiguous, T4 -> Tiled).

**Model** (all effective input-normalized GB/s, matching
`figure_rstar.py`'s conventions):

- `T_B(S) = max(S/BW_pcie, m·S/BW_hbm)` — with `m` looked up per GPU
  receive kernel; reduces to link-bound when `m < hiding.ratio`
  (R4-validated). Default `m` = the plan-appropriate kernel; callers may
  override.
- `T_A(S, r, pattern) `: pipelined form
  `S / min(BW_cpu(pattern), BW_pcie/r)` and serial form
  `S·(1/BW_cpu + r/BW_pcie)`; two-pass CPU stages compose harmonically in
  source-normalized GB/s (the figure_rstar rule). The dtype ratio `r` is
  a first-class parameter.
- Multi-GPU: with `K` receivers, B delivers its total bytes at the
  calibrated aggregate rate (`multigpu.delivery_gbps.kK`), A likewise
  for its DMA leg; total bytes per scenario: scatter A = r·S, B = S;
  broadcast A = K·r·S, B = K·S. A's single CPU transform stays flat in K
  — this reproduces R3's inequality (A wins iff CPU transform BW exceeds
  the aggregate delivery rate).
- **Prefold arm**: `decide(..., nReuse)` — when `nReuse` is provided,
  the A-side transform cost is amortized via the existing
  `reloc::prefold::prefoldWins(nReuse, tTransform, tPrefoldCold,
  penalty)`; `tPrefoldCold` defaults to `tTransform` + a calibration
  `prefold.alloc_ms_per_gib` term. No new arithmetic — V4's validated
  rule is called, not reimplemented. Both sides of that rule scale
  linearly in `S`, so the single-symbol threshold precompute remains
  valid with the prefold arm active; the stored threshold covers the
  chosen-method boundary, and `decide()` records which arms were
  compared.

**Decision**:

```cpp
struct MethodDecision {
  enum class Method { A, B, APrefold };
  Method method;
  double tAMs, tBMs;          // predicted, at the bound size
  double thresholdBytes = -1; // single-symbol precompute (see below)
  // + provenance: pattern classified, m used, K, nReuse
};
MethodDecision decide(const PlanParams &, const CostModel &, int K = 1,
                      int64_t nReuse = -1);
```

**Threshold precompute (single free symbol)**: both cost forms are affine
in `S` with no `S`-dependent min/max arm switches, so the crossover is
degenerate (always-A or always-B) unless the model carries per-transfer
fixed costs; the calibration therefore includes `overhead.a_ms` /
`overhead.b_ms` intercepts (two-point fits over committed N=2048 vs
N=16384-class medians — see the implementation plan's header note,
`docs/superpowers/plans/2026-07-29-v3-costmodel.md`), and `thresholdBytes`
is the argmin-boundary of the resulting affine cost lines. `bind` then
only compares `totalBytes` against the stored threshold (the compile-time
story, made real for every plan in the tree — all have one symbol). Plans with ≥2 free symbols:
evaluate-at-bind fallback (`decide()` at the bound values — still
microseconds). Piecewise-linear multi-symbol *regions* are explicitly
deferred (YAGNI: no such plan exists; note kept in this spec).
Acceptance (from #97): threshold agrees with brute-force
evaluate-at-many-sizes on synthetic calibrations — unit-tested.

### 3. `bind()` integration

- `bind(plan, symbolMap, Strategy override = Auto,
  const CostModel *model = nullptr)` — default nullptr keeps every
  existing caller source-compatible. When non-null, `BoundPlan` gains a
  populated `std::optional<MethodDecision> decision`.
- The P2 fixed heuristic's magic constants (`kL2Bytes`,
  `kMultiThreadMaxBytes`, `Bind.cpp:315-327`) are **replaced by
  calibration-derived values when a model is present** (e.g. the
  single-thread/multi-thread boundary from the T=1 vs T=8 roofline
  ratio), constants retained as the no-model fallback — this is the
  literal "replaces the heuristic" item in #97. The A/B/APrefold method
  decision is a *new axis* (Method B is not a `Strategy`); the spec
  states this distinction explicitly.

### 4. pyreloc surface

`pyreloc.load_calibration(path)` -> capsule/handle;
`pyreloc.predict(params_dict | bound_plan, calibration, k=1,
n_reuse=None)` -> dict `{method, t_a_ms, t_b_ms, threshold_bytes,
pattern, ...}`. Prediction test and any notebook call the identical C++
code. Strategy-string conventions follow the existing binding style.

### 5. Pre-registered prediction test

`bench/rtrack/v3_gate.py` (bars fixed in code, committed **before**
the test first runs — the standing discipline) + a pytest that walks the
**committed** measured cells:

- Cells: R1 Gen3 matrix (`v1_gen3_nsweep`, b_fair rows), R2/V1 Gen4
  matrix + rsweep (`v1_gen4_*`), R3 multi-GPU scatter/broadcast
  (`r3_*`, `v5_*`), each labeled with machine + pattern.
- Bars (verbatim from #97): winner misclassification **< 15 %** across
  all measured cells; `|r*_pred − r*_meas| ≤ 0.15` absolute on families
  where an r\* exists; regret p90 **< 20 %**, regret
  `= (T_chosen − T_oracle)/T_oracle` from measured times.
- Ablation table: model vs always-A vs always-B vs measured-oracle.
- Cells the model gets wrong are **reported and explained, never
  refit**.

### 6. Compiler-emitted plan row (title-gap closure)

- Generate a **symbolic blocked-transpose** wire plan through the real
  path (`generate_corpus.py` flow: MLIR chain → `sym-opt --reloc-fold` →
  `encodePlan`) — correct by construction, unlike the #63 frozen blob;
  commit it as a corpus-style artifact with its recipe JSON.
- `bench-rtrack` gains a `--plan-wire <file>` workload source: decode →
  `bind({N: 8192}, model)` → run the standard A/B_fair measurement on
  it.
- Acceptance: the wire-driven T1b-equivalent cell matches the
  hand-authored T1b cell within noise (pre-stated tolerance: median
  within the row's IQR), and the model's decision for it matches the
  measured winner. This single row converts "the machinery exists" into
  "the machinery is what we measured".

## Testing

- Unit (CPU CI): calibration parser (valid + malformed + duplicate-key +
  wrong-version), classifier truth table over the six T-plans, decision
  arithmetic on synthetic calibrations, threshold-vs-brute-force
  agreement, prefold-arm delegation.
- Integration: pyreloc `predict` round-trip; prediction pytest over
  committed data (gates in `v3_gate.py`).
- Hardware (this box): only the compiler-emitted row measurement.
- MLIR-free contract: CostModel touches no MLIR includes; CI tests
  enforce as usual.

## Out of scope

- Multi-symbol piecewise-linear decision regions (no plan exercises
  them; fallback covers correctness).
- P5 torch integration; auto-recalibration; scheduling the model into
  the compiler itself (the threshold is computed in libreloc from the
  decoded plan — compiler-side emission of thresholds is future work).
- Changing any R-track verdict.

## Size estimate

~500 LOC C++ (parser+classifier+model+threshold) + ~150 pybind + ~200
tests + ~250 Python (assembler + gate + pytest) + bench row wiring —
consistent with #97's 3–4 day estimate.

## Acceptance (from #97 + review additions)

- Component in-tree, consumed by `bind`; unit tests incl. brute-force
  threshold agreement.
- Prediction test run against R1/R3 (merged) and R2/V1 Gen4 cells with
  the three bars reported; ablation table included.
- Wrong cells reported with explanations.
- Prefold arm delegates to `prefoldWins`; compiler-emitted row measured
  and matching.
