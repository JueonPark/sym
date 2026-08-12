# R6 — Cross-box no-recompile bind demo (issue #87, narrowed)

Date: 2026-08-12. Basis: Build Document v3 §3.3 (R6 축소판) and issue #107's
"Reconciliation with #87" section, which is settled: the held-out
prediction-accuracy half of #87 was absorbed into CM4 (registration,
#112/#121) and CM5 (evaluation, #113/#126). What remains — and what this
design delivers — is #87's second bullet verbatim:

> End-to-end demo: same `sym` relocation plan, symbol-bound at runtime,
> placement chosen automatically by cost model → correct choice on both
> Gen3 and Gen4 boxes without recompilation.

This is the cross-box generalization of V3's wire row
(`docs/v3-costmodel.md` §6), which ran on one box and confirmed one
decision.

## What exists already (verified against the working tree)

- `libreloc/test/corpus/blocked_transpose_sym.bin` — the MLIR-folded
  symbolic wire blob (chain → `sym-opt --reloc-fold` → `encodePlan`),
  the same plan the V3 wire row measured. `pyreloc.load_plan` decodes it
  (`symbols=['N']`) and `pyreloc.bind(plan, {"N": N})` binds at any
  64-divisible N.
- C++ `reloc::bind(plan, syms, strategy, model, wireRatio, K, nReuse)`
  already populates `BoundPlan::decision` (`MethodDecision`: method,
  tAMs/tBMs, thresholdBytes, pattern, bPlacement) when given a
  `costmodel::CostModel*` — built by V3/CM1.
- **The only missing machinery is pybind plumbing**: `pyreloc.bind()`
  does not forward a model, and `BoundPlan.decision` is not exposed.
- Both boxes' calibrations are committed (`calibration/epyc7351-2080ti.cal`,
  `calibration/7800x3d-4070tis.cal`; `make_calibration.py` regen-locked).
- The measured ground truth is committed: BP rsweep CSVs
  (`bench/results/bp_rsweep{,_rerun}_{box}.csv`), with the
  stabler-preference merge implemented in `bench/rtrack/cm5_eval.py`
  (the rule; `gates.py --exp bp`'s merge is explicitly NOT the rule).

## Demo definition

**Plan**: `blocked_transpose_sym.bin` only — "same sym plan" is literal.
**Grid**: r ∈ {0.25, 0.5, 1.0} × N ∈ {2048, 4096, 8192, 16384} × both
boxes = 24 cells. r maps to the measured rsweep wire tiers (0.25 → s8,
0.5 → f16, 1.0 → f32). r = 0.125 (s4) is excluded because the
calibrations carry no blocked s4 gather key — `predict` raises on it
and `bind` leaves the decision unset (see the amended negative-test
bullet below); that boundary belongs to the CM track and gets a
negative test plus a doc note here, not a workaround.
**Decision inputs**: `bind(plan, {"N": N}, model=<box .cal>,
wire_ratio=r)` with defaults k=1, n_reuse=-1. The bind-time decision is
threads=8 + `BPlacement::Overlapped` by construction
(`libreloc/src/Bind.cpp:346` — the deployment default; B's competent
implementation is pipelined, per BP). **The bind-hook decision is the
headline.** The serial (`b_fair`) pairing is a secondary
model-arithmetic check via the already-exposed
`pyreloc.predict(..., b_placement="serial")` — it does not require (and
does not get) a bind-signature change.

**Expected outcomes — computed from committed artifacts before any code
is written, stated here so the demo cannot quietly move them**:

- Overlapped pairing: 23/24 decisions match the stabler-merged measured
  winner. The single miss is (Gen4, r=0.5, N=2048): measured `a` by a
  4.4% margin (0.669 vs 0.700 ms), model says `b`. Small-N WSL2 cell —
  the standing caveat class (BP admissibility gates Gen4 at N ≥ 8192).
  Disclosed, not excluded.
- Serial pairing: same 23/24, same single miss cell (0.669 vs 0.719 ms).
- **The flip row, r = 0.25 (all 8 cells correct)**: Gen3 chooses `b`
  (its host gather is slow relative to its link; shipping raw f32 wins
  ~4× at every N), Gen4 chooses `a` (host gather+quantize wins
  1.32–1.73×). Same wire bytes, same shapes, opposite placements, both
  correct — the cross-box claim in one row.
- r = 1.0 row: both boxes choose `b` at every N (matches measurement) —
  shows the machinery isn't hardwired to flip.

## Components

1. **pybind surface** (`libreloc/python/PyReloc.cpp` + stub/test
   updates): `bind(plan, symbols, strategy="auto", model=None,
   wire_ratio=1.0, k=1, n_reuse=-1)` — new args keyword-only with
   defaults so every existing caller is unaffected; mirrors the C++
   signature exactly. `BoundPlan.decision` → `None` (bound without a
   model) or dict `{method, t_a_ms, t_b_ms, threshold_bytes, pattern,
   b_placement, k, n_reuse}`. No libreloc core changes.
2. **Demo script** (`bench/rtrack/r6_bind_demo.py`): `--machine <name>`
   (no hostname magic; the runbook supplies it) → loads
   `calibration/<machine>.cal` + the corpus blob, binds the 12 (r, N)
   cells under both placements, writes
   `bench/results/r6_bind_demo_<machine>.json` — the bind-hook
   (overlapped) decision per cell plus the serial `predict` check.
   **Fully deterministic**: sorted keys, no timestamps, no hostnames,
   no git revs; contains the machine name, sha256 of the blob and the
   .cal, and per-cell {decision dict, bound extents/strides} (the bound
   geometry is the runtime-symbol-bind evidence). Determinism is what
   makes AC3 a bar.
3. **CI lock test** (`libreloc/python/tests/test_r6_crossbox_bind.py`),
   CPU-only, `importorskip("pyreloc")` like its siblings:
   - byte-equality: regenerate each committed artifact's content
     in-process and compare — skip-with-reason while a box's artifact is
     not yet committed (draft-PR window), strict once present;
   - winner match: bind-hook (overlapped) decisions vs cm5-merged
     winners (winner extraction **imported from `cm5_eval`**, not
     reimplemented — CM3 discipline), asserting exactly 23/24 with the
     pinned miss cell; serial (via `predict`) vs `b_fair` winners
     likewise;
   - flip row asserted explicitly (Gen3 `b` ×4, Gen4 `a` ×4 at r=0.25);
   - negative test at r=0.125 (missing blocked s4 key): `predict`
     raises ValueError; `bind` succeeds with `decision is None`
     (Bind.cpp step 8 is opt-in advice, never a bind failure) —
     amended 2026-08-12 when the as-built behavior was confirmed.
4. **Result doc** (`docs/r6-crossbox-bind.md`): the 24-cell table with
   measured medians and threshold_bytes column (bind-time decide is a
   constant compare — V3's threshold precompute), flip-row narrative,
   the disclosed miss, why there is no #73 gate registration (**no new
   stochastic measurement exists in this track**: decisions are
   deterministic functions of committed calibrations + committed code;
   the measured winners were committed by BP3; acceptance = AC2/AC3,
   both CI-checked), the Gen4 runbook, cross-links (#87, #107
   reconciliation, `cm5_eval_report.json`, V3 §6).
5. **Claim ledger** (`docs/claim-ledger.md`): add the machinery row —
   same folded plan, bind-time auto-placement, correct choice on both
   boxes without recompilation; status `survives`; authoritative source
   = `docs/r6-crossbox-bind.md` + the two artifacts.
6. **Issue hygiene**: edit #87's body per #107's reconciliation (narrow
   to the demo bullet, cross-link CM4/CM5 for the held-out half); close
   #87 when the PR merges with both artifacts.

## Acceptance criteria

- **AC1 (machinery)**: `pyreloc.bind` with a model populates
  `decision`; without one, `decision is None`; `wire_ratio` moves the
  decision (unit tests). Existing bind callers unchanged.
- **AC2 (cross-box correctness)**: decision tables match the committed
  measured winners exactly as stated in "Expected outcomes" — 23/24
  per pairing, pinned miss cell, flip row all-correct.
- **AC3 (no-recompile portability)**: each committed
  `r6_bind_demo_<machine>.json` byte-equals CI's in-process
  regeneration from the committed .cal + blob. The Gen3 artifact is
  produced on `rebel-gpu1`, the Gen4 artifact on the home box (BP3
  runbook pattern); the artifacts contain nothing
  environment-dependent (their only inputs are the committed .cal and
  blob, both sha-pinned inside), so equality proves the same plan +
  same code produce the same decisions everywhere — no per-box
  recompilation or refit is even expressible.
- **AC4 (record)**: result doc + ledger row merged; #87 edited and
  closed.

## Flow (BP3 pattern)

1. pybind extension + unit tests (CI green, no artifacts yet).
2. Demo script; run on `rebel-gpu1` → commit
   `r6_bind_demo_epyc7351-2080ti.json`.
3. Lock test + result doc + ledger row; open **draft PR** (Gen4
   byte-check skips with reason).
4. User runs the one-command runbook on the home box (build pyreloc if
   stale → `python3 bench/rtrack/r6_bind_demo.py --machine
   7800x3d-4070tis` → commit artifact).
5. Gen4 skip disappears in CI; undraft; merge; edit + close #87.

## Out of scope

- r = 0.125 / s4 calibration keys (CM track owns model coverage).
- Any change to model arithmetic, calibrations, bars, or committed
  verdicts; any new measurement (BP3's CSVs are the ground truth).
- Multi-family cells (quant/tiled/single-element): CM5's 48-cell gates
  already cover model quality; adding them here would duplicate that
  and drag in the known Gen4 quant r=1.0 one-sided miss without
  strengthening the machinery claim.
- Executing the chosen placement (transfer + kernels): execution
  fidelity was proven bit-exact by the V3 wire row and BP3's
  `--verify`; the demo's deliverable is the decision.
- torch.compile / FX integration (future work per Build Doc v3 §2.7).

## Risks

1. **Float formatting breaks byte-equality across boxes** — artifact
   floats are C++ doubles serialized by `json.dumps` (repr-based,
   deterministic for identical doubles on CPython ≥ 3.1); the lock test
   regenerating on CI is the continuous check. If a platform ever
   diverges, the bar fails loudly — which is the point.
2. **Gen4 box needs a pyreloc build** — runbook includes the build
   step; the box built the whole tree for BP3.
3. **pybind signature drift** — new args keyword-only with defaults;
   `test_bindings.py` continues to pass unmodified.
