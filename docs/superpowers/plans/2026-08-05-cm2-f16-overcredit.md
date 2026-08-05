# CM2 — f16 Over-credit Attribution + cpu_pipe Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #110 (CM2): document the attribution of the Gen3 r=0.5 ~26% over-credit (in-pipeline DRAM-contention derate) and correct it via per-box `cpu_pipe.*` calibration keys, with a ±0.10 pin test against the committed measurement.

**Architecture:** The spec is `docs/superpowers/specs/2026-08-05-cm2-f16-overcredit-design.md` — read it first. Three small pieces: (1) `cpuBw`'s r=0.5 contiguous branch prefers a new `cpu_pipe.t{threads}.contiguous.convert_f32_f16_gbps` key, falling back to the roofline key; (2) `make_calibration.py` derives that key for both boxes from committed rsweep CSVs (`srcBytes/cpu_stage_ms`, best chunk); (3) `test_prediction.py`'s report output moves to pytest `tmp_path` so the frozen V3 report is never rewritten, plus a self-contained pin test.

**Tech Stack:** C++17 (gtest), Python 3 (make_calibration.py, pytest), pybind11 (pyreloc).

## Global Constraints

- Method A's cost **form** does not change (#107 "Out"): `pathCosts`' `max(aCpuSlope, aDmaSlope)` stays; only the value `cpuBw` returns changes, and only via calibration keys.
- `bench/results/v3_prediction_report.json` must never be rewritten — it is V3's as-measured record (#107). After this change, running the old `test_prediction.py` WOULD rewrite it with moved predictions; Task 2 fixes that before Task 3 lands the keys. Never run the full pytest suite between Task 3's calibration change and Task 2's redirect (the plan orders them safely: Task 2 before Task 3).
- Existing `.cal` lines must not change — new keys are pure insertions; `v3_gate.py` CALIBRATION-REGEN must PASS byte-identically for both machines.
- CPU tests: `ninja -C build/sym libreloc-test` → `build/sym/libreloc/test/libreloc-test`. Python: `ninja -C build/sym` → `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q`. Gate: `python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json` (needs the explicit `--report` arg).
- clang-format clean on touched C++ files (`build/cm1-tools/fmt/bin/clang-format` exists from CM1; if missing: `python3 -m venv build/cm1-tools/fmt && build/cm1-tools/fmt/bin/pip install clang-format`).
- Branch: `cm2-f16-overcredit` (already exists, spec committed). Deliverable: a **regular (non-draft) PR** — no new measurement is needed.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: cpuBw prefers the cpu_pipe key at r=0.5 contiguous

**Files:**
- Modify: `libreloc/src/CostModel.cpp` (the `cpuBw` r=0.5 branch, near line 137)
- Test: `libreloc/test/CostModelTest.cpp`

**Interfaces:**
- Consumes: existing `cpuBw(const CostModel &m, Pattern p, double r, int threads)` and the `kSynth` fixture (`CostModelTest.cpp`, `cpu.t8.contiguous.convert_f32_f16_gbps 20`).
- Produces: `cpuBw` reads key `cpu_pipe.t{threads}.contiguous.convert_f32_f16_gbps` when present (r==0.5, contiguous only). Task 3's calibration keys and pin test rely on this exact key spelling.

- [ ] **Step 1: Write the failing test**

Append to `libreloc/test/CostModelTest.cpp` (after `CostModelCpuBw.FigureRstarComposition`):

```cpp
TEST(CostModelCpuBw, PipeKeyPreferredAtR05Contiguous) {
  // Issue #110 (CM2): the isolated convert_f32_f16 roofline over-credits
  // A's r=0.5 stage under concurrent DMA (in-pipeline host-DRAM
  // contention). When the calibration carries the measured in-pipeline
  // value, cpuBw must prefer it; absent, the roofline fallback keeps
  // current behavior.
  CostModel base = mustParse(kSynth);
  EXPECT_DOUBLE_EQ(*cpuBw(base, Pattern::Contiguous, 0.5, 8), 20.0);
  CostModel withPipe = mustParse(
      std::string(kSynth) +
      "cpu_pipe.t8.contiguous.convert_f32_f16_gbps 10\n");
  EXPECT_DOUBLE_EQ(*cpuBw(withPipe, Pattern::Contiguous, 0.5, 8), 10.0);
  // Scope guards: other tiers and non-contiguous r=0.5 are untouched by
  // the pipe key (blocked r=0.5 stays harmonic(gather 5, convert 20)=4).
  EXPECT_DOUBLE_EQ(*cpuBw(withPipe, Pattern::Contiguous, 1.0, 8), 20.0);
  EXPECT_DOUBLE_EQ(*cpuBw(withPipe, Pattern::Blocked, 0.5, 8), 4.0);
  // Thread count is part of the key: t1 has no pipe key -> nullopt path
  // unchanged (kSynth has no t1 keys at all).
  EXPECT_FALSE(cpuBw(withPipe, Pattern::Contiguous, 0.5, 1).has_value());
  // And the preferred value flows through pathCosts' A slope:
  // aCpuSlope = 1e-6/10 = 1e-7 > aDmaSlope = 0.5e-6/10 = 5e-8.
  auto pc = pathCosts(withPipe, Pattern::Contiguous, 1000000000, 0.5, 8);
  ASSERT_TRUE(pc.has_value());
  EXPECT_DOUBLE_EQ(pc->aSlopeMsPerByte, 1e-7);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build/sym libreloc-test && build/sym/libreloc/test/libreloc-test --gtest_filter='*PipeKey*'`
Expected: FAIL — `cpuBw(withPipe, …, 0.5, 8)` returns 20.0 (roofline), not 10.0.

- [ ] **Step 3: Implement**

In `libreloc/src/CostModel.cpp`, change the r=0.5 branch of `cpuBw` (currently `if (r == 0.5) { auto conv = need("convert_f32_f16"); if (contig) return conv; … }`) to:

```cpp
  if (r == 0.5) {
    if (contig) {
      // Issue #110 (CM2): prefer the measured in-pipeline bandwidth --
      // the isolated roofline over-credits the convert stage under
      // concurrent DMA (mutual host-DRAM contention; on the EPYC box
      // both overlapped stages derate ~16% vs their isolated rates).
      // Absent key -> roofline fallback, current behavior.
      const std::string pipeKey = "cpu_pipe.t" + std::to_string(threads) +
                                  ".contiguous.convert_f32_f16_gbps";
      if (m.has(pipeKey))
        return m.at(pipeKey);
      return need("convert_f32_f16");
    }
    auto conv = need("convert_f32_f16");
    auto g = need("gather_f32");
    if (!g || !conv)
      return std::nullopt;
    return harmonic(*g, *conv);
  }
```

(Note the non-contig path keeps its exact current arithmetic; only the contig early-return gains the preference.)

- [ ] **Step 4: Run the full C++ suite**

Run: `ninja -C build/sym libreloc-test && build/sym/libreloc/test/libreloc-test 2>&1 | tail -3`
Expected: ALL PASS (no committed calibration carries the key yet, so every existing test sees fallback behavior).

- [ ] **Step 5: Commit**

```bash
git add libreloc/src/CostModel.cpp libreloc/test/CostModelTest.cpp
git commit -m "feat(libreloc): cpuBw prefers in-pipeline cpu_pipe key at r=0.5 contiguous (#110)"
```

---

### Task 2: test_prediction.py writes to tmp_path (frozen-report guard)

**Files:**
- Modify: `libreloc/python/tests/test_prediction.py:21-24, 331-341`

**Interfaces:**
- Consumes: nothing from other tasks (deliberately ordered BEFORE Task 3's calibration change so the frozen report can never be dirtied by a pytest run).
- Produces: `test_write_prediction_report(tmp_path)` — the committed `bench/results/v3_prediction_report.json` is never written again by any test.

- [ ] **Step 1: Check for external references**

Run: `grep -rn "REPORT_PATH\|test_prediction" libreloc/ bench/ --include='*.py' | grep -v test_prediction.py`
Expected: only `bench/rtrack/v3_gate.py` mentions the report (by its own path constant, not by importing test_prediction). If anything imports `REPORT_PATH` from test_prediction.py, STOP and report.

- [ ] **Step 2: Make the change**

In `libreloc/python/tests/test_prediction.py`:

1. Delete the module-level constant `REPORT_PATH = RESULTS / "v3_prediction_report.json"` (line 24).
2. Replace `test_write_prediction_report` (lines 331-341) with:

```python
def test_write_prediction_report(tmp_path):
    # CM2 (#110): the report is written to a scratch path, never to the
    # committed bench/results/v3_prediction_report.json -- that file is
    # V3's as-measured record against the model that produced it (#107;
    # v3_gate.py's REPORT-REGEN guards it). Re-registering predictions
    # against the corrected model is #CM4's job, not a pytest side
    # effect.
    report = build_report()
    report_path = tmp_path / "v3_prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Structural sanity only -- the BARS are judged by v3_gate.py, never
    # here (measurement and judgment stay separate).
    assert report_path.exists()
    assert report["summary"]["n_modelable"] >= 30, (
        f"only {report['summary']['n_modelable']} modelable cells "
        "(expected >= 30)")
```

3. In the module docstring (lines 1-11), change "writes `bench/results/v3_prediction_report.json` -- the single source `bench/rtrack/v3_gate.py` judges against" to "writes the report to a pytest scratch path; the committed `bench/results/v3_prediction_report.json` is V3's frozen as-measured record (#107) that `bench/rtrack/v3_gate.py` judges/guards".

- [ ] **Step 3: Run and verify the tree stays clean**

```bash
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/test_prediction.py -q
git status --short bench/results/
```
Expected: test passes; `git status` prints nothing for bench/results/ (the committed report untouched).

- [ ] **Step 4: Commit**

```bash
git add libreloc/python/tests/test_prediction.py
git commit -m "test(pyreloc): prediction report writes to tmp_path -- frozen V3 report never rewritten (#110)"
```

---

### Task 3: cpu_pipe calibration keys + pin test

**Files:**
- Modify: `bench/rtrack/make_calibration.py` (new shared helper + two call sites + SOURCES docstring)
- Modify (regenerated): `calibration/epyc7351-2080ti.cal`, `calibration/7800x3d-4070tis.cal`
- Create: `libreloc/python/tests/test_f16_overcredit_pin.py`

**Interfaces:**
- Consumes: Task 1's key spelling `cpu_pipe.t8.contiguous.convert_f32_f16_gbps`; Task 2's guarantee that pytest cannot dirty the frozen report; existing helpers `load_csv_rows`, `source_bytes`, `Emitter.emit` in make_calibration.py.
- Produces: the key in both `.cal` files (epyc ≈14.48, gen4 ≈32.02); the acceptance pin test.

- [ ] **Step 1: Add the shared helper** to `bench/rtrack/make_calibration.py` (next to `emit_recv_from_cm1_run`):

```python
def emit_cpu_pipe_convert(e, csv_path):
    """cpu_pipe.t8.contiguous.convert_f32_f16_gbps (issue #110/CM2): the
    convert stage's IN-PIPELINE effective bandwidth -- srcBytes /
    cpu_stage_ms of the best-chunk (min median_ms) method=a,
    transform=quant, r=0.5, N=16384, t8 row. The isolated roofline
    over-credits this stage under concurrent DMA (mutual host-DRAM
    contention; prior art docs/poc-reproduction-v2.md). Sourced from the
    same rsweep file that feeds the frozen r* comparison for this box,
    so the model is judged against the regime it was calibrated in."""
    rows = [r for r in load_csv_rows(csv_path, e.machine)
            if r["method"] == "a" and r["transform"] == "quant"
            and float(r["r"]) == 0.5 and int(r["N"]) == 16384
            and int(r["threads"]) == 8]
    if not rows:
        sys.exit(f"error: no method=a quant r=0.5 N=16384 t8 rows in "
                 f"{csv_path}")
    best = min(rows, key=lambda r: float(r["median_ms"]))
    bw = source_bytes(16384) / (float(best["cpu_stage_ms"]) * 1e-3) / 1e9
    e.emit("cpu_pipe.t8.contiguous.convert_f32_f16_gbps", round(bw, 2),
           csv_path,
           note=f"srcBytes/cpu_stage_ms, best chunk "
                f"({best['chunk_req_mib']} MiB), in-pipeline vs isolated "
                f"roofline (issue #110 contention derate)")
```

- [ ] **Step 2: Wire into both builders**

In `build_epyc`, immediately after the recv-m block (after the `emit_recv_from_cm1_run(...)` call), add:

```python
    emit_cpu_pipe_convert(
        e, f"{RESULTS}/v2_isa_gen3_rsweep_avx2_epyc7351-2080ti.csv")
```

In `build_gen4`, immediately after its `emit_recv_from_cm1_run(...)` call, add:

```python
    emit_cpu_pipe_convert(
        e, f"{RESULTS}/v1_gen4_rsweep_7800x3d_4070tis.csv")
```

Add to the SOURCES docstring under BOTH machines:

```
    - cpu_pipe.t8.contiguous.convert_f32_f16_gbps
                                   <- <that box's rsweep CSV named above>
      (cpu_stage_ms of the best-chunk a/quant/r=0.5/N=16384 row -- the
      in-pipeline effective convert BW; issue #110 contention derate.
      Same file as the box's frozen r* comparison source.)
```

- [ ] **Step 3: Regenerate both files + verify additive-only and expected values**

```bash
python3 bench/rtrack/make_calibration.py --machine epyc7351-2080ti --out calibration/epyc7351-2080ti.cal
python3 bench/rtrack/make_calibration.py --machine 7800x3d-4070tis --out calibration/7800x3d-4070tis.cal
git diff -U0 calibration/ | grep '^-' | grep -v '^---'   # MUST print nothing
git diff calibration/ | grep '^+' | grep -v '^+++'
```

Expected: exactly one inserted line per file — epyc `cpu_pipe.t8.contiguous.convert_f32_f16_gbps 14.48` (= 1073741824 B / 74.1677 ms), gen4 `… 32.02` (= 1073741824 B / 33.5301 ms). Different values → STOP and reconcile the row-selection rule (do not accept silently).

- [ ] **Step 4: Write the pin test**

Create `libreloc/python/tests/test_f16_overcredit_pin.py`:

```python
"""CM2 (issue #110): pins the corrected Gen3 r=0.5 quant prediction
within +-0.10 of the committed measurement. Self-contained -- loads only
the epyc calibration and reads only the committed
v2_isa_gen3_rstar_avx2 JSON; never touches test_prediction.py's report
writer (bench/results/v3_prediction_report.json is V3's frozen
as-measured record, guarded by v3_gate.py REPORT-REGEN)."""
import json
import pathlib

import pytest

pyreloc = pytest.importorskip("pyreloc")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_gen3_r05_quant_prediction_within_tolerance():
    cal = pyreloc.load_calibration(
        str(REPO_ROOT / "calibration" / "epyc7351-2080ti.cal"))
    doc = json.loads(
        (REPO_ROOT / "bench" / "results" /
         "v2_isa_gen3_rstar_avx2_epyc7351-2080ti.json").read_text())
    measured = doc["families"]["quant"]["speedup_measured"]["0.5"]
    assert measured == pytest.approx(1.0601163292018307)  # committed value

    src_bytes = 16384 * 16384 * 4
    # Same quantity as docs/v3-costmodel.md S4's table: speedup(r) =
    # t_b_pred(r=1.0) / t_a_pred(r).
    t_b1 = pyreloc.predict(cal, pattern="contiguous", src_bytes=src_bytes,
                           r=1.0)["t_b_ms"]
    t_a = pyreloc.predict(cal, pattern="contiguous", src_bytes=src_bytes,
                          r=0.5)["t_a_ms"]
    predicted = t_b1 / t_a
    # Pre-fix this was 1.3317 (|delta| 0.272); the cpu_pipe key (14.48
    # GB/s) brings it to ~1.107 (|delta| ~0.047). The residual is the
    # unmodelled A-side recv drain, documented in the CM2 spec.
    assert abs(predicted - measured) <= 0.10, (predicted, measured)
```

- [ ] **Step 5: Run everything**

```bash
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
git status --short   # bench/results/ must stay clean (Task 2's guarantee)
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
```

Expected: full pytest green (pin test included), clean tree, C++ suite green, CALIBRATION-REGEN PASS ×2 + REPORT-REGEN PASS. (The gate's V3 verdict lines will still show the historical RSTAR FAIL 0.363 — that is the frozen record, correct and expected.)

- [ ] **Step 6: Commit**

```bash
git add bench/rtrack/make_calibration.py calibration/ libreloc/python/tests/test_f16_overcredit_pin.py
git commit -m "feat(bench, pyreloc): cpu_pipe in-pipeline convert keys + Gen3 r=0.5 pin test (#110)"
```

---

### Task 4: Attribution comment, clang-format, PR

**Files:**
- No repo file changes beyond possible clang-format reflows.

**Interfaces:**
- Consumes: everything above.
- Produces: the attribution comment on issue #110 (acceptance: "attribution documented") and the PR.

- [ ] **Step 1: Post the attribution comment on issue #110**

`gh issue comment 110 --repo JueonPark/sym --body "<the text below>"`:

```markdown
**Attribution (recorded per the issue's first task), evidence in the PR/spec:**

Both named candidates are refuted:
- (a) *wrong bandwidth key*: `cpuBw` at r=0.5 contiguous already reads
  `cpu.t8.contiguous.convert_f32_f16_gbps` (17.42 on Gen3), not `contig_read`
  (that key is consulted only at r=1.0). Substituting contig_read moves the
  prediction to 1.997 — away from the measured 1.060.
- (b) *convert cannot overlap its own DMA*: the harness is genuinely
  double-buffered (`rtrack_bench.cu` gates chunk c's CPU stage on
  `h2dEnd[c-2]`), and the measured decomposition shows overlap:
  cpu_stage 74.17 ms + h2d 48.63 ms = 122.8 ms against a 77.47 ms wall.
  The serialized counterfactual predicts 0.80 — wrong the other way.

**Confirmed attribution: in-pipeline DRAM-contention derate.** The model
credits the CPU convert stage with its *isolated* roofline; in the live
pipeline both overlapped stages compete for the same host DRAM (EPYC 7351,
4 of 8 channels, triad 44 GB/s) and both derate ~16% in lockstep — CPU
17.42→14.48 GB/s, DMA 13.07→11.04 GB/s. Prior art for the same effect:
`docs/poc-reproduction-v2.md` (in-pipeline gather 9.5 vs isolated 14.3 GB/s).

Corollaries recorded for CM4/CM5: the A under-prediction is monotone in
concurrent CPU work, not f16-specific (Gen3 +12.7% at r=1.0 [DMA-bound],
+18.8% at r=0.25, +25.5% at r=0.5); Gen4's sign is reversed (in-pipeline
convert 32.02 GB/s > its 26.20 roofline), so the correction is a per-box
calibrated value; A's post-DMA recv drain (~3.3 ms here) stays unmodelled —
a term would change A's cost form (#107 Out).

Fix (per the attribution): `cpu_pipe.t8.contiguous.convert_f32_f16_gbps`
keys derived from the committed rsweep `cpu_stage_ms` (Gen3 14.48, Gen4
32.02), preferred by `cpuBw` at r=0.5 contiguous. Corrected Gen3 prediction
1.107 vs measured 1.060 → |Δ| ≈ 0.047 ≤ 0.10 (pinned in
`libreloc/python/tests/test_f16_overcredit_pin.py`).
```

- [ ] **Step 2: clang-format + final verification**

```bash
build/cm1-tools/fmt/bin/clang-format -i libreloc/src/CostModel.cpp libreloc/test/CostModelTest.cpp
git diff --stat   # if anything reflowed, re-run the C++ suite, then commit the reflow
build/sym/libreloc/test/libreloc-test 2>&1 | tail -3
PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q
python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json 2>&1 | grep -E 'REGEN.*(PASS|FAIL)'
git status --short   # clean tree required
```

- [ ] **Step 3: Push and open the PR (regular, not draft)**

```bash
git push -u origin cm2-f16-overcredit
gh pr create --title "feat(libreloc, bench): CM2 — f16 over-credit attribution + cpu_pipe correction (#110)" --body "<body>"
```

PR body must contain, in order: (1) verdict-first: corrected Gen3 r=0.5 prediction 1.107 vs measured 1.060, |Δ|=0.047 ≤ 0.10, pinned by `test_f16_overcredit_pin.py`; CALIBRATION-REGEN PASS ×2, REPORT-REGEN PASS, frozen report untouched; (2) the attribution summary (link the #110 comment) including both candidate refutations; (3) what changed: `cpuBw` preference (form unchanged, #107-compliant), two `.cal` insertions (14.48/32.02 with provenance), `test_prediction.py` → tmp_path with the frozen-record rationale; (4) out-of-scope residuals recorded for CM4/CM5 (r=1.0/r=0.25 same-mechanism residuals, recv drain, Gen4's remaining reversed-sign misses); (5) `Refs #110, #107`; (6) the standard generated-with footer.

---

## Verification (end-to-end, after all tasks)

1. `build/sym/libreloc/test/libreloc-test` — all pass (baseline + `PipeKeyPreferredAtR05Contiguous`).
2. `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/ -q` — all pass including the pin test; `git status` clean afterward (frozen report never rewritten).
3. `python3 bench/rtrack/v3_gate.py --report bench/results/v3_prediction_report.json` — CALIBRATION-REGEN PASS ×2, REPORT-REGEN PASS.
4. `git diff main -- calibration/ | grep '^-' | grep -v '^---'` prints nothing (pure insertions, one line per box).
5. Issue #110 acceptance: attribution documented (issue comment + spec + PR body); corrected prediction within ±0.10 on existing data only (the pin test; no new measurement anywhere).
