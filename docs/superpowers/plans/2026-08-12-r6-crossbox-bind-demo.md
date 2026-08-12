# R6 Cross-box No-recompile Bind Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demonstrate #87's narrowed claim — the same MLIR-folded sym plan, symbol-bound at runtime with each box's calibration as the bind model, yields the correct (measured-winner-matching) placement on both Gen3 and Gen4 without recompilation — including the cross-box flip at r=0.25.

**Architecture:** Plumb the existing C++ `reloc::bind(plan, syms, strategy, model, wireRatio, K, nReuse)` model hook through pybind (`pyreloc.bind` gains keyword-only `model/wire_ratio/k/n_reuse`; `BoundPlan.decision` exposed). A deterministic demo script binds the committed corpus blob over r ∈ {0.25, 0.5, 1.0} × N ∈ {2048, 4096, 8192, 16384} and writes `bench/results/r6_bind_demo_<machine>.json`. A CI test regenerates both boxes' tables in-process, checks them against the stabler-merged BP measured winners (merge rule imported from `cm5_eval`), and byte-compares committed artifacts.

**Tech Stack:** pybind11 (existing `pyreloc_ext` target), pytest (existing CI job `pytest libreloc/python/tests`), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md` (committed on this branch).

## Global Constraints

- No libreloc core (`libreloc/src`, `libreloc/include`) changes — pybind layer only.
- No changes to model arithmetic, calibrations, bars, committed verdicts, or measured CSVs; no new measurements.
- New `pyreloc.bind` args are **keyword-only with defaults**; every existing caller must pass unmodified (`test_bindings.py` untouched and green).
- Demo artifacts are **fully deterministic**: sorted keys, no timestamps/hostnames/git revs; inputs sha256-pinned inside the artifact.
- Winner extraction is **imported from `bench/rtrack/cm5_eval.py`** (`load_rows`, `merge_points`, `fmt_r`) — never reimplemented (CM3 single-implementation discipline).
- Expected outcomes are fixed by the spec: 23/24 correct per pairing; the only miss is (7800x3d-4070tis, r=0.5, N=2048); flip row r=0.25 = Gen3 `b` ×4, Gen4 `a` ×4.
- Build: `ninja -C build/sym pyreloc_ext`; run tests with `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests -q`.
- This box is `rebel-gpu1` = machine name `epyc7351-2080ti` (Gen3). The Gen4 artifact is produced by the user on the home box via the runbook (Task 6).
- Branch: `r6-crossbox-bind-demo`. Repo merges are squash-merges titled like existing history (e.g. `feat(pyreloc, bench): R6 — cross-box no-recompile bind demo (#87)`).

---

### Task 1: pybind bind-with-model surface

**Files:**
- Modify: `libreloc/python/PyReloc.cpp:127-134` (bindPlan wrapper), `:263-290` (BoundPlan class — add `decision` property), `:292-295` (move + extend the `m.def("bind", ...)`)
- Test: `libreloc/python/tests/test_bind_decision.py` (new)
- Modify: `docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md` (amendment paragraph)

**Interfaces:**
- Consumes: `reloc::bind(plan, symbolMap, Strategy, const costmodel::CostModel*, double wireRatio, int K, int64_t nReuse)` (`libreloc/include/reloc/Bind.h:99-102`, exists); `BoundPlan::decision` (`std::optional<costmodel::MethodDecision>`, `Bind.h:80`).
- Produces: `pyreloc.bind(plan, symbols, strategy="auto", *, model=None, wire_ratio=1.0, k=1, n_reuse=-1) -> BoundPlan`; `BoundPlan.decision -> None | dict` with keys `method` (`"a"|"b"|"a_prefold"`), `t_a_ms`, `t_b_ms`, `threshold_bytes` (floats), `pattern` (`"contiguous"|"blocked"|"single_element"|"tiled"`), `b_placement` (always `"overlapped"` from bind — `Bind.cpp:346`), `k` (int), `n_reuse` (int). Tasks 2–4 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

Create `libreloc/python/tests/test_bind_decision.py`:

```python
"""R6 (issue #87): pyreloc bind-with-model surface (spec AC1).

The C++ bind() has accepted an optional CostModel* since V3/CM1
(libreloc/src/Bind.cpp step 8: threads=8, BPlacement::Overlapped —
the deployment default; the library's real B is the double-buffered
pipeline). These tests cover the pybind plumbing added for the R6 demo.
Committed calibrations are the fixtures (test_wire_row_decision.py
precedent)."""
import pathlib

import pytest

pyreloc = pytest.importorskip("pyreloc")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CORPUS = (REPO_ROOT / "libreloc" / "test" / "corpus" /
          "blocked_transpose_sym.bin")
GEN3_CAL = REPO_ROOT / "calibration" / "epyc7351-2080ti.cal"
GEN4_CAL = REPO_ROOT / "calibration" / "7800x3d-4070tis.cal"


@pytest.fixture(scope="module")
def plan():
    return pyreloc.load_plan(CORPUS.read_bytes())


def test_bind_without_model_decision_is_none(plan):
    bound = pyreloc.bind(plan, {"N": 8192})
    assert bound.decision is None


def test_bind_with_model_populates_decision(plan):
    cal = pyreloc.load_calibration(str(GEN3_CAL))
    bound = pyreloc.bind(plan, {"N": 8192}, model=cal, wire_ratio=1.0)
    d = bound.decision
    assert d is not None
    assert d["method"] == "b"           # Gen3 blocked r=1.0 — the V3 wire row
    assert d["pattern"] == "blocked"    # classified from the bound plan
    assert d["b_placement"] == "overlapped"   # Bind.cpp:346
    assert d["t_a_ms"] > 0 and d["t_b_ms"] > 0
    assert d["k"] == 1 and d["n_reuse"] == -1
    assert "threshold_bytes" in d


def test_wire_ratio_moves_the_decision(plan):
    # Gen4 blocked: r* (overlapped) = 0.3605 measured / 0.2914 predicted
    # (claim ledger) — r=0.25 sits on the A side, r=1.0 on the B side.
    cal = pyreloc.load_calibration(str(GEN4_CAL))
    assert pyreloc.bind(plan, {"N": 8192}, model=cal,
                        wire_ratio=0.25).decision["method"] == "a"
    assert pyreloc.bind(plan, {"N": 8192}, model=cal,
                        wire_ratio=1.0).decision["method"] == "b"


def test_missing_tier_key_leaves_decision_none(plan):
    # No blocked s4 (r=0.125) key exists in either calibration: predict
    # raises, but bind succeeds with decision unset — Bind.cpp step 8 is
    # opt-in advice, never a bind failure.
    cal = pyreloc.load_calibration(str(GEN3_CAL))
    with pytest.raises(ValueError):
        pyreloc.predict(cal, pattern="blocked", src_bytes=8192 * 8192 * 4,
                        r=0.125)
    bound = pyreloc.bind(plan, {"N": 8192}, model=cal, wire_ratio=0.125)
    assert bound.strategy       # bound fine
    assert bound.decision is None


def test_bind_rejects_positional_model(plan):
    cal = pyreloc.load_calibration(str(GEN3_CAL))
    with pytest.raises(TypeError):
        pyreloc.bind(plan, {"N": 8192}, "auto", cal)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/test_bind_decision.py -v`
Expected: FAIL — `test_bind_without_model_decision_is_none` with `AttributeError: 'BoundPlan' object has no attribute 'decision'`; the model-kwarg tests with `TypeError: bind(): incompatible function arguments`.

- [ ] **Step 3: Implement the pybind changes**

In `libreloc/python/PyReloc.cpp`:

(a) Replace the `bindPlan` wrapper (currently lines 127–134):

```cpp
reloc::BoundPlan bindPlan(const reloc::RelocationPlan &plan,
                          const std::map<std::string, int64_t> &symbols,
                          const std::string &strategy,
                          const reloc::costmodel::CostModel *model,
                          double wireRatio, int k, int64_t nReuse) {
  auto result = reloc::bind(plan, symbols, parseStrategy(strategy), model,
                            wireRatio, k, nReuse);
  if (auto *err = std::get_if<reloc::BindError>(&result))
    throw BindException(err->message);
  return std::get<reloc::BoundPlan>(std::move(result));
}
```

(b) In the `py::class_<reloc::BoundPlan>` chain (before `.def("__repr__", ...)`), add:

```cpp
      .def_property_readonly(
          "decision",
          [](const reloc::BoundPlan &b) -> py::object {
            if (!b.decision)
              return py::none();
            const reloc::costmodel::MethodDecision &d = *b.decision;
            py::dict out;
            out["method"] = std::string(reloc::costmodel::methodName(d.method));
            out["t_a_ms"] = d.tAMs;
            out["t_b_ms"] = d.tBMs;
            out["threshold_bytes"] = d.thresholdBytes;
            out["pattern"] =
                std::string(reloc::costmodel::patternName(d.pattern));
            out["b_placement"] =
                std::string(reloc::costmodel::placementName(d.bPlacement));
            out["k"] = d.k;
            out["n_reuse"] = d.nReuse;
            return std::move(out);
          },
          "Cost-model decision populated when bind() was given a model and "
          "the calibration had the needed pattern/r keys; None otherwise "
          "(bind-time pricing is t8 + Overlapped -- Bind.cpp step 8).")
```

(c) Delete the current `m.def("bind", ...)` at lines 292–295 and re-add it **after** the `load_calibration` def (so the `Calibration` type is registered before `bind`'s signature references it):

```cpp
  m.def("bind", &bindPlan, py::arg("plan"), py::arg("symbols"),
        py::arg("strategy") = "auto", py::kw_only(),
        py::arg("model") =
            static_cast<const reloc::costmodel::CostModel *>(nullptr),
        py::arg("wire_ratio") = 1.0, py::arg("k") = 1,
        py::arg("n_reuse") = -1,
        "Bind a plan against {symbol: value}. Raises BindError on symbol "
        "mismatch or violated correctness constraints. With `model`, "
        "populates BoundPlan.decision (bind-time t8/Overlapped pricing); "
        "decision stays None if the calibration lacks the needed keys.");
```

Check the include list at the top of the file: `reloc/CostModel.h` (or whichever header `predict` already uses for `costmodel::decide`) must be present — it is, since `predict` compiles; add nothing new unless the compiler complains about `MethodDecision`, in which case add `#include "reloc/MethodDecision.h"`.

- [ ] **Step 4: Rebuild and run the new tests**

Run: `ninja -C build/sym pyreloc_ext && PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/test_bind_decision.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Run the full python test suite (regression gate)**

Run: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests -q`
Expected: everything passes; `test_bindings.py` unmodified. If clang-format is available, run `clang-format --dry-run libreloc/python/PyReloc.cpp` and hand-fix only lines this task touched (do not reformat the rest of the file).

- [ ] **Step 6: Amend the spec's negative-test sentence**

In `docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md`, replace the line:

```
   - negative test: r=0.125 raises (missing blocked s4 key).
```

with:

```
   - negative test at r=0.125 (missing blocked s4 key): `predict`
     raises ValueError; `bind` succeeds with `decision is None`
     (Bind.cpp step 8 is opt-in advice, never a bind failure) —
     amended 2026-08-12 when the as-built behavior was confirmed.
```

- [ ] **Step 7: Commit**

```bash
git add libreloc/python/PyReloc.cpp libreloc/python/tests/test_bind_decision.py docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md
git commit -m "feat(pyreloc): R6 — bind-with-model surface, BoundPlan.decision (#87)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Demo script + Gen3 artifact

**Files:**
- Create: `bench/rtrack/r6_bind_demo.py`
- Create (generated): `bench/results/r6_bind_demo_epyc7351-2080ti.json`

**Interfaces:**
- Consumes: Task 1's `pyreloc.bind(..., model=, wire_ratio=)` + `BoundPlan.decision`; `pyreloc.predict(cal, pattern=, src_bytes=, r=, b_placement="serial")` (exists).
- Produces: `build_demo(machine: str) -> dict` and `render(report: dict) -> str` (module-importable, used by Task 3's regen twin); artifact schema `{generated_by, issue, machine, inputs: {relpath: sha256}, cells: [{r, N, bound: {extents, src_strides, dst_strides, total_bytes, strategy}, bind_decision: <decision dict>, serial_check: <predict dict>}]}`.

- [ ] **Step 1: Write the script**

Create `bench/rtrack/r6_bind_demo.py`:

```python
#!/usr/bin/env python3
# bench/rtrack/r6_bind_demo.py
"""R6 (issue #87): cross-box no-recompile bind demo.

Loads the MLIR-folded symbolic wire blob
(libreloc/test/corpus/blocked_transpose_sym.bin -- the V3 wire-row plan),
binds it at runtime symbol values N in {2048, 4096, 8192, 16384} with
this box's committed calibration as the bind model, and records the
bind-time placement decision (t8 + Overlapped -- libreloc/src/Bind.cpp
step 8) plus a serial-priced predict() check per cell, for wire ratios
r in {0.25, 0.5, 1.0} (the measured rsweep tiers s8/f16/f32; r=0.125 is
excluded -- no blocked s4 calibration key exists, a CM-track boundary,
see docs/r6-crossbox-bind.md).

The output is FULLY DETERMINISTIC: sorted keys, no timestamps, no
hostnames, no git revs; inputs sha256-pinned inside the artifact.
Byte-equality between the artifact a box commits and a CI in-process
regeneration is the demo's no-recompile portability bar (spec AC3).

  PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
      --machine epyc7351-2080ti
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import pyreloc
except ImportError:
    sys.exit("error: pyreloc not importable -- build it (ninja -C build/sym "
             "pyreloc_ext) and run with PYTHONPATH=build/sym/python")

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = (REPO_ROOT / "libreloc" / "test" / "corpus" /
          "blocked_transpose_sym.bin")
MACHINES = ("epyc7351-2080ti", "7800x3d-4070tis")
R_GRID = (0.25, 0.5, 1.0)      # measured rsweep wire tiers s8/f16/f32
N_GRID = (2048, 4096, 8192, 16384)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_demo(machine):
    cal_path = REPO_ROOT / "calibration" / f"{machine}.cal"
    cal = pyreloc.load_calibration(str(cal_path))
    plan = pyreloc.load_plan(CORPUS.read_bytes())
    cells = []
    for r in R_GRID:
        for n in N_GRID:
            bound = pyreloc.bind(plan, {"N": n}, model=cal, wire_ratio=r)
            decision = bound.decision
            assert decision is not None, f"no decision at r={r} N={n}"
            serial = pyreloc.predict(
                cal, pattern=decision["pattern"],
                src_bytes=bound.total_bytes, r=r, b_placement="serial")
            cells.append({
                "r": r, "N": n,
                "bound": {
                    "extents": list(bound.extents),
                    "src_strides": list(bound.src_strides),
                    "dst_strides": list(bound.dst_strides),
                    "total_bytes": bound.total_bytes,
                    "strategy": bound.strategy,
                },
                "bind_decision": decision,
                "serial_check": serial,
            })
    return {
        "generated_by": "bench/rtrack/r6_bind_demo.py",
        "issue": "#87 (R6): cross-box no-recompile bind demo",
        "machine": machine,
        "inputs": {
            str(CORPUS.relative_to(REPO_ROOT)): sha256(CORPUS),
            str(cal_path.relative_to(REPO_ROOT)): sha256(cal_path),
        },
        "cells": cells,
    }


def render(report):
    return json.dumps(report, indent=1, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="R6 cross-box bind demo (issue #87)")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: bench/results/r6_bind_demo_<machine>.json")
    args = ap.parse_args()
    out = args.out or (REPO_ROOT / "bench" / "results" /
                       f"r6_bind_demo_{args.machine}.json")
    report = build_demo(args.machine)
    out.write_text(render(report))
    for c in report["cells"]:
        d = c["bind_decision"]
        print(f"r={c['r']:<5g} N={c['N']:<6d} -> {d['method']}  "
              f"(t_a={d['t_a_ms']:.4g} ms, t_b={d['t_b_ms']:.4g} ms, "
              f"threshold_bytes={d['threshold_bytes']:.6g})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on this box (Gen3) and check determinism**

```bash
PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py --machine epyc7351-2080ti
PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py --machine epyc7351-2080ti --out /tmp/claude-2017/-home-jueonpark-sym/39763a14-843a-4b74-9323-36fd876da897/scratchpad/r6_twin.json
diff bench/results/r6_bind_demo_epyc7351-2080ti.json /tmp/claude-2017/-home-jueonpark-sym/39763a14-843a-4b74-9323-36fd876da897/scratchpad/r6_twin.json && echo DETERMINISTIC
```

Expected: printed table shows `b` for all 12 Gen3 cells (Gen3 blocked never crosses — V2); `DETERMINISTIC` prints.

- [ ] **Step 3: Sanity-check the Gen4 table from the committed calibration (no artifact written)**

```bash
PYTHONPATH=build/sym/python python3 -c "
import sys; sys.path.insert(0, 'bench/rtrack')
import r6_bind_demo
rep = r6_bind_demo.build_demo('7800x3d-4070tis')
print([ (c['r'], c['N'], c['bind_decision']['method']) for c in rep['cells'] ])"
```

Expected: `a` at all four N for r=0.25; `b` everywhere at r=0.5 and r=1.0 (the model's known miss at (0.5, 2048) is against the *measurement*, not visible here).

- [ ] **Step 4: Commit (script + Gen3 artifact only — the Gen4 artifact is produced on the Gen4 box, Task 6)**

```bash
git add bench/rtrack/r6_bind_demo.py bench/results/r6_bind_demo_epyc7351-2080ti.json
git commit -m "bench(rtrack, results): R6 — bind demo script + Gen3 artifact (#87)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Cross-box CI lock test

**Files:**
- Test: `libreloc/python/tests/test_r6_crossbox_bind.py` (new)

**Interfaces:**
- Consumes: `r6_bind_demo.build_demo/render` (Task 2); `cm5_eval.load_rows/merge_points/fmt_r` (exists); committed `bench/results/bp_rsweep{,_rerun}_{box}.csv`; committed `bench/results/r6_bind_demo_<machine>.json` (Gen3 from Task 2; Gen4 pending until Task 6 — that byte-check must **skip with reason**, not fail).
- Produces: nothing downstream; this is the AC2+AC3 lock.

- [ ] **Step 1: Write the test file**

Create `libreloc/python/tests/test_r6_crossbox_bind.py`:

```python
"""R6 (issue #87): cross-box no-recompile bind demo lock (spec AC2+AC3).

AC2: bind-hook decision tables derived from the two committed
calibrations match the stabler-merged BP measured winners
(bench/results/bp_rsweep*.csv; merge rule imported from cm5_eval --
CM3 single-implementation discipline) at 23/24 cells per pairing, the
single miss pinned exactly; the r=0.25 row flips between boxes.

AC3: each committed bench/results/r6_bind_demo_<machine>.json
byte-equals an in-process regeneration (skip-with-reason while a box's
artifact is pending in the draft-PR window)."""
import sys
from pathlib import Path

import pytest

pyreloc = pytest.importorskip("pyreloc")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "bench" / "rtrack"))

import r6_bind_demo                                  # noqa: E402
from cm5_eval import fmt_r, load_rows, merge_points  # noqa: E402

RESULTS = REPO_ROOT / "bench" / "results"
BOXES = ("epyc7351-2080ti", "7800x3d-4070tis")
# The one expected miss (spec "Expected outcomes"): Gen4 r=0.5 N=2048,
# measured a by a 4.4% margin -- small-N WSL2 caveat class, disclosed.
KNOWN_MISSES = {("7800x3d-4070tis", 0.5, 2048)}


def merged_rsweep(box):
    merged, _ = merge_points(
        load_rows(RESULTS / f"bp_rsweep_{box}.csv"),
        load_rows(RESULTS / f"bp_rsweep_rerun_{box}.csv"))
    return merged


def measured_winner(merged, r, n, b_method):
    # rsweep design: A is measured at each r tier; B ships f32 at r=1
    # (its cost is r-independent), so every cell compares a(r) vs
    # b_method(r=1) -- the same pairing cm5_eval's r* rows use.
    a = merged[("blocked_transpose", n, "a", "rsweep", fmt_r(r))]
    b = merged[("blocked_transpose", n, b_method, "rsweep", "1")]
    return "a" if float(a["median_ms"]) < float(b["median_ms"]) else "b"


@pytest.fixture(scope="module", params=BOXES)
def box_report(request):
    return request.param, r6_bind_demo.build_demo(request.param)


def test_overlapped_decisions_match_measured_winners(box_report):
    box, report = box_report
    merged = merged_rsweep(box)
    misses = {(box, c["r"], c["N"]) for c in report["cells"]
              if c["bind_decision"]["method"]
              != measured_winner(merged, c["r"], c["N"], "b_pipelined")}
    assert misses == {m for m in KNOWN_MISSES if m[0] == box}


def test_serial_check_matches_b_fair_winners(box_report):
    box, report = box_report
    merged = merged_rsweep(box)
    misses = {(box, c["r"], c["N"]) for c in report["cells"]
              if c["serial_check"]["method"]
              != measured_winner(merged, c["r"], c["N"], "b_fair")}
    assert misses == {m for m in KNOWN_MISSES if m[0] == box}


def test_flip_row_r025(box_report):
    box, report = box_report
    expect = "b" if box == "epyc7351-2080ti" else "a"
    row = [c for c in report["cells"] if c["r"] == 0.25]
    assert len(row) == 4
    assert all(c["bind_decision"]["method"] == expect for c in row)


def test_all_cells_classified_blocked(box_report):
    _, report = box_report
    assert all(c["bind_decision"]["pattern"] == "blocked"
               for c in report["cells"])
    assert all(c["bind_decision"]["b_placement"] == "overlapped"
               for c in report["cells"])


@pytest.mark.parametrize("box", BOXES)
def test_committed_artifact_byte_equals_regeneration(box):
    path = RESULTS / f"r6_bind_demo_{box}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed yet (draft-PR window; see "
                    "the R6 Gen4 runbook in bench/rtrack/README.md)")
    regen = r6_bind_demo.render(r6_bind_demo.build_demo(box))
    assert path.read_text() == regen
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests/test_r6_crossbox_bind.py -v`
Expected: 9 PASS + 1 SKIP (`test_committed_artifact_byte_equals_regeneration[7800x3d-4070tis]` — artifact pending). If any winner-match test fails, STOP and report — the spec's expected outcomes were pre-verified against the committed CSVs, so a failure means the implementation (not the data) is wrong.

- [ ] **Step 3: Run the full suite once more**

Run: `PYTHONPATH=build/sym/python python3 -m pytest libreloc/python/tests -q`
Expected: all green, 1 skip.

- [ ] **Step 4: Commit**

```bash
git add libreloc/python/tests/test_r6_crossbox_bind.py
git commit -m "test(pyreloc): R6 — cross-box decision lock vs BP winners (#87)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Result doc, README sections, claim-ledger row

**Files:**
- Create: `docs/r6-crossbox-bind.md`
- Modify: `bench/rtrack/README.md` (new `## R6 bind demo` section + `### R6 Gen4 runbook`, placed after the existing `### CM1 Gen4 recv-kernel run` section, line ~428 area, before `## V3 cost-model tools`)
- Modify: `docs/claim-ledger.md` (new `## Machinery` section at end)

**Interfaces:**
- Consumes: Task 2's artifact + `build_demo`; merged BP CSVs via `cm5_eval` (table generation snippet below).
- Produces: the citable result doc; the runbook Task 6's user step follows.

- [ ] **Step 1: Generate the 24-cell table**

Run this snippet; paste its markdown output into the doc in Step 2:

```bash
PYTHONPATH=build/sym/python python3 - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, 'bench/rtrack')
from cm5_eval import fmt_r, load_rows, merge_points
import r6_bind_demo
RESULTS = Path('bench/results')
for box in ("epyc7351-2080ti", "7800x3d-4070tis"):
    report = r6_bind_demo.build_demo(box)
    merged, _ = merge_points(load_rows(RESULTS/f'bp_rsweep_{box}.csv'),
                             load_rows(RESULTS/f'bp_rsweep_rerun_{box}.csv'))
    print(f"\n### {box}\n")
    print("| r | N | a (ms) | b_pipelined (ms) | measured | bind decision "
          "| threshold_bytes | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for c in report['cells']:
        a = merged[('blocked_transpose', c['N'], 'a', 'rsweep', fmt_r(c['r']))]
        b = merged[('blocked_transpose', c['N'], 'b_pipelined', 'rsweep', '1')]
        meas = 'a' if float(a['median_ms']) < float(b['median_ms']) else 'b'
        d = c['bind_decision']
        verdict = 'match' if d['method'] == meas else '**MISS** (disclosed)'
        print(f"| {c['r']} | {c['N']} | {float(a['median_ms']):.3f} "
              f"| {float(b['median_ms']):.3f} | {meas} | {d['method']} "
              f"| {d['threshold_bytes']:.6g} | {verdict} |")
EOF
```

- [ ] **Step 2: Write `docs/r6-crossbox-bind.md`**

Use exactly this structure (fill `<TABLES>` with Step 1's output; adjust nothing else without noting why in the commit message):

```markdown
# R6 — cross-box no-recompile bind demo (issue #87)

**Claim demonstrated**: the same MLIR-folded symbolic relocation plan
(`libreloc/test/corpus/blocked_transpose_sym.bin`, the V3 wire-row plan:
chain → `sym-opt --reloc-fold` → `encodePlan`), symbol-bound at runtime
(`pyreloc.bind(plan, {"N": N}, model=<box calibration>, wire_ratio=r)`),
gets its placement chosen automatically by the calibrated cost model at
bind time — and the choice is correct on both boxes, with zero per-box
recompilation or refitting. Cross-box generalization of V3's wire row
(`docs/v3-costmodel.md` §6: one box, one decision).

Narrowing provenance: issue #107 "Reconciliation with #87" — the
held-out prediction-accuracy half of #87 went to CM4 (#112/#121) and
CM5 (#113/#126); this demo is the remaining half.

## Setup

- Grid: r ∈ {0.25, 0.5, 1.0} (the measured rsweep wire tiers s8/f16/f32)
  × N ∈ {2048, 4096, 8192, 16384} × both boxes = 24 cells. r = 0.125 is
  excluded: neither calibration carries a blocked s4 CPU key, so
  `predict` raises and `bind` leaves the decision unset (opt-in advice,
  never a bind failure) — a CM-track model-coverage boundary, not a
  machinery limit.
- Decision: bind-time, t8 + Overlapped placement by construction
  (`libreloc/src/Bind.cpp` step 8 — the deployment default; the
  library's real Method B is the double-buffered pipeline). A
  serial-priced (`b_fair`) `predict` check rides along per cell.
- Ground truth: stabler-preference-merged BP rsweep medians
  (`bench/results/bp_rsweep{,_rerun}_{box}.csv`; merge rule =
  `cm5_eval.merge_points`, the CM5 rule). Cell pairing follows the
  rsweep design: `a` at tier r vs `b_pipelined` (or `b_fair`) at r=1 —
  B ships f32 regardless of A's wire ratio.
- Artifacts: `bench/results/r6_bind_demo_{epyc7351-2080ti,
  7800x3d-4070tis}.json` — fully deterministic (sorted keys, no
  timestamps/hostnames/revs, inputs sha256-pinned). Each was produced
  on its own box; CI regenerates both in-process and byte-compares
  (`libreloc/python/tests/test_r6_crossbox_bind.py`). Byte-equality is
  the no-recompile portability bar: the artifact's only inputs are the
  committed calibration and blob, so a box cannot have "adapted"
  anything and still match.

## Verdict

- **23/24 bind-time decisions match the measured winner** under the
  overlapped pairing; the serial pairing gives the same 23/24.
- **The flip row, r = 0.25 — all 8 cells correct**: Gen3 chooses `b`
  at every N (its host gather is slow relative to its gen3 link;
  shipping raw f32 wins ~4×), Gen4 chooses `a` at every N (host
  gather+quantize wins 1.32–1.73×). Same plan bytes, same shapes,
  opposite placements, both correct — the boundary law
  (`docs/claim-ledger.md`) deciding placement per host at bind time.
- r = 1.0 row: both boxes choose `b` at every N, matching measurement —
  the machinery is not hardwired to flip (and this is the r=1.0
  pure-relocation loss both boxes' ledger rows record).
- **The 1 miss, disclosed**: (Gen4, r=0.5, N=2048) — measured `a` by
  4.4% (0.669 vs 0.700 ms), model says `b`. Small-N WSL2 cell — the
  caveat class BP's admissibility gate excludes below N=8192; CM5's
  MISCLASS handling is the authority on model quality. Not excluded
  from this table; counted, named, and left standing.

<TABLES>

## Why there is no #73 gate registration for this demo

Nothing here is a new stochastic measurement. The decision tables are
deterministic functions of committed calibrations + committed code; the
measured winners were committed by BP3 (#116/#124) and merged by the
CM5 rule (#113/#126). The acceptance bars (23/24 with the pinned miss;
artifact byte-equality) are enforced continuously in CI, which is
stronger than a one-shot pre-registered run. The expected outcomes were
still written into the spec before the code existed
(`docs/superpowers/specs/2026-08-12-r6-crossbox-bind-demo-design.md`).

## Reproduction

    PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
        --machine <epyc7351-2080ti|7800x3d-4070tis>
    PYTHONPATH=build/sym/python python3 -m pytest \
        libreloc/python/tests/test_r6_crossbox_bind.py -v

Gen4 session: see "R6 Gen4 runbook" in `bench/rtrack/README.md`.

## Cross-links

#87 (this demo) · #107 reconciliation (scope split) · CM4/CM5 (held-out
model-quality gates, `bench/results/cm5_eval_report.json`) · V3 §6 wire
row (single-box precedent) · BP3 (`bp_rsweep*` ground truth) ·
`docs/claim-ledger.md` "Machinery" row.
```

- [ ] **Step 3: Add the README sections**

In `bench/rtrack/README.md`, after the `### CM1 Gen4 recv-kernel run` section ends (before `## V3 cost-model tools`), insert:

```markdown
## R6 bind demo (issue #87)

`r6_bind_demo.py` — the cross-box no-recompile bind demo: binds the
committed MLIR-folded corpus blob (`blocked_transpose_sym.bin`) at
runtime N with this box's calibration as the bind model and writes the
deterministic decision artifact `bench/results/r6_bind_demo_<machine>.json`.
No GPU, no measurement — decisions only; verified against the BP
measured winners by `libreloc/python/tests/test_r6_crossbox_bind.py`.
See `docs/r6-crossbox-bind.md`.

    PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
        --machine epyc7351-2080ti

### R6 Gen4 runbook (issue #87; run at the home box, then undraft PR)

No session ritual needed — this writes no measurement; any load state
is fine.

    git pull
    ninja -C build/sym pyreloc_ext
    PYTHONPATH=build/sym/python python3 bench/rtrack/r6_bind_demo.py \
        --machine 7800x3d-4070tis
    git add bench/results/r6_bind_demo_7800x3d-4070tis.json
    git commit -m "bench(results): R6 — Gen4 bind-demo artifact (#87)"
    git push

CI's byte-equality check (previously skipping with "not committed yet")
goes live on push; when it is green, undraft the PR.
```

- [ ] **Step 4: Add the claim-ledger Machinery section**

Append to `docs/claim-ledger.md`:

```markdown
## Machinery

| claim | boxes | result | status | authoritative source |
|---|---|---|---|---|
| Same folded plan, runtime symbol-bind, bind-time auto-placement — correct choice on both boxes, no recompilation | Gen3 + Gen4 | 23/24 decisions match measured winners (1 small-N miss disclosed); r=0.25 row flips correctly (Gen3 `b` ×4, Gen4 `a` ×4); artifacts byte-equal CI regeneration | survives | `docs/r6-crossbox-bind.md`; `bench/results/r6_bind_demo_*.json` (#87) |
```

- [ ] **Step 5: Commit**

```bash
git add docs/r6-crossbox-bind.md bench/rtrack/README.md docs/claim-ledger.md
git commit -m "docs(bench): R6 — result doc, Gen4 runbook, ledger machinery row (#87)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Draft PR

**Files:** none (git/gh operations only)

**Interfaces:**
- Consumes: all prior commits on `r6-crossbox-bind-demo`.
- Produces: the draft PR whose number Task 6's issue edit references.

- [ ] **Step 1: Push and open the draft PR**

```bash
git push -u origin r6-crossbox-bind-demo
gh pr create --draft --title "feat(pyreloc, bench): R6 — cross-box no-recompile bind demo (#87)" --body "$(cat <<'EOF'
Resolves the narrowed #87 (per #107's reconciliation; held-out half went to CM4/CM5): same MLIR-folded sym plan (`blocked_transpose_sym.bin`), runtime symbol-bind, bind-time auto-placement from each box's committed calibration — correct choice on both boxes, no recompilation.

- pyreloc: `bind(..., model=, wire_ratio=, k=, n_reuse=)` (kw-only) + `BoundPlan.decision` — plumbs the existing C++ bind hook (Bind.cpp step 8) through pybind; existing callers untouched.
- `bench/rtrack/r6_bind_demo.py` + deterministic artifacts `bench/results/r6_bind_demo_<machine>.json` (inputs sha-pinned; byte-equality vs CI regeneration = the no-recompile bar).
- Lock test `test_r6_crossbox_bind.py`: 23/24 decisions match stabler-merged BP winners per pairing (the 1 miss pinned: Gen4 r=0.5 N=2048, 4.4% margin, small-N WSL2 caveat class); **r=0.25 flip row all-correct (Gen3 b ×4 / Gen4 a ×4)**; winner merge imported from `cm5_eval` (CM3 discipline).
- `docs/r6-crossbox-bind.md` (24-cell table + why no #73 registration), README runbook, claim-ledger Machinery row.

**Draft until** the Gen4 artifact lands — runbook: `bench/rtrack/README.md` §"R6 Gen4 runbook" (one command on the home box; CI byte-check goes live on push).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Verify CI goes green on the draft (1 expected skip)**

Run: `gh pr checks --watch` (or re-check after a few minutes).
Expected: build + pytest green; `test_committed_artifact_byte_equals_regeneration[7800x3d-4070tis]` skips.

---

### Task 6: Gen4 leg, undraft, merge, close #87 (user-gated)

**Files:**
- Created on the Gen4 box by the user (runbook): `bench/results/r6_bind_demo_7800x3d-4070tis.json`

**Interfaces:**
- Consumes: Task 4's runbook; Task 5's PR.
- Produces: merged PR; #87 edited + closed.

- [ ] **Step 1 (USER, on the home box): run the R6 Gen4 runbook** (bench/rtrack/README.md) — one script invocation, commit, push.

- [ ] **Step 2: Confirm CI byte-check went live and green**

Run: `gh pr checks`
Expected: pytest green with **0 skips** in `test_r6_crossbox_bind.py`.

- [ ] **Step 3: Undraft and merge (squash, repo convention)**

```bash
gh pr ready
gh pr merge --squash
```

- [ ] **Step 4: Edit #87's body (per #107's reconciliation) and close**

```bash
gh issue edit 87 --repo JueonPark/sym --body "$(cat <<'EOF'
(Narrowed 2026-08-12 per #107's "Reconciliation with #87": the held-out prediction-accuracy half — train N∈{2048,8192} → test N∈{4096,16384} — was absorbed into CM4's registration (#112/#121) and CM5's held-out verdicts (#113/#126). What remained, delivered by the R6 PR (`docs/r6-crossbox-bind.md`):)

- End-to-end demo: same `sym` relocation plan, symbol-bound at runtime, placement chosen automatically by cost model → correct choice on both Gen3 and Gen4 boxes without recompilation. This is the machinery claim regardless of (a)/(b) framing.
- **W7 end = eval freeze (2026-09-02).** All CSVs frozen, figures regenerate from frozen data only.
EOF
)"
gh issue close 87 --repo JueonPark/sym --comment "Delivered (narrowed scope per #107): docs/r6-crossbox-bind.md — 23/24 bind-time decisions match the BP measured winners on both boxes from the same folded plan with runtime symbol-bind (1 disclosed small-N miss); the r=0.25 row flips correctly between boxes (Gen3 b / Gen4 a); artifacts byte-equal CI regeneration (no-recompile bar). Held-out model-quality gates live in CM5 (bench/results/cm5_eval_report.json). Eval-freeze date (2026-09-02) remains in force per Build Doc v3 §4."
```

---

## Self-review notes

- **Spec coverage**: AC1 → Task 1; AC2 → Tasks 2–3; AC3 → Tasks 2–3 (determinism + byte twin) and 6 (Gen4 artifact); AC4 → Tasks 4 and 6. Spec components 1–6 map to Tasks 1, 2, 3, 4 (doc+ledger), 5–6 (flow + issue hygiene). Spec amendment (bind-vs-predict behavior at r=0.125) → Task 1 Step 6.
- **Type consistency**: decision dict keys (`method/t_a_ms/t_b_ms/threshold_bytes/pattern/b_placement/k/n_reuse`) identical across Task 1 (producer), Task 2 (artifact), Task 3 (assertions), Task 4 (doc table uses `method`, `threshold_bytes`). `build_demo/render` names match between Tasks 2 and 3.
- **Known-fragile points**: pybind `py::kw_only()` position (after `strategy`); `bind` def must live after `Calibration` registration; `MethodDecision` include may be transitively present — compiler is the arbiter, fallback include named in Task 1 Step 3.
