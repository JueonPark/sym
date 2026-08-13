"""R6 (issue #87): pyreloc bind-with-model surface (spec AC1).

The C++ bind() has accepted an optional CostModel* since V3/CM1
(libreloc/src/Bind.cpp step 8: threads=8, BPlacement::Overlapped --
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
    assert d["method"] == "b"           # Gen3 blocked r=1.0 -- the V3 wire row
    assert d["pattern"] == "blocked"    # classified from the bound plan
    assert d["b_placement"] == "overlapped"   # Bind.cpp:346
    assert d["t_a_ms"] > 0 and d["t_b_ms"] > 0
    assert d["k"] == 1 and d["n_reuse"] == -1
    assert "threshold_bytes" in d


def test_wire_ratio_moves_the_decision(plan):
    # Gen4 blocked: r* (overlapped) = 0.3605 measured / 0.2914 predicted
    # (claim ledger) -- r=0.25 sits on the A side, r=1.0 on the B side.
    cal = pyreloc.load_calibration(str(GEN4_CAL))
    assert pyreloc.bind(plan, {"N": 8192}, model=cal,
                        wire_ratio=0.25).decision["method"] == "a"
    assert pyreloc.bind(plan, {"N": 8192}, model=cal,
                        wire_ratio=1.0).decision["method"] == "b"


def test_missing_tier_key_leaves_decision_none(plan):
    # No blocked s4 (r=0.125) key exists in either calibration: predict
    # raises, but bind succeeds with decision unset -- Bind.cpp step 8 is
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
