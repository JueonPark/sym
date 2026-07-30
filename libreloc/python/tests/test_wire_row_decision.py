"""V3 (issue #97), spec §6 acceptance: the model's decision for the
compiler-emitted wire row matches the measured winner. Self-contained --
loads only the epyc calibration and reads only
bench/results/v3_wire_row_epyc_2080ti.csv; does NOT import or otherwise
touch test_prediction.py's report writer (that report's committed JSON
must stay byte-identical, see v3_gate.py's REPORT-REGEN check)."""
import csv
import pathlib

import pytest

pyreloc = pytest.importorskip("pyreloc")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CAL_PATH = REPO_ROOT / "calibration" / "epyc7351-2080ti.cal"
WIRE_ROW_PATH = REPO_ROOT / "bench" / "results" / "v3_wire_row_epyc_2080ti.csv"


def _best_chunk_median(rows, method):
    """Min median_ms across chunk sizes for this method -- the same
    'best-chunk' selection test_prediction.py's matrix_cells uses."""
    candidates = [float(r["median_ms"]) for r in rows if r["method"] == method]
    assert candidates, f"no '{method}' rows found in {WIRE_ROW_PATH.name}"
    return min(candidates)


def test_wire_row_decision_matches_measured_winner():
    cal = pyreloc.load_calibration(str(CAL_PATH))

    # Spec §6 acceptance, verbatim params: N=8192 blocked-transpose wire
    # row, r=1.0 (no dtype reduction -- both methods ship full fp32).
    decision = pyreloc.predict(cal, pattern="blocked",
                               src_bytes=8192 * 8192 * 4, r=1.0)
    assert decision["method"] == "b"

    with open(WIRE_ROW_PATH) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    a_median = _best_chunk_median(rows, "a")
    b_fair_median = _best_chunk_median(rows, "b_fair")

    # The model predicted "b"; the measured best-chunk winner must also be
    # b_fair (lower median than a) for the decision to match the
    # measurement, per docs/v3-costmodel.md §6.
    assert b_fair_median < a_median, (
        f"model predicted method=b but measured b_fair median "
        f"({b_fair_median}) is not below measured a median ({a_median})")
