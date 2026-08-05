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
