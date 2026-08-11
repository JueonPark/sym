"""CM5 evaluation-tool tests (issue #113). Synthetic fixtures for the pure
functions; Task 5 adds an end-to-end regen twin over committed inputs."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "bench" / "rtrack"))

import cm5_eval  # noqa: E402


def _row(transform="quant", N="2048", method="a", variant="matrix", r="0.25",
         chunk="4", median="1.0", iqr="1.0"):
    return {"transform": transform, "N": N, "method": method, "variant": variant,
            "r": r, "chunk_req_mib": chunk, "median_ms": median,
            "iqr_over_median_pct": iqr}


def test_merge_prefers_lower_iqr_rerun():
    orig = [_row(chunk="4", median="1.00", iqr="8.0")]
    rerun = [_row(chunk="4", median="1.10", iqr="2.0")]
    merged, audit = cm5_eval.merge_points(orig, rerun)
    key = cm5_eval.point_key(orig[0])
    assert merged[key]["median_ms"] == "1.10"
    assert audit == [{"transform": "quant", "N": 2048, "method": "a",
                      "variant": "matrix", "r": "0.25", "chose": "rerun",
                      "orig_chunk": "4", "orig_median_ms": 1.0, "orig_iqr": 8.0,
                      "rerun_chunk": "4", "rerun_median_ms": 1.1,
                      "rerun_iqr": 2.0}]


def test_merge_tie_keeps_original():
    orig = [_row(median="1.00", iqr="3.0")]
    rerun = [_row(median="1.10", iqr="3.0")]
    merged, audit = cm5_eval.merge_points(orig, rerun)
    assert merged[cm5_eval.point_key(orig[0])]["median_ms"] == "1.00"
    assert audit[0]["chose"] == "original"


def test_merge_missing_rerun_uses_original_no_audit():
    orig = [_row()]
    merged, audit = cm5_eval.merge_points(orig, [])
    assert merged[cm5_eval.point_key(orig[0])] is orig[0]
    assert audit == []


def test_merge_best_chunk_per_file_before_comparison():
    # original: best chunk is 16MiB (1.5ms) with noisy IQR 9; rerun best is
    # 1.6ms IQR 2 -> rerun wins because per-file best-chunk rows compare 9 vs 2.
    orig = [_row(chunk="4", median="2.0", iqr="1.0"),
            _row(chunk="16", median="1.5", iqr="9.0")]
    rerun = [_row(chunk="4", median="1.6", iqr="2.0")]
    merged, audit = cm5_eval.merge_points(orig, rerun)
    assert merged[cm5_eval.point_key(orig[0])]["median_ms"] == "1.6"
    assert audit[0]["orig_chunk"] == "16"


def test_merge_rerun_only_point_is_loud_error():
    with pytest.raises(SystemExit):
        cm5_eval.merge_points([], [_row()])
