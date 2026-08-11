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


def _reg_cell(box="boxA", family="quant", N=2048, split="train",
              wf="a", wp="a"):
    return {"box": box, "family": family, "N": N, "split": split,
            "winner_vs_b_fair": wf, "winner_vs_b_pipelined": wp}


def _merged(entries):
    """entries: list of (transform, N, method, variant, r, median_ms)."""
    return {(t, n, m, v, r): _row(transform=t, N=str(n), method=m, variant=v,
                                  r=r, median=str(med))
            for t, n, m, v, r, med in entries}


def test_winner_cells_and_exclusion():
    reg = {"cells": [_reg_cell(wf="a", wp="b")]}
    merged = {"boxA": _merged([
        ("quant", 2048, "a", "matrix", "0.25", 1.0),
        ("quant", 2048, "b_fair", "matrix", "0.25", 2.0),
        # no b_pipelined row -> that pairing excluded
    ])}
    reg["cells"][0]["r_native"] = 0.25
    cells, excluded = cm5_eval.eval_winner_cells(merged, reg)
    assert [c["winner_meas"] for c in cells["b_fair"]] == ["a"]
    assert cells["b_fair"][0]["winner_pred"] == "a"
    assert cells["b_pipelined"] == []
    assert excluded == [{"box": "boxA", "family": "quant", "N": 2048,
                         "pairing": "b_pipelined",
                         "reason": "no measured b_pipelined matrix row"}]


def test_misclass_and_regret():
    cells = [
        {"winner_pred": "a", "winner_meas": "a", "t_a_meas_ms": 1.0,
         "t_b_meas_ms": 2.0},
        {"winner_pred": "b", "winner_meas": "a", "t_a_meas_ms": 1.0,
         "t_b_meas_ms": 1.5},  # wrong: model pays 1.5 vs oracle 1.0
    ]
    mc = cm5_eval.misclass(cells, 0.15)
    assert (mc["n_cells"], mc["n_wrong"], mc["verdict"]) == (2, 1, "FAIL")
    assert cm5_eval.cell_regret(cells[1], "model") == pytest.approx(0.5)
    assert cm5_eval.cell_regret(cells[1], "always_a") == pytest.approx(0.0)
    rg = cm5_eval.regret_gate(cells, 0.20)
    assert rg["p90"] == pytest.approx(0.5) and rg["verdict"] == "FAIL"


def test_empty_universe_verdict_none():
    assert cm5_eval.misclass([], 0.15)["verdict"] is None
    assert cm5_eval.regret_gate([], 0.20)["verdict"] is None


def _reg_rstar(box="boxA", family="quant", placement="serial", n=16384,
               grid=(0.25, 0.5, 1.0), pred=0.5):
    return {"box": box, "family": family, "placement": placement, "n": n,
            "grid_used": list(grid), "rstar_predicted": pred}


def _rsweep_merged(family, n, b_method, a_medians, b_median):
    """a_medians: {r_str: median}. B measured once at r=1."""
    entries = [(family, n, "a", "rsweep", r, m) for r, m in a_medians.items()]
    entries.append((family, n, b_method, "rsweep", "1", b_median))
    return _merged(entries)


def test_rstar_grid_restriction_and_both_exist():
    # speedup(r) = B(1)/A(r). With grid_used=(0.25, 0.5, 1.0): r=0.25 -> 2.0/1.0=2.0
    # (above 1), r=0.5 -> 2.0/1.0=2.0 (above 1), r=1 -> 2.0/4.0=0.5 (below 1)
    # -> crossing between 0.5 and 1.0. The r=0.125 row (speedup 2.0/25.0=0.08,
    # also below 1.0) must be ignored; if grid_used were leaked/ignored, the
    # crossing would instead be found between r=0.125 and r=0.25, giving r* < 0.25
    # and failing the assertion. This tests that grid restriction is enforced.
    merged = {"boxA": _rsweep_merged(
        "quant", 16384, "b_fair",
        {"1": 4.0, "0.5": 1.0, "0.25": 1.0, "0.125": 25.0}, 2.0)}
    reg = {"rstar": [_reg_rstar(grid=(0.25, 0.5, 1.0), pred=0.9)]}
    rows, excluded = cm5_eval.eval_rstar(merged, reg)
    assert excluded == []
    (row,) = rows
    assert row["classification"] == "both_exist"
    assert 0.5 < row["rstar_meas"] < 1.0
    assert row["abs_delta"] == pytest.approx(abs(0.9 - row["rstar_meas"]))


def test_rstar_one_sided_mismatch_fails_gate():
    # predicted crossing, measured curve never crosses (speedup always < 1)
    merged = {"boxA": _rsweep_merged(
        "quant", 16384, "b_fair", {"1": 4.0, "0.5": 4.0, "0.25": 4.0}, 2.0)}
    reg = {"rstar": [_reg_rstar(pred=0.5)]}
    rows, _ = cm5_eval.eval_rstar(merged, reg)
    assert rows[0]["classification"] == "mismatch_one_sided"
    gate = cm5_eval.rstar_gate(rows, "serial", 0.15)
    assert gate["verdict"] == "FAIL" and gate["n_one_sided_mismatch"] == 1


def test_rstar_non_rsweep_family_excluded():
    reg = {"rstar": [_reg_rstar(family="transpose", pred=None)]}
    rows, excluded = cm5_eval.eval_rstar({"boxA": {}}, reg)
    assert rows == []
    assert excluded[0]["reason"].startswith("family not in the BP r-sweep")


def test_rstar_all_no_crossing_passes():
    merged = {"boxA": _rsweep_merged(
        "quant", 16384, "b_fair", {"1": 4.0, "0.5": 4.0, "0.25": 4.0}, 2.0)}
    reg = {"rstar": [_reg_rstar(pred=None)]}
    rows, _ = cm5_eval.eval_rstar(merged, reg)
    assert rows[0]["classification"] == "both_no_crossing"
    assert cm5_eval.rstar_gate(rows, "serial", 0.15)["verdict"] == "PASS"


def test_crossing_is_the_shared_implementation():
    import figure_rstar
    assert cm5_eval.crossing is figure_rstar.crossing


def test_rstar_gate_filters_by_placement_and_empty_is_none():
    # rstar_gate must filter rows by placement and return None verdict when
    # no rows match the placement (not just when the input is empty).
    rows = [
        {"box": "boxA", "family": "quant", "placement": "serial",
         "classification": "both_exist", "abs_delta": 0.05},
        {"box": "boxA", "family": "quant", "placement": "overlapped",
         "classification": "mismatch_one_sided", "abs_delta": None},
    ]
    gate_serial = cm5_eval.rstar_gate(rows, "serial", 0.15)
    assert gate_serial["n_rows"] == 1
    assert gate_serial["n_one_sided_mismatch"] == 0
    assert gate_serial["verdict"] == "PASS"
    gate_overlapped = cm5_eval.rstar_gate(rows, "overlapped", 0.15)
    assert gate_overlapped["verdict"] == "FAIL"
    gate_empty = cm5_eval.rstar_gate(rows, "nonexistent", 0.15)
    assert gate_empty["n_rows"] == 0
    assert gate_empty["verdict"] is None


def test_ablation_policies():
    cells = [{"winner_pred": "b", "winner_meas": "a", "t_a_meas_ms": 1.0,
              "t_b_meas_ms": 2.0}]
    ab = cm5_eval.ablation(cells)
    assert ab["model"]["mean"] == pytest.approx(1.0)     # picked b, paid 2x
    assert ab["always_a"]["mean"] == pytest.approx(0.0)
    assert ab["always_b"]["mean"] == pytest.approx(1.0)
    assert ab["oracle"] == {"mean": 0.0, "p90": 0.0}


def test_report_shape_and_determinism():
    report = cm5_eval.build_report()
    assert sorted(report) == ["ablation", "excluded_cells", "gates",
                              "generated_by", "issue", "merge_audit",
                              "misses", "provenance", "rstar_rows"]
    # all-cells universes match the known data facts
    ac = report["gates"]["all_cells"]
    assert ac["b_fair"]["misclass"]["n_cells"] == 48
    assert ac["b_pipelined"]["misclass"]["n_cells"] == 40
    ho = report["gates"]["held_out"]
    assert ho["b_fair"]["misclass"]["n_cells"] == 24
    assert ho["b_pipelined"]["misclass"]["n_cells"] == 20
    assert "caveat" in ho and "two-point-fit" in ho["caveat"]
    rs = report["gates"]["rstar"]
    assert rs["serial"]["n_rows"] == 8 and rs["overlapped"]["n_rows"] == 8
    assert len(report["rstar_rows"]) == 16
    assert len(report["provenance"]) == 9
    # byte determinism
    assert cm5_eval.render(report) == cm5_eval.render(cm5_eval.build_report())


def test_committed_report_regenerates_byte_identical():
    """CI-side twin of v3_gate.py's CM5-REPORT-REGEN (CI runs pytest, not
    v3_gate.py): the committed report must reproduce from committed inputs."""
    committed = REPO_ROOT / "bench" / "results" / "cm5_eval_report.json"
    assert committed.exists(), "cm5_eval_report.json not committed"
    assert cm5_eval.render(cm5_eval.build_report()) == committed.read_text()
