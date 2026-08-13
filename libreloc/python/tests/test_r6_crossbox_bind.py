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
