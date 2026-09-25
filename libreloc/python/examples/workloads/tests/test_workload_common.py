"""Unit tests for the workload examples' shared helpers (common.py)."""
import json
import sys

import pytest
import torch

import common


def test_a_check_passes_only_when_every_observation_passed():
    report = common.Report("demo")
    assert report.check("a", True) and report.check("a", True) and report.check("b", 1 == 1)
    assert report.ok
    assert report.check("b", False, lambda: "second observation differs") is False
    assert not report.ok
    assert report.data["checks"]["b"] == {"passed": 1, "failed": 1, "first_failure": "second observation differs"}


def test_a_detail_is_evaluated_only_when_the_check_fails():
    calls = []
    report = common.Report("demo")
    report.check("a", True, lambda: calls.append("evaluated"))
    assert calls == []
    report.check("a", False, lambda: calls.append("evaluated") or "why")
    assert calls == ["evaluated"] and report.data["checks"]["a"]["first_failure"] == "why"


def test_a_report_without_checks_is_not_ok():
    assert not common.Report("demo").ok


def test_finish_writes_the_report_and_returns_zero_when_every_check_passed(tmp_path, capsys):
    report = common.Report("demo", devices=["cuda:0"], rows=3)
    report.check("a", True)
    report.add_bytes("x", source=400, wire=200, destination=200)
    report.add_bytes("x", source=400, wire=200, destination=200)
    report.add_dispatch("cpu_reference", 2)
    report.add_plan_compiles(1)
    for path, samples in (("torch", [5.0, 1.0, 3.0]), ("sym", [50.0, 10.0, 30.0])):
        for milliseconds in samples:
            report.time(path, "x", milliseconds)
    output = tmp_path / "report.json"
    assert report.finish(output) == 0
    data = json.loads(output.read_text())
    assert data["ok"] is True and data["devices"] == ["cuda:0"] and data["sizes"] == {"rows": 3}
    assert data["bytes"]["x"] == {"transfers": 2, "source": 800, "wire": 400, "destination": 400}
    assert data["dispatches"] == {"cpu_reference": 2}
    assert data["timings_ms"]["x"]["sym"] == {"count": 3, "first_call": 50.0, "median": 20.0, "total": 90.0}
    assert data["summary"] == {
        "checks_passed": 1, "checks_total": 1, "plan_compiles": 1, "sym_transfers": 2, "wire_bytes": 400,
        "destination_bytes": 400, "fallbacks": 0, "steady_transfer_ms": {"torch": 4.0, "sym": 40.0}}
    assert "demo: PASS (1/1 checks)" in capsys.readouterr().out


def test_finish_returns_one_and_names_the_failed_check(capsys):
    report = common.Report("demo")
    report.check("values_equal", False, "batch 3 differs")
    assert report.finish() == 1
    out = capsys.readouterr().out
    assert "demo: FAIL (0/1 checks)" in out
    assert "FAILED values_equal: 1 of 1 -- batch 3 differs" in out


def test_a_report_with_wire_bytes_states_that_equivalent_pytorch_moves_the_same_bytes():
    report = common.Report("demo")
    report.check("a", True)
    report.finish()
    assert common.WIRE_CAVEAT not in report.data["notes"]           # nothing moved, nothing to qualify
    report.add_bytes("x", source=400, wire=100, destination=400)
    report.finish()
    report.finish()
    assert report.data["notes"].count(common.WIRE_CAVEAT) == 1

def test_backend_stats_feed_counters_dispatches_and_plan_compiles():
    report = common.Report("demo")
    report.set_backend_stats({
        "dynamo_compiles": 1, "plan_compiles": 2, "symbol_binds": 4, "runtime_executions": 4,
        "typed_executions": 2, "typed_payload_bytes": 64, "fallbacks": {},
        "exclusions": {"conditional_materialization": 1}, "dispatches": {"cpu_reference": 2}, "cache_hits": 7})
    assert report.data["counters"] == {
        "dynamo_compiles": 1, "plan_compiles": 2, "symbol_binds": 4, "runtime_executions": 4,
        "typed_executions": 2, "typed_payload_bytes": 64, "fallbacks": {},
        "exclusions": {"conditional_materialization": 1}}
    assert report.data["dispatches"] == {"cpu_reference": 2}
    report.check("a", True)
    report.finish()
    assert report.data["summary"]["plan_compiles"] == 2 and report.data["summary"]["fallbacks"] == 1


def test_transfer_wrappers_time_compare_and_count_bytes():
    report = common.Report("demo")
    clock = common.Clock([])                               # no devices to synchronize: runs on the CPU
    reference = common.reference_transfer(report, clock, "x", lambda t: t * 2)
    sym = common.sym_transfer(report, clock, "x", lambda t: t * 2, lambda t: t * 2)
    wrong = common.sym_transfer(report, clock, "y", lambda t: t * 2, lambda t: t * 3, count_bytes=False)
    source = torch.arange(4.0)
    assert torch.equal(reference(source), source * 2) and torch.equal(sym(source), source * 2)
    wrong(source)
    assert report.data["checks"]["x_equal"] == {"passed": 1, "failed": 0}
    assert report.data["checks"]["y_equal"]["failed"] == 1
    assert "max abs diff 3" in report.data["checks"]["y_equal"]["first_failure"]
    assert report.data["bytes"] == {"x": {"transfers": 1, "source": 16, "wire": 16, "destination": 16}}
    report.finish()
    assert set(report.data["timings_ms"]["x"]) == {"torch", "sym"}


def test_same_tensor_compares_values_and_metadata():
    a = torch.arange(6.0).reshape(2, 3)
    assert common.same_tensor(a, a.clone())
    assert not common.same_tensor(a, a.t().contiguous().t())      # same values, other strides
    assert not common.same_tensor(a, a.to(torch.float16))
    changed = a.clone()
    changed[1, 2] = 7.0
    assert not common.same_tensor(changed, a)
    assert "max abs diff 2" in common.difference(changed, a)
    assert "metadata differs" in common.difference(a, a.to(torch.float16))
    assert common.same_tensor(torch.zeros(1, 4), torch.zeros(4, 1).t())   # extent-1 strides are ignored


def test_same_tensor_compares_bit_patterns_not_just_values():
    for dtype in (torch.float32, torch.float16):
        assert not common.same_tensor(torch.tensor([0.0], dtype=dtype), torch.tensor([-0.0], dtype=dtype))
    nan = torch.tensor([float("nan"), 1.0])
    assert common.same_tensor(nan, nan.clone())                    # identical bits, even though NaN != NaN
    assert "bit patterns differ" in common.difference(torch.tensor([0.0]), torch.tensor([-0.0]))

def test_calibration_is_chosen_by_gpu_name():
    assert common.calibration_path("NVIDIA GeForce RTX 2080 Ti").name == "epyc7351-2080ti.cal"
    assert common.calibration_path("NVIDIA GeForce RTX 4070 Ti SUPER").name == "7800x3d-4070tis.cal"
    assert common.calibration_path("NVIDIA A100-SXM4-80GB") is None


def test_resolve_calibration_modes(tmp_path):
    assert common.resolve_calibration("none", "NVIDIA GeForce RTX 2080 Ti") == (None, "none")
    model, label = common.resolve_calibration("auto", "NVIDIA GeForce RTX 2080 Ti")
    assert model.machine == "epyc7351-2080ti" and label.endswith("calibration/epyc7351-2080ti.cal")
    model, label = common.resolve_calibration(str(common.REPO / "calibration" / "7800x3d-4070tis.cal"), "any")
    assert model.machine == "7800x3d-4070tis"
    model, label = common.resolve_calibration("auto", "Some Other GPU")
    assert model is None and "no committed calibration matches" in label
    with pytest.raises(common.PrerequisiteError, match="cannot load calibration"):
        common.resolve_calibration(str(tmp_path / "missing.cal"), "NVIDIA GeForce RTX 2080 Ti")


def test_quantize_per_channel_is_symmetric_int8_per_output_channel():
    weight = torch.randn(64, 48, generator=torch.Generator().manual_seed(0))
    weight[:, 5] = 0.0                                     # an all-zero channel still gets a positive scale
    q, scale = common.quantize_per_channel(weight)
    assert q.dtype == torch.int8 and q.shape == (64, 48) and q.is_contiguous()
    assert scale.dtype == torch.float32 and scale.shape == (48,) and bool((scale > 0).all())
    assert int(q.abs().max()) <= 127 and bool((q[:, 5] == 0).all())
    assert torch.allclose(scale[:5], weight[:, :5].abs().amax(dim=0) / 127)
    assert bool(((q.float() * scale - weight).abs() <= scale / 2 + 1e-6).all())


def test_check_environment_reports_every_problem(monkeypatch):
    monkeypatch.setenv("SYM_RELOC_EXPORT", "/nonexistent/sym-reloc-export")
    problems = common.check_environment(["cuda:99", "not-a-device"])
    assert any("exporter is missing" in p for p in problems)
    assert any("cuda:99" in p or "no CUDA device" in p for p in problems)
    assert any("not-a-device" in p or "no CUDA device" in p for p in problems)


def test_check_environment_reports_a_missing_torch_instead_of_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)       # `import torch` now raises ImportError
    problems = common.check_environment(["cuda:0"])
    assert len(problems) == 1 and "torch is not importable" in problems[0]

def test_run_example_turns_a_prerequisite_error_into_exit_two(capsys):
    def main():
        raise common.PrerequisiteError("no GPU here")

    assert common.run_example(main) == 2
    assert "prerequisite: no GPU here" in capsys.readouterr().err
    assert common.run_example(lambda: 0) == 0


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
needs_two_gpus = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                                    reason="needs two GPUs")


@pytest.mark.gpu
@needs_cuda
def test_weight_fetcher_matches_pytorch_for_every_shape_with_one_artifact():
    report = common.Report("fetch")
    fetcher = common.WeightFetcher(report)
    generator = torch.Generator().manual_seed(1)
    for rows, cols in ((64, 96), (96, 64), (128, 128)):
        q, scale = common.quantize_per_channel(torch.randn(rows, cols, generator=generator))
        actual = fetcher.fetch(q, scale, "cuda:0")
        assert actual.shape == (cols, rows) and actual.dtype == torch.float32
        assert common.same_tensor(actual, common.WeightFetcher.reference(q, scale, "cuda:0"))
    assert fetcher.compiles == 1 and fetcher.shapes == {(64, 96), (96, 64), (128, 128)}
    assert report.data["plan_compiles"] == 1
    weights = report.data["bytes"]["weights"]
    assert weights["transfers"] == 3 and weights["payload"] >= weights["wire"] == weights["destination"]
    assert report.data["dispatches"] == {"cpu_reference": 3}              # no calibration -> the reference row


@pytest.mark.gpu
@needs_two_gpus
@pytest.mark.parametrize("row", ["cpu_stages_cuda_stages@0", "cuda_dequant_relocate"])
def test_weight_fetcher_reaches_a_device_that_is_not_current(row):
    """Pins the workaround for the runtime's missing device scope around typed
    kernel launches: GPU rows alternate between the current device and another."""
    report = common.Report("fetch")
    fetcher = common.WeightFetcher(report, implementation=row, kind="expert_weights")
    q, scale = common.quantize_per_channel(torch.randn(512, 256, generator=torch.Generator().manual_seed(2)))
    torch.cuda.set_device(0)
    for device in ("cuda:0", "cuda:1", "cuda:0", "cuda:1"):
        actual = fetcher.fetch(q, scale, device)
        assert common.same_tensor(actual, common.WeightFetcher.reference(q, scale, device))
    assert report.data["dispatches"] == {row: 4}
    assert report.data["bytes"]["expert_weights"]["transfers"] == 4
