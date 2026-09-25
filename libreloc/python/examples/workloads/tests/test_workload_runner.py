"""The workload runner's accounting (CPU; small fake examples stand in for the real ones)."""
import json
import pathlib
import subprocess
import sys

import run_workloads as runner

EXAMPLE_HEAD = (
    "import argparse, json, sys\n"
    "parser = argparse.ArgumentParser()\n"
    "for flag in ('--output', '--device', '--devices', '--calibration'):\n"
    "    parser.add_argument(flag)\n"
    "parser.add_argument('--quick', action='store_true')\n"
    "a = parser.parse_args()\n"
)


def fake_example(path, body):
    path.write_text(EXAMPLE_HEAD + body)
    return path


def fake_build(root):
    (root / "python" / "pyreloc").mkdir(parents=True)
    tools = root / "sym" / "tools"
    tools.mkdir(parents=True)
    (tools / "sym-reloc-export").write_text("#!/bin/sh\n")
    return root


def report_body(ok):
    return (f"json.dump({{'ok': {ok}, 'summary': {{'checks_passed': {int(ok)}, 'checks_total': 1}}}}, "
            "open(a.output, 'w'))\n")


def test_preflight_names_each_missing_piece(tmp_path):
    problems = runner.preflight(tmp_path / "no-python", tmp_path / "no-build", [tmp_path / "missing.py"])
    assert len(problems) == 4
    assert "interpreter" in problems[0] and "pyreloc" in problems[1]
    assert "exporter" in problems[2] and "missing.py" in problems[3]
    assert runner.preflight(pathlib.Path(sys.executable), fake_build(tmp_path / "b"), [pathlib.Path(__file__)]) == []


def test_a_missing_build_exits_two_before_running_anything(tmp_path):
    proc = subprocess.run([sys.executable, runner.__file__, "--build", str(tmp_path / "absent"),
                           "--python", sys.executable, "--output-dir", str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "prerequisite: no staged pyreloc" in proc.stderr
    assert not (tmp_path / "out").exists()


def test_child_arguments_route_devices_and_calibration():
    args = runner.parse_args(["--quick", "--devices", "cuda:1,cuda:2", "--calibration", "none"])
    output = pathlib.Path("o.json")
    assert runner.child_arguments("dlrm", args, output) == ["--output", "o.json", "--quick", "--device", "cuda:1"]
    assert runner.child_arguments("llm", args, output) == [
        "--output", "o.json", "--quick", "--device", "cuda:1", "--calibration", "none"]
    assert runner.child_arguments("moe", args, output) == [
        "--output", "o.json", "--quick", "--devices", "cuda:1,cuda:2", "--calibration", "none"]
    assert "--quick" not in runner.child_arguments("gnn", runner.parse_args([]), output)


def test_classify_and_overall_exit():
    assert runner.classify(0, {"ok": True}) == "pass"
    assert runner.classify(0, {"ok": False}) == "fail"
    assert runner.classify(1, {"ok": False}) == "fail"
    assert runner.classify(0, None) == "crash"                  # exit 0 without a report
    assert runner.classify(1, None) == "crash"
    assert runner.classify(-11, {"ok": True}) == "crash"        # killed by a signal
    assert runner.classify(2, None) == "prerequisite"
    assert runner.overall_exit(["pass", "pass"]) == 0
    assert runner.overall_exit(["pass", "fail"]) == 1
    assert runner.overall_exit(["timeout", "crash", "pass"]) == 1
    assert runner.overall_exit(["fail", "prerequisite"]) == 2


def test_run_collects_reports_logs_and_the_worst_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "EXAMPLES", {
        "dlrm": fake_example(tmp_path / "good.py", report_body(True)),
        "gnn": fake_example(tmp_path / "bad.py", report_body(False) + "sys.exit(1)\n"),
        "llm": fake_example(tmp_path / "crash.py", "raise RuntimeError('boom')\n"),
    })
    out = tmp_path / "out"
    code = runner.main(["--build", str(fake_build(tmp_path / "build")), "--python", sys.executable,
                        "--output-dir", str(out)])
    assert code == 1
    summary = json.loads((out / "summary.json").read_text())
    assert [(r["name"], r["status"]) for r in summary] == [("dlrm", "pass"), ("gnn", "fail"), ("llm", "crash")]
    assert summary[0]["summary"] == {"checks_passed": 1, "checks_total": 1}
    assert "boom" in (out / "llm.log").read_text()
    table = capsys.readouterr().out
    assert "example" in table and "1/1" in table and "crash" in table


def test_only_runs_the_selected_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXAMPLES", {
        "dlrm": fake_example(tmp_path / "good.py", report_body(True)),
        "gnn": tmp_path / "never-created.py",
    })
    out = tmp_path / "out"
    code = runner.main(["--build", str(fake_build(tmp_path / "build")), "--python", sys.executable,
                        "--output-dir", str(out), "--only", "dlrm"])
    assert code == 0
    assert [r["name"] for r in json.loads((out / "summary.json").read_text())] == ["dlrm"]


def test_a_timeout_is_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXAMPLES", {"dlrm": fake_example(tmp_path / "slow.py", "import time\ntime.sleep(60)\n")})
    out = tmp_path / "out"
    code = runner.main(["--build", str(fake_build(tmp_path / "build")), "--python", sys.executable,
                        "--output-dir", str(out), "--timeout", "2"])
    assert code == 1
    assert json.loads((out / "summary.json").read_text())[0]["status"] == "timeout"


def test_a_stale_report_is_never_reused(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "dlrm.json").write_text(json.dumps({"ok": True}))
    monkeypatch.setattr(runner, "EXAMPLES", {"dlrm": fake_example(tmp_path / "silent.py", "sys.exit(0)\n")})
    code = runner.main(["--build", str(fake_build(tmp_path / "build")), "--python", sys.executable,
                        "--output-dir", str(out)])
    assert code == 1
    assert json.loads((out / "summary.json").read_text())[0]["status"] == "crash"


def test_relative_paths_resolve_against_the_callers_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXAMPLES", {"moe": fake_example(
        tmp_path / "echo.py", "json.dump({'ok': True, 'calibration': a.calibration}, open(a.output, 'w'))\n")})
    fake_build(tmp_path / "build")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    code = runner.main(["--build", "../build", "--python", sys.executable, "--output-dir", "out",
                        "--calibration", "my.cal"])
    assert code == 0
    assert json.loads((work / "out" / "moe.json").read_text())["calibration"] == str(work / "my.cal")
