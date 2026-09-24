"""R4 (issue #148): combined compiler/runtime/frontend regressions and the
handoff runner's own accounting.

Component owners keep their exhaustive cases (R2 stream/lifetime in
test_transfers_gpu.py and test_transport.py, R3 rows in test_dispatch.py,
C4 numerics in test_typed_conformance*.py, T4 weights in test_weights.py).
This file adds what only the combination can show: the handoff runner fails
on missing prerequisites, failed children and malformed child output; the
standalone C++ consumer runs generated plans; and the T3 public entry point
carries the non-self-inverse permutation + pad witness and a typed D2H cast
through the real runtime with exact metadata.
"""
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[4]
RUNNER = REPO / "libreloc" / "python" / "examples" / "compiler_runtime_handoff.py"


def runner_module():
    spec = importlib.util.spec_from_file_location("compiler_runtime_handoff", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_runner(tmp_path, *args, env=None):
    out = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, str(RUNNER), "--device", "cpu", "--output", str(out), *args],
                          capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=900)
    return proc, json.loads(out.read_text()) if out.exists() else None


#===----------------------------------------------------------------------===#
# Runner accounting (CPU).
#===----------------------------------------------------------------------===#

def test_missing_exporter_is_a_failed_prerequisite_not_a_skip(tmp_path):
    proc, result = run_runner(tmp_path, env={"SYM_RELOC_EXPORT": str(tmp_path / "absent" / "sym-reloc-export")})
    assert proc.returncode == 2, proc.stderr
    assert result["scenarios"] == []
    assert any("exporter is missing" in p for p in result["prerequisites"])
    assert result["summary"]["prerequisites_failed"] >= 1


def test_a_failed_required_child_fails_the_run(tmp_path):
    build = tmp_path / "build"
    fake = build / "libreloc" / "examples" / "reloc-run-artifact"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\necho 'injected failure' >&2\nexit 3\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    proc, result = run_runner(tmp_path, "--build", str(build), "--only", "cpp_host_layout")
    assert proc.returncode == 1, proc.stderr
    scenario = {s["name"]: s for s in result["scenarios"]}["cpp_host_layout"]
    assert scenario["status"] == "failed" and "injected failure" in scenario["error"]
    assert result["summary"]["required_failed"] == 1


def test_malformed_child_output_is_a_failure():
    module = runner_module()
    assert module.child_json('log line\n{"ok": true}\n', "child") == {"ok": True}
    for text in ("", "no json here", '{"truncated": ', "[1, 2]"):
        with pytest.raises(module.ScenarioFailure):
            module.child_json(text, "child")


def test_the_cpp_consumer_runs_generated_plans_and_rejects_bad_input(tmp_path):
    """The public C++ path on the host: export -> reloc-run-artifact -> numpy
    replay, rebinding one artifact at 64/128/192; unknown symbols, guard
    violations, short input, malformed and typed plans fail with diagnostics."""
    proc, result = run_runner(tmp_path, "--only", "export_example_recipes", "cpp_host_layout", "cpp_rejections")
    assert proc.returncode == 0, proc.stderr + json.dumps(result, indent=1)[-2000:]
    names = {s["name"]: s for s in result["scenarios"]}
    cases = names["cpp_host_layout"]["details"]["split_transpose_rebind"]["cases"]
    assert [c["symbols"]["s0"] for c in cases] == [64, 128, 192]
    assert set(names["cpp_rejections"]["details"]) == {
        "unknown_symbol", "guard_violation", "short_input", "malformed_plan", "typed_plan_refused"}
    assert result["environment"]["source_revision"]


#===----------------------------------------------------------------------===#
# T3 public entry point + real runtime (GPU).
#===----------------------------------------------------------------------===#

def _metadata(t):
    return tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device.type


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_permutation_and_pad_witness_through_torch_compile_both_directions():
    from reloc_torch import RelocBackend

    def h2d(x):
        y = x.permute(1, 2, 0).contiguous()
        return torch.nn.functional.pad(y, (0, 0, 0, 0, 1, 2), value=1.5).to("cuda")

    def d2h(x):
        y = x.permute(1, 2, 0).contiguous()
        return torch.nn.functional.pad(y, (0, 0, 0, 0, 1, 2), value=1.5).cpu()

    for fn, source_device in ((h2d, "cpu"), (d2h, "cuda")):
        backend = RelocBackend()
        try:
            compiled = torch.compile(fn, backend=backend, dynamic=False)
            x = (torch.arange(24, dtype=torch.float32) * 0.25 - 3).reshape(2, 3, 4).to(source_device)
            with torch.no_grad():
                actual = compiled(x)
                expected = fn(x)
            torch.cuda.synchronize()
            assert _metadata(actual) == _metadata(expected)
            assert torch.equal(actual.cpu(), expected.cpu())
            # Not an inverse scatter: the permutation is not its own inverse.
            assert not torch.equal(actual.cpu()[1:4], x.cpu().permute(2, 0, 1).contiguous().reshape(actual[1:4].shape))
            stats = backend.stats()
            assert stats["runtime_executions"] == 1 and stats["replaced_regions"] == 1 and not stats["fallbacks"], stats
        finally:
            backend.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_typed_d2h_cast_runs_the_forward_program_through_torch_compile():
    from reloc_torch import RelocBackend

    def fn(x):
        return x.t().contiguous().to("cpu", torch.float16)

    backend = RelocBackend()
    try:
        compiled = torch.compile(fn, backend=backend, dynamic=False)
        x = (torch.randn(6, 8) * 1000).cuda()
        x[0, 0], x[0, 1] = 70000.0, -0.0
        with torch.no_grad():
            actual = compiled(x)
            expected = fn(x)
        assert _metadata(actual) == _metadata(expected)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        stats = backend.stats()
        assert stats["typed_executions"] == 1, stats
        assert stats["dispatches"] == {"cpu_reference": 1}  # no GPU narrowing kernel: the CPU reference ran
        assert stats["typed_payload_bytes"] == 6 * 8 * 4  # D2H moves the dense f32 source
    finally:
        backend.close()
