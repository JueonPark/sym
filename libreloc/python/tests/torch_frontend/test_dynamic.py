"""Symbolic plan reuse across bindings, guard fallback and separate counters (T4 Task 1)."""
import pytest
import torch

from conftest import make_entry


gpu = pytest.mark.gpu
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def split_transpose(x):
    return x.reshape(x.shape[0] // 64, 64).t().contiguous().to("cuda")


def _spying_backend(compiler, runtime):
    """A RelocBackend whose importer records every ImportReport it produced."""
    from reloc_torch import RelocBackend, import_graph

    reports = []

    def importer(gm, example_inputs):
        report = import_graph(gm, example_inputs)
        reports.append(report)
        return report

    backend = RelocBackend(compiler=compiler, runtime=runtime, importer=importer)
    return backend, reports


@gpu
@needs_cuda
def test_dynamic_plan_reuse(backend):
    def fn(x):
        return x.reshape(x.shape[0] // 64, 64).t().contiguous().to("cuda")

    compiled_fn = torch.compile(fn, backend=backend, dynamic=True)
    start = backend.stats()
    with torch.no_grad():
        for n in (128, 192, 256):
            x = torch.arange(n, dtype=torch.float32)
            actual = compiled_fn(x)
            torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    end = backend.stats()
    assert end["plan_compiles"] - start["plan_compiles"] == 1
    assert end["symbol_binds"] - start["symbol_binds"] == 3
    assert end["runtime_executions"] - start["runtime_executions"] == 3
    assert end["dynamo_compiles"] > start["dynamo_compiles"]


@gpu
@needs_cuda
def test_dynamic_capture_is_symbolic_and_one_artifact_serves_every_binding(compiler, real_runtime):
    from reloc_torch.symbolic import FloorDiv, Symbol

    backend, reports = _spying_backend(compiler, real_runtime)
    compiled_fn = torch.compile(split_transpose, backend=backend, dynamic=True)
    with torch.no_grad():
        for n in (128, 192, 256):
            x = torch.arange(n, dtype=torch.float32)
            actual = compiled_fn(x)
            expected = split_transpose(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.stride() == expected.stride() and actual.storage_offset() == 0
    candidates = [c for report in reports for c in report.candidates]
    assert candidates, [report.exclusions for report in reports]
    recipe = candidates[0].recipe
    # Dimensions stay symbolic: the source is (s0,), the destination (64, s0 // 64).
    assert recipe.source.shape == (Symbol("s0"),)
    assert recipe.destination.shape[0].value == 64 and recipe.destination.shape[1] == FloorDiv(Symbol("s0"), 64)
    assert {c.recipe.canonical_identity for c in candidates} == {recipe.canonical_identity}
    stats = backend.stats()
    assert stats["plan_compiles"] == 1
    assert stats["runtime_executions"] == 3
    assert stats["symbol_binds"] == 3
    assert stats["dynamo_compiles"] == len(reports) >= 1
    backend.close()


def test_one_artifact_rebinds_across_sizes_on_the_host_reference_path(compiler, counting_runtime):
    """CPU variant: the real compiler, binder and host executor, one artifact, three bindings."""
    from reloc_torch import import_graph
    from reloc_torch.compat import symbolic_capture
    from reloc_torch.runtime import execute_or_fallback

    gm = symbolic_capture(
        lambda x: torch.ops.aten._to_copy.default(
            x.reshape(x.shape[0] // 64, 64).t().contiguous(), device=torch.device("cuda:0")),
        torch.ones(128),
    )
    candidate, = import_graph(gm).candidates
    compiled = compiler.compile(candidate.recipe)
    entry = make_entry(compiled, counting_runtime, lambda src, *s: src.reshape(-1, 64).t().contiguous())
    for n in (128, 192, 256):
        x = torch.arange(n, dtype=torch.float32)
        actual = execute_or_fallback(entry, x, [n], torch.device("cpu"))
        expected = x.reshape(n // 64, 64).t().contiguous()
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()
    snapshot = entry.diagnostics.snapshot()
    assert counting_runtime.executions == 3
    assert snapshot["symbol_binds"] == 3
    assert snapshot["runtime_executions"] == 3
    assert snapshot["fallbacks"] == {}


@gpu
@needs_cuda
def test_invalid_divisibility_preserves_pytorch_exception_without_launch(backend):
    compiled_fn = torch.compile(split_transpose, backend=backend, dynamic=True)
    with torch.no_grad():
        compiled_fn(torch.arange(128, dtype=torch.float32))
        before = backend.stats()
        x = torch.arange(130, dtype=torch.float32)
        with pytest.raises(RuntimeError) as eager_failure:
            split_transpose(x)
        with pytest.raises(RuntimeError) as compiled_failure:
            compiled_fn(x)
    # Dynamo re-traces the failing shape and surfaces PyTorch's own reshape
    # error (wrapped in its RuntimeError subclass); nothing of ours launched.
    assert isinstance(compiled_failure.value, RuntimeError)
    assert "shape" in str(compiled_failure.value) and "shape" in str(eager_failure.value)
    after = backend.stats()
    assert after["runtime_executions"] == before["runtime_executions"]
    assert after["plan_compiles"] == before["plan_compiles"]


@gpu
@needs_cuda
def test_noncontiguous_input_falls_back_while_pytorch_succeeds(compiler, real_runtime):
    backend, reports = _spying_backend(compiler, real_runtime)
    compiled_fn = torch.compile(split_transpose, backend=backend, dynamic=True)
    with torch.no_grad():
        compiled_fn(torch.arange(128, dtype=torch.float32))
        executed = backend.stats()["runtime_executions"]
        strided = torch.arange(256, dtype=torch.float32)[::2]
        actual = compiled_fn(strided)
        expected = split_transpose(strided)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert backend.stats()["runtime_executions"] == executed
    reasons = {e.reason for report in reports for e in report.exclusions}
    # Dynamo gives the strided input a free stride symbol; the importer either
    # rejects the non-dense root or cannot express that stride. Both fall back.
    assert reasons & {"source_layout", "unsupported_symbolic_expr"}
    backend.close()


@gpu
@needs_cuda
def test_new_dtype_is_a_distinct_family_and_excluded_inputs_fall_back(backend):
    compiled_fn = torch.compile(split_transpose, backend=backend, dynamic=True)
    with torch.no_grad():
        compiled_fn(torch.arange(128, dtype=torch.float32))
        compiled_fn(torch.arange(192, dtype=torch.float32))
        assert backend.stats()["plan_compiles"] == 1
        half = compiled_fn(torch.arange(128, dtype=torch.float16))
        torch.testing.assert_close(half, split_transpose(torch.arange(128, dtype=torch.float16)), rtol=0, atol=0)
        assert backend.stats()["plan_compiles"] == 2
        executed = backend.stats()["runtime_executions"]
        empty = compiled_fn(torch.empty(0, dtype=torch.float32))
        assert empty.shape == (64, 0) and empty.device.type == "cuda"
        # A scalar has no dim 0: PyTorch (and Dynamo re-tracing it) raise the
        # indexing error; the frontend never launched anything for it.
        with pytest.raises((IndexError, RuntimeError)):
            split_transpose(torch.ones((), dtype=torch.float32))
        with pytest.raises((IndexError, RuntimeError)):
            compiled_fn(torch.ones((), dtype=torch.float32))
        assert backend.stats()["runtime_executions"] == executed
