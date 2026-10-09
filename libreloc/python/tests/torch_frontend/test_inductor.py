"""Real Inductor codegen, opaque Sym ops, and the composition contract."""
import pytest
import torch

from reloc_torch import RelocBackend, compat


def test_compute_only_graph_reaches_inductor_and_rebinds():
    from torch._inductor import metrics

    backend = RelocBackend(compute_backend="inductor")
    fn = lambda x: (x.sin() + x.cos()).relu()
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    before = metrics.generated_kernel_count
    try:
        with torch.no_grad():
            for n in (17, 33, 65):
                x = torch.randn(n, 7)
                torch.testing.assert_close(compiled(x), fn(x))
        assert metrics.generated_kernel_count > before
        stats = backend.stats()
        assert stats["inductor_compiles"] == stats["dynamo_compiles"] == 1
        assert stats["inductor_executions"] == 3
        assert stats["plan_compiles"] == 0
        assert stats["replaced_regions"] == 0
        backend.close()
        with torch.no_grad(), pytest.raises(RuntimeError, match="closed"):
            compiled(x)
    finally:
        backend.close()


def test_explicit_inference_contract_and_options():
    with pytest.raises(ValueError, match="compute_backend"):
        RelocBackend(compute_backend="unknown")
    with pytest.raises(ValueError, match="requires"):
        RelocBackend(inductor_options={})
    with pytest.raises(ValueError, match="CUDA graph"):
        RelocBackend(compute_backend="inductor", inductor_options={"triton.cudagraphs": True})
    backend = RelocBackend(compute_backend="inductor")
    gm = torch.fx.symbolic_trace(lambda x: x.sin())
    x = torch.ones(5, requires_grad=True)
    try:
        with pytest.raises(RuntimeError, match="requires torch.no_grad"):
            backend(gm, [x])
        with torch.no_grad():
            compiled = backend(gm, [x])
            assert not compiled(x).requires_grad
        with pytest.raises(RuntimeError, match="requires torch.no_grad"):
            compiled(x)
    finally:
        backend.close()


def test_compile_failure_propagates_and_releases_handles(compiler, counting_runtime, monkeypatch):
    from test_backend import capture, cpu_candidate, transpose_recipe, snapshot

    gm = capture(lambda x: x.t().contiguous())
    recipe, sources = transpose_recipe()
    members = tuple(n.name for n in gm.graph.nodes if n.op == "call_function")
    importer = cpu_candidate(gm, lambda x: x.t().contiguous(), recipe, sources, members)
    backend = RelocBackend(compiler=compiler, runtime=counting_runtime,
                           importer=importer, compute_backend="inductor")
    before = snapshot(gm)
    def fail(graph, inputs, options):
        assert any("reloc_torch.transfer" in str(n.target) for n in graph.graph.nodes)
        raise RuntimeError("injected Inductor failure")
    monkeypatch.setattr(compat, "compile_inductor", fail)
    try:
        with torch.no_grad(), pytest.raises(RuntimeError, match="injected Inductor failure"):
            backend(gm, [torch.ones(3, 4)])
        assert snapshot(gm) == before
        assert backend.stats()["live_handles"] == 0
        assert backend.stats()["inductor_compile_failures"] == 1
        assert backend.stats()["inductor_compiles"] == 0
    finally:
        backend.close()


def test_guard_fallback_replays_only_the_transfer_under_inductor(compiler, counting_runtime, monkeypatch):
    from test_backend import capture, cpu_candidate, transpose_recipe, snapshot
    from reloc_torch.artifact import UnsupportedRecipe

    gm = capture(lambda x: (x.t().contiguous() + 2).sin())
    recipe, sources = transpose_recipe()
    members = tuple(n.name for n in gm.graph.nodes
                    if n.target in (torch.ops.aten.t.default, torch.ops.aten.clone.default))
    importer = cpu_candidate(gm, lambda x: x.t().contiguous(), recipe, sources, members)
    backend = RelocBackend(compiler=compiler, runtime=counting_runtime,
                           importer=importer, compute_backend="inductor")
    x = torch.randn(3, 4)
    before = snapshot(gm)
    try:
        with torch.no_grad():
            compiled = backend(gm, [x])
            assert snapshot(gm) == before
            torch.testing.assert_close(compiled(x), (x.t() + 2).sin())
            assert counting_runtime.executions == 1
            def reject(*args, **kwargs):
                raise UnsupportedRecipe("test_preflight", "explicit preflight rejection")
            monkeypatch.setattr(counting_runtime, "preflight", reject)
            torch.testing.assert_close(compiled(x + 1), (x.t() + 3).sin())
        assert backend.stats()["fallbacks"] == {"test_preflight": 1}
        assert counting_runtime.executions == 1
        assert backend.stats()["inductor_executions"] == 2
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["layout", "cast", "indexed", "d2h"])
def test_custom_op_composes_with_compute_dynamic_and_current_stream(cuda_device, kind):
    import pyreloc
    from torch._inductor import metrics

    if not pyreloc.cuda_enabled:
        pytest.skip("CUDA runtime unavailable")
    def fn(x, indices):
        if kind == "indexed":
            y = x.index_select(0, indices).to(cuda_device, dtype=torch.float16)
        elif kind == "d2h":
            y = x.t().contiguous().cpu()
        else:
            y = x.t().contiguous().to(cuda_device, dtype=(torch.float16 if kind == "cast" else x.dtype))
        return y, (y.float().sin() + y.float().cos()).relu()

    backend = RelocBackend(compute_backend="inductor")
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    stream = torch.cuda.Stream(device=cuda_device)
    before = metrics.generated_kernel_count
    try:
        with torch.no_grad(), torch.cuda.stream(stream):
            for n in (11, 19, 31):
                x = torch.randn(n, 7, device=cuda_device if kind == "d2h" else "cpu")
                # A CUDA producer on the nondefault caller stream must finish
                # before the Sym D2H source is read.
                x.add_(0.5)
                indices = torch.tensor([n-1, 0, n-1, 2])
                actual = compiled(x, indices)
                expected = fn(x, indices)
                assert actual[0].shape == expected[0].shape
                assert actual[0].stride() == expected[0].stride()
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1])
            stream.synchronize()
        stats = backend.stats()
        assert stats["runtime_executions"] == 3
        assert stats["inductor_compiles"] == stats["dynamo_compiles"] == 1
        assert stats["replaced_regions"] == 1
        assert not stats["fallbacks"]
        assert metrics.generated_kernel_count > before
    finally:
        backend.close()


@pytest.mark.gpu
def test_typed_parameters_remain_live_under_inductor(cuda_device):
    import pyreloc

    if not pyreloc.cuda_enabled:
        pytest.skip("CUDA runtime unavailable")
    qd = compat.quantized_decomposed()
    def fn(q, scale, zero):
        y = qd.dequantize_per_channel(q, scale, zero, 1, -128, 127, torch.int8)
        y = y.t().contiguous().to(cuda_device)
        return y, y.square() + 1

    backend = RelocBackend(compute_backend="inductor")
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
    q = torch.randint(-127, 127, (13, 7), dtype=torch.int8)
    scale, zero = torch.ones(7), torch.zeros(7, dtype=torch.int64)
    try:
        with torch.no_grad():
            for iteration in range(3):
                scale.numpy()[:] = 0.25 + iteration
                q.numpy()[0, 0] = iteration
                actual, expected = compiled(q, scale, zero), fn(q, scale, zero)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert backend.stats()["typed_executions"] == 3
        assert backend.stats()["inductor_compiles"] == 1
        assert not backend.stats()["fallbacks"]
    finally:
        backend.close()
