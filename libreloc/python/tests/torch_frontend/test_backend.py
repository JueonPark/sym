"""Graph rewriting safety, eager routing, reentrancy and lifetime."""
import gc
import threading

import pytest
import torch

from conftest import CountingRuntime


def backend_module():
    from reloc_torch import backend

    return backend


def eager_module():
    from reloc_torch import eager

    return eager


def op():
    from reloc_torch import ops

    return torch.ops.reloc_torch.transfer.default


def capture(fn, *shapes, **options):
    from reloc_torch.compat import symbolic_capture

    shapes = shapes or ((3, 4),)
    return symbolic_capture(fn, *(torch.ones(shape, **options) for shape in shapes))


def transfer(x, **kwargs):
    return torch.ops.aten._to_copy.default(x, device=torch.device("cuda:0"), **kwargs)


def snapshot(gm):
    return str(gm.graph), [(n.name, tuple(u.name for u in n.users), dict(n.meta)) for n in gm.graph.nodes]


def op_nodes(gm):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is op()]


def targets(gm):
    return [str(n.target) for n in gm.graph.nodes if n.op != "placeholder" and n.op != "output"]


@pytest.fixture
def cpu_backend(compiler, counting_runtime):
    backend = backend_module().RelocBackend(compiler=compiler, runtime=counting_runtime)
    yield backend
    backend.close()


def cpu_candidate(gm, fn, recipe, symbol_sources, members):
    """Hand-built candidate over a CPU-only layout region for control-flow tests."""
    from reloc_torch.fx_import import Candidate, ImportReport

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    candidate = Candidate(names[0], members[-1], tuple(members), recipe, tuple(symbol_sources), (), torch.fx.symbolic_trace(fn))
    return lambda graph, inputs: ImportReport((candidate,), ())


def transpose_recipe(dtype="float32"):
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol, SymbolSource

    rows, columns = Symbol("s0"), Symbol("s1")
    recipe = Recipe(
        TensorSpec((rows, columns), (columns, Const(1)), Const(0), dtype),
        (Transpose((1, 0)),),
        TensorSpec((columns, rows), (rows, Const(1)), Const(0), dtype),
        "h2d",
    )
    return recipe, (SymbolSource("s0", 0), SymbolSource("s1", 1))


def test_rewrite_inserts_symbolic_custom_op_and_preserves_the_original_graph(cpu_backend):
    from reloc_torch.cache import REGISTRY

    gm = capture(lambda x: transfer(x.transpose(0, 1).contiguous()))
    before = snapshot(gm)
    compiled = cpu_backend(gm, None)
    assert snapshot(gm) == before
    rewritten = compiled.rewritten
    node, = op_nodes(rewritten)
    assert not any(t in targets(rewritten) for t in ("aten._to_copy.default", "aten.transpose.int", "aten.clone.default"))
    root, handle, symbols, out_shape, out_strides, device = node.args
    assert root.op == "placeholder"
    assert isinstance(handle, str) and handle.startswith("reloc-")
    assert [n.target for n in symbols] == [torch.ops.aten.sym_size.int] * 2
    assert [tuple(n.args) for n in symbols] == [(root, 0), (root, 1)]
    assert out_shape == [symbols[1], symbols[0]]
    assert out_strides == [symbols[0], 1]
    assert device == torch.device("cuda", 0)
    assert node.meta["val"].device.type == "cuda"
    output = next(n for n in rewritten.graph.nodes if n.op == "output")
    assert output.args[0] is node or output.args[0] == (node,) or node in output.args[0]
    rewritten.graph.lint()
    entry = REGISTRY.lookup(handle)
    assert entry.compiled.recipe.direction == "h2d"
    stats = cpu_backend.stats()
    assert stats["dynamo_compiles"] == 1
    assert stats["plan_compiles"] == 1
    assert stats["replaced_regions"] == 1
    assert stats["live_handles"] == 1
    assert stats["runtime_executions"] == 0


def test_rejected_region_retains_original_nodes_and_records_the_reason(cpu_backend):
    def fn(x):
        y = x.transpose(0, 1)
        return transfer(y.contiguous()), y

    gm = capture(fn)
    compiled = cpu_backend(gm, None)
    assert not op_nodes(compiled.rewritten)
    assert targets(compiled.rewritten) == targets(gm)
    stats = cpu_backend.stats()
    assert stats["exclusions"]["escaping_intermediate"] == 1
    assert stats["replaced_regions"] == 0
    assert stats["plan_compiles"] == 0


def _copy_into(x):
    y = torch.ops.aten.empty.memory_format([4, 3], dtype=torch.float32, device=torch.device("cuda:0"))
    return torch.ops.aten.copy_.default(y, x.transpose(0, 1))


def _view_mutation(x):
    alias = x.view(-1)
    y = transfer(x.transpose(0, 1).contiguous())
    alias.add_(1)
    return y


def _shared_user(x):
    y = x.transpose(0, 1).contiguous()
    return transfer(y), y.sum()


def _two_transfers(x):
    y = transfer(x)
    return torch.ops.aten._to_copy.default(y, device=torch.device("cpu"))


def _unknown_effect(x):
    y = x.transpose(0, 1)
    torch.ops.aten.rand.default([1])
    return transfer(y.contiguous())


@pytest.mark.parametrize(
    ("fn", "reasons"),
    [
        (lambda x: (transfer(x.transpose(0, 1).contiguous()), x.transpose(0, 1).contiguous()), set()),
        (_shared_user, {"escaping_intermediate"}),
        (_copy_into, set()),
        (_view_mutation, {"mutation"}),
        (_unknown_effect, {"unknown_side_effect"}),
        (_two_transfers, {"multiple_transfers"}),
        (lambda x: transfer(x.transpose(0, 1)), {"destination_layout"}),
        (lambda x: x.transpose(0, 1), set()),
    ],
    ids=["returned_intermediate", "shared_user", "copy_", "view_mutation", "unknown_effect", "two_transfers", "noncontiguous_result", "view_only"],
)
def test_adversarial_graphs_are_never_replaced(cpu_backend, fn, reasons):
    gm = capture(fn)
    before = snapshot(gm)
    compiled = cpu_backend(gm, None)
    assert snapshot(gm) == before
    if fn.__name__ == "<lambda>" and reasons == set() and "contiguous" in str(gm.graph) and "_to_copy" in str(gm.graph):
        # Two independent copies of one region: only the transferred one is a
        # candidate and it may be replaced; the returned view chain survives.
        assert "aten.transpose.int" in targets(compiled.rewritten)
    else:
        assert not op_nodes(compiled.rewritten)
        assert targets(compiled.rewritten) == targets(gm)
    stats = cpu_backend.stats()
    assert reasons <= set(stats["exclusions"])
    assert stats["runtime_executions"] == 0


def test_cpu_region_executes_through_the_adapter_and_replays_only_itself_on_guard_miss(compiler, counting_runtime):
    def fn(x, counter):
        counter.add_(1)
        return x.transpose(0, 1).contiguous()

    gm = capture(fn, (3, 4), (1,))
    recipe, sources = transpose_recipe()
    importer = cpu_candidate(gm, lambda src: src.transpose(0, 1).contiguous(), recipe, sources, ("transpose", "clone"))
    backend = backend_module().RelocBackend(compiler=compiler, runtime=counting_runtime, importer=importer)
    compiled = backend(gm, None)
    node, = op_nodes(compiled.rewritten)
    assert node.args[5] == torch.device("cpu")
    assert "aten.add_.Tensor" in targets(compiled.rewritten)

    counter = torch.zeros(1)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    actual = compiled(x, counter)
    expected = fn(x, torch.zeros(1))
    assert torch.equal(actual, expected) and actual.stride() == expected.stride()
    assert counter.item() == 1
    assert counting_runtime.executions == 1

    strided = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
    actual = compiled(strided, counter)
    assert torch.equal(actual, strided.transpose(0, 1).contiguous())
    assert counter.item() == 2
    assert counting_runtime.executions == 1
    stats = backend.stats()
    assert stats["fallbacks"]["unsupported_layout"] == 1
    assert stats["runtime_executions"] == 1

    other = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    assert torch.equal(compiled(other, counter), other.t().contiguous())
    assert counting_runtime.executions == 2
    assert backend.stats()["plan_compiles"] == 1

    grad_input = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    result = compiled(grad_input, counter)
    assert result.requires_grad is True
    assert torch.equal(result.detach(), grad_input.detach().t().contiguous())
    assert counting_runtime.executions == 2
    assert backend.stats()["fallbacks"]["unsupported_layout"] == 1
    backend.close()


def test_training_capture_leaves_the_graph_unmodified(compiler, counting_runtime):
    gm = capture(lambda x: x.transpose(0, 1).contiguous())
    recipe, sources = transpose_recipe()
    importer = cpu_candidate(gm, lambda src: src.transpose(0, 1).contiguous(), recipe, sources, ("transpose", "clone"))
    backend = backend_module().RelocBackend(compiler=compiler, runtime=counting_runtime, importer=importer)
    compiled = backend(gm, [torch.ones(3, 4, requires_grad=True)])
    assert compiled.rewritten is None
    x = torch.ones(3, 4, requires_grad=True)
    assert compiled(x).requires_grad
    assert backend.stats()["exclusions"]["requires_grad"] == 1
    assert backend.stats()["plan_compiles"] == 0
    backend.close()


def test_graph_callable_owns_handles_and_survives_cache_eviction(compiler, counting_runtime):
    from reloc_torch.cache import REGISTRY

    gm = capture(lambda x: x.transpose(0, 1).contiguous())
    recipe, sources = transpose_recipe()
    importer = cpu_candidate(gm, lambda src: src.transpose(0, 1).contiguous(), recipe, sources, ("transpose", "clone"))
    backend = backend_module().RelocBackend(compiler=compiler, runtime=counting_runtime, importer=importer, cache_capacity=1)
    compiled = backend(gm, None)
    handle = op_nodes(compiled.rewritten)[0].args[1]
    other_recipe, other_sources = transpose_recipe("float16")
    gm16 = capture(lambda x: x.transpose(0, 1).contiguous(), (3, 4), dtype=torch.float16)
    backend._importer = cpu_candidate(gm16, lambda src: src.transpose(0, 1).contiguous(), other_recipe, other_sources, ("transpose", "clone"))
    compiled16 = backend(gm16, None)
    assert backend.stats()["cache_entries"] == 1
    assert backend.stats()["plan_compiles"] == 2
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(compiled(x), x.t().contiguous())
    assert torch.equal(compiled16(x.half()), x.half().t().contiguous())
    assert REGISTRY.lookup(handle) is not None
    assert backend.stats()["live_handles"] == 2
    del compiled
    gc.collect()
    with pytest.raises(RuntimeError, match="unknown or closed"):
        REGISTRY.lookup(handle)
    assert backend.stats()["live_handles"] == 1
    backend.close()
    backend.close()
    assert backend.stats()["closed"] is True
    assert backend.stats()["live_handles"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend(gm, None)
    with pytest.raises(RuntimeError, match="unknown or closed"):
        compiled16(x.half())


def test_default_construction_resolves_compiler_and_transport_lazily(monkeypatch):
    from reloc_torch.runtime import TransportAdapter

    monkeypatch.delenv("SYM_RELOC_EXPORT", raising=False)
    monkeypatch.delenv("SYM_OPT", raising=False)
    backend = backend_module().RelocBackend()
    assert isinstance(backend.runtime, TransportAdapter)
    with pytest.raises(RuntimeError, match="SYM_RELOC_EXPORT"):
        backend.compiler
    monkeypatch.setenv("SYM_RELOC_EXPORT", "/nonexistent/sym-reloc-export")
    with pytest.raises(RuntimeError, match="absent"):
        backend_module().RelocBackend().compiler
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="absent"):
        with eager_module().eager_transfers(backend=backend_module().RelocBackend()):
            x.to(copy=True)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda: (torch.ones(3, 4), dict(device=torch.device("cuda:0"))), None),
        (lambda: (torch.ones(3, 4), dict(device=torch.device("cuda:0"), non_blocking=True)), "nonblocking_unavailable"),
        (lambda: (torch.ones(3, 4), dict(device=torch.device("cuda:0"), dtype=torch.float16)), "typed_transform_unavailable"),
        (lambda: (torch.ones(2, 3, 4, 5), dict(device=torch.device("cuda:0"), memory_format=torch.channels_last)), "unsupported_memory_format"),
        (lambda: (torch.ones(3, 4), dict(device=torch.device("cuda:0"), pin_memory=True)), "pinned_transfer_unavailable"),
        (lambda: (torch.ones(3, 4, requires_grad=True), dict(device=torch.device("cuda:0"))), "requires_grad"),
        (lambda: (torch.ones(3, 4), dict()), "same_device_copy"),
        (lambda: (torch.ones(3, 4), dict(device=torch.device("cpu"))), "same_device_copy"),
        (lambda: (torch.ones(3, 4), dict(device=torch.device("meta"))), "unsupported_device"),
        (lambda: (torch.ones(3, 8)[:, ::2], dict(device=torch.device("cuda:0"))), "unsupported_layout"),
        (lambda: (torch.ones(8)[2:], dict(device=torch.device("cuda:0"))), "storage_offset"),
        (lambda: (torch.ones(0, 4), dict(device=torch.device("cuda:0"))), "empty_tensor"),
        (lambda: (torch.ones(()), dict(device=torch.device("cuda:0"))), "unsupported_rank"),
        (lambda: (torch.ones(3, 4, dtype=torch.float64), dict(device=torch.device("cuda:0"))), "unsupported_dtype"),
    ],
)
def test_eager_decision_uses_the_t1_overload_inventory(make, reason):
    src, options = make()
    decision = eager_module().decide(torch.ops.aten._to_copy.default, (src,), options)
    assert decision.reason == reason
    if reason is None:
        assert decision.direction == "h2d"
        assert decision.device == torch.device("cuda:0")


def test_eager_decision_rejects_subclasses_and_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    class Subclass(torch.Tensor):
        pass

    decide = eager_module().decide
    assert decide(torch.ops.aten._to_copy.default, (torch.ones(3).as_subclass(Subclass),), dict(device=torch.device("cuda:0"))).reason == "tensor_subclass"
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.ones(3))
    assert decide(torch.ops.aten._to_copy.default, (fake,), dict(device=torch.device("cuda:0"))).reason == "tensor_subclass"
    assert decide(torch.ops.aten.copy_.default, (torch.ones(3), torch.ones(3)), {}).reason == "unsupported_operator"


def test_eager_identity_artifact_is_shared_per_rank_dtype_and_direction(cpu_backend):
    entry_a = cpu_backend.eager_entry(torch.ones(3, 4), torch.device("cuda:0"), lambda src: src)
    entry_b = cpu_backend.eager_entry(torch.ones(7, 2), torch.device("cuda:0"), lambda src: src)
    entry_c = cpu_backend.eager_entry(torch.ones(5), torch.device("cuda:0"), lambda src: src)
    assert entry_a.compiled is entry_b.compiled
    assert entry_a.compiled is not entry_c.compiled
    assert entry_a.compiled.recipe.operations == ()
    assert entry_a.compiled.recipe.direction == "h2d"
    assert len(entry_a.compiled.symbols) == 2
    assert cpu_backend.stats()["plan_compiles"] == 2
    assert cpu_backend.stats()["cache_hits"] == 1
    entry_d = cpu_backend.eager_entry(torch.ones(3, 4, device="meta"), torch.device("cpu"), lambda src: src)
    assert entry_d.compiled.recipe.direction == "d2h"


def test_eager_scope_redispatches_ineligible_transfers_and_restores_state(cpu_backend):
    eager = eager_module()
    depth = torch._C._len_torch_dispatch_stack()
    x = torch.arange(6, dtype=torch.float32)
    with eager.eager_transfers(backend=cpu_backend):
        y = x.to(copy=True)
        with eager.eager_transfers(backend=cpu_backend):
            z = x.to(torch.float16)
        assert torch._C._len_torch_dispatch_stack() == depth + 1
    assert torch._C._len_torch_dispatch_stack() == depth
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()
    assert z.dtype == torch.float16
    stats = cpu_backend.stats()
    assert stats["redispatches"]["same_device_copy"] == 1
    assert stats["redispatches"]["typed_transform_unavailable"] == 1
    assert stats["runtime_executions"] == 0
    with pytest.raises(ValueError):
        with eager.eager_transfers(backend=cpu_backend):
            raise ValueError("boom")
    assert torch._C._len_torch_dispatch_stack() == depth


def test_eager_scope_is_thread_local_and_suspension_blocks_interception(cpu_backend):
    eager = eager_module()
    from reloc_torch.runtime import suspend_interception

    seen = {}
    x = torch.arange(6, dtype=torch.float32)

    def other_thread():
        x.to(copy=True)
        seen["stack"] = torch._C._len_torch_dispatch_stack()

    with eager.eager_transfers(backend=cpu_backend):
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
        with suspend_interception():
            x.to(copy=True)
        assert cpu_backend.stats()["redispatches"] == {}
        x.to(copy=True)
    assert seen["stack"] == 0
    assert cpu_backend.stats()["redispatches"] == {"same_device_copy": 1}


def test_eager_activation_fails_early_on_unresolvable_configuration(monkeypatch, counting_runtime):
    monkeypatch.delenv("SYM_RELOC_EXPORT", raising=False)
    monkeypatch.delenv("SYM_OPT", raising=False)
    backend = backend_module().RelocBackend(runtime=counting_runtime)
    x = torch.arange(6, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="SYM_RELOC_EXPORT"):
        with eager_module().eager_transfers(backend=backend):
            x.to(copy=True)
    assert torch._C._len_torch_dispatch_stack() == 0
    assert backend.stats()["redispatches"] == {}


def test_closed_backend_rejects_eager_activation(compiler, counting_runtime):
    backend = backend_module().RelocBackend(compiler=compiler, runtime=counting_runtime)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        with eager_module().eager_transfers(backend=backend):
            pass


def test_public_entry_points_are_lazy_and_torch_free():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys; from reloc_torch import RelocBackend, eager_transfers; assert callable(RelocBackend) and callable(eager_transfers); assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def backend(compiler, real_runtime):
    result = backend_module().RelocBackend(compiler=compiler, runtime=real_runtime)
    yield result
    result.close()


@pytest.fixture
def production_backend(compiler):
    from reloc_torch.runtime import TransportAdapter

    result = backend_module().RelocBackend(compiler=compiler, runtime=TransportAdapter())
    yield result
    result.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_layout_transfer(backend):
    def fn(x):
        return x.to("cuda").transpose(0, 1).contiguous()

    compiled_fn = torch.compile(fn, backend=backend, dynamic=True)
    with torch.no_grad():
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        actual = compiled_fn(x)
        expected = fn(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert backend.stats()["runtime_executions"] >= 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_transfer_rewrites_and_falls_back_correctly_through_the_production_adapter(production_backend):
    def fn(x):
        return x.to("cuda").transpose(0, 1).contiguous()

    compiled_fn = torch.compile(fn, backend=production_backend, dynamic=True)
    with torch.no_grad():
        for rows in (4, 6):
            x = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6)
            actual = compiled_fn(x)
            expected = fn(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.stride() == expected.stride()
            assert actual.storage_offset() == 0
            assert actual.device == expected.device
    stats = production_backend.stats()
    assert stats["dynamo_compiles"] >= 1
    assert stats["plan_compiles"] == 1
    assert stats["replaced_regions"] >= 1
    if production_backend.runtime.available:
        assert stats["runtime_executions"] >= 2
    else:
        assert stats["runtime_executions"] == 0
        assert stats["fallbacks"]["runtime_unavailable"] >= 2


def _gpu_copy_(x):
    y = torch.empty(4, 3, device="cuda")
    y.copy_(x.transpose(0, 1))
    return y


def _gpu_view_mutation(x):
    alias = x.view(-1)
    y = x.transpose(0, 1).contiguous().to("cuda")
    alias.add_(1)
    return y, x


def _gpu_returned_intermediate(x):
    y = x.transpose(0, 1).contiguous()
    return y.to("cuda"), y


def _gpu_two_transfers(x):
    return x.to("cuda").cpu()


def _gpu_noncontiguous(x):
    return x.to("cuda").transpose(0, 1)


def _gpu_view_only(x):
    return x.transpose(0, 1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("fn", [_gpu_copy_, _gpu_view_mutation, _gpu_returned_intermediate, _gpu_two_transfers, _gpu_noncontiguous, _gpu_view_only])
def test_excluded_regions_match_pytorch_values_aliases_and_versions_on_cuda(production_backend, fn):
    compiled_fn = torch.compile(fn, backend=production_backend, dynamic=True)
    with torch.no_grad():
        x_actual = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        x_expected = x_actual.clone()
        actual = compiled_fn(x_actual)
        expected = fn(x_expected)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
        assert a.stride() == e.stride() and a.device == e.device
    assert torch.equal(x_actual, x_expected)
    assert x_actual._version == x_expected._version
    stats = production_backend.stats()
    assert stats["replaced_regions"] == 0
    assert stats["runtime_executions"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_eager_transfers_route_h2d_and_d2h_once_through_the_adapter(production_backend):
    eager = eager_module()
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    with torch.no_grad(), eager.eager_transfers(backend=production_backend):
        on_gpu = x.to("cuda")
        back = on_gpu.cpu()
        skipped = x.to("cuda", non_blocking=True)
    assert on_gpu.device.type == "cuda" and torch.equal(on_gpu.cpu(), x)
    assert torch.equal(back, x) and back.device.type == "cpu"
    assert torch.equal(skipped.cpu(), x)
    stats = production_backend.stats()
    assert stats["redispatches"]["nonblocking_unavailable"] == 1
    if production_backend.runtime.available:
        assert stats["runtime_executions"] == 2
    else:
        assert stats["runtime_executions"] == 0
        assert stats["fallbacks"]["runtime_unavailable"] == 2
    assert stats["plan_compiles"] == 2


def _attempts(stats):
    return stats["runtime_executions"] + sum(stats["fallbacks"].values())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_eager_scope_around_a_compiled_function_offloads_each_transfer_once(production_backend):
    eager = eager_module()

    def fn(x):
        return x.to("cuda").transpose(0, 1).contiguous()

    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    # Dynamo does not trace while a dispatch mode is active: the first call
    # inside the scope runs eagerly and the eager path offloads the transfer
    # exactly once.
    compiled_fn = torch.compile(fn, backend=production_backend, dynamic=True)
    with torch.no_grad(), eager.eager_transfers(backend=production_backend):
        actual = compiled_fn(x)
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    stats = production_backend.stats()
    assert stats["dynamo_compiles"] == 0
    assert _attempts(stats) == 1
    # Compiled outside the scope, the graph's custom op runs with interception
    # suspended inside it: still exactly one attempt per call, no duplicate.
    torch._dynamo.reset()
    with torch.no_grad():
        compiled_fn(x)
    before = _attempts(production_backend.stats())
    assert production_backend.stats()["replaced_regions"] == 1
    with torch.no_grad(), eager.eager_transfers(backend=production_backend):
        actual = compiled_fn(x)
    torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
    stats = production_backend.stats()
    assert stats["dynamo_compiles"] == 1
    assert _attempts(stats) == before + 1
    assert stats["redispatches"] == {}


def test_parameters_are_eligible_only_under_no_grad(cpu_backend):
    decide = eager_module().decide
    weight = torch.nn.Parameter(torch.ones(3, 4))
    options = dict(device=torch.device("cuda:0"))
    assert decide(torch.ops.aten._to_copy.default, (weight,), options).reason == "requires_grad"
    with torch.no_grad():
        decision = decide(torch.ops.aten._to_copy.default, (weight,), options)
    assert decision.reason is None and decision.direction == "h2d"

    from reloc_torch import import_graph

    # Raw capture with a parameter-like example input: normalization keeps
    # requires_grad on the fake root and the importer decides by the grad mode
    # at import time (the same path as T2's requires_grad root-contract test).
    gm = torch.fx.symbolic_trace(lambda x: x.transpose(0, 1).contiguous().to("cuda"))
    grad_input = torch.ones(3, 4, requires_grad=True)
    assert "requires_grad" in {e.reason for e in import_graph(gm, [grad_input]).exclusions}
    with torch.no_grad():
        assert len(import_graph(gm, [grad_input]).candidates) == 1
        compiled = cpu_backend(gm, [grad_input])
    assert compiled.rewritten is not compiled.original
    assert len(op_nodes(compiled.rewritten)) == 1
    assert cpu_backend.stats()["replaced_regions"] == 1
    with torch.no_grad():
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
        from reloc_torch.runtime import source_reason

        assert source_reason(source) is None
    assert source_reason(source) == "requires_grad"
