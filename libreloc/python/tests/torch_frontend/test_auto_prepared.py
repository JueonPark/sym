"""Compiler artifacts lowered to guarded, bounded prepared execution."""
import pytest
import torch
import pyreloc

from reloc_torch.fast import FastTransfers, TypedProgram
from reloc_torch.resources import TransferResources
from test_fast import dlrm_recipe


@pytest.mark.gpu
def test_borrowed_resources_survive_fast_owner_close(compiler, cuda_device):
    with TransferResources() as resources:
        source = torch.randn(26, 7, 64)
        for _ in range(2):
            with FastTransfers(cuda_device, compiler=compiler, resources=resources) as owner:
                out = owner.typed(dlrm_recipe(), source).execute(source)
                expected = source.transpose(0, 1).contiguous().to(cuda_device, torch.float16)
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
                assert owner.stats()['logical_transfers'] == 1
            assert not resources.closed


@pytest.mark.gpu
def test_borrowed_resource_close_rejects_submission(compiler, cuda_device):
    resources = TransferResources()
    owner = FastTransfers(cuda_device, compiler=compiler, resources=resources)
    try:
        source = torch.randn(26, 7, 64)
        program = owner.typed(dlrm_recipe(), source)
        resources.close()
        with pytest.raises(RuntimeError, match='closed'):
            program.execute(source)
    finally:
        owner.close()
        resources.close()


@pytest.mark.gpu
def test_compiler_template_factory_uses_existing_plan(compiler, cuda_device, monkeypatch):
    source = torch.randn(26, 7, 64)
    compiled = compiler.compile(dlrm_recipe())
    bindings = compiled.bind_values(source)
    bound = pyreloc.bind_typed(compiled.decoded_plan, bindings, {})
    template = pyreloc.prepare_dispatch_template(bound, 'h2d', 'cuda', threads=4)
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        def unexpected(*args, **kwargs):
            pytest.fail('an existing compiler template must not be rebuilt or registered')
        monkeypatch.setattr(compiler, 'compile', unexpected)
        monkeypatch.setattr(pyreloc, 'bind_typed', unexpected)
        monkeypatch.setattr(TypedProgram, '_register_op', unexpected)
        program = TypedProgram._from_template(owner, compiled, source, bindings, bound, template, 0)
        assert program.bound is bound and program.template is template
        assert program.handle is None
        group = owner.prepare_group([program])
        for _ in range(2):
            source.neg_()
            out = group.submit([source]).wait()[0]
            expected = source.transpose(0, 1).contiguous().to(cuda_device, torch.float16)
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert owner.stats()['logical_transfers'] == 2
        with pytest.raises(ValueError):
            program.check(source.half())
        with pytest.raises(ValueError):
            program.check(torch.randn(26, 8, 64))


def policy(**kwargs):
    from reloc_torch import PreparedExecutionPolicy
    return PreparedExecutionPolicy(min_bytes=0, min_reuses=0, **kwargs)


def backend(compiler, **kwargs):
    from reloc_torch import RelocBackend
    return RelocBackend(compiler=compiler, prepared_execution=kwargs.pop('prepared_execution', policy()),
                        transfer_options=kwargs.pop('transfer_options', dict(pinning='pinned', n_streams=1)), **kwargs)


def cast(x):
    return x.transpose(0, 1).contiguous().to('cuda:0', torch.float16)


def captured_entry(owner, fn, *args):
    """Exercise real FX import/compiler lowering; expose its entry for race tests."""
    graph = owner(torch.fx.symbolic_trace(fn), args)
    return graph, owner._registry.lookup(graph.handles[0])


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == expected.device and actual.stride() == expected.stride()
    dtype = torch.int16 if actual.dtype in (torch.float16, torch.bfloat16) else torch.int32
    assert torch.equal(actual.view(dtype), expected.view(dtype))


@pytest.mark.gpu
def test_automatic_capture_revisits_live_values_and_held_outputs(compiler, cuda_device):
    owner = backend(compiler)
    try:
        run = torch.compile(cast, backend=owner, dynamic=True, fullgraph=True)
        held = []
        with torch.no_grad():
            for n in (17, 19, 17, 19, 17):
                x = torch.randn(7, n, 9)
                out = run(x)
                exact(out, cast(x))
                held.append((out, out.clone()))
                x.fill_(100)
        for out, expected in held:
            exact(out, expected)
        assert len({out.data_ptr() for out, _ in held}) == 5
        stats = owner.stats()
        prepared = stats['prepared_execution']
        assert stats['plan_compiles'] == 1
        assert prepared['preparations'] == 2 and prepared['cache_hits'] == 3
        assert prepared['executions'] == stats['runtime_executions'] == stats['typed_executions'] == 5
        assert prepared['native']['prepared_groups']['rebinds'] == 3
        assert prepared['native']['report_materializations'] == 0
        assert 'cuda_relocate_f32' in stats['dispatches']
    finally:
        owner.close()


@pytest.mark.gpu
def test_same_address_parameter_revision_changes_row_legality(compiler, cuda_device):
    import torch.ao.quantization.fx._decomposed
    def dequant(q, s, z):
        return torch.ops.quantized_decomposed.dequantize_per_channel(
            q, s, z, 1, -128, 127, torch.int8).t().contiguous().to('cuda:0', torch.float16)
    owner = backend(compiler)
    q = torch.randint(-100, 100, (13, 11), dtype=torch.int8)
    s, z = torch.linspace(.01, .5, 11), torch.zeros(11, dtype=torch.int32)
    try:
        run, _ = captured_entry(owner, dequant, q, s, z)
        held = []
        with torch.no_grad():
            for zero in (0, 3, 0, 3):
                z.fill_(zero)
                out = run(q, s, z)
                exact(out, dequant(q, s, z))
                held.append((out, out.clone()))
        for out, expected in held:
            exact(out, expected)
        stats = owner.stats()['prepared_execution']
        assert stats['preparations'] == 2 and stats['cache_hits'] == 2
        assert stats['parameter_snapshots'] == 4
        assert len(stats['native']['dispatches']) == 2
        assert stats['retained_parameter_bytes'] == 2 * 11 * 8
    finally:
        owner.close()


@pytest.mark.gpu
def test_live_indices_rebind_every_call_and_reject_range(compiler, cuda_device):
    def select(x, index):
        return x.index_select(0, index).to('cuda:0', torch.float16)
    owner = backend(compiler)
    x, index = torch.randn(19, 7), torch.tensor([1, 5, 3, 1])
    try:
        run, entry = captured_entry(owner, select, x, index)
        for offset in range(3):
            index.add_(offset)
            exact(run(x, index), select(x, index))
        stats = owner.stats()['prepared_execution']
        assert stats['index_bindings'] == 3 and stats['preparations'] == 1
        assert stats['cache_hits'] == 2 and stats['native']['groups'] == 0
        index[0] = 19
        with pytest.raises((RuntimeError, IndexError)):
            run(x, index)
        assert entry.fallback_calls == 1
        assert owner.stats()['prepared_execution']['executions'] == 3
    finally:
        owner.close()


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['small', 'reuse'])
def test_admission_runs_original_once_without_preparation(compiler, cuda_device, mode):
    from reloc_torch import PreparedExecutionPolicy
    configured = PreparedExecutionPolicy(min_bytes=1 << 30, min_reuses=0) if mode == 'small' else PreparedExecutionPolicy(min_bytes=0, min_reuses=1)
    owner = backend(compiler, prepared_execution=configured)
    try:
        x = torch.randn(7, 9)
        run, entry = captured_entry(owner, cast, x)
        exact(run(x), cast(x))
        assert entry.fallback_calls == 1
        assert owner.stats()['prepared_execution']['preparations'] == 0
        exact(run(x), cast(x))
        assert owner.stats()['prepared_execution']['preparations'] == (mode == 'reuse')
    finally:
        owner.close()


@pytest.mark.gpu
def test_bounded_cache_eviction_preserves_preflighted_call(compiler, cuda_device):
    owner = backend(compiler, prepared_execution=policy(max_entries=1))
    try:
        # Symbolic capture supplies one real compiler artifact for both shapes.
        from reloc_torch.compat import symbolic_capture
        from reloc_torch import import_graph
        x = torch.randn(7, 9)
        candidate, = import_graph(symbolic_capture(cast, x)).candidates
        artifact = owner.compile_recipe(candidate.recipe)
        first = owner.runtime.preflight(artifact, x, cuda_device)
        second_source = torch.randn(8, 11)
        second = owner.runtime.preflight(artifact, second_source, cuda_device)
        assert owner.stats()['prepared_execution']['cache_entries'] == 1
        exact(owner.runtime.execute(first), cast(x))
        exact(owner.runtime.execute(second), cast(second_source))
        with pytest.raises(RuntimeError, match='already executed'):
            owner.runtime.execute(first)
        stats = owner.stats()['prepared_execution']
        assert stats['evictions'] == 1 and stats['executions'] == 2
    finally:
        owner.close()


@pytest.mark.gpu
@pytest.mark.parametrize('change', ['source', 'parameter', 'index'])
def test_preflight_rechecks_mutable_operands_before_launch(compiler, cuda_device, change):
    import torch.ao.quantization.fx._decomposed
    owner = backend(compiler)
    try:
        x = torch.randn(7, 9)
        if change == 'source':
            _, entry = captured_entry(owner, cast, x)
            call = owner.runtime.preflight(entry.compiled, x, cuda_device)
            x.transpose_(0, 1)
        elif change == 'index':
            fn = lambda x, i: x.index_select(0, i).to('cuda:0', torch.float16)
            i = torch.tensor([1, 3, 1])
            _, entry = captured_entry(owner, fn, x, i)
            call = owner.runtime.preflight(entry.compiled, x, cuda_device, index=i)
            i[0] = 2
        else:
            fn = lambda q, s, z: torch.ops.quantized_decomposed.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).t().contiguous().to('cuda:0', torch.float16)
            x = torch.randint(-100, 100, (7, 9), dtype=torch.int8)
            s, z = torch.ones(9), torch.zeros(9, dtype=torch.int32)
            _, entry = captured_entry(owner, fn, x, s, z)
            params = dict(zip((p.name for p in entry.compiled.parameters), (s, z)))
            call = owner.runtime.preflight(entry.compiled, x, cuda_device, parameters=params)
            z.add_(1)
        with pytest.raises(RuntimeError, match='stale'):
            owner.runtime.execute(call)
        assert owner.stats()['prepared_execution']['executions'] == 0
    finally:
        owner.close()


@pytest.mark.gpu
@pytest.mark.parametrize('config,reason', [({'transfer_resources': None}, 'per_call_resources'),
    ({'transfer_options': {'n_streams': 3}}, 'transfer_options'), ({'prepared_execution': False}, None)])
def test_explicit_configuration_uses_legacy_path(compiler, cuda_device, config, reason):
    owner = backend(compiler, **config)
    try:
        x = torch.randn(7, 9)
        run, _ = captured_entry(owner, cast, x)
        exact(run(x), cast(x))
        stats = owner.stats()['prepared_execution']
        if reason is None:
            assert stats is None
        else:
            assert stats['legacy_bypasses'] == {reason: 1}
            assert stats['executions'] == 0
    finally:
        owner.close()


@pytest.mark.gpu
def test_auto_owner_borrows_budgets_and_closes_handles(compiler, cuda_device):
    resources = TransferResources(max_typed_retained_bytes=65536, max_typed_live_bytes=65536)
    owner = backend(compiler, transfer_resources=resources)
    try:
        x = torch.randn(7, 9)
        run, _ = captured_entry(owner, cast, x)
        exact(run(x), cast(x))
        assert owner.runtime._resources is resources
        owner.close()
        assert not resources.closed
        assert owner.stats()['prepared_execution']['cache_entries'] == 0
        assert owner.stats()['prepared_execution']['native']['prepared_groups']['live_slots'] == 0
        with pytest.raises(RuntimeError, match='closed'):
            run(x)
    finally:
        owner.close()
        resources.close()


def test_policy_validates_limits_and_custom_runtime(counting_runtime):
    from reloc_torch import PreparedExecutionPolicy, RelocBackend
    for kwargs in ({'max_entries': 0}, {'max_parameter_bytes': -1}, {'min_reuses': -1}, {'min_bytes': True}):
        with pytest.raises((TypeError, ValueError)):
            PreparedExecutionPolicy(**kwargs)
    with pytest.raises(ValueError, match='custom runtime'):
        RelocBackend(runtime=counting_runtime, prepared_execution=True)
    with pytest.raises(TypeError, match='prepared_execution'):
        RelocBackend(prepared_execution='auto')


@pytest.mark.gpu
def test_parameter_budget_and_revisions_are_bounded_without_tensor_retention(compiler, cuda_device):
    import gc
    import weakref
    import torch.ao.quantization.fx._decomposed
    def fn(q, s, z):
        return torch.ops.quantized_decomposed.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).t().contiguous().to('cuda:0', torch.float16)
    owner = backend(compiler, prepared_execution=policy(max_entries=3, max_parameter_bytes=80))
    try:
        q, s, z = torch.ones(7, 5, dtype=torch.int8), torch.ones(5), torch.zeros(5, dtype=torch.int32)
        run, _ = captured_entry(owner, fn, q, s, z)
        for value in range(6):
            s.fill_(value + 1)
            exact(run(q, s, z), fn(q, s, z))
        stats = owner.stats()['prepared_execution']
        assert stats['cache_entries'] == 2 and stats['retained_parameter_bytes'] == 80
        assert stats['evictions'] == 4 and stats['preparations'] == 6
        refs = [weakref.ref(t) for t in (q, s, z)]
        del q, s, z
        gc.collect()
        assert all(ref() is None for ref in refs)
    finally:
        owner.close()
    owner = backend(compiler, prepared_execution=policy(max_parameter_bytes=1))
    try:
        q, s, z = torch.ones(7, 5, dtype=torch.int8), torch.ones(5), torch.zeros(5, dtype=torch.int32)
        run, entry = captured_entry(owner, fn, q, s, z)
        exact(run(q, s, z), fn(q, s, z))
        assert entry.fallback_calls == 1
        stats = owner.stats()['prepared_execution']
        assert stats['cache_entries'] == 0 and stats['exclusions'] == {'parameter_budget': 1}
    finally:
        owner.close()


@pytest.mark.gpu
def test_automatic_layout_concurrency_and_explicit_placement(compiler, cuda_device):
    from concurrent.futures import ThreadPoolExecutor
    from reloc_torch import PlacementPolicy
    fn = lambda x: x.t().contiguous().to('cuda:0')
    owner = backend(compiler)
    try:
        x = torch.randn(13, 17)
        run, _ = captured_entry(owner, fn, x)
        sources = [x + n for n in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(run, sources))
        for out, source in zip(outs, sources):
            exact(out, fn(source))
        assert owner.stats()['prepared_execution']['preparations'] == 1
        assert owner.stats()['prepared_execution']['executions'] == 8
        assert owner.stats()['typed_executions'] == 0
    finally:
        owner.close()
    placement = PlacementPolicy()
    owner = backend(compiler, placement=placement)
    try:
        run, _ = captured_entry(owner, cast, x)
        with placement.force('sym_cpu'):
            exact(run(x), cast(x))
        stats = owner.stats()
        assert stats['dispatches'] == {'cpu_reference': 1}
        assert stats['prepared_execution']['legacy_bypasses'] == {'placement': 1}
    finally:
        owner.close()


@pytest.mark.gpu
@pytest.mark.parametrize('kind', ['offset', 'stride', 'empty', 'dtype'])
def test_automatic_metadata_exclusions_execute_original_once(compiler, cuda_device, kind):
    from reloc_torch.runtime import execute_or_fallback
    owner = backend(compiler)
    try:
        x = torch.randn(7, 9)
        _, entry = captured_entry(owner, cast, x)
        source = {'offset': torch.randn(8, 9)[1:], 'stride': torch.randn(7, 18)[:, ::2],
                  'empty': torch.empty(0, 9), 'dtype': x.double()}[kind]
        exact(execute_or_fallback(entry, source, None, cuda_device), cast(source))
        assert entry.fallback_calls == 1
        assert owner.stats()['prepared_execution']['preparations'] == 0
    finally:
        owner.close()


@pytest.mark.gpu
def test_resource_budget_failure_is_terminal_and_close_invalidates_preflight(compiler, cuda_device):
    from reloc_torch.runtime import ExecutionError
    with TransferResources(max_typed_live_bytes=1) as resources:
        owner = backend(compiler, transfer_resources=resources)
        try:
            x = torch.randn(7, 9)
            run, entry = captured_entry(owner, cast, x)
            with pytest.raises(ExecutionError, match='live-byte limit'):
                run(x)
            assert entry.fallback_calls == 0
            assert owner.stats()['prepared_execution']['executions'] == 0
        finally:
            owner.close()
    owner = backend(compiler)
    x = torch.randn(7, 9)
    _, entry = captured_entry(owner, cast, x)
    call = owner.runtime.preflight(entry.compiled, x, cuda_device)
    owner.close()
    with pytest.raises(RuntimeError, match='closed'):
        owner.runtime.execute(call)
    assert owner.stats()['prepared_execution']['native']['prepared_groups']['live_slots'] == 0


@pytest.mark.gpu
def test_prepared_path_does_not_resolve_a_second_compiler(compiler, cuda_device, monkeypatch):
    from reloc_torch import CompilerClient
    owner = backend(compiler)
    try:
        x = torch.randn(7, 9)
        run, _ = captured_entry(owner, cast, x)
        def unexpected(*args, **kwargs):
            pytest.fail('the adapter already has a compiler artifact')
        monkeypatch.setattr(CompilerClient, 'from_environment', unexpected)
        exact(run(x), cast(x))
    finally:
        owner.close()


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['1', '2', '5', '6', 'fork'])
def test_automatic_native_failure_and_fork_subprocess(mode):
    import os
    import subprocess
    import sys
    from pathlib import Path
    shim = Path(pyreloc.__file__).resolve().parent.parent / 'libtyped_dispatch_faults.so'
    assert shim.exists(), 'build the native dispatch fault harness'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('automatic_fault_scenario.py')), mode],
        env=dict(os.environ, LD_PRELOAD=str(shim), SYM_DISPATCH_FAULT_SHIM=str(shim)),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
def test_no_grad_parameters_and_negative_views_preserve_semantics(compiler, cuda_device):
    owner = backend(compiler)
    try:
        source = torch.nn.Parameter(torch.randn(7, 9))
        with torch.no_grad():
            run, entry = captured_entry(owner, cast, source)
            exact(run(source), cast(source))
            assert owner.stats()['prepared_execution']['executions'] == 1
            neg = torch._neg_view(source.detach())
            exact(run(neg), cast(neg))
            # The Torch dispatcher resolves negative views before custom ops.
            # Direct adapter entry still has to reject the unresolved view.
            from reloc_torch.runtime import execute_or_fallback
            exact(execute_or_fallback(entry, neg, None, cuda_device), cast(neg))
        assert entry.fallback_calls == 1
    finally:
        owner.close()


def test_closed_prepared_statistics_are_independent_snapshots():
    from reloc_torch.automatic import AutomaticPreparedExecution
    manager = AutomaticPreparedExecution(policy())
    manager.close()
    snapshot = manager.stats()
    snapshot['native']['logical_transfers'] = 99
    snapshot['native']['dispatches']['invented'] = 99
    assert manager.stats()['native']['logical_transfers'] == 0
    assert manager.stats()['native']['dispatches'] == {}
