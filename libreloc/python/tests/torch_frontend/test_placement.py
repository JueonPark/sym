"""Qualified choices, profile compatibility and real stream/precision semantics."""
from types import SimpleNamespace

import pytest
import torch
import pyreloc

from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend
from reloc_torch import placement
from reloc_torch.recipe import Cast, Recipe, Reshape, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides

CONTEXT = dict(cpu_load='idle', gpu_load='idle', overlap='none')


def recipe(typed=False, direction='h2d'):
    a, b = Symbol('s0'), Symbol('s1')
    source = TensorSpec((a,b), dense_strides((a,b)), Const(0), 'float32')
    destination = TensorSpec((b,a), dense_strides((b,a)), Const(0), 'float16' if typed else 'float32')
    ops = (Transpose((1,0)),) + ((Cast('float16','ieee_rne'),) if typed else ())
    return Recipe(source, ops, destination, direction)


class Runtime:
    epoch = 0
    def _require_open(self):
        pass
    def placement_configuration(self):
        return {}, self.epoch, True


def observations(r, path, costs=(1.,2.), context=CONTEXT):
    return [dict(family=placement.family(r), shape=[n,8], strides=[8,1], path=path,
                 wire_bytes=placement.wire_bytes(r,(n,8),path), resource_state=state,
                 completed_ms=cost, context=context)
            for state in ('cold','warm') for n,cost in zip((8,32),costs)]


def test_interpolation_compatibility_and_conservative_misses(monkeypatch, tmp_path):
    monkeypatch.setattr(placement, 'hardware', lambda d: {'gpu':'test'})
    r = recipe()
    rows = observations(r,'native',(2.,3.)) + observations(r,'torch_gpu',(.2,.3))
    profile = PlacementProfile(hardware={'gpu':'test'},execution={},observations=rows)
    path = tmp_path/'profile.json'
    profile.save(path)
    policy = PlacementPolicy(PlacementProfile.load(path),context=CONTEXT)
    runtime = Runtime()
    def choose(n):
        return policy.choose(SimpleNamespace(recipe=r),torch.zeros(n,8),torch.device('cuda:0'),runtime)
    decision = choose(16)
    assert decision[0] == 'torch_gpu'
    assert .2 < decision[2]['predicted_ms']['torch_gpu'] < .3
    policy.succeeded(decision,runtime)
    assert choose(16)[2]['resource_state'] == 'warm'
    runtime.epoch += 1
    assert choose(16)[2]['resource_state'] == 'cold'
    assert choose(64)[0] == 'native'
    assert policy.stats()['last']['reason'] == 'profile_missing_native_coverage'
    monkeypatch.setattr(placement,'hardware',lambda d:{'gpu':'other'})
    assert choose(16)[2]['reason'] == 'profile_hardware_mismatch'


def test_context_options_state_and_wire_do_not_cross_profiles(monkeypatch):
    monkeypatch.setattr(placement,'hardware',lambda d:{})
    r, runtime = recipe(True), Runtime()
    rows = observations(r,'torch_cpu',(2.,3.)) + observations(r,'torch_gpu',(.2,.3))
    context = dict(CONTEXT)
    policy = PlacementPolicy(PlacementProfile(hardware={},execution={},observations=rows),context=lambda:context)
    choose = lambda:policy.choose(SimpleNamespace(recipe=r),torch.zeros(16,8),torch.device('cuda:0'),runtime)
    assert choose()[0] == 'torch_gpu'
    context['gpu_load'] = 'busy'
    assert choose()[0] == 'torch_cpu'
    assert choose()[2]['reason'] == 'profile_missing_native_coverage'
    context['gpu_load'] = 'idle'
    runtime.placement_configuration = lambda:({'pinning':'pageable'},0,True)
    assert choose()[2]['reason'] == 'profile_execution_mismatch'
    # A profile for half-width wire must not rank an FP32-upload GPU path.
    bad = observations(r,'torch_gpu')
    for row in bad:
        row['wire_bytes'] //= 2
    profile = PlacementProfile(hardware={},execution={},observations=bad)
    assert profile.predict(placement.family(r),(16,8),'torch_gpu','cold',CONTEXT,512) is None


def test_validation_forced_scope_and_bounded_warm_tracking():
    with pytest.raises(ValueError,match='format'):
        PlacementProfile.from_dict({'format':'pinning/1'})
    r, runtime = recipe(), Runtime()
    rows = observations(r,'native')
    rows[0]['completed_ms'] = float('nan')
    with pytest.raises(ValueError,match='invalid'):
        PlacementProfile(hardware={},execution={},observations=rows)
    policy = PlacementPolicy(capacity=2)
    for n in (8,16,32):
        decision = policy.choose(SimpleNamespace(recipe=r),torch.zeros(n,8),torch.device('cuda:0'),runtime)
        assert decision[0] == 'native' and decision[2]['reason'] == 'no_profile'
        policy.succeeded(decision,runtime)
    assert policy.stats()['cached_paths'] == 2
    with pytest.raises(ValueError,match='not qualified'), policy.force('native'):
        policy.choose(SimpleNamespace(recipe=recipe(True)),torch.zeros(8,8),torch.device('cuda:0'),runtime)


def test_discovery_failure_is_conservative(monkeypatch):
    def unavailable(device):
        raise OSError('driver inventory unavailable')
    monkeypatch.setattr(placement,'hardware',unavailable)
    r = recipe()
    policy = PlacementPolicy(PlacementProfile(hardware={},execution={},
        observations=observations(r,'native')),context=CONTEXT)
    decision = policy.choose(SimpleNamespace(recipe=r),torch.zeros(16,8),torch.device('cuda:0'),Runtime())
    assert decision[0] == 'native'
    assert decision[2]['reason'] == 'profile_hardware_unavailable'


def test_unqualified_recipe_keeps_existing_runtime():
    r = recipe()
    r = Recipe(r.source, (Reshape(r.source.shape),)+r.operations,r.destination,r.direction)
    policy = PlacementPolicy()
    args = (SimpleNamespace(recipe=r),torch.zeros(16,8),torch.device('cuda:0'),Runtime())
    assert policy.choose(*args) is None
    assert policy.stats()['choices'] == {'legacy:unsupported_recipe':1}
    with policy.force('torch_gpu'),pytest.raises(ValueError,match='not qualified'):
        policy.choose(*args)


def test_d2h_warm_history_is_specific_to_source_device():
    policy, runtime = PlacementPolicy(), Runtime()
    spec = SimpleNamespace(recipe=recipe(direction='d2h'))
    def choose(index):
        source = SimpleNamespace(shape=(16,8),stride=lambda:(8,1),device=torch.device('cuda',index))
        return policy.choose(spec,source,torch.device('cpu'),runtime)
    policy.succeeded(choose(0),runtime)
    assert choose(0)[2]['resource_state'] == 'warm'
    assert choose(1)[2]['resource_state'] == 'cold'


@pytest.mark.gpu
def test_selected_execution_failure_never_retries(cuda_device,monkeypatch):
    from reloc_torch.runtime import ExecutionError
    policy = PlacementPolicy()
    backend = RelocBackend(placement=policy)
    compiled = torch.compile(lambda x:x.t().contiguous().to(cuda_device),backend=backend,fullgraph=True)
    attempts=[]
    def fail(*args):
        attempts.append(args[-1])
        raise RuntimeError('injected launch failure')
    try:
        with torch.no_grad():
            compiled(torch.ones(8,16))
            monkeypatch.setattr(placement,'replay',fail)
            with policy.force('torch_gpu'),pytest.raises(ExecutionError,match='injected launch failure'):
                compiled(torch.ones(8,16))
        assert attempts == ['torch_gpu']
        assert not backend.stats()['fallbacks']
        assert backend.stats()['runtime_executions'] == 0
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize('direction',['h2d','d2h'])
@pytest.mark.parametrize('typed',[False,True])
def test_forced_paths_current_stream_shape_and_rounding(cuda_device,direction,typed):
    if not pyreloc.cuda_enabled:
        pytest.skip('CUDA runtime unavailable')
    def fn(x):
        value = x.t().contiguous()
        if typed:
            value = value.to(dtype=torch.float16)
        return value.to(cuda_device if direction == 'h2d' else 'cpu')
    policy = PlacementPolicy(context=CONTEXT)
    backend = RelocBackend(placement=policy)
    compiled = torch.compile(fn,backend=backend,fullgraph=True,dynamic=True)
    stream = torch.cuda.Stream(device=cuda_device)
    paths = ['torch_cpu','torch_gpu','sym_cpu'] + (['sym_gpu'] if typed else ['native'])
    try:
        with torch.no_grad(),torch.cuda.stream(stream):
            for n in (11,19):
                values = torch.randn(n,8)
                values[0,:] = torch.tensor([0.,-0.,2**-24,2**-25,65504.,65520.,float('inf'),float('nan')])
                x = values.to(cuda_device) if direction == 'd2h' else values
                expected = fn(x)
                saved = []
                for path in paths:
                    with policy.force(path):
                        out = compiled(x)
                    assert policy.stats()['last']['path'] == path
                    assert out.shape == expected.shape and out.stride() == expected.stride()
                    a,b = out.cpu(),expected.cpu()
                    assert torch.equal(torch.isnan(a),torch.isnan(b))
                    good = ~torch.isnan(b)
                    dtype = torch.int16 if typed else torch.int32
                    assert torch.equal(a.view(dtype)[good],b.view(dtype)[good])
                    saved.append((out,out.clone()))
                for out,copy in saved:
                    torch.testing.assert_close(out,copy,rtol=0,atol=0,equal_nan=True)
            stream.synchronize()
        assert not backend.stats()['fallbacks']
        assert backend.stats()['plan_compiles'] == 1
    finally:
        backend.close()


@pytest.mark.gpu
def test_automatic_profile_uses_live_values_and_composes_with_inductor(cuda_device,monkeypatch):
    if not pyreloc.cuda_enabled:
        pytest.skip('CUDA runtime unavailable')
    policy = PlacementPolicy(context=CONTEXT)
    backend = RelocBackend(placement=policy,compute_backend='inductor')
    fn = lambda x: (x.t().contiguous().to(cuda_device) + 1).sin()
    compiled = torch.compile(fn,backend=backend,fullgraph=True,dynamic=True)
    try:
        with torch.no_grad():
            x = torch.randn(16,8)
            torch.testing.assert_close(compiled(x),fn(x))
            assert policy.stats()['last']['path'] == 'native'
            monkeypatch.setattr(placement,'hardware',lambda d:{})
            execution,_,_ = backend.runtime.placement_configuration()
            policy.profile = PlacementProfile(hardware={},execution=execution,
                observations=observations(recipe(),'native',(2.,3.)) +
                             observations(recipe(),'torch_gpu',(.2,.3)))
            x.numpy()[:] *= 2
            torch.testing.assert_close(compiled(x),fn(x))
            assert policy.stats()['last']['path'] == 'torch_gpu'
        assert backend.stats()['inductor_compiles'] == 1
        assert backend.stats()['runtime_executions'] == 0
        assert not backend.stats()['fallbacks']
        backend.close()
        with torch.no_grad(),pytest.raises(RuntimeError,match='closed'):
            compiled(x)
    finally:
        backend.close()
