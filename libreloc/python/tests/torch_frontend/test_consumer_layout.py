"""Private layout copies may disappear only before fresh functional cat."""
from contextlib import ExitStack

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx


def module():
    from reloc_torch import consumer_layout
    return consumer_layout


def capture(fn, *args):
    return make_fx(fn)(*args)


def copies(gm):
    return [n for n in gm.graph.nodes if n.target in (torch.ops.aten.clone.default, torch.ops.aten.contiguous.default)]


@pytest.fixture
def wire():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return torch.randn(7, 4, 8, device='cuda', dtype=torch.float16)


def append(wire, new):
    return torch.cat((wire.transpose(0, 1).contiguous(), new), dim=1)


def test_layout_rule_removes_private_copy_and_preserves_held_values(wire):
    new = torch.randn(4, 2, 8, device='cuda', dtype=wire.dtype)
    gm = capture(append, wire, new)
    original = str(gm.graph)
    lowered, decisions = module().lower_consumer_layout(gm, (wire, new))
    assert len(copies(gm)) == 1 and not copies(lowered)
    assert str(gm.graph) == original
    assert len(decisions) == 1 and decisions[0].selected
    assert decisions[0].eliminated_bytes == wire.numel() * wire.element_size()
    assert decisions[0].read_write_bytes == 2 * decisions[0].eliminated_bytes
    held = lowered(wire, new); snapshot = held.clone()
    for _ in range(3):
        wire.normal_(); new.normal_()
        actual = lowered(wire, new)
        assert actual.is_contiguous() and actual.dtype == wire.dtype
        assert torch.equal(actual.view(torch.int16), gm(wire, new).view(torch.int16))
        assert torch.equal(held.view(torch.int16), snapshot.view(torch.int16))


def test_layout_rule_preserves_nan_payloads_and_signed_zero(wire):
    wire.view(torch.int16).flatten()[:6] = torch.tensor([0, -32768, 31745, 32257, 31744, -1024], device='cuda', dtype=torch.int16)
    new = wire.transpose(0, 1)[:, :1].contiguous()
    gm = capture(append, wire, new)
    lowered, rows = module().lower_consumer_layout(gm, (wire, new))
    assert rows[0].selected
    assert torch.equal(lowered(wire, new).view(torch.int16), gm(wire, new).view(torch.int16))


def test_layout_rule_accepts_packed_positive_offset(wire):
    packed = torch.stack((wire, wire + 1))
    view = packed[1]; assert view.storage_offset() > 0
    new = torch.randn(4, 1, 8, device='cuda', dtype=wire.dtype)
    gm = capture(append, view, new)
    lowered, rows = module().lower_consumer_layout(gm, (view, new))
    assert rows[0].selected
    assert torch.equal(lowered(view, new), gm(view, new))


@pytest.mark.parametrize('kind', ['visible','alias','mixed','mutation','padding','channel','noncontiguous','cpu','cast','out'])
def test_layout_rule_preserves_unsupported_materialization(wire, kind):
    if kind == 'cpu': wire = wire.cpu()
    if kind == 'noncontiguous': wire = torch.randn(7, 4, 16, device='cuda', dtype=wire.dtype)[..., ::2]
    new = torch.zeros(4, 1, 8, device=wire.device, dtype=wire.dtype)
    def fn(x, y):
        if kind == 'padding': x = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 0))
        if kind == 'channel': x = x.index_select(0, torch.tensor([1, 0], device=x.device))
        if kind == 'cast': x = x.float(); y = y.float()
        view = x.transpose(0, 1)
        copied = view.contiguous()
        if kind == 'mutation': x.add_(1)
        if kind == 'out':
            out = torch.empty(4, copied.shape[1] + 1, 8, dtype=x.dtype, device=x.device)
            return torch.cat((copied, y), dim=1, out=out)
        result = torch.cat((copied, y), dim=1)
        if kind == 'visible': return result, copied
        if kind == 'alias': return result, copied.view(-1)
        if kind == 'mixed': return result, copied.sum()
        return result
    gm = capture(fn, wire.clone() if kind == 'mutation' else wire, new)
    lowered, rows = module().lower_consumer_layout(gm, (wire, new))
    assert not any(r.selected for r in rows)
    assert len(copies(lowered)) == len(copies(gm))
    x1, x2 = wire.clone(), wire.clone()
    a, b = gm(x1, new), lowered(x2, new)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_layout_policy_threshold_and_invalid_values(wire):
    mod = module()
    with pytest.raises(ValueError): mod.ConsumerLayoutPolicy(min_saved_bytes=-1)
    with pytest.raises(TypeError): mod.ConsumerLayoutPolicy(min_saved_bytes=True)
    new = wire.transpose(0, 1)[:, :1].contiguous()
    gm = capture(append, wire, new)
    lowered, rows = mod.lower_consumer_layout(gm, (wire, new), policy=mod.ConsumerLayoutPolicy(min_saved_bytes=1 << 20))
    assert not any(r.selected for r in rows) and copies(lowered)
    assert rows[0].eligible and rows[0].reason == 'below_threshold'


def test_backend_is_opt_in_and_preserves_graph_and_training_fallback(wire):
    from reloc_torch import RelocBackend
    new = wire.transpose(0, 1)[:, :1].contiguous()
    gm = capture(append, wire, new)
    original = str(gm.graph)
    with ExitStack() as life:
        off = RelocBackend(); life.callback(off.close)
        on = RelocBackend(consumer_layout=module().ConsumerLayoutPolicy()); life.callback(on.close)
        plain = off(gm, (wire, new)); candidate = on(gm, (wire, new))
        assert copies(plain.rewritten or plain.original)
        assert not copies(candidate.rewritten)
        assert str(gm.graph) == original
        assert torch.equal(candidate(wire, new), plain(wire, new))
        assert on.stats()['consumer_layout']['selected'] == 1
        differentiable = wire.detach().requires_grad_()
        candidate(differentiable, new).sum().backward()
        assert torch.equal(differentiable.grad, torch.ones_like(differentiable))


def test_cpu_only_fake_cuda_rule_and_rank_two_permutation():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty(5, 3, device='cuda', dtype=torch.float16)
        y = torch.empty(3, 2, device='cuda', dtype=torch.float16)
        gm = make_fx(lambda a,b: torch.cat((a.permute(1,0).contiguous(),b),1))(x,y)
    original = str(gm.graph)
    lowered, rows = module().lower_consumer_layout(gm,(x,y))
    assert rows[0].selected and rows[0].eliminated_bytes == 30
    assert not copies(lowered) and str(gm.graph) == original


def test_cpu_only_missing_metadata_preserves_original():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty(5,3,device='cuda');y = torch.empty(3,2,device='cuda')
        gm = make_fx(lambda a,b:torch.cat((a.t().contiguous(),b),1))(x,y)
    for n in gm.graph.nodes:
        if n.target is torch.ops.aten.cat.default: n.meta.clear()
    lowered, rows = module().lower_consumer_layout(gm,(x,y))
    assert lowered is gm and not rows[0].selected


def test_cpu_only_unknown_call_cannot_hide_mutation():
    from torch._subclasses.fake_tensor import FakeTensorMode
    def opaque(): raise AssertionError('compiler must not execute opaque Python')
    with FakeTensorMode():
        x = torch.empty(5,3,device='cuda');y = torch.empty(3,2,device='cuda')
        gm = make_fx(lambda a,b:torch.cat((a.t().contiguous(),b),1))(x,y)
    clone = copies(gm)[0]
    with gm.graph.inserting_after(clone): gm.graph.call_function(opaque,())
    gm.recompile()
    lowered, rows = module().lower_consumer_layout(gm,(x,y))
    assert lowered is gm and rows[0].reason == 'opaque_operator'


def test_dynamo_inductor_rule_matches_fresh_values(wire):
    from reloc_torch import RelocBackend
    new = torch.randn(4,1,8,device=wire.device,dtype=wire.dtype)
    backend = RelocBackend(consumer_layout=True,compute_backend='inductor')
    try:
        compiled = torch.compile(append,backend=backend,fullgraph=True,dynamic=False)
        with torch.no_grad():
            first = compiled(wire,new); snapshot = first.clone()
            wire.normal_(); new.normal_()
            assert torch.equal(compiled(wire,new), append(wire,new))
            assert torch.equal(first,snapshot)
        assert backend.stats()['consumer_layout']['selected'] == 1
        backend.close()
        with torch.no_grad(), pytest.raises(RuntimeError,match='closed'):
            compiled(wire,new)
    finally: backend.close()


def test_dynamo_attention_consumer_sees_identical_cache(wire):
    from reloc_torch import RelocBackend
    new = torch.randn(4,1,8,device=wire.device,dtype=wire.dtype)
    q = torch.randn(4,1,8,device=wire.device,dtype=wire.dtype)
    def consume(x,y,q):
        cache = torch.cat((x.transpose(0,1).contiguous(),y),1)
        mask = torch.ones(1,cache.shape[1],device=x.device,dtype=torch.bool).tril(cache.shape[1]-1)
        scores = (q @ cache.transpose(1,2)).masked_fill(~mask, -float("inf"))
        return scores.softmax(-1), cache
    backend = RelocBackend(consumer_layout=True)
    try:
        compiled = torch.compile(consume,backend=backend,fullgraph=True,dynamic=False)
        with torch.no_grad():
            result = compiled(wire,new,q)
            expected = consume(wire,new,q)
        assert backend.stats()['consumer_layout']['selected'] == 1
        for a,b in zip(result,expected): assert torch.equal(a,b)
    finally: backend.close()


@pytest.mark.parametrize('kind',['out','method','getitem','getattr'])
def test_high_level_calls_cannot_hide_source_mutation(wire,kind):
    import operator
    new=torch.zeros(4,1,8,device=wire.device,dtype=wire.dtype)
    gm=capture(append,wire,new)
    source=next(n for n in gm.graph.nodes if n.op=='placeholder')
    clone=copies(gm)[0]
    class Opaque:
        def mutate(self):wire.zero_();return 0
        def clone(self):return self.mutate()
        def __getitem__(self,key):return self.mutate()
        @property
        def value(self):return self.mutate()
    if kind!='out':
        with gm.graph.inserting_before(source):opaque=gm.graph.placeholder('opaque')
    with gm.graph.inserting_after(clone):
        if kind=='out':gm.graph.call_function(torch.zeros,(tuple(wire.shape),),{'out':source})
        elif kind=='method':gm.graph.call_method('clone',(opaque,))
        elif kind=='getitem':gm.graph.call_function(operator.getitem,(opaque,0))
        else:gm.graph.call_function(getattr,(opaque,'value'))
    gm.recompile()
    lowered,rows=module().lower_consumer_layout(gm)
    assert lowered is gm and not any(r.selected for r in rows)
    wire.fill_(1)
    result=lowered(wire,new) if kind=='out' else lowered(Opaque(),wire,new)
    assert torch.equal(result[:,:7],torch.ones_like(result[:,:7]))
    assert torch.count_nonzero(wire)==0


def test_eager_layout_only_graph_rejects_execution_after_close(wire):
    from reloc_torch import RelocBackend
    new=torch.zeros(4,1,8,device=wire.device,dtype=wire.dtype)
    backend=RelocBackend(consumer_layout=True)
    compiled=backend(capture(append,wire,new),(wire,new))
    assert backend.stats()['consumer_layout']['selected']==1
    assert torch.equal(compiled(wire,new),append(wire,new))
    backend.close()
    with pytest.raises(RuntimeError,match='closed'):compiled(wire,new)


def test_unrelated_tensor_subclass_operand_cannot_hide_mutation(wire):
    new=torch.zeros(4,1,8,device=wire.device,dtype=wire.dtype)
    gm=capture(append,wire,new);source=next(n for n in gm.graph.nodes if n.op=='placeholder')
    class OpaqueTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls,func,types,args=(),kwargs=None):
            if func is torch.clone:wire.zero_()
            return super().__torch_function__(func,types,args,kwargs or {})
    value=torch.ones(1,device=wire.device).as_subclass(OpaqueTensor)
    with gm.graph.inserting_before(source):opaque=gm.graph.placeholder('opaque')
    opaque.meta['val']=value
    with gm.graph.inserting_after(copies(gm)[0]):gm.graph.call_function(torch.clone,(opaque,))
    gm.recompile()
    lowered,rows=module().lower_consumer_layout(gm)
    assert lowered is gm and not any(r.selected for r in rows)
    wire.fill_(1);result=lowered(value,wire,new)
    assert torch.equal(result[:,:7],torch.ones_like(result[:,:7]))
    assert torch.count_nonzero(wire)==0


@pytest.mark.parametrize('method',['contiguous','clone'])
def test_noop_contiguous_does_not_claim_saved_allocation(method):
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x=torch.empty(7,4,8,device='cuda',dtype=torch.float16)
        y=torch.empty(7,1,8,device='cuda',dtype=torch.float16)
        graph=torch.fx.Graph();a=graph.placeholder('a');b=graph.placeholder('b')
        a.meta['val']=x;b.meta['val']=y
        first=graph.call_method('transpose',(a,0,1));first.meta['val']=x.transpose(0,1)
        second=graph.call_method('transpose',(first,0,1));second.meta['val']=x
        material=graph.call_method(method,(second,)) if method=='contiguous' else graph.call_function(torch.clone,(second,))
        material.meta['val']=x.clone()
        cat=graph.call_function(torch.cat,((material,b),1));cat.meta['val']=torch.cat((x,y),1)
        graph.output(cat);gm=torch.fx.GraphModule({},graph)
    lowered,rows=module().lower_consumer_layout(gm,policy=module().ConsumerLayoutPolicy(min_saved_bytes=1))
    if method=='contiguous':
        assert lowered is gm and not rows[0].selected
        assert rows[0].eliminated_bytes==rows[0].read_write_bytes==0
    else:
        assert rows[0].selected and rows[0].eliminated_bytes==x.numel()*x.element_size()
