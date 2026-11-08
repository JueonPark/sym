"""Numerical acceptance gate: direct differential checks, no performance assertions."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import pytest
import torch
import pyreloc
from reloc_torch import CompilerClient
from cases import make_case,FAMILIES

def bind(client,recipe,case):
    artifact=client.compile(recipe)
    params={'scale':('float32',[case.m],case.scales.numpy().tobytes())} if recipe.parameter_bindings else {}
    return (pyreloc.bind_typed(artifact.decoded_plan,{},params) if artifact.typed else
            pyreloc.bind(artifact.decoded_plan,{}))

def equal_bits(got,want):
    assert got.shape == want.shape and got.dtype == want.dtype
    width=torch.int16 if got.dtype==torch.float16 else torch.int32
    mask=torch.isnan(want)
    assert torch.equal(torch.isnan(got),mask)
    assert torch.equal(got.view(width)[~mask],want.view(width)[~mask])

@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('threads',[1,8])
@pytest.mark.parametrize('n,m',[(33,65),(1025,129)])
def test_folded_materialized_and_manual_preserve_ordered_values(family,threads,n,m):
    assert importlib.util.find_spec('_folding_native') is not None, 'native runner has not been built'
    from _folding_native import Runner
    case=make_case(family,n,m)
    if case.source.dtype==torch.float32:
        case.source.flatten()[:12]=torch.tensor([-0.,0.,float('inf'),-float('inf'),float('nan'),
            2**-24,2**-25,65519.,65520.,1.00048828125,-1.00048828125,1.0007])
    client=CompilerClient.from_environment()
    runners=[Runner([bind(client,p,case) for p in case.parts],threads),
             Runner([bind(client,case.recipe,case)],threads),
             Runner([bind(CompilerClient.from_environment(canonicalize=False),case.recipe,case)],threads),
             Runner([],threads)]
    buffers=[case.source]+[torch.empty(s,dtype=getattr(torch,d)) for s,d in zip(case.shapes[1:],case.dtypes[1:])]
    out=torch.empty_like(buffers[-1])
    for iteration in range(2):
        expected=case.reference(case.source)
        for runner in runners[:-1]:
            selected=buffers if len(runner.kernels())>1 else [case.source,out]
            runner.run([b.numpy() for b in selected]); equal_bits(selected[-1],expected)
        runners[-1].manual(family,case.source.numpy(),out.numpy(),case.scales.numpy(),case.n,case.m)
        equal_bits(out,expected)
        case.source.neg_()
    if family.startswith('roundtrip'):
        assert not torch.equal(case.source,expected), 'control must not turn lossy roundtrip into memcpy'

def test_native_rejects_wrong_capacity():
    assert importlib.util.find_spec('_folding_native') is not None, 'native runner has not been built'
    from _folding_native import Runner
    case=make_case('roundtrip1',3,5)
    runner=Runner([bind(CompilerClient.from_environment(),case.recipe,case)],1)
    with pytest.raises(ValueError,match='capacity'):
        runner.run([case.source.numpy(),torch.empty(1).numpy()])


@pytest.mark.parametrize('family,outer', [('roundtrip7',8),('rank3_chain',5),
    ('bf16_roundtrip',8),('dequant_roundtrip',8),('quant_roundtrip',8)])
def test_heldout_chains_match_independent_eager_expression(family,outer):
    from _folding_native import Runner
    case=make_case(family,17,13,outer=outer)
    out=torch.empty(case.shapes[-1],dtype=getattr(torch,case.dtypes[-1]))
    runner=Runner([bind(CompilerClient.from_environment(),case.recipe,case)],1)
    runner.run([case.source.numpy(),out.numpy()])
    equal_bits(out,case.reference(case.source))

@pytest.mark.parametrize('family',FAMILIES+['roundtrip7','bf16_roundtrip','dequant_roundtrip','quant_roundtrip'])
def test_inductor_preserves_exceptional_values(family):
    torch._dynamo.reset()
    torch._inductor.config.emulate_precision_casts=True
    case=make_case(family,33,65)
    if case.source.dtype==torch.float32:
        case.source.flatten()[:10]=torch.tensor([-0.,0.,float('inf'),-float('inf'),float('nan'),
                                                2**-24,2**-25,65520.,1.00048828125,-1.00048828125])
    else:
        case.scales[:5]=torch.tensor([2**-24,1.00048828125,65520.,1e-30,1e30])
    expected=case.reference(case.source)
    destination=torch.empty_like(expected)
    def into(x,out):return out.copy_(case.reference(x))
    compiled=torch.compile(into,fullgraph=True,dynamic=False)
    compiled(case.source,destination)
    equal_bits(destination,expected)
