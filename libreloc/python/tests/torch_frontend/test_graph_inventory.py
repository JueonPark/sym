import dataclasses
import json

import torch


def test_symbolic_layout_inventory():
    from reloc_torch import graph_inventory
    from torch.fx.experimental.proxy_tensor import make_fx
    def recipe(x):
        return x.reshape(x.shape[0] // 4, 4).transpose(0, 1).contiguous()
    gm = make_fx(recipe, tracing_mode="symbolic")(torch.ones(16))
    before = str(gm.graph)
    records = graph_inventory(gm)
    assert {"aten.view.default", "aten.transpose.int", "aten.clone.default", "aten.sym_size.int"} <= {r.target for r in records}
    assert str(gm.graph) == before
    assert any(isinstance(d, str) for r in records if r.tensor_metadata for d in r.tensor_metadata.shape)
    assert all(not r.tensor_metadata.is_subclass for r in records if r.tensor_metadata)
    json.dumps([dataclasses.asdict(r) for r in records])


def test_raw_dynamo_inventory():
    from reloc_torch import graph_inventory, classify
    captured = []
    def backend(gm, inputs):
        captured.append(gm)
        return gm.forward
    def recipe(x):
        y = x.reshape(4, 4).transpose(0, 1).contiguous()
        z = y.to(dtype=torch.float16)
        y.add_(1)
        return torch.nn.functional.pad(z, (1, 1)), y, y + 2
    x = torch.ones(16)
    expected = recipe(x.clone())
    actual = torch.compile(recipe, backend=backend, fullgraph=True)(x.clone())
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    gm = captured[0]
    before = str(gm.graph)
    records = graph_inventory(gm)
    assert {"reshape", "transpose", "contiguous", "to", "add_", "torch._C._nn.pad"} <= {r.target for r in records}
    assert any(len(r.user_nodes) > 1 for r in records)
    assert any(classify(r).reason == "mutation" for r in records)
    assert any(classify(r).reason == "typed_transform_unavailable" for r in records)
    assert str(gm.graph) == before
    json.dumps([dataclasses.asdict(r) for r in records])


def test_missing_metadata():
    from reloc_torch import graph_inventory, classify
    gm = torch.fx.symbolic_trace(lambda x: x.to("cpu"))
    records = graph_inventory(gm)
    assert all(r.metadata_reason == "metadata_unavailable" for r in records)
    assert all(classify(r).reason == "metadata_unavailable" for r in records)


import pytest


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_raw_boundaries():
    from reloc_torch import graph_inventory, classify
    graphs = []
    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward
    def recipe(x):
        a = x.to("cuda")
        b = x.to("cuda", torch.float32, True)
        c = x.cuda()
        return a.cpu(), b.cpu(), c.cpu()
    result = torch.compile(recipe, backend=backend, fullgraph=True)(torch.ones(16))
    assert all(torch.equal(x, torch.ones(16)) for x in result)
    records = graph_inventory(graphs[0])
    decisions = [classify(r) for r in records]
    assert any(d.candidate for d in decisions)
    assert any(d.reason == "nonblocking_unavailable" for d in decisions)
    assert {"to", "cpu", "cuda"} <= {r.target for r in records}


def test_aten_pad_cast_mutation_and_aliases():
    from reloc_torch import graph_inventory
    from torch.fx.experimental.proxy_tensor import make_fx
    def recipe(x):
        y = x.reshape(4, 4).transpose(0, 1).contiguous()
        z = y.to(torch.float16)
        y.add_(1)
        return torch.nn.functional.pad(z, (1, 1)), y, y + 2
    gm = make_fx(recipe)(torch.ones(16))
    records = graph_inventory(gm)
    assert {"aten.constant_pad_nd.default", "aten._to_copy.default", "aten.add_.Tensor"} <= {r.target for r in records}
    assert next(r for r in records if r.target == "aten.view.default").transfer.aliases_source
    assert not next(r for r in records if r.target == "aten.clone.default").transfer.aliases_source


def test_copy_source_alias_is_source_not_destination():
    from reloc_torch import graph_inventory
    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(lambda dst, src: dst.copy_(src))(torch.zeros(4), torch.ones(4))
    record = next(r for r in graph_inventory(gm) if r.target == "aten.copy_.default")
    assert record.transfer.mutates
    assert not record.transfer.aliases_source
