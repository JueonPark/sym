"""C4 (issue #144): the typed conformance corpus on a real GPU.

Every committed fixture (libreloc/test/corpus/typed) runs through
``reloc_torch.dispatch`` on actual hardware in both directions, once per
row the runtime qualified for a CUDA device end, and must reproduce the
stored oracle bytes exactly. A skip here never certifies a GPU row: the
evidence file docs/typed-evidence/c4-cuda-run.txt records the real run.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

import pyreloc

CORPUS = pathlib.Path(__file__).resolve().parents[3] / "test" / "corpus" / "typed"
FIXTURES = sorted(p.stem for p in CORPUS.glob("*.json"))
_BITS = {"float32": np.uint32, "float16": np.uint16, "int8": np.uint8, "int32": np.uint32}
_DTYPES = {"float32": np.float32, "float16": np.float16, "int8": np.int8, "int32": np.int32}


def array(block):
    raw = np.asarray([int(b, 16) for b in block["bits"]], dtype=np.uint64).astype(_BITS[block["dtype"]])
    return raw.view(_DTYPES[block["dtype"]]).reshape(tuple(block["shape"]))


def rows(bound, direction):
    return [r["implementation"] for r in pyreloc.query_capability(bound, direction, "cuda")["eligible"]]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", FIXTURES)
def test_every_qualified_cuda_row_reproduces_the_fixture(name):
    from reloc_torch import dispatch
    from reloc_torch.artifact import CompiledRecipe

    meta = json.loads((CORPUS / f"{name}.json").read_text())
    compiled = CompiledRecipe.load(CORPUS / f"{name}.reloc")  # the portable artifact, reloaded
    assert compiled.typed and compiled.plan_bytes == (CORPUS / f"{name}.bin").read_bytes()
    parameters = {n: torch.from_numpy(array(b).copy()) for n, b in meta["parameters"].items()}
    executed = 0
    for binding in meta["bindings"]:
        source = torch.from_numpy(array(binding["source"]).copy())
        expected = array(binding["expected"])
        bound = pyreloc.bind_typed(
            pyreloc.load_typed_plan(compiled.plan_bytes),
            {n: binding["symbols"][n] for n in compiled.symbols},
            {n: (v.numpy().dtype.name, list(v.shape), v.numpy().tobytes()) for n, v in parameters.items()},
        )
        # Host to device: every row, explicitly selected.
        for row in rows(bound, "h2d"):
            request = dispatch.prepare_typed_transfer(compiled, source, "cuda", parameters=parameters, implementation=row)
            result = dispatch.execute_typed_transfer(request)
            assert result.tensor.device.type == "cuda"
            assert result.tensor.cpu().numpy().tobytes() == expected.tobytes(), (name, row, "h2d")
            assert result.report["implementation"] == row and result.report["executed"]
            executed += 1
        # Device to host: the same program from a device source, forward only.
        d2h = compiler_direction(compiled, "d2h")
        for row in rows(bound, "d2h"):
            request = dispatch.prepare_typed_transfer(d2h, source.cuda(), "cpu", parameters=parameters, implementation=row)
            result = dispatch.execute_typed_transfer(request)
            assert result.tensor.device.type == "cpu"
            assert result.tensor.numpy().tobytes() == expected.tobytes(), (name, row, "d2h")
            executed += 1
    assert executed >= 2 * len(meta["bindings"])


def compiler_direction(compiled, direction):
    """The same artifact declared for the other direction: the plan bytes
    are direction-free, only the recipe's direction field differs."""
    import dataclasses

    from reloc_torch.artifact import CompiledRecipe

    payload = json.loads(compiled.to_bytes())
    payload["recipe"]["direction"] = direction
    return CompiledRecipe.from_bytes(json.dumps(payload).encode())
