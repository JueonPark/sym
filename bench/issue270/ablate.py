"""Run the same bound artifact and dispatch row with and without region lowering."""
import argparse
import hashlib
import json
from pathlib import Path

import pyreloc
import torch
from libreloc.python.tests import typed_reference as oracle

from .region import Invocation, capture, lower, materialized

CORPUS = Path(__file__).resolve().parents[2] / "libreloc/test/corpus/typed"


def bindings():
    return [(path.stem, i) for path in sorted(CORPUS.glob("*.json"))
            for i in range(len(json.loads(path.read_text())["bindings"]))]


def array(block):
    return oracle.from_bits(block["dtype"], [int(b, 16) for b in block["bits"]], tuple(block["shape"]))


def tensor(block):
    result = torch.from_numpy(array(block).copy())
    return result.view(torch.bfloat16) if block["dtype"] in ("bfloat16", "uint16") else result


def fixture(name, index, device):
    meta = json.loads((CORPUS / f"{name}.json").read_text())
    blob = (CORPUS / f"{name}.bin").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == meta["plan_sha256"]
    binding = meta["bindings"][index]
    params = {key: array(block) for key, block in meta["parameters"].items()}
    bound = pyreloc.bind_typed(pyreloc.load_typed_plan(blob), binding["symbols"],
                               {k: (p.dtype.name, list(p.shape), p.tobytes()) for k, p in params.items()})
    source, expected = tensor(binding["source"]), tensor(binding["expected"])
    prefix = torch.full((1, *expected.shape[1:]), 11, dtype=expected.dtype, device=device)
    return Invocation(bound, prefix.device), source, prefix, expected


def compare(name, index, device):
    call, source, prefix, expected = fixture(name, index, device)
    original = capture(materialized)
    enabled = lower(original)
    reference = torch.cat((prefix.cpu(), expected), dim=0)
    kind = "cuda" if prefix.is_cuda else "host"
    rows = pyreloc.query_capability(call.bound, "h2d", kind)["eligible"]
    meta = json.loads((CORPUS / f"{name}.json").read_text())
    declared = meta["bindings"][index]["capability"][kind]["h2d"]
    assert declared and [r["implementation"] for r in rows] == declared, (name, index, kind)
    for row in rows:
        call.implementation = row["implementation"]
        temporaries = []
        dispatches = []
        for run in (original, enabled):
            out = run(call, source, prefix)
            assert torch.equal(out.cpu().view(torch.uint8), reference.view(torch.uint8)), (name, index, row)
            assert out.device == prefix.device and out.dtype == prefix.dtype and out.is_contiguous()
            assert call.report["executed"] and call.report["implementation"] == call.implementation
            source_copy = source.nbytes if source.data_ptr() != call.source_address else 0
            dest_copy = expected.nbytes if out[prefix.size(0):].data_ptr() != call.destination_address else 0
            temporaries.append(source_copy + dest_copy)
            dispatches.append(tuple(call.report[k] for k in (
                "wire_boundary", "payload_bytes_transferred", "host_kernel")))
        assert dispatches[0] == dispatches[1], (name, index, row)
        boundary, payload, host_kernel = dispatches[0]
        assert temporaries == [source.nbytes + expected.nbytes, 0], (name, index, row)
        yield dict(fixture=name, binding=index, implementation=call.implementation,
                   source_bytes=source.nbytes, destination_bytes=expected.nbytes,
                   wire_boundary=boundary, transfer_bytes=payload, host_kernel=host_kernel,
                   temporary_bytes=temporaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not __debug__:
        parser.error("qualification requires Python assertions enabled")
    pairs = bindings()
    assert pairs, f"no typed fixtures found under {CORPUS}"
    for name, index in pairs:
        for result in compare(name, index, args.device):
            print(json.dumps(result))


if __name__ == "__main__":
    main()
