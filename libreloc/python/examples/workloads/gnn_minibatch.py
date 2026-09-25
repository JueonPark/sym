#!/usr/bin/env python3
"""GraphSAGE mini-batch inference with a CPU feature store (workload example).

Node features live in CPU memory. Each mini-batch samples a two-hop
neighbourhood (uniformly, with replacement) and gathers the feature rows it
needs on the CPU with PyTorch (index-based work that Sym cannot express),
then moves the gathered ``[n, 128]`` float32 block to the GPU as float16.
The Sym pass uses one plan compiled by ``RelocBackend`` whose row count
``n`` changes with every mini-batch; the reference pass runs the same
function eagerly. Two SAGE-mean layers run on the GPU with a fixed fanout,
so aggregation is a gather plus a mean (no atomics) and both passes must
produce identical logits.

    python gnn_minibatch.py [--device cuda:0] [--quick] [--output report.json]

See README.md in this directory for the environment and the report fields.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import common

FEATURES = 128
HIDDEN = 128
CLASSES = 16
SIZES = {
    "default": {"nodes": 500_000, "avg_degree": 15, "seeds": 1024, "fanouts": [10, 5], "batches": 5},
    "quick": {"nodes": 20_000, "avg_degree": 10, "seeds": 256, "fanouts": [5, 3], "batches": 3},
}


def random_graph(nodes, avg_degree, generator):
    """CSR adjacency: node v has randint(1, 2 * avg_degree) uniform random neighbours."""
    degrees = torch.randint(1, 2 * avg_degree, (nodes,), generator=generator)
    indptr = torch.zeros(nodes + 1, dtype=torch.long)
    indptr[1:] = degrees.cumsum(0)
    return indptr, torch.randint(0, nodes, (int(indptr[-1]),), generator=generator)


def sample(graph, nodes, fanout, generator):
    """``fanout`` neighbours of every node, uniformly with replacement -> [len(nodes), fanout]."""
    indptr, indices = graph
    start = indptr[nodes]
    degree = indptr[nodes + 1] - start
    picks = (torch.rand(len(nodes), fanout, generator=generator) * degree.unsqueeze(1)).long()
    return indices[start.unsqueeze(1) + picks]


def minibatch(graph, seeds, fanouts, generator):
    """Two-hop sample around sorted unique ``seeds`` (CPU, PyTorch).

    ``nodes`` are the sorted global ids whose features the batch needs. Layer 1
    reads rows of ``nodes`` for the one-hop set ``frontier`` and its sampled
    neighbours; layer 2 reads rows of the layer-1 output for the seeds and theirs."""
    hop0 = sample(graph, seeds, fanouts[0], generator)
    frontier = torch.unique(torch.cat([seeds, hop0.flatten()]))
    hop1 = sample(graph, frontier, fanouts[1], generator)
    nodes = torch.unique(torch.cat([frontier, hop1.flatten()]))
    return {
        "nodes": nodes,
        "layer1": (torch.searchsorted(nodes, frontier), torch.searchsorted(nodes, hop1)),
        "layer2": (torch.searchsorted(frontier, seeds), torch.searchsorted(frontier, hop0)),
    }


class GraphSAGE:
    """Two SAGE-mean layers with a fixed fanout: aggregation is a gather plus a mean (no atomics)."""

    def __init__(self, generator, device):
        def weight(fan_in, fan_out):
            return (torch.randn(fan_out, fan_in, generator=generator) / fan_in ** 0.5).to(device)

        self.layer1 = (weight(FEATURES, HIDDEN), weight(FEATURES, HIDDEN))
        self.layer2 = (weight(HIDDEN, CLASSES), weight(HIDDEN, CLASSES))

    def forward(self, features, layer1, layer2):
        """features [n, 128] float16 on the GPU and the batch's local indices -> seed logits [seeds, 16]."""
        x = features.float()
        own, neighbours = layer1
        h = torch.relu(F.linear(x[own], self.layer1[0]) + F.linear(x[neighbours].mean(dim=1), self.layer1[1]))
        own, neighbours = layer2
        return F.linear(h[own], self.layer2[0]) + F.linear(h[neighbours].mean(dim=1), self.layer2[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="a small graph and few batches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args()
    common.require_environment([args.device])
    common.seed_everything(args.seed)
    from reloc_torch import RelocBackend

    device = torch.device(args.device)
    sizes = SIZES["quick" if args.quick else "default"]
    report = common.Report("gnn_minibatch", devices=[device], features=FEATURES, **sizes)
    clock = common.Clock([device])
    generator = torch.Generator().manual_seed(args.seed)
    graph = random_graph(sizes["nodes"], sizes["avg_degree"], generator)
    features = torch.randn(sizes["nodes"], FEATURES, generator=generator)
    model = GraphSAGE(generator, device)
    blocks = [minibatch(graph, torch.randperm(sizes["nodes"], generator=generator)[:sizes["seeds"]].sort().values,
                        sizes["fanouts"], generator)
              for _ in range(sizes["batches"])]

    def to_device(gathered):
        return gathered.to(device, torch.float16)

    backend = RelocBackend()
    try:
        transfers = {
            "torch": common.reference_transfer(report, clock, "node_features", to_device),
            "sym": common.sym_transfer(report, clock, "node_features",
                                       torch.compile(to_device, backend=backend, dynamic=True), to_device),
        }
        logits = {}
        with torch.no_grad():
            for path, transfer in transfers.items():
                logits[path] = [model.forward(transfer(features[block["nodes"]]),     # CPU gather, then transfer
                                              tuple(t.to(device) for t in block["layer1"]),
                                              tuple(t.to(device) for t in block["layer2"]))
                                for block in blocks]
        stats = backend.stats()
    finally:
        backend.close()
    for index, (sym, reference) in enumerate(zip(logits["sym"], logits["torch"])):
        report.check("logits_equal", common.same_tensor(sym, reference),
                     lambda: f"batch {index}: {common.difference(sym, reference)}")
    rows = [len(block["nodes"]) for block in blocks]
    report.data["workload"]["rows_per_batch"] = rows
    report.set_backend_stats(stats)
    count = len(blocks)
    report.check("one_plan_for_every_row_count", stats["plan_compiles"] == 1,
                 f"plan_compiles={stats['plan_compiles']}")
    report.check("rows_change_between_batches", len(set(rows)) >= 2, f"rows per batch: {rows}")
    report.check("one_execution_per_batch", stats["typed_executions"] == count,
                 f"typed_executions={stats['typed_executions']} for {count} batches")
    report.check("cpu_reference_row", report.data["dispatches"] == {"cpu_reference": count},
                 f"dispatches={report.data['dispatches']}")
    report.check("observed_payload_matches_wire",
                 stats["typed_payload_bytes"] == report.data["bytes"]["node_features"]["wire"],
                 f"typed_payload_bytes={stats['typed_payload_bytes']}")
    report.check("no_fallbacks", not stats["fallbacks"] and not stats["exclusions"],
                 f"fallbacks={dict(stats['fallbacks'])}, exclusions={dict(stats['exclusions'])}")
    report.note("sampling, the feature gather and the index tensors' moves are PyTorch; Sym moves the feature block")
    report.note("the PyTorch reference moves the same float16 bytes: the dtype choice halves the wire, not Sym")
    report.note("transfers are blocking; every Sym call includes per-call runtime setup")
    return report.finish(args.output)


if __name__ == "__main__":
    raise SystemExit(common.run_example(main))
