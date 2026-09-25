#!/usr/bin/env python3
"""DLRM inference with CPU-resident embedding tables (workload example).

The embedding tables live in CPU memory, as they do when they are too large
for the GPU. Each batch looks up and sum-pools its bags on the CPU with
PyTorch (an index-based gather that Sym cannot express) and moves the
pooled ``[T, B, 64]`` float32 block to the GPU as ``[B, T, 64]`` float16 for
the dot interaction. The Sym pass moves it through one plan compiled by
``RelocBackend`` and reused for every batch size; the reference pass runs
the same function eagerly. Both passes must produce identical predictions.

    python dlrm_embeddings.py [--device cuda:0] [--quick] [--output report.json]

See README.md in this directory for the environment and the report fields.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import common

DENSE_FEATURES = 13
EMBEDDING_DIM = 64
SIZES = {
    "default": {"tables": 26, "rows": 100_000, "batches": [2048, 2048, 1536, 512]},
    "quick": {"tables": 8, "rows": 20_000, "batches": [256, 256, 192, 64]},
}


def linear_stack(sizes, generator, device):
    """(weight, zero bias) pairs of an MLP with these layer widths, on ``device``."""
    layers = []
    for fan_in, fan_out in zip(sizes, sizes[1:]):
        weight = torch.randn(fan_out, fan_in, generator=generator) / fan_in ** 0.5
        layers.append((weight.to(device), torch.zeros(fan_out, device=device)))
    return layers


def run_stack(layers, x):
    """ReLU between layers, none after the last."""
    for index, (weight, bias) in enumerate(layers):
        x = F.linear(x, weight, bias)
        if index < len(layers) - 1:
            x = torch.relu(x)
    return x


class DLRM:
    """Sum-pooled embedding tables on the CPU; bottom MLP, dot interaction and top MLP on the GPU."""

    def __init__(self, tables, rows, device, generator):
        self.tables = [
            torch.nn.EmbeddingBag.from_pretrained(torch.randn(rows, EMBEDDING_DIM, generator=generator) * 0.05,
                                                  freeze=True, mode="sum")
            for _ in range(tables)
        ]
        self.bottom = linear_stack([DENSE_FEATURES, 512, 256, EMBEDDING_DIM], generator, device)
        self.top = linear_stack([EMBEDDING_DIM + (tables + 1) * tables // 2, 512, 256, 1], generator, device)
        self.pairs = torch.triu_indices(tables + 1, tables + 1, offset=1, device=device)

    def pooled(self, indices, offsets):
        """CPU, PyTorch: every table's sum-pooled bags -> [T, B, 64] float32."""
        return torch.stack([table(i, o) for table, i, o in zip(self.tables, indices, offsets)])

    def predict(self, dense, embeddings):
        """GPU: dense [B, 13] float32 and embeddings [B, T, 64] float16 -> click probabilities [B]."""
        bottom = run_stack(self.bottom, dense)
        vectors = torch.cat([bottom.unsqueeze(1), embeddings.float()], dim=1)       # [B, T + 1, 64]
        dots = torch.bmm(vectors, vectors.transpose(1, 2))
        interactions = dots[:, self.pairs[0], self.pairs[1]]                      # [B, (T + 1) T / 2]
        return torch.sigmoid(run_stack(self.top, torch.cat([bottom, interactions], dim=1))).squeeze(1)


def make_batch(batch, tables, rows, generator):
    """Dense features and, per table, one bag of 1-4 uniform random rows per sample."""
    dense = torch.randn(batch, DENSE_FEATURES, generator=generator)
    indices, offsets = [], []
    for _ in range(tables):
        bags = torch.randint(1, 5, (batch,), generator=generator)
        offsets.append(torch.cat([torch.zeros(1, dtype=torch.long), bags.cumsum(0)[:-1]]))
        indices.append(torch.randint(0, rows, (int(bags.sum()),), generator=generator))
    return dense, indices, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="small tables and batches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args()
    common.require_environment([args.device])
    common.seed_everything(args.seed)
    from reloc_torch import RelocBackend

    device = torch.device(args.device)
    sizes = SIZES["quick" if args.quick else "default"]
    report = common.Report("dlrm_embeddings", devices=[device], embedding_dim=EMBEDDING_DIM, **sizes)
    clock = common.Clock([device])
    generator = torch.Generator().manual_seed(args.seed)
    model = DLRM(sizes["tables"], sizes["rows"], device, generator)
    batches = [make_batch(batch, sizes["tables"], sizes["rows"], generator) for batch in sizes["batches"]]

    def to_device(pooled):
        return pooled.transpose(0, 1).contiguous().to(device, torch.float16)

    backend = RelocBackend()
    try:
        transfers = {
            "torch": common.reference_transfer(report, clock, "pooled_embeddings", to_device),
            "sym": common.sym_transfer(report, clock, "pooled_embeddings",
                                       torch.compile(to_device, backend=backend, dynamic=True), to_device),
        }
        predictions = {}
        with torch.no_grad():
            for path, transfer in transfers.items():
                predictions[path] = [model.predict(dense.to(device), transfer(model.pooled(indices, offsets)))
                                     for dense, indices, offsets in batches]
        stats = backend.stats()
    finally:
        backend.close()
    for index, (sym, reference) in enumerate(zip(predictions["sym"], predictions["torch"])):
        report.check("predictions_equal", common.same_tensor(sym, reference),
                     lambda: f"batch {index}: {common.difference(sym, reference)}")
    report.set_backend_stats(stats)
    count = len(batches)
    report.check("one_plan_for_every_batch_size", stats["plan_compiles"] == 1,
                 f"plan_compiles={stats['plan_compiles']}")
    report.check("one_execution_per_batch", stats["typed_executions"] == count,
                 f"typed_executions={stats['typed_executions']} for {count} batches")
    report.check("cpu_reference_row", report.data["dispatches"] == {"cpu_reference": count},
                 f"dispatches={report.data['dispatches']}")
    report.check("observed_payload_matches_wire",
                 stats["typed_payload_bytes"] == report.data["bytes"]["pooled_embeddings"]["wire"],
                 f"typed_payload_bytes={stats['typed_payload_bytes']}")
    report.check("no_fallbacks", not stats["fallbacks"] and not stats["exclusions"],
                 f"fallbacks={dict(stats['fallbacks'])}, exclusions={dict(stats['exclusions'])}")
    report.data["workload"]["batch_sizes"] = sizes["batches"]
    report.note("Sym transposes and narrows on the CPU (cpu_reference is the only f32->f16 row), then copies float16")
    report.note("the PyTorch reference moves the same float16 bytes: the dtype choice halves the wire, not Sym")
    report.note("transfers are blocking; every Sym call includes per-call runtime setup")
    return report.finish(args.output)


if __name__ == "__main__":
    raise SystemExit(common.run_example(main))
