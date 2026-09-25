#!/usr/bin/env python3
"""Mixture-of-experts FFN blocks with CPU-offloaded int8 experts (workload example).

Expert weights stay in CPU memory as int8 ``[in, out]`` matrices with float32
per-output-channel scales. For every batch the router (resident on the
primary GPU) picks the top-2 experts per token; only the experts that were
selected move to a GPU, as float32 ``[out, in]``, run on their tokens and
are combined. The Sym pass moves them with ``WeightFetcher`` (one symbolic
typed recipe for both matrix shapes); the reference pass moves them with
plain PyTorch. With ``--devices cuda:0,cuda:1,...`` expert ``e`` lives on
device ``e % n`` and tokens travel between GPUs with PyTorch. Routing, token
dispatch and the combine are PyTorch in both passes, and both must produce
identical outputs.

    python moe_experts.py [--devices cuda:0[,cuda:1,...]] [--quick] [--calibration auto|none|PATH]
                          [--output report.json]

See README.md in this directory for the environment and the report fields.
"""
from __future__ import annotations

import argparse

import common

torch = common.import_torch()          # exits 2 with a one-line message when torch is missing
F = torch.nn.functional

TOP_K = 2
SIZES = {
    "default": {"d_model": 1024, "d_ff": 2048, "experts": 16, "blocks": 2, "batches": [256, 32, 8]},
    "quick": {"d_model": 256, "d_ff": 512, "experts": 4, "blocks": 1, "batches": [64, 8]},
}


class OffloadedMoE:
    """Pre-LN MoE FFN blocks: routers on the primary device, experts offloaded to the CPU as int8."""

    def __init__(self, sizes, devices, generator):
        d, ff = sizes["d_model"], sizes["d_ff"]
        self.d_model, self.devices = d, devices
        self.routers = [(torch.randn(d, sizes["experts"], generator=generator) / d ** 0.5).to(devices[0])
                        for _ in range(sizes["blocks"])]
        self.experts = [
            [{"w1": common.quantize_per_channel(torch.randn(d, ff, generator=generator) / d ** 0.5),
              "w2": common.quantize_per_channel(torch.randn(ff, d, generator=generator) / ff ** 0.5)}
             for _ in range(sizes["experts"])]
            for _ in range(sizes["blocks"])
        ]

    def device_of(self, expert):
        return self.devices[expert % len(self.devices)]

    def forward(self, x, fetch, on_route):
        """x [T, d] on the primary device. ``fetch(q, scale, device)`` returns a float32
        [out, in] weight on ``device``; ``on_route(active)`` sees each block's sorted active experts."""
        for router, experts in zip(self.routers, self.experts):
            h = F.layer_norm(x, (self.d_model,))
            top_logits, top_experts = (h @ router).topk(TOP_K, dim=1)
            gates = top_logits.softmax(dim=1)
            active = torch.unique(top_experts).tolist()
            on_route(active)
            out = torch.zeros_like(x)
            for expert in active:
                tokens, slot = (top_experts == expert).nonzero(as_tuple=True)
                device = self.device_of(expert)
                w1 = fetch(*experts[expert]["w1"], device)
                w2 = fetch(*experts[expert]["w2"], device)
                y = F.linear(F.gelu(F.linear(h[tokens].to(device), w1)), w2).to(x.device)
                out.index_add_(0, tokens, gates[tokens, slot].unsqueeze(1) * y)   # unique rows per expert
            x = x + out
        return x


def parse_devices(parser, value, experts):
    """``--devices`` -> distinct torch devices, at most one per expert (usage errors exit 2)."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    try:
        devices = [torch.device(name) for name in names]
    except RuntimeError as error:
        parser.error(f"invalid --devices {value!r}: {error}")
    if not devices:
        parser.error("--devices is empty")
    if len(set(devices)) != len(devices):
        parser.error(f"--devices repeats a device: {value}")
    if len(devices) > experts:
        parser.error(f"--devices lists {len(devices)} devices but the model has only {experts} experts")
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="cuda:0", help="comma-separated; expert e lives on device e %% n")
    parser.add_argument("--quick", action="store_true", help="a small model and two batches")
    parser.add_argument("--calibration", default="auto", help="auto, none or a .cal path for the dispatch cost model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args()
    sizes = SIZES["quick" if args.quick else "default"]
    devices = parse_devices(parser, args.devices, sizes["experts"])
    common.require_environment(devices)
    common.seed_everything(args.seed)
    report = common.Report("moe_experts", devices=devices, top_k=TOP_K, **sizes)
    calibration, label = common.resolve_calibration(args.calibration, torch.cuda.get_device_name(devices[0]))
    report.note(f"calibration: {label}")
    clock = common.Clock(devices)
    generator = torch.Generator().manual_seed(args.seed)
    model = OffloadedMoE(sizes, devices, generator)
    inputs = [torch.randn(tokens, sizes["d_model"], generator=generator).to(devices[0]) for tokens in sizes["batches"]]
    fetcher = common.WeightFetcher(report, calibration, kind="expert_weights")
    fetches = {str(device): 0 for device in devices}

    def sym_fetch(q, scale, device):
        fetches[str(device)] += 1
        return fetcher.fetch(q, scale, device)

    reference = common.WeightFetcher.reference
    paths = {"torch": common.reference_transfer(report, clock, "expert_weights", reference),
             "sym": common.sym_transfer(report, clock, "expert_weights", sym_fetch, reference, count_bytes=False)}
    routes, outputs = {}, {}
    with torch.no_grad():
        for path, fetch in paths.items():
            routes[path] = []
            outputs[path] = [model.forward(x, fetch, routes[path].append) for x in inputs]
    for index, (sym, reference_out) in enumerate(zip(outputs["sym"], outputs["torch"])):
        report.check("outputs_equal", common.same_tensor(sym, reference_out),
                     lambda: f"batch {index}: {common.difference(sym, reference_out)}")
    active = sum(len(experts) for experts in routes["sym"])
    possible = len(inputs) * sizes["blocks"] * sizes["experts"]
    report.check("routing_identical", routes["sym"] == routes["torch"], "the two passes routed differently")
    report.check("only_active_experts_fetched", sum(fetches.values()) == 2 * active,
                 f"{sum(fetches.values())} fetches for {active} active experts")
    report.check("every_device_received_experts", all(fetches.values()), f"fetches per device: {fetches}")
    report.check("one_weight_artifact_for_both_shapes", fetcher.compiles == 1 and len(fetcher.shapes) == 2,
                 f"compiles={fetcher.compiles}, shapes={sorted(fetcher.shapes)}")
    report.data["workload"].update(active_experts=routes["sym"], active_expert_count=active,
                                   possible_expert_count=possible, fetches_per_device=fetches)
    report.note(f"{active} of {possible} expert slots were active, and only those experts' weights moved")
    report.note("routing, token dispatch, GPU-to-GPU token moves and the combine are PyTorch in both passes")
    report.note("several GPUs show placement and correctness only: the multi-GPU throughput gate failed on this "
                "box (R3)")
    report.note("transfers are blocking; every Sym call includes per-call runtime setup")
    return report.finish(args.output)


if __name__ == "__main__":
    raise SystemExit(common.run_example(main))
