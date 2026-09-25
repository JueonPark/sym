#!/usr/bin/env python3
"""GPT-style decoding with CPU-offloaded int8 weights and KV-cache eviction (workload example).

Every layer's projection matrices stay in CPU memory as int8 with float32
per-output-channel scales, in the ``[in, out]`` layout of GPT-2-style
checkpoints; they are quantized once, offline, with PyTorch. Just before a
layer runs, its four matrices move to the GPU as float32 ``[out, in]`` (the
``nn.Linear`` layout) and are dropped afterwards, so only one layer's
weights occupy the GPU. The Sym pass moves them with ``WeightFetcher``: one
symbolic typed recipe for every matrix shape, and the runtime's cost model
decides where the dequantize runs. After every decoding step the KV cache
leaves the GPU (``[H, S, Dh]`` -> ``[S, H, Dh]`` on the CPU) and comes back
before the next step, as a server does when it swaps sessions; ``S`` grows
by one per step and each direction keeps one plan. The reference pass does
the same with plain PyTorch, and both must produce identical tokens and logits.

    python llm_offload.py [--device cuda:0] [--quick] [--calibration auto|none|PATH] [--output report.json]

See README.md in this directory for the environment and the report fields.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import common

SIZES = {
    "default": {"vocab": 8192, "d_model": 1024, "heads": 16, "layers": 8, "d_ff": 4096, "prompt": 32,
                "new_tokens": 8},
    "quick": {"vocab": 1024, "d_model": 256, "heads": 4, "layers": 2, "d_ff": 1024, "prompt": 8, "new_tokens": 4},
}
MATRICES = ("qkv", "proj", "fc1", "fc2")


class OffloadedGPT:
    """Pre-LN decoder: embeddings and positions resident on the GPU, projection matrices offloaded as int8."""

    def __init__(self, sizes, device, generator):
        d, ff = sizes["d_model"], sizes["d_ff"]
        self.d_model, self.heads = d, sizes["heads"]
        self.token = (torch.randn(sizes["vocab"], d, generator=generator) * 0.02).to(device)
        self.position = (torch.randn(sizes["prompt"] + sizes["new_tokens"], d, generator=generator) * 0.02).to(device)
        shapes = {"qkv": (d, 3 * d), "proj": (d, d), "fc1": (d, ff), "fc2": (ff, d)}
        self.layers = [
            {name: common.quantize_per_channel(torch.randn(rows, cols, generator=generator) / rows ** 0.5)
             for name, (rows, cols) in shapes.items()}
            for _ in range(sizes["layers"])
        ]

    def forward(self, tokens, start, cache, fetch):
        """Run ``tokens`` (GPU, positions ``start``, ``start + 1``, ...) through every layer.

        ``cache`` holds each layer's (K, V) as [H, S, Dh] or None; ``fetch(q, scale)``
        returns a float32 [out, in] weight on the GPU. Returns logits [n, vocab] and the new cache."""
        d, heads = self.d_model, self.heads
        head_dim = d // heads
        x = self.token[tokens] + self.position[start:start + len(tokens)]
        new_cache = []
        for layer, past in zip(self.layers, cache):
            w = {name: fetch(*layer[name]) for name in MATRICES}
            q, k, v = F.linear(F.layer_norm(x, (d,)), w["qkv"]).split(d, dim=1)
            q, k, v = (t.view(-1, heads, head_dim).transpose(0, 1) for t in (q, k, v))    # [H, n, Dh]
            if past is not None:
                k, v = torch.cat([past[0], k], dim=1), torch.cat([past[1], v], dim=1)
            k, v = k.contiguous(), v.contiguous()
            new_cache.append((k, v))
            total, n = k.shape[1], q.shape[1]
            mask = torch.ones(n, total, dtype=torch.bool, device=x.device).tril(total - n)
            scores = (q @ k.transpose(1, 2) / head_dim ** 0.5).masked_fill(~mask, float("-inf"))
            attention = scores.softmax(dim=-1) @ v                                          # [H, n, Dh]
            x = x + F.linear(attention.transpose(0, 1).reshape(n, d), w["proj"])
            x = x + F.linear(F.gelu(F.linear(F.layer_norm(x, (d,)), w["fc1"])), w["fc2"])
        return F.layer_norm(x, (d,)) @ self.token.t(), new_cache


def generate(model, prompt, steps, fetch, evict, restore):
    """Greedy decoding for ``steps`` forward passes. Between passes every
    layer's K and V leave the GPU (``evict``) and come back (``restore``).
    Returns the generated token ids and the last pass's logits."""
    tokens, start, offloaded, generated = prompt, 0, None, []
    for step in range(steps):
        if offloaded is None:
            cache = [None] * len(model.layers)
        else:
            cache = [(restore(k), restore(v)) for k, v in offloaded]
        logits, cache = model.forward(tokens, start, cache, fetch)
        token = logits[-1].argmax().view(1)
        generated.append(int(token))
        start += len(tokens)
        tokens = token
        offloaded = [(evict(k), evict(v)) for k, v in cache] if step < steps - 1 else None
    return generated, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="a tiny model and a short generation")
    parser.add_argument("--calibration", default="auto", help="auto, none or a .cal path for the dispatch cost model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args()
    common.require_environment([args.device])
    common.seed_everything(args.seed)
    from reloc_torch import RelocBackend

    device = torch.device(args.device)
    sizes = SIZES["quick" if args.quick else "default"]
    report = common.Report("llm_offload", devices=[device], **sizes)
    calibration, label = common.resolve_calibration(args.calibration, torch.cuda.get_device_name(device))
    report.note(f"calibration: {label}")
    clock = common.Clock([device])
    generator = torch.Generator().manual_seed(args.seed)
    model = OffloadedGPT(sizes, device, generator)
    prompt = torch.randint(0, sizes["vocab"], (sizes["prompt"],), generator=generator).to(device)
    fetcher = common.WeightFetcher(report, calibration)

    def reference_weight(q, scale):
        return common.WeightFetcher.reference(q, scale, device)

    def evict(kv):
        return kv.transpose(0, 1).contiguous().cpu()

    def restore(kv):
        return kv.transpose(0, 1).contiguous().to(device)

    backend = RelocBackend()
    try:
        paths = {
            "torch": (common.reference_transfer(report, clock, "weights", reference_weight),
                      common.reference_transfer(report, clock, "kv_evict", evict),
                      common.reference_transfer(report, clock, "kv_restore", restore)),
            "sym": (common.sym_transfer(report, clock, "weights", lambda q, scale: fetcher.fetch(q, scale, device),
                                        reference_weight, count_bytes=False),
                    common.sym_transfer(report, clock, "kv_evict",
                                        torch.compile(evict, backend=backend, dynamic=True), evict),
                    common.sym_transfer(report, clock, "kv_restore",
                                        torch.compile(restore, backend=backend, dynamic=True), restore)),
        }
        outputs = {}
        with torch.no_grad():
            for path, (fetch, evict_kv, restore_kv) in paths.items():
                outputs[path] = generate(model, prompt, sizes["new_tokens"], fetch, evict_kv, restore_kv)
        stats = backend.stats()
    finally:
        backend.close()
    (sym_tokens, sym_logits), (torch_tokens, torch_logits) = outputs["sym"], outputs["torch"]
    report.check("tokens_equal", sym_tokens == torch_tokens, f"sym {sym_tokens} vs torch {torch_tokens}")
    report.check("logits_equal", common.same_tensor(sym_logits, torch_logits),
                 lambda: common.difference(sym_logits, torch_logits))
    report.set_backend_stats(stats)
    steps = sizes["new_tokens"]
    kv_transfers = sizes["layers"] * 2 * 2 * (steps - 1)        # K and V, evicted and restored between passes
    report.check("one_weight_artifact_for_every_matrix_shape", fetcher.compiles == 1 and len(fetcher.shapes) == 4,
                 f"compiles={fetcher.compiles}, shapes={sorted(fetcher.shapes)}")
    report.check("one_kv_plan_per_direction", stats["plan_compiles"] == 2, f"plan_compiles={stats['plan_compiles']}")
    report.check("one_kv_execution_per_transfer", stats["runtime_executions"] == kv_transfers,
                 f"runtime_executions={stats['runtime_executions']}, expected {kv_transfers}")
    report.check("no_fallbacks", not stats["fallbacks"] and not stats["exclusions"],
                 f"fallbacks={dict(stats['fallbacks'])}, exclusions={dict(stats['exclusions'])}")
    report.data["workload"].update(generated_tokens=sym_tokens,
                                   kv_lengths_evicted=[sizes["prompt"] + step for step in range(steps - 1)])
    report.note("weights: the PyTorch reference also moves int8 and dequantizes on the GPU, so the wires match "
                "whenever the runtime picks a GPU row; the dispatch rows show the cost model's choice")
    report.note("transfers are blocking, so weight streaming does not overlap compute; this toy model fits on the "
                "GPU anyway -- the example shows the mechanism, not a capacity win")
    return report.finish(args.output)


if __name__ == "__main__":
    raise SystemExit(common.run_example(main))
