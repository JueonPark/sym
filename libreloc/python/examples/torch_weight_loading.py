"""Repeated module weight loading with observation, eager routing and explicit preparation.

Four scenarios, each asserting the numerical result against plain PyTorch:

1. Ordinary module APIs under observation only: ``module.to(device)`` and
   repeated ``load_state_dict`` outside ``forward``. Observation classifies
   every transfer (module parameters/buffers are candidates; ``load_state_dict``
   is ``copy_`` mutation and stays excluded) but replaces nothing.
2. ``module.to(device)`` inside ``eager_transfers`` under ``torch.no_grad()``:
   parameters and buffers are relocated through the runtime adapter.
3. Explicit stable preparation of a buffer and a nested parameter with a
   transpose recipe: repeated ``get`` reuses one preparation; an in-place
   update and a ``load_state_dict`` rebuild it; every result is fresh.
4. Explicitly requested quantized preparation is gated on the typed contract
   (C3/C4/R3); the gate's reason is reported and the layout-only scenarios stay
   runnable.

With ``--device cpu`` the adapter reports ``direction_mismatch`` and every
relocation runs through the recorded PyTorch fallback, which exercises the
lifecycle mechanics without CUDA.
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict

import torch

from reloc_torch import RelocBackend, check_version, classify, eager_transfers, observe_transfers, prepare_weights


def transpose_recipe():
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, dense_strides

    source = (Const(4), Const(6))
    destination = (Const(6), Const(4))
    return Recipe(
        TensorSpec(source, dense_strides(source), Const(0), "float32"),
        (Transpose((1, 0)),),
        TensorSpec(destination, dense_strides(destination), Const(0), "float32"),
        "h2d",
    )


def counters(backend):
    stats = backend.stats()
    return {k: stats[k] for k in ("plan_compiles", "symbol_binds", "dynamo_compiles", "runtime_executions",
                                  "weight_preparations", "weight_invalidations")} | {
        "fallbacks": stats["fallbacks"], "exclusions": stats["exclusions"], "redispatches": stats["redispatches"]}


def check(report, name, condition):
    report.setdefault("checks", {})[name] = bool(condition)
    return bool(condition)


def scenario_observation(device, report):
    module = torch.nn.Linear(4, 4).requires_grad_(False)
    module.register_buffer("scale", torch.ones(4))
    observer = observe_transfers()
    with observer, observer.phase("module_to"):
        module.to(device)
    with observer, observer.phase("load_state_dict"):
        for value in (2.0, 3.0):
            module.load_state_dict(dict(weight=torch.full((4, 4), value), bias=torch.full((4,), value), scale=torch.full((4,), value)))
    records = observer.records
    classes = Counter((r.phase, classify(r).category, classify(r).reason) for r in records)
    report["observation"] = {
        "events": len(records),
        "cross_device_transfers": sum(r.source.device_type != r.destination.device_type for r in records),
        "classification": {f"{phase}/{category}/{reason}": n for (phase, category, reason), n in sorted(classes.items())},
        "records": [asdict(r) for r in records[:6]],
    }
    return check(report, "observation_keeps_values", torch.equal(module.weight.detach().cpu(), torch.full((4, 4), 3.0)))


def scenario_eager_module_to(device, backend, report):
    module = torch.nn.Linear(4, 4)
    module.register_buffer("scale", torch.arange(4.0))
    reference = {k: v.detach().clone() for k, v in module.state_dict().items()}
    before = backend.stats()
    with torch.no_grad(), eager_transfers(backend=backend):
        module.to(device)
    after = backend.stats()
    report["eager_module_to"] = {
        "device": str(module.weight.device),
        "runtime_executions": after["runtime_executions"] - before["runtime_executions"],
        "fallbacks": after["fallbacks"],
        "redispatches": after["redispatches"],
    }
    same = all(torch.equal(v.detach().cpu(), reference[k]) for k, v in module.state_dict().items())
    return check(report, "eager_module_to_values", same and module.weight.device.type == device.type)


def scenario_stable_preparation(device, backend, report):
    module = torch.nn.Module()
    module.register_buffer("weight", torch.arange(24.0).reshape(4, 6))
    module.layer = torch.nn.Module()
    module.layer.weight = torch.nn.Parameter(torch.arange(24.0).reshape(4, 6) + 100)
    recipes = {"weight": transpose_recipe(), "layer.weight": transpose_recipe()}
    ok = True
    with torch.no_grad(), prepare_weights(module, recipes, backend=backend, stable=True) as prepared:
        outputs = [prepared.get("weight", device=device) for _ in range(3)]
        ok &= all(torch.equal(o.cpu(), module.weight.t().contiguous()) for o in outputs)
        ok &= len({o.data_ptr() for o in outputs}) == 3
        module.weight.add_(1)
        updated = prepared.get("weight", device=device)
        ok &= torch.equal(updated.cpu(), module.weight.t().contiguous())
        module.load_state_dict({"weight": torch.full((4, 6), 9.0), "layer.weight": torch.full((4, 6), 7.0)})
        ok &= torch.equal(prepared.get("weight", device=device).cpu(), torch.full((6, 4), 9.0))
        ok &= torch.equal(prepared.get("layer.weight", device=device).cpu(), torch.full((6, 4), 7.0))
        ok &= module.layer.weight.requires_grad and isinstance(module.layer.weight, torch.nn.Parameter)
        stats = backend.stats()
    report["stable_preparation"] = {
        "weight_preparations": stats["weight_preparations"],
        "weight_invalidations": stats["weight_invalidations"],
        "device": str(device),
    }
    return check(report, "stable_preparation_values", ok)


def scenario_quantized(report):
    from reloc_torch.prefold import typed_prefold_capability

    reason = typed_prefold_capability()
    report["quantized_preparation"] = {"status": "gated", "reason": reason}
    return check(report, "quantized_gate_reported", reason is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    check_version()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    report = {"device": str(device), "torch": str(torch.__version__)}
    backend = RelocBackend()
    ok = scenario_observation(device, report)
    ok &= scenario_eager_module_to(device, backend, report)
    ok &= scenario_stable_preparation(device, backend, report)
    ok &= scenario_quantized(report)
    report["counters"] = counters(backend)
    backend.close()
    text = json.dumps(report, indent=2, default=str) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
