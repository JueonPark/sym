"""One isolated E1/held-out/consumer process with a common completed H2D path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import pyreloc
from reloc_torch import CompilerClient
from reloc_torch.mlir_emit import emit_mlir
from _folding_native import Runner
from cases import make_case
from test_controls import equal_bits
import contract
from evidence import provenance, cache_inventory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def clock(fn):
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1e6


def array(tensor):
    return tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compile_bound(client, recipe, case, folder, name):
    artifact, compile_ms = clock(lambda: client.compile(recipe))
    params = {'scale': ('float32', [case.m], case.scales.numpy().tobytes())} if recipe.parameter_bindings else {}
    bound, bind_ms = clock(lambda: pyreloc.bind_typed(artifact.decoded_plan, {}, params)
                          if artifact.typed else pyreloc.bind(artifact.decoded_plan, {}))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + '.mlir')).write_text(emit_mlir(recipe))
    (folder / (name + '.bin')).write_bytes(artifact.plan_bytes)
    (folder / (name + '.manifest.json')).write_text(json.dumps(artifact.manifest, indent=2))
    return bound, dict(compile_ms=compile_ms, bind_ms=bind_ms, plan_bytes=len(artifact.plan_bytes),
                      source_operations=len(recipe.operations),
                      value_stages=len(bound.stages) if artifact.typed else 0,
                      plan_sha256=hashlib.sha256(artifact.plan_bytes).hexdigest())


def one_case(args, specification):
    job = args.job
    case = make_case(**specification)
    key = contract.case_key(specification)
    folder = args.output.parent / 'plans' / job['id'] / key
    start = time.perf_counter_ns()
    wire = torch.empty(case.shapes[-1], dtype=getattr(torch, case.dtypes[-1]), pin_memory=True)
    gpu = torch.empty_like(wire, device='cuda:0')
    inter = [torch.empty(s, dtype=getattr(torch, d))
             for s, d in zip(case.shapes[1:-1], case.dtypes[1:-1])]
    arrays = [array(t) for t in [case.source, *inter, wire]]
    fused_arrays = [arrays[0], arrays[-1]]
    allocation_ms = (time.perf_counter_ns() - start) / 1e6
    client = CompilerClient.from_environment()
    bounds, parts = [], []
    for i, recipe in enumerate(case.parts):
        bound, facts = compile_bound(client, recipe, case, folder, f'unfused-{i:02d}')
        bounds.append(bound)
        parts.append(facts)
    bound, folded_facts = compile_bound(client, case.recipe, case, folder, 'folded')
    unfused, unfused_prepare = clock(lambda: Runner(bounds, job['threads']))
    folded, folded_prepare = clock(lambda: Runner([bound], job['threads']))
    manual, manual_prepare = clock(lambda: Runner([], job['threads']))
    functions = {'sym_unfused': lambda: unfused.run(arrays),
                 'sym_folded': lambda: folded.run(fused_arrays),
                 'torch_eager': lambda: wire.copy_(case.reference(case.source))}
    if 'manual_fused' in contract.arms(specification):
        functions['manual_fused'] = lambda: manual.manual(case.family, arrays[0], arrays[-1],
                                                         case.scales.numpy(), case.n, case.m)
    torch._dynamo.reset()

    def into(source, destination):
        return destination.copy_(case.reference(source))

    compiled = torch.compile(into, backend='inductor', fullgraph=True, dynamic=False)
    functions['torch_inductor'] = lambda: compiled(case.source, wire)
    active_arms = contract.arms(specification)
    cache_before = cache_inventory(args.cache)
    paths = {arm: {} for arm in active_arms}
    expected = case.reference(case.source)
    initial_input = digest(case.source)
    initial_oracle = digest(expected)
    # Deterministic consumer, identical operands/kernel for all producer arms.
    activation = None
    if job['suite'] == 'consumer':
        generator = torch.Generator().manual_seed(267)
        activation = torch.randn((32, wire.shape[0]), generator=generator, dtype=wire.dtype).cuda()
        matmul_output = torch.empty((32, wire.shape[1]), dtype=wire.dtype, device='cuda:0')

    def consume(weights):
        torch.mm(activation, weights, out=matmul_output)
        return torch.nn.functional.gelu(matmul_output)

    def completed(fn):
        fn()
        gpu.copy_(wire, non_blocking=True)
        output = consume(gpu) if activation is not None else gpu
        torch.cuda.synchronize()
        return output

    shift = job['round'] % len(active_arms)
    first_order = active_arms[shift:] + active_arms[:shift]
    for arm in first_order:
        _, elapsed = clock(functions[arm])
        equal_bits(wire, expected)
        paths[arm]['first_producer_ms'] = elapsed
        start = time.perf_counter_ns()
        for _ in range(contract.WARMUP):
            completed(functions[arm])
        paths[arm]['warmup_total_ms'] = (time.perf_counter_ns() - start) / 1e6
        equal_bits(gpu.cpu(), expected)
    checks = 0
    for mode in contract.modes(job['suite']):
        for arm in active_arms:
            paths[arm][mode] = {'samples_ms': []}
        for sample in range(contract.SAMPLES):
            case.source.neg_()
            expected = case.reference(case.source)
            consumer_expected = consume(expected.cuda()).clone() if activation is not None else None
            torch.cuda.synchronize()
            shift = (sample + job['round']) % len(active_arms)
            for arm in active_arms[shift:] + active_arms[:shift]:
                torch.cuda.synchronize()
                if mode == 'producer':
                    _, elapsed = clock(functions[arm])
                    equal_bits(wire, expected)
                else:
                    result, elapsed = clock(lambda: completed(functions[arm]))
                    equal_bits(gpu.cpu(), expected)
                    if consumer_expected is not None:
                        equal_bits(result.cpu(), consumer_expected.cpu())
                paths[arm][mode]['samples_ms'].append(elapsed)
                checks += 1
    preparation = {
        'common_allocation_ms': allocation_ms,
        'sym_unfused': dict(compile_ms=sum(p['compile_ms'] for p in parts),
                            bind_ms=sum(p['bind_ms'] for p in parts), native_ms=unfused_prepare),
        'sym_folded': dict(compile_ms=folded_facts['compile_ms'],
                          bind_ms=folded_facts['bind_ms'], native_ms=folded_prepare),
        'manual_fused': dict(compile_ms=0., bind_ms=0., native_ms=manual_prepare),
        'torch_eager': dict(compile_ms=0., bind_ms=0., native_ms=0.),
        'torch_inductor': dict(compile_ms=0., bind_ms=0., native_ms=0.),
        'inductor_note': 'Lazy compilation is included in first_producer_ms, with a fresh per-process cache.',
        'parts': parts, 'folded': folded_facts}
    return dict(case=specification, checks=checks, input_sha256=initial_input,
                oracle_sha256=initial_oracle, scale_sha256=digest(case.scales), paths=paths,
                h2d_copies_per_call=1, source_bytes=case.source.nbytes, wire_bytes=wire.nbytes,
                materialized_boundaries=len(inter), intermediate_bytes=sum(t.nbytes for t in inter),
                kernels={'sym_unfused': unfused.kernels(), 'sym_folded': folded.kernels()},
                preparation=preparation, cache_before=cache_before, cache_after=cache_inventory(args.cache),
                runtime_shape_code_cache_bytes=0, consumer_batch=32 if activation is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.job = next(j for j in contract.jobs() if j['id'] == args.job)
    args.cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    assert not args.output.exists()
    torch.set_num_threads(args.job['threads'])
    torch.set_num_interop_threads(1)
    torch._inductor.config.emulate_precision_casts = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    # Loader provenance must match the selected runtime, including when rpath
    # would otherwise silently resolve a different libreloc.
    maps = Path('/proc/self/maps').read_text()
    actual_libraries = {line.split()[-1] for line in maps.splitlines() if 'libreloc_runtime.so' in line}
    assert actual_libraries == {str((args.runtime_root / 'libreloc/libreloc_runtime.so').resolve())}
    record = dict(job=args.job, complete=False, samples=contract.SAMPLES, warmup=contract.WARMUP,
                  affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
                  interop_threads=torch.get_num_interop_threads(),
                  emulate_precision_casts=torch._inductor.config.emulate_precision_casts,
                  torch=torch.__version__, python=os.sys.version, gpu=torch.cuda.get_device_name(0),
                  provenance=provenance(REPO, args.runtime_root, Path(os.environ['SYM_RELOC_EXPORT'])),
                  cases={})
    with torch.no_grad():
        for case in contract.cases(args.job['suite']):
            key = contract.case_key(case)
            record['cases'][key] = one_case(args, case)
            args.output.write_text(json.dumps(record, indent=2) + '\n')
            print(key, 'complete', flush=True)
    record['complete'] = True
    args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
