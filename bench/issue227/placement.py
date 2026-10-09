#!/usr/bin/env python3
"""Forced completed-call calibration and held-out placement qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend, TransferResources
from reloc_torch import placement
from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides

CONTEXT = dict(cpu_load='idle', gpu_load='idle', overlap='none')
OPTIONS = dict(gather_threads=8, n_streams=1, n_buffers=2, pinning='pinned')
LIMITS = dict(max_retained_bytes=64 << 20, max_live_staging_bytes=64 << 20,
              max_typed_retained_bytes=64 << 20, max_typed_live_bytes=64 << 20)


def measure(fn, device):
    start = time.perf_counter_ns()
    value = fn()
    torch.cuda.current_stream(device).synchronize()
    return value, (time.perf_counter_ns()-start)/1e6


def summary(samples):
    ordered = sorted(samples)
    return dict(p50_ms=statistics.median(samples), p95_ms=ordered[int(.95*(len(ordered)-1))],
                samples_ms=[round(x,6) for x in samples])


def check(value, expected):
    assert value.stride() == expected.stride()
    assert value.dtype == expected.dtype and value.device == expected.device
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


def case(direction, typed, kv, device):
    perm = (1,0,2) if kv else (1,0)
    destination = device if direction == 'h2d' else 'cpu'
    def transform(x):
        y = x.transpose(0,1).contiguous()
        return y.to(dtype=torch.float16) if typed else y
    def original(x):
        return transform(x).to(destination)
    def control(x, gpu, compiled):
        target = device if gpu else 'cpu'
        y = x.to(target)
        return compiled(y).to(destination)
    return perm, original, transform, control


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(227)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 256
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    options = OPTIONS | dict(gather_threads=args.threads)
    result = dict(hardware=placement.hardware(device), options=options, limits=LIMITS,
                  round=args.round, protocol='sym227/1', cases={})
    observations = []
    for kv,typed in ((False,False),(False,True),(True,False)):
        for direction in ('h2d','d2h'):
            torch._dynamo.reset()
            name = f'{"kv" if kv else "matrix"}_{"cast" if typed else "layout"}_{direction}'
            perm, original, transform, control = case(direction,typed,kv,device)
            policy = PlacementPolicy(context=CONTEXT,capacity=512)
            with TransferResources(**LIMITS) as resources:
                backend = RelocBackend(placement=policy,transfer_resources=resources,transfer_options=options)
                legacy = RelocBackend(transfer_resources=resources,transfer_options=options)
                compiled = torch.compile(original,backend=backend,fullgraph=True,dynamic=True)
                old = torch.compile(original,backend=legacy,fullgraph=True,dynamic=True)
                cpu = torch.compile(transform,backend='inductor',fullgraph=True,dynamic=True,
                                    options={'triton.cudagraphs':False})
                gpu = torch.compile(transform,backend='inductor',fullgraph=True,dynamic=True,
                                    options={'triton.cudagraphs':False})
                training = ([4,8,32,128,512] if kv else
                            [(n,w) for n in (64,256,1024,4096) for w in (256,1024)])
                heldout = ([12,24,64,256] if kv else [(128,512),(512,512),(2048,512),(2048,1024)])
                if args.quick:
                    training,heldout = training[:2],heldout[:1]
                rows = result['cases'][name] = []
                paths = ['torch_cpu','torch_gpu','sym_cpu'] + (['sym_gpu'] if typed else ['native'])
                profile = None
                for split,sizes in (('train',training),('heldout',heldout)):
                    if split == 'heldout':
                        profile = PlacementProfile(hardware=result['hardware'],
                            execution=backend.runtime.placement_configuration()[0],observations=observations)
                        policy.profile = profile
                    for size in sizes:
                        shape = ((size,16,64) if direction=='h2d' else (16,size,64)) if kv else size
                        x = torch.randn(shape,device=device if direction=='d2h' else 'cpu')
                        expected,_ = measure(lambda:original(x),device)
                        # Compile both compute locations and warm the process allocator.
                        controls = dict(torch=lambda:original(x),
                            inductor_cpu=lambda:control(x,False,cpu),
                            inductor_gpu=lambda:control(x,True,gpu), legacy_sym=lambda:old(x))
                        for fn in controls.values():
                            check(measure(fn,device)[0],expected)
                        with policy.force(paths[0]):
                            check(measure(lambda:compiled(x),device)[0],expected)
                        assert policy.stats()['last'] is not None, backend.stats()
                        row = dict(split=split,shape=list(shape),paths={})
                        rows.append(row)
                        order = paths[args.round%len(paths):]+paths[:args.round%len(paths)]
                        for path in order:
                            for state in ('cold','warm'):
                                if state == 'warm':
                                    with policy.force(path):
                                        for _ in range(3):
                                            compiled(x)
                                samples=[]
                                count = 3 if state == 'cold' else args.samples
                                for _ in range(count):
                                    if state == 'cold':
                                        resources.clear()
                                    with policy.force(path):
                                        value,ms = measure(lambda:compiled(x),device)
                                    check(value,expected)
                                    samples.append(ms)
                                record = policy.stats()['last']
                                assert record['path'] == path and record['resource_state'] == state, record
                                row['paths'][path+'_'+state] = summary(samples)
                                if split == 'train':
                                    observations.append({k:record[k] for k in
                                        ('family','shape','strides','path','wire_bytes','resource_state','context')} |
                                        dict(completed_ms=statistics.median(samples)))
                        if split == 'heldout':
                            # All alternatives have seen this exact shape in the current owner generation.
                            for path in paths:
                                with policy.force(path):
                                    compiled(x)
                            runs = dict(controls,automatic=lambda:compiled(x))
                            def forced(path):
                                def execute():
                                    with policy.force(path):
                                        return compiled(x)
                                return execute
                            runs.update({f'forced_{path}':forced(path) for path in paths})
                            values = {n:[] for n in runs}
                            automatic_decision = None
                            for i in range(args.samples):
                                order = list(runs)
                                order = order[(i+args.round)%len(order):]+order[:(i+args.round)%len(order)]
                                for path in order:
                                    value,ms = measure(runs[path],device)
                                    if path == 'automatic':
                                        automatic_decision = policy.stats()['last']
                                    check(value,expected)
                                    values[path].append(ms)
                            row['controls'] = {n:summary(v) for n,v in values.items()}
                            row['decision'] = automatic_decision
                            best = min(row['controls']['forced_'+p]['p50_ms'] for p in paths)
                            row['regret'] = row['controls']['automatic']['p50_ms']/best-1
                            row['path_regret'] = row['controls']['forced_'+row['decision']['path']]['p50_ms']/best-1
                            # Selector only: same shape/family, no copy, binding or GPU work.
                            dims = tuple(Const(n) for n in shape)
                            output = tuple(dims[i] for i in perm)
                            recipe = Recipe(TensorSpec(dims,dense_strides(dims),Const(0),'float32'),
                                (Transpose(perm),)+((Cast('float16','ieee_rne'),) if typed else ()),
                                TensorSpec(output,dense_strides(output),Const(0),'float16' if typed else 'float32'),direction)
                            spec = SimpleNamespace(recipe=recipe)
                            latencies=[]
                            for _ in range(100):
                                start=time.perf_counter_ns()
                                policy.choose(spec,x,device if direction=='h2d' else 'cpu',backend.runtime)
                                latencies.append((time.perf_counter_ns()-start)/1e6)
                            row['selection_only'] = summary(latencies)
                            resources.clear()
                            value,ms = measure(lambda:compiled(x),device)
                            check(value,expected)
                            row['automatic_cold'] = dict(ms=ms,decision=policy.stats()['last'])
                        print(name,split,shape,{k:round(v['p50_ms'],3) for k,v in row['paths'].items() if k.endswith('warm')},flush=True)
                        args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')
                result['cases'][name+'_stats'] = backend.stats()
                backend.close()
                legacy.close()
    profile.save(args.output.with_suffix('.profile.json'))
    result['benchmark_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--round',type=int,default=0)
    parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--quick',action='store_true')
    args = parser.parse_args()
    with torch.no_grad():
        run(args)
