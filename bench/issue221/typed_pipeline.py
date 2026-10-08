#!/usr/bin/env python3
"""Completed typed H2D calls; native traces and headline timings are separate.

Recipes, Inductor compilation, input changes and byte checks are outside timing.
Allocation, binding/dispatch, host transform, copy and completion are inside.
Run independent processes with rotated --paths for independent rounds.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = ('small', 'cast', 'transpose', 'split')


def distribution(values):
    ordered = sorted(values)
    i = .95 * (len(values) - 1)
    lo = int(i)
    return dict(p50_ms=statistics.median(values),
                p95_ms=ordered[lo] + (ordered[min(lo+1, len(values)-1)]-ordered[lo])*(i-lo),
                samples_ms=values)


def make_case(name):
    import torch
    from reloc_torch import CompilerClient
    from reloc_torch.recipe import Cast, Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol, dense_strides

    def spec(shape, dtype):
        return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    a, b, c = (Symbol('s'+str(i)) for i in range(3))
    shape = (128, 128, 1024) if name == 'transpose' else ((65539,) if name == 'small' else (16777219,))
    source_shape = (a,b,c) if name == 'transpose' else (a,)
    dest_shape = (b,a,c) if name == 'transpose' else (a,)
    operations = ([Transpose((1,0,2))] if name == 'transpose' else []) + [Cast('float16','ieee_rne')]
    if name == 'split': operations.append(Cast('float32','exact'))
    recipe = CompilerClient.from_environment().compile(Recipe(spec(source_shape,'float32'),
        tuple(operations), spec(dest_shape,'float32' if name == 'split' else 'float16'),'h2d'))
    # Nontrivial, finite values make the intermediate f16 rounding observable.
    source = torch.randn(shape)
    def host_transform(x):
        if name == 'transpose': x = x.transpose(0,1)
        return x.to(torch.float16).contiguous()
    return recipe, source, host_transform


def metadata():
    import torch, pyreloc, reloc_torch
    build = Path(pyreloc.__file__).parents[2]
    paths = [Path(__file__), Path(os.environ['SYM_RELOC_EXPORT']),
             build/'libreloc/libreloc_runtime.so',
             *Path(pyreloc.__file__).parent.glob('_pyreloc*.so')]
    pyfiles = sorted(Path(reloc_torch.__file__).parent.glob('*.py'))
    return dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime_revision=os.environ.get('SYM_RUNTIME_REVISION','see binary hashes'),
        runtime_package=reloc_torch.__file__, torch=torch.__version__, cuda=torch.version.cuda,
        affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem',
                                     '--format=csv,noheader'],text=True).splitlines(),
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        frontend_sha256=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in pyfiles)).hexdigest())


def main():
    import torch
    from reloc_torch import CompiledRecipe, TransferResources, dispatch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',nargs='+',choices=CASES,default=CASES)
    p.add_argument('--paths',nargs='+',default=['whole','ring1','ring2','ring4','torch','torch_pinned','inductor'],
                   choices=['legacy','whole','ring1','ring2','ring4','torch','torch_pinned','inductor'])
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--chunk-size',type=int,default=0,help='zero selects the runtime default')
    p.add_argument('--pinning',choices=['pinned','pageable','auto'],default='pinned')
    p.add_argument('--min-pinned-bytes',type=int)
    p.add_argument('--trace',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); torch.manual_seed(221)
    torch.cuda.init()
    result = {'metadata':metadata(), 'configuration':vars(args)|{'output':str(args.output)}, 'cases':{}}

    def completed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        out = fn()
        torch.cuda.synchronize()
        return out, (time.perf_counter_ns()-start)/1e6

    with torch.no_grad():
        for name in args.case:
            recipe, source, host_transform = make_case(name)
            compile_ms = None
            if 'inductor' in args.paths:
                compiled_host = torch.compile(host_transform,fullgraph=True)
                start = time.perf_counter_ns(); compiled_host(source)
                compile_ms = (time.perf_counter_ns()-start)/1e6
            def torch_transfer(host):
                wire = host(source)
                out = wire.to('cuda:0',non_blocking=True)
                return out.float() if name == 'split' else out
            pinned = None
            def torch_pinned():
                nonlocal pinned
                logical = source.transpose(0,1) if name == 'transpose' else source
                if pinned is None:
                    pinned = torch.empty(logical.shape,dtype=torch.float16,pin_memory=True)
                pinned.copy_(logical)
                out = pinned.to('cuda:0',non_blocking=True)
                return out.float() if name == 'split' else out
            expected = host_transform(source)
            if name == 'split': expected = expected.float()
            def check(out, oracle):
                assert torch.equal(out.cpu().view(torch.uint8),oracle.view(torch.uint8)), name
            latest = {}
            with ExitStack() as stack:
                owners = {path:stack.enter_context(TransferResources(max_typed_retained_bytes=128<<20,
                          max_typed_live_bytes=128<<20, max_typed_background_workers=args.threads-1))
                          for path in args.paths if path not in ('torch','torch_pinned','inductor')}
                recipes = {path:CompiledRecipe.from_bytes(recipe.to_bytes()) for path in owners}
                def sym(path):
                    request = dispatch.prepare_typed_transfer(recipes[path],source,'cuda:0',threads=args.threads,
                        implementation='cpu_stages_cuda_stages@1' if name == 'split' else 'cpu_reference')
                    options = dict(n_buffers=int(path[-1]) if path.startswith('ring') else 2,
                        n_streams=1, gather_threads=args.threads, pinning=args.pinning,
                        min_pinned_bytes=args.min_pinned_bytes, resources=owners[path])
                    if path != 'legacy': options.update(pipeline=path!='whole',chunk_size=args.chunk_size or None)
                    output = dispatch.execute_typed_transfer(request,**options)
                    latest[path] = output.report
                    return output.tensor
                methods = {path:(lambda path=path:sym(path)) if path in owners else
                           (lambda:torch_transfer(host_transform)) if path=='torch' else
                           torch_pinned if path=='torch_pinned' else
                           (lambda:torch_transfer(compiled_host)) for path in args.paths}
                row = {'source_bytes':source.numel()*4,'wire_bytes':source.numel()*2,
                       'result_bytes':expected.numel()*expected.element_size(),
                       'recipe_sha256':hashlib.sha256(recipe.to_bytes()).hexdigest(),
                       'inductor_compile_and_prime_ms':compile_ms,'paths':{}}
                for path, fn in methods.items():
                    out, elapsed = completed(fn); check(out,expected)
                    row['paths'][path] = {'first_completed_ms':elapsed}
                for _ in range(8):
                    for fn in methods.values(): check(completed(fn)[0],expected)
                times = {path:[] for path in methods}; saved = {}
                for i in range(args.samples):
                    source.neg_()
                    expected = host_transform(source)
                    if name == 'split': expected = expected.float()
                    order = list(methods); order = order[i%len(order):]+order[:i%len(order)]
                    for path in order:
                        out, elapsed = completed(methods[path]); check(out,expected)
                        if path in saved: check(*saved[path])
                        saved[path] = (out,expected)
                        times[path].append(elapsed)
                for path, fn in methods.items():
                    out, _ = completed(fn); check(out,expected)
                    row['paths'][path].update(distribution(times[path]),
                        resources=owners[path].stats()['typed'] if path in owners else None,
                        torch_retained_host_bytes=source.numel()*2 if path=='torch_pinned' else 0,
                        report=dict(latest[path]) if path in owners else None)
                if args.trace:
                    torch.cuda.profiler.start()
                    for path, fn in methods.items():
                        for _ in range(5):
                            torch.cuda.nvtx.range_push('typed221/'+name+'/'+path)
                            pending = fn(); torch.cuda.synchronize()
                            torch.cuda.nvtx.range_pop()
                            out = pending; check(out,expected)
                    torch.cuda.profiler.stop()
                result['cases'][name] = row
                print(name,{path:(round(v['p50_ms'],3),round(v['p95_ms'],3))
                            for path,v in row['paths'].items()},flush=True)
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__ == '__main__': main()
