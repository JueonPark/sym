#!/usr/bin/env python3
"""Completed consumer-sized groups, individual Sym and asynchronously batched Torch.

Output allocation and final completion are timed. Input mutations, oracle checks,
compiler construction and diagnostic memory/counter samples are outside timing.
No packing of source payloads or aliasing of logical outputs is performed.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def distribution(values):
    ordered = sorted(values)
    index = .95 * (len(ordered) - 1)
    low = int(index)
    return {'p50_ms': statistics.median(values),
            'p95_ms': ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(index-low),
            'samples_ms': values}


def compiled_recipes():
    from reloc_torch import CompilerClient
    from reloc_torch.recipe import BindingParam, Dequantize, Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol, dense_strides

    compiler = CompilerClient.from_environment()
    def spec(shape, dtype):
        return TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    a, b, c = (Symbol('s'+str(i)) for i in range(3))
    recipes = {direction: compiler.compile(Recipe(spec((a,b,c), 'float32'),
               (Transpose((1,0,2)),), spec((b,a,c), 'float32'), direction)) for direction in ('h2d','d2h')}
    recipes['weight'] = compiler.compile(Recipe(spec((a,b), 'int8'),
        (Dequantize('float32', BindingParam('scale','float32',(b,)), None, 1, 'affine'),
         Transpose((1,0))), spec((b,a),'float32'), 'h2d'))
    return recipes


CASES = {
    'kv_h2d_small': [('h2d',(16,33,64))]*8,
    'kv_d2h_small': [('d2h',(16,33,64))]*8,
    'kv_h2d_large': [('h2d',(16,1024,64))]*4,
    'kv_d2h_large': [('d2h',(16,1024,64))]*4,
    'weights_small': [('weight',(128,128))]*8,
    'weights_large': [('weight',(1024,4096))]*4,
    'heterogeneous': [('h2d',(3,5,64)), ('weight',(256,512)), ('d2h',(16,33,64)),
                      ('weight',(128,512)), ('h2d',(8,257,32)), ('weight',(512,1024))],
}


def make_case(name, recipes):
    import torch
    scales = {}
    items = []
    for kind, shape in CASES[name]:
        if kind == 'weight':
            source = torch.randint(-127,128,shape,dtype=torch.int8)
            scale = scales.setdefault(shape[1], torch.full((shape[1],), .5))
        else:
            source = torch.randn(shape,device='cuda:0' if kind == 'd2h' else 'cpu')
            scale = None
        items.append((kind, source, scale))
    def change(i):
        for kind, source, _ in items:
            source.neg_() if kind == 'weight' else source.add_(1)
        for scale in scales.values():
            scale.fill_(.25 if i%2 else .5)
    def oracle():
        return tuple(((source.float()*scale).t().contiguous() if kind == 'weight' else
                      source.transpose(0,1).contiguous().cpu()) for kind,source,scale in items)
    return items, change, oracle


def prepare(items, recipes, calibration, threads):
    from reloc_torch import dispatch, transport
    return (dispatch.prepare_typed_transfer(recipes[kind], src, 'cuda:0',
                parameters={'scale': scale}, calibration=calibration, threads=threads)
            if kind == 'weight' else transport.prepare_transfer(recipes[kind], src,
                'cuda:0' if kind == 'h2d' else 'cpu') for kind,src,scale in items)


def torch_batch(items):
    import torch
    # One upload per shared scale tensor in this invocation. No value is cached
    # between invocations, and the input list owns every source until completion.
    uploaded, outputs = {}, []
    for kind, source, scale in items:
        if kind == 'weight':
            if id(scale) not in uploaded:
                uploaded[id(scale)] = scale.to('cuda:0',non_blocking=True)
            out = (source.to('cuda:0',non_blocking=True).float()*uploaded[id(scale)]).t().contiguous()
        else:
            out = source.transpose(0,1).contiguous().to('cuda:0' if kind == 'h2d' else 'cpu',non_blocking=True)
        outputs.append(out)
    return tuple(outputs)


def metadata():
    import torch, pyreloc, reloc_torch
    paths = [Path(__file__), ROOT/'calibration/epyc7351-2080ti.cal',
             *Path(reloc_torch.__file__).parent.glob('*.py'), *Path(pyreloc.__file__).parent.glob('*.py'),
             *Path(pyreloc.__file__).parent.glob('_pyreloc*.so'),
             *Path(pyreloc.__file__).parents[2].glob('libreloc/libreloc_runtime.so')]
    paths += [Path(os.environ[name]) for name in ('SYM_RELOC_EXPORT','SYM_OPT')]
    return {'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_changes':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'runtime_revision':os.environ.get('SYM_RUNTIME_REVISION','unspecified; see hashes'),
            'runtime_package':reloc_torch.__file__, 'python':sys.version,'torch':torch.__version__,
            'torch_cuda':torch.version.cuda,'affinity':sorted(os.sched_getaffinity(0)),
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,clocks.sm,clocks.mem',
                        '--format=csv,noheader'],text=True).splitlines(),
            'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def main():
    import torch, pyreloc
    from reloc_torch import TransferResources, dispatch, transport
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--individual-only',action='store_true',help='current-main control')
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    if args.samples < 1: p.error('samples must be positive')
    if args.trace and len(args.case) != 1: p.error('trace one case per process')
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(220)
    calibration=pyreloc.load_calibration(str(ROOT/'calibration/epyc7351-2080ti.cal'))
    result={'metadata':metadata(),'cases':{},'scope':'Completed transfer groups including allocation and synchronization; not model inference. Eight warmups, changed inputs/scales, fixed group topology. Recipe compilation is outside first-call timing. Torch executes member operations on one current stream with non_blocking copies, shared-scale uploads and a single final synchronization. Raw rounds are independent processes.'}
    def completed(fn):
        torch.cuda.synchronize()
        begin=time.perf_counter_ns()
        out=fn()
        torch.cuda.synchronize()
        return out,(time.perf_counter_ns()-begin)/1e6
    def check(outputs,expected):
        assert len(outputs)==len(expected)
        assert all(torch.equal(a.cpu(),b) for a,b in zip(outputs,expected))
    with torch.no_grad():
        for name in args.case:
            recipes=compiled_recipes()
            # Independent caches: the first individual call must not prewarm
            # the first grouped call's binding/selection templates.
            from reloc_torch import CompiledRecipe
            grouped_recipes={key:CompiledRecipe.from_bytes(recipe.to_bytes()) for key,recipe in recipes.items()}
            items,change,oracle=make_case(name,recipes)
            with ExitStack() as stack:
                owners={path:stack.enter_context(TransferResources(max_retained_bytes=64<<20,
                    max_live_staging_bytes=64<<20,max_background_workers=7,
                    max_typed_retained_bytes=64<<20,max_typed_live_bytes=64<<20,
                    max_typed_background_workers=7)) for path in ('individual','grouped') if path != 'grouped' or not args.individual_only}
                latest={}
                def individual():
                    outputs=[];reports=[]
                    for request in prepare(items,recipes,calibration,8):
                        if hasattr(request,'template'):
                            out=dispatch.execute_typed_transfer(request,resources=owners['individual'],n_streams=1,gather_threads=8)
                            outputs.append(out.tensor);reports.append(dict(out.report))
                        else:
                            outputs.append(transport.execute_transfer(request,resources=owners['individual'],n_streams=1,n_buffers=1,gather_threads=8))
                    latest['individual']=reports
                    return tuple(outputs)
                def grouped():
                    from reloc_torch import prepare_transfer_group,execute_transfer_group
                    out=execute_transfer_group(prepare_transfer_group(prepare(items,grouped_recipes,calibration,8)),
                        resources=owners['grouped'],gather_threads=8,max_scratch_bytes=64<<20)
                    latest['grouped']=dict(out.report)
                    return out.tensors
                methods={'individual':individual,'torch_batched':lambda:torch_batch(items)}
                if not args.individual_only: methods['grouped']=grouped
                row={'members':[{'kind':kind,'shape':list(src.shape),'source_bytes':src.numel()*src.element_size()} for kind,src,_ in items],
                     'first_completed_ms':{},'warm':{},'diagnostics':{}}
                expected=oracle()
                for path,fn in methods.items():
                    outputs,elapsed=completed(fn);check(outputs,expected)
                    row['first_completed_ms'][path]=elapsed
                for _ in range(8):
                    for fn in methods.values(): check(completed(fn)[0],expected)
                saved={};times={path:[] for path in methods}
                for i in range(args.samples):
                    change(i);expected=oracle()
                    order=list(methods)
                    order=order[i%len(order):]+order[:i%len(order)]
                    for path in order:
                        outputs,elapsed=completed(methods[path]);check(outputs,expected)
                        if path in saved: check(*saved[path])
                        saved[path]=(outputs,tuple(e.clone() for e in expected))
                        times[path].append(elapsed)
                row['warm']={path:distribution(values) for path,values in times.items()}
                for path,fn in methods.items():
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    before=torch.cuda.memory_allocated()
                    before_stats=owners[path].stats() if path in owners else None
                    outputs,_=completed(fn);check(outputs,expected)
                    row['diagnostics'][path]={
                        'torch_peak_extra_allocated_bytes':torch.cuda.max_memory_allocated()-before,
                        'torch_output_bytes':sum(out.numel()*out.element_size() for out in outputs if out.is_cuda),
                        'resources_before':before_stats,'resources_after':owners[path].stats() if path in owners else None,
                        'report':latest.get(path), 'packing_bytes':0}
                if args.trace:
                    torch.cuda.profiler.start()
                    for path,fn in methods.items():
                        for _ in range(5):
                            torch.cuda.nvtx.range_push('group220/'+name+'/'+path)
                            pending=fn();torch.cuda.synchronize()
                            torch.cuda.nvtx.range_pop()
                            # Keep the preceding output alive throughout the
                            # range. Releasing a Torch pinned CPU output can
                            # record allocator events; attribute that retirement
                            # outside the next implementation's call.
                            outputs=pending
                            check(outputs,expected)
                    torch.cuda.profiler.stop()
                row['correctness']=True
                result['cases'][name]=row
                print(name,{path:(d['p50_ms'],d['p95_ms']) for path,d in row['warm'].items()},flush=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
