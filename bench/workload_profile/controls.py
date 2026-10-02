#!/usr/bin/env python3
"""Matched warm transfer controls and Python call attribution.

Prepared-before-timer rows intentionally exclude preparation: they are diagnostic
lower bounds, not end-to-end claims or proposals to remove validation. All calls
still use fresh single-use requests, fresh outputs and blocking completion.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import random
import statistics
import sys
import time

import torch
from reloc_torch import CompilerClient, RelocBackend, TransferResources, dispatch, transport
from reloc_torch.recipe import Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, Symbol, dense_strides


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workloads',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.workloads/'libreloc/python/examples/workloads'))
    import common
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(162)
    rng=random.Random(162)
    result=dict(scope=__doc__,rows=[],python_profiles={})
    def check(out,expected):
        assert torch.equal(out.cpu().contiguous().view(torch.uint8),expected.cpu().contiguous().view(torch.uint8))
    def measure(case,variants,expected,rounds=3,calls=40):
        for round_id in range(rounds):
            order=list(variants);rng.shuffle(order)
            for mode in order:
                fn,prepare=variants[mode]
                for _ in range(5):check(fn(prepare() if prepare else None),expected)
                samples=[]
                for _ in range(calls):
                    prepared=prepare() if prepare else None
                    torch.cuda.synchronize();start=time.perf_counter_ns()
                    out=fn(prepared);torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns()-start)/1e6)
                    check(out,expected)
                    del out,prepared
                result['rows'].append(dict(case=case,variant=mode,round=round_id,samples_ms=samples,median_ms=statistics.median(samples)))
        print(case,'checked',flush=True)
    def profile(case,fn,calls=60):
        for _ in range(5):fn();torch.cuda.synchronize()
        prof=cProfile.Profile(timer=time.thread_time);prof.enable();wall_start=time.perf_counter()
        for _ in range(calls):fn();torch.cuda.synchronize()
        wall_ms=(time.perf_counter()-wall_start)*1000;prof.disable();prof.dump_stats(str(args.output/(case+'.pstats')))
        stream=io.StringIO();stats=pstats.Stats(prof,stream=stream).sort_stats('cumulative');stats.print_stats(65)
        (args.output/(case+'-cprofile.txt')).write_text(stream.getvalue())
        rows=[]
        for (file,line,name),(primitive,total,self_s,cum_s,callers) in stats.stats.items():
            rows.append(dict(file=file,line=line,function=name,calls=total,primitive_calls=primitive,self_ms=self_s*1000,cumulative_ms=cum_s*1000))
        result['python_profiles'][case]=dict(calls=calls,timer='calling-thread CPU time (excludes descheduling and worker CPU)',profiled_wall_ms=wall_ms,profiled_total_ms=stats.total_tt*1000,
            functions=sorted(rows,key=lambda r:r['cumulative_ms'],reverse=True))
    with torch.no_grad():
        q=torch.randint(-127,128,(1024,4096),dtype=torch.int8);scale=torch.rand(4096)+.01
        fetcher=common.WeightFetcher(common.Report('diagnostic'),implementation='cuda_dequant_relocate')
        def prepare_weight():
            return dispatch.prepare_typed_transfer(fetcher.compiled,q,'cuda:0',parameters={'scale':scale},implementation='cuda_dequant_relocate')
        with TransferResources() as owner:
            def run_weight(prepared=None,resources=owner):
                req=prepared if prepared is not None else prepare_weight()
                return dispatch.execute_typed_transfer(req,resources=resources).tensor
            expected=fetcher.reference(q,scale,'cuda:0')
            measure('weight_4MiB',{
                'torch':(lambda _:fetcher.reference(q,scale,'cuda:0'),None),
                'sym_ephemeral':(lambda _:run_weight(resources=None),None),
                'sym_retained':(lambda _:run_weight(),None),
                'sym_prepared_outside_timer':(run_weight,prepare_weight)},expected)
            profile('weight-sym',run_weight)
            profile('weight-torch',lambda:fetcher.reference(q,scale,'cuda:0'))
            result['weight_resources']=owner.stats()['typed']
        for direction in ('h2d','d2h'):
            shape=(32,16,64) if direction=='h2d' else (16,32,64)
            destination='cuda:0' if direction=='h2d' else 'cpu'
            src=torch.randn(shape,device='cpu' if direction=='h2d' else 'cuda:0')
            def reference(x):return x.transpose(0,1).contiguous().to(destination)
            backend=RelocBackend();frontend=torch.compile(reference,backend=backend,dynamic=True)
            dims=tuple(Symbol(f's{i}') for i in range(3))
            def spec(extents):return TensorSpec(extents,dense_strides(extents),Const(0),'float32')
            recipe=Recipe(spec(dims),(Transpose((1,0,2)),),spec((dims[1],dims[0],dims[2])),direction)
            compiled=CompilerClient.from_environment().compile(recipe)
            try:
                with TransferResources() as owner:
                    def prepare():return transport.prepare_transfer(compiled,src,destination)
                    def direct(prepared=None):
                        return transport.execute_transfer(prepared if prepared is not None else prepare(),resources=owner)
                    expected=reference(src)
                    measure('kv_'+direction,{
                        'torch':(lambda _:reference(src),None),
                        'sym_frontend':(lambda _:frontend(src),None),
                        'sym_direct':(lambda _:direct(),None),
                        'sym_prepared_outside_timer':(direct,prepare)},expected)
                    profile('kv-'+direction+'-sym-frontend',lambda:frontend(src))
                    profile('kv-'+direction+'-sym-direct',direct)
                    profile('kv-'+direction+'-torch',lambda:reference(src))
                    if direction=='h2d':
                        # Isolate CPU transpose thread-pool cost, changing only Torch's thread count.
                        torch.set_num_threads(1)
                        measure('kv_h2d_torch_one_thread',{'torch':(lambda _:reference(src),None)},expected)
                        torch.set_num_threads(8)
                    snapshot=backend.stats();assert not snapshot['fallbacks'] and not snapshot['exclusions']
                    result['kv_'+direction+'_stats']=dict(snapshot)
            finally:backend.close()
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,default=dict)+'\n')


if __name__=='__main__':main()
