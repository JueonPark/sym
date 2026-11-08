"""Supplemental replay/shape/budget sweep and independent launch profiling."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bench.issue268 import contract
from bench.issue268.run import quiet
from bench.issue266.isolation import audit


def kernel(x,p):
    for _ in range(8): x=(x*p).sin()+.01
    return x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compute',choices=('eager','inductor'),required=True)
    parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true'); parser.add_argument('--profile',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    preflight=quiet(True,args.pilot)
    observed=[]; stop=threading.Event()
    def watch():
        while not stop.is_set():
            observed.append(audit({os.getpid()})); stop.wait(3)
    watcher=threading.Thread(target=watch); watcher.start()
    import torch
    from reloc_torch import GPUReplay,GPUReplayPolicy
    from bench.issue266.support import load_inputs
    inputs=load_inputs(args.artifact)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.init(); torch.cuda.mem_get_info()
    result=dict(compute=args.compute,round=args.round,qualification=not args.pilot,rows=[],preflight=preflight,
                environment=contract.environment(torch,dict(callers=1)),
                provenance=contract.provenance(args.runtime,args.artifact,Path(os.environ['SYM_RELOC_EXPORT'])))
    sampler=inputs.memory.Sampler()
    elapsed=lambda:(time.perf_counter_ns()-sampler.start_ns)/1e6
    try:
        with torch.no_grad():
            for scenario in ('stable','varying','pressure'):
                modes=list(contract.MODES); shift=args.round%3; modes=modes[shift:]+modes[:shift]
                for mode in modes:
                    torch._dynamo.reset()
                    fn=kernel if args.compute=='eager' else torch.compile(kernel,fullgraph=True,
                           options={'triton.cudagraphs':False})
                    owner=None
                    if mode!='plain':
                        owner=GPUReplay(fn,device='cuda:0',policy=GPUReplayPolicy(enabled=mode=='on',slots=2,
                            max_buckets=2,max_bytes=(1<<20) if scenario=='pressure' else 32<<20))
                    x=torch.ones(1024,device='cuda:0'); p=torch.full_like(x,.5)
                    row=dict(scenario=scenario,mode=mode,completed_ms=[],submission_ms=[],checks=0,retained_checks=0)
                    row['prepare_start_ms']=elapsed(); start=time.perf_counter_ns()
                    if owner is not None: owner.prepare(x,p).freeze()
                    torch.cuda.synchronize(); row['preparation_ms']=(time.perf_counter_ns()-start)/1e6
                    row['prepare_end_ms']=elapsed()
                    call=owner if owner is not None else fn
                    row['first_start_ms']=elapsed(); start=time.perf_counter_ns(); first=call(x,p); torch.cuda.synchronize()
                    row['first_ms']=(time.perf_counter_ns()-start)/1e6
                    row['first_end_ms']=elapsed()
                    torch.testing.assert_close(first,kernel(x,p),rtol=2e-5,atol=2e-5)
                    for _ in range(5): call(x,p)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    row['before']=owner.stats() if owner else None; row['warm_start_ms']=elapsed()
                    previous=None
                    for index in range(5 if args.pilot else 100):
                        n=(1024*(1+index%4)) if scenario=='varying' else 1024
                        x=torch.full((n,),(-1 if index%2 else 1)*(1+index*.01),device='cuda:0')
                        p=torch.full_like(x,.4+index*.001)
                        reference=kernel(x,p); torch.cuda.synchronize()
                        start=time.perf_counter_ns(); actual=call(x,p)
                        submission=(time.perf_counter_ns()-start)/1e6; torch.cuda.synchronize()
                        row['completed_ms'].append((time.perf_counter_ns()-start)/1e6)
                        row['submission_ms'].append(submission)
                        torch.testing.assert_close(actual,reference,rtol=2e-5,atol=2e-5); row['checks']+=1
                        if previous:
                            assert torch.equal(previous[0],previous[1]); row['retained_checks']+=1
                        previous=(actual,actual.clone())
                    row['warm_end_ms']=elapsed(); row['memory']=inputs.memory.torch_stats(torch)
                    row['after']=owner.stats() if owner else None
                    if args.profile and scenario=='stable' and mode in ('off','on'):
                        trace=args.output/f'{mode}-trace.json'
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
                            for _ in range(10): call(x,p)
                            torch.cuda.synchronize()
                        profile.export_chrome_trace(str(trace))
                        events=json.loads(trace.read_text())['traceEvents']
                        counts={name:sum(e.get('name')==name for e in events) for name in
                                ('cudaGraphLaunch','cudaLaunchKernel','cuLaunchKernel','cudaLaunchKernelExC')}
                        counts['gpu_kernel_events']=sum(e.get('cat')=='kernel' for e in events)
                        counts['graph_replays']=sum('CUDAGraph::replay' in e.get('name','') for e in events)
                        compressed=trace.with_suffix('.json.gz'); compressed.write_bytes(gzip.compress(trace.read_bytes()))
                        trace.unlink(); row['profile']=dict(file=compressed.name,sha256=contract.sha(compressed),counts=counts)
                    if owner: owner.close()
                    assert torch.equal(previous[0],previous[1]); row['retained_checks']+=1
                    row['closed']=owner.stats() if owner else None
                    result['rows'].append(row); contract.save(args.output/'record.json',result)
                    del owner,call,fn,actual,reference,previous,first,x,p
                    torch.cuda.synchronize(); torch.cuda.empty_cache()  # Between separate preparations only.
    finally:
        stop.set(); watcher.join()
        result['audit']=dict(observations=len(observed),foreign_busy_processes=sorted(set(v for row in observed for v in row)))
        trace=sampler.stop(); path=args.output/'memory.json.gz'
        path.write_bytes(gzip.compress(json.dumps(trace,separators=(',',':')).encode()))
        result['memory']=dict(file=path.name,sha256=contract.sha(path))
        contract.save(args.output/'record.json',result)
    if result['audit']['foreign_busy_processes']: raise RuntimeError('contaminated supplement; preserve and rerun')


if __name__=='__main__': main()
