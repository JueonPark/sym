"""Fresh-process matched E2/E5 arm; original numerical functions are unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.issue268 import contract
from bench.issue266.support import load_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args=parser.parse_args()
    if args.profile and not args.pilot: parser.error('profiling is diagnostic only')
    job=next(j for j in contract.jobs() if j['id']==args.job)
    import torch
    from reloc_torch import GPUReplay, GPUReplayPolicy
    inputs=load_inputs(args.artifact)
    torch.set_num_threads(8//job['callers']); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    inputs.e2.legacy.COMPUTE_DTYPE='float16'
    torch.cuda.init(); torch.manual_seed(225)
    result=dict(job=job, protocol=contract.protocol(job), environment=contract.environment(torch,job),
                provenance=contract.provenance(args.runtime,args.artifact,Path(os.environ['SYM_RELOC_EXPORT'])))
    if args.pilot: result['pilot']=True
    # Materialize the runtime context before NVML baseline/reservation, as E5 does.
    result["cuda_context_baseline_free_total"]=torch.cuda.mem_get_info()
    sampler=inputs.memory.Sampler()
    elapsed=lambda: (time.perf_counter_ns()-sampler.start_ns)/1e6
    reserve=None; snapshots=None; workers=[]; executors=[]; previous=[]
    result['initial_nvml']=sampler.device_info()
    if job['kind']=='e5':
        reserve=inputs.memory.Reserve(1024<<20)
        result['budget']=reserve.stats()
        result['nvml_after_reservation']=sampler.device_info()
        assert abs(result['nvml_after_reservation']['used']-result['initial_nvml']['used']-reserve.bytes)<=16<<20
        assert abs(result['nvml_after_reservation']['free']-reserve.after['free'])<=16<<20
    # E2 uses its original larger prefill/dense shapes and gather budget.
    if job['kind']=='e2':
        original_fast=inputs.workload.FastTransfers
        def fast(*a,**kw):
            kw['gather_threads']=8
            return original_fast(*a,**kw)
        inputs.workload.FastTransfers=fast

    class Region:
        def __init__(self,fn,index):
            budget=(256<<20) if job['kind']=='e2' else ((192 if index==0 else 32)<<20)//job['callers']
            # MoE's expert is the large region; route receives the small share.
            if job['kind']=='e5' and job['case']=='moe_sparse':
                budget=((192 if index==1 else 32)<<20)//job['callers']
            self.owner=GPUReplay(fn, device='cuda:0', policy=GPUReplayPolicy(
                enabled=job['mode']=='on', slots=2,max_buckets=8,max_bytes=budget))
            self.frozen=False
        def __call__(self,*a,**kw):
            if not self.frozen:
                torch.cuda.synchronize()
                self.owner.prepare(*a,**kw)
            return self.owner(*a,**kw)
        def freeze(self):
            self.owner.freeze(); self.frozen=True

    class Worker(inputs.workload.Worker):
        def __init__(self,core,snapshots):
            super().__init__(core,snapshots,compute=job['compute'],scratch_bytes=(128<<20)//job['callers'])
            self.regions=[]
            if job['mode']!='plain':
                functions=[]
                for index, fn in enumerate(self.functions):
                    if core.case=='moe_sparse' and index==2:
                        functions.append(fn)  # original combine mutates its input
                    else:
                        region=Region(fn,index); self.regions.append(region)
                        self.life.callback(region.owner.close); functions.append(region)
                self.functions=tuple(functions)
                self.set_input(core.x)
            self.submission_ms=0.
        def request(self,x=None):
            with torch.cuda.stream(self.stream):
                self.stream.synchronize(); start=time.perf_counter_ns()
                if x is not None: self.set_input(x)
                out=self.fn()
                self.submission_ms=(time.perf_counter_ns()-start)/1e6
                self.stream.synchronize()
            return out,(time.perf_counter_ns()-start)/1e6

    def replay_stats():
        return [r.owner.stats() for w in workers for r in w.regions]
    def check(out,ref):
        return inputs.workload.check(out,ref)
    def exact(out,ref):
        inputs.workload.exact_retained(out,ref)
    def cpu(out):
        return inputs.workload.cpu_result(out)

    result['phase']='fixture'
    try:
        with torch.no_grad():
            if job['kind']=='e2':
                model,x,sizes,pairs,reference=inputs.e2.make_model(job['case'],torch.float16)
                names={id(q):str(i) for i,(q,s) in enumerate(pairs)}
                llm=job['case'].startswith('llm')
                groups=([[names[id(layer[k][0])] for k in inputs.e2.legacy.MATRICES] for layer in model.layers] if llm else
                        [[names[id(expert[k][0])] for k in ('w1','w2')] for block in model.experts for expert in block])
                core=SimpleNamespace(case='llm_decode' if llm else 'moe_sparse',model=model,x=x,sizes=sizes,
                                     groups=groups,pairs=pairs,names=names,by_name={names[id(q)]:(q,s) for q,s in pairs})
                values=[x]; references=[cpu(reference)]; del reference
            else:
                core=inputs.workload.Model(job['case'])
                values=[core.input(8 if job['case']=='llm_decode' else 1) for _ in range(8)]
                references=[cpu(core.oracle(x)) for x in values]
            torch.cuda.synchronize()
            result['input_sha256']=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in values]
            result['sizes']=core.sizes
        result['phase']='prepare'; start=time.perf_counter_ns()
        with torch.no_grad(): snapshots=inputs.workload.Snapshots(core,job['runtime'])
        result['snapshot_ms']=(time.perf_counter_ns()-start)/1e6
        preparations=[]; first=[]; warmups=[]
        for i in range(job['callers']):
            executor=ThreadPoolExecutor(1); executors.append(executor)
            def initialize(i=i):
                span=8//job['callers']; os.sched_setaffinity(0,contract.CALLER_CPUS[i*span:(i+1)*span])
                torch.cuda.set_device(0)
                with torch.no_grad():
                    start=time.perf_counter_ns(); worker=Worker(core,snapshots)
                    workers.append(worker); preparations.append((time.perf_counter_ns()-start)/1e6)
                    result['phase']='first'; out,cost=worker.request(values[i if job['kind']=='e5' else 0])
                    first.append(cost); check(out,references[i if job['kind']=='e5' else 0])
                    result['phase']='warmup'; start=time.perf_counter_ns()
                    for _ in range(3):
                        for ordinal in range(i,len(values),job['callers']):
                            out,_=worker.request(values[ordinal]); check(out,references[ordinal])
                    warmups.append((time.perf_counter_ns()-start)/1e6)
                    for region in worker.regions: region.freeze()
                return worker
            executor.submit(initialize).result()
        result.update(preparation_ms=result['snapshot_ms']+sum(preparations), first_ms=sum(first),
                      worker_prepare_ms=preparations,worker_first_ms=first,warmup_ms=warmups,
                      replay_before=replay_stats(), compile_before=[w.state.stats() for w in workers])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        result['phase']='warm'; result['warm_start_ms']=elapsed()
        result.update(completed_ms=[],submission_ms=[],requests=[],checks=0,retained_checks=0)
        samples=2 if args.pilot else result['protocol']['samples']
        if job['kind']=='e2':
            for index in range(samples):
                def call():
                    with torch.no_grad():
                        out,cost=workers[0].request(values[0]); return out,cost,workers[0].submission_ms
                out,cost,submission=executors[0].submit(call).result()
                result['completed_ms'].append(cost); result['submission_ms'].append(submission)
                check(out,references[0]); result['checks']+=1
                if previous:
                    exact(*previous[0]); result['retained_checks']+=1
                previous=[(out,cpu(out))]
        else:
            def call(i,offered,batch,ordinal):
                with torch.no_grad():
                    started=time.perf_counter_ns(); out,cost=workers[i].request(values[ordinal]); done=time.perf_counter_ns()
                return out,dict(caller=i,batch=batch,ordinal=ordinal,service_ms=cost,
                    cpu_submission_ms=workers[i].submission_ms,queue_ms=(started-offered)/1e6,sojourn_ms=(done-offered)/1e6)
            for batch in range(samples):
                start=time.perf_counter_ns(); futures=[]
                for ordinal in range(8):
                    futures.append(executors[ordinal%job['callers']].submit(call,ordinal%job['callers'],
                                   time.perf_counter_ns(),batch,ordinal))
                submission=(time.perf_counter_ns()-start)/1e6
                finished=[f.result() for f in futures]
                result['completed_ms'].append((time.perf_counter_ns()-start)/1e6)
                result['submission_ms'].append(submission)
                for held,ref in previous: exact(held,ref); result['retained_checks']+=1
                previous=[]
                for out,record in finished:
                    record['max_abs_error']=check(out,references[record['ordinal']]); result['checks']+=1
                    result['requests'].append(record); previous.append((out,cpu(out)))
        result['warm_end_ms']=elapsed()
        result['torch_memory']=inputs.memory.torch_stats(torch)
        result['replay_after']=replay_stats()
        result['compile_after']=[w.state.stats() for w in workers]
        result['transfer_after']=[w.stats() for w in workers]
        if args.profile:
            from collections import Counter
            trace_path=args.output.with_suffix('.profile.json')
            def profile_requests():
                with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                    for _ in range(10):
                        actual,_=workers[0].request(values[0])
                    torch.cuda.synchronize()
                check(actual,references[0])
                profile.export_chrome_trace(str(trace_path))
            executors[0].submit(profile_requests).result()
            events=json.loads(trace_path.read_text())['traceEvents']
            counts={name:sum(e.get('name')==name for e in events) for name in
                    ('cudaGraphLaunch','cudaLaunchKernel','cuLaunchKernel','cudaLaunchKernelExC')}
            kernels=dict(Counter(e['name'] for e in events if e.get('cat')=='kernel'))
            compressed=trace_path.with_suffix('.json.gz')
            compressed.write_bytes(gzip.compress(trace_path.read_bytes())); trace_path.unlink()
            result['profile']=dict(file=compressed.name,sha256=contract.sha(compressed),counts=counts,
                                   gpu_kernels=kernels,requests=10,scope='separate diagnostic; timings excluded')
        start=time.perf_counter_ns()
        for executor,worker in zip(executors,workers): executor.submit(worker.close).result()
        snapshots.close(); snapshots=None
        result['retirement_ms']=(time.perf_counter_ns()-start)/1e6
        for held,ref in previous: exact(held,ref); result['retained_checks']+=1
        result['held_after_close']=len(previous)
        result['replay_closed']=replay_stats()
        result['torch_held_after_close']=inputs.memory.torch_stats(torch)
        result['outcome']='success'
    except Exception as error:
        if result['phase']!='fixture' and (isinstance(error,torch.OutOfMemoryError) or
            ('out of memory' in str(error).lower() and 'cuda' in str(error).lower())):
            result.update(outcome='oom',error=repr(error))
        else:
            result.update(outcome='invalid',error=repr(error))
            raise
    finally:
        for executor,worker in zip(executors,workers):
            if worker.core is not None: executor.submit(worker.close).result()
        for executor in executors: executor.shutdown(wait=True)
        if snapshots is not None: snapshots.close()
        torch.cuda.synchronize()
        trace=sampler.stop(); path=args.output.with_suffix('.memory.json.gz')
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as stream: json.dump(trace,stream,separators=(',',':'))
        result['memory']=dict(file=path.name,sha256=contract.sha(path),scope=trace['scope'])
        if reserve is not None: reserve.close()
        contract.save(args.output,result)
    print(job['id'],result['outcome'],flush=True)


if __name__=='__main__': main()
