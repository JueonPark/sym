"""Fresh process model/supplement execution; diagnostics remain separate."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bench.issue269 import contract
from bench.issue266.support import load_inputs


def profile(args,result,torch,call):
    path=args.output.with_suffix('.profile.json')
    with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as captured:
        for _ in range(10):call()
        torch.cuda.synchronize()
    captured.export_chrome_trace(str(path))
    events=json.loads(path.read_text())['traceEvents']
    count=Counter(e.get('name') for e in events)
    info=dict(requests=10,scope='separate diagnostic; no acceptance timings',
        launches={n:count[n] for n in ('cudaGraphLaunch','cudaLaunchKernel','cuLaunchKernel','cudaLaunchKernelExC')},
        kernels=dict(Counter(e['name'] for e in events if e.get('cat')=='kernel')),
        copies=[dict(name=e['name'],args=e.get('args',{})) for e in events if e.get('cat')=='gpu_memcpy'],
        allocation_events=[e.get('args',{}) for e in events if e.get('name')=='[memory]'])
    compressed=path.with_suffix('.json.gz');compressed.write_bytes(gzip.compress(path.read_bytes()));path.unlink()
    info.update(file=compressed.name,sha256=contract.sha(compressed));result['profile']=info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    for name in ('runtime','artifact','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.profile and not args.pilot:parser.error('profiles are diagnostics only')
    job=next(j for j in contract.jobs() if j['id']==args.job)
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=load_inputs(args.artifact);torch.cuda.init();torch.cuda.mem_get_info()
    result=dict(job=job,protocol=contract.protocol(job),environment=contract.environment(torch,job),phases=[],
        provenance=contract.provenance(args.runtime,args.artifact,Path(os.environ['SYM_RELOC_EXPORT'])))
    if args.pilot:result['pilot']=True
    sampler=inputs.memory.Sampler();result['initial_nvml']=sampler.device_info()
    elapsed=lambda:(time.perf_counter_ns()-sampler.start_ns)/1e6
    try:
        if job['kind']=='model':
            from bench.issue269.model import run
        else:
            from bench.issue269.micro import run
        run(args,job,result,inputs,torch,elapsed)
    except Exception as error:
        if isinstance(error,torch.OutOfMemoryError) or ('out of memory' in str(error).lower() and 'cuda' in str(error).lower()):
            result.update(outcome='oom',error=repr(error))
        else:
            result.update(outcome='invalid',error=repr(error));raise
    finally:
        torch.cuda.synchronize();trace=sampler.stop();path=args.output.with_suffix('.memory.json.gz')
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as stream:json.dump(trace,stream,separators=(',',':'))
        result['memory']=dict(file=path.name,sha256=contract.sha(path),scope=trace['scope'])
        contract.save(args.output,result)
    print(job['id'],result['outcome'],flush=True)


if __name__=='__main__':main()
