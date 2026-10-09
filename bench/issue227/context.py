#!/usr/bin/env python3
"""Held-out hardware/context: conservative fallback and load-specific costs."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import torch
from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend, TransferResources
from reloc_torch import placement as policy_module
from placement import CONTEXT, LIMITS, OPTIONS, case, check, measure, summary


@contextmanager
def load(kind,device):
    stop = threading.Event()
    metrics = dict(gpu_iterations=0, cpu_worker_started=False)
    process = thread = None
    if kind == 'cpu_busy':
        process = subprocess.Popen([sys.executable,'-c',
            'import numpy as np; a=np.ones(2**20); b=a.copy();\nwhile True: np.add(a,b,out=a)'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        metrics['cpu_worker_started'] = True
    elif kind == 'gpu_busy':
        def work():
            with torch.no_grad(),torch.cuda.device(device),torch.cuda.stream(torch.cuda.Stream(device=device)):
                a=torch.randn(2048,2048,device=device)
                b=torch.empty_like(a)
                while not stop.is_set():
                    torch.mm(a,a,out=b)
                    torch.cuda.current_stream(device).synchronize()
                    metrics['gpu_iterations'] += 1
        thread=threading.Thread(target=work)
        thread.start()
    try:
        time.sleep(.5)
        yield metrics
    finally:
        stop.set()
        if thread:
            thread.join()
        if process:
            assert process.poll() is None, 'controlled CPU worker stopped unexpectedly'
            process.terminate()
            process.wait()
        if kind == 'gpu_busy':
            assert metrics['gpu_iterations'] > 0, 'controlled GPU worker did not execute'


def run(args):
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(227)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    context=dict(CONTEXT)
    policy=PlacementPolicy(PlacementProfile.load(args.profile),context=lambda:context)
    _,original,transform,control=case('h2d',True,False,device)
    result=dict(hardware=policy_module.hardware(device),scenario=args.scenario,rows=[])
    with TransferResources(**LIMITS) as resources:
        backend=RelocBackend(placement=policy,transfer_resources=resources,transfer_options=OPTIONS)
        compiled=torch.compile(original,backend=backend,fullgraph=True,dynamic=True)
        paths=['torch_cpu','torch_gpu','sym_cpu','sym_gpu']
        x=torch.randn(128,512)
        expected=original(x)
        # Precompile before introducing controlled load.
        for n in (64,256,128):
            with policy.force('torch_cpu'):
                compiled(torch.randn(n,512))
        if args.scenario in ('cpu_busy','gpu_busy'):
            context['cpu_load' if args.scenario=='cpu_busy' else 'gpu_load']='busy'
        elif args.scenario=='overlap':
            context['overlap']='consumer'
        with load(args.scenario,device) as metrics:
            result['controlled_load'] = metrics
            ms=[]
            for _ in range(30):
                value,elapsed=measure(lambda:compiled(x),device)
                check(value,expected)
                ms.append(elapsed)
            result['foreign_profile']=summary(ms)|dict(decision=policy.stats()['last'])
            if args.scenario in ('cpu_busy','gpu_busy'):
                observations=[]
                for n in (64,256):
                    source=torch.randn(n,512)
                    for path in paths:
                        for state in ('cold','warm'):
                            ms=[]
                            for _ in range(30 if state=='warm' else 3):
                                if state=='cold':resources.clear()
                                with policy.force(path):
                                    _,elapsed=measure(lambda:compiled(source),device)
                                ms.append(elapsed)
                            record=policy.stats()['last']
                            observations.append({k:record[k] for k in ('family','shape','strides','path','wire_bytes','resource_state','context')}|
                                dict(completed_ms=statistics.median(ms)))
                            result['rows'].append(dict(shape=list(source.shape),path=path,state=state,**summary(ms)))
                policy.profile=PlacementProfile(hardware=result['hardware'],
                    execution=backend.runtime.placement_configuration()[0],observations=observations)
                result['local_profile']=policy.profile.to_dict()
                results={}
                for path in paths:
                    with policy.force(path):
                        compiled(x)
                        ms=[measure(lambda:compiled(x),device)[1] for _ in range(30)]
                    results[path]=summary(ms)
                ms=[]
                for _ in range(30):
                    value,elapsed=measure(lambda:compiled(x),device)
                    check(value,expected)
                    ms.append(elapsed)
                result['qualified_profile']=summary(ms)|dict(decision=policy.stats()['last'],forced=results)
        backend.close()
    args.output.write_text(json.dumps(result,separators=(',',':'))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--scenario',choices=['hardware','cpu_busy','gpu_busy','overlap'],required=True)
    args=parser.parse_args()
    with torch.no_grad():run(args)
