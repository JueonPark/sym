#!/usr/bin/env python3
"""Run uninstrumented main measurements and separate Nsight captures serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['measure','profile','controls'],required=True)
    p.add_argument('--workloads',type=Path,required=True,help='checkout containing unchanged #162 bodies')
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--nsys',default='/tmp/sym-cuda-toolkit-12.6.3/bin/nsys')
    p.add_argument('--aten-nvtx',action='store_true')
    p.add_argument('--torch-hooks',type=Path)
    p.add_argument('--retained-only',action='store_true')
    p.add_argument('--examples',default='dlrm,gnn,llm,moe')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(args.build/'python'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    commands=[]
    def monitor(stop):
        previous={}
        with (args.output/'host-monitor.jsonl').open('a') as stream:
            while not stop.is_set():
                cpus={}
                for line in Path('/proc/stat').read_text().splitlines():
                    fields=line.split()
                    if fields and fields[0] in ['cpu']+[f'cpu{i}' for i in (4,5,6,7,20,21,22,23)]:
                        counters=list(map(int,fields[1:9]));total=sum(counters);idle=counters[3]+counters[4]
                        old=previous.get(fields[0]);previous[fields[0]]=(total,idle)
                        if old and total>old[0]:cpus[fields[0]]=100*(1-(idle-old[1])/(total-old[0]))
                compilers=0;simulators=0
                for proc in Path('/proc').glob('[0-9]*/comm'):
                    try:
                        name=proc.read_text().strip()
                        compilers+=name in {'cc1plus','cc1','clang','clang++','ld','ld.lld'}
                        simulators+=name=='gem5.opt'
                    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
                stream.write(json.dumps(dict(time_unix=time.time(),cpu_busy_percent=cpus,compiler_processes=compilers,simulation_processes=simulators))+'\n');stream.flush()
                stop.wait(2)
    def run(command, log):
        commands.append(command)
        (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        stop=threading.Event();watcher=threading.Thread(target=monitor,args=(stop,),daemon=True);watcher.start()
        try:
            with log.open('w') as stream:
                subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        finally:
            stop.set();watcher.join()
    if args.phase=='controls':
        run(['taskset','-c','4-7,20-23',sys.executable,str(Path(__file__).with_name('controls.py')),
             '--workloads',str(args.workloads),'--output',str(args.output)],args.output/'run.log')
    elif args.phase=='measure':
        command=['taskset','-c','4-7,20-23',sys.executable,str(root/'bench/issue189/run_matrix.py'),
            '--repo',str(args.workloads),'--output',str(args.output/'matrix'),
            '--rounds',str(args.rounds),'--variants','default,typed_reuse']
        run(command,args.output/'matrix.log')
    else:
        cases=[(name,True) for name in ('dlrm','gnn','llm','moe')]+[(name,False) for name in ('llm','moe')]
        if args.retained_only:cases=cases[:4]
        cases=[(name,reuse) for name,reuse in cases if name in args.examples.split(',')]
        if args.torch_hooks:env['LD_PRELOAD']=str(args.torch_hooks.resolve())
        for name,reuse in cases:
            stem=name+('-retained' if reuse else '-default')
            dest=args.output/stem
            command=['taskset','-c','4-7,20-23',args.nsys,'profile','--trace=cuda,nvtx','--sample=none',
                '--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--force-overwrite=true','--output',str(dest),sys.executable,
                str(Path(__file__).with_name('capture.py')),'--repo',str(args.workloads),
                '--example',name,'--output',str(dest)+'.json']
            if reuse:command.append('--typed-reuse')
            if args.aten_nvtx:command.append('--aten-nvtx')
            if args.torch_hooks:command.append('--native-hooks')
            run(command,args.output/(stem+'-capture.log'))
            assert json.loads(Path(str(dest)+'.json').read_text())['ok']
            run([args.nsys,'export','--type=sqlite','--force-overwrite=true',
                '--output',str(dest)+'.sqlite',str(dest)+'.nsys-rep'],args.output/(stem+'-export.log'))
            print(stem,'correct; capture/export complete',flush=True)


if __name__=='__main__':main()
