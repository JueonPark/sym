"""Run the immutable E5 lifecycle/burst program, with optional scoped tracing."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from bench.issue263.instrument import Instrumentation
from bench.issue263.interventions import Intervention
from bench.issue263.support import add_arguments, configure, digest, load_artifact, load_module, provenance, save


def install(meter,artifact,args,kind):
    if meter.mode=='off': return
    Worker,Snapshots=artifact.e5.Worker,artifact.e5.Snapshots
    prefix=f'e5/{kind}/{args.case}/{args.runtime}/{args.variant}/{getattr(args,"compute","eager")}'
    counts={}; next_worker=[0]; setup_counts={}
    def patch(owner,name,fn):
        original=getattr(owner,name);meter.undo.append((owner,name,original,True));setattr(owner,name,fn(original))
    def initialize(original):
        def wrapped(worker,*a,**kw):
            ordinal=next_worker[0];next_worker[0]+=1
            counts[id(worker)]=[ordinal,0]
            with meter.request(f'{prefix}/caller{ordinal}/owner_setup/0'):
                with meter.phase('owner_setup'): original(worker,*a,**kw)
                worker.functions=tuple(meter.consumer(fn,f'consumer{i}') for i,fn in enumerate(worker.functions))
                worker.set_input(worker.core.x)
        return wrapped
    def request(original):
        def wrapped(worker,*a,**kw):
            caller,index=counts[id(worker)];counts[id(worker)][1]+=1
            if kind=='lifecycle':
                if index<args.requests: phase='initial' if index==0 else 'warm'
                elif index<args.requests+10: phase='shape_change'
                else: phase='scale_change'
            else: phase='warmup' if index<3*(8//args.callers) else 'warm'
            with meter.request(f'{prefix}/caller{caller}/{phase}/{index}'):
                return original(worker,*a,**kw)
        return wrapped
    def scope(name):
        def factory(original):
            def wrapped(*a,**kw):
                index=setup_counts.get(name,0);setup_counts[name]=index+1
                context=(nullcontext() if getattr(meter.local,'state',None) is not None
                         else meter.request(f'{prefix}/setup/{name}/{index}'))
                with context,meter.phase(name):return original(*a,**kw)
            return wrapped
        return factory
    patch(Worker,'__init__',initialize);patch(Worker,'request',request)
    patch(Snapshots,'__init__',scope('snapshot_setup'))
    patch(Snapshots,'refresh',scope('snapshot_refresh'))
    patch(Worker,'close',scope('owner_retirement'))
    patch(Snapshots,'close',scope('snapshot_retirement'))
    meter.wrap(artifact.e5.legacy,'host_pack','cpu_transform')
    from reloc_torch import PreparedWireWeights
    meter.wrap(PreparedWireWeights,'prepare','snapshot_prepare')
    meter.wrap(Worker,'set_input','input_binding')


def main(kind):
    p=argparse.ArgumentParser(description=__doc__);add_arguments(p)
    p.add_argument('--case',choices=('llm_decode','moe_sparse'),required=True)
    p.add_argument('--runtime',choices=('sym','torch'),required=True)
    if kind=='lifecycle':
        p.add_argument('--compute',choices=('eager','inductor'),required=True)
        p.add_argument('--requests',type=int,default=100)
    else:
        p.add_argument('--callers',type=int,choices=(1,2,4),required=True)
        p.add_argument('--batches',type=int,default=30)
    args=p.parse_args()
    if args.runtime=='torch' and args.variant!='baseline':p.error('Torch control uses baseline only')
    runtime=configure(args,threads=None);artifact=load_artifact(args.artifact)
    path=args.artifact/'05_lifecycle_concurrency'/f'{kind}.py'
    inventory={r['path']:r['sha256'] for r in json.loads((args.artifact/'SHA256-MANIFEST.json').read_text())['files']}
    assert digest(path)==inventory[str(path.relative_to(args.artifact))]
    runner=load_module(f'frozen_{kind}',path)
    forwarded=['--output',str(args.output),'--round',str(args.round),'--case',args.case,'--runtime',args.runtime]
    if kind=='lifecycle':forwarded+=['--compute',args.compute,'--requests',str(args.requests)]
    else:forwarded+=['--callers',str(args.callers),'--batches',str(args.batches)]
    argv=sys.argv
    try:
        with Intervention(args.variant,qualified_metadata=True) as intervention,Instrumentation(args.instrument) as meter:
            meter.install(artifact.controls);install(meter,artifact,args,kind)
            sys.argv=[str(path),*forwarded];runner.main()
        result=json.loads(args.output.read_text())
        assert result['complete']
        assert result['held_output_checks_after_close']==(4 if kind=='lifecycle' else 8)
        closed=result['native_after_close']
        for item in (closed if isinstance(closed,list) else [closed]):
            if item is not None:
                typed=item['typed']
                assert all(typed[k]==0 for k in ('retained_bytes','live_bytes','streams','background_workers','active_contexts'))
        if kind=='concurrency':
            assert result['physical_meter_valid']
            for before,after in zip(result['ready']['workers'],result['warm_after']['workers']):
                assert before['compile']==after['compile'], 'warm compute recompile'
                if args.runtime=='sym':
                    a,b=before['transfer']['resources'],after['transfer']['resources']
                    assert all(a[k]==b[k] for k in ('context_creations','device_allocations','host_allocations')), 'warm native growth'
        result.update(configuration=vars(args),metadata=provenance(args,artifact,runtime),host_phases=meter.records,
                      intervention=intervention.description(),frozen_runner_sha256=digest(path))
        save(args.output,result)
    finally:sys.argv=argv
