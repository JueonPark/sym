"""Original decoder equations with explicit, matched representation boundaries."""
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

from bench.issue269 import contract


class CompileState:
    def __init__(self, torch, compute, lowered):
        from reloc_torch import RelocBackend
        self.torch=torch;self.backend=RelocBackend(compute_backend=compute,consumer_layout=lowered)
        self.callbacks=0;self.milliseconds=0.;self.graphs=[]
    def compile(self,fn):
        def backend(gm,values):
            start=time.perf_counter_ns();result=self.backend(gm,values)
            self.callbacks+=1;self.milliseconds+=(time.perf_counter_ns()-start)/1e6
            self.graphs.append(dict(original=gm.code,lowered=(result.rewritten or gm).code))
            return result
        return self.torch.compile(fn,backend=backend,fullgraph=True,dynamic=False)
    def close(self):self.backend.close()
    def decisions(self):return self.backend.stats()['consumer_layout']['decisions']


class ResidentProducer:
    def __init__(self, weights):self.weights=weights
    @contextmanager
    def prefetch(self,groups,slots=2):
        yield ([self.weights[name] for name in group] for group in groups)
    def stats(self):return dict(resident_bytes=sum(t.numel()*t.element_size() for t in self.weights.values()))
    def close(self):self.weights.clear()


class ResidentKV:
    def restore(self,stored):return list(stored)
    def evict(self,values):return list(values)
    def drain(self):pass


class WireKV:
    def __init__(self, original, helper):self.original,self.helper=original,helper
    def evict(self,values):return self.original.evict(values)
    def restore(self,values):return self.helper(values)


class Region:
    def __init__(self,fn,torch):
        from reloc_torch import GPUReplay,GPUReplayPolicy
        self.torch=torch;self.frozen=False
        self.owner=GPUReplay(fn,device='cuda:0',policy=GPUReplayPolicy(slots=2,max_buckets=8,max_bytes=256<<20))
    def __call__(self,*args):
        if not self.frozen:
            self.torch.cuda.synchronize();self.owner.prepare(*args)
        return self.owner(*args)
    def freeze(self):self.owner.freeze();self.frozen=True


class Worker:
    def __init__(self,core,snapshots,inputs,job,torch):
        from reloc_torch.fast import FastTransfers,FastWeights,KVCacheSwap
        from reloc_torch import PackedWireRestore
        self.torch,self.core,self.inputs,self.job=torch,core,inputs,job
        self.life=ExitStack();self.stream=torch.cuda.Stream();self.fast=None;self.raw=None;self.regions=[]
        if job['variant']=='resident':
            resident={name:inputs.e2.legacy.common.WeightFetcher.reference(q,s,'cuda:0').half()
                      for name,(q,s) in core.by_name.items()}
            self.producer=ResidentProducer(resident);self.life.callback(self.producer.close)
            self.kv=ResidentKV()
        elif job['runtime']=='sym':
            self.fast=self.life.enter_context(FastTransfers('cuda:0',compiler=snapshots.compiler,
                gather_threads=8,contexts=2,max_scratch_bytes=64<<20,pinning='pinned'))
            self.producer=FastWeights(self.fast,snapshots.weights)
            self.kv=KVCacheSwap(self.fast,16,64,dtype='float16');self.life.callback(self.kv.drain)
        else:
            self.producer=inputs.controls.TorchProducer(snapshots.images,
                lambda q,s:inputs.controls.fused_dequant(q,s,torch.float16),torch.float16)
            self.life.callback(self.producer.close)
            self.kv=inputs.controls.TorchKV();self.life.callback(self.kv.drain)
        swap=self.kv
        if job['variant'] in ('materialized','lowered'):
            if self.fast:
                swap=WireKV(self.kv,self.kv.restore_wire)
            else:
                self.raw=PackedWireRestore('cuda:0',max_bytes=128<<20);self.life.callback(self.raw.close)
                swap=WireKV(self.kv,self.raw.restore)
        self.swap=inputs.controls.Swaps(swap,'grouped')
        self.state=CompileState(torch,job['compute'],job['variant']=='lowered');self.life.callback(self.state.close)
        original=inputs.e2.legacy.models.decoder_layer
        def wire_layer(x,past,weights):
            if past is not None:
                past=tuple(t.transpose(0,1).contiguous() for t in past)
            return original(x,past,weights)
        fn=wire_layer if job['variant'] in ('materialized','lowered') else original
        self.decoder_reference=fn
        self.functions=tuple(self.state.compile(f) for f in
            (fn,inputs.e2.legacy.models.embed,inputs.e2.legacy.models.logits))
        if job['replay']:
            self.regions=[Region(f,torch) for f in self.functions]
            for region in self.regions:self.life.callback(region.owner.close)
            self.functions=tuple(self.regions)
        self.set_input(core.x)
    def set_input(self,x):
        self.fn=self.inputs.e2.gpt_path(self.core.model,x,self.core.sizes,self.producer,
            self.swap,self.functions,self.core.groups,'whole')
    def request(self,x):
        torch=self.torch
        with torch.cuda.stream(self.stream),torch.no_grad():
            self.stream.synchronize();start=time.perf_counter_ns();self.set_input(x)
            out=self.fn();submit=(time.perf_counter_ns()-start)/1e6
            self.stream.synchronize();completed=(time.perf_counter_ns()-start)/1e6
        return out,completed,submit
    def audit_caches(self,values):
        # Run after all timings: identical actual region inputs and compute
        # backend, with only the private-copy rule disabled in the reference.
        reference=CompileState(self.torch,self.job['compute'],False)
        original=self.functions
        checks=0
        try:
            baseline=reference.compile(self.decoder_reference)
            def checked(*args):
                nonlocal checks
                actual=original[0](*args)
                expected=baseline(*args)
                for a,b in zip(actual[1],expected[1]):
                    assert a.shape==b.shape and a.stride()==b.stride()
                    assert self.torch.equal(a.view(self.torch.int16),b.view(self.torch.int16))
                    checks+=1
                return actual
            self.functions=(checked,*original[1:])
            for value in values:self.request(value)
            return dict(cache_checks=checks,cache_requests=len(values),
                        cache_reference_compiles=reference.callbacks)
        finally:
            self.functions=original;reference.close()

    def freeze(self):
        for region in self.regions:region.freeze()
    def replay_stats(self):return [r.owner.stats() for r in self.regions]
    def stats(self):
        return dict(weights=self.fast.stats() if self.fast else self.producer.stats(),
            wire=(self.kv._wire_restore.stats() if self.fast and self.kv._wire_restore else
                  self.raw.stats() if self.raw else None),
            kv=self.kv.stats() if hasattr(self.kv,'stats') else None)
    def close(self):
        self.life.close();self.fn=None;self.functions=()


def fixture(inputs,job,torch):
    legacy=inputs.e2.legacy;legacy.COMPUTE_DTYPE='float16'
    torch.manual_seed(225);gen=torch.Generator().manual_seed(225)
    prompt,new_tokens=contract.CASES[job['case']]
    sizes=dict(vocab=2048,d_model=1024,heads=16,d_ff=4096,layers=4,prompt=prompt,new_tokens=new_tokens)
    model=legacy.OffloadedGPT(sizes,torch.device('cuda:0'),gen)
    model.token,model.position=model.token.half(),model.position.half()
    values=[torch.randint(2048,(prompt,),device='cuda:0') for _ in range(8)]
    reference=lambda q,s,d=torch.device('cuda:0'):legacy.common.WeightFetcher.reference(q,s,d).half()
    refs=[inputs.workload.cpu_result(legacy.generate(model,x,new_tokens,reference,
        lambda t:t.transpose(0,1).contiguous().cpu(),lambda t:t.transpose(0,1).contiguous().cuda())) for x in values]
    pairs=[pair for layer in model.layers for pair in layer.values()]
    names={id(q):str(i) for i,(q,s) in enumerate(pairs)}
    groups=[[names[id(layer[k][0])] for k in legacy.MATRICES] for layer in model.layers]
    core=SimpleNamespace(case='llm_decode',model=model,x=values[0],sizes=sizes,groups=groups,
        pairs=pairs,names=names,by_name={names[id(q)]:(q,s) for q,s in pairs})
    return core,values,refs


def run(args,job,result,inputs,torch,elapsed):
    worker=None;snapshots=None;previous=None
    def begin(name):result['phase']=name;return elapsed(),time.perf_counter_ns()
    def end(name,start):
        result['phases'].append(dict(name=name,start_ms=start[0],end_ms=elapsed()))
        return (time.perf_counter_ns()-start[1])/1e6
    try:
        with torch.no_grad():
            start=begin('fixture');core,values,refs=fixture(inputs,job,torch);torch.cuda.synchronize()
            result['sizes']=core.sizes
            result['input_sha256']=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in values]
            end('fixture',start)
            start=begin('prepare')
            if job['variant']!='resident':snapshots=inputs.workload.Snapshots(core,job['runtime'])
            worker=Worker(core,snapshots,inputs,job,torch);torch.cuda.synchronize()
            result['preparation_ms']=end('prepare',start)
            start=begin('first');out,_,_=worker.request(values[0]);result['first_ms']=end('first',start)
            inputs.workload.check(out,refs[0])
            start=begin('warmup')
            for i in range(5):
                out,_,_=worker.request(values[i%8]);inputs.workload.check(out,refs[i%8])
            worker.freeze();result['warmup_ms']=end('warmup',start);result['warmups']=5
            result.update(compile_before=worker.state.callbacks,replay_before=worker.replay_stats(),
                transfer_before=worker.stats(),completed_ms=[],submission_ms=[],checks=0,retained_checks=0)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=begin('warm')
            result['warm_start_ms']=start[0]
            for i in range(2 if args.pilot else 30):
                out,cost,submit=worker.request(values[i%8])
                result['completed_ms'].append(cost);result['submission_ms'].append(submit)
                inputs.workload.check(out,refs[i%8]);result['checks']+=1
                if previous:inputs.workload.exact_retained(*previous);result['retained_checks']+=1
                previous=(out,inputs.workload.cpu_result(out))
            end('warm',start);result['warm_end_ms']=result['phases'][-1]['end_ms']
            result.update(torch_memory=inputs.memory.torch_stats(torch),compile_after=worker.state.callbacks,
                compile_ms=worker.state.milliseconds,lowering=worker.state.decisions(),
                replay_after=worker.replay_stats(),transfer_after=worker.stats())
            result['compiler_graphs']=[]
            for i,graph in enumerate(worker.state.graphs):
                for kind,code in graph.items():
                    p=args.output.parent/'graphs'/f'{i}-{kind}.py';p.parent.mkdir(exist_ok=True);p.write_text(code)
                    result['compiler_graphs'].append(dict(file=str(p.relative_to(args.output.parent)),sha256=contract.sha(p)))
            start=begin('cache_audit');result.update(worker.audit_caches(values))
            result['cache_audit_ms']=end('cache_audit',start)
            if args.profile:
                from bench.issue269.worker import profile
                profile(args,result,torch,lambda:worker.request(values[0]))
            start=begin('retirement');worker.close()
            if snapshots:snapshots.close();snapshots=None
            result['retirement_ms']=end('retirement',start)
            inputs.workload.exact_retained(*previous);result['retained_checks']+=1;result['held_after_close']=1
            result['outcome']='success'
    finally:
        if worker:worker.close()
        if snapshots:snapshots.close()
