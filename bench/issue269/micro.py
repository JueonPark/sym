"""Exact cache, lifetime, consumer-cost and physical-pressure supplements."""
import gc
import hashlib
import time
from types import SimpleNamespace

from bench.issue269 import contract
from bench.issue269.model import CompileState


def bitcheck(torch, actual, expected):
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected):
        assert a.shape==b.shape and a.is_contiguous()
        assert torch.equal(a.cpu().view(torch.int16),b.view(torch.int16))


def one(args,job,case,variant,inputs,torch,elapsed):
    from reloc_torch import PackedWireRestore
    row=dict(case=case,variant=variant,outcome='pending',phases=[],completed_ms=[],submission_ms=[],
             checks=0,retained_checks=0,consumer_ms=[],transfer_ms=[])
    helper=state=reserve=None;previous=None
    def begin(name):row['phase']=name;return elapsed(),time.perf_counter_ns()
    def end(name,start):
        row['phases'].append(dict(name=name,start_ms=start[0],end_ms=elapsed()))
        return (time.perf_counter_ns()-start[1])/1e6
    try:
        start=begin('prepare')
        torch._dynamo.reset();torch.cuda.synchronize();torch.cuda.empty_cache()
        if case['budget_mib']:
            reserve=inputs.memory.Reserve(case['budget_mib']<<20);row['budget']=reserve.stats()
        s,h,d=case['length'],case['heads'],case['dim']
        gen=torch.Generator().manual_seed(269+s+h+d)
        hosts=[torch.randn((2,s,h,d),generator=gen,dtype=torch.float16).pin_memory() for _ in range(4)]
        news_cpu=[torch.randn((2,h,1,d),generator=gen,dtype=torch.float16) for _ in range(4)]
        news=[tuple(t.cuda() for t in v.unbind(0)) for v in news_cpu]
        refs=[]
        for host,new in zip(hosts,news_cpu):
            restored=tuple(t.transpose(0,1).contiguous() for t in host.unbind(0))
            result=tuple(torch.cat((a,b),1) for a,b in zip(restored,new))
            refs.append(result+restored if case['visible'] else result)
        row['input_sha256']=[hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()).hexdigest()
                             for a,b in zip(hosts,news_cpu)]
        helper=inputs.controls.TorchKV() if variant=='native' else PackedWireRestore('cuda:0',max_bytes=128<<20)
        state=CompileState(torch,job['compute'],variant=='lowered')
        def consumer(past,new):
            restored=tuple(past) if variant=='native' else tuple(t.transpose(0,1).contiguous() for t in past)
            result=tuple(torch.cat((a,b),1) for a,b in zip(restored,new))
            return result+restored if case['visible'] else result
        fn=state.compile(consumer)
        def call(index):return fn(helper.restore(list(hosts[index].unbind(0))),news[index])
        def timed(fn):
            torch.cuda.synchronize();t=time.perf_counter_ns();out=fn()
            submit=(time.perf_counter_ns()-t)/1e6;torch.cuda.synchronize()
            return out,(time.perf_counter_ns()-t)/1e6,submit
        torch.cuda.synchronize();row['preparation_ms']=end('prepare',start)
        start=begin('first');out,_,_=timed(lambda:call(0));row['first_ms']=end('first',start);bitcheck(torch,out,refs[0]);del out
        start=begin('warmup')
        for i in range(5):
            out,_,_=timed(lambda:call(i%4));bitcheck(torch,out,refs[i%4]);del out
        row['warmup_ms']=end('warmup',start);row['warmups']=5
        row.update(compile_before=state.callbacks,transfer_before=helper.stats())
        torch.cuda.reset_peak_memory_stats();start=begin('warm');row['warm_start_ms']=start[0]
        for i in range(5 if args.pilot else 100):
            out,cost,submit=timed(lambda:call(i%4))
            row['completed_ms'].append(cost);row['submission_ms'].append(submit)
            bitcheck(torch,out,refs[i%4]);row['checks']+=1
            if previous:bitcheck(torch,*previous);row['retained_checks']+=1
            previous=(out,tuple(t.cpu() for t in out));del out
        end('warm',start);row['warm_end_ms']=row['phases'][-1]['end_ms']
        row.update(torch_memory=inputs.memory.torch_stats(torch),compile_after=state.callbacks,
                   compile_ms=state.milliseconds,lowering=state.decisions(),transfer_after=helper.stats())
        # Independent intervals, never subtracted from completed request cost.
        start=begin('isolated')
        for i in range(5 if args.pilot else 100):
            restored,cost,_=timed(lambda:helper.restore(list(hosts[i%4].unbind(0))))
            row['transfer_ms'].append(cost)
            output,cost,_=timed(lambda:fn(restored,news[i%4]));row['consumer_ms'].append(cost)
            bitcheck(torch,output,refs[i%4]);del restored,output
        end('isolated',start)
        row['compiler_graphs']=[]
        for i,graph in enumerate(state.graphs):
            for kind,code in graph.items():
                p=args.output.parent/'graphs'/f"{case['name']}-{variant}-{i}-{kind}.py"
                p.parent.mkdir(exist_ok=True);p.write_text(code)
                row['compiler_graphs'].append(dict(file=str(p.relative_to(args.output.parent)),sha256=contract.sha(p)))
        if args.profile and case['name'] in ('canonical','held_long'):
            from bench.issue269.worker import profile
            profile(SimpleNamespace(output=args.output.parent/f"{case['name']}-{variant}.json"),row,torch,lambda:call(0))
        start=begin('retirement')
        helper.drain();state.close()
        if hasattr(helper,'close'):helper.close()
        row['retirement_ms']=end('retirement',start)
        bitcheck(torch,*previous);row['retained_checks']+=1;row['held_after_close']=1
        row['outcome']='success'
    except Exception as error:
        if isinstance(error,torch.OutOfMemoryError) or ('out of memory' in str(error).lower() and 'cuda' in str(error).lower()):
            row.update(outcome='oom',error=repr(error))
        else:raise
    finally:
        if helper:
            helper.drain()
            if hasattr(helper,'close'):helper.close()
        if state:state.close()
        if reserve:reserve.close()
    return row


def run(args,job,result,inputs,torch,elapsed):
    result['rows']=[]
    with torch.no_grad():
        modes=contract.VARIANTS[job['round']%3:]+contract.VARIANTS[:job['round']%3]
        for case in contract.MICROS:
            for variant in modes:
                result['rows'].append(one(args,job,case,variant,inputs,torch,elapsed))
                gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    result['outcome']='success'
