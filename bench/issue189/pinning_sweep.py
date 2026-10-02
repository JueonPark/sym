#!/usr/bin/env python3
"""Completed layout-transpose transfers: cold/warm, pinned/pageable, same bytes."""
import argparse, json, os, random, statistics, time
from pathlib import Path
import torch
from reloc_torch import CompilerClient, TransferResources
from reloc_torch.recipe import Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Symbol, Const, dense_strides
from reloc_torch.transport import prepare_transfer, execute_transfer

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--direction',choices=['h2d','d2h'],default='h2d');a=p.parse_args()
torch.set_num_threads(8);torch.set_num_interop_threads(1)
torch.manual_seed(189)
n,m=Symbol('s0'),Symbol('s1')
def spec(shape):return TensorSpec(shape,dense_strides(shape),Const(0),'float32')
compiled=CompilerClient.from_environment().compile(Recipe(spec((n,m)),(Transpose((1,0)),),spec((m,n)),a.direction))
rows=[];rng=random.Random(189)
with torch.no_grad():
 for wire in [64<<10,256<<10,1<<20,4<<20,8<<20,16<<20,32<<20,64<<20]:
  cpu=torch.randn(wire//(512*4),512);expected=cpu.t().contiguous();src=cpu if a.direction=='h2d' else cpu.to('cuda:0')
  variants=[(reuse,mode) for reuse in [False,True] for mode in ['pinned','pageable']]
  for round_id in range(3):
   rng.shuffle(variants)
   for reuse,mode in variants:
    owner=TransferResources() if reuse else None
    def call():
     req=prepare_transfer(compiled,src,'cuda:0' if a.direction=='h2d' else 'cpu')
     return execute_transfer(req,resources=owner,pinning=mode,gather_threads=8)
    samples=[]
    for i in range(15):
     torch.cuda.synchronize();start=time.perf_counter();out=call();torch.cuda.synchronize()
     elapsed=(time.perf_counter()-start)*1000
     if i>=3:samples.append(elapsed)
    assert torch.equal(out.cpu(),expected)
    rows.append(dict(bytes=wire,reuse=reuse,pinning=mode,round=round_id,samples_ms=samples,median_ms=statistics.median(samples)))
    if owner:owner.close()
  print(wire,[(r['reuse'],r['pinning'],round(r['median_ms'],3)) for r in rows[-4:]],flush=True)
Path(a.output).write_text(json.dumps(dict(direction=a.direction,rows=rows,affinity=sorted(os.sched_getaffinity(0)),threads=8,correct=True,scope=__doc__),indent=2)+'\n')
