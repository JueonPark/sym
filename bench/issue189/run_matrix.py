#!/usr/bin/env python3
"""Run unchanged PR #162 workload bodies in fresh processes, serially."""
import argparse, json, os, random, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=3);p.add_argument('--variants',default='default,typed_reuse,staged_pageable,direct_pageable,cold_metadata');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
variants={'default':[], 'typed_reuse':['--typed-reuse'],
          'staged_pageable':['--typed-reuse','--staged-upload','--pinning','pageable'],
          'direct_pageable':['--typed-reuse','--pinning','pageable'],
          'cold_metadata':['--typed-reuse','--cold-metadata'],
          'auto_default':['--typed-reuse','--pinning','auto'],
          'auto_configured':['--typed-reuse','--pinning','auto','--min-pinned-bytes','8388608'],
          'pinned':['--typed-reuse','--pinning','pinned'],
          'pageable':['--typed-reuse','--pinning','pageable']}
variants={key:variants[key] for key in a.variants.split(',')}
rows=[]
for round_id in range(a.rounds):
 jobs=[(example,variant) for variant in variants for example in ['dlrm','gnn','llm','moe']]
 random.Random(189+round_id).shuffle(jobs)
 for example,variant in jobs:
  stem=f'{variant}-{example}-{round_id}'
  dest=out/(stem+'.json')
  cmd=[sys.executable,str(Path(__file__).with_name('measure_example.py')),'--repo',a.repo,'--example',example,'--output',str(dest),*variants[variant]]
  with (out/(stem+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  r=json.loads(dest.read_text());assert r['ok']
  row=dict(example=example,variant=variant,round=round_id,**r['summary']['steady_transfer_ms'])
  rows.append(row);print(row,flush=True)
  (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
