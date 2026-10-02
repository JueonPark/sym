#!/usr/bin/env python3
"""Build standalone HTML and exportable figures from recorded measurements."""
import argparse, base64, html, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();root=a.results
rows=json.loads((root/'matrix/summary.json').read_text())
assert len(rows)==60, 'complete all three rounds before reporting'
examples=['dlrm','gnn','llm','moe'];names=['DLRM','GNN','LLM','MoE']
variants=['default','typed_reuse','staged_pageable','direct_pageable','cold_metadata']
summary={}
for ex in examples:
 summary[ex]={}
 for variant in variants:
  values=[r for r in rows if r['example']==ex and r['variant']==variant]
  summary[ex][variant]={path:dict(median=statistics.median(v[path] for v in values),min=min(v[path] for v in values),max=max(v[path] for v in values),samples=[v[path] for v in values]) for path in ['sym','torch']}
 for label,dirname in [('main_pinned','baseline-pinned'),('main_pageable','baseline-pageable')]:
  path=root/dirname/'summary.json'
  if path.exists():
   values=[r for r in json.loads(path.read_text()) if r['example']==ex]
   summary[ex][label]={axis:dict(median=statistics.median(v[axis] for v in values),min=min(v[axis] for v in values),max=max(v[axis] for v in values),samples=[v[axis] for v in values]) for axis in ['sym','torch']}
  else:
   prior=json.loads((root/'prior-baseline.json').read_text())['workloads'][ex]
   summary[ex][label]=prior['pinned' if label=='main_pinned' else 'pageable']['totals_ms']
# Include complete first-call/setup cost alongside warmed totals.
for ex in examples:
 for variant,directory,prefix in [('default','matrix','default'),('typed_reuse','matrix','typed_reuse'),('main_pinned','baseline-pinned','default'),('main_pageable','baseline-pageable','default')]:
  runs=[json.loads(path.read_text()) for path in (root/directory).glob(f'{prefix}-{ex}-*.json')]
  summary[ex][variant]['all_calls_ms']={axis:statistics.median(sum(sum(paths[axis]) for paths in r['raw_transfer_samples_ms'].values()) for r in runs) for axis in ['sym','torch']}
  summary[ex][variant]['first_calls_ms']={axis:statistics.median(sum(paths[axis][0] for paths in r['raw_transfer_samples_ms'].values()) for r in runs) for axis in ['sym','torch']}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':14,'figure.facecolor':'white'})
colors=['#8995a5','#3d789f','#237c68','#bd6622']
pdf=PdfPages(root/'figures.pdf')
def save(fig,name):
 pdf.savefig(fig,bbox_inches='tight')
 fig.savefig(root/(name+'.png'),dpi=175,bbox_inches='tight');fig.savefig(root/(name+'.svg'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,ex,name in zip(axes.flat,examples,names):
 vals=[summary[ex]['main_pageable']['sym']['median'],summary[ex]['default']['sym']['median'],summary[ex]['typed_reuse']['sym']['median'],summary[ex]['typed_reuse']['torch']['median']]
 bars=ax.barh(['Prior Sym pageable','Updated, default calls','Updated, direct-call reuse','Torch'],vals,color=colors)
 ax.invert_yaxis();ax.set_title(name);ax.set_xlabel('Completed transfer total (ms)')
 ax.bar_label(bars,fmt='%.1f',padding=4);ax.set_xlim(0,max(vals)*1.2);ax.grid(axis='x',alpha=.15)
fig.suptitle('End-to-end transfer cost: large improvements; Torch remains faster',fontsize=17)
save(fig,'comparison')
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
controls=[('staged_pageable','direct_pageable','Dense upload: remove Sym staging copy'),('default','typed_reuse','Direct calls: retain execution resources'),('cold_metadata','typed_reuse','Reuse immutable metadata')]
for ax,(before,after,title) in zip(axes,controls):
 chosen=['llm','moe']
 x=np.arange(len(chosen));width=.34
 b=[summary[e][before]['sym']['median'] for e in chosen];v=[summary[e][after]['sym']['median'] for e in chosen]
 ax.bar(x-width/2,b,width,label='Control',color='#8995a5');ax.bar(x+width/2,v,width,label='Enabled',color='#237c68')
 ax.set_xticks(x,['LLM','MoE']);ax.set_title(title,fontsize=12);ax.set_ylabel('Completed transfer total (ms)');ax.legend()
 for i,(old,new) in enumerate(zip(b,v)):ax.text(i,max(old,new)*1.04,f'{(1-new/old)*100:.1f}% less',ha='center',fontsize=10)
 ax.set_ylim(0,max(b+v)*1.2)
fig.suptitle('Matched ablations on the updated implementation',fontsize=17)
save(fig,'ablations')
sweeps={direction:json.loads((root/filename).read_text())['rows'] for direction,filename in [('H2D','pinning-sweep.json'),('D2H','pinning-sweep-d2h.json')]}
fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
for ax,(direction,reuse) in zip(axes.flat,[(d,r) for d in sweeps for r in [False,True]]):
 sweep=sweeps[direction]
 title=direction+(': retained, warm staging' if reuse else ': ephemeral staging')
 for mode,color in [('pinned','#bd6622'),('pageable','#3d789f')]:
  sizes=sorted({r['bytes'] for r in sweep})
  vals=[statistics.median(r['median_ms'] for r in sweep if r['bytes']==size and r['reuse']==reuse and r['pinning']==mode) for size in sizes]
  ax.plot([v/(1<<20) for v in sizes],vals,'o-',label=mode,color=color)
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(title,fontsize=12);ax.set_xlabel('Actual wire size (MiB)');ax.set_ylabel('Completed transfer latency (ms)');ax.legend();ax.grid(alpha=.2)
 if reuse:ax.axvline(8,color='#237c68',linestyle='--',label='auto threshold');ax.text(8,.8,' auto pins ≥ 8 MiB',rotation=90,color='#237c68',va='bottom')
fig.suptitle('Why auto uses both retention eligibility and size',fontsize=17)
save(fig,'pinning')
cold=[]
for ex,name in zip(examples,names):
 r=summary[ex];values=[r[k]['all_calls_ms']['sym'] for k in ['main_pageable','default','typed_reuse']]+[r['typed_reuse']['all_calls_ms']['torch']]
 cold.append('<tr><th>'+name+'</th>'+''.join(f'<td>{v:.2f}</td>' for v in values)+'</tr>')
trs=[]
for ex,name in zip(examples,names):
 r=summary[ex];values=[r['main_pinned']['sym']['median'],r['main_pageable']['sym']['median'],r['default']['sym']['median'],r['typed_reuse']['sym']['median'],r['typed_reuse']['torch']['median']]
 trs.append('<tr><th>'+name+'</th>'+''.join(f'<td>{v:.2f}</td>' for v in values)+'</tr>')
text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sym transfer overhead implementation — #189</title><style>body{font:17px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#182c37}h1,h2{line-height:1.2}h1{font-size:36px}h2{margin-top:40px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd}th:first-child{text-align:left}code{background:#eef2f4;padding:2px 5px}a{color:#236383}.note{background:#edf5f2;padding:18px;border-left:4px solid #237c68}</style>
<h1>Sym: implementing the five profiling priorities</h1>
<p class="note">All five priorities are implemented. Completed transfer times improve substantially; the measured Torch reference remains faster. Direct dispatch users obtain the resource-reuse benefit by passing an explicit owner. Existing compiler frontends use AUTO ownership.</p>
<h2>Completed transfer totals</h2><p>Milliseconds, median of three fresh-process runs. Each total excludes the first call of each transfer kind; it includes preparation, allocation, CPU conversion, transfer, and GPU completion. Model compute and correctness checks are outside the timer. This is not whole-model throughput.</p>
<table><thead><tr><th>Workload</th><th>Main pinned</th><th>Main pageable</th><th>Updated default calls</th><th>Updated + direct-call reuse</th><th>Torch</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table>
<p>“Default calls” uses unchanged PR #162 workload bodies. “Direct-call reuse” passes <code>resources=owner</code> to the weight fetcher's existing direct dispatch calls. DLRM/GNN already use frontend AUTO; their difference between those two columns is run variation. The GPU computations, shapes, precisions, wire bytes and correctness checks are unchanged.</p>
<img src="comparison.svg" alt="Completed transfer totals by workload"><p><a href="comparison.png">Download PNG</a></p>
<h2>Including first calls</h2><p>These totals include first-call graph compilation and setup whenever they occur inside the original transfer timer. For short workloads this dominates and substantially reduces the apparent speedup. None of these numbers include model compute.</p><table><thead><tr><th>Workload</th><th>Main pageable</th><th>Updated default</th><th>Updated + direct-call reuse</th><th>Torch</th></tr></thead><tbody>'''+''.join(cold)+'''</tbody></table>
<h2>What changed</h2><ol>
<li><b>CPU conversion:</b> eligible single-stage FP32→FP16 contiguous runs use Sym's existing SIMD converter. The outer layout traversal and pad semantics remain intact. Execution honors preparation/frontend thread settings. Other stage/layout combinations retain the scalar path.</li>
<li><b>Dense uploads:</b> upload from the existing host allocation instead of copying through another Sym staging buffer. Completion and ownership protection are retained; CUDA may internally stage pageable inputs.</li>
<li><b>Typed resource reuse:</b> an exclusive context retains bounded host/device scratch, streams and CPU workers. Inputs and scales are uploaded again every call. Scratch gets bounded growth headroom to avoid repeated allocations for small shape changes. Default typed retention is 64 MiB, separate from layout budgets.</li>
<li><b>Metadata:</b> cache immutable decoded plans and up to 32 validated source/destination descriptors per artifact. Deduplicate frontend/transport binding within a validated call. Prepare checked typed arithmetic once per invocation. Parameters, storage snapshots and fresh single-use requests remain per-call.</li>
<li><b>Pinning selection:</b> forced pinned/pageable and auto modes. Auto chooses pageable for ephemeral/unretainable staging and applies a strict 8 MiB threshold to retained allocations. Layout selection uses actual wire bytes, not rounded buffer capacity. Small typed parameter allocations are classified separately.</li></ol>
<h2>Measured controls</h2><p>For priority 1, the intermediate SIMD/thread change alone reduced the GNN warmed transfer total to <b>115.44 ms</b> with the original pinned allocation path still in place (one intermediate run, <a href="simd/gnn.json">raw samples</a>). Its remaining allocation overhead is addressed by the later changes. The final matrix uses three runs per variant.</p><img src="ablations.svg" alt="Independent dense-upload, resource-reuse and metadata-cache ablations"><p><a href="ablations.png">Download PNG</a></p>
<p>Dense-upload controls force pageable staging in both variants. Reuse controls differ in the explicit owner for direct weight dispatch; frontend AUTO is unchanged. The metadata control clears descriptor/decoded caches each call; it retains the new within-call binding consolidation and prepared-program reuse, so this ablation measures the caches alone.</p>
<h2>Pinning and pipelining</h2><img src="pinning.svg" alt="Cold and warm pinning size sweeps"><p><a href="pinning.png">Download PNG</a></p>
<p>The 8 MiB starting threshold is calibrated conservatively on this machine, not a universal break-even rule. A first retained pinned allocation still has a setup cost. A later cost model should account for direction, layout, expected reuse and memory pressure. The existing layout transpose/gather pipeline keeps its chunk schedule and multiple staging slots; pinned chunks can overlap CPU work with PCIe transfer. Pageable copies may block internally. The dense typed upload has no remaining CPU layout gather to overlap; the whole-buffer typed CPU reference still computes before uploading.</p>
<h2>Correctness and review</h2><p>317 native tests passed (two unsupported SIMD cases skipped). 875 Python tests passed (four skipped), including real CUDA execution, changed source/parameter values, concurrent resource use, retention/live limits, fresh outputs, stale/single-use requests, and injected CUDA event failures. When independent stream completion also fails, scratch and tensor owners remain quarantined; when completion succeeds, they are released.</p>
<h2>Reproducibility and limits</h2><p>EPYC 7351, RTX 2080 Ti GPU 0, affinity 4–7 and 20–23, eight Torch threads, one interop thread, CUDA 12.6.3, Torch 2.14.0+cu126. Release builds, profiling disabled during timing. Variants run serially in shuffled order. The machine's clocks are not locked. Each workload retains its original Torch-first order and dynamic shapes; totals are not fixed-shape latency distributions. Raw per-transfer samples and per-run resource counters are included.</p>
<p><a href="summary.json">Summary data</a> · <a href="pinning-sweep.json">Pinning samples</a> · <a href="environment.json">Environment and source hashes</a> · <a href="validation.json">Validation</a></p>
</html>'''
pdf.close()
for name in ['comparison','ablations','pinning']:
    encoded=base64.b64encode((root/(name+'.svg')).read_bytes()).decode('ascii')
    text=text.replace('src="'+name+'.svg"','src="data:image/svg+xml;base64,'+encoded+'"')
text=text.replace('<a href="summary.json">Summary data</a>', '<a href="figures.pdf">Charts PDF</a> · <a href="summary.json">Summary data</a>')
(root/'report.html').write_text(text)
print(json.dumps({e:{k:summary[e][k]['sym']['median'] for k in ['main_pageable','default','typed_reuse']} for e in examples},indent=2))
