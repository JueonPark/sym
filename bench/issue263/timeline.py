#!/usr/bin/env python3
"""Plot a request timeline from correlated trace JSON (requires matplotlib)."""
import argparse
import gzip
import json
from pathlib import Path

COLORS={'consumer':'#8064a2','native_submission':'#4c78a8','native_group_preparation':'#72b7b2',
        'completion_report':'#e45756','report_accounting':'#f58518','output_allocation':'#54a24b',
        'completion_dependency':'#b279a2','source_validation':'#eeca3b','other':'#b8b8b8',
        'KERNEL':'#4c78a8','MEMCPY':'#f58518','MEMSET':'#54a24b'}


def render(row,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    start,end=row['start_ns'],row['end_ns']
    streams=sorted({a['streamId'] for a in row['activities']})
    labels=['Host exclusive']+[f'CUDA stream {s}' for s in streams]
    fig,ax=plt.subplots(figsize=(14,max(3,1.5+.45*len(labels))))
    def bar(a,b,lane,color):
        ax.broken_barh([((a-start)/1e6,(b-a)/1e6)],(lane-.35,.7),facecolors=color,edgecolors='none')
    for phase in row['phases']:
        bar(phase['start'],phase['end'],0,COLORS.get(phase['category'],'#999999'))
    for activity in row['activities']:
        bar(activity['start'],activity['end'],1+streams.index(activity['streamId']),COLORS[activity['kind']])
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.set_xlim(0,(end-start)/1e6);ax.set_xlabel('Elapsed time within instrumented request (ms)')
    ax.set_title(row['label'],fontsize=10,loc='left');ax.grid(axis='x',alpha=.2)
    ax.legend(handles=[Patch(color=COLORS[k],label=label) for k,label in (
        ('consumer','Host consumer'),('completion_report','Host completion report'),
        ('KERNEL','GPU kernel'),('MEMCPY','GPU copy'),('MEMSET','GPU memset'))],
        loc='upper center',bbox_to_anchor=(.5,-.22),ncol=5,fontsize=8,frameon=False)
    fig.text(.01,.01,'Diagnostic tracing adds overhead. Host and GPU intervals overlap; these widths are not attributable savings.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(path,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with gzip.open(args.trace,'rt') as f:trace=json.load(f)
    rows=[r for r in trace['requests'] if r['label']==args.label]
    if len(rows)!=1:raise ValueError('expected exactly one request with this label')
    render(rows[0],args.output)


if __name__=='__main__':main()
