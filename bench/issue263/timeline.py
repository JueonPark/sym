#!/usr/bin/env python3
"""Export a diagnostic request timeline from the correlated trace JSON."""
import argparse
import gzip
import html
import json
from pathlib import Path

COLORS={'consumer':'#8064a2','native_submission':'#4c78a8','native_group_preparation':'#72b7b2',
        'completion_report':'#e45756','report_accounting':'#f58518','output_allocation':'#54a24b',
        'completion_dependency':'#b279a2','source_validation':'#eeca3b','other':'#b8b8b8',
        'KERNEL':'#4c78a8','MEMCPY':'#f58518','MEMSET':'#54a24b'}


def render(row,path):
    start,end=row['start_ns'],row['end_ns'];width=1400;left=190;plot=width-left-20
    streams=sorted({a['streamId'] for a in row['activities']});lanes=['Host exclusive']+[f'CUDA stream {s}' for s in streams]
    height=165+42*len(lanes)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font:13px sans-serif;fill:#222} .title{font-size:16px;font-weight:bold}</style>',
        f'<text x="15" y="25" class="title">{html.escape(row["label"])}</text>',
        '<text x="15" y="48">Instrumented diagnostic; host/GPU intervals overlap. Width is elapsed time, not attributable savings.</text>']
    def x(t):return left+plot*(t-start)/(end-start)
    def rect(a,b,y,color,title):
        parts.append(f'<rect x="{x(a):.3f}" y="{y}" width="{max(.5,x(b)-x(a)):.3f}" height="23" fill="{color}"><title>{html.escape(title)}</title></rect>')
    for i,label in enumerate(lanes):
        y=80+42*i;parts.append(f'<text x="15" y="{y+16}">{label}</text>')
        parts.append(f'<line x1="{left}" x2="{width-20}" y1="{y+25}" y2="{y+25}" stroke="#ddd"/>')
    for p in row['phases']:rect(p['start'],p['end'],80,COLORS.get(p['category'],'#999'),p['category'])
    for a in row['activities']:
        y=80+42*(1+streams.index(a['streamId']))
        rect(a['start'],a['end'],y,COLORS[a['kind']],f'{a["name"]}; {a.get("group")}; correlation {a["correlationId"]}')
    axis=80+42*len(lanes)
    for i in range(6):
        t=start+(end-start)*i/5
        parts.append(f'<text x="{x(t):.1f}" y="{axis+20}" text-anchor="middle">{(t-start)/1e6:.2f} ms</text>')
    parts.append(f'<text x="15" y="{axis+48}">GPU: blue kernel, orange copy, green memset. Host: red completion report; purple consumer; gray other phases.</text>')
    parts.append('</svg>');Path(path).write_text('\n'.join(parts)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with gzip.open(args.trace,'rt') as f:trace=json.load(f)
    rows=[r for r in trace['requests'] if r['label']==args.label]
    if len(rows)!=1:raise ValueError('expected exactly one request with this label')
    render(rows[0],args.output)


if __name__=='__main__':main()
