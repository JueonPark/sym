#!/usr/bin/env python3
"""Compact request-correlated CPU, DMA and kernel evidence from Nsight SQLite."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.issue221.analyze_trace import union


def duration(rows):
    return sum(b-a for a, b in union((r['start'], r['end']) for r in rows))/1e6


def analyze(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = {r['id']: r['value'] for r in db.execute('select id,value from StringIds')}
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    events = [dict(r) for r in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
    for r in events:
        r['label'] = r.pop('text') or strings.get(r.pop('textId'), '')
    roots = [r for r in events if r['label'].startswith('request219/')]
    apis = [dict(r) for r in db.execute('select start,end,nameId,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')]
    for r in apis:
        r['name'] = strings[r.pop('nameId')]
    activity = {}
    for table, kind in [('CUPTI_ACTIVITY_KIND_MEMCPY', 'dma'), ('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel')]:
        if table not in tables:
            continue
        for r in db.execute('select * from '+table):
            event = {k: r[k] for k in ('start', 'end', 'correlationId', 'deviceId', 'streamId')}
            event['kind'] = kind
            if kind == 'dma':
                event.update(bytes=r['bytes'], copy_kind=r['copyKind'])
            else:
                event['name'] = strings[r['demangledName']]
            activity.setdefault(r['correlationId'], []).append(event)
    requests, witnesses, seen = [], [], set()
    for root in roots:
        start, end, tid = root['start'], root['end'], root['globalTid']
        contained = lambda r: start <= r['start'] <= r['end'] <= end
        host = [r for r in apis if contained(r) and r['globalTid'] == tid]
        # Worker-thread CUDA submissions are also part of the serialized root.
        calls = [r for r in apis if contained(r)]
        gpu = [r for a in calls for r in activity.get(a['correlationId'], [])]
        assert all(contained(r) for r in gpu), 'completed root must contain correlated GPU activity'
        nested = [r for r in events if contained(r) and r is not root]
        work = [r for r in nested if r['label'] in ('reloc.typed.transform.work', 'reloc.typed.fill', 'reloc.gather.work')]
        dma = [r for r in gpu if r['kind'] == 'dma']
        kernels = [r for r in gpu if r['kind'] == 'kernel']
        native_kernels = [r for r in kernels if 'reloc::cuda::' in r['name']]
        native = [r for r in nested if r['label'].startswith('phase219/native_including_completion/')]
        waits = [r for r in host if 'Synchronize' in r['name']]
        native_waits = [r for r in waits if any(n['start'] <= r['start'] <= r['end'] <= n['end'] for n in native)]
        overlap = sum(max(0, min(b, d)-max(a, c)) for a, b in union((r['start'], r['end']) for r in work)
                      for c, d in union((r['start'], r['end']) for r in dma))/1e6
        requests.append(dict(label=root['label'], completed_ms=(end-start)/1e6,
            native_host_ms=duration(native), native_completion_ms=duration(native_waits),
            cuda_completion_ms=duration(waits), cuda_copy_api_ms=duration([r for r in host if 'Memcpy' in r['name']]),
            cuda_allocation_api_ms=duration([r for r in host if 'Alloc' in r['name'] or 'Malloc' in r['name'] or 'Free' in r['name']]),
            cpu_transform_union_ms=duration(work), dma_union_ms=duration(dma), kernel_union_ms=duration(kernels),
            sym_kernel_union_ms=duration(native_kernels), cpu_dma_overlap_ms=overlap,
            wire_bytes=sum(r['bytes'] for r in dma if r['copy_kind'] in (1, 2)),
            kernel_names=sorted({r['name'] for r in native_kernels})))
        if root['label'] not in seen:
            seen.add(root['label'])
            def relative(rows):
                return [r | dict(start=r['start']-start, end=r['end']-start) for r in rows]
            witnesses.append(dict(label=root['label'], duration_ns=end-start,
                host_apis=relative(host), phases=relative(nested), gpu=relative(gpu)))
    assert requests and witnesses
    return dict(sqlite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Instrumented diagnostics. CPU phases and CUDA API subsets overlap GPU activity; do not sum them. '
              'GPU unions avoid double counting concurrent streams. Timings are milliseconds; witness intervals are relative nanoseconds.',
        requests=requests, first_call_witnesses=witnesses)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    with gzip.open(args.output, 'wt') as file:
        json.dump(analyze(args.sqlite), file, separators=(',', ':'))
