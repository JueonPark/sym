#!/usr/bin/env python3
"""GPU DMA overlap and CPU work correlation; CUDA API waits are excluded."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/issue221'))
from analyze_trace import union


def overlap(a, b):
    return sum(max(0, min(y, v)-max(x, u)) for x, y in union(a) for u, v in union(b))


def main(args):
    db = sqlite3.connect(args.sqlite)
    db.row_factory = sqlite3.Row
    names = dict(db.execute('select id,value from StringIds'))
    ranges = [dict(row) for row in db.execute('select * from NVTX_EVENTS where end is not null order by start')]
    for r in ranges: r['label'] = r['text'] or names.get(r['textId'], '')
    batches = [r for r in ranges if r['label'].startswith('sym228/batch/')]
    copies = [dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1')]
    work = [r for r in ranges if r['label'] in ('reloc.typed.transform.work', 'reloc.typed.fill')]
    summaries, witnesses, counts = [], {}, {}
    for batch in batches:
        label = batch['label']
        ordinal = counts.get(label, 0)
        counts[label] = ordinal + 1
        if ordinal < 8: continue  # three cold-owner calls and five warmups
        dma = [r for r in copies if batch['start'] <= r['start'] < r['end'] <= batch['end']]
        cpu = union([[r['start'], r['end']] for r in work if batch['start'] <= r['start'] < r['end'] <= batch['end']])
        devices = sorted({r['deviceId'] for r in dma})
        assert len(devices) == 2 and cpu, (label, devices, len(cpu))
        by_device = {d: [[r['start'], r['end']] for r in dma if r['deviceId']==d] for d in devices}
        intervals = [v for values in by_device.values() for v in values]
        item = dict(path=label, sample=ordinal-8, batch_ms=(batch['end']-batch['start'])/1e6,
                    dma_copies=len(dma), wire_bytes=sum(r['bytes'] for r in dma),
                    dma_overlap_ms=overlap(*by_device.values())/1e6,
                    cpu_dma_overlap_ms=overlap(cpu, intervals)/1e6)
        summaries.append(item)
        if label not in witnesses and item['dma_overlap_ms'] > 0 and item['cpu_dma_overlap_ms'] > 0:
            base = batch['start']
            witnesses[label] = dict(sample=ordinal-8, start_ns=base, end_ns=batch['end'],
                cpu_work_ns=[[x-base, y-base] for x,y in cpu],
                dma=[dict(device=r['deviceId'], start_ns=r['start']-base, end_ns=r['end']-base,
                          bytes=r['bytes'], correlation_id=r['correlationId']) for r in dma])
    assert summaries and witnesses, 'no qualified overlap witness captured'
    result = dict(sqlite_sha256=hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),
                  runtime_sha256=hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
                  run=json.loads(args.run.read_text()),
                  summaries=summaries, witnesses=witnesses)
    args.output.write_bytes(gzip.compress((json.dumps(result, separators=(',', ':'))+'\n').encode(), mtime=0))
    for label in counts:
        rows = [r for r in summaries if r['path']==label]
        print(label, 'samples', len(rows), 'DMA overlapping', sum(r['dma_overlap_ms']>0 for r in rows),
              'CPU/DMA overlapping', sum(r['cpu_dma_overlap_ms']>0 for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
