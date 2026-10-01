#!/usr/bin/env python3
"""Prove gather(n+1)/DMA(n) overlap from an Nsight Systems SQLite export.

Only GPU activity timestamps establish DMA. CUDA API or NVTX submission ranges
are used for correlation, never as substitutes for GPU activity. Native worker
ranges cover gatherChunk itself, excluding the driver's dispatch/barrier wait.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


class MissingEvidence(ValueError):
    pass


def union_ns(intervals):
    end = None
    total = 0
    for begin, finish in sorted(intervals):
        if finish > begin:
            total += max(0, finish - max(begin, end if end is not None else begin))
            end = max(finish, end if end is not None else finish)
    return total


def analyze(path, tolerance_ns=1000):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        nvtx = [dict(row) for row in db.execute('''
            SELECT n.start, n.end, n.globalTid, n.uint64Value AS chunk,
                   coalesce(n.text, s.value) AS name
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
            WHERE n.end IS NOT NULL ORDER BY n.start''')]
        apis = [dict(row) for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        copies = [dict(row) for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1')]
    requests = [row for row in nvtx if (row['name'] or '').startswith('reloc.benchmark/')]
    if not requests or not copies:
        raise MissingEvidence('missing request annotations or GPU H2D activity; overlap is unqualified')
    results = []
    for request in requests:
        match = re.fullmatch(r'reloc.benchmark/buffers=(1|4)/request=(\d+)', request['name'])
        if not match:
            raise MissingEvidence('unsupported request annotation')
        buffers, index = map(int, match.groups())
        children = [row for row in nvtx if request['start'] <= row['start'] < row['end'] <= request['end']]
        work = [row for row in children if row['name'] == 'reloc.gather.work']
        submits = [row for row in children if row['name'] == 'reloc.h2d.submit']
        if not work or len(submits) < 2:
            raise MissingEvidence('missing native work/submit ranges or fewer than two chunks')
        chunks = []
        if [row['chunk'] for row in submits] != list(range(len(submits))):
            raise MissingEvidence('chunk submissions are not uniquely ordered')
        for submit in submits:
            calls = [api for api in apis if api['globalTid'] == submit['globalTid']
                     and submit['start'] <= api['start'] < api['end'] <= submit['end']]
            gpu = [copy for copy in copies if any(
                copy['correlationId'] == api['correlationId']
                and copy['globalPid'] >> 24 == api['globalTid'] >> 24 for api in calls)]
            if len(gpu) != 1:
                raise MissingEvidence('each submitted chunk must correlate to exactly one GPU H2D copy')
            dma = gpu[0]
            own_work = [row for row in work if row['chunk'] == submit['chunk']]
            next_work = [row for row in work if row['chunk'] == submit['chunk'] + 1]
            if not own_work or max(row['end'] for row in own_work) > submit['start']:
                raise MissingEvidence('gather/barrier ordering is incomplete')
            if not request['start'] <= dma['start'] < dma['end'] <= request['end'] + tolerance_ns:
                raise MissingEvidence('DMA extends beyond the completed request')
            overlap = union_ns([(max(row['start'], dma['start']), min(row['end'], dma['end']))
                                for row in next_work])
            chunks.append(dict(chunk=submit['chunk'], bytes=dma['bytes'], stream=dma['streamId'],
                               correlation_id=dma['correlationId'], dma_start_ns=dma['start'], dma_end_ns=dma['end'],
                               submit_start_ns=submit['start'], submit_end_ns=submit['end'],
                               gather=[dict(start_ns=row['start'], end_ns=row['end'], thread=row['globalTid']) for row in own_work],
                               next_chunk_work_overlap_ns=overlap))
        observed = any(chunk['next_chunk_work_overlap_ns'] > tolerance_ns for chunk in chunks)
        results.append(dict(request=index, buffers=buffers, start_ns=request['start'], end_ns=request['end'],
                            chunks=chunks, observed_overlap=observed,
                            gate_passed=observed if buffers > 1 else not observed))
    return dict(schema_version=1, status='qualified' if all(r['gate_passed'] for r in results) else 'failed',
                sqlite_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                timestamp_tolerance_ns=tolerance_ns, requests=results)


def chrome_trace(result):
    origin = min(row['start_ns'] for row in result['requests'])
    events = []
    def span(name, start, end, pid, tid, args):
        events.append(dict(name=name, ph='X', ts=(start-origin)/1000, dur=(end-start)/1000,
                           pid=pid, tid=tid, args=args))
    for request in result['requests']:
        for chunk in request['chunks']:
            args = dict(request=request['request'], chunk=chunk['chunk'], bytes=chunk['bytes'])
            span(f"H2D chunk {chunk['chunk']}", chunk['dma_start_ns'], chunk['dma_end_ns'], 'GPU', chunk['stream'], args)
            for work in chunk['gather']:
                span(f"gather chunk {chunk['chunk']}", work['start_ns'], work['end_ns'], 'CPU', work['thread'], args)
    return dict(traceEvents=events, displayTimeUnit='ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chrome', type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.sqlite)
    except (MissingEvidence, sqlite3.Error) as error:
        result = dict(status='unavailable', error=str(error))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if args.chrome and result.get('requests'):
        args.chrome.write_text(json.dumps(chrome_trace(result), indent=2) + '\n')
    if result['status'] != 'qualified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
