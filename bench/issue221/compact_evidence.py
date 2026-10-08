#!/usr/bin/env python3
"""Keep lossless per-call samples and provenance without verbose runtime reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('inputs',nargs='+',type=Path)
    args = p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows, metadata = [], []
    for path in sorted(args.inputs):
        run = json.loads(path.read_text())
        metadata.append(dict(run=path.stem,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            configuration=run['configuration'],metadata=run['metadata'],
            recipe_sha256={k:v.get('recipe_sha256') for k,v in run['cases'].items()}))
        for name, case in run['cases'].items():
            for method, data in case['paths'].items():
                stats, report = data.get('resources') or {}, data.get('report') or {}
                rows.append(dict(run=path.stem,case=name,path=method,
                    source_bytes=case['source_bytes'],wire_bytes=case['wire_bytes'],result_bytes=case['result_bytes'],
                    first_completed_ms=data['first_completed_ms'],p50_ms=data['p50_ms'],p95_ms=data['p95_ms'],
                    host_retained_bytes=stats.get('host_bytes',data.get('torch_retained_host_bytes',0)),
                    device_scratch_bytes=stats.get('device_bytes',0),peak_native_scratch_bytes=stats.get('peak_live_bytes',0),
                    host_pipeline=report.get('host_pipeline','legacy_whole' if method=='legacy' else ''),
                    chunks=report.get('host_chunks',''),chunk_bytes=report.get('host_chunk_bytes',''),
                    buffers=report.get('host_buffers',''),
                    inductor_compile_and_prime_ms=case.get('inductor_compile_and_prime_ms') if method=='inductor' else '',
                    samples_ms=json.dumps(data['samples_ms'],separators=(',',':'))))
    with (args.output_dir/'timings.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    (args.output_dir/'provenance.jsonl').write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in metadata))


if __name__=='__main__': main()
