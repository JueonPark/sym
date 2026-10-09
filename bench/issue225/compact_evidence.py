#!/usr/bin/env python3
"""Keep raw timing samples and scalar evidence; exclude verbose runtime logs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rounds', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'timings.csv').open('w') as timings, \
         (args.output/'diagnostics.jsonl').open('w') as diagnostics, \
         (args.output/'provenance.jsonl').open('w') as provenance:
        writer = csv.writer(timings, lineterminator='\n')
        writer.writerow(['round','case','path','phase','p50_ms','p95_ms','samples_ms'])
        for round_id, source in enumerate(args.rounds, 1):
            result = json.loads(source.read_text())
            provenance.write(json.dumps(dict(round=round_id, **{k:result[k] for k in
                ('metadata','configuration')}), separators=(',',':'))+'\n')
            for case, row in result['cases'].items():
                for name, values in row['paths'].items():
                    for phase in ('end_to_end','transfer','kv_roundtrips'):
                        dist = values if phase == 'end_to_end' else values.get(phase)
                        if dist:
                            writer.writerow([round_id,case,name,phase,dist['p50_ms'],dist['p95_ms'],
                                             json.dumps(dist['samples_ms'],separators=(',',':'))])
                    record = dict(round=round_id,case=case,path=name,configuration=row['configuration'],
                        first_completed_ms=values['first_completed_ms'],
                        first_generated_kernels=values['first_generated_kernels'],
                        first_compile_callbacks=values['first_stats']['compile_callbacks'],
                        first_backend_compile_ms=values['first_stats']['compile_ms'],
                        max_abs_error=max(values['max_abs_error'],values['first_max_abs_error']),
                        warm_compile_callbacks=values['warm_compile_callbacks'],
                        dynamo_counters=row['dynamo_counters'])
                    backend = values['warm_stats']['backend']
                    if backend:
                        record['backend'] = {k:backend[k] for k in ('compute_backend','inductor_compiles',
                            'inductor_compile_failures','inductor_executions','dynamo_compiles','plan_compiles',
                            'replaced_regions','runtime_executions','typed_executions','typed_payload_bytes',
                            'fallbacks','exclusions')}
                    if 'resources' in values:
                        record['resources'] = values['resources']
                    diagnostics.write(json.dumps(record,separators=(',',':'))+'\n')


if __name__ == '__main__':
    main()
