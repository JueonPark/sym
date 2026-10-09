#!/usr/bin/env python3
"""Keep raw timing arrays and scalar diagnostics; omit logs and trace databases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def compact(value):
    return json.dumps(value, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples, details, provenance = [], [], []
    for path in args.runs:
        run = json.loads(path.read_text())
        raw_only = all(name.endswith('_raw') for name in run['configuration']['paths'])
        provenance.append(dict(run=path.stem, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata=run['metadata'], configuration=run['configuration'], limits=run['limits'],
            sym_artifact_setup_ms=run['sym_artifact_setup_ms']))
        for section in ('matrices', 'models'):
            for case, record in run.get(section, {}).items():
                details.append(dict(run=path.stem, section=section, case=case,
                    record={k: v for k, v in record.items() if k != 'paths'}))
                for name, values in record['paths'].items():
                    metrics = {'warm': values['warm'] if section == 'matrices' else
                               {k: values[k] for k in ('p50_ms', 'p95_ms', 'samples_ms')}}
                    metrics.update({key: values[key] for key in ('replace', 'invalidate_then_prepare') if key in values})
                    # Raw-only runs qualify existing-path warm regressions; their
                    # unused reuse sweeps are not part of the reported comparison.
                    if not raw_only:
                        metrics.update({'reuse_' + key: value for key, value in values.get('reuse_total', {}).items()})
                    for metric, value in metrics.items():
                        samples.append(dict(run=path.stem, section=section, case=case, path=name,
                            metric=metric, p50_ms=value['p50_ms'], p95_ms=value['p95_ms'],
                            samples_ms=compact(value['samples_ms'])))
                    details.append(dict(run=path.stem, section=section, case=case, path=name,
                        record={k: v for k, v in values.items() if k not in
                                {'warm','replace','invalidate_then_prepare','reuse_total','p50_ms','p95_ms','samples_ms'}}))
    with (args.output / 'timings.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(samples)
    for name, records in [('diagnostics', details), ('provenance', provenance)]:
        (args.output / (name + '.jsonl')).write_text(''.join(compact(row)+'\n' for row in records))


if __name__ == '__main__':
    main()
