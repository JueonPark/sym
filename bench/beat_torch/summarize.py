#!/usr/bin/env python3
"""Aggregate round JSONs into compact evidence and the README tables.

    python bench/beat_torch/summarize.py <rounds-dir> <evidence-dir>

Writes gzip-compressed round JSONs, summary.csv (median of round p50s and
p95s per regime/path, min/max round p50, max abs error) and manifest.json
(revision, hashes). Prints the latency table in Markdown.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(rounds_dir, evidence_dir):
    rounds_dir, evidence_dir = Path(rounds_dir), Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rounds = []
    for path in sorted(rounds_dir.glob('workloads-r*.json')):
        data = json.loads(path.read_text())
        rounds.append(data)
        with open(path, 'rb') as src, gzip.open(evidence_dir / (path.name + '.gz'), 'wb', compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
    cases = list(rounds[0]['cases'])
    paths = list(rounds[0]['cases'][cases[0]]['paths'])
    rows = []
    for case in cases:
        for path in paths:
            p50 = [r['cases'][case]['paths'][path]['p50_ms'] for r in rounds]
            p95 = [r['cases'][case]['paths'][path]['p95_ms'] for r in rounds]
            transfer = [r['cases'][case]['paths'][path]['transfer']['p50_ms'] for r in rounds]
            error = max(r['cases'][case]['paths'][path]['max_abs_error'] for r in rounds)
            rows.append(dict(case=case, path=path, p50_ms=statistics.median(p50), p50_min=min(p50), p50_max=max(p50),
                             p95_ms=statistics.median(p95), transfer_p50_ms=statistics.median(transfer),
                             max_abs_error=error, rounds=len(rounds)))
    with open(evidence_dir / 'summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(source_revision=rounds[0].get('source_revision'), rounds=len(rounds),
                    samples=rounds[0]['configuration']['samples'], warmup=rounds[0]['configuration']['warmup'],
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence_dir.glob('*')) if p.name != 'manifest.json'})
    (evidence_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    by = {(r['case'], r['path']): r for r in rows}
    print('| Regime | ' + ' | '.join(paths) + ' | sym_fast vs best control |')
    print('| --- | ' + ' | '.join('---:' for _ in paths) + ' | ---: |')
    for case in cases:
        cells = [f"{by[(case, p)]['p50_ms']:.3f} [{by[(case, p)]['p50_min']:.3f}–{by[(case, p)]['p50_max']:.3f}]" for p in paths]
        controls = [by[(case, p)]['p50_ms'] for p in ('torch', 'inductor') if (case, p) in by]
        fast = by.get((case, 'sym_fast'))
        verdict = ''
        if fast and controls:
            best = min(controls)
            every_round = all(
                r['cases'][case]['paths']['sym_fast']['p50_ms'] < 0.95 * min(r['cases'][case]['paths'][p]['p50_ms'] for p in ('torch', 'inductor'))
                for r in rounds)
            verdict = f"{(fast['p50_ms'] / best - 1) * 100:+.1f}%" + (' (win in every round)' if every_round else '')
        print(f"| {case} | " + ' | '.join(cells) + f" | {verdict} |")


if __name__ == '__main__':
    main(*sys.argv[1:3])
