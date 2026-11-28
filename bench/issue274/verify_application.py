"""Validate the published application receipts and reconstruct their paired summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from . import application as app


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def verify(folder):
    manifest = read(folder / 'manifest.json')
    for key, file in [('protocol_sha256', app.PROTOCOL), ('assets_sha256', app.ASSETS),
                      ('runner_sha256', Path(app.__file__))]:
        require(hashlib.sha256(file.read_bytes()).hexdigest() == manifest[key], file)
    require(manifest['arm_order'] == list(map(list, app.ORDER)), 'arm order')
    expected = {f'r{i}-{arm}.json' for i, arms in enumerate(app.ORDER) for arm in arms}
    require({p.name for p in folder.glob('r*-*.json')} == expected, 'process matrix')
    reference = read(folder / 'r0-eager.json')['requests']
    require(len({r['input_sha256'] for r in reference}) == app.REQUESTS, 'input diversity')
    sha = re.compile('[0-9a-f]{64}')
    phases = {'startup_configuration', 'inputs_and_model_cpu', 'cuda_and_residency',
              'callable_setup', 'owner_retirement'}
    for i, arms in enumerate(app.ORDER):
        for arm in arms:
            name = f'r{i}-{arm}'
            r = read(folder / (name + '.json'))
            require((r['round'], r['arm']) == (i, arm), name)
            require(set(r['phases_ns']) == phases and all(t > 0 for t in r['phases_ns'].values()), name)
            if arm == 'sym':
                stats = r['timed_stats']
                require(stats['inductor_compile_failures'] == 0 and
                        stats['inductor_executions'] == app.REQUESTS and stats['replaced_regions'] == 1,
                        f'{name} compiled consumer')
            for j, (row, oracle) in enumerate(zip(r['requests'], reference, strict=True)):
                position = f'{name} request {j}'
                role = 'cold' if j == 0 else 'warmup' if j <= app.WARMUP else 'warm'
                require(row['role'] == role and row['completed_ns'] > 0, position)
                require(row['input_sha256'] == oracle['input_sha256'] and
                        row['reference_sha256'] == oracle['logits_sha256'] and
                        row['top1'] == oracle['top1'], position)
                require(sha.fullmatch(row['logits_sha256']) and math.isfinite(row['max_abs_error'])
                        and row['max_abs_error'] >= 0, position)
                if arm == 'eager':
                    require(row['max_abs_error'] == 0 and row['logits_sha256'] == row['reference_sha256'],
                            position)
            if arm == 'sym':
                d = r['diagnostics']
                require(d['checks'] == d['held_transfers_after_close'] == len(set(d['transfer_digests'])) == 3,
                        f'{name} ownership')
                require(d['closed_prepared_state']['cache_entries'] ==
                        d['closed_prepared_state']['native']['prepared_groups']['live_slots'] == 0,
                        f'{name} retirement')
                require(d['after_dense']['prepared_execution']['executions'] ==
                        d['after_rejection']['prepared_execution']['executions'] == 114 and
                        d['after_rejection']['exclusions'].get('source_layout', 0) >
                        d['after_dense']['exclusions'].get('source_layout', 0), f'{name} strided fallback')
    summary = app.score(folder)
    require(summary == read(folder / 'summary.json'), 'paired summary differs from the receipts')
    return len(expected), len(reference)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    if not __debug__:
        parser.error('qualification requires Python assertions')
    n, k = verify(parser.parse_args().folder)
    print(f'Verified {n} process receipts, {n * k:,} completed requests and the paired summary.')
