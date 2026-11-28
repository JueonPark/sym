"""Render the #272 capture comparison directly from its archived process records."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'issue272/captured-quality-records.tar.gz'
ROWS = (
    ('Materialized stages', 'specialized', 'materialized'),
    ('Captured plan, prior kernels', 'generic', 'captured_artifact'),
    ('Captured plan, current kernels', 'specialized', 'captured_artifact'),
    ('Manual fusion', 'specialized', 'manual'),
    ('Eager Torch', 'specialized', 'eager_torch'),
    ('Inductor', 'specialized', 'inductor'),
    ('Full automatic endpoint', 'specialized', 'automatic_endpoint'),
)


def read_rounds():
    with tarfile.open(ARCHIVE) as archive:
        records = {}
        for runtime in ('generic', 'specialized'):
            group = []
            for number in range(1, 4):
                stem = f'r{number}-{runtime}'
                receipt = json.load(archive.extractfile(f'{stem}.audit.json'))
                row = json.load(archive.extractfile(f'{stem}/result.json'))
                plan = archive.extractfile(f'{stem}/captured.plan').read()
                if (receipt['returncode'] or receipt['observations'] or
                        row['round'] != number or row['runtime'] != runtime or
                        row['capture']['plan_sha256'] != hashlib.sha256(plan).hexdigest() or
                        row['samples'] != 30 or row['warmups'] != 5):
                    raise ValueError(f'Invalid process record: {stem}')
                group.append(row)
            records[runtime] = group
        return records


def run(output):
    records = read_rounds()
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.hashsalt': 'sym-issue274-v1', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9.2, 3.6))
    for axis, (field, title, unit, ticks) in zip(axes, [
        ('steady_p50_ms', 'Warm completed request', 'milliseconds', [.15, .25, .5, 1]),
        ('accounted_total_ms', 'Accounted 36-call trace', 'seconds', [1.5, 2, 3, 5, 8]),
    ]):
        for i, (_, runtime, arm) in enumerate(ROWS):
            values = [r['arms'][arm][field] for r in records[runtime]]
            if field == 'accounted_total_ms':
                values = [v / 1000 for v in values]
            color = '#166769' if arm in ('captured_artifact', 'automatic_endpoint') else '#454e61'
            axis.plot([min(values), max(values)], [i, i], color=color, lw=1.1, zorder=1)
            axis.scatter(values, [i - .09, i, i + .09], s=24,
                         edgecolors=color, facecolors='white', lw=.9, zorder=2)
            axis.scatter(statistics.median(values), i, s=30, marker='D',
                         color=color, zorder=3)
        axis.set_xscale('log')
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.minorticks_off()
        axis.grid(axis='x', color='#d8dde3', lw=.7)
        axis.set_axisbelow(True)
        axis.set_title(title, loc='left', fontsize=12, weight='bold', pad=12)
        axis.set_xlabel(unit + ' (log scale)')
        axis.spines[['left', 'right', 'top']].set_visible(False)
        axis.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(ROWS)), [r[0] for r in ROWS])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.29, right=.98, top=.87, bottom=.23, wspace=.15)
    fig.text(.29, .03, 'Circles denote process rounds. Diamonds mark their median.',
             color='#454e61', fontsize=9)
    fig.savefig(output, metadata={'Date': None})
    plt.close(fig)
    if output.suffix == '.svg':
        lines = output.read_text().splitlines()
        output.write_text(''.join(line.rstrip() + '\n' for line in lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'capture-cost.svg')
    run(parser.parse_args().output)
