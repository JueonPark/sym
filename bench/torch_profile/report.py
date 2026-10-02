#!/usr/bin/env python3
"""Render symmetric Torch/Sym attribution, keeping unresolved work explicit."""
import argparse
import base64
import gzip
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUPS = {
    'Observed preparation / guards': (['preparation_guards'], '#7862a3'),
    'Output setup / allocation': (['output_setup_allocation'], '#bc93c7'),
    'Other Python / C++ host work': (['frontend_execution', 'torch_operator_other', 'native_other', 'outer_host'], '#aeb8c2'),
    'CPU transform region': (['cpu_transform'], '#d3a143'),
    'Host/device memory management': (['staging_management', 'device_management'], '#c97245'),
    'CUDA copy API (host)': (['cuda_copy_api'], '#235789'),
    'CUDA wait API (host)': (['cuda_wait_api'], '#168574'),
    'Other CUDA APIs': (['cuda_other_api'], '#73b8bb'),
}


def read(file):
    if file.suffix == '.gz':
        with gzip.open(file, 'rt') as f:
            return json.load(f)
    return json.loads(file.read_text())


def table(headers, rows):
    def cells(values, tag):
        return ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)
    return '<div class="table"><table><thead><tr>'+cells(headers, 'th')+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells(row, 'td')+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    root = a.directory
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    profiles = read(root/'profiles/shared-summary.json')
    detail = {name: read(root/'profiles'/f'{name}.analysis.json.gz') for name in profiles}
    coarse = read(root/'coarse/llm-retained.analysis.json.gz')
    measurements = read(root/'measurements/matrix/summary.json')
    controls = read(root/'controls/summary.json')
    examples = ['dlrm', 'gnn', 'llm', 'moe']
    summary = dict(measurements={}, preparation={}, tensor_iterator={}, monitor={}, coarse_control={}, controls={})
    for e in examples:
        summary['measurements'][e] = {}
        for variant in ('default', 'typed_reuse'):
            rows = [r for r in measurements if r['example'] == e and r['variant'] == variant]
            assert len(rows) == 3
            summary['measurements'][e][variant] = {who: dict(median_ms=statistics.median(r[who] for r in rows),
                min_ms=min(r[who] for r in rows), max_ms=max(r[who] for r in rows)) for who in ('torch', 'sym')}
        profile = profiles[e+'-retained']
        assert profile['model']['kernel_sequence_equal'] and profile['model']['kernel_signatures_equal']
        summary['preparation'][e] = {kind: {who: dict(calls=s['calls'], total_ms=s['exclusive_ms']['preparation_guards'],
            us_per_call=s['exclusive_ms']['preparation_guards']*1000/s['calls']) for who, s in paths.items()}
            for kind, paths in profile['by_kind'].items()}
        for kind in profile['by_kind']:
            rows = [r for r in detail[e+'-retained']['transfers'] if r['path'] == 'torch' and r['kind'] == kind and r['index'] > 0]
            scopes = {}
            for method in ['build', 'compute_mem_overlaps', 'compute_shape', 'compute_types', 'fast_set_up', 'allocate_or_resize_outputs']:
                vals = [r['inclusive_scopes'].get('torch.native.ti.'+method, {'count': 0, 'ms': 0}) for r in rows]
                scopes[method] = dict(count=sum(v['count'] for v in vals), inclusive_ms=sum(v['ms'] for v in vals),
                                     mean_us_per_transfer=sum(v['ms'] for v in vals)*1000/len(rows))
            summary['tensor_iterator'][e+'/'+kind] = dict(calls=len(rows), methods=scopes)
    for phase in ('measurements', 'controls', 'profiles', 'coarse'):
        rows = [json.loads(s) for s in (root/phase/'host-monitor.jsonl').read_text().splitlines()]
        summary['monitor'][phase] = dict(samples=len(rows), compiler_max=max(r['compiler_processes'] for r in rows),
                                       simulator_max=max(r['simulation_processes'] for r in rows))
    for mode, profile in [('full', detail['llm-retained']), ('build_only', coarse)]:
        summary['coarse_control'][mode] = {}
        for kind in ('weights', 'kv_restore', 'kv_evict'):
            rows = [r for r in profile['transfers'] if r['path'] == 'torch' and r['kind'] == kind and r['index'] > 0]
            summary['coarse_control'][mode][kind] = dict(
                median_ti_build_us=statistics.median(r['inclusive_scopes']['torch.native.ti.build']['ms']*1000 for r in rows),
                median_transfer_us=statistics.median(r['wall_ms']*1000 for r in rows),
                builds=sum(r['inclusive_scopes']['torch.native.ti.build']['count'] for r in rows))
    for case in {r['case'] for r in controls['rows']}:
        summary['controls'][case] = {v: statistics.median(r['median_ms'] for r in controls['rows'] if r['case'] == case and r['variant'] == v)
                                    for v in {r['variant'] for r in controls['rows'] if r['case'] == case}}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(figures/f'{name}.{ext}', dpi=160, bbox_inches='tight')
        plt.close(fig)
    def stacked(ax, paths):
        for y, who in enumerate(('torch', 'sym')):
            left = 0
            s = paths[who]
            for keys, color in GROUPS.values():
                value = sum(s['exclusive_ms'][key] for key in keys)
                ax.barh(y, value, left=left, color=color, height=.52)
                left += value
            assert abs(left-s['wall_ms']) < 1e-6
            ax.text(left*1.01, y, f'{left:.2f}', va='center')
        ax.set_yticks([0, 1], ['Torch', 'Sym retained'])
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1]*1.12)
        ax.set_xlabel('Profiled caller wall time (ms)')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.3))
    for ax, e in zip(axes.flat, examples):
        stacked(ax, profiles[e+'-retained']['totals'])
        ax.set_title(e.upper())
    fig.suptitle('Torch also pays preparation and guard costs', fontsize=17)
    fig.legend([Patch(color=color) for _, color in GROUPS.values()], list(GROUPS), ncol=3,
               loc='lower center', bbox_to_anchor=(.5, .015), fontsize=9)
    fig.text(.5, -.014, 'Shared classification, disjoint host wall time. Residual host/CPU regions can still contain uninstrumented checks.\nDiagnostic traces include instrumentation overhead. CPU regions include execution setup; GPU time is not added.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .96))
    save(fig, 'shared-attribution')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, kind, title in zip(axes, ['weights', 'kv_restore', 'kv_evict'], ['LLM weights', 'KV restore (H2D)', 'KV eviction (D2H)']):
        paths = profiles['llm-retained']['by_kind'][kind]
        for pos, who, color in [(0, 'torch', '#235789'), (1, 'sym', '#168574')]:
            value = paths[who]['exclusive_ms']['preparation_guards']*1000/paths[who]['calls']
            ax.bar(pos, value, .6, color=color)
            ax.text(pos, value+8, f'{value:.1f}', ha='center')
        ax.set_xticks([0, 1], ['Torch', 'Sym'])
        ax.set_title(title)
        ax.set_ylabel('Observed preparation / guards (µs/call)')
        ax.set_ylim(0, max(s['exclusive_ms']['preparation_guards']*1000/s['calls'] for s in paths.values())*1.25)
    fig.suptitle('Preparation is nonzero on both paths', fontsize=17)
    fig.text(.5, .015, 'Instrumented means over calls after the first. Observed preparation includes metadata work, not only predicates.\nThese are not overhead-free costs or a complete census of every guard.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .94))
    save(fig, 'preparation-per-call')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    methods = ['compute_mem_overlaps', 'compute_shape', 'compute_types']
    labels = ['Memory-overlap checks', 'Broadcast shape', 'Dtype / device']
    for ax, kind, title in zip(axes, ['weights', 'kv_restore'], ['LLM weight fetch', 'KV restore']):
        data = summary['tensor_iterator']['llm/'+kind]
        for pos, (method, label) in enumerate(zip(methods, labels)):
            val = data['methods'][method]['mean_us_per_transfer']
            ax.barh(pos, val, color=['#7862a3', '#bc93c7', '#73b8bb'][pos], height=.55)
            ax.text(val+.15, pos, f'{val:.2f} µs', va='center')
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1]*1.3)
        ax.set_xlabel('Instrumented function time per transfer (µs)')
        builds = data['methods']['build']['count']/data['calls']
        ax.set_title(f'{title}: {builds:g} TensorIterator builds / transfer')
    fig.suptitle('Inside Torch’s native preparation', fontsize=17)
    fig.text(.5, -.015, 'Three selected native functions. Additional operand, stride, dimension and output setup work is measured separately.\nSub-microsecond functions are sensitive to observer overhead; the next control exposes that effect.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .94))
    save(fig, 'torch-native-checks')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for group, kind in enumerate(('weights', 'kv_restore', 'kv_evict')):
        for off, mode, color in [(-.18, 'full', '#7862a3'), (.18, 'build_only', '#235789')]:
            value = summary['coarse_control'][mode][kind]['median_ti_build_us']
            ax.bar(group+off, value, .32, color=color)
            ax.text(group+off, value+1, f'{value:.1f}', ha='center')
    ax.set_xticks(range(3), ['LLM weights', 'KV restore', 'KV eviction'])
    ax.set_ylabel('Median sum of TI build scopes / transfer (µs)')
    ax.set_ylim(0, ax.get_ylim()[1]*1.2)
    ax.legend([Patch(color='#7862a3'), Patch(color='#235789')], ['Build + inner function ranges', 'Build ranges only'])
    ax.set_title('Detailed instrumentation changes the time being observed')
    fig.text(.5, -.015, 'Same operators and build counts; separate captures. Both include ATen and CPU-copy observers.\nBuild includes output setup/allocation. The difference is a sensitivity check, not an exact overhead subtraction.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, 'instrumentation-sensitivity')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.7))
    colors = {key: color for keys, color in GROUPS.values() for key in keys}
    for ax, kind in zip(axes, ['weights', 'kv_restore']):
        rows = [next(r for r in detail['llm-retained']['transfers'] if r['path'] == who and r['kind'] == kind and r['index'] == 20)
                for who in ('torch', 'sym')]
        for group, r in enumerate(rows):
            for piece in r['exclusive_timeline']:
                ax.broken_barh([(piece['start_ms'], piece['end_ms']-piece['start_ms'])], (group*2-.3, .6), facecolors=colors[piece['phase']])
            for key, color in [('copies', '#235789'), ('kernels', '#168574')]:
                for event in r['gpu'][key]:
                    ax.broken_barh([(event['start_ms'], event['duration_ms'])], (group*2+.7, .6), facecolors=color)
            ax.text(r['wall_ms']+.01, group*2, f"{r['wall_ms']:.3f} ms", va='center', fontsize=9)
        ax.set_yticks([0, 1, 2, 3], ['Torch host', 'Torch GPU', 'Sym host', 'Sym GPU'])
        ax.set_ylim(3.7, -.7)
        ax.set_xlim(0, max(r['wall_ms'] for r in rows)*1.25)
        ax.set_title(f'LLM {kind} · fixed call index 20 (illustrative, not median)')
        ax.set_xlabel('Time since each call began (ms)')
    fig.suptitle('Both paths prepare work before the GPU runs', fontsize=17)
    fig.text(.5, -.02, 'Host colors match shared attribution. GPU blue = copies; green = conversion kernels.\nSeparate Torch/Sym calls aligned at entry, not concurrent execution.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .96))
    save(fig, 'shared-timeline')

    def image(name, caption):
        blob = base64.b64encode((figures/f'{name}.png').read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{blob}" alt="{html.escape(caption)}"><figcaption>{caption} · <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'
    measure_rows = []
    for e in examples:
        for v in ('default', 'typed_reuse'):
            r = summary['measurements'][e][v]
            measure_rows.append([e.upper(), v]+[f"{r[w]['median_ms']:.3f} [{r[w]['min_ms']:.3f}, {r[w]['max_ms']:.3f}]" for w in ('torch', 'sym')])
    prep_rows = [[e.upper(), k, r['torch']['calls'], f"{r['torch']['us_per_call']:.2f}", f"{r['sym']['us_per_call']:.2f}"]
                 for e, kinds in summary['preparation'].items() for k, r in kinds.items()]
    coverage_rows = [[key, r['calls'], r['methods']['build']['count'],
                      r['methods']['compute_mem_overlaps']['count'], r['methods']['compute_shape']['count'],
                      r['methods']['compute_types']['count']] for key, r in summary['tensor_iterator'].items()]
    model_rows = [[e.upper(), profiles[e+'-retained']['model']['torch']['kernel_count'],
                   profiles[e+'-retained']['model']['sym']['kernel_count'], 'identical ordered names and launch dimensions'] for e in examples]
    ctrl = summary['controls']['weight_4MiB']
    torch_sha = '08187d9e0fba026dc8217405802ab5381dc88d90'
    source = f'https://github.com/pytorch/pytorch/blob/{torch_sha}/aten/src/ATen/TensorIterator.cpp'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Torch preparation and guards · Sym comparison</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#263644;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 30px 70px;background:white}
h1{font-size:2.5rem;line-height:1.2}h2{margin-top:2.3em;padding-top:1em;border-top:1px solid #dce3e9}a{color:#235789}code{background:#edf1f4;padding:2px 4px}.lead{font-size:1.2rem;color:#476174}.note{padding:15px 20px;background:#eff5f8;border-left:4px solid #235789}
figure{margin:28px 0}img{max-width:100%}figcaption{font-size:.88rem;color:#5c6e7c}.table{overflow:auto;margin:20px 0}table{border-collapse:collapse;min-width:650px;width:100%;font-size:.9rem}td,th{padding:9px 12px;border-bottom:1px solid #dae3ea;text-align:left}th{background:#edf2f6}li{margin:9px 0}summary{font-weight:600;cursor:pointer}
</style><main><p>SYM PERFORMANCE EXPERIMENT · 2026-10-02</p><h1>Torch also prepares and checks its requests</h1>
<p class="lead">Native Torch preparation is now visible beside Sym’s preparation. The old graph’s zero for Torch was an annotation gap, not zero work.</p>
<p>Runtime main <code>519f6a1</code>; unchanged workload bodies from open PR #162 at <code>cdc44e2</code>. Torch 2.14.0+cu126, Python 3.14.7, EPYC 7351 / RTX 2080 Ti GPU 0, eight Torch threads and one interop thread. No production runtime changes or PR merge.</p>
<div class="note"><strong>Two independent measurements:</strong> headline transfer latency uses a normal build without preload hooks or profiler. Diagnostic captures enable ATen ranges for both paths, native TensorIterator hooks, and a C++ observer that identifies actual CPU-to-CPU copy operators. Observed preparation is a subset, not every possible validation instruction.</div>
<h2>What changed in the experiment</h2>
<p>The installed Torch wheel exports TensorIterator preparation functions. A diagnostic ELF interposer forwards each call once to its original implementation and records its duration. It covers build, operand/output setup, memory overlap, broadcast shape, dtype/device selection, strides, dimension order/coalescing and CPU iteration. A Torch RecordFunction callback reads the input/output devices of copy_ and annotates CPU copies, including conversion kernels that do not use the exported for_each loop. The same observers apply to Torch calls made by Sym.</p>
<p>These boundaries were checked against the exact installed Torch revision: <a href="'''+source+'''#L1462">TensorIteratorBase::build</a>. The probe was compiled with Release-compatible <code>NDEBUG</code> and the installed headers. Value checks, invalid-shape and overlapping-output errors pass under the probe. Capture validation also checks that all expected shape/dtype/overlap callbacks occur.</p>
'''+table(['Torch path / kind', 'Transfers', 'TI builds', 'Overlap checks', 'Shape checks', 'Type checks'], coverage_rows)+'''
<h2>Shared attribution, with unknown work left visible</h2>
'''+image('shared-attribution', 'Both paths use the same disjoint host-time classification.')+table(
        ['Workload', 'Kind', 'Calls after first', 'Torch observed prep µs/call', 'Sym observed prep µs/call'], prep_rows)+image(
        'preparation-per-call', 'The Torch preparation category is no longer zero.')+'''
<p><strong>Preparation/guards</strong> includes observed metadata, binding, snapshot/recheck and TensorIterator work; it is not exclusively boolean predicates. <strong>Output setup/allocation</strong> separates empty operations and TensorIterator output setup, which can itself include metadata work. Native CUDA allocation/free APIs and scratch bookkeeping are assigned to <strong>host/device memory management</strong>. <strong>Other Python/C++ host work</strong> combines residual frontend, operator and native work on both paths; uninstrumented guards can remain here. It is deliberately not called Python-only time.</p>
<p><strong>CPU transform regions</strong> include CPU copy bodies identified from real tensor devices, TensorIterator CPU loops, and Sym’s native transform/gather scopes. Observed preparation and CUDA API subranges are removed from these regions, but residual kernel setup and checks can remain. Durations are caller wall time, including worker dispatch/wait, not summed CPU utilization. CUDA API durations can already contain GPU execution; the separate GPU track must not be added to the host stack.</p>
'''+image('torch-native-checks', 'Native Torch checks, measured at actual function boundaries.')+image('shared-timeline', 'Fixed call index 20 shows preparation and GPU activity on both paths.')+'''
<h2>Instrumentation sensitivity</h2>
'''+image('instrumentation-sensitivity', 'A second LLM capture retains build-level hooks and omits inner TensorIterator ranges.')+'''
<p>The build-only capture keeps the same original operators, ATen observers, CPU-copy observer and TensorIterator build counts. It disables inner native range recording. The build scope includes output setup/allocation in both cases, making it a comparable sensitivity control. It is still instrumented and is not a measurement of pure guard time. Comparing separate captures cannot provide an exact per-marker cost, so no overhead subtraction is applied to the reported results.</p>
<p>The detailed timings therefore establish where work occurs and its observed scale. They do not support treating each small native duration as its uninstrumented cost, or treating the difference between Torch’s and Sym’s preparation categories as a guaranteed optimization gain.</p>
<h2>Profiler-free end-to-end transfer results</h2>
'''+table(['Workload', 'Variant', 'Torch ms: median [min, max]', 'Sym ms: median [min, max]'], measure_rows)+'''
<p>Twenty-four fresh-process runs, three rounds per workload/variant. Each value sums completed transfers after omitting the first call of each kind; shapes vary within workloads. Request work, allocation, conversion and completion are included. Model computation and untimed oracle checks are excluded. These are not whole-model throughput measurements. Raw JSON preserves all calls, including first-call effects.</p>
<p>Default calls preserve #162. The typed_reuse variant supplies an explicit owner to direct weight dispatch. The compiled DLRM/GNN paths already retain resources, so this flag does not change their execution path. Pinning remains auto without a configured threshold; captured typed staging is pageable.</p>
<p>The fresh matched 4 MiB control is <strong>'''+f"{ctrl['torch']:.3f} ms Torch versus {ctrl['sym_retained']:.3f} ms retained Sym"+'''</strong>. Executing with preparation outside the timer is <strong>'''+f"{ctrl['sym_prepared_outside_timer']:.3f} ms"+'''</strong>; this is a diagnostic bound, not an end-to-end speedup. Preparation and frontend overhead remain useful Sym optimization targets, but Torch is also doing real preparation and checking.</p>
<h2>Correctness, coverage and evidence</h2>
'''+table(['Workload', 'Torch model kernels', 'Sym model kernels', 'Comparison'], model_rows)+'''
<p>All 24 uninstrumented workload runs and five final diagnostic captures pass workload checks; the controls compare every timed output bitwise. The analyzer verifies GPU activity correlation/completion, exact host-time partitioning and matching model kernel sequences within the model regions. It excludes transfer/oracle work from model comparisons. Initialization and argument work outside annotated model regions are not covered by that claim.</p>
<p>GPU clocks are unlocked. CPU affinity is 4–7,20–23. The unchanged workloads run Torch before Sym and perform untimed Torch oracle calls between Sym transfers. Two-second monitors record zero of the previously observed compiler/simulator jobs during the final runs; this is not a guarantee of exclusive host access. Profiling is more intrusive on short operations and the native probe is specific to this Torch ABI.</p>
<p>The initial v1 probe captured iterator internals but missed some CPU conversion bodies. Those exploratory results remain local. The published final captures use the v2 CPU-copy observer; the build-only control uses the same v2 binary. Earlier aborted prototype runs are excluded.</p>
<ul><li><a href="summary.json">Numerical summary</a> · <a href="provenance.json">provenance and hashes</a> · <a href="README.md">reproduction and artifact index</a></li>
<li><a href="profiles/shared-summary.json">Full capture attribution and native counts</a> · <a href="coarse/shared-summary.json">build-only sensitivity control</a></li>
<li><a href="controls/summary.json">Fresh controls and paired Python call profiles</a> · <a href="measurements/matrix/summary.json">uninstrumented workload measurements</a></li>
<li>Each capture includes its Nsight report, gzipped SQLite export, gzipped detailed analysis, original workload report, logs and commands. <a href="SHA256SUMS">Checksums</a> cover the published artifacts.</li></ul></main></html>'''
    (root/'report.html').write_text(document)
    print(root/'report.html')


if __name__ == '__main__':
    main()
