#!/usr/bin/env python3
"""Render the archived workload measurements without running GPU work."""
import argparse
import base64
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


TORCH = '#235789'
SYM = '#d36736'
REUSE = '#168574'
PHASES = {
    'frontend_prepare': ('Request preparation', '#7b62a3'),
    'frontend_execute': ('Python execution / guards', '#bd91cc'),
    'outer_host': ('Outer host / frontend', '#b5bac2'),
    'cpu_transform': ('Annotated CPU transform', '#d8a442'),
    'native_other': ('Other native execution', '#6f8c93'),
    'staging_management': ('Host scratch management', '#8cab55'),
    'device_management': ('Device scratch management', '#d36736'),
    'cuda_copy_api': ('CUDA copy API (host)', '#235789'),
    'cuda_wait_api': ('CUDA wait API (host)', '#168574'),
    'cuda_other_api': ('Other CUDA APIs (host)', '#70b4bd'),
}


def read(file):
    return json.loads(file.read_text())


def stat(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), rounds=values)


def table(headers, rows):
    def cells(values, tag):
        return ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)
    return '<div class="table"><table><thead><tr>'+cells(headers, 'th')+'</tr></thead><tbody>'+''.join(
        '<tr>'+cells(row, 'td')+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    root = args.directory
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    measurements = root/'measurements/matrix'
    examples = ['dlrm', 'gnn', 'llm', 'moe']
    cases = {e: {v: [read(measurements/f'{v}-{e}-{i}.json') for i in range(3)]
                 for v in ('default', 'typed_reuse')} for e in examples}
    traces = read(root/'profiles/summary.json')
    controls = read(root/'controls/summary.json')
    analysis = {case: read(root/'profiles'/f'{case}.analysis.json') for case in traces}
    summary = dict(workloads={}, controls={}, monitor={}, model={}, allocations={})
    for e, variants in cases.items():
        summary['workloads'][e] = {}
        for variant, reports in variants.items():
            assert all(r['ok'] and r['summary']['fallbacks'] == 0 for r in reports)
            summary['workloads'][e][variant] = {}
            for who in ('torch', 'sym'):
                raw = [r['raw_transfer_samples_ms'] for r in reports]
                summary['workloads'][e][variant][who] = {
                    'after_first_ms': stat([sum(sum(k[who][1:]) for k in r.values()) for r in raw]),
                    'all_calls_ms': stat([sum(sum(k[who]) for k in r.values()) for r in raw]),
                    'by_kind_ms': {k: stat([sum(r[k][who][1:]) for r in raw]) for k in raw[0]},
                }
    for case in {r['case'] for r in controls['rows']}:
        summary['controls'][case] = {
            variant: stat([r['median_ms'] for r in controls['rows'] if r['case'] == case and r['variant'] == variant])
            for variant in {r['variant'] for r in controls['rows'] if r['case'] == case}}
    for phase in ('controls', 'measurements', 'profiles'):
        rows = [json.loads(line) for line in (root/phase/'host-monitor.jsonl').read_text().splitlines()]
        summary['monitor'][phase] = dict(samples=len(rows),
            compiler_max=max(r['compiler_processes'] for r in rows),
            simulator_max=max(r['simulation_processes'] for r in rows))
    for case, trace in traces.items():
        assert trace['model']['kernel_signatures_equal'] and trace['model']['kernel_sequence_equal']
        summary['model'][case] = trace['model']
        apis = {}
        for transfer in analysis[case]['transfers']:
            if transfer['path'] != 'sym' or transfer['index'] == 0:
                continue
            for name, value in transfer['api_totals'].items():
                apis[name] = apis.get(name, 0)+value['count']
        summary['allocations'][case] = apis
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    def save(fig, name):
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(figures/f'{name}.{extension}', dpi=160, bbox_inches='tight')
        plt.close(fig)

    def draw_bar(ax, position, value, color, width=.32, hatch=None):
        med = value['median']
        bar = ax.bar(position, med, width, color=color, hatch=hatch, edgecolor='white', linewidth=.5)
        ax.errorbar(position, med, yerr=[[med-value['minimum']], [value['maximum']-med]],
                    color='#27313b', capsize=4, linewidth=1, fmt='none')
        ax.annotate(f'{med:.2f}', (position, value['maximum']), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=10)
        return bar

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for e, ax in zip(examples, axes.flat):
        for group, variant in enumerate(('default', 'typed_reuse')):
            for offset, who, color in ((-.18, 'torch', TORCH), (.18, 'sym', SYM if group == 0 else REUSE)):
                draw_bar(ax, group+offset, summary['workloads'][e][variant][who]['after_first_ms'], color)
        ax.set_xticks([0, 1], ['Unchanged #162 calls', '+ explicit direct-call owner'])
        ax.set_title(e.upper()+(' (owner does not change this path)' if e in ('dlrm', 'gnn') else ''))
        ax.set_ylabel('Completed transfer total (ms)')
        ax.set_ylim(0, ax.get_ylim()[1]*1.16)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Latest main: retention helps, but Torch still leads', fontsize=17, y=1.01)
    fig.legend(handles=[Patch(color=TORCH, label='Torch (paired baseline)'), Patch(color=SYM, label='Sym unchanged calls'),
                        Patch(color=REUSE, label='Sym + explicit owner')], loc='lower center', ncol=3, bbox_to_anchor=(.5, -.02))
    fig.text(.5, -.035, 'No profiler. Median of 3 fresh-process totals; whiskers show min–max. First call of each transfer kind omitted.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .97))
    save(fig, 'workload-latency')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
    kinds = ['weights', 'kv_evict', 'kv_restore']
    kind_colors = ['#235789', '#c8863d', '#7b62a3']
    for row, (variant, who, label) in enumerate([
        ('typed_reuse', 'torch', 'Torch (paired reuse baseline)'),
        ('typed_reuse', 'sym', 'Sym + explicit owner'),
        ('default', 'sym', 'Sym unchanged calls')]):
        left = 0
        for kind, color in zip(kinds, kind_colors):
            val = summary['workloads']['llm'][variant][who]['by_kind_ms'][kind]['median']
            axes[0].barh(row, val, left=left, color=color, height=.6)
            if val >= 30:
                axes[0].text(left+val/2, row, f'{val:.1f}', ha='center', va='center', color='white', fontsize=9)
            left += val
        if who == 'torch':
            axes[0].text(left+7, row, 'KV: 11.8 + 13.8 ms', va='center', fontsize=9)
    axes[0].set_yticks(range(3), ['Torch', 'Sym retained', 'Sym unchanged'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Transfer total by kind (ms), uninstrumented')
    axes[0].set_title('LLM: tiny KV calls add up')
    axes[0].legend([Patch(color=c) for c in kind_colors], ['255 weights', '111 KV evictions', '111 KV restores'], loc='upper left', bbox_to_anchor=(0, -.2), ncol=1)
    for pos, case in enumerate(('llm-retained', 'moe-retained')):
        key = 'weights' if case.startswith('llm') else 'expert_weights'
        for off, who, color in ((-.18, 'torch', TORCH), (.18, 'sym', REUSE)):
            val = traces[case]['by_kind'][key][who]['gpu_kernel_ms']
            axes[1].bar(pos+off, val, .32, color=color)
            axes[1].text(pos+off, val+.6, f'{val:.1f}', ha='center')
    axes[1].set_xticks([0, 1], ['LLM weights', 'MoE weights'])
    axes[1].set_ylim(0, 54)
    axes[1].set_ylabel('GPU conversion kernels, sum (ms)')
    axes[1].set_title('Sym’s GPU conversion is faster')
    axes[1].legend([Patch(color=TORCH), Patch(color=REUSE)], ['Torch', 'Sym retained'])
    fig.text(.5, -.12, 'Left: medians per kind, so stacks can differ slightly from median total. Right: separate Nsight runs; kernel time excludes copies.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'llm-and-gpu')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, e in zip(axes.flat, examples):
        case = e+'-retained'
        for y, who in enumerate(('torch', 'sym')):
            left = 0
            for phase, (_, color) in PHASES.items():
                value = traces[case]['totals'][who]['exclusive_ms'][phase]
                ax.barh(y, value, left=left, height=.5, color=color)
                left += value
            ax.text(left*1.01, y, f'{left:.1f}', va='center')
        ax.set_yticks([0, 1], ['Torch', 'Sym retained'])
        ax.invert_yaxis()
        ax.set_title(e.upper())
        ax.set_xlabel('Profiled caller wall time (ms)')
        ax.set_xlim(0, ax.get_xlim()[1]*1.1)
    fig.suptitle('Host-side attribution explains the remaining gap', fontsize=17)
    fig.legend([Patch(color=color) for _, color in PHASES.values()], [label for label, _ in PHASES.values()],
               loc='lower center', bbox_to_anchor=(.5, -.035), ncol=3, fontsize=9)
    fig.text(.5, -.065, 'Disjoint host phases. CUDA API time can include GPU waits; never add GPU durations to these bars. Instrumentation increases latency.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .96))
    save(fig, 'host-attribution')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, case, title in zip(axes, ['weight_4MiB', 'kv_h2d', 'kv_d2h'],
                               ['4 MiB int8 weight → 16 MiB float32', '128 KiB KV restore (H2D)', '128 KiB KV eviction (D2H)']):
        if case == 'weight_4MiB':
            variants = ['torch', 'sym_ephemeral', 'sym_retained', 'sym_prepared_outside_timer']
            labels = ['Torch', 'Sym\nephemeral', 'Sym\nretained', 'Sym execution\nonly*']
        else:
            variants = ['torch', 'sym_frontend', 'sym_direct', 'sym_prepared_outside_timer']
            labels = ['Torch', 'Sym compiled\nfrontend', 'Sym direct\nprepare + run', 'Sym execution\nonly*']
        for i, variant in enumerate(variants):
            draw_bar(ax, i, summary['controls'][case][variant], [TORCH, SYM, REUSE, '#87b8ae'][i], .62, '///' if i == 3 else None)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Completed transfer latency (ms)')
        ax.set_ylim(0, ax.get_ylim()[1]*1.17)
    fig.suptitle('Matched controls isolate preparation and frontend costs', fontsize=17)
    fig.text(.5, .01, '*Preparation outside timer: diagnostic bound, NOT end-to-end performance. Fresh request/output, guards and completion retained.\nNo profiler; median of 3 round medians × 40 calls, 5 warmups per variant/round. Whiskers show round min–max.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .105, 1, .96))
    save(fig, 'matched-controls')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6))
    for ax, kind, title in zip(axes, ['weights', 'kv_restore'], ['LLM weight fetch', 'LLM KV restore']):
        rows = [next(r for r in analysis['llm-retained']['transfers'] if r['path'] == who and r['kind'] == kind and r['index'] == 1)
                for who in ('torch', 'sym')]
        for group, row in enumerate(rows):
            host_y, gpu_y = group*2, group*2+1
            for piece in row['exclusive_timeline']:
                ax.broken_barh([(piece['start_ms'], piece['end_ms']-piece['start_ms'])], (host_y-.32, .64), facecolors=PHASES[piece['phase']][1])
            for key, color in [('copies', TORCH), ('kernels', REUSE)]:
                for event in row['gpu'][key]:
                    ax.broken_barh([(event['start_ms'], event['duration_ms'])], (gpu_y-.28, .56), facecolors=color)
            ax.text(row['wall_ms']+.01, host_y, f"{row['wall_ms']:.3f} ms", va='center', fontsize=9)
        ax.set_yticks([0, 1, 2, 3], ['Torch host', 'Torch GPU', 'Sym host', 'Sym GPU'])
        ax.set_ylim(3.7, -.7)
        ax.set_xlim(0, max(r['wall_ms'] for r in rows)*1.22)
        ax.set_title(title+' · second call, fixed index 1', fontsize=12)
        ax.set_xlabel('Time since this transfer began (ms); rows aligned at request entry')
    fig.suptitle('The GPU waits for host-side request work', fontsize=17)
    fig.text(.5, -.01, 'Illustrative early calls, not median latency; new shapes may need setup. GPU blue = copy; green = kernel. Host colors match attribution.\nSeparate Torch/Sym calls aligned at entry, not concurrent execution.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .96))
    save(fig, 'request-timeline')

    def image(name, caption):
        data = base64.b64encode((figures/f'{name}.png').read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption} · <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.pdf">PDF</a></figcaption></figure>'

    detailed_rows, cold_rows = [], []
    for e in examples:
        for v in ('default', 'typed_reuse'):
            entry = summary['workloads'][e][v]
            def cell(who, scope):
                value = entry[who][scope]
                return f"{value['median']:.3f} [{value['minimum']:.3f}, {value['maximum']:.3f}]"
            detailed_rows.append([e.upper(), v, cell('torch', 'after_first_ms'), cell('sym', 'after_first_ms'),
                                  f"{entry['sym']['after_first_ms']['median']/entry['torch']['after_first_ms']['median']:.2f}×"])
            cold_rows.append([e.upper(), v, cell('torch', 'all_calls_ms'), cell('sym', 'all_calls_ms')])
    model_rows = [[case, r['model']['torch']['kernel_count'], r['model']['sym']['kernel_count'],
                   'Identical ordered names + grid/block dimensions', f"{r['model']['torch']['kernel_ms']:.2f}", f"{r['model']['sym']['kernel_ms']:.2f}"]
                  for case, r in traces.items()]
    alloc_rows = []
    for case in ['llm-default', 'llm-retained', 'moe-default', 'moe-retained']:
        apis = summary['allocations'][case]
        alloc_rows.append([case, apis.get('cudaMalloc_v3020', 0), apis.get('cudaFree_v3020', 0),
                           apis.get('cudaStreamCreateWithFlags_v5000', 0), apis.get('cudaStreamDestroy_v5050', 0)])
    control_rows = [[case, variant, f"{s['median']:.4f}", f"{s['minimum']:.4f}–{s['maximum']:.4f}"]
                    for case, variants in sorted(summary['controls'].items()) for variant, s in sorted(variants.items())]
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Torch vs Sym · latest main workload profile</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#253342;font:16px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:45px 30px 80px;background:white}h1{font-size:2.55rem;line-height:1.16;margin:.25em 0}h2{font-size:1.6rem;margin-top:2.4em;border-top:1px solid #d8e0e7;padding-top:1.1em}
p{max-width:1000px}a{color:#235789}code{background:#edf1f5;padding:2px 5px;border-radius:3px;font-size:.9em}.tag{color:#53748b;font-size:.8rem;font-weight:700;letter-spacing:.12em}
.lead{font-size:1.2rem;color:#43586d}.note{background:#eef5f4;border-left:4px solid #168574;padding:14px 20px}figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{color:#5c6c7b;font-size:.88rem}
table{border-collapse:collapse;min-width:650px;width:100%;font-size:.9rem}td,th{padding:10px 12px;border-bottom:1px solid #dce3e8;text-align:left}th{background:#edf2f6}.table{overflow:auto;margin:18px 0}li{margin:10px 0}details{margin:20px 0}summary{cursor:pointer;font-weight:600}
</style><main><div class="tag">SYM PERFORMANCE NOTE · 2026-10-02</div><h1>Torch vs Sym after the merged patches</h1>
<p class="lead">Resource retention removes repeated allocation and stream setup. The remaining gap is largely request preparation and frontend execution; Sym’s fused weight conversion kernel is already faster in these traces.</p>
<p>Runtime: <a href="https://github.com/JueonPark/sym/commit/519f6a1c3699d10a3b59f352f89f55cf733f3113">main 519f6a1</a>, including #200, #206, #207 and #209. Workloads: unchanged bodies from <a href="https://github.com/JueonPark/sym/pull/162">open PR #162</a> at <code>cdc44e2</code>. No runtime optimization or PR merge was performed for this report.</p>
<div class="note"><strong>What is timed:</strong> the complete transfer function through device completion, including output allocation, transforms, request binding/checks and any compilation inside that function. Model computation, source preparation and oracle comparisons are outside the timer. These are transfer totals within real workload examples, not whole-model throughput.</div>
<h2>1. End-to-end transfer measurements</h2>
<p>The headline sums calls after dropping the first call of each transfer kind, then takes the median of three fresh-process rounds. Shapes change within a workload; this is not a fixed-shape steady-state benchmark. Every run passed the examples’ correctness checks with zero fallbacks. The plotted baseline is paired with each variant.</p>
'''+image('workload-latency', 'Uninstrumented transfer totals; independent process rounds and observed ranges.')+table(
        ['Workload', 'Variant', 'Torch ms [min, max]', 'Sym ms [min, max]', 'Sym / Torch'], detailed_rows)+'''
<p><code>default</code> preserves #162’s calls. Its compiled frontend already retains resources, but <code>WeightFetcher.fetch()</code> calls <code>execute_typed_transfer(request)</code> without an owner. <code>typed_reuse</code> supplies one shared <code>TransferResources</code> owner for those direct calls. DLRM and GNN do not use that direct weight path, so the flag does not change their implementation. Their small run-to-run differences are not reuse gains.</p>
<p>For LLM, explicit ownership lowers Sym’s transfer total from <strong>467.210 to 362.174 ms (22.5%)</strong>. For MoE, it lowers it from <strong>205.363 to 127.452 ms (37.9%)</strong>. Retained Sym is still about <strong>1.98×</strong> and <strong>1.69×</strong> the paired Torch time. A few DLRM Torch runs vary despite the absence of the observed background jobs; all per-round values are retained.</p>
<details><summary>All calls, including first-call effects</summary><p>This includes compilation/setup only when it occurs inside the transfer timer. Weight recipe compilation and model initialization outside that timer remain excluded. Cold costs therefore differ by frontend and transfer kind.</p>'''+table(
        ['Workload', 'Variant', 'Torch all calls ms [min, max]', 'Sym all calls ms [min, max]'], cold_rows)+'''</details>
<h2>2. Same model kernels; different relocation work</h2>
<p>The model NVTX regions have identical ordered GPU kernel names and launch grid/block dimensions for Torch and Sym. Their copy/memset counts and bytes also match. This checks the captured model regions (DLRM predict, GNN forward, LLM generate and MoE forward), excluding transfers and oracle checks; setup and argument preparation outside those regions are not covered by this claim.</p>
'''+table(['Capture', 'Torch kernels', 'Sym kernels', 'Launch comparison', 'Torch GPU ms', 'Sym GPU ms'], model_rows)+image('llm-and-gpu', 'Weight conversion is not the remaining bottleneck; many small KV calls carry substantial host overhead.')+'''
<p>Relocation kernels intentionally differ. For 255 warm LLM weight fetches, Torch’s conversion kernels total <strong>45.22 ms</strong>, versus <strong>22.10 ms</strong> for Sym’s fused dequantize/transpose. MoE is <strong>19.55 vs 10.32 ms</strong>. H2D payloads are equivalent: int8 weights plus scales, with float32 output allocated on the GPU. LLM KV restore performs no GPU conversion kernel in either path; its DMA totals about <strong>1.39 ms over 111 calls</strong> in both traces. These GPU durations come from separate instrumented captures.</p>
<h2>3. Where the caller spends time</h2>
'''+image('host-attribution', 'Disjoint caller-wall phases, after excluding the first transfer of each kind.')+'''
<p>Retained LLM weights spend <strong>71.51 ms</strong> in annotated request preparation and <strong>44.66 ms</strong> in Python execution/guard work across 255 calls. The corresponding MoE values are <strong>44.24 and 29.08 ms</strong>. These are profiled values, not overhead-free estimates. The matched controls below are the stronger evidence for the end-to-end cost of each layer.</p>
<p>DLRM/GNN use CPU transpose plus float32-to-float16 conversion. Their annotated CPU transform scopes total <strong>2.70/4.34 ms</strong>, while preparation plus Python execution total <strong>2.33/3.36 ms</strong>. Torch’s unannotated host bucket also contains its CPU operators, so the chart does not establish a direct CPU-kernel speed ratio.</p>
<p>The bars partition the caller’s wall time exactly. A blocking CUDA copy/wait API can already include GPU time; GPU durations must not be added to these bars. CPU-transform scopes include worker dispatch/wait, not just CPU utilization. Layout-only KV CPU work remains under native/host buckets because it lacks the typed host-program annotation. Nested Python/native NVTX instrumentation inflates short calls; headline numbers use a separate build with NVTX disabled.</p>
'''+image('request-timeline', 'Fixed second-call examples expose the host delay before GPU activity.')+'''
<p>Allocation evidence confirms that the merged retention machinery works when the direct call supplies an owner:</p>
'''+table(['Sym capture', 'cudaMalloc calls', 'cudaFree calls', 'Stream creates', 'Stream destroys'], alloc_rows)+'''
<p>Counts exclude each kind’s first call. LLM retained has one later scratch growth allocation; MoE retained needs none. All typed staging decisions in these captures are pageable. Auto pinning has no configured threshold, and source tensors are ordinary pageable allocations. The remaining gap therefore does not require forced pinning as an explanation.</p>
<h2>4. Controls isolate the largest remaining costs</h2>
'''+image('matched-controls', 'Fresh output and completion included; the hatched rows deliberately exclude preparation.')+'''
<p>The 4 MiB weight control is <strong>0.889 ms Torch vs 1.260 ms retained Sym</strong>. A newly prepared, single-use request executed with preparation outside the timer takes <strong>0.895 ms</strong>. This nearly closes the gap only in the diagnostic measurement; moving work outside a timer is not an end-to-end optimization. It identifies an opportunity to make preparation cheaper or amortize only invariant work while preserving mutation guards.</p>
<p>For a 128 KiB KV restore, Torch is <strong>0.084 ms</strong>, the Sym compiled frontend <strong>0.529 ms</strong>, and direct Sym preparation plus execution <strong>0.247 ms</strong>. Execution with preparation outside the timer is <strong>0.156 ms</strong>. KV eviction shows the same layering: <strong>0.097 / 0.529 / 0.258 / 0.180 ms</strong>. These controls use the same transpose, source values and output contract; changing the frontend route changes the work performed on the caller.</p>
<p>Calling-thread CPU profiles locate repeated work in metadata eligibility, symbol binding, source/storage snapshots, parameter snapshots and CUDA device/current-stream helpers. For 60 typed transfers, parameter snapshots occur 120 times (preparation plus pre-execution recheck). KV frontend profiles show the Dynamo/FX/custom-operator chain and repeated preflight checks above the direct transport API. These checks have correctness roles; deleting them is not the proposed fix. One-thread Torch KV restore is 0.0765 ms versus 0.0843 ms at eight threads, too small a difference to explain Sym’s gap.</p>
<details><summary>All matched-control medians and observed round ranges</summary>'''+table(
        ['Case', 'Variant', 'Median ms', 'Round min–max ms'], control_rows)+'''</details>
<h2>5. Priorities suggested by this evidence</h2>
<ol><li><strong>Use the existing explicit owner in #162’s direct weight fetcher.</strong> This is an integration change with measured LLM/MoE benefits. Preserve owner lifetime and close/error behavior; this report does not modify or merge #162.</li>
<li><strong>Reduce per-call typed preparation and guard overhead.</strong> Consolidate repeated metadata/device queries and native crossings; specialize invariant recipe work without caching mutable source or parameter values unchecked. Keep fresh outputs, caller-stream ordering and stale-request rejection.</li>
<li><strong>Reduce the compiled frontend cost for small layout transfers.</strong> The matched direct path saves about 0.28 ms per KV call. Investigate the custom-operator/preflight chain and, where workload dependencies permit, batching. The 222 warm LLM KV calls make small fixed costs significant. Bypassing the frontend in a diagnostic is not a drop-in production fix.</li>
<li><strong>Then reassess native event/stream bookkeeping and CPU conversion.</strong> Small-call execution still exceeds Torch after preparation is excluded. Existing traces do not isolate every remaining native sub-cost, and they do not justify replacing the faster fused GPU kernel.</li></ol>
<h2>Method, limitations and raw evidence</h2>
<p>EPYC 7351 and one RTX 2080 Ti (GPU 0), CPU affinity <code>4-7,20-23</code>, eight Torch threads and one interop thread; Torch 2.14.0+cu126, Python 3.14.7, CUDA 12.6.3, driver 595.71.05. GPU clocks were not locked. Workload/variant job order was shuffled per round, but each original example runs its whole Torch pass before Sym and performs untimed Torch oracle checks after Sym transfers. This is not a throughput/overlap benchmark.</p>
<p>Two earlier batches were affected by unrelated CPU compilation/simulation jobs and are excluded from headline results. Their measurements and available monitors remain under <code>contended/</code>; their duplicate large traces remain local. Final controls, measurements and captures recorded zero of those compiler/simulator processes at two-second monitor samples. This detects the observed interference, not every possible source of contention or sub-second job. The final run still shows ordinary timing variation.</p>
<p>All 24 final workload runs and six captures passed their example checks. Every timed control output passed a bitwise comparison. The analyzer asserts completed GPU activity fits within transfer ranges and verifies the model kernel comparisons. <code>cProfile</code> uses calling-thread CPU time; its absolute times include profiler overhead and omit descheduling/worker CPU.</p>
<ul><li><a href="summary.json">Machine-readable report summary</a> · <a href="provenance.json">source/build provenance</a> · <a href="SHA256SUMS">artifact checksums</a></li>
<li><a href="measurements/matrix/summary.json">Uninstrumented workload results</a> · <a href="controls/summary.json">control samples and CPU call profiles</a></li>
<li><a href="profiles/summary.json">Nsight attribution and model signatures</a>; each capture includes .nsys-rep, SQLite, JSON, command and export logs.</li>
<li><a href="build/native-annotations.patch">Exact native annotation patch</a> · <a href="README.md">reproduction and artifact index</a></li></ul>
</main></html>'''
    (root/'report.html').write_text(document)
    print(root/'report.html')


if __name__ == '__main__':
    main()
