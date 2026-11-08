"""Strict saved-data analysis. Standard library only; no GPU or local artifacts."""
import argparse
import bisect
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.issue268 import contract


def require(condition, message):
    if not condition: raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name)


def membership(actual, expected):
    require(len(actual) == len(set(actual)) and set(actual) == set(expected), 'matrix membership')


def validate_audit(row):
    require(row['exit_code'] == 0 and not row['foreign_busy_processes'], 'failed/contaminated run')
    require(row['observations'] > 0, 'missing audit observations')
    require(row['preflight']['quiet_seconds'] >= 60 and row['preflight']['quiet_observations'] >= 20,
            'missing measured quiet interval')


def memory_peak(samples, start, end):
    last = -1.
    warm = []
    for row in samples:
        require(len(row) == 3, 'memory schema')
        for value in row: finite(value, 'memory value')
        require(row[0] > last, 'unordered memory observations')
        last = row[0]
        if start <= row[0] <= end: warm.append(row[1])
    require(bool(warm), 'no warm physical memory observations')
    return max(warm)


def phase_peaks(samples, phases):
    times=[r[0] for r in samples]
    result=[]
    for phase in phases:
        selected=samples[bisect.bisect_left(times,phase['start_ms']):bisect.bisect_right(times,phase['end_ms'])]
        result.append(phase | dict(observations=len(selected),peak_bytes=max(r[1] for r in selected) if selected else None))
    return result


def validate_phases(row, job):
    phases=row['phases']
    expected=[(name,None) for name in ('fixture','snapshot','warm','retirement')]
    expected += [(name,i) for i in range(job['callers']) for name in ('worker_prepare','first','warmup')]
    actual=[(p['name'],p.get('caller')) for p in phases]
    require(len(actual)==len(set(actual)) and set(actual)==set(expected),'phase membership')
    previous=0.
    for phase in phases:
        finite(phase['start_ms'],'phase start');finite(phase['end_ms'],'phase end')
        require(previous<=phase['start_ms']<phase['end_ms'],'phase ordering')
        previous=phase['end_ms']
        if phase['name']=='warm':
            require(phase['start_ms']==row['warm_start_ms'] and phase['end_ms']==row['warm_end_ms'],'warm phase mismatch')


def validate_record(row, job):
    require(row['job'] == job, 'job changed')
    require(row['protocol'] == contract.protocol(job), 'controls changed')
    require(row['environment'] == contract.expected_environment(job), 'environment changed')
    require(row['outcome'] in ('success', 'oom'), 'unknown outcome')
    if row['outcome'] == 'oom':
        require(bool(row['error']) and row['phase'] in ('prepare', 'first', 'warmup', 'warm'), 'OOM provenance')
        return
    require(row['warmup_requests_per_caller'] == [row['protocol']['warmup_requests_per_caller']]*job['callers'],
            'executed warmup count')
    validate_phases(row, job)
    count = 60 if job['kind'] == 'e2' else 240
    require(row['checks'] == count and row['retained_checks'] == count, 'correctness counts')
    require(row['held_after_close'] == (1 if job['kind'] == 'e2' else 8), 'retirement checks')
    for name in ('completed_ms', 'submission_ms'):
        require(len(row[name]) == (60 if job['kind'] == 'e2' else 30), 'timing membership')
        for value in row[name]: finite(value, name)
    for value in row['warmup_ms']: finite(value, 'warmup cost')
    for name in ('first_ms', 'preparation_ms', 'retirement_ms', 'warm_start_ms', 'warm_end_ms'):
        finite(row[name], name)
    require(row['warm_end_ms'] > row['warm_start_ms'], 'empty warm interval')
    require(len(row['compile_before']) == len(row['compile_after']) == job['callers'], 'compile owner count')
    require(all(a['compile_callbacks'] == b['compile_callbacks']
                for a,b in zip(row['compile_before'],row['compile_after'])), 'warm compile callback')
    memory = row['torch_memory']
    require(memory['max_allocated'] >= memory['allocated'] >= 0 and
            memory['max_reserved'] >= memory['reserved'] >= memory['allocated'], 'Torch memory bounds')
    before, after = row['replay_before'], row['replay_after']
    require(len(before) == len(after), 'region count changed')
    for old, new in zip(before, after):
        require(old['captures'] == new['captures'], 'capture during frozen serving')
        require(new['frozen'] and 0 <= new['retained_bytes'] <= new['max_bytes'], 'replay budget/freeze')
        if job['mode'] != 'on': require(new['replays'] == 0, 'graphs enabled in control')
    if job['kind'] == 'e5':
        requests = row['requests']
        membership([(r['batch'], r['ordinal']) for r in requests], [(b, i) for b in range(30) for i in range(8)])
        for request in requests:
            require(request['caller'] == request['ordinal'] % job['callers'], 'caller mapping')
            for key in ('service_ms', 'queue_ms', 'sojourn_ms', 'cpu_submission_ms'): finite(request[key], key)
        require(row['budget']['requested_free_budget'] == 1024 << 20, 'physical budget changed')
        require((1024 << 20) - (16 << 20) <= row['budget']['after']['free'] <= 1024 << 20,
                'physical budget not achieved')


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def summarize(row, physical_peak):
    job = row['job']
    result = job | dict(outcome=row['outcome'])
    if row['outcome'] != 'success': return result | dict(error=row['error'], phase=row['phase'])
    scale = 8 if job['kind'] == 'e5' else 1
    warm = statistics.median(row['completed_ms']) / scale
    setup=math.fsum([row['preparation_ms'], row['first_ms'], *row['warmup_ms']])
    baseline=row.get('nvml_after_reservation', row['initial_nvml'])['used']
    return result | dict(setup_ms=setup, p50_ms=warm, p95_ms=percentile(row['completed_ms'], .95)/scale,
        warm_replays=sum(b['replays']-a['replays'] for a,b in zip(row['replay_before'],row['replay_after'])),
        warm_fallbacks={k:sum(b.get('fallbacks',{}).get(k,0)-a.get('fallbacks',{}).get(k,0)
                             for a,b in zip(row['replay_before'],row['replay_after']))
                        for k in sorted({k for b in row['replay_after'] for k in b.get('fallbacks',{})})},
        submission_p50_ms=statistics.median(row['submission_ms'])/scale,
        request_submission_p50_ms=statistics.median([r['cpu_submission_ms'] for r in row['requests']]) if job['kind']=='e5' else None,
        request_submission_p95_ms=percentile([r['cpu_submission_ms'] for r in row['requests']], .95) if job['kind']=='e5' else None,
        queue_p95_ms=percentile([r['queue_ms'] for r in row['requests']], .95) if job['kind']=='e5' else None,
        service_p95_ms=percentile([r['service_ms'] for r in row.get('requests', [])], .95) if job['kind']=='e5' else None,
        sojourn_p95_ms=percentile([r['sojourn_ms'] for r in row.get('requests', [])], .95) if job['kind']=='e5' else None,
        preparation_ms=row['preparation_ms'], first_ms=row['first_ms'], warmup_ms=math.fsum(row['warmup_ms']), retirement_ms=row['retirement_ms'],
        capture_warmup_ms=math.fsum(r.get('warmup_ms', 0) for r in row['replay_after']),
        capture_ms=math.fsum(r.get('capture_ms', 0) for r in row['replay_after']),
        retained_bytes=sum(r['retained_bytes'] for r in row['replay_after']),
        pool_reserved_bytes=sum(r.get('pool_reserved_bytes', 0) for r in row['replay_after']),
        physical_peak_bytes=physical_peak, app_increment_peak_bytes=physical_peak-baseline, torch_memory=row['torch_memory'],
        total_ms={str(n): setup + n*warm for n in (1,100,1000)},
        fallbacks={k: sum(r.get('fallbacks', {}).get(k,0) for r in row['replay_after'])
                   for k in sorted({k for r in row['replay_after'] for k in r.get('fallbacks', {})})})


def analyze(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['qualification'], 'pilot is not qualification')
    expected = contract.jobs()
    require(manifest['jobs'] == expected, 'matrix definition changed')
    membership(list(manifest['records']), [j['id'] for j in expected])
    rows = []
    input_identities = {}
    for job in expected:
        saved = manifest['records'][job['id']]
        def read(name, digest):
            path = root / saved[name]
            require(contract.sha(path) == saved[digest], 'record checksum')
            return json.loads(path.read_text())
        record = read('path', 'sha256'); audit = read('audit_path', 'audit_sha256')
        validate_audit(audit); validate_record(record, job)
        require(record['provenance'] == manifest['provenance'], 'provenance mismatch')
        identity = (job['kind'], job['case'])
        require(len(record['input_sha256']) == (1 if job['kind'] == 'e2' else 8), 'input membership')
        require(record['input_sha256'] == input_identities.setdefault(identity, record['input_sha256']),
                'different inputs across matched arms/rounds')
        trace_path = root / saved['path']; trace_path = trace_path.parent / record['memory']['file']
        require(contract.sha(trace_path) == record['memory']['sha256'], 'trace checksum')
        trace = json.loads(gzip.decompress(trace_path.read_bytes()))
        peak = memory_peak(trace['samples'], record['warm_start_ms'], record['warm_end_ms']) if record['outcome']=='success' else None
        require(bool(trace['samples']), 'missing physical memory trace')
        rows.append(summarize(record, peak) | dict(
            whole_process_peak_bytes=max(r[1] for r in trace['samples']),
            whole_process_rss_peak_bytes=max(r[2] for r in trace['samples']),
            phase_memory=phase_peaks(trace['samples'],record['phases'])))
    for excluded in manifest['excluded']:
        for name, digest in excluded['files'].items():
            require(contract.sha(root / excluded['path'] / name) == digest, 'excluded evidence changed')
    return rows


def validate_micro(record):
    require(record['qualification'], 'pilot supplement')
    require(record['compute'] in ('eager', 'inductor') and record['round'] in (1, 2, 3), 'supplement identity')
    require(record['environment'] == contract.expected_environment(dict(callers=1)), 'supplement environment')
    validate_audit(record['audit'] | dict(exit_code=0, preflight=record['preflight']))
    membership([(r['scenario'], r['mode']) for r in record['rows']],
               [(s, m) for s in ('stable', 'varying', 'pressure') for m in contract.MODES])
    for row in record['rows']:
        require(row['checks'] == row['retained_checks'] == 100, 'supplement correctness')
        for name in ('completed_ms', 'submission_ms'):
            require(len(row[name]) == 100, 'supplement timings')
            for value in row[name]: finite(value, name)
        for name in ('preparation_ms', 'first_ms', 'warm_start_ms', 'warm_end_ms'): finite(row[name], name)
        require(0<=row['prepare_start_ms']<row['prepare_end_ms']<=row['first_start_ms']<row['first_end_ms']<=row['warm_start_ms'],
                'supplement cold phase boundaries')
        require(row['warm_end_ms'] > row['warm_start_ms'], 'supplement warm interval')
        if row['mode'] != 'plain':
            before, after = row['before'], row['after']
            require(before['captures'] == after['captures'] and after['frozen'], 'warm supplement capture')
            require(0 <= after['retained_bytes'] <= after['max_bytes'], 'supplement budget')
            require(row['closed']['closed'] and row['closed']['retained_bytes'] == 0, 'supplement retirement')
            if row['mode'] == 'off': require(after['replays'] == 0, 'supplement control replays')


def supplements(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = []
    for compute in ('eager', 'inductor'):
        for round_ in (1, 2, 3):
            directory = root / 'supplements' / f'{compute}-r{round_}'
            record = json.loads((directory / 'record.json').read_text())
            validate_micro(record)
            require(record['compute'] == compute and record['round'] == round_, 'supplement membership')
            require(record['provenance'] == manifest['provenance'], 'supplement provenance')
            trace_path = directory / record['memory']['file']
            require(contract.sha(trace_path) == record['memory']['sha256'], 'supplement memory checksum')
            trace = json.loads(gzip.decompress(trace_path.read_bytes()))
            for row in record['rows']:
                peak = memory_peak(trace['samples'], row['warm_start_ms'], row['warm_end_ms'])
                if round_ == 1 and row['scenario'] == 'stable' and row['mode'] in ('off', 'on'):
                    profile = row['profile']
                    require(contract.sha(directory / profile['file']) == profile['sha256'], 'profile checksum')
                    events = json.loads(gzip.decompress((directory / profile['file']).read_bytes()))['traceEvents']
                    counts = profile['counts']
                    for name in ('cudaGraphLaunch', 'cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC'):
                        require(counts[name] == sum(e.get('name') == name for e in events), 'profile counts')
                    require(counts['gpu_kernel_events'] == sum(e.get('cat') == 'kernel' for e in events) > 0,
                            'missing GPU profiling')
                    require(counts['cudaGraphLaunch'] == (10 if row['mode'] == 'on' else 0), 'profile graph launches')
                after = row['after'] or {}
                rows.append(dict(compute=compute, round=round_, scenario=row['scenario'], mode=row['mode'],
                    p50_ms=statistics.median(row['completed_ms']), p95_ms=percentile(row['completed_ms'], .95),
                    max_ms=max(row['completed_ms']), total_warm_ms=math.fsum(row['completed_ms']),
                    submission_p50_ms=statistics.median(row['submission_ms']), preparation_ms=row['preparation_ms'],
                    first_ms=row['first_ms'], retained_bytes=after.get('retained_bytes', 0),
                    pool_reserved_bytes=after.get('pool_reserved_bytes', 0), physical_peak_bytes=peak,
                    fallbacks=after.get('fallbacks', {}), profile=row.get('profile'),
                    phase_memory=phase_peaks(trace['samples'],[dict(name=name,start_ms=row[start],end_ms=row[end])
                        for name,start,end in (('prepare','prepare_start_ms','prepare_end_ms'),
                                              ('first','first_start_ms','first_end_ms'),('warm','warm_start_ms','warm_end_ms'))])))
    return rows


def comparisons(rows):
    fields = ('kind', 'case', 'compute', 'callers')
    groups = sorted({tuple(r[k] for k in fields) for r in rows})
    result = []
    for key in groups:
        members = [r for r in rows if tuple(r[k] for k in fields) == key]
        def arm(runtime, mode):
            selected = [r for r in members if r['runtime'] == runtime and r['mode'] == mode]
            require(len(selected) == 3, 'three-round arm membership')
            successful = [r for r in selected if r['outcome'] == 'success']
            if len(successful) != 3: return dict(successful_rounds=len(successful))
            return dict(successful_rounds=3, replay_active_rounds=sum(r['warm_replays']>0 for r in successful),
                        **{k: statistics.median(r[k] for r in successful) for k in
                        ('p50_ms', 'p95_ms', 'setup_ms', 'retained_bytes', 'app_increment_peak_bytes')})
        summary = {f'{rt}_{mode}': arm(rt, mode) for rt in ('torch', 'sym') for mode in contract.MODES}
        relative = {}
        for rt in ('torch', 'sym'):
            on = summary[f'{rt}_on']
            for control in ('plain', 'off'):
                off = summary[f'{rt}_{control}']
                if min(on['successful_rounds'], off['successful_rounds']) == 3:
                    saving = off['p50_ms'] - on['p50_ms']
                    relative[f'{rt}_on_vs_{control}'] = dict(speedup=off['p50_ms']/on['p50_ms'],
                        break_even_reuses=max(0, math.ceil((on['setup_ms']-off['setup_ms'])/saving)) if saving > 0 else None)
        for mode in contract.MODES:
            sym, torch = summary[f'sym_{mode}'], summary[f'torch_{mode}']
            if min(sym['successful_rounds'], torch['successful_rounds']) == 3:
                relative[f'sym_vs_torch_{mode}'] = dict(speedup=torch['p50_ms']/sym['p50_ms'])
        result.append(dict(zip(fields, key)) | dict(arms=summary, comparisons=relative))
    return result


def parent_targets(rows):
    """Parent #247: 5% below BOTH matched controls in EVERY independent round."""
    result=[]
    identities=sorted({(r['case'],r['compute'],r['mode']) for r in rows
                       if r['kind']=='e2' and r['runtime']=='sym'})
    for case,compute,mode in identities:
        outcomes=[]
        for round_ in (1,2,3):
            candidates=[r for r in rows if r['kind']=='e2' and r['case']==case and
                        r['mode']==mode and r['round']==round_ and r['outcome']=='success']
            sym=[r for r in candidates if r['runtime']=='sym' and r['compute']==compute]
            controls=[r for r in candidates if r['runtime']=='torch']
            available=len(sym)==1 and len(controls)==2 and {r['compute'] for r in controls}=={'eager','inductor'}
            threshold=.95*min(r['p50_ms'] for r in controls) if available else None
            outcomes.append(dict(round=round_,threshold_ms=threshold,
                sym_p50_ms=sym[0]['p50_ms'] if sym else None,
                passed=bool(available and sym[0]['p50_ms']<=threshold)))
        result.append(dict(case=case,compute=compute,mode=mode,rounds=outcomes,
                           passed=all(r['passed'] for r in outcomes),rounds_passed=sum(r['passed'] for r in outcomes)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); rows=analyze(args.root)
    contract.save(args.output, rows)
    contract.save(args.output.with_name('comparisons.json'), comparisons(rows))
    contract.save(args.output.with_name('parent-targets.json'), parent_targets(rows))
    contract.save(args.output.with_name('supplements.json'), supplements(args.root))
    table=['# Issue 268 per-round measurements', '', '| Experiment | Case | Compute | Callers | Round | Runtime | Mode | p50 ms | p95 ms | Outcome |',
           '|---|---|---|---:|---:|---|---|---:|---:|---|']
    for r in rows:
        table.append('| '+' | '.join(str(r.get(k, '—')) if k not in ('p50_ms','p95_ms') else f"{r[k]:.6f}" if k in r else '—'
                     for k in ('kind','case','compute','callers','round','runtime','mode','p50_ms','p95_ms','outcome'))+' |')
    args.output.with_suffix('.md').write_text('\n'.join(table)+'\n')


if __name__=='__main__': main()
