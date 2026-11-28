"""Read-only replay of complete E1–E5 records from the November archive."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import median


MANIFEST = 'd23e45465fbf7d5e188a4fdc0e2b5320d815a777d07c4eaa054361be9a81c741'
ASSESSMENT = '10f2faa59326ad0d4e92fe2a76a3c57c45ba468245a0fc97d18c6a504fe3cf11'
REVISION = '2617ba166f98a7476a5f472d829a573317fc0cfb'
WIKI = 'https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04'
E1, E2, E3, E4, E5 = ('01_folding', '02_fp16_attribution', '03_dynamic_shapes',
                     '04_memory_budget', '05_lifecycle_concurrency')
FAMILIES = ('roundtrip1', 'roundtrip2', 'roundtrip4', 'rank3_chain',
            'dequant_transpose', 'padded_roundtrip')
FOLD_ARMS = ('sym_unfused', 'sym_folded', 'sym_no_canonicalize',
             'manual_fused', 'torch_eager', 'torch_inductor')
MODELS = ('llm_decode', 'llm_prefill', 'moe_sparse', 'moe_dense')
COMPUTE = ('eager', 'inductor')
E3_CASES = ('layout_cast', 'kv_restore', 'dequant_live', 'index_live')
E3_ARMS = ('torch', 'inductor', 'sym_auto_eager', 'sym_auto_inductor',
           'sym_fast_reprepare', 'sym_fast_cache')
E3_SEQUENCE = [128, 256, 128, 33, 512, 33, 1, 128, 0, 65, 256, 65]
E4_CASES = tuple(product((16, 32), (16, 512), (512, 1024)))
E4_ARMS = ('resident', 'partial_torch', 'partial_sym', 'offload_torch', 'offload_sym')
E5_MODELS = ('llm_decode', 'moe_sparse')
MIB = 1 << 20
FIVE_ROUNDS = range(1, 6)
THREE_ROUNDS = range(1, 4)


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_archive(root, *, expected=MANIFEST):
    manifest = root / 'SHA256-MANIFEST.json'
    require(hash_file(manifest) == expected, 'archive manifest digest differs')
    index = read(manifest)
    require(index['algorithm'] == 'sha256', 'unexpected manifest algorithm')
    entries = {row['path']: row for row in index['files']}
    require(len(entries) == len(index['files']), 'duplicate paths in inventory')
    files = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if (path.is_file() and path != manifest
                and {'__pycache__', '.pytest_cache'}.isdisjoint(relative.parts)):
            files[relative.as_posix()] = path
    require(files.keys() == entries.keys(), 'archive file inventory differs')
    for name, row in entries.items():
        path = files[name]
        require(path.stat().st_size == row['bytes'] and hash_file(path) == row['sha256'],
                f'archive file changed: {name}')
    return dict(files=len(entries), bytes=sum(row['bytes'] for row in entries.values()),
                manifest_sha256=expected)


def tail(values, *, interpolate):
    ordered = sorted(values)
    if not interpolate:
        return ordered[math.ceil(len(ordered) * .95) - 1]
    offset = .95 * (len(ordered) - 1)
    lower, upper = math.floor(offset), math.ceil(offset)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (offset - lower)


def valid_times(values, count, label, *, allow_zero=False):
    require(len(values) == count and all(math.isfinite(v) and (v >= 0 if allow_zero else v > 0) for v in values),
            f'invalid sample count or value: {label}')
    return values


def distribution(values, *, interpolate=False):
    return dict(p50_ms=median(values), p95_ms=tail(values, interpolate=interpolate))


def contrast(control, candidate, *, higher_better=False):
    require(control.keys() == candidate.keys(), 'paired arms must have the same rounds')
    rounds = sorted(control)
    if not rounds:
        return dict(paired_rounds=[], classification='no paired success')
    a, b = [control[r] for r in rounds], [candidate[r] for r in rounds]
    valid_times(a, len(rounds), 'paired control')
    valid_times(b, len(rounds), 'paired candidate')
    require(len(rounds) <= 5, 'exact bootstrap intended for at most five rounds')
    sign = 1 if higher_better else -1
    ratios = [y / x for x, y in zip(a, b)]
    reductions = [100 * sign * (ratio - 1) for ratio in ratios]
    bootstrap = sorted(median(replicate) for replicate in product(reductions, repeat=len(rounds)))
    lo = bootstrap[math.ceil(.025 * len(bootstrap)) - 1]
    hi = bootstrap[math.ceil(.975 * len(bootstrap)) - 1]
    classification = 'inconclusive'
    if len(rounds) == 1:
        classification = 'single pair'
    elif lo > 0:
        classification = 'benefit'
    elif hi < 0:
        classification = 'regression'
    return dict(control_median=median(a), candidate_median=median(b),
                candidate_over_control=median(b) / median(a),
                median_of_paired_ratios=median(ratios),
                paired_effect_pct=median(reductions), paired_bootstrap95_pct=[lo, hi],
                paired_rounds=rounds, control_values=a, candidate_values=b, pair_effects_pct=reductions,
                paired_benefits=sum(v > 0 for v in reductions), paired_regressions=sum(v < 0 for v in reductions),
                paired_ties=sum(v == 0 for v in reductions), classification=classification)


class Replay:
    def __init__(self, root):
        self.root = root
        self.provenance = hash_file(root / '00_protocol/provenance.json')
        require(read(root / '00_protocol/provenance.json')['revision'] == REVISION,
                'common source revision differs')
        self.panels = defaultdict(dict)
        self.comparisons = defaultdict(list)
        self.checks = Counter()
        self.workers = Counter()

    def records(self, panel, directory, pattern):
        for path in sorted((self.root / directory).glob(pattern)):
            if path.name.endswith('.run.json'):
                continue
            row = read(path)
            receipt = read(path.with_suffix('.run.json'))
            require(receipt['exit_code'] == 0 and receipt['foreign_busy_processes'] == [],
                    f'invalid process audit: {path}')
            if panel != 'E1':
                require(row['complete'] and row['common_provenance_sha256'] == self.provenance,
                        f'incomplete record or changed provenance: {path}')
            self.workers[panel] += 1
            yield path.relative_to(self.root).as_posix(), row

    def add(self, panel, cell, arm, round_number, path, metrics, **facts):
        key = (cell, arm, round_number)
        require(key not in self.panels[panel], f'duplicate record: {panel} {key}')
        self.panels[panel][key] = dict(cell=cell, arm=arm, round=round_number, raw=path,
                                       metrics=metrics, **facts)

    def check_grid(self, panel, expected):
        require(self.panels[panel].keys() == set(expected), f'changed or incomplete matrix: {panel}')

    def compare(self, panel, cell, control, candidate, metrics, category):
        a = {r: row['metrics'] for (c, arm, r), row in self.panels[panel].items() if (c, arm) == (cell, control)}
        b = {r: row['metrics'] for (c, arm, r), row in self.panels[panel].items() if (c, arm) == (cell, candidate)}
        require(a and a.keys() == b.keys(), f'missing or unpaired arms: {panel} {cell} {control} {candidate}')
        present = [r for r in sorted(a) if all(metric in a[r] and metric in b[r] for metric in metrics)]
        out = {}
        for metric in metrics:
            out[metric] = contrast({r: a[r][metric] for r in present}, {r: b[r][metric] for r in present},
                                   higher_better=metric == 'within_burst_rps')
        self.comparisons[panel].append(dict(cell=cell, category=category, control=control, candidate=candidate,
                                          excluded_unpaired_success=sorted(a.keys() - set(present)), metrics=out))


def folding(replay):
    cells = [(family, n, workers, mode) for family, n, workers, mode in
             product(FAMILIES, (33, 512, 8192), (1, 8), ('completed_h2d', 'producer'))]
    for path, row in replay.records('E1', f'{E1}/acceptance', 'round*-t*.json'):
        meta, cases = row['metadata'], row['cases']
        config = meta['configuration']
        workers, round_number = config['threads'], config['round']
        require(meta['source_revision'] == REVISION and (config['samples'], config['warmup']) == (60, 5),
                f'changed E1 protocol: {path}')
        for label, case in cases.items():
            require(case['checks'] == 720 and case['h2d_copies_per_call'] == 1 and case['threads'] == workers
                    and set(case['paths']) == set(FOLD_ARMS), f'invalid E1 controls: {path} {label}')
            replay.checks['E1'] += case['checks']
            for arm, data in case['paths'].items():
                for mode in ('completed_h2d', 'producer'):
                    obs = data[mode]
                    samples = valid_times(obs['samples_ms'], 60, f'{path} {label} {mode} {arm}')
                    m = distribution(samples, interpolate=True)
                    require(all(math.isclose(m[k], obs[k], rel_tol=1e-10) for k in m),
                            f'E1 archived p50/p95 disagree: {path} {mode} {arm}')
                    replay.add('E1', (case['family'], case['n'], workers, mode), arm, round_number,
                               path, m, wire_bytes=case['wire_bytes'],
                               explicit_intermediates_bytes=case['intermediate_allocated_bytes'] if arm == 'sym_unfused' else None,
                               first_producer_ms=data['first_producer_ms'], preparation=case['preparation'])
    replay.check_grid('E1', product(cells, FOLD_ARMS, FIVE_ROUNDS))
    for cell in cells:
        for arm in FOLD_ARMS:
            if arm != 'sym_folded':
                replay.compare('E1', cell, arm, 'sym_folded', ('p50_ms', 'p95_ms'), 'folded versus control')


def e2_arms(dtype, case):
    groups, windows = ('grouped',), ('whole',)
    if (dtype, case) == ('float16', 'llm_decode'):
        groups, windows = ('grouped', 'split'), ('whole', 'step')
    return {f'{rt}_{c}_native_{g}_{w}' for rt, c, g, w in product(('torch', 'sym'), COMPUTE, groups, windows)} | {
        f'torch_{c}_arithmetic_grouped_whole' for c in COMPUTE}


def attribution(replay):
    cells = [('float16', model) for model in MODELS] + [('float32', 'llm_decode')]
    for path, row in replay.records('E2', f'{E2}/acceptance', 'round*-float*.json'):
        config, protocol = row['configuration'], row['protocol']
        dtype = config['dtype']
        require((config['samples'], config['warmup'], config['only_compute']) == (60, 3, None)
                and (protocol['slots'], protocol['native_scratch_limit']) == (2, 128 * MIB)
                and not protocol['cudagraphs'], f'changed E2 protocol: {path}')
        for case, values in row['cases'].items():
            require(set(values['arms']) == e2_arms(dtype, case) and values['warm_compile_callbacks'] == 0,
                    f'changed E2 arms or compiled in warm window: {path} {case}')
            payloads = set()
            for arm, data in values['arms'].items():
                require(data['checks'] == 60, f'E2 missing correctness checks: {path} {case} {arm}')
                before, after = (data['diagnostics'][phase] for phase in ('before', 'after'))
                scopes = ('native',) if arm.startswith('sym_') else ('torch_weights', 'torch_kv')
                payload = sum(after[scope]['payload_bytes'] - before[scope]['payload_bytes'] for scope in scopes)
                payloads.add(payload)
                if case == 'llm_decode':
                    expected = 6 if data['factors']['grouping'] == 'grouped' else 48
                    d = lambda scope, field: after[scope][field] - before[scope][field]
                    require(d('native', 'groups') == 16 + expected if scopes == ('native',)
                            else (d('torch_weights', 'groups'), d('torch_kv', 'copy_calls')) == (16, expected),
                            f'changed E2 grouping: {path} {arm}')
                samples = valid_times(data['samples_ms'], 60, f'{path} {arm}')
                m = distribution(samples)
                require(m['p50_ms'] == data['median_ms'], f'E2 archived p50 differs: {path} {arm}')
                replay.checks['E2'] += 60
                replay.add('E2', (dtype, case), arm, config['round'], path,
                           m | {'first_ms': data['first_ms']}, payload_bytes=payload,
                           snapshot_ms={k: values[k] for k in ('torch_common_snapshot_ms', 'sym_snapshot_and_preparation_ms')})
            require(len(payloads) == 1 and next(iter(payloads)) > 0, f'unmatched E2 payloads: {path} {case}')
    replay.check_grid('E2', ((cell, arm, r) for cell in cells for arm in e2_arms(*cell) for r in FIVE_ROUNDS))
    for cell in cells:
        for arm in sorted(e2_arms(*cell)):
            if arm.startswith('sym_'):
                replay.compare('E2', cell, arm.replace('sym_', 'torch_', 1), arm,
                               ('p50_ms', 'p95_ms'), 'runtime')
        for mode in COMPUTE:
            replay.compare('E2', cell, f'torch_{mode}_arithmetic_grouped_whole',
                           f'torch_{mode}_native_grouped_whole', ('p50_ms', 'p95_ms'), 'native conversion path')
    cell = ('float16', 'llm_decode')
    for rt, mode, window in product(('torch', 'sym'), COMPUTE, ('whole', 'step')):
        replay.compare('E2', cell, f'{rt}_{mode}_native_split_{window}', f'{rt}_{mode}_native_grouped_{window}',
                       ('p50_ms', 'p95_ms'), 'KV grouping')
    for rt, mode, group in product(('torch', 'sym'), COMPUTE, ('grouped', 'split')):
        replay.compare('E2', cell, f'{rt}_{mode}_native_{group}_step', f'{rt}_{mode}_native_{group}_whole',
                       ('p50_ms', 'p95_ms'), 'prefetch window')


def changing_shapes(replay):
    for path, row in replay.records('E3', f'{E3}/acceptance', 'round[0-9]*.json'):
        config = row['configuration']
        require(config['epochs'] == 5 and set(config['paths']) == set(E3_ARMS)
                and row['sequence'] == E3_SEQUENCE
                and all(x['values_and_mutation_correct'] for x in row['coverage'].values()),
                f'changed E3 trace or failed coverage witness: {path}')
        for case, record in row['cases'].items():
            requests = record['requests']
            require([r['n'] for r in requests] == E3_SEQUENCE * 5
                    and [r['step'] for r in requests] == list(range(60)), f'changed sequence: {path} {case}')
            require(all(set(r['paths']) == set(E3_ARMS) for r in requests), f'missing E3 request path: {path} {case}')
            for arm in E3_ARMS:
                observations = [req['paths'][arm] for req in requests]
                require(all(x['correct'] for x in observations), f'E3 oracle failure: {path} {case} {arm}')
                times = valid_times([x['completed_ms'] for x in observations], 60, f'{path} {case} {arm}')
                final = record['final_stats'][arm]
                require(observations[11]['after']['dynamo_callbacks'] == final['dynamo_callbacks'],
                        f'E3 recompiled after first cycle: {path} {case} {arm}')
                replay.checks['E3'] += 60
                replay.add('E3', case, arm, config['round'], path, distribution(times[12:]) |
                           dict(first_cycle_ms=sum(times[:12]), total60_ms=sum(times), repeated48_ms=sum(times[12:])),
                           compiler_callbacks=final['dynamo_callbacks'], compiler_callback_ms=final['callback_ms'],
                           backend=final['backend'], explicit=final['explicit'])
    replay.check_grid('E3', product(E3_CASES, E3_ARMS, FIVE_ROUNDS))
    comparisons = (
        ('torch', 'sym_auto_eager', 'automatic eager'),
        ('torch', 'sym_auto_inductor', 'automatic Inductor versus eager, original reference'),
        ('inductor', 'sym_auto_inductor', 'automatic Inductor versus matched compute'),
        ('sym_fast_reprepare', 'sym_fast_cache', 'explicit cache'),
    )
    for case in E3_CASES:
        for control, candidate, category in comparisons:
            replay.compare('E3', case, control, candidate,
                           ('p50_ms', 'p95_ms', 'first_cycle_ms', 'repeated48_ms', 'total60_ms'), category)


def capacity(replay):
    for panel, directory in (('E4', 'acceptance'), ('E4 packed', 'supplement-resident-packed')):
        for path, row in replay.records(panel, f'{E4}/{directory}', 'round*-L*-S*-B*-*.json'):
            config, budget = row['configuration'], row['budget']
            cell = (config['layers'], config['context'], config['budget_mib'])
            goal = cell[-1] * MIB
            require(row['physical_meter_valid'] and budget['requested_free_budget'] == goal
                    and goal - 16 * MIB <= budget['after']['free'] <= goal
                    and (config['samples'], config['warmup']) == (20, 3), f'invalid E4 budget: {path}')
            require(row['outcome'] in ('success', 'oom'), f'invalid E4 outcome: {path}')
            arm = config['path']
            if panel == 'E4 packed':
                policy = row['supplemental_policy']
                require(arm == 'resident' and policy['name'] == 'resident_packed_serial_preload'
                        and policy['preload_window_restored'], f'invalid packed control: {path}')
                arm = 'resident_packed'
            metrics = {}
            timed_counts = {}
            if row['outcome'] == 'success':
                times = valid_times(row['samples_ms'], 20, path)
                metrics = distribution(times) | {'first_ms': row['first_request_ms']}
                require(row['checks'] == 20 and metrics['p50_ms'] == row['median_ms'], f'invalid E4 checks: {path}')
                replay.checks[panel] += 20
                if arm.startswith(('partial_', 'offload_')):
                    before, after = [row[phase]['resources'] for phase in ('warm_before', 'warm_after')]
                    if arm.endswith('_sym'):
                        before, after = before['native'], after['native']
                    timed_counts = {name: after[name] - before[name] for name in ('groups', 'payload_bytes')}
                    expected = 20 * (cell[0] - (8 if arm.startswith('partial_') else 0))
                    require(timed_counts['groups'] == expected and timed_counts['payload_bytes'] > 0,
                            f'changed E4 payload/group contract: {path}')
            else:
                require(row.get('oom_phase') in ('preparation_start', 'residency_start',
                        'first_request', 'warmup_request', 'timed_request'), f'invalid E4 OOM phase: {path}')
            replay.add(panel, cell, arm, config['round'], path, metrics, outcome=row['outcome'],
                oom_phase=row.get('oom_phase'), achieved_mib=budget['after']['free'] / MIB,
                peak_mib=row['observed_application_device_peak_above_initial_context'] / MIB,
                pinned_checkpoint_mib=row['owned_pinned_checkpoint_bytes'] / MIB,
                rss_mib=row['physical_memory']['observed_process_rss_peak'] / MIB,
                snapshot_preparation_ms=row.get('snapshot_preparation_ms'), preload_ms=row.get('residency_preload_ms'),
                phase_timestamps_ms=row['phases'], timed_counts=timed_counts)
    replay.check_grid('E4', product(E4_CASES, E4_ARMS, THREE_ROUNDS))
    replay.check_grid('E4 packed', product(((16, 16, 512), (16, 512, 512)), ('resident_packed',), THREE_ROUNDS))
    for cell, placement in product(E4_CASES, ('partial', 'offload')):
        control, candidate = placement + '_torch', placement + '_sym'
        for r in THREE_ROUNDS:
            t, s = (replay.panels['E4'][cell, arm, r]['timed_counts'] for arm in (control, candidate))
            require(not (t and s) or t == s, f'unmatched E4 timed payload/groups: {cell} round {r}')
        replay.compare('E4', cell, control, candidate, ('p50_ms', 'p95_ms', 'first_ms'), placement)


def check_closed(stats, path):
    typed = stats['typed']
    require(stats['closed'] and typed['closed'] and not any(
        typed[k] for k in ('retained_bytes', 'live_bytes', 'background_workers', 'streams')),
        f'native resources did not retire: {path}')


def lifecycle(replay):
    for path, row in replay.records('E5 lifecycle', f'{E5}/acceptance', 'lifecycle-r*-*.json'):
        c, phases = row['configuration'], row['phases']
        requests = row['requests']
        require([r['phase'] for r in requests] == ['initial'] * 100 + ['shape_change'] * 10 + ['scale_change'] * 10
                and all(r['correct'] for r in requests) and row['held_output_checks_after_close'] == 4,
                f'E5 lifecycle checks or phases changed: {path}')
        if c['runtime'] == 'sym':
            check_closed(row['native_after_close'], path)
            require(row['before_retirement']['snapshots']['revisions'] == row['initial_stats']['snapshots']['revisions'] + 1,
                    f'E5 snapshot did not advance: {path}')
        times = valid_times([r['completed_ms'] for r in requests], 120, path)
        setup = row['launch_to_pre_cuda_ms'] + sum(phases[k] for k in
                    ('cuda_init_ms', 'fixture_ms', 'snapshot_total_ms', 'request_owner_ms'))
        retirement = phases['owner_retirement_ms'] + phases['caller_release_and_cache_clear_ms']
        full = setup + sum(times) + phases['shape_update_ms'] + phases['scale_reprepare_ms'] + retirement
        replay.checks['E5 lifecycle'] += 120
        replay.add('E5 lifecycle', (c['case'], c['compute']), c['runtime'], c['round'], path,
            distribution(times[1:100]) | dict(setup_ms=setup, first_ms=times[0], prefix10_ms=setup + sum(times[:10]),
                prefix100_ms=setup + sum(times[:100]), shape_first_ms=times[100], scale_first_ms=times[110],
                retirement_ms=retirement, full120_ms=full),
            phases_ms=phases, launch_to_pre_cuda_ms=row['launch_to_pre_cuda_ms'],
            peak_mib=row['observed_deployment_device_peak'] / MIB, after_release=row['after_release'])
    cells = tuple(product(E5_MODELS, COMPUTE))
    replay.check_grid('E5 lifecycle', product(cells, ('torch', 'sym'), THREE_ROUNDS))
    for cell in cells:
        replay.compare('E5 lifecycle', cell, 'torch', 'sym',
            ('p50_ms', 'p95_ms', 'setup_ms', 'first_ms', 'shape_first_ms', 'scale_first_ms', 'retirement_ms', 'full120_ms'),
            'matched lifecycle')


def bursts(replay):
    offered_inputs = {}
    for path, row in replay.records('E5 burst', f'{E5}/acceptance', 'concurrency-r*-*.json'):
        c, protocol = row['configuration'], row['protocol']
        batches = row['batches']
        require(len(batches) == 30 and all([r['input_index'] for r in b['requests']] == list(range(8)) for b in batches)
                and row['checks'] == 240 and row['held_output_checks_after_close'] == 8,
                f'E5 burst geometry or held output checks changed: {path}')
        inputs = row['input_sha256']
        require(len(inputs) == len(set(inputs)) == 8 and offered_inputs.setdefault(c['case'], inputs) == inputs,
                f'E5 offered request set differs: {path}')
        budget = row['budget']
        require(row['physical_meter_valid'] and budget['requested_free_budget'] == 1024 * MIB
                and 1008 * MIB <= budget['after']['free'] <= 1024 * MIB
                and protocol['torch_threads_per_caller'] * c['callers'] == 8
                and protocol['aggregate_native_scratch_bytes'] == 128 * MIB, f'E5 burst budget differs: {path}')
        if c['runtime'] == 'sym':
            require(len(row['native_after_close']) == c['callers'], f'E5 missing owner close: {path}')
            for stats in row['native_after_close']:
                check_closed(stats, path)
        requests = [r for batch in batches for r in batch['requests']]
        require(all(r['correct'] for r in requests), f'E5 burst output check failed: {path}')
        m = {}
        for label, field in (('service', 'service_ms'), ('sojourn', 'sojourn_ms'),
                             ('executor', 'executor_execution_ms'), ('queue', 'queue_wait_ms')):
            observations = valid_times([r[field] for r in requests], 240, path, allow_zero=label == 'queue')
            m.update({f'{label}_{k}': v for k, v in distribution(observations).items()})
        elapsed = sum(valid_times([b['completed_ms'] for b in batches], 30, path))
        replay.checks['E5 burst'] += 240
        replay.add('E5 burst', (c['case'], c['callers']), c['runtime'], c['round'], path,
            m | {'within_burst_rps': 240000 / elapsed}, achieved_mib=budget['after']['free'] / MIB,
            peak_mib=row['observed_warm_app_increment_peak'] / MIB, after_release=row['after_release'])
    cells = tuple(product(E5_MODELS, (1, 2, 4)))
    replay.check_grid('E5 burst', product(cells, ('torch', 'sym'), THREE_ROUNDS))
    for cell in cells:
        replay.compare('E5 burst', cell, 'torch', 'sym',
            ('service_p50_ms', 'service_p95_ms', 'sojourn_p50_ms', 'sojourn_p95_ms', 'queue_p95_ms', 'within_burst_rps'),
            'matched finite bursts')


def replay_baseline(root):
    inventory = verify_archive(root)
    require(hash_file(root / 'sym-benefit-assessment.md') == ASSESSMENT, 'original assessment digest differs')
    replay = Replay(root)
    for analyze in (folding, attribution, changing_shapes, capacity, lifecycle, bursts):
        analyze(replay)
    require(replay.checks == {'E1': 129600, 'E2': 12600, 'E3': 7200, 'E4': 2100, 'E4 packed': 60,
                              'E5 lifecycle': 2880, 'E5 burst': 8640}, 'output counts differ from published audit')
    require(replay.workers == {'E1': 10, 'E2': 10, 'E3': 5, 'E4': 120, 'E4 packed': 6,
                              'E5 lifecycle': 24, 'E5 burst': 36}, 'process counts differ from published audit')
    return dict(revision=REVISION, inventory=inventory, common_provenance_sha256=replay.provenance,
                workers=dict(replay.workers), oracle_checks=dict(replay.checks),
                arms={panel: list(rows.values()) for panel, rows in replay.panels.items()},
                contrasts=dict(replay.comparisons))


def table(columns, rows):
    def line(items):
        return '| ' + ' | '.join(map(str, items)) + ' |'

    return '\n'.join([line(columns), line(['---'] * len(columns)), *(line(row) for row in rows)])


def fnum(value):
    return '—' if value is None else f'{value:.3f}'


def paired_cell(obs):
    if not obs['paired_rounds']:
        return 'OOM'
    lo, hi = obs['paired_bootstrap95_pct']
    return (f'{fnum(obs["control_median"])} → {fnum(obs["candidate_median"])} '
            f'({obs["paired_effect_pct"]:+.2f}% [{lo:+.2f}, {hi:+.2f}])')


def text_key(cell):
    return ' / '.join(map(str, cell)) if isinstance(cell, tuple) else cell


def by_arm(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['cell'], row['arm']].append(row)
    return groups


def observed_median(values):
    actual = [value for value in values if value is not None]
    return median(actual) if actual else None


def metric_median(rows, key):
    return observed_median(row['metrics'].get(key) for row in rows)


def timing_cell(rows):
    return ' / '.join(fnum(metric_median(rows, key)) for key in ('p50_ms', 'p95_ms'))


def pair_table(rows, keys):
    return table(('Cell', 'Contrast', 'Control → candidate', *keys), [
        (text_key(row['cell']), row['category'], f'{row["control"]} → {row["candidate"]}',
         *(paired_cell(row['metrics'][metric]) for metric in keys))
        for row in rows
    ])


def summarize(result):
    arms, comparisons = result['arms'], result['contrasts']
    checked = sum(result['oracle_checks'].values())
    e1 = [row for row in comparisons['E1'] if row['cell'][-1] == 'completed_h2d']
    e1_index = {(row['cell'], row['control']): row['metrics'] for row in e1}
    e1_cells = list(dict.fromkeys(row['cell'] for row in e1))
    e1_wins = sum(e1_index[cell, 'sym_unfused']['p50_ms']['paired_effect_pct'] > 0 for cell in e1_cells)
    manual = sum(e1_index[cell, 'manual_fused']['p50_ms']['candidate_over_control'] > 1 for cell in e1_cells)
    primary = [d for d in comparisons['E2'] if d['cell'][0] == 'float16' and d['category'] == 'runtime'
               and d['control'].endswith('grouped_whole')]
    slow = [(row['metrics']['p50_ms']['median_of_paired_ratios'] - 1) * 100 for row in primary]
    slow_tail = [(row['metrics']['p95_ms']['median_of_paired_ratios'] - 1) * 100 for row in primary]
    e4_ooms = sum(d['outcome'] == 'oom' for d in arms['E4'])
    life_losses = sum(row['metrics']['full120_ms']['paired_regressions'] for row in comparisons['E5 lifecycle'])
    burst_losses = sum(row['metrics']['within_burst_rps']['paired_regressions'] for row in comparisons['E5 burst'])
    intro = f"""# Replaying the matched E1–E5 baseline

The original campaign isolates a benefit from folding and selected reuse.
The matched model, capacity and lifecycle comparisons retain adverse outcomes.

- E1 folding beats true materialization in **{e1_wins}/36** cells by paired median reduction.
  The manual loop has a lower p50 median of process medians in **{manual}/36**.
- E2 Sym takes **{min(slow):.1f}–{max(slow):.1f}% longer** in the eight principal FP16 comparisons
  against Torch with the same native kernels, grouping and prefetch. The
  derived p95 comparison takes **{min(slow_tail):.1f}–{max(slow_tail):.1f}% longer** across those cells.
- E3 has zero additional Dynamo callbacks after the first cycle. The explicit cache and
  captured execution use different paths. Automatic execution trails the archived eager control.
- E4 retains **{e4_ooms}** OOMs in the fixed placement grid. Both all-offload arms fit.
  At 32 layers, context 512, eight resident layers and a 512 MiB budget,
  Torch succeeds and Sym OOMs in all three rounds.
- E5 increases full-lifecycle cost in **{life_losses}/12** paired rounds and reduces
  within-burst throughput in **{burst_losses}/18**. Held-output and native-close checks pass.

## Reproduction and estimators

The replay verifies all **{result['inventory']['files']:,}** published inventory
files and audits **{checked:,}** recorded output checks. It reads the raw latency
vectors, process phase counters and dispositions on CPU with Python 3.11+.
[Versioned archive, original analyses and benefit assessment]({WIKI})

~~~sh
python -m bench.issue274.baseline_analysis /path/to/sym_experiments_261104 \\
  --output /tmp/BASELINE_RESULTS.md --json-output /tmp/baseline-analysis.json
~~~

Sym revision `{REVISION}`. The immutable manifest SHA-256 is
`{MANIFEST}`. Common build and environment are pinned by
`00_protocol/provenance.json` (SHA-256 `{result['common_provenance_sha256']}`).
Original runs use the EPYC 7351, RTX 2080 Ti GPU 0, CUDA 12.6, Torch 2.14.0+cu126,
Python 3.14.7 and CPU affinity 4–7,20–23. The archive records commands, driver
and clocks, native/compiler hashes, stream settings and per-arm budgets.
The output JSON gives the raw path and phase measurements for every arm/round.

Each p50/p95 is computed inside its process first. The displayed arm
summaries are medians of those process statistics. E1 uses its original
linear p95. E5 bursts use the original nearest-rank p95, also used for the
newly derived E2, E3, E4 and E5 lifecycle tails. At 20 samples this is the
nineteenth order statistic.

Each paired percentage is the median of same-round effects. For latency
the effect is 100 × (1 − candidate/control). For throughput it is
100 × (candidate/control − 1). Positive favors the candidate. The JSON
separately reports candidate/control as a ratio of arm medians and as a
median of same-round ratios. Table brackets give exploratory 95% percentile
intervals from exhaustive resampling of three or five paired processes,
with requests kept inside their processes. These unadjusted intervals have
coarse resolution. A band that touches zero is inconclusive. The JSON retains
individual pairs and ties.
An OOM yields no latency comparison for that pair.

## E1 · True materialization

All cells use one pinned wire and one completed H2D copy. The genuine
materialized control writes each operation's output into separate storage
inside one native invocation. Disabling canonicalization retains folding.
The manual producer uses F16C and can exploit the roundtrip idempotence
allowed by the value contract. Shared warm buffers, one or eight workers,
five independent rounds, five warmups and 60 samples per arm define the
boundary. The JSON also contains producer-only comparisons and preparation
including compile, binding and intermediate storage.
"""
    groups = by_arm(arms['E1'])
    labels = ('Unfused', 'Folded', 'No canonicalize', 'Manual', 'Eager', 'Inductor')
    fold_table = table(('Chain / N / workers', *(label + ' p50/p95 ms' for label in labels),
                        'Fold vs unfused p50', 'Fold vs manual p50'), [
        (text_key(cell[:3]), *(timing_cell(groups[cell, arm]) for arm in FOLD_ARMS),
         paired_cell(e1_index[cell, 'sym_unfused']['p50_ms']), paired_cell(e1_index[cell, 'manual_fused']['p50_ms']))
        for cell in e1_cells
    ])
    e2_intro = """## E2 · Matched request attribution

The timer covers the full completed model request. The nominal producer
envelope is 128 MiB with two slots. Inductor arms share compiled consumer
functions. Source snapshot, first calls and diagnostic counters sit outside
the warm 60 samples. These use five process rounds and three warmups with
no warm callbacks. Conversion, grouping and window effects interact.
The archive's factorial interaction table retains those contrasts."""
    e3_intro = """## E3 · Revisit trace and fixed paths

All six arms execute the original 12 requests five times, with empty and
singleton cases, changed stride, live parameters and live indices. Arms
share a process and compiler cache. The first cycle includes compile and
preparation. The repeated distribution uses the final 48 completed calls,
including any fallback. Native execution counts and capture exclusions
describe different event populations. Live dequantization changes its
parameter revision on every eligible explicit call. Both explicit index
arms retain their native descriptor cache, limiting that contrast to the
timing of those specific paths."""
    groups = by_arm(arms['E3'])
    shape_table = table(('Case', 'Arm', 'First cycle ms', 'All 60 ms', 'Repeated 48 ms',
                         'Repeated p50/p95 ms', 'Dynamo callbacks', 'Native calls', 'Explicit prepares / hits'), [
        (case, arm, *(fnum(metric_median(rows, key)) for key in ('first_cycle_ms', 'total60_ms', 'repeated48_ms')),
         timing_cell(rows), int(median(r['compiler_callbacks'] for r in rows)),
         int(median(r['backend']['runtime_executions'] for r in rows)) if rows[0]['backend'] else '—',
         '/'.join(str(int(median(r['explicit'][key] for r in rows)))
                  for key in ('program_preparations', 'program_cache_hits')) if rows[0]['explicit'] else '—')
        for (case, arm), rows in groups.items()
    ])
    return '\n\n'.join((intro.strip(), fold_table, e2_intro, pair_table(comparisons['E2'], ('p50_ms', 'p95_ms')),
        e3_intro, shape_table, pair_table(comparisons['E3'], ('repeated48_ms', 'p95_ms')),
        capacity_report(arms, comparisons), lifecycle_report(arms, comparisons), LIMITS)) + '\n'


def capacity_report(arms, comparisons):
    intro = """## E4 · Physical capacity and OOMs

A direct CUDA reservation sets actual free VRAM after context initialization.
Warm time excludes setup, residency preload, first request, three warmups
and oracle work. Three fresh processes per cell record 20 samples on success.
The packed resident arm is a separate supplement that changes preload and
storage allocation. Success counts and OOM phases stay with each policy."""
    tables = []
    for panel in ('E4', 'E4 packed'):
        groups = by_arm(arms[panel])
        rows = []
        for (cell, arm), observations in sorted(groups.items()):
            outcomes = Counter(r['oom_phase'] for r in observations if r['oom_phase'])
            oom_phases = ', '.join(f'{k} ({v})' for k, v in sorted(outcomes.items()))
            achieved = [r['achieved_mib'] for r in observations]
            rows.append((text_key(cell), arm, f'{len(observations) - sum(outcomes.values())}/{len(observations)}',
                oom_phases or '—', f'{min(achieved):.2f}–{max(achieved):.2f}',
                fnum(observed_median(r['preload_ms'] for r in observations)),
                fnum(metric_median(observations, 'first_ms')), timing_cell(observations),
                f'{max(r["peak_mib"] for r in observations):.1f}'))
        tables.append(table(('Layers / context / budget MiB', 'Policy', 'Success', 'OOM phase (rounds)',
            'Achieved free MiB', 'Preload median ms', 'First ms', 'Warm p50/p95 ms',
            'Max observed app increment MiB'), rows))
    return '\n\n'.join((intro, *tables, pair_table(comparisons['E4'], ('p50_ms', 'p95_ms'))))


def lifecycle_report(arms, comparisons):
    intro = """## E5 · Lifecycle, finite bursts and memory

Full120 charges launch, context, fixtures, snapshots and owners, 100
initial requests, ten after a shape change, ten after a scale revision,
and retirement including held-output release. Inductor's first requests
include lazy compile where observed. Shape changes also change model
work. Warm distributions use initial calls 2–100.

E5's separate concurrent trial applies the 1024 MiB physical reservation
with eight aggregate threads, 128 MiB scratch/staging, and one, two or
four callers. Process statistics cover 30 batches of eight fixed inputs.
Burst throughput excludes validation and gaps between batches. Sojourn
includes queue and execution. Three process pairs per cell describe
finite bursts."""
    groups = by_arm(arms['E5 lifecycle'])
    lifecycle_table = table(('Workload / compute', 'Arm', 'Setup ms', 'First ms', 'Prefix 10 / 100 ms',
                             'Shape first / scale first ms', 'Warm p50/p95 ms', 'Retirement ms', 'Full120 ms',
                             'Observed deployment peak MiB'), [
        (text_key(cell), arm, fnum(metric_median(rows, 'setup_ms')), fnum(metric_median(rows, 'first_ms')),
         ' / '.join(fnum(metric_median(rows, key)) for key in ('prefix10_ms', 'prefix100_ms')),
         ' / '.join(fnum(metric_median(rows, key)) for key in ('shape_first_ms', 'scale_first_ms')),
         timing_cell(rows), fnum(metric_median(rows, 'retirement_ms')), fnum(metric_median(rows, 'full120_ms')),
         f'{median(r["peak_mib"] for r in rows):.1f}')
        for (cell, arm), rows in sorted(groups.items())
    ])
    groups = by_arm(arms['E5 burst'])
    burst_peak_table = table(('Workload / callers', 'Arm', 'Observed warm app increment median MiB'), [
        (text_key(cell), arm, f'{median(r["peak_mib"] for r in rows):.1f}')
        for (cell, arm), rows in sorted(groups.items())
    ])
    tail_scope = """E4 and E5 burst application increments use their post-reservation
baselines. E5 lifecycle peak is an absolute NVML value in its own phase
window. Sampled peaks are lower bounds on transient peak. OOM peaks
describe observation through failure. Torch allocator, native retention,
pinned-host, RSS and physical-device counters have overlapping scopes.
Keep these values separate."""
    return '\n\n'.join((intro, lifecycle_table, pair_table(comparisons['E5 lifecycle'],
        ('full120_ms', 'p50_ms', 'p95_ms')), pair_table(comparisons['E5 burst'],
        ('within_burst_rps', 'service_p50_ms', 'service_p95_ms', 'sojourn_p95_ms', 'queue_p95_ms')),
        burst_peak_table, tail_scope))


LIMITS = """## Follow-up evidence and remaining scope

The frozen campaign uses synthetic workloads on one GPU architecture.
#267 qualifies native kernel quality. #272 links actual capture to the
same isolated kernel comparison and independently varies cache, groups
and the typed-kernel build on the E3 trace. #271 scores a supplied
executable catalog against a measured oracle and retains calibration
costs, arena refusals, regret and finite-arrival deadline misses. Those
studies use their own machines, phase boundaries and process pairs.
[#267 native results](../issue267/RESULTS.md) · [#272 E3 controls](../issue272/RESULTS.md) ·
[#272 captured quality](../issue272/CAPTURED_QUALITY.md) · [#271 planner](../issue271/README.md)

The separate #274 application evaluates pretrained SqueezeNet 1.1 on one
RTX 4070 Ti SUPER with FP32, batch one and a fixed input shape. That campaign
provides a real application boundary outside the original synthetic workloads.
Portability of the same application across hardware, BF16 model behavior,
productivity and sustained open-loop serving still require evidence.
[Application protocol](APPLICATION_PROTOCOL.md) · [Issue 274 synthesis](README.md)"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    root = args.archive.resolve()
    paths = [p.resolve() for p in (args.output, args.json_output) if p]
    require(len(paths) == len(set(paths)) and all(not p.is_relative_to(root) for p in paths),
            'outputs must be distinct and outside the original archive')
    result = replay_baseline(root)
    args.output.write_text(summarize(result))
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
