"""Run and re-score the frozen single-image SqueezeNet application campaign."""
import argparse
import ast
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
from time import perf_counter_ns, time_ns
import types

from bench.issue265.support import digest, environment, runtime_hashes, save

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'APPLICATION_PROTOCOL.md'
ASSETS = HERE / 'application-assets.json'
ARMS = ('eager', 'inductor', 'sym')
ORDER = tuple(permutations(ARMS))
AFFINITY = tuple(range(8))
COLD, WARMUP, WARM = 1, 15, 96
REQUESTS = COLD + WARMUP + WARM
MB = 1 << 20
ALLOCATOR_LIMIT = 4096 * MB
ATOL, RTOL = 1e-4, 2e-4
COMPILE_OPTIONS = {'triton.cudagraphs': False}
TRANSFER_OPTIONS = {'pinning': 'pinned', 'n_streams': 1, 'gather_threads': 8, 'n_buffers': 2}


def nearest_rank(values, percentile):
    data = sorted(values)
    if not data or not 0 < percentile <= 1:
        raise ValueError('a nonempty sample and percentile in (0, 1] are required')
    return data[math.ceil(percentile * len(data)) - 1]


def paired_ratio(treatment, control):
    ratios = [a / b for a, b in zip(treatment, control, strict=True)]
    logs = list(map(math.log, ratios))
    rng = random.Random(274)
    draws = [math.exp(statistics.fmean(rng.choices(logs, k=len(logs)))) for _ in range(20_000)]
    return {
        'ratios_by_round': ratios, 'geomean': math.exp(statistics.fmean(logs)),
        'bootstrap_95': [nearest_rank(draws, p) for p in (.025, .975)],
    }


def verified_assets(folder):
    manifest = json.loads(ASSETS.read_text())
    files = {}
    for name, spec in (('source', manifest['model']['source']),
                       ('weights', manifest['model']['weights']), ('photo', manifest['photo'])):
        path = folder / spec['file']
        if digest(path) != spec['sha256']:
            raise ValueError(f'asset digest mismatch for {name}')
        files[name] = path
    return manifest, files


def input_tensors(torch, photo, pillow_version):
    import numpy as np
    import PIL
    from PIL import Image

    if PIL.__version__ != pillow_version:
        raise ValueError(f'expected Pillow {pillow_version}, got {PIL.__version__}')
    with Image.open(photo) as opened:
        width, height = opened.size
        shortest = min(width, height)
        pixels = opened.convert('RGB').resize((256 * width // shortest, 256 * height // shortest),
                                              Image.Resampling.BILINEAR)
    width, height = pixels.size
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32)
    std = torch.tensor([.229, .224, .225], dtype=torch.float32)
    inputs = []
    for step in range(REQUESTS):
        x, y = (17 * step) % (width - 223), (7 * step) % (height - 223)
        crop = pixels.crop((x, y, x + 224, y + 224))
        if step % 2:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        sample = torch.from_numpy(np.array(crop)).to(torch.float32).div_(255).sub_(mean).div_(std)
        assert sample.is_contiguous() and sample.storage_offset() == 0
        inputs.append(sample)
    return inputs


def load_model(torch, source, weights):
    tree = ast.parse(source.read_text(), filename=str(source))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
               node.name in ('Fire', 'SqueezeNet')]
    assert [node.name for node in classes] == ['Fire', 'SqueezeNet']
    module = types.ModuleType('_sym274_torchvision_squeezenet')
    module.__file__ = str(source)
    module.torch, module.nn, module.init = torch, torch.nn, torch.nn.init
    # The upstream hook emits API usage telemetry only.
    module._log_api_usage_once = lambda obj: None
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), module.__dict__)
    net = module.SqueezeNet(version='1_1').eval()
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    assert sum(p.numel() for p in net.parameters()) == 1_235_496
    return net


def memory(torch):
    free, total = torch.cuda.mem_get_info()
    return {
        'device_total': total, 'device_used_global': total - free,
        'torch_allocated': torch.cuda.memory_allocated(), 'torch_reserved': torch.cuda.memory_reserved(),
        'torch_peak_allocated': torch.cuda.max_memory_allocated(), 'torch_peak_reserved': torch.cuda.max_memory_reserved(),
        'process_peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def logits_check(torch, actual, expected):
    assert (actual.shape == expected.shape == (1, 1000)
            and actual.dtype == expected.dtype == torch.float32 and actual.device == expected.device)
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    top1 = int(actual.argmax())
    assert top1 == int(expected.argmax()), (top1, int(expected.argmax()))
    return {
        'max_abs_error': float((actual - expected).abs().max()), 'top1': top1,
        'logits_sha256': tensor_hash(actual), 'reference_sha256': tensor_hash(expected),
    }


def regular_stats(stats, requests):
    p = stats['prepared_execution']
    assert stats['dynamo_compiles'] == stats['plan_compiles'] == stats['inductor_compiles'] == 1, stats
    assert stats['fallbacks'] == {'prepared_first_use': 1}, stats['fallbacks']
    assert p['attempts'] == requests and p['executions'] == stats['runtime_executions'] == requests - 1, p
    assert p['preparations'] == 1 and p['cache_hits'] == requests - 2, p
    g = p['native']['prepared_groups']
    assert g['group_constructions'] == 1 and g['rebinds'] == requests - 2, g
    assert p['native']['logical_transfers'] == requests - 1 and p['native']['report_materializations'] == 0
    assert p['native']['payload_bytes'] == (requests - 1) * 602_112, p


def inspect_sym(torch, run, reference, owner, inputs):
    from bench.issue272.witness import observe_submissions

    snapshots = []
    stats_before = owner.stats()
    with observe_submissions() as submits:
        for host in inputs[:3]:
            actual = run(host)
            torch.cuda.synchronize()
            (group, completed), = submits
            out, = completed.wait()
            expected = host.permute(2, 0, 1).contiguous().to('cuda:0').unsqueeze(0)
            assert (out.shape, out.stride(), out.dtype, out.device) == (
                expected.shape, expected.stride(), expected.dtype, expected.device), (out.shape, out.stride())
            assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
            assert all(out.data_ptr() != previous.data_ptr() for previous, _ in snapshots)
            snapshots.append((out, tensor_hash(expected)))
            logits_check(torch, actual, reference(host))
            detailed = completed.report
            submits.clear()
    after_dense = owner.stats()
    assert after_dense['prepared_execution']['executions'] == stats_before['prepared_execution']['executions'] + 3
    assert detailed['logical_transfers'] == 1 and detailed['items'][0]['executed']
    strided = torch.empty(224, 224, 6, dtype=torch.float32)[..., ::2]
    strided.copy_(inputs[0])
    assert not strided.is_contiguous()
    with observe_submissions() as submits:
        rejected = logits_check(torch, run(strided), reference(strided))
        assert not submits, 'unsupported source submitted native work'
    after_rejection = owner.stats()
    assert after_rejection['prepared_execution']['executions'] == after_dense['prepared_execution']['executions']
    assert (after_rejection['fallbacks'] != after_dense['fallbacks']
            or after_rejection['exclusions'] != after_dense['exclusions']), after_rejection
    return {
        'checks': 3, 'transfer_digests': [checksum for _, checksum in snapshots],
        'native_report': detailed, 'strided_shape': list(strided.shape), 'strided_stride': list(strided.stride()),
        'strided_logits': rejected, 'after_dense': after_dense, 'after_rejection': after_rejection,
    }, snapshots


def worker(args):
    sys.path.insert(0, str(args.build / 'python'))
    os.environ['SYM_RELOC_EXPORT'] = str(args.build / 'sym/tools/sym-reloc-export')
    import torch
    import pyreloc
    from reloc_torch import RelocBackend

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(274)
    assert pyreloc.cuda_enabled
    assert sorted(os.sched_getaffinity(0)) == list(AFFINITY)
    timed = not args.untimed
    phases = {}
    if timed:
        phases['startup_configuration'] = time_ns() - int(os.environ['SYM274_LAUNCH_NS'])
    started = perf_counter_ns()
    asset_manifest, files = verified_assets(args.inputs)
    samples = input_tensors(torch, files['photo'], asset_manifest['pillow'])
    model = load_model(torch, files['source'], files['weights'])
    if timed:
        phases['inputs_and_model_cpu'] = perf_counter_ns() - started
    input_hashes = [tensor_hash(sample) for sample in samples]
    assert len(set(input_hashes)) == REQUESTS
    started = perf_counter_ns()
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_LIMIT / torch.cuda.get_device_properties(0).total_memory)
    model.cuda()
    torch.cuda.synchronize()
    if timed:
        phases['cuda_and_residency'] = perf_counter_ns() - started

    def reference(x):
        return model(x.permute(2, 0, 1).contiguous().to('cuda:0').unsqueeze(0))

    owner, trace = None, []
    started = perf_counter_ns()
    if args.arm == 'sym':
        owner = RelocBackend(compute_backend='inductor', inductor_options=COMPILE_OPTIONS,
                             prepared_execution=True, transfer_options=TRANSFER_OPTIONS)

        def capture(graph, examples):
            result = owner(graph, examples)
            trace.append((graph, result))
            return result

        run = torch.compile(reference, backend=capture, fullgraph=True, dynamic=False)
    elif args.arm == 'inductor':
        run = torch.compile(reference, backend='inductor', fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    else:
        run = reference
    if timed:
        phases['callable_setup'] = perf_counter_ns() - started

    requests = REQUESTS if timed else 3
    latencies, outputs = [], []
    with torch.no_grad():
        try:
            residency_start = memory(torch)
            for sample in samples[:requests]:
                torch.cuda.synchronize()
                if timed:
                    started = perf_counter_ns()
                logits = run(sample)
                torch.cuda.synchronize()
                if timed:
                    latencies.append(perf_counter_ns() - started)
                outputs.append(logits)
            residency_end = memory(torch)
            timed_stats = owner.stats() if owner is not None else None
            diagnostics = None
            retained_transfers = []
            if owner is not None:
                regular_stats(timed_stats, requests)
                [(graph, rewritten)] = trace
                [handle] = rewritten.handles
                compiled = owner._registry.lookup(handle).compiled
                capture_record = {
                    'dynamo_graph_sha256': hashlib.sha256(graph.code.encode()).hexdigest(),
                    'rewritten_graph_sha256': hashlib.sha256(rewritten.rewritten.code.encode()).hexdigest(),
                    'plan_sha256': hashlib.sha256(compiled.plan_bytes).hexdigest(),
                    'plan_bytes': len(compiled.plan_bytes),
                }

                # The untimed observer and replay must not enter the accepted measurements.
                diagnostics, retained_transfers = inspect_sym(torch, run, reference, owner, samples)
                diagnostics['capture'] = capture_record
            assert len({logits.data_ptr() for logits in outputs}) == requests
            checks = [logits_check(torch, logits, reference(x))
                      for x, logits in zip(samples[:requests], outputs, strict=True)]
            retained_initial = checks[0]['logits_sha256']
        finally:
            started = perf_counter_ns()
            if owner is not None:
                owner.close()
            torch.cuda.synchronize()
            if timed:
                phases['owner_retirement'] = perf_counter_ns() - started
        assert tensor_hash(outputs[0]) == retained_initial
        assert all(tensor_hash(out) == checksum for out, checksum in retained_transfers)
    if owner is not None:
        retired = owner.stats()['prepared_execution']
        assert retired['cache_entries'] == 0 and retired['native']['prepared_groups']['live_slots'] == 0
        diagnostics['held_transfers_after_close'] = len(retained_transfers)
        diagnostics['closed_prepared_state'] = retired

    rows = [dict(step=i, role='cold' if i == 0 else 'warmup' if i < COLD + WARMUP else 'warm',
                 completed_ns=latencies[i] if timed else None, input_sha256=input_hashes[i], **check)
            for i, check in enumerate(checks)]
    save(args.output, {
        'complete': True, 'arm': args.arm, 'round': args.round, 'timed': timed,
        'phases_ns': phases, 'requests': rows, 'input_trace_sha256': input_hashes,
        'held_initial_after_close': True, 'residency_start': residency_start, 'residency_end': residency_end,
        'timed_stats': timed_stats, 'diagnostics': diagnostics, 'environment': environment(),
    })


def verify_freeze(revision):
    for file in (PROTOCOL, ASSETS):
        published = subprocess.check_output(['git', 'show', f'{revision}:{file.relative_to(ROOT)}'], cwd=ROOT)
        if published != file.read_bytes():
            raise ValueError(f'{file.name} differs from frozen revision {revision}')


def run_campaign(args):
    is_preflight = args.action == 'preflight'
    if not is_preflight:
        verify_freeze(args.freeze_revision)
        for arm in ARMS:
            path = args.preflight / f'r0-{arm}.json'
            smoke = json.loads(path.read_text())
            assert smoke['complete'] and not smoke['timed'] and smoke['arm'] == arm, path
        smoke_manifest = json.loads((args.preflight / 'manifest.json').read_text())
        assert smoke_manifest['runner_sha256'] == digest(__file__), 'the runner changed after preflight'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build, inputs = args.build.resolve(), args.inputs.resolve()
    manifest, _ = verified_assets(inputs)
    provenance = {
        'protocol_sha256': digest(PROTOCOL), 'assets_sha256': digest(ASSETS), 'runner_sha256': digest(__file__),
        'freeze_revision': args.freeze_revision, 'asset_manifest': manifest,
        'runtime_sha256': runtime_hashes(build), 'arm_order': ORDER,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
    }
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
    provenance['gpu_before'] = {'exit': gpu.returncode, 'stdout': gpu.stdout, 'stderr': gpu.stderr}
    save(output / 'manifest.json', provenance)
    rounds = (ARMS,) if is_preflight else ORDER
    for round_no, arms in enumerate(rounds):
        for arm in arms:
            name = f'r{round_no}-{arm}'
            logfile = output / (name + '.log')
            command = ['taskset', '-c', '0-7', sys.executable, '-m', 'bench.issue274.application', 'worker',
                       '--inputs', str(inputs), '--build', str(build), '--output', str(output / (name + '.json')),
                       '--round', str(round_no), '--arm', arm]
            if is_preflight:
                command.append('--untimed')
            env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TORCHINDUCTOR_COMPILE_THREADS='1',
                       TORCHINDUCTOR_CACHE_DIR=str(output / 'caches' / name / 'inductor'),
                       TRITON_CACHE_DIR=str(output / 'caches' / name / 'triton'),
                       TORCHINDUCTOR_FORCE_DISABLE_CACHES='0', TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',
                       TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',
                       SYM274_LAUNCH_NS=str(time_ns()))
            with logfile.open('w') as log:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(name, 'completed' if process.returncode == 0 else 'failed', logfile, flush=True)
            if process.returncode:
                save(output / (name + '.failure.json'), {'exit': process.returncode, 'log': logfile.name})
                raise SystemExit(process.returncode)
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
    provenance['gpu_after'] = {'exit': gpu.returncode, 'stdout': gpu.stdout, 'stderr': gpu.stderr}
    save(output / 'manifest.json', provenance)
    if not is_preflight:
        save(output / 'summary.json', score(output))


def score(folder):
    processes = {arm: [] for arm in ARMS}
    records = [json.loads((folder / f'r{i}-{arm}.json').read_text())
               for i in range(len(ORDER)) for arm in ARMS]
    reference = records[0]
    for record in records:
        assert record['complete'] and record['timed'] and record['held_initial_after_close']
        assert record['environment'] == reference['environment']
        assert record['input_trace_sha256'] == reference['input_trace_sha256']
        rows = record['requests']
        assert len(rows) == REQUESTS and [row['step'] for row in rows] == list(range(REQUESTS))
        assert [row['input_sha256'] for row in rows] == record['input_trace_sha256']
        assert [row['reference_sha256'] for row in rows] == [row['reference_sha256'] for row in reference['requests']]
        warm = [row['completed_ns'] for row in rows[COLD + WARMUP:]]
        assert len(warm) == WARM
        latency = [row['completed_ns'] for row in rows]
        stats = record['timed_stats']
        if record['arm'] == 'sym':
            regular_stats(stats, REQUESTS)
            assert record['diagnostics']['held_transfers_after_close'] == 3
        process = {
            'round': record['round'], 'cold_completed_ns': latency[0],
            'warm_p50_ns': nearest_rank(warm, .5), 'warm_p95_ns': nearest_rank(warm, .95),
            'warmup_ns': sum(latency[1:COLD + WARMUP]), 'warm_total_ns': sum(warm),
            'lifecycle_ns': sum(record['phases_ns'].values()) + sum(latency),
            'max_abs_error': max(row['max_abs_error'] for row in rows),
        }
        processes[record['arm']].append(process)
    medians = {arm: {metric: statistics.median(row[metric] for row in rows)
                     for metric in rows[0] if metric != 'round'} for arm, rows in processes.items()}
    comparison = {control: {metric: paired_ratio([row[metric] for row in processes['sym']],
                                                 [row[metric] for row in processes[control]])
                            for metric in ('warm_p50_ns', 'warm_p95_ns', 'cold_completed_ns', 'lifecycle_ns')}
                  for control in ('eager', 'inductor')}
    return {
        'processes': processes, 'median_of_processes_ns': medians,
        'sym_over_baseline_ratios': comparison,
        'primary_hypothesis_supported': comparison['inductor']['warm_p95_ns']['bootstrap_95'][1] < 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('preflight', 'run', 'worker'):
        command = actions.add_parser(name)
        for arg in ('build', 'inputs', 'output'):
            command.add_argument('--' + arg, required=True, type=Path)
        if name == 'run':
            command.add_argument('--preflight', required=True, type=Path)
            command.add_argument('--freeze-revision', required=True)
        elif name == 'preflight':
            command.add_argument('--freeze-revision', default=None)
        else:
            command.add_argument('--arm', choices=ARMS, required=True)
            command.add_argument('--round', type=int, required=True)
            command.add_argument('--untimed', action='store_true')
    actions.add_parser('score').add_argument('folder', type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error('qualification requires Python assertions')
    if args.action == 'score':
        print(json.dumps(score(args.folder), indent=2))
    elif args.action == 'worker':
        worker(args)
    else:
        run_campaign(args)


if __name__ == '__main__':
    main()
