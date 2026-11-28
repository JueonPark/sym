"""Compare producer kernels using the same artifact from an actual Torch capture."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from bench.issue227.run import audit
from bench.issue265.analyze import paired
from bench.issue265.support import ROOT, digest, save
from bench.issue267.contract import SAMPLES, WARMUP
from bench.issue272.ablate import verify_builds

SOURCE_SHAPE = (8, 5, 29, 151)
PERMUTATION = (1, 2, 3, 0)
RESULT_SHAPE = tuple(SOURCE_SHAPE[i] for i in PERMUTATION)
WIRE_BYTES = math.prod(RESULT_SHAPE) * 4
WIRE_BUDGET = 1 << 20
ARMS = ('materialized', 'captured_artifact', 'manual', 'eager_torch', 'inductor', 'automatic_endpoint')
SEED = 272_151


def host_chain(source):
    return source.half().permute(PERMUTATION).contiguous().float()


def captured_chain(source):
    import torch

    return source.to(torch.float16).permute(PERMUTATION).contiguous().to('cuda:0', torch.float32)


def into_wire(source, wire):
    return wire.copy_(host_chain(source))


def clock(fn, *args):
    start = time.perf_counter_ns()
    result = fn(*args)
    return result, (time.perf_counter_ns() - start) / 1e6


def exact(actual, expected, torch):
    if (actual.shape, actual.dtype, actual.stride()) != (expected.shape, expected.dtype, expected.stride()):
        raise ValueError('output shape, dtype or strides differ from the oracle')
    bits = actual.cpu().view(torch.int32)
    if not torch.equal(bits, expected.view(torch.int32)):
        raise ValueError('output bits differ from the oracle')
    return hashlib.sha256(bits.numpy().tobytes()).hexdigest()


def materialized_programs(compiler, pyreloc):
    from bench.issue267.cases import spec
    from reloc_torch.recipe import Cast, Recipe, Transpose

    operations = (Cast('float16', 'ieee_rne'), Transpose(PERMUTATION), Cast('float32', 'exact'))
    boundaries = (spec(SOURCE_SHAPE, 'float32'), spec(SOURCE_SHAPE, 'float16'),
                  spec(RESULT_SHAPE, 'float16'), spec(RESULT_SHAPE, 'float32'))
    artifacts = [compiler.compile(Recipe(src, (op,), dst, 'h2d'))
                 for src, op, dst in zip(boundaries, operations, boundaries[1:])]
    plans = [pyreloc.bind_typed(artifact.decoded_plan, {}, {}) if artifact.typed
             else pyreloc.bind(artifact.decoded_plan, {}) for artifact in artifacts]
    return artifacts, plans


def worker(args):
    import torch
    import pyreloc
    from _folding_native import Runner
    from reloc_torch import CompilerClient, PreparedExecutionPolicy, RelocBackend, TransferResources
    from bench.issue265.support import environment
    from bench.issue272.witness import observe_submissions

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._inductor.config.emulate_precision_casts = True
    torch.backends.cuda.matmul.allow_tf32 = False
    setup = {'process_import_ms': (time.time_ns() - int(os.environ['EXPERIMENT_LAUNCH_NS'])) / 1e6}
    _, setup['cuda_init_ms'] = clock(torch.cuda.init)
    generator = torch.Generator().manual_seed(SEED)

    def inputs():
        value = torch.randn(SOURCE_SHAPE, generator=generator)
        value.reshape(-1)[:8] = torch.tensor([-0., 0., 2**-25, 2**-24,
            1.00048828125, 1.00146484375, -2.0009765625, 65520.])
        return value

    source = inputs()
    wire, setup['wire_allocation_ms'] = clock(
        lambda: torch.empty(RESULT_SHAPE, dtype=torch.float32, pin_memory=True))
    if not wire.is_pinned() or wire.nbytes != WIRE_BYTES or not 0 < WIRE_BYTES <= WIRE_BUDGET:
        raise ValueError('the common wire does not meet the frozen budget')
    compiler = CompilerClient.from_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    capture = {}
    with ExitStack() as resources, torch.no_grad():
        budget = resources.enter_context(TransferResources(max_contexts=1, max_contexts_per_device=1,
            max_background_workers=0, max_typed_background_workers=0, max_typed_streams=1,
            max_typed_retained_bytes=WIRE_BUDGET, max_typed_live_bytes=WIRE_BUDGET))
        backend, setup['backend_setup_ms'] = clock(lambda: RelocBackend(compiler=compiler,
            prepared_execution=PreparedExecutionPolicy(min_reuses=0, min_bytes=0, row_policy='cpu'),
            transfer_resources=budget,
            transfer_options=dict(gather_threads=1, n_buffers=1, n_streams=1, pinning='pinned')))
        resources.callback(backend.close)

        def on_capture(graph, examples):
            lowered, duration = clock(backend, graph, examples)
            if capture or len(lowered.handles) != 1:
                raise ValueError(f'expected one captured and lowered region: {backend.stats()["exclusions"]}')
            entry = backend._registry.lookup(lowered.handles[0])
            capture.update(artifact=entry.compiled, original=graph.code, lowered=lowered.rewritten.code,
                           callback_ms=duration)
            return lowered

        auto, setup['automatic_wrapper_ms'] = clock(lambda: torch.compile(
            captured_chain, backend=on_capture, fullgraph=True, dynamic=False))

        def automatic(value):
            out = auto(value)
            torch.cuda.synchronize()
            return out

        first_auto, auto_first_ms = clock(automatic, source)
        compiled = capture.pop('artifact')
        if not compiled.typed or compiled.indexed:
            raise ValueError('capture did not emit a non-indexed typed plan')
        stages = [(stage['input_dtype'], stage['output_dtype']) for stage in compiled.manifest['stages']]
        if stages != [('float32', 'float16'), ('float16', 'float32')]:
            raise ValueError(f'unexpected captured numerical stages: {stages}')
        plan_path = args.output / 'captured.plan'
        plan_path.write_bytes(compiled.plan_bytes)
        capture.update(plan_sha256=digest(plan_path), plan_bytes=plan_path.stat().st_size,
            manifest=compiled.manifest, bindings=compiled.bind_values(source))
        bound, setup['captured_decode_bind_ms'] = clock(lambda: pyreloc.bind_typed(
            pyreloc.load_typed_plan(plan_path.read_bytes()), capture['bindings'], {}))
        folded, setup['captured_native_prepare_ms'] = clock(Runner, [bound], 1)
        host_kernel, = folded.kernels()
        stage_run, = folded.stage_metadata()
        if host_kernel not in ('tiled_transpose', 'tiled_transpose_runs', 'strided_runs', 'contiguous_stages'):
            raise ValueError(f'executed host kernel family does not use ordered stage runs: {host_kernel}')
        expected_kind = 'register_cast_chain' if args.runtime == 'specialized' else 'legacy'
        if stage_run['eligible_kind'] != expected_kind:
            raise ValueError(f'unexpected stage metadata: {stage_run}')

        (parts, bounds), setup['materialized_compile_bind_ms'] = clock(materialized_programs, compiler, pyreloc)
        for i, part in enumerate(parts):
            (args.output / f'materialized-{i}.plan').write_bytes(part.plan_bytes)
        materialized, setup['materialized_native_prepare_ms'] = clock(Runner, bounds, 1)
        intermediates, setup['materialized_allocations_ms'] = clock(lambda: (
            torch.empty(SOURCE_SHAPE, dtype=torch.float16), torch.empty(RESULT_SHAPE, dtype=torch.float16)))
        manual, setup['manual_prepare_ms'] = clock(Runner, [], 1)
        dummy_scales = torch.empty(0, dtype=torch.float32).numpy()
        cast_part, permute_part = [tensor.numpy() for tensor in intermediates]
        wire_array = wire.numpy()
        inductor, setup['inductor_wrapper_ms'] = clock(lambda: torch.compile(
            into_wire, backend='inductor', fullgraph=True, dynamic=False,
            options={'triton.cudagraphs': False}))

        producers = {
            'materialized': lambda value: materialized.run([value.numpy(), cast_part, permute_part, wire_array]),
            'captured_artifact': lambda value: folded.run([value.numpy(), wire_array]),
            'manual': lambda value: manual.manual('rank3_chain', value.numpy(), wire_array, dummy_scales,
                SOURCE_SHAPE[1] * SOURCE_SHAPE[2], SOURCE_SHAPE[3]),
            'eager_torch': lambda value: into_wire(value, wire),
            'inductor': lambda value: inductor(value, wire),
        }

        def complete(arm, value):
            if arm == 'automatic_endpoint':
                return automatic(value)
            producers[arm](value)
            out = torch.empty(RESULT_SHAPE, dtype=wire.dtype, device='cuda:0')
            out.copy_(wire, non_blocking=True)
            torch.cuda.synchronize()
            return out

        expected = host_chain(source)
        bit_hash = exact(first_auto, expected, torch)
        held = {'automatic_endpoint': (first_auto, expected)}
        results = {arm: dict(first_ms=None, warm_ms=[], repeated_ms=[], outputs=[]) for arm in ARMS}
        results['automatic_endpoint'].update(first_ms=auto_first_ms, first_sha256=bit_hash)
        offset = (args.round - 1) % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]

        def checked_call(arm, values, oracle):
            out, elapsed = clock(complete, arm, values)
            bit_hash = exact(out, oracle, torch)
            if arm != 'automatic_endpoint':
                exact(wire, oracle, torch)
            if arm in held:
                old, old_expected = held[arm]
                if out.data_ptr() == old.data_ptr():
                    raise ValueError(f'{arm} reused a live output')
                exact(old, old_expected, torch)
            else:
                held[arm] = out, oracle
            return elapsed, bit_hash

        for arm in order:
            row = results[arm]
            if arm != 'automatic_endpoint':
                row['first_ms'], row['first_sha256'] = checked_call(arm, source, expected)
            for _ in range(WARMUP):
                elapsed, _ = checked_call(arm, source, expected)
                row['warm_ms'].append(elapsed)
        for sample in range(SAMPLES):
            values = inputs()
            oracle = host_chain(values)
            start = (sample + offset) % len(ARMS)
            for arm in ARMS[start:] + ARMS[:start]:
                elapsed, output = checked_call(arm, values, oracle)
                results[arm]['repeated_ms'].append(elapsed)
                results[arm]['outputs'].append(output)

        timed_stats = backend.stats()
        with observe_submissions() as submissions:
            observed = automatic(source)
        exact(observed, expected, torch)
        (group, completion), = submissions
        native_report = completion.report
        item, = native_report['items']
        (program,) = group._programs
        if (hashlib.sha256(program.compiled.plan_bytes).hexdigest() != capture['plan_sha256']
                or item['implementation'] != 'cpu_reference' or item['wire_boundary'] != len(stages)
                or item['wire_bytes'] != WIRE_BYTES or item['payload_bytes_transferred'] != WIRE_BYTES
                or not item['executed']):
            raise ValueError(f'automatic endpoint dispatched a different plan or wire: {item}')
        counters = timed_stats['prepared_execution']
        if (counters['executions'] != 1 + WARMUP + SAMPLES or timed_stats['fallbacks']
                or timed_stats['plan_compiles'] != 1 or timed_stats['dynamo_compiles'] != 1):
            raise ValueError('automatic endpoint did not execute one captured plan for every request')
        _, setup['retirement_ms'] = clock(resources.close)

    for output, expected in held.values():
        exact(output, expected, torch)
    common = setup['process_import_ms'] + setup['cuda_init_ms']
    acquisition = setup['backend_setup_ms'] + setup['automatic_wrapper_ms'] + auto_first_ms
    arm_setup = {
        'materialized': setup['materialized_compile_bind_ms'] + setup['materialized_native_prepare_ms']
            + setup['materialized_allocations_ms'],
        'captured_artifact': acquisition + setup['captured_decode_bind_ms'] + setup['captured_native_prepare_ms'],
        'manual': setup['manual_prepare_ms'], 'eager_torch': 0,
        'inductor': setup['inductor_wrapper_ms'],
        'automatic_endpoint': setup['backend_setup_ms'] + setup['automatic_wrapper_ms'],
    }
    for arm, row in results.items():
        init = common + arm_setup[arm] + (0 if arm == 'automatic_endpoint' else setup['wire_allocation_ms'])
        row.update(cold_ms=init + row['first_ms'], steady_p50_ms=statistics.median(row['repeated_ms']),
            measured_call_sum_ms=row['first_ms'] + sum(row['warm_ms']) + sum(row['repeated_ms']))
        row['accounted_total_ms'] = init + row['measured_call_sum_ms']
        if arm in ('automatic_endpoint', 'captured_artifact'):
            row['accounted_total_ms'] += setup['retirement_ms']
    result = dict(runtime=args.runtime, round=args.round, environment=environment(),
        wire_bytes=WIRE_BYTES, output_bytes=WIRE_BYTES, source_shape=SOURCE_SHAPE, permute=PERMUTATION,
        destination_shape=RESULT_SHAPE, common_h2d_copies=1, stages=stages, capture=capture,
        materialized_plan_bytes=[len(part.plan_bytes) for part in parts],
        materialized_intermediate_bytes=sum(t.nbytes for t in intermediates),
        host_kernel=host_kernel, stage_run=stage_run, automatic_report=item,
        automatic_stats=timed_stats, setup_ms=setup, arms=results,
        held_checks_after_close=len(held), samples=SAMPLES, warmups=WARMUP,
        compile_cache_files=sum(path.is_file() for path in Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*')),
        compile_cache_bytes=sum(path.stat().st_size for path in
            Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*') if path.is_file()),
        helper_sha256=digest(sys.modules['_folding_native'].__file__))
    save(args.output / 'result.json', result)


def load_rows(directory):
    contract = json.loads((directory / 'contract.json').read_text())
    rows = []
    for round_number in range(1, contract['rounds'] + 1):
        for runtime in ('specialized', 'generic'):
            name = f'r{round_number}-{runtime}'
            row = json.loads((directory / name / 'result.json').read_text())
            receipt = json.loads((directory / f'{name}.audit.json').read_text())
            if receipt['returncode'] != 0 or receipt['observations'] or (
                    row['round'], row['runtime']) != (round_number, runtime):
                raise ValueError(f'invalid process receipt: {name}')
            if digest(directory / name / 'captured.plan') != row['capture']['plan_sha256']:
                raise ValueError(f'captured plan hash differs: {name}')
            if row['helper_sha256'] != contract['helpers'][runtime]:
                raise ValueError(f'native helper differs: {name}')
            rows.append(row)
    return rows


def summarize(rows):
    first = rows[0]
    expected_output = first['arms']['captured_artifact']['outputs']
    for row in rows:
        if (row['capture']['plan_sha256'] != first['capture']['plan_sha256']
                or row['environment'] != first['environment'] or row['wire_bytes'] != WIRE_BYTES
                or row['common_h2d_copies'] != 1 or row['held_checks_after_close'] != len(ARMS)):
            raise ValueError('artifacts, environment, wire or retirement checks differ')
        for arm in ARMS:
            record = row['arms'][arm]
            if (len(record['warm_ms']) != WARMUP or len(record['repeated_ms']) != SAMPLES
                    or len(expected_output) != SAMPLES or record['outputs'] != expected_output
                    or record['first_sha256'] != first['arms']['captured_artifact']['first_sha256']):
                raise ValueError('request sequence or output hashes differ')
    specialized = [r for r in rows if r['runtime'] == 'specialized']
    generic = [r for r in rows if r['runtime'] == 'generic']
    comparisons = []
    controls = [(arm, specialized) for arm in ARMS if arm not in ('captured_artifact', 'automatic_endpoint')]
    controls.insert(1, ('prior_captured_artifact', generic))
    for arm, rounds in controls:
        source = 'captured_artifact' if arm == 'prior_captured_artifact' else arm
        comparisons.append(dict(control=arm, candidate='captured_artifact', metrics={
            metric: paired([row['arms'][source][metric] for row in rounds],
                           [row['arms']['captured_artifact'][metric] for row in specialized])
            for metric in ('cold_ms', 'steady_p50_ms', 'accounted_total_ms')
        }))
    return dict(complete=True, processes=len(rows), matched_arms=ARMS[:-1],
        separate_endpoint='automatic_endpoint uses the backend-owned pinned staging and scheduling',
        artifact_acquisition='captured_artifact cold and total charge one completed automatic capture call',
        cold_scope='shared-process first calls after common CUDA init; fresh Inductor caches',
        comparisons=comparisons, rows=rows)


def run(args):
    directory = args.output.resolve()
    builds = {name: getattr(args, name + '_build').resolve() for name in ('specialized', 'generic')}
    helpers = {name: getattr(args, name + '_helper').resolve() for name in builds}
    helper_hashes = {}
    for name, helper in helpers.items():
        files = list(helper.glob('_folding_native*.so'))
        if len(files) != 1:
            raise ValueError(f'expected one native runner in {helper}')
        helper_hashes[name] = digest(files[0])
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / 'contract.json', dict(rounds=args.rounds, wire_bytes=WIRE_BYTES,
        configurations=dict(affinity=args.affinity, cuda_device=args.device, omp_threads=1),
        builds=verify_builds(builds), helpers=helper_hashes, runner_sha256=digest(__file__)))
    keep = ('HOME', 'USER', 'LOGNAME', 'PATH', 'SHELL', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR')
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1')
    for round_number in range(1, args.rounds + 1):
        order = ('generic', 'specialized') if round_number % 2 else ('specialized', 'generic')
        for runtime in order:
            foreign = audit(set())
            if foreign:
                raise RuntimeError(f'foreign work before completed-H2D campaign: {foreign}')
            name = f'r{round_number}-{runtime}'
            build = builds[runtime]
            variables = env | dict(PYTHONPATH=os.pathsep.join((str(build / 'python'), str(helpers[runtime]))),
                LD_LIBRARY_PATH=str(build / 'libreloc'), SYM_RELOC_EXPORT=str(build / 'sym/tools/sym-reloc-export'),
                SYM_OPT=str(build / 'sym/tools/sym-opt'), TORCHINDUCTOR_CACHE_DIR=str(directory / (name + '-cache')))
            command = ['taskset', '-c', args.affinity, os.path.abspath(args.python), '-m',
                'bench.issue272.captured_quality', 'worker', '--output', str(directory / name),
                '--round', str(round_number), '--runtime', runtime]
            observations = []
            variables['EXPERIMENT_LAUNCH_NS'] = str(time.time_ns())
            with (directory / f'{name}.log').open('w') as log:
                with subprocess.Popen(command, cwd=ROOT, env=variables, stdout=log,
                                      stderr=subprocess.STDOUT) as process:
                    while process.poll() is None:
                        foreign = audit({process.pid})
                        if foreign:
                            observations.append(foreign)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
            save(directory / f'{name}.audit.json', dict(command=command,
                returncode=process.returncode, observations=observations))
            if process.returncode or observations:
                raise RuntimeError(f'{name} failed or had external interference. See {directory / (name + ".log")}')
            print(name, 'complete', flush=True)
    save(directory / 'analysis.json', summarize(load_rows(directory)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    campaign = sub.add_parser('run')
    for key in ('specialized-build', 'generic-build', 'specialized-helper', 'generic-helper', 'output'):
        campaign.add_argument('--' + key, type=Path, required=True)
    campaign.add_argument('--python', default=sys.executable)
    campaign.add_argument('--rounds', type=int, choices=range(2, 11), default=3)
    campaign.add_argument('--affinity', default='0-7')
    campaign.add_argument('--device', default='0')
    child = sub.add_parser('worker')
    child.add_argument('--runtime', required=True, choices=('specialized', 'generic'))
    child.add_argument('--round', type=int, required=True)
    child.add_argument('--output', type=Path, required=True)
    scoring = sub.add_parser('score')
    scoring.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    elif args.command == 'worker':
        worker(args)
    else:
        save(args.directory / 'analysis.json', summarize(load_rows(args.directory)))


if __name__ == '__main__':
    main()
