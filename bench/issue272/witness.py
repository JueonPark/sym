"""Trace normal Torch capture through actually completed native prepared groups."""

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
from math import prod
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch
from weakref import WeakKeyDictionary

from bench.issue265.support import digest, environment, save

ROOT = Path(__file__).resolve().parents[2]
SEED = 272013
ROW_POLICIES = ('device', 'cpu')
STAGE_KINDS = ('buffered', 'register_cast_chain', 'register_dequant_cast_chain')
STAGE_BOUNDARIES = 9
STAGE_SLOTS = len(STAGE_KINDS) * STAGE_BOUNDARIES ** 2


@dataclass(frozen=True)
class Request:
    n: int
    phase: str
    revision: str = 'none'


TRACES = {
    'ordered_cast': (
        Request(29, 'first_use'), Request(29, 'prepare'),
        Request(37, 'first_use'), Request(37, 'prepare'),
        Request(29, 'reuse'), Request(3, 'small'),
        Request(37, 'reuse'), Request(29, 'reuse'),
    ),
    'parameterized_dequant': (
        Request(503, 'first_use', 'zero'), Request(503, 'prepare', 'zero'),
        Request(503, 'first_use', 'nonzero'), Request(503, 'prepare', 'nonzero'),
        Request(503, 'first_use', 'scale_only'), Request(503, 'prepare', 'scale_only'),
        Request(571, 'first_use', 'zero'), Request(571, 'prepare', 'zero'),
        Request(503, 'reuse', 'zero'), Request(503, 'reuse', 'nonzero'),
        Request(37, 'small', 'zero'), Request(503, 'reuse', 'scale_only'),
        Request(571, 'reuse', 'zero'),
    ),
}


def ordered_cast(x, torch):
    return x.to(torch.float16).permute(2, 0, 1, 3).contiguous().to('cuda:0', torch.float32)


def parameterized_dequant(x, scale, zero_point, torch):
    f = torch.ops.quantized_decomposed.dequantize_per_channel(
        x, scale, zero_point, 1, -128, 127, torch.int8,
    )
    return f.transpose(0, 1).contiguous().to(torch.float16).to('cuda:0', torch.float32)


def bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def tensor_metadata(value):
    return {
        'shape': list(value.shape), 'strides': list(value.stride()),
        'dtype': str(value.dtype), 'device': str(value.device),
        'storage_offset': value.storage_offset(),
    }


def tensor_bytes(value):
    import torch

    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def checked_output(actual, expected):
    metadata = tensor_metadata(actual)
    assert metadata == tensor_metadata(expected), ('output metadata', metadata, tensor_metadata(expected))
    actual_bits, expected_bits = tensor_bytes(actual), tensor_bytes(expected)
    assert actual_bits == expected_bits, 'output bits differ from the eager oracle'
    return dict(metadata, actual_sha256=bytes_sha256(actual_bits),
                expected_sha256=bytes_sha256(expected_bits), exact=True)


def make_workload(case, torch):
    generator = torch.Generator().manual_seed(SEED)
    if case == 'ordered_cast':
        def function(source):
            return ordered_cast(source, torch)

        def inputs(request):
            source = torch.randn(request.n, 7, 13, 11, generator=generator)
            # Exercise round ties, signed zero, and the initial half rounding.
            source.reshape(-1)[:4] = torch.tensor([-0., 1.00048828125, 1.00146484375, -2.0009765625])
            return (source,)

        return function, inputs

    import torch.ao.quantization.fx._decomposed  # noqa: F401

    # These two allocations stay live across every revision and both shapes.
    base_scale = torch.linspace(.0078125, .25, 137)
    scale = base_scale.clone()
    zero_point = torch.zeros(137, dtype=torch.int32)

    def function(source, scales, zero_points):
        return parameterized_dequant(source, scales, zero_points, torch)

    def inputs(request):
        scale.copy_(base_scale)
        if request.revision == 'scale_only':
            scale.mul_(1.75)
        zero_point.fill_(3 if request.revision == 'nonzero' else 0)
        source = torch.randint(-128, 128, (request.n, 137), dtype=torch.int8, generator=generator)
        return source, scale, zero_point

    return function, inputs


@contextmanager
def observe_submissions():
    """Retain the real completion until its report can be read after timing."""
    from reloc_torch.prepared_group import PreparedGroup

    original = PreparedGroup.submit
    submissions = []

    def observe(group, *args, **kwargs):
        completion = original(group, *args, **kwargs)
        submissions.append((group, completion))
        return completion

    with patch.object(PreparedGroup, 'submit', observe):
        yield submissions


def capture_artifact(owner, graph, lowered):
    handle, = lowered.handles
    entry = owner._registry.lookup(handle)
    compiled = entry.compiled
    payload = compiled.to_bytes()
    captured, rewritten = graph.code, lowered.rewritten.code
    return {
        'capture_graph': captured, 'capture_sha256': bytes_sha256(captured.encode()),
        'rewritten_graph': rewritten, 'rewritten_sha256': bytes_sha256(rewritten.encode()),
        'artifact': json.loads(payload), 'artifact_bytes': len(payload),
        'artifact_sha256': bytes_sha256(payload),
        'plan_bytes': len(compiled.plan_bytes), 'plan_sha256': bytes_sha256(compiled.plan_bytes),
        'wire_version': compiled.wire_version, 'compiler': asdict(compiled.compiler),
        'fallback_calls': entry.fallback_calls,
    }


def counters(stats):
    prepared = stats['prepared_execution'] or {}
    native = prepared.get('native', {})
    groups = native.get('prepared_groups', {})
    return {
        **{key: stats[key] for key in ('dynamo_compiles', 'plan_compiles', 'runtime_executions')},
        **{key: prepared.get(key, 0) for key in (
            'attempts', 'executions', 'preparations', 'cache_hits', 'cache_misses',
            'preparation_ns', 'parameter_snapshot_ns', 'parameter_snapshots',
        )},
        **{key: native.get(key, 0) for key in ('groups', 'payload_bytes', 'report_materializations')},
        **{key: groups.get(key, 0) for key in ('group_constructions', 'rebinds', 'handles_created')},
        'fallbacks': stats['fallbacks'], 'dispatches': native.get('typed_dispatches', {}),
    }


def counter_changes(previous, current):
    changes = {}
    for name, value in current.items():
        old = previous[name]
        if isinstance(value, dict):
            changes[name] = {key: value.get(key, 0) - old.get(key, 0)
                             for key in sorted(value.keys() | old.keys())
                             if value.get(key, 0) != old.get(key, 0)}
        else:
            changes[name] = value - old
    return changes


class StageProbe:
    """Read the benchmark-only ELF interposer after work has completed."""

    def __init__(self, probe_path, runtime):
        probe_path, runtime = probe_path.resolve(strict=True), runtime.resolve(strict=True)
        preloads = os.environ.get('LD_PRELOAD', '').replace(':', ' ').split()
        if probe_path not in {Path(path).resolve() for path in preloads}:
            raise ValueError('start Python with --stage-probe also named in LD_PRELOAD')
        os.environ['SYM_ISSUE272_STAGE_RUNTIME'] = str(runtime)
        process = ctypes.CDLL(None)
        abi = process.sym_issue272_stage_probe_abi
        abi.argtypes, abi.restype = [], ctypes.c_uint
        if abi() != 1:
            raise ValueError('incompatible issue272 stage probe ABI')
        self._counts = process.sym_issue272_stage_counts
        self._counts.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self._counts.restype = ctypes.c_int
        eligible = process.sym_issue272_stage_cpu_eligible
        eligible.argtypes, eligible.restype = [], ctypes.c_int
        self.metadata = {
            'abi': 1, 'file': probe_path, 'sha256': digest(probe_path),
            'source_sha256': digest(Path(__file__).with_name('stage_probe.cpp')),
            'runtime_file': runtime, 'runtime_sha256': digest(runtime),
            'cpu_supports_avx2_f16c_fma': bool(eligible()),
            'scope': 'GNU Linux x86-64 ELF interposition of runStageRun in this diagnostic process',
            'interpretation': 'successful StageRun calls and processed elements by kind and stage bounds. '
                              'The ISA query reports eligibility and does not observe individual instructions.',
        }

    def snapshot(self):
        values = (ctypes.c_uint64 * (2 * STAGE_SLOTS + 1))()
        assert self._counts(values, len(values)) == 0 and values[-1] == 0, 'stage probe ABI or stage range mismatch'
        return [(values[2 * i], values[2 * i + 1]) for i in range(STAGE_SLOTS)]


def stage_changes(previous, current):
    rows = []
    for slot, ((old_calls, old_elements), (calls, elements)) in enumerate(zip(previous, current, strict=True)):
        if (calls, elements) == (old_calls, old_elements):
            continue
        kind, boundary_pair = divmod(slot, STAGE_BOUNDARIES ** 2)
        start, stop = divmod(boundary_pair, STAGE_BOUNDARIES)
        rows.append(dict(kind=STAGE_KINDS[kind], start_stage=start, stop_stage=stop,
                         calls=calls - old_calls, elements=elements - old_elements))
    return rows


def validate_stage_runs(case, row_policy, row):
    runs, native = row['observed_stage_runs'], row['native']
    if native is None:
        assert runs == [], ('fallback ran native stage code', row)
        return
    item, = native['report']['items']
    if row_policy == 'cpu':
        kind = 'register_cast_chain' if case == 'ordered_cast' else 'register_dequant_cast_chain'
        assert len(runs) == 1, ('missing native register stage observations', row)
        observed, = runs
        assert (observed['kind'], observed['start_stage'], observed['stop_stage']) == (
            kind, 0, item['wire_boundary'],
        ), ('unexpected native register stage range', row)
        assert observed['calls'] > 0 and observed['elements'] == prod(row['source']['shape']), (
            'native register stages did not cover the full source', row,
        )
    elif item['wire_boundary'] == 0:
        assert runs == [], ('all-device row ran host stage code', row)


def validate_execution(phase, delta, native):
    prepared, reused = phase == 'prepare', phase == 'reuse'
    ran = prepared or reused
    expected = {
        'runtime_executions': ran, 'executions': ran, 'preparations': prepared,
        'cache_hits': reused, 'groups': ran, 'group_constructions': prepared, 'rebinds': reused,
    }
    assert all(delta[key] == count for key, count in expected.items()), (phase, expected, delta)
    assert delta['fallbacks'] == ({} if ran else {'prepared_' + phase: 1}), ('fallback reason', phase, delta)
    if not ran:
        assert native is None and delta['dispatches'] == {}, ('fallback submitted native work', delta)
        return
    assert native is not None, 'completed dispatch observation is missing'
    report, selection = native['report'], native['selection']
    item, = report['items']
    comparable = {'implementation', 'method', 'wire_boundary', 'wire_bytes', 'device_temp_bytes'} & selection.keys()
    assert (report['logical_transfers'] == 1 and item['executed']
            and all(item[key] == selection[key] for key in comparable)
            and delta['dispatches'] == {item['implementation']: 1}), ('completed dispatch', selection, report, delta)


def observe_native(submissions, group_ids, specializations):
    if not submissions:
        return None
    (group, completion), = submissions
    program, = group._programs
    group_id = group_ids.get(group)
    counts = dict(group._binder.stats())
    if group_id is None:
        group_id = group_ids[group] = len(specializations)
        specializations.append({
            'group_id': group_id, 'plan_sha256': bytes_sha256(program.compiled.plan_bytes),
            'symbol_bindings': dict(program.bindings), 'parameter_bytes': program.parameter_bytes,
            'selection': dict(program.template.selection), 'capability': dict(program.template.capability),
            'cuts': list(program.bound.cuts),
        })
    # The report is a snapshot from the completed group in the normal execution path.
    report = completion.report
    assert report is not None, 'automatic execution returned before native group completion'
    return {
        'group_id': group_id, 'binder_counters': counts,
        'selection': specializations[group_id]['selection'], 'report': report,
    }


def validate_case(case, row_policy, rows, specializations, captures):
    capture, = captures
    assert capture['wire_version'] == 1
    assert 'torch.ops.reloc.' in capture['rewritten_graph'] or 'torch.ops.reloc_torch.' in capture['rewritten_graph'], (
        'the normal frontend did not replace the region', capture['rewritten_graph'],
    )
    assert capture['fallback_calls'] == sum(row['native'] is None for row in rows)
    manifest = capture['artifact']['manifest']
    expected = [('cast', 'float32', 'float16'), ('cast', 'float16', 'float32')]
    if case == 'parameterized_dequant':
        expected.insert(0, ('dequantize', 'int8', 'float32'))
    assert [(stage['transform'], stage['input_dtype'], stage['output_dtype'])
            for stage in manifest['stages']] == expected, manifest['stages']
    assert all(s['plan_sha256'] == capture['plan_sha256'] for s in specializations)
    if case == 'parameterized_dequant':
        assert {p['dtype'] for p in manifest['parameters']} == {'float32', 'int32'}
    groups = {}
    for row in rows:
        native = row['native']
        if native is None:
            continue
        group_id = native['group_id']
        key = row['n'], row['revision']
        if row['phase'] == 'prepare':
            assert group_id not in groups.values()
            groups[key] = group_id
        else:
            assert groups[key] == group_id, ('reuse bound a different native group', row)
        selected = native['selection']['implementation']
        if row_policy == 'cpu':
            assert selected == 'cpu_reference', (row_policy, row)
        elif case == 'ordered_cast':
            assert selected == 'cuda_relocate_f32', row
        elif row['revision'] != 'nonzero':
            assert selected == 'cuda_dequant_relocate', row
        else:
            excluded = specializations[group_id]['capability']['excluded']
            assert selected != 'cuda_dequant_relocate' and any(
                entry['implementation'] == 'cuda_dequant_relocate' and 'nonzero_zero_point' in entry['reason']
                for entry in excluded
            ), (row, excluded)


def run_case(case, row_policy, stage_probe=None):
    import torch
    from reloc_torch import PreparedExecutionPolicy, RelocBackend

    function, inputs = make_workload(case, torch)
    policy = PreparedExecutionPolicy(row_policy=row_policy)
    started = perf_counter_ns()
    owner = RelocBackend(prepared_execution=policy, transfer_options={'pinning': 'pinned', 'n_streams': 1})
    callbacks, retained, rows, specializations = [], [], [], []
    group_ids = WeakKeyDictionary()
    parameter_addresses, revisions = None, {}

    def capture(graph, examples):
        lowered = owner(graph, examples)
        callbacks.append((graph, lowered))
        return lowered

    try:
        run = torch.compile(function, backend=capture, dynamic=True, fullgraph=True)
        owner_setup_ns = perf_counter_ns() - started
        previous = counters(owner.stats())
        previous_stages = stage_probe.snapshot() if stage_probe is not None else None
        with torch.no_grad(), observe_submissions() as submissions:
            for step, request in enumerate(TRACES[case]):
                values = inputs(request)
                source, *parameters = values
                source_hash = bytes_sha256(tensor_bytes(source))
                parameter_hashes = [bytes_sha256(tensor_bytes(value)) for value in parameters]
                addresses = tuple(value.data_ptr() for value in parameters)
                if parameter_addresses is None:
                    parameter_addresses = addresses
                assert addresses == parameter_addresses, 'parameter storage changed across live revisions'
                assert revisions.setdefault(request.revision, parameter_hashes) == parameter_hashes
                torch.cuda.synchronize()
                started = perf_counter_ns()
                actual = run(*values)
                torch.cuda.synchronize()
                completed_ns = perf_counter_ns() - started
                native = observe_native(submissions, group_ids, specializations)
                submissions.clear()
                stats = owner.stats()
                current = counters(stats)
                delta = counter_changes(previous, current)
                validate_execution(request.phase, delta, native)
                previous = current
                expected = function(*values)
                checked = checked_output(actual, expected)
                assert all(old.data_ptr() != actual.data_ptr() for old, _ in retained), 'held output storage was reused'
                for old, snapshot in retained:
                    checked_output(old, snapshot)
                retained.append((actual, expected))
                row = dict(asdict(request), step=step,
                           completed_ns=completed_ns if stage_probe is None else None,
                           source=dict(tensor_metadata(source), sha256=source_hash),
                           parameter_sha256=parameter_hashes, output=checked,
                           held_checks_before_append=step, counter_delta=delta, native=native)
                if stage_probe is not None:
                    current_stages = stage_probe.snapshot()
                    row['observed_stage_runs'] = stage_changes(previous_stages, current_stages)
                    validate_stage_runs(case, row_policy, row)
                    previous_stages = current_stages
                rows.append(row)
        captures = [capture_artifact(owner, graph, lowered) for graph, lowered in callbacks]
        validate_case(case, row_policy, rows, specializations, captures)
        assert len({row['source']['sha256'] for row in rows}) == len(rows), 'source values were repeated'
        assert len({tuple(items) for items in revisions.values()}) == len(revisions), 'revisions repeated values'
        if case == 'parameterized_dequant':
            assert revisions['zero'][0] == revisions['nonzero'][0] != revisions['scale_only'][0]
            assert revisions['zero'][1] == revisions['scale_only'][1] != revisions['nonzero'][1]
        final_stats = owner.stats()
    finally:
        started = perf_counter_ns()
        owner.close()
        owner_close_ns = perf_counter_ns() - started
    retired = owner.stats()
    assert retired['prepared_execution']['cache_entries'] == 0
    assert retired['prepared_execution']['native']['prepared_groups']['live_slots'] == 0
    for actual, expected in retained:
        checked_output(actual, expected)
    return {
        'policy': asdict(policy), 'owner_setup_ns': owner_setup_ns if stage_probe is None else None,
        'owner_close_ns': owner_close_ns if stage_probe is None else None,
        'captures': captures, 'specializations': specializations, 'requests': rows,
        'parameter_values_same_storage': bool(parameter_addresses),
        'parameter_revision_sha256': revisions, 'final_stats': final_stats,
        'retired_stats': retired, 'held_checks_after_close': len(retained),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-probe', type=Path, help='preloaded benchmark-only stage probe (untimed)')
    args = parser.parse_args()
    if not __debug__:
        parser.error('witness validation requires Python assertions')
    if args.output.exists():
        raise FileExistsError(args.output)
    build = args.build.resolve()
    sys.path.insert(0, str(build / 'python'))
    import torch
    import pyreloc
    import reloc_torch
    from reloc_torch.compiler import CompilerClient

    assert torch.cuda.is_available() and pyreloc.cuda_enabled, 'a CUDA-enabled Torch and Sym runtime are required'
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    for default_exporter in (build / 'sym/tools/sym-reloc-export', build / 'bin/sym-reloc-export'):
        if default_exporter.is_file() and not os.environ.get('SYM_OPT'):
            os.environ.setdefault('SYM_RELOC_EXPORT', str(default_exporter))
            break
    compiler = CompilerClient.from_environment()
    stage_probe = StageProbe(args.stage_probe, build / 'libreloc/libreloc_runtime.so') if args.stage_probe else None
    sources = ('libreloc/python/reloc_torch/backend.py', 'libreloc/python/reloc_torch/automatic.py',
               'libreloc/python/reloc_torch/prepared_group.py', 'libreloc/src/TypedStageKernels.cpp',
               'libreloc/src/Dispatch.cpp', 'bench/issue265/POLICY.json')
    loaded = {name: {'file': path, 'sha256': digest(path)} for name, path in sorted({
        'exporter': compiler.executable.resolve(),
        'pyreloc': Path(pyreloc.__file__).resolve(),
        'reloc_torch': Path(reloc_torch.__file__).resolve(),
        **{path.name: path.resolve() for path in (build / 'libreloc').glob('libreloc_runtime.so*') if path.is_file()},
        **{name: Path(module.__file__).resolve() for name, module in sys.modules.copy().items()
           if name.startswith(('pyreloc.', 'reloc_torch.')) and getattr(module, '__file__', None)},
    }.items())}
    cpu_info = Path('/proc/cpuinfo')
    flags = next((line.partition(':')[2].split() for line in cpu_info.read_text().splitlines()
                  if line.startswith('flags')), []) if cpu_info.exists() else []
    result = {
        'schema_version': 1, 'complete': True, 'seed': SEED,
        'environment': dict(environment(), cpu_architecture=platform.machine(),
                            cpu_flags_observed=sorted(set(flags) & {'avx', 'avx2', 'avx512f', 'f16c'})),
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'witness_sha256': digest(__file__), 'source_sha256': {name: digest(ROOT / name) for name in sources},
        'runtime_files': loaded,
        'stage_probe': stage_probe.metadata if stage_probe is not None else None,
        'provenance': {
            'capture': 'normal torch.compile Dynamo callback delegated to RelocBackend',
            'compiler_output': 'portable typed plan and admitted stage manifest emitted by sym-reloc-export',
            'execution': 'automatic guarded selection and reuse of existing native prepared-group kernels',
            'fallback': 'original saved Torch region, with a matching eager invocation as the bitwise oracle',
            'cpu_flags': 'OS-reported ISA eligibility; completion reports are the runtime evidence',
            'manual_producer': False, 'per_shape_executable_codegen': False,
            'timing': ('timings omitted with the stage interposer preloaded' if stage_probe is not None else
                       'synchronized requests include a small observation hook. Detailed reports, hashes and oracle are read after timing.'),
        },
        'cases': {},
    }
    for case in TRACES:
        result['cases'][case] = {}
        for row_policy in ROW_POLICIES:
            torch._dynamo.reset()
            result['cases'][case][row_policy] = run_case(case, row_policy, stage_probe)
            print(case, row_policy, 'completed', len(TRACES[case]), 'exact requests', flush=True)
    # The Torch view bridge is loaded lazily by the first prepared group.
    for name, module in sys.modules.copy().items():
        path = getattr(module, '__file__', None)
        if name.startswith(('pyreloc.', 'reloc_torch.')) and path and name not in loaded:
            resolved = Path(path).resolve()
            loaded[name] = {'file': resolved, 'sha256': digest(resolved)}
    save(args.output, result)


if __name__ == '__main__':
    main()
