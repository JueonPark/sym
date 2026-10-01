"""Benchmark policy/control checks and conservative GPU-timeline evidence."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'bench' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load('transfer_resource_reuse')
overlap = load('analyze_resource_overlap')


def test_fixed_chunk_control_and_small_effective_buffer_count():
    for shape in bench.SHAPES:
        one = bench.transpose_schedule(shape, 1)
        four = bench.transpose_schedule(shape, 4)
        assert one['chunk_bytes'] == four['chunk_bytes']
        assert sum(one['chunk_bytes']) == shape[0] * shape[1] * 4
    assert bench.transpose_schedule((4096, 1024), 4)['chunk_bytes'] == [4 << 20] * 4
    assert bench.transpose_schedule((256, 1024), 4)['effective_buffers'] == 1
    assert bench.distribution([1, 2, 3, 4, 5])['p95_ms'] == 5
    with pytest.raises(SystemExit):
        bench.parser().parse_args(['--output', 'unused', '--samples', '0'])


def fixture_trace(path, buffers=4, overlap_work=True, gpu=True):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, uint64Value INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, bytes INTEGER, streamId INTEGER, correlationId INTEGER, globalPid INTEGER, copyKind INTEGER);
        ''')
        db.execute('INSERT INTO NVTX_EVENTS VALUES(0,200000,1,NULL,?,NULL)',
                   (f'reloc.benchmark/buffers={buffers}/request=0',))
        for chunk, begin in enumerate([1000, 25000 if overlap_work else 55000]):
            db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,1,?,"reloc.gather.work",NULL)', (begin, begin+10000, chunk))
            db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,1,?,"reloc.h2d.submit",NULL)', (begin+11000, begin+14000, chunk))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,1,?)', (begin+12000, begin+13000, chunk))
            if gpu:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,4194304,7,?,0,1)', (begin+13000, begin+49000, chunk))


@pytest.mark.parametrize('buffers,work,expected', [(4, True, 'qualified'), (4, False, 'failed'),
                                                 (1, False, 'qualified'), (1, True, 'failed')])
def test_only_actual_next_chunk_work_during_gpu_dma_passes(tmp_path, buffers, work, expected):
    path = tmp_path / 'capture.sqlite'
    fixture_trace(path, buffers, work)
    result = overlap.analyze(path)
    assert result['status'] == expected
    assert len(overlap.chrome_trace(result)['traceEvents']) == 4


def test_missing_gpu_activity_cannot_masquerade_as_overlap(tmp_path):
    path = tmp_path / 'capture.sqlite'
    fixture_trace(path, gpu=False)
    with pytest.raises(overlap.MissingEvidence, match='GPU H2D'):
        overlap.analyze(path)
    # Multiple workers overlap each other; count their union, not their sum.
    assert overlap.union_ns([(0, 10), (5, 20), (25, 30)]) == 25


@pytest.mark.gpu
def test_resource_benchmark_completed_frontend_smoke(tmp_path, cuda_device):
    output = tmp_path / 'measurements.json'
    result = subprocess.run([sys.executable, str(ROOT / 'bench/transfer_resource_reuse.py'),
                             '--output', str(output), '--device', str(cuda_device.index),
                             '--threads', '1', '--shapes', '67x97', '--warmup', '2',
                             '--samples', '2', '--rounds', '1'],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert {row['scenario'] for row in report['results']} == {'steady', 'd2h', 'changing', 'concurrent', 'control'}
    for row in report['results']:
        if 'methods' in row:
            for method in row['methods'].values():
                assert method[0]['byte_exact']
                assert len(method[0]['samples_ms']) == 2
        else:
            assert row['byte_exact'] and row['completed_calls_per_s'] > 0
