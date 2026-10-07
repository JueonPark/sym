"""Compiler -> wire v2 -> native binding/dispatch, without importing Torch."""
import json
import struct
import subprocess
import sys

import numpy as np
import pytest
import pyreloc
from typed_support import REPO_ROOT, reloc_export_executable


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp('indexed-native')
    source = REPO_ROOT / 'test/dialect/reloc/indexed_export.mlir'
    plan, manifest = root / 'plan.bin', root / 'manifest.json'
    subprocess.run([str(reloc_export_executable()), str(source), '--indexed', '--typed',
                    '--output', str(plan), '--manifest', str(manifest)], check=True, capture_output=True)
    return plan, json.loads(manifest.read_text())


def view(array):
    return pyreloc.BufferView(array.ctypes.data, array.nbytes, 0, list(array.shape),
        [s // array.itemsize for s in array.strides], array.itemsize, 'host')


def bind(blob, values=(7, 0, 7), **symbols):
    indices = np.asarray(values, dtype='<i8')
    return pyreloc.bind_indexed(pyreloc.load_indexed_plan(blob),
        {'N': 8, 'D': 5, 'M': len(values), **symbols}, ('int64', [len(values)], indices.tobytes()))


@pytest.mark.parametrize('rows,width,count', [(8, 5, 3), (3, 129, 11), (19, 1, 7)])
def test_symbolic_native_execution_and_accounting(artifact, rows, width, count):
    path, _ = artifact
    x = np.arange(rows * width, dtype=np.float32).reshape(rows, width)
    indices = (np.arange(count) * 7 + 2) % rows
    bound = bind(path.read_bytes(), indices, N=rows, D=width)
    assert bound.source_extents == [rows, width]
    assert bound.result_extents == [count, width]
    out = np.empty((count, width), np.float16)
    program = pyreloc.prepare_index_select_program(bound, view(x))
    request = pyreloc.prepare_dispatch(program, view(x), view(out), 'h2d')
    report = pyreloc.execute_dispatch(request, gather_threads=2)
    np.testing.assert_array_equal(out, x[indices].astype(np.float16))
    assert report['artifact_version'] == 2
    assert report['source_bytes'] == x.nbytes
    assert report['parameter_bytes'] == count * 8
    assert report['payload_bytes_transferred'] == out.nbytes
    with pytest.raises(pyreloc.TransferError):
        pyreloc.prepare_dispatch(program, view(x), view(out), 'd2h')


def test_fresh_process_only_needs_saved_wire_and_runtime(artifact):
    proc = subprocess.run([sys.executable, '-c', '''
import sys, struct, numpy as np, pyreloc
from pathlib import Path
plan = pyreloc.load_indexed_plan(Path(sys.argv[1]).read_bytes())
bound = pyreloc.bind_indexed(plan, {'N': 4, 'D': 3, 'M': 5},
    ('int64', [5], struct.pack('<5q', 3, 0, 3, 1, 2)))
x = np.arange(12, dtype=np.float32).reshape(4, 3)
y = np.zeros((5, 3), dtype=np.float16)
def view(a):
    return pyreloc.BufferView(a.ctypes.data, a.nbytes, 0, list(a.shape),
        [s // a.itemsize for s in a.strides], a.itemsize, 'host')
program = pyreloc.prepare_index_select_program(bound, view(x))
request = pyreloc.prepare_dispatch(program, view(x), view(y), 'h2d')
pyreloc.execute_dispatch(request)
np.testing.assert_array_equal(y, x[[3, 0, 3, 1, 2]].astype(np.float16))
assert 'torch' not in sys.modules and 'reloc_torch' not in sys.modules
''', str(artifact[0])], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_decoder_rejects_corruption_and_wrong_loaders(artifact):
    blob = artifact[0].read_bytes()
    for end in range(len(blob)):
        with pytest.raises(pyreloc.DecodeError):
            pyreloc.load_indexed_plan(blob[:end])
    for bad in (blob + b'\x00', b'FAIL' + blob[4:], blob[:4] + struct.pack('<I', 1) + blob[8:],
                blob[:-5] + struct.pack('<IB', 1, 0), blob[:-1] + b'\xff', blob[:-1] + b'\x01'):
        with pytest.raises(pyreloc.DecodeError):
            pyreloc.load_indexed_plan(bad)
    for load in (pyreloc.load_plan, pyreloc.load_typed_plan):
        with pytest.raises(pyreloc.DecodeError):
            load(blob)


@pytest.mark.parametrize('values,symbols', [([-1], {}), ([8], {}), ([], {}),
    ([0], {'N': 0}), ([0], {'D': 0}), ([0], {'D': 2**62}), ([0], {'M': 2}), ([0], {'extra': 1})])
def test_binder_rejects_invalid_values_and_shapes(artifact, values, symbols):
    with pytest.raises(pyreloc.BindError):
        bind(artifact[0].read_bytes(), values, **symbols)


@pytest.mark.parametrize('operand', [('int32', [3], bytes(12)), ('int64', [3, 1], bytes(24)),
                                    ('int64', [2], bytes(16)), ('int64', [3], bytes(23))])
def test_binder_rejects_index_descriptor_and_bytes(artifact, operand):
    plan = pyreloc.load_indexed_plan(artifact[0].read_bytes())
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_indexed(plan, {'N': 8, 'D': 5, 'M': 3}, operand)
    with pytest.raises(pyreloc.BindError):
        pyreloc.bind_indexed(plan, {'N': 8, 'M': 3}, operand)


def test_symbolic_result_must_agree_with_both_operands(artifact):
    blob = artifact[0].read_bytes()
    # Replace the final extent's symbol D with N (both valid symbol indices).
    symbol_d = struct.pack('<IBI', 1, 0, 1)
    position = blob.rfind(symbol_d)
    assert position > 0
    bad = blob[:position] + struct.pack('<IBI', 1, 0, 0) + blob[position + 9:]
    with pytest.raises(pyreloc.BindError, match='result shape'):
        bind(bad)


def test_binding_owns_index_values(artifact):
    indices = np.array([7, 0, 7], dtype=np.int64)
    bound = bind(artifact[0].read_bytes(), indices)
    indices[:] = 1
    x = np.arange(40, dtype=np.float32).reshape(8, 5)
    out = np.empty((3, 5), np.float16)
    program = pyreloc.prepare_index_select_program(bound, view(x))
    pyreloc.execute_dispatch(pyreloc.prepare_dispatch(program, view(x), view(out), 'h2d'))
    np.testing.assert_array_equal(out, x[[7, 0, 7]].astype(np.float16))
