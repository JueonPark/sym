"""Indexed compiler contract, independent of Torch and libreloc."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

tool = shutil.which(sys.argv[1])
source = Path(sys.argv[2]).read_text()

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    def run(text, flags=('--indexed', '--typed'), code=0, reason=None):
        inp, plan, manifest = (root / n for n in ('in.mlir', 'plan.bin', 'manifest.json'))
        plan.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        inp.write_text(text)
        proc = subprocess.run([tool, str(inp), '--output', str(plan), '--manifest', str(manifest), *flags],
                              capture_output=True, text=True)
        assert proc.returncode == code, (proc.returncode, proc.stderr, text)
        if code == 1:
            assert proc.stderr and not plan.exists() and not manifest.exists()
            return
        meta = json.loads(manifest.read_text())
        if code == 2:
            assert meta['status'] == 'unsupported' and meta['reason'] == reason, meta
            assert not plan.exists()
            return
        blob = plan.read_bytes()
        assert meta['plan_sha256'] == hashlib.sha256(blob).hexdigest()
        assert meta['input_sha256'] == hashlib.sha256(inp.read_bytes()).hexdigest()
        return blob, meta

    blob, meta = run(source)
    assert blob[:8] == b'RPLN\x02\x00\x00\x00'
    assert meta['schema_version'] == 3 and meta['wire_version'] == 2
    assert meta['symbols'] == ['N', 'D', 'M']
    assert meta['logical_source']['shape'] == [['symbol', 'N'], ['symbol', 'D']]
    assert meta['logical_destination']['shape'] == [['symbol', 'M'], ['symbol', 'D']]
    assert meta['index_select']['indices']['dtype'] == 'int64'
    assert meta['index_select']['policy'] == 'ieee_rne'
    assert meta['index_select']['axis'] == 0
    assert run(source) == (blob, meta)
    run(source, flags=(), code=2, reason='indexed_unsupported')
    run(source, flags=('--indexed',), code=2, reason='typed_unsupported')
    for invalid in (source.replace('axis 0', 'axis 1'), source.replace('["M"], i64', '["M"], f32'),
                    source.replace('policy ieee_rne', 'policy exact')):
        run(invalid, code=1)

    static = 'func.func @s(%x: !sym.tensor<[8, 5], f32>, %i: !sym.tensor<[3], i64>) -> !sym.tensor<[3, 5], f32> {\n%0 = reloc.index_select %x, %i axis 0 : !sym.tensor<[8, 5], f32>, !sym.tensor<[3], i64> -> !sym.tensor<[3, 5], f32>\nreturn %0 : !sym.tensor<[3, 5], f32>\n}\n'
    identity, imeta = run(static, flags=('--indexed',))
    assert imeta['index_select']['policy'] == 'exact'
    assert identity[-5:] == struct.pack('<IB', 0, 1)
    # Frozen v2 bytes: no symbols; f32[8,5], i64[3], f32[3,5]; axis 0, exact.
    assert identity.hex() == (
        '52504c4e02000000000000000200000001000000010800000000000000010000000105'
        '000000000000000000000001000000010000000000000000002000000001000000'
        '010000000103000000000000000000000001000000010000000000000000024000'
        '000002000000010000000103000000000000000100000001050000000000000000'
        '0000000100000001000000000000000000200000000000000001'
    )
    run(static.replace('[3, 5]', '[4, 5]'), code=1)
    run(static.replace('[3], i64', '[0], i64'), code=1)
    run(static.replace('[3], i64', '[3, 1], i64'), code=1)

    # Cast-before-select and layout-after-select are not partially exported.
    prefix = static.replace('%0 = reloc.index_select %x', '%v = reloc.reshape %x to [8, 5] : !sym.tensor<[8, 5], f32> -> !sym.tensor<[8, 5], f32>\n%0 = reloc.index_select %v')
    run(prefix, code=2, reason='indexed_chain_unsupported')
    layout = static.replace('return %0', '%1 = reloc.reshape %0 to [3, 5] : !sym.tensor<[3, 5], f32> -> !sym.tensor<[3, 5], f32>\nreturn %1')
    run(layout, code=2, reason='fold_unsupported')
    folded = subprocess.run([str(Path(tool).with_name('sym-opt')), '--reloc-fold'],
                            input=source, capture_output=True, text=True, check=True).stdout
    assert 'reloc.indexed_plan_result' in folded
    run(folded, code=2, reason='prefolded_input')
    # The result operation verifies both operands against the plan descriptors.
    bad = folded.replace('%arg1: !sym.tensor<["M"], i64>', '%arg1: !sym.tensor<["M"], i32>')
    bad = bad.replace(', !sym.tensor<["M"], i64> ->', ', !sym.tensor<["M"], i32> ->')
    run(bad, code=1)

    # Opting in does not change existing v0/v1 bytes or manifests.
    for text in (
        'func.func @t(%x: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f32> {\n%0 = reloc.reshape %x to [4] : !sym.tensor<[4], f32> -> !sym.tensor<[4], f32>\nreturn %0 : !sym.tensor<[4], f32>\n}',
        'func.func @t(%x: !sym.tensor<[4], f32>) -> !sym.tensor<[4], f16> {\n%0 = reloc.cast %x policy ieee_rne : !sym.tensor<[4], f32> -> !sym.tensor<[4], f16>\nreturn %0 : !sym.tensor<[4], f16>\n}',
    ):
        assert run(text, flags=('--typed',)) == run(text)
print('indexed export contract passed')
