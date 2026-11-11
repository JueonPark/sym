from pathlib import Path

import pytest

from bench.issue263.support import load_module, verify_files


def test_repository_override_applies_before_import_side_effects_without_editing_source(tmp_path):
    source = tmp_path / 'original.py'
    source.write_text("from pathlib import Path\nREPO=Path('/unavailable-original-machine')\nvalue=(REPO/'witness').read_text()\n")
    original = source.read_bytes()
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / 'witness').write_text('portable import')
    module = load_module('issue263_portability_witness', source, repository=checkout)
    assert module.value == 'portable import'
    assert module.REPO == checkout
    assert source.read_bytes() == original


def test_manifest_check_rejects_changed_input(tmp_path):
    file = tmp_path / 'data'
    file.write_bytes(b'abc')
    expected = {'data': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}
    assert verify_files(tmp_path, expected) == expected
    file.write_bytes(b'abd')
    with pytest.raises(ValueError, match='data'):
        verify_files(tmp_path, expected)


def test_baseline_transitive_workload_change_is_rejected(tmp_path):
    import json
    from bench.issue263.support import baseline_sources, verify_repository
    expected=baseline_sources()['repository_sha256']
    for name in expected:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).resolve().parents[2]/name).read_bytes())
    assert verify_repository(tmp_path)==expected
    target=tmp_path/'libreloc/python/examples/workloads/llm_offload.py'
    target.write_text(target.read_text()+'\n# changed dependency\n')
    with pytest.raises(ValueError,match='llm_offload.py'):verify_repository(tmp_path)
