"""The canonical/raw accounting arms must both include H2D transport."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_identity_accounting_times_h2d_in_both_modes(cuda_device, tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[4] / 'bench/beat_torch/plans.py'
    spec = importlib.util.spec_from_file_location('plan_accounting', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'plans.json'
    monkeypatch.setattr('sys.argv', [str(path), '--samples', '1', '--json', str(output)])
    module.main()
    rows = json.loads(output.read_text())
    for mode in ('canonical', 'raw'):
        assert rows[mode]['identity_view_2048x128']['per_call_us'] > 0
