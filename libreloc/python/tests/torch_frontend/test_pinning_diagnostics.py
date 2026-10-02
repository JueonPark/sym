"""Pinning reports describe the actual managed allocation, never tensor data."""
import pytest
import torch

from reloc_torch import TransferResources
from reloc_torch.recipe import Dequantize, InlineParam, Recipe, TensorSpec, Transpose
from reloc_torch.symbolic import Const, dense_strides
from reloc_torch import dispatch, transport


def spec(shape, dtype='float32'):
    shape = tuple(Const(n) for n in shape)
    return TensorSpec(shape, dense_strides(shape), Const(0), dtype)


@pytest.mark.gpu
def test_layout_report_distinguishes_wire_capacity_and_reuse(compiler, cuda_device):
    recipe = Recipe(spec((8, 4)), (Transpose((1, 0)),), spec((4, 8)), 'h2d')
    compiled = compiler.compile(recipe)
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    with TransferResources() as owner:
        reports = []
        for threshold in (None, 128, 128, 129):
            request = transport.prepare_transfer(compiled, source, cuda_device)
            out = transport.execute_transfer(request, resources=owner, min_pinned_bytes=threshold)
            assert torch.equal(out.cpu(), source.t())
            assert len(request.staging) == 1
            reports.append(request.staging[0])
        assert [r['reason'] for r in reports] == [
            'unconfigured_threshold', 'configured_size_gate',
            'configured_size_gate', 'below_threshold']
        assert [r['memory_kind'] for r in reports] == ['pageable', 'pinned', 'pinned', 'pageable']
        assert [r['reused'] for r in reports] == [False, False, True, True]
        assert all(r['wire_bytes'] == 128 for r in reports)
        assert all(r['staging_capacity_bytes'] == 256 << 10 for r in reports)
        assert reports[0]['min_pinned_bytes'] is None
        assert reports[1]['min_pinned_bytes'] == 128


@pytest.mark.gpu
def test_typed_parameter_budget_includes_busy_device_scratch(compiler, cuda_device):
    recipe = Recipe(spec((32, 2), 'int8'),
        (Dequantize('float32', InlineParam('float32', (), (0x3f000000,)), None, None, 'affine'),
         Transpose((1, 0))), spec((2, 32)), 'h2d')
    compiled = compiler.compile(recipe)
    source = torch.arange(64, dtype=torch.int8).reshape(32, 2)
    for limit, expected_kind in [(64, 'pageable'), (128, 'pinned')]:
        with TransferResources(max_typed_retained_bytes=limit) as owner:
            request = dispatch.prepare_typed_transfer(compiled, source, cuda_device,
                implementation='cuda_dequant_relocate')
            result = dispatch.execute_typed_transfer(request, resources=owner, min_pinned_bytes=1)
            assert torch.equal(result.tensor.cpu(), source.t().float() * .5)
            rows = result.report['staging']
            assert len(rows) == 1  # dense input uploads directly; only scales need staging
            assert rows[0]['wire_bytes'] == 8
            assert rows[0]['memory_kind'] == expected_kind
            assert rows[0]['retention_eligible'] == (limit == 128)
            assert rows[0]['reason'] == ('configured_size_gate' if limit == 128 else 'ephemeral_staging')
            assert owner.stats()['typed']['retained_bytes'] <= limit


def test_optional_threshold_validation_and_scalar_diagnostic_aggregation():
    from reloc_torch.resources import _transfer_configuration
    from reloc_torch.diagnostics import Diagnostics
    assert _transfer_configuration(None, {'min_pinned_bytes': None}) == {'min_pinned_bytes': None}
    for invalid in (True, -1, 1.5, '8MiB'):
        with pytest.raises((TypeError, ValueError)):
            _transfer_configuration(None, {'min_pinned_bytes': invalid})
    diagnostics = Diagnostics()
    row = {'memory_kind': 'pageable', 'reason': 'unconfigured_threshold', 'reused': False}
    diagnostics.record_staging([row])
    row.clear()
    assert diagnostics.snapshot()['staging_decisions'] == {'pageable:unconfigured_threshold': 1}
    assert diagnostics.snapshot()['staging_reuse'] == {'allocated': 1}
