"""Compiler artifacts lowered to guarded, bounded prepared execution."""
import pytest
import torch
import pyreloc

from reloc_torch.fast import FastTransfers, TypedProgram
from reloc_torch.resources import TransferResources
from test_fast import dlrm_recipe


@pytest.mark.gpu
def test_borrowed_resources_survive_fast_owner_close(compiler, cuda_device):
    with TransferResources() as resources:
        source = torch.randn(26, 7, 64)
        for _ in range(2):
            with FastTransfers(cuda_device, compiler=compiler, resources=resources) as owner:
                out = owner.typed(dlrm_recipe(), source).execute(source)
                expected = source.transpose(0, 1).contiguous().to(cuda_device, torch.float16)
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
                assert owner.stats()['logical_transfers'] == 1
            assert not resources.closed


@pytest.mark.gpu
def test_borrowed_resource_close_rejects_submission(compiler, cuda_device):
    resources = TransferResources()
    owner = FastTransfers(cuda_device, compiler=compiler, resources=resources)
    try:
        source = torch.randn(26, 7, 64)
        program = owner.typed(dlrm_recipe(), source)
        resources.close()
        with pytest.raises(RuntimeError, match='closed'):
            program.execute(source)
    finally:
        owner.close()
        resources.close()


@pytest.mark.gpu
def test_compiler_template_factory_uses_existing_plan(compiler, cuda_device, monkeypatch):
    source = torch.randn(26, 7, 64)
    compiled = compiler.compile(dlrm_recipe())
    bindings = compiled.bind_values(source)
    bound = pyreloc.bind_typed(compiled.decoded_plan, bindings, {})
    template = pyreloc.prepare_dispatch_template(bound, 'h2d', 'cuda', threads=4)
    with FastTransfers(cuda_device, compiler=compiler, gather_threads=4) as owner:
        def unexpected(*args, **kwargs):
            pytest.fail('an existing compiler template must not be rebuilt or registered')
        monkeypatch.setattr(compiler, 'compile', unexpected)
        monkeypatch.setattr(pyreloc, 'bind_typed', unexpected)
        monkeypatch.setattr(TypedProgram, '_register_op', unexpected)
        program = TypedProgram._from_template(owner, compiled, source, bindings, bound, template, 0)
        assert program.bound is bound and program.template is template
        assert program.handle is None
        group = owner.prepare_group([program])
        for _ in range(2):
            source.neg_()
            out = group.submit([source]).wait()[0]
            expected = source.transpose(0, 1).contiguous().to(cuda_device, torch.float16)
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert owner.stats()['logical_transfers'] == 2
        with pytest.raises(ValueError):
            program.check(source.half())
        with pytest.raises(ValueError):
            program.check(torch.randn(26, 8, 64))
