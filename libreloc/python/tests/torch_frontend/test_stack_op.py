"""reloc_torch::stack_transfer (torch.stack support): a Tensor[] custom op with an exact,
metadata-only fake kernel; the single-source op schemas stay unchanged."""
import inspect

import pytest
import torch

from conftest import StackedCountingRuntime, make_entry, stacked_recipe

DEFAULT_OPCHECK_UTILS = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")


def test_schemas():
    from reloc_torch import ops

    assert str(ops.STACKED_OP._schema) == (
        "reloc_torch::stack_transfer(Tensor[] srcs, str handle, SymInt[] symbols, "
        "SymInt[] out_shape, SymInt[] out_strides, Device device) -> Tensor"
    )
    assert ops.SCHEMA == (
        "(Tensor src, str handle, SymInt[] symbols, SymInt[] out_shape, "
        "SymInt[] out_strides, Device device) -> Tensor"
    )


@pytest.fixture
def registered(compiler):
    from reloc_torch.cache import REGISTRY

    runtime = StackedCountingRuntime()
    entry = make_entry(compiler.compile(stacked_recipe(count=3, dim=1)), runtime, lambda *xs: torch.stack(xs, 1))
    registration = REGISTRY.register(entry)
    yield registration.handle, entry, runtime
    registration.release()


def symbols(entry):
    return [{"s0": 4, "s1": 5}[name] for name in entry.compiled.symbols]


def test_opcheck_default_checks_pass_for_tensor_list_sources(registered):
    from reloc_torch import ops

    handle, entry, runtime = registered
    assert tuple(inspect.signature(torch.library.opcheck).parameters["test_utils"].default) == DEFAULT_OPCHECK_UTILS
    with torch.inference_mode():
        xs = [torch.arange(20, dtype=torch.float32).reshape(4, 5) + 100 * i for i in range(3)]
    args = (xs, handle, symbols(entry), [4, 3, 5], [15, 5, 1], torch.device("cpu"))
    assert torch.library.opcheck(ops.STACKED_OP, args) == {name: "SUCCESS" for name in DEFAULT_OPCHECK_UTILS}
    assert torch.equal(ops.STACKED_OP(*args), torch.stack(xs, 1))
    assert runtime.executions >= 1


def test_gradient_requiring_calls_fail_at_call_time(registered):
    from reloc_torch import ops

    handle, entry, _ = registered
    xs = [torch.ones(4, 5, requires_grad=True) for _ in range(3)]
    with pytest.raises(RuntimeError, match="do not implement autograd"):
        ops.STACKED_OP(xs, handle, symbols(entry), [4, 3, 5], [15, 5, 1], torch.device("cpu"))
