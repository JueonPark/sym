"""Functional relocation custom op with an exact, metadata-only fake kernel.

``reloc_torch::transfer`` is registered once at module initialization through
``torch.library.custom_op`` with an explicit layout-only schema. Output dtype
equals the source dtype, the storage offset is zero, and the output never
aliases the input. Symbolic output metadata travels as explicit ``SymInt[]``
arguments so the fake kernel never consults process-local runtime objects.
"""

import torch
from torch import library

from . import compat
from .cache import lookup_handle
from .runtime import ConcreteDescriptor, execute_or_fallback


QUALIFIED_NAME = "reloc_torch::transfer"
SCHEMA = (
    "(Tensor src, str handle, SymInt[] symbols, SymInt[] out_shape, "
    "SymInt[] out_strides, Device device) -> Tensor"
)


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _verify(result, src, declared):
    if not isinstance(result, torch.Tensor):
        raise RuntimeError("reloc_torch::transfer produced a non-tensor result")
    if compat.tensors_alias(result, src):
        raise RuntimeError("reloc_torch::transfer result must not alias its input")
    actual = (tuple(result.shape), tuple(result.stride()), result.dtype, result.storage_offset())
    expected = (declared.shape, declared.strides, src.dtype, 0)
    if actual != expected:
        raise RuntimeError(
            f"reloc_torch::transfer output metadata mismatch: got {actual}, declared {expected}"
        )
    device = declared.device
    if result.device.type != device.type or (
        device.index is not None and result.device.index != device.index
    ):
        raise RuntimeError(
            f"reloc_torch::transfer output device {result.device} does not match declared {device}"
        )


_AUTOGRAD_MESSAGE = (
    "reloc_torch::transfer does not implement autograd; T3 entry points fall "
    "back to the original region before reaching the op for gradient-requiring inputs"
)


def _reject_autograd(ctx, inputs, output):
    # Runs only when autograd is recording (grad mode on, an input requires
    # grad). The shared preflight has already refused to launch for a source
    # that requires grad, so this turns a direct gradient-requiring call into
    # a clear call-time error instead of a deferred backward failure.
    raise RuntimeError(_AUTOGRAD_MESSAGE)


def _no_backward(ctx, grad):
    raise RuntimeError(_AUTOGRAD_MESSAGE)


def _execute(src, handle, symbols, out_shape, out_strides, device):
    entry = lookup_handle(handle)
    device = torch.device(device)
    declared = ConcreteDescriptor(
        tuple(int(dim) for dim in out_shape),
        tuple(int(dim) for dim in out_strides),
        _dtype_name(src.dtype),
        device,
    )
    result = execute_or_fallback(entry, src, list(symbols), device, declared=declared)
    _verify(result, src, declared)
    return result


def _define():
    @library.custom_op(QUALIFIED_NAME, mutates_args=(), schema=SCHEMA)
    def transfer(src, handle, symbols, out_shape, out_strides, device):
        return _execute(src, handle, symbols, out_shape, out_strides, device)

    @transfer.register_fake
    def transfer_fake(src, handle, symbols, out_shape, out_strides, device):
        return torch.empty_strided(out_shape, out_strides, dtype=src.dtype, device=device)

    transfer.register_autograd(_no_backward, setup_context=_reject_autograd)
    return transfer


transfer = _define()
OP = torch.ops.reloc_torch.transfer.default


__all__ = ("OP", "QUALIFIED_NAME", "SCHEMA", "transfer")
