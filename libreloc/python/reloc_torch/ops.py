"""Functional relocation custom ops with exact, metadata-only fake kernels.

``reloc_torch::transfer`` (T3) is the layout-only op: output dtype equals the
source dtype. ``reloc_torch::typed_transfer`` (C4, issue #144) carries a
typed program: its output dtype is an explicit ``ScalarType`` argument and
its runtime parameters (scales, zero points) travel as an explicit
``Tensor[]`` operand so the fake kernel never consults process-local
runtime objects and the graph keeps every parameter dependency visible.
Both are registered once at module initialization through
``torch.library.custom_op``; outputs never alias inputs and have storage
offset zero.
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
TYPED_QUALIFIED_NAME = "reloc_torch::typed_transfer"
TYPED_SCHEMA = (
    "(Tensor src, Tensor[] parameters, str handle, SymInt[] symbols, "
    "SymInt[] out_shape, SymInt[] out_strides, Device device, ScalarType dtype) -> Tensor"
)


_AUTOGRAD_MESSAGE = (
    "reloc_torch custom ops do not implement autograd; T3 entry points fall "
    "back to the original region before reaching the op for gradient-requiring inputs"
)


def _reject_autograd(ctx, inputs, output):
    # Runs only when autograd is recording (grad mode on, an input requires
    # grad), after the kernel already ran below autograd with grad disabled;
    # the kernel cannot tell that redispatch from a user's no_grad scope, in
    # which requires_grad parameters are legitimately eligible. The produced
    # result is discarded and the call fails here instead of in a deferred
    # backward. T3's entry points never reach the op with autograd live: the
    # graph callable runs the original graph and the eager mode redispatches.
    raise RuntimeError(_AUTOGRAD_MESSAGE)


def _no_backward(ctx, grad):
    raise RuntimeError(_AUTOGRAD_MESSAGE)


def _execute(src, handle, symbols, out_shape, out_strides, device):
    entry = lookup_handle(handle)
    device = torch.device(device)
    declared = ConcreteDescriptor(
        tuple(int(dim) for dim in out_shape),
        tuple(int(dim) for dim in out_strides),
        compat.dtype_name(src.dtype),
        device,
    )
    return execute_or_fallback(entry, src, list(symbols), device, declared=declared)


def _execute_typed(src, parameters, handle, symbols, out_shape, out_strides, device, dtype):
    entry = lookup_handle(handle)
    device = torch.device(device)
    declared = ConcreteDescriptor(
        tuple(int(dim) for dim in out_shape),
        tuple(int(dim) for dim in out_strides),
        compat.dtype_name(dtype),
        device,
    )
    return execute_or_fallback(
        entry, src, list(symbols), device, declared=declared, parameters=list(parameters)
    )


def _define():
    # Registration is process-global and happens once: a reload or duplicate
    # import reuses the live definition, because re-registering would replace
    # the dispatcher entry and invalidate OpOverload objects already captured
    # in FX graphs.
    existing = compat.existing_custom_op(QUALIFIED_NAME)
    if existing is not None:
        return existing

    @library.custom_op(QUALIFIED_NAME, mutates_args=(), schema=SCHEMA)
    def transfer(src, handle, symbols, out_shape, out_strides, device):
        return _execute(src, handle, symbols, out_shape, out_strides, device)

    @transfer.register_fake
    def transfer_fake(src, handle, symbols, out_shape, out_strides, device):
        return torch.empty_strided(out_shape, out_strides, dtype=src.dtype, device=device)

    transfer.register_autograd(_no_backward, setup_context=_reject_autograd)
    return transfer


def _define_typed():
    existing = compat.existing_custom_op(TYPED_QUALIFIED_NAME)
    if existing is not None:
        return existing

    @library.custom_op(TYPED_QUALIFIED_NAME, mutates_args=(), schema=TYPED_SCHEMA)
    def typed_transfer(src, parameters, handle, symbols, out_shape, out_strides, device, dtype):
        return _execute_typed(src, parameters, handle, symbols, out_shape, out_strides, device, dtype)

    @typed_transfer.register_fake
    def typed_transfer_fake(src, parameters, handle, symbols, out_shape, out_strides, device, dtype):
        # The declared dtype is the recipe's destination dtype (C3 logical
        # descriptor); the real kernel verifies the produced tensor against it.
        return torch.empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    typed_transfer.register_autograd(_no_backward, setup_context=_reject_autograd)
    return typed_transfer


transfer = _define()
typed_transfer = _define_typed()
OP = torch.ops.reloc_torch.transfer.default
TYPED_OP = torch.ops.reloc_torch.typed_transfer.default


__all__ = ("OP", "QUALIFIED_NAME", "SCHEMA", "TYPED_OP", "TYPED_QUALIFIED_NAME", "TYPED_SCHEMA",
           "transfer", "typed_transfer")
