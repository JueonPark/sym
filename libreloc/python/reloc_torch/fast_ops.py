"""The fast lane inside ``torch.compile``: prepared programs as custom ops.

A prepared program (``TypedProgram``, ``IndexedProgram``, ``PooledProgram``)
is bound to one shape signature, so its output metadata is known before any
call. ``program.op(...)`` calls one of the functional custom ops below: the
fake kernel returns ``empty_strided`` of the declared metadata and the real
kernel runs the program. Dynamo therefore captures the call as one opaque
graph node (no graph break, ``fullgraph=True`` holds) and Inductor schedules
the surrounding compute around it; the op is also an ordinary eager call.

The handle is a string looked up in a process-local registry at execution
time only; the fake kernel never consults it. A program's handle dies with
the program or its ``FastTransfers`` owner, and a graph that still carries
it fails loudly instead of fabricating output. Stateful conveniences
(``FastPrefetch``, ``KVCacheSwap``) are not ops: they carry completions
across calls and sit between compiled regions, exactly where the Torch
controls keep their own prefetch queues.
"""

import torch
from torch import library

from . import compat
from .cache import HandleRegistry
from .ops import _no_backward, _reject_autograd

TYPED_QUALIFIED_NAME = "reloc_torch::fast_typed"
TYPED_SCHEMA = (
    "(Tensor src, str handle, SymInt[] out_shape, SymInt[] out_strides, "
    "Device device, ScalarType dtype) -> Tensor"
)
INDEXED_QUALIFIED_NAME = "reloc_torch::fast_indexed"
INDEXED_SCHEMA = (
    "(Tensor src, Tensor index, str handle, SymInt[] out_shape, SymInt[] out_strides, "
    "Device device, ScalarType dtype) -> Tensor"
)
POOLED_QUALIFIED_NAME = "reloc_torch::fast_pooled"
POOLED_SCHEMA = (
    "(Tensor[] indices, Tensor[] offsets, str handle, SymInt[] out_shape, "
    "Device device, ScalarType dtype) -> Tensor"
)

# Prepared programs by handle. Separate from reloc_torch.cache.REGISTRY (the
# backend's execution entries): a program is a per-shape object owned by a
# FastTransfers, and lookup() refuses it once that owner closed.
PROGRAMS = HandleRegistry()


def lookup_program(handle):
    return PROGRAMS.lookup(handle)


def _verify(out, out_shape, out_strides, device, dtype):
    # The declared metadata is the program's destination descriptor; a
    # mismatch means the graph was captured against another program.
    if (tuple(out.shape) != tuple(int(d) for d in out_shape)
            or out.dtype != dtype or out.device != torch.device(device)
            or (out_strides is not None and tuple(out.stride()) != tuple(int(s) for s in out_strides))):
        raise RuntimeError("fast op output does not match the declared metadata "
                           f"({tuple(out.shape)}, {out.dtype}, {out.device})")
    return out


def _define(name, schema, real, fake):
    existing = compat.existing_custom_op(name)
    if existing is not None:
        return existing
    op = library.custom_op(name, real, mutates_args=(), schema=schema)
    op.register_fake(fake)
    op.register_autograd(_no_backward, setup_context=_reject_autograd)
    return op


def _typed(src, handle, out_shape, out_strides, device, dtype):
    out = lookup_program(handle).execute(src)
    return _verify(out, out_shape, out_strides, device, dtype)


def _typed_fake(src, handle, out_shape, out_strides, device, dtype):
    return torch.empty_strided(out_shape, out_strides, dtype=dtype, device=device)


def _indexed(src, index, handle, out_shape, out_strides, device, dtype):
    out = lookup_program(handle).execute(src, index)
    return _verify(out, out_shape, out_strides, device, dtype)


def _indexed_fake(src, index, handle, out_shape, out_strides, device, dtype):
    return torch.empty_strided(out_shape, out_strides, dtype=dtype, device=device)


def _pooled(indices, offsets, handle, out_shape, device, dtype):
    # Asynchronous producer: the output is ordered on the caller's current
    # stream and the completion stays registered with the owner, so the
    # compute that follows in the graph overlaps the DMA tail.
    out = lookup_program(handle).submit(list(indices), list(offsets)).wait_stream()[0]
    return _verify(out, out_shape, None, device, dtype)


def _pooled_fake(indices, offsets, handle, out_shape, device, dtype):
    return torch.empty(out_shape, dtype=dtype, device=device)


fast_typed = _define(TYPED_QUALIFIED_NAME, TYPED_SCHEMA, _typed, _typed_fake)
fast_indexed = _define(INDEXED_QUALIFIED_NAME, INDEXED_SCHEMA, _indexed, _indexed_fake)
fast_pooled = _define(POOLED_QUALIFIED_NAME, POOLED_SCHEMA, _pooled, _pooled_fake)
TYPED_OP = torch.ops.reloc_torch.fast_typed.default
INDEXED_OP = torch.ops.reloc_torch.fast_indexed.default
POOLED_OP = torch.ops.reloc_torch.fast_pooled.default


def register(program):
    """Register a prepared program; returns the Registration that owns the handle."""
    return PROGRAMS.register(program)


__all__ = ("PROGRAMS", "TYPED_OP", "INDEXED_OP", "POOLED_OP", "TYPED_QUALIFIED_NAME", "TYPED_SCHEMA",
           "INDEXED_QUALIFIED_NAME", "INDEXED_SCHEMA", "POOLED_QUALIFIED_NAME", "POOLED_SCHEMA",
           "fast_typed", "fast_indexed", "fast_pooled", "lookup_program", "register")
