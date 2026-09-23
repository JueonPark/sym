"""Version qualification and version-pinned Torch access points."""

from __future__ import annotations

import platform
import re
import sys
import sysconfig


_QUALIFIED_PYTHON = (3, 14, 7)
_QUALIFIED_SOABI = "cpython-314-x86_64-linux-gnu"
_QUALIFIED_TORCH = (2, 14, 0)
_TORCH_VERSION = re.compile(
    r"^(?P<base>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>\+[a-z0-9.]+)?$"
)


class CompatibilityError(RuntimeError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def _validate_version(
    *,
    python_version: tuple[int, int, int],
    python_releaselevel: str,
    implementation: str,
    gil_enabled: bool,
    soabi: str | None,
    torch_version: str,
    torch_cuda: str | None,
) -> None:
    if python_version != _QUALIFIED_PYTHON or python_releaselevel != "final":
        raise CompatibilityError(
            "unsupported_python_version", f"expected 3.14.7, got {python_version}"
        )
    if (
        implementation != "CPython"
        or not gil_enabled
        or soabi != _QUALIFIED_SOABI
    ):
        raise CompatibilityError(
            "unsupported_python_build",
            f"expected regular-GIL CPython cp314, got {implementation} {soabi}",
        )

    match = _TORCH_VERSION.fullmatch(torch_version)
    if match is None or tuple(
        int(match.group(name)) for name in ("base", "minor", "patch")
    ) != _QUALIFIED_TORCH:
        raise CompatibilityError(
            "unsupported_torch_version", f"expected 2.14.0, got {torch_version}"
        )

    suffix = match.group("suffix")
    if suffix == "+cpu" and torch_cuda is None:
        return
    if suffix == "+cu126" and torch_cuda == "12.6":
        return
    raise CompatibilityError(
        "unsupported_torch_build",
        f"expected +cpu/None or +cu126/12.6, got {suffix}/{torch_cuda}",
    )


def check_version() -> None:
    """Reject execution outside the one qualified Python/Torch baseline."""
    import torch

    gil_probe = getattr(sys, "_is_gil_enabled", None)
    _validate_version(
        python_version=sys.version_info[:3],
        python_releaselevel=sys.version_info.releaselevel,
        implementation=platform.python_implementation(),
        gil_enabled=bool(gil_probe and gil_probe()),
        soabi=sysconfig.get_config_var("SOABI"),
        torch_version=str(torch.__version__),
        torch_cuda=torch.version.cuda,
    )


def torch_dispatch_mode_type():
    """Return the version-pinned Python dispatch-mode base class."""
    from torch.utils._python_dispatch import TorchDispatchMode

    return TorchDispatchMode


def operator_name(func) -> str:
    return str(func)


def bound_schema_arguments(func, args, kwargs):
    bound = []
    for index, argument in enumerate(func._schema.arguments):
        if index < len(args):
            value = args[index]
        elif argument.name in kwargs:
            value = kwargs[argument.name]
        else:
            continue
        alias = argument.alias_info
        bound.append((argument.name, value, bool(alias is not None and alias.is_write)))
    return tuple(bound)


def is_layout_operator(func) -> bool:
    import torch

    return func in {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.as_strided.default,
        torch.ops.aten.clone.default,
    }


def is_foreach_copy_operator(func) -> bool:
    import torch

    return func is torch.ops.aten._foreach_copy_.default


def tensors_alias(left, right) -> bool:
    import torch

    return bool(torch._C._is_alias_of(left, right))


def is_tensor(value) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def is_plain_tensor_or_parameter(value) -> bool:
    import torch

    return type(value) in (torch.Tensor, torch.nn.Parameter)


from .records import TensorMetadata

def _dimension(value):
    return value if type(value) is int else str(value)


def tensor_metadata(tensor, *, graph=False):
    try:
        device = tensor.device
        device_type, device_index = device.type, device.index
    except Exception:
        device_type, device_index = "unknown", None
    try:
        layout = str(tensor.layout).removeprefix("torch.")
    except Exception:
        layout = "unknown"
    capacity = None
    if layout == "strided":
        try:
            value = tensor.untyped_storage().nbytes()
            capacity = value if type(value) is int else None
        except Exception:
            pass
    pinned = False
    if device_type == "cpu" and layout == "strided":
        try:
            pinned = bool(tensor.is_pinned())
        except Exception:
            pass
    try:
        shape = tuple(_dimension(value) for value in tensor.shape)
    except Exception:
        shape = ()
    try:
        strides = tuple(_dimension(value) for value in tensor.stride())
    except Exception:
        strides = ()
    try:
        storage_offset = _dimension(tensor.storage_offset())
    except Exception:
        storage_offset = 0
    try:
        dtype = str(tensor.dtype).removeprefix("torch.")
    except Exception:
        dtype = "unknown"
    try:
        requires_grad = bool(tensor.requires_grad)
    except Exception:
        requires_grad = True
    return TensorMetadata(
        shape=shape,
        strides=strides,
        storage_offset=storage_offset,
        dtype=dtype,
        device_type=device_type,
        device_index=device_index,
        requires_grad=requires_grad,
        layout=layout,
        pinned=pinned,
        is_subclass=not (is_plain_tensor_or_parameter(tensor) or (graph and is_fake_tensor(tensor))),
        storage_capacity_bytes=capacity,
    )



def is_fake_tensor(value):
    from torch._subclasses.fake_tensor import FakeTensor
    return type(value) is FakeTensor


def graph_value(node):
    return node.meta.get("val", node.meta.get("example_value"))


def graph_target(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "_schema"):
        return str(target)
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
    if module == "_operator":
        module = "operator"
    return f"{module}.{name}" if module else name


def graph_mutates(node):
    if node.op == "call_method":
        return str(node.target).endswith("_")
    schema = getattr(node.target, "_schema", None)
    return bool(schema and any(a.alias_info and a.alias_info.is_write for a in schema.arguments))


def graph_nonblocking(node):
    if hasattr(node.target, "_schema"):
        options = dict((k, v) for k, v, _ in bound_schema_arguments(node.target, node.args, node.kwargs))
        return options.get("non_blocking", False) is not False
    if "non_blocking" in node.kwargs:
        return node.kwargs["non_blocking"] is not False
    args = node.args[1:]
    # The pinned Tensor.to overloads put non_blocking after dtype, device+dtype,
    # or other tensor; checking literal booleans avoids interpreting FX values.
    if node.target == "to":
        import torch
        index = 2 if args and isinstance(args[0], (str, torch.device)) else 1
        return len(args) > index and args[index] is not False
    if node.target in {"cuda", "copy_"}:
        return len(args) > 1 and args[1] is not False
    return False


# Pinned view semantics remain authoritative when schemas omit alias metadata
# or real-mode make_fx stores independently allocated fake tensor snapshots.
# In particular, aten._unsafe_view shares storage despite an unannotated return.
_KNOWN_VIEW_TARGETS = frozenset({
    "aten.view.default", "aten._unsafe_view.default", "aten.transpose.int",
    "aten.permute.default", "aten.as_strided.default",
})


def graph_may_alias_inputs(node):
    """Conservative alias edges using pinned exceptions plus schema contracts."""
    if graph_target(node.target) in _KNOWN_VIEW_TARGETS:
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is not None and any(result.alias_info for result in schema.returns):
        return True
    return node.op == "call_method" and node.target in {
        "view", "reshape", "transpose", "permute", "detach", "contiguous", "to"
    }


def graph_alias_semantics(node, source, destination):
    # Real-mode make_fx snapshots can have independently allocated fake metadata;
    # schema view guarantees remain authoritative when metadata loses storage IDs.
    target = graph_target(node.target)
    if target in _KNOWN_VIEW_TARGETS:
        return "aliases"
    if tensors_alias(source, destination):
        return "aliases"
    if source.device != destination.device or source.dtype != destination.dtype:
        return "distinct"
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        # An alias annotation permits aliasing; independently snapshotted fake
        # storage cannot disprove it. Unannotated tensor returns are fresh.
        return "unknown" if any(r.alias_info is not None for r in schema.returns) else "distinct"
    if node.op == "call_method" and target == "to" and node.kwargs.get("copy") is True:
        return "distinct"
    return "unknown"


class SymbolicContext:
    """Pinned SymInt conversion; environment identity is part of provenance.

    Only atomic, backed source dimensions introduce symbols. Reading expressions
    and membership of backed_var_to_val never reads a concrete hint or asks Torch
    to install a guard. Other dimensions may reference those source symbols.
    """
    def __init__(self):
        self._symbols = {}
        self.sources = ()

    @classmethod
    def from_tensor(cls, tensor):
        import sympy
        import torch
        from .symbolic import SymbolSource, UnsupportedSymbolicExpr
        context = cls()
        sources = []
        for axis, value in enumerate(tensor.shape):
            if type(value) is int:
                continue
            if not isinstance(value, torch.SymInt):
                raise UnsupportedSymbolicExpr('noninteger source dimension')
            node = value.node
            expr, env = node.expr, node.shape_env
            if not isinstance(expr, sympy.Symbol) or env is None or expr not in env.backed_var_to_val:
                raise UnsupportedSymbolicExpr('source dimension is not an independent backed symbol')
            key = (env, expr)
            if key in context._symbols:
                index = context._symbols[key]
                source = sources[index]
                sources[index] = SymbolSource(source.name, source.axis, source.equal_axes + (axis,))
            else:
                index = len(sources)
                context._symbols[key] = index
                sources.append(SymbolSource(f's{index}', axis))
        context.sources = tuple(sources)
        return context

    def expression(self, value, *, positive=()):
        import sympy
        import torch
        from torch.utils._sympy.functions import FloorDiv as TorchFloorDiv, PythonMod, Mod as TorchMod, Max as TorchMax
        from .symbolic import Const, Symbol, FloorDiv, Mod, UnsupportedSymbolicExpr, add, mul
        from functools import reduce
        if type(value) is int:
            return Const(value)
        if not isinstance(value, torch.SymInt):
            raise UnsupportedSymbolicExpr('expected integer or backed SymInt')
        env = value.node.shape_env

        def convert(expr):
            if isinstance(expr, sympy.Integer):
                return Const(int(expr))
            if isinstance(expr, sympy.Symbol):
                index = self._symbols.get((env, expr))
                if index is None or env is None or expr not in env.backed_var_to_val:
                    raise UnsupportedSymbolicExpr('unbacked or non-source symbol')
                return Symbol(self.sources[index].name)
            if expr.func in (sympy.Max, TorchMax) and len(expr.args) == 2 and sympy.Integer(1) in expr.args:
                other = next(a for a in expr.args if a != sympy.Integer(1))
                converted = convert(other)
                if converted in positive:
                    return converted
                raise UnsupportedSymbolicExpr('Max extent lacks a positive guard')
            if expr.func is sympy.Add:
                return reduce(add, (convert(a) for a in expr.args))
            if expr.func is sympy.Mul:
                return reduce(mul, (convert(a) for a in expr.args))
            if expr.func in (TorchFloorDiv, PythonMod, TorchMod, sympy.Mod):
                lhs, divisor = expr.args
                if not isinstance(divisor, sympy.Integer) or divisor <= 0:
                    raise UnsupportedSymbolicExpr('symbolic or nonpositive divisor')
                kind = FloorDiv if expr.func is TorchFloorDiv else Mod
                return kind(convert(lhs), int(divisor))
            # SymPy combines repeated source dimensions (N*N) into an integer
            # power. Expand that structural abbreviation into the Mul vocabulary.
            if expr.func is sympy.Pow and isinstance(expr.args[1], sympy.Integer) and expr.args[1] >= 0:
                return reduce(mul, (convert(expr.args[0]) for _ in range(int(expr.args[1]))), Const(1))
            raise UnsupportedSymbolicExpr(f'unknown expression function {expr.func}')
        return convert(value.node.expr)

    def tensor_spec(self, tensor, *, require_dense=True, positive_shape=()):
        from .recipe import TensorSpec
        from .symbolic import Const, dense_strides, UnsupportedSymbolicExpr
        shape = tuple(self.expression(d) for d in tensor.shape)
        strides = tuple(self.expression(d, positive=positive_shape) for d in tensor.stride())
        offset = self.expression(tensor.storage_offset())
        if require_dense and (not shape or strides != dense_strides(shape) or offset != Const(0)):
            raise UnsupportedSymbolicExpr('source_layout: expected dense zero-offset tensor')
        return TensorSpec(shape, strides, offset, str(tensor.dtype).removeprefix('torch.'))


def dtype_name(dtype):
    """Canonical frontend dtype name ('float32') for a torch.dtype or its str."""
    return str(dtype).removeprefix("torch.")


def storage_span(tensor):
    """Allocation base, capacity in bytes and the logical byte offset of
    element 0, read from the tensor's untyped storage rather than from
    ``data_ptr()``/``numel()`` (R2, issue #146)."""
    storage = tensor.untyped_storage()
    return (
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()) * tensor.element_size(),
    )


def storage_snapshot(tensor):
    """Metadata plus storage identity used to detect a stale prepared request."""
    base, capacity, offset = storage_span(tensor)
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        offset,
        str(tensor.dtype),
        str(tensor.device),
        base,
        capacity,
    )


def cuda_stream_handle(device):
    """The caller's current CUDA stream on `device` as a raw cudaStream_t
    handle (0 is the legacy default stream)."""
    import torch

    return int(torch.cuda.current_stream(torch.device(device)).cuda_stream)


def existing_custom_op(qualname):
    """Return the live ``CustomOpDef`` registered under ``qualname``, or ``None``.

    Pinned internal access: re-registering a custom op replaces its dispatcher
    entry and invalidates ``OpOverload`` objects captured earlier (including FX
    node targets), so module reloads must reuse the existing definition.
    """
    from torch._library.custom_ops import _maybe_get_opdef

    return _maybe_get_opdef(qualname)


def statically_at_least(value, bound):
    """True when ``value >= bound`` is provable from the pinned ShapeEnv without
    installing a guard or reading a hint; False for anything unprovable."""
    import torch

    if type(value) is int:
        return value >= bound
    if isinstance(value, torch.SymInt):
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        try:
            if statically_known_true(value >= bound):
                return True
            # Dynamo records contiguity/reshape decisions as inequality
            # guards (Ne(s // 64, 1), Ne(s // 64, 0)); for a non-negative
            # integer, excluding every value below the bound proves it.
            return bool(statically_known_true(value >= 0)) and all(
                bool(statically_known_true(value != k)) for k in range(bound))
        except Exception:
            return False
    return False


def symbolic_capture(function, *inputs):
    """Capture canonical ATen code without real transfers on the pinned wheel."""
    check_version()
    from torch.fx.experimental.proxy_tensor import make_fx
    return make_fx(function, tracing_mode='symbolic')(*inputs)


def fx_kind(node):
    """Closed normalization vocabulary, from the T1 pinned capture inventory."""
    import operator
    import torch
    aten = torch.ops.aten
    kinds = {
        aten._to_copy.default: 'transfer', aten.to.device: 'transfer',
        aten.to.dtype: 'transfer', aten.to.other: 'transfer',
        aten.permute.default: 'permute', aten.transpose.int: 'transpose',
        aten.t.default: 'transpose', torch.t: 'transpose',
        aten.view.default: 'reshape', aten.reshape.default: 'reshape',
        aten._unsafe_view.default: 'reshape',
        aten.clone.default: 'materialize', aten.contiguous.default: 'materialize',
        aten.constant_pad_nd.default: 'pad', aten.sym_size.int: 'scalar',
        torch.transpose: 'transpose', torch.permute: 'permute',
        torch.reshape: 'reshape', torch.clone: 'materialize',
        torch.nn.functional.pad: 'pad', torch._C._nn.pad: 'pad',
        operator.floordiv: 'scalar', operator.mul: 'scalar',
        operator.add: 'scalar', operator.sub: 'scalar', operator.mod: 'scalar',
        operator.getitem: 'scalar', getattr: 'scalar',
    }
    if node.op == 'call_function':
        return kinds.get(node.target)
    if node.op == 'call_method':
        return {'to': 'transfer', 'cpu': 'transfer', 'cuda': 'transfer',
                'permute': 'permute', 'transpose': 'transpose', 't': 'transpose', 'view': 'reshape',
                'reshape': 'reshape', 'contiguous': 'materialize',
                'size': 'scalar'}.get(node.target)
    return None


def fx_canonical_call(node, source_value):
    """Return a 1:1 canonical call; constants and explicit options are retained.

    Tensor.to's Python overload parser is used only to bind arguments, never to
    execute a tensor operation. The original node remains the fallback authority.
    """
    import torch
    aten = torch.ops.aten
    kind = fx_kind(node)
    if kind is None:
        raise ValueError('unrecognized_fx_target')
    if node.target in (aten.t.default, torch.t) or (node.op == 'call_method' and node.target == 't'):
        # Tensor.t() is transpose(0, 1) for rank 2 (and rank 0/1 identity,
        # which the rank check in the recipe rejects conservatively).
        args = node.args or (node.kwargs.get('self') or node.kwargs.get('input'),)
        if not args or args[0] is None:
            raise ValueError('unrecognized_fx_target')
        return aten.transpose.int, (args[0], 0, 1), {}
    if node.op == 'call_function' and hasattr(node.target, '_schema'):
        return node.target, node.args, dict(node.kwargs)
    args, kwargs = node.args, dict(node.kwargs)
    if not args and 'input' in kwargs:
        args = (kwargs.pop('input'),)
    if not args:
        raise ValueError('unrecognized_fx_target')
    source = args[0]
    if kind == 'transfer':
        if source_value is None:
            raise ValueError('metadata_unavailable')
        if node.target == 'to':
            to_args = args[1:]
            copy = kwargs.get('copy', False)
            copy_index = 3 if to_args and isinstance(to_args[0], (str, torch.device)) else 2
            if len(to_args) > copy_index:
                if len(to_args) != copy_index + 1 or 'copy' in kwargs:
                    raise ValueError('unrecognized_fx_target')
                copy = to_args[-1]
                to_args = to_args[:-1]
            if type(copy) is not bool:
                raise ValueError('unrecognized_fx_target')
            device, dtype, nonblocking, memory_format = torch._C._nn._parse_to(*to_args, **{k: v for k, v in kwargs.items() if k != 'copy'})
            device = source_value.device if device is None else device
            dtype = source_value.dtype if dtype is None else dtype
            return aten.to.device, (source, device, dtype), {
                'non_blocking': nonblocking, 'copy': copy,
                'memory_format': memory_format}
        if node.target == 'cpu':
            if args[1:]:
                kwargs['memory_format'] = args[1]
            return aten.to.device, (source, torch.device('cpu'), source_value.dtype), kwargs
        device = args[1] if len(args) > 1 else kwargs.pop('device', None)
        if device is None:
            device = torch.device('cuda')
        elif type(device) is int:
            device = torch.device('cuda', device)
        else:
            device = torch.device(device)
        if len(args) > 2:
            kwargs['non_blocking'] = args[2]
        if len(args) > 3:
            kwargs['memory_format'] = args[3]
        return aten.to.device, (source, device, source_value.dtype), kwargs
    if kind in ('permute', 'reshape'):
        key = 'dims' if kind == 'permute' else ('size' if node.target == 'view' else 'shape')
        values = args[1:]
        if not values:
            values = kwargs.pop(key)
        elif len(values) == 1 and isinstance(values[0], (tuple, list)):
            values = values[0]
        target = aten.permute.default if kind == 'permute' else (
            aten.view.default if node.target == 'view' else aten.reshape.default)
        return target, (source, values), kwargs
    if kind == 'transpose':
        return aten.transpose.int, args, kwargs
    if kind == 'materialize':
        target = aten.contiguous.default if node.target == 'contiguous' else aten.clone.default
        if len(args) > 1:
            kwargs['memory_format'] = args[1]
        return target, (source,), kwargs
    if kind == 'pad':
        pad = args[1] if len(args) > 1 else kwargs.pop('pad')
        mode = args[2] if len(args) > 2 else kwargs.pop('mode', 'constant')
        value = args[3] if len(args) > 3 else kwargs.pop('value', None)
        if mode != 'constant':
            raise ValueError('unsupported_padding')
        return aten.constant_pad_nd.default, (source, pad, 0 if value is None else value), kwargs
    if node.op == 'call_method' and node.target == 'size' and len(args) == 2:
        return aten.sym_size.int, args, kwargs
    if node.op == 'call_method':
        raise ValueError('unrecognized_fx_target')
    return node.target, args, kwargs


def fx_validate_scalar(target, args):
    """Reject Python numeric overloads without evaluating symbolic arithmetic."""
    import operator
    import torch
    if target is getattr:
        if len(args) != 2 or args[1] != 'shape' or not is_fake_tensor(args[0]):
            raise ValueError('unrecognized_fx_target')
    if target is operator.getitem:
        if type(args[0]) not in (tuple, list, torch.Size) or type(args[1]) is not int:
            raise ValueError('unrecognized_fx_target')
    if target in (operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod):
        if any(type(value) not in (int, torch.SymInt) for value in args):
            raise ValueError('unrecognized_fx_target')


def fx_fake_call(target, args, kwargs):
    """Evaluate only audited pure calls; bypass CPU-wheel Tensor.to CUDA checks."""
    import torch
    aten = torch.ops.aten
    fx_validate_scalar(target, args)
    if target in (aten.to.device, aten.to.dtype, aten.to.other):
        options = {k: v for k, v, _ in bound_schema_arguments(target, args, kwargs)}
        source = args[0]
        other = options.get('other')
        dtype = other.dtype if other is not None else options.get('dtype', source.dtype)
        device = other.device if other is not None else options.get('device', source.device)
        memory_format = options.get('memory_format')
        if device == source.device and dtype == source.dtype and not options.get('copy', False) and memory_format in (None, torch.preserve_format):
            return source
        return aten._to_copy.default(source, device=device, dtype=dtype,
                                     non_blocking=options.get('non_blocking', False),
                                     memory_format=memory_format)
    return target(*args, **kwargs)
