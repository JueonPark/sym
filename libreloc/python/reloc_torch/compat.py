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


def graph_alias_semantics(node, source, destination):
    # Real-mode make_fx snapshots can have independently allocated fake metadata;
    # schema view guarantees remain authoritative when metadata loses storage IDs.
    target = graph_target(node.target)
    if target in {"aten.view.default", "aten.transpose.int", "aten.permute.default", "aten.as_strided.default"}:
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
