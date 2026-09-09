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


def tensors_alias(left, right) -> bool:
    import torch

    return bool(torch._C._is_alias_of(left, right))


def is_tensor(value) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def is_plain_tensor_or_parameter(value) -> bool:
    import torch

    return type(value) in (torch.Tensor, torch.nn.Parameter)
