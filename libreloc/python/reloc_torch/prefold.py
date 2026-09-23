"""Typed prefold capability gate and the owned prefolder bridge (T4 Task 3).

Quantized weight preparation may reuse libreloc's existing load-time
prefolder only for an explicitly requested typed recipe whose declared
scale/zero-point/channel semantics match the kernel (inverse per-channel
scales, zero point zero, round-to-nearest-even, clamp to int8). The typed
recipe and parameter-binding contract belongs to C3/C4 and typed execution to
R3; until they exist ``typed_prefold_capability()`` reports
``typed_artifacts_unavailable`` and ``prepare_weights`` never quantizes a float
weight. ``prefold_s8_image`` is the Torch-facing bridge the future typed path
will call; it is exercised directly by conformance tests today.
"""

from __future__ import annotations

import pyreloc

from . import compat

CAPABILITY_REASON = "typed_artifacts_unavailable"
OUTPUT_SPECS = ("s8_quant_pack", "s8_gather_quant")


def typed_prefold_capability():
    """Reason the typed prefold path is unavailable, or ``None`` once C3/C4/R3
    supply a verified typed recipe/parameter contract."""
    return CAPABILITY_REASON


def prefold_eligibility(recipe, parameters=None):
    """Reason a recipe cannot take the prefold path (never ``None`` before C4).

    A layout-only recipe is not a quantization request (``not_typed_recipe``);
    ordinary float weight loading must keep its dtype and values.
    """
    if getattr(recipe, "value_transforms", None) in (None, ()):
        return "not_typed_recipe"
    return typed_prefold_capability()


def prefold_s8_image(bound, source, inv_scales, *, output_spec, shape=None, gather_threads=1):
    """Fold a float32 CPU tensor into an owned int8 host tensor through ``pyreloc.prefold_s8``.

    ``bound`` is the bound plan whose outer (channel) extent matches
    ``inv_scales``. The result has dtype ``int8`` because quantization was
    explicitly requested and takes ``shape`` (the logical destination shape)
    when given; otherwise the bound plan's coalesced extents, which may merge
    logical axes. The byte count must match the artifact either way.
    """
    import torch
    from pyreloc.torch_interop import as_ptr

    compat.check_version()
    if output_spec not in OUTPUT_SPECS:
        raise ValueError(f"output_spec must be one of {OUTPUT_SPECS}")
    if not compat.is_plain_tensor_or_parameter(source) or source.dtype != torch.float32 or source.device.type != "cpu":
        raise TypeError("prefold_s8_image needs a plain float32 CPU tensor")
    if inv_scales.dtype != torch.float32 or inv_scales.device.type != "cpu":
        raise TypeError("inverse scales must be a float32 CPU tensor")
    contiguous = source.detach().contiguous()
    scales = inv_scales.detach().contiguous()
    with pyreloc.prefold_s8(
        bound, *as_ptr(contiguous), *as_ptr(scales), output_spec=output_spec,
        gather_threads=gather_threads,
    ) as handle:
        result_shape = list(shape) if shape is not None else list(bound.extents)
        image = torch.empty(result_shape, dtype=torch.int8)
        if image.numel() != handle.nbytes:
            raise RuntimeError(
                f"prefold image shape {tuple(result_shape)} holds {image.numel()} bytes "
                f"but the artifact has {handle.nbytes}"
            )
        handle.copy_to(*as_ptr(image))
    return image


__all__ = ("CAPABILITY_REASON", "OUTPUT_SPECS", "prefold_eligibility", "prefold_s8_image", "typed_prefold_capability")
