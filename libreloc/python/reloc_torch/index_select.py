"""Dense CPU row selection fused with optional cast and blocking H2D.

The compiler's wire-v2 plan describes the physical source, runtime index
operand, selection and cast. Native binding checks that plan before dispatch.
"""

from . import compat
from .artifact import UnsupportedRecipe
from .symbolic import GuardError


def bind_index_select(compiled, source, index):
    import torch
    from .runtime import source_reason

    reason = source_reason(source)
    if reason is not None:
        raise UnsupportedRecipe(reason, f"index_select source rejected: {reason}")
    if (source.device.type != 'cpu' or source.dim() < 1 or not source.is_contiguous()
            or source.is_neg() or source.is_conj()):
        raise UnsupportedRecipe('unsupported_index_select', 'expected a dense CPU source')
    if (not compat.is_plain_tensor_or_parameter(index) or index.device.type != 'cpu'
            or index.layout != torch.strided or index.dim() != 1
            or index.dtype not in (torch.int32, torch.int64)):
        raise UnsupportedRecipe('unsupported_index_select', 'expected a CPU int32/int64 index vector')
    try:
        return compiled.bind_values(source, index)
    except GuardError as error:
        raise UnsupportedRecipe(error.reason, str(error)) from error


def index_snapshot(index):
    return compat.storage_snapshot(index), index.detach().resolve_neg().contiguous().numpy().tobytes()


def prepare_index_select_transfer(compiled, source, index, device, *, threads=8):
    import torch
    import pyreloc
    from .dispatch import PreparedTypedTransfer
    from .runtime import destination_descriptor
    from .transport import _code, _storage_view

    bindings = bind_index_select(compiled, source, index)
    recipe = compiled.recipe
    if not compiled.indexed or recipe.direction != 'h2d':
        raise UnsupportedRecipe('unsupported_index_select', 'expected row selection and optional cast')
    device = torch.device(device)
    if device.type != 'cuda':
        raise UnsupportedRecipe('direction_mismatch', 'index_select requires H2D')
    snapshot = index_snapshot(index)
    view = _storage_view(source, 'host', -1)
    try:
        bound = pyreloc.bind_indexed(compiled.decoded_plan, bindings,
            (compat.dtype_name(index.dtype), list(index.shape), snapshot[1]))
        program = pyreloc.prepare_index_select_program(bound, view)
        capability = pyreloc.query_capability(program, 'h2d', 'cuda')
        selected = pyreloc.select_dispatch(program, 'h2d', 'cuda', threads=threads)
    except pyreloc.BindError as error:
        raise UnsupportedRecipe('index_out_of_range' if 'index_out_of_range' in str(error) else 'bind_error', str(error)) from error
    except pyreloc.TransferError as error:
        raise UnsupportedRecipe(_code(error), str(error)) from error
    if not pyreloc.cuda_enabled or not torch.cuda.is_available():
        raise UnsupportedRecipe('cuda_unavailable', 'no CUDA-capable runtime')
    target = torch.device('cuda', device.index if device.index is not None else torch.cuda.current_device())
    destination = destination_descriptor(compiled, bindings, target)
    return PreparedTypedTransfer(
        compiled, source, bindings, bound, destination, 'h2d', target, view,
        'auto', None, threads, {}, {}, capability, selected, program=program,
        index=index, index_snapshot=snapshot,
    )
