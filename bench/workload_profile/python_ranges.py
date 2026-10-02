"""Profiler-only NVTX wrappers; return values and execution order unchanged."""
from functools import wraps

def install():
    import torch
    import pyreloc
    from reloc_torch import dispatch, transport, runtime, artifact

    def wrap(owner, key, label):
        original = getattr(owner, key)
        @wraps(original)
        def traced(*args, **kwargs):
            with torch.cuda.nvtx.range(label):
                return original(*args, **kwargs)
        setattr(owner, key, traced)

    for key in ('load_typed_plan', 'bind_typed', 'query_capability',
                'select_dispatch', 'prepare_typed_program', 'prepare_dispatch', 'execute_dispatch'):
        wrap(pyreloc, key, 'python.native.'+key)
    for key in ('prepare_typed_transfer', 'execute_typed_transfer',
                '_parameter_snapshot', 'bind_symbols', 'destination_descriptor', '_storage_view'):
        wrap(dispatch, key, 'python.'+key)
    wrap(dispatch.PreparedTypedTransfer, 'recheck', 'python.recheck')
    for key in ('prepare_transfer','execute_transfer','bind_plan','bind_symbols','destination_descriptor'):
        wrap(transport, key, 'python.layout.'+key)
    for key in ('bind_symbols','destination_descriptor','verify_result'):
        wrap(runtime, key, 'python.runtime.'+key)
    for key in ('preflight','execute'):
        wrap(runtime.TransportAdapter, key, 'python.adapter.'+key)
    wrap(artifact.CompiledRecipe, 'bind_values', 'python.artifact.bind_values')
    for key in ('load_plan','bind','make_transfer','execute_transfer','validate_transfer_source'):
        wrap(pyreloc, key, 'python.native.'+key)
