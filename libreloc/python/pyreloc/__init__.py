"""pyreloc: Python surface of libreloc (issue #46).

Wheel-less install: point PYTHONPATH at the build tree's python/ directory
(see libreloc/README.md). Buffers cross the C++ boundary as
(pointer, nbytes) integer pairs -- design decision 2; use
pyreloc.torch_interop.as_ptr to map torch tensors / numpy arrays.
"""
from ._pyreloc import (  # noqa: F401
    BindError,
    BoundPlan,
    BufferView,
    Calibration,
    DecodeError,
    DispatchRequest,
    GatherPool,
    PlanHandle,
    PrefoldError,
    PrefoldHandle,
    TransferError,
    TransferRequest,
    TypedBoundPlan,
    TypedPlanHandle,
    bind,
    bind_typed,
    cuda_enabled,
    cuda_pointer_device,
    d2h,
    execute_dispatch,
    execute_transfer,
    h2d,
    load_calibration,
    load_plan,
    load_typed_plan,
    make_transfer,
    predict,
    prefold_s8,
    prepare_dispatch,
    query_capability,
    relocate,
    relocate_inverse,
    select_dispatch,
    typed_prefold_spec,
    validate_transfer_source,
    wire_version,
)
