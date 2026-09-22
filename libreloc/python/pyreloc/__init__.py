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
    GatherPool,
    PlanHandle,
    TransferError,
    TransferRequest,
    bind,
    cuda_enabled,
    cuda_pointer_device,
    d2h,
    execute_transfer,
    h2d,
    load_calibration,
    load_plan,
    make_transfer,
    predict,
    relocate,
    relocate_inverse,
    validate_transfer_source,
)
