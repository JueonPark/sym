"""pyreloc: Python surface of libreloc (issue #46).

Wheel-less install: point PYTHONPATH at the build tree's python/ directory
(see libreloc/README.md). Buffers cross the C++ boundary as
(pointer, nbytes) integer pairs -- design decision 2; use
pyreloc.torch_interop.as_ptr to map torch tensors / numpy arrays.
"""
from ._pyreloc import (  # noqa: F401
    BindError,
    BoundPlan,
    DecodeError,
    PlanHandle,
    bind,
    cuda_enabled,
    d2h,
    h2d,
    load_plan,
    relocate,
    relocate_inverse,
)
