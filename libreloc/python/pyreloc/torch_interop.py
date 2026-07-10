"""Map torch tensors / numpy arrays to the (ptr, nbytes) integer pairs the
pyreloc C++ API takes (issue #40 design decision 2).

Deliberately duck-typed: torch is never imported here, so pyreloc works in
torch-less environments and libtorch is never a dependency. CUDA tensors
work the same way -- ``Tensor.data_ptr()`` returns a device pointer, which
is exactly what h2d/d2h expect for their device-side argument.
"""


def as_ptr(obj):
    """Return ``(ptr, nbytes)`` for a supported buffer object.

    Accepts: a torch tensor (CPU / pinned / CUDA; must be contiguous), a
    numpy ndarray (must be C-contiguous), or an already-made
    ``(ptr, nbytes)`` tuple of two ints (passed through).
    """
    if isinstance(obj, tuple):
        if len(obj) != 2 or not all(isinstance(v, int) for v in obj):
            raise ValueError(
                f"expected a (ptr, nbytes) tuple of two ints, got {obj!r}")
        return obj

    data_ptr = getattr(obj, "data_ptr", None)
    if callable(data_ptr):  # torch.Tensor (duck-typed)
        if not obj.is_contiguous():
            raise ValueError("tensor must be contiguous for as_ptr()")
        return data_ptr(), obj.numel() * obj.element_size()

    if hasattr(obj, "__array_interface__"):  # numpy.ndarray
        if not obj.flags["C_CONTIGUOUS"]:
            raise ValueError("ndarray must be C-contiguous for as_ptr()")
        return obj.ctypes.data, obj.nbytes

    raise TypeError(
        f"as_ptr() supports torch tensors, numpy arrays, and (ptr, nbytes) "
        f"tuples; got {type(obj).__name__}")
