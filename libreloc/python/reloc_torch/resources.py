"""Transfer resource ownership and configuration; Torch-free import."""

from collections.abc import Mapping
from enum import Enum
import operator
import os


class _ResourcePolicy(Enum):
    AUTO = "auto"


AUTO = _ResourcePolicy.AUTO


def _transfer_configuration(resources, options):
    """Validate before execution, copying knobs while borrowing object owners."""
    if resources is not None and resources is not AUTO and not isinstance(resources, TransferResources):
        raise TypeError("transfer_resources must be AUTO, TransferResources or None")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("transfer_options must be a mapping or None")
    options = dict(options) if options is not None else {}
    for name, value in options.items():
        if name == "gather_pool":
            if value is not None:
                import pyreloc

                if not isinstance(value, pyreloc.GatherPool):
                    raise TypeError("gather_pool must be GatherPool or None")
        elif name in {"n_buffers", "n_streams", "gather_threads"}:
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            try:
                value = operator.index(value)
            except TypeError:
                raise TypeError(f"{name} must be an integer") from None
            minimum = 0 if name == "gather_threads" else 1
            if not minimum <= value <= (1 << 31) - 1:
                raise ValueError(f"{name} must be between {minimum} and 2147483647")
            options[name] = value
        else:
            raise ValueError(f"unknown transfer option: {name}")
    return options


class TransferResources:
    """Keep bounded staging, streams and workers across blocking transfers.

    Layout limits and typed limits are separate: max_retained_bytes bounds
    layout staging; max_typed_retained_bytes bounds combined typed host/device
    scratch (default 64 MiB). Typed calls serialize within one owner. Optional
    max_typed_live_bytes also caps scratch during execution (0 = unlimited).
    Construction and stats do not initialize CUDA.
    Use a context manager or close() before shutdown. clear() retires idle
    resources and older active generations. Unknown completion retains resources
    and tensor owners for the process lifetime; close() reports that failure.
    """

    __slots__ = ("_native", "_typed", "_pid", "__weakref__")

    def __init__(self, *, max_retained_bytes=256 << 20, max_contexts=4,
                 max_contexts_per_device=2, max_background_workers=64,
                 max_live_staging_bytes=None, acquire_timeout_ms=None,
                 max_typed_retained_bytes=64 << 20, max_typed_live_bytes=0,
                 max_typed_streams=8, max_typed_background_workers=64):
        import pyreloc

        self._pid = os.getpid()
        self._native = pyreloc.TransferResourceCache(
            max_retained_bytes=max_retained_bytes, max_contexts=max_contexts,
            max_contexts_per_device=max_contexts_per_device,
            max_background_workers=max_background_workers,
            max_live_staging_bytes=max_live_staging_bytes,
            acquire_timeout_ms=acquire_timeout_ms,
        )

        self._typed = pyreloc.DispatchResources(
            max_retained_bytes=max_typed_retained_bytes,
            max_live_bytes=max_typed_live_bytes,
            max_background_workers=max_typed_background_workers,
            max_streams=max_typed_streams,
        )

    @property
    def native_typed(self):
        self.native  # PID check before taking any native lock
        return self._typed

    @property
    def native(self):
        # Reject inherited use before a frontend can allocate output or touch
        # CUDA/pool state. Native PID/epoch checks independently protect bindings.
        if os.getpid() != self._pid:
            raise RuntimeError("process_mismatch: transfer resources belong to another process")
        return self._native

    @property
    def closed(self):
        return self.native.closed

    def stats(self):
        """Return scalar counters/gauges and per-device snapshots."""
        result = self.native.stats()
        result["typed"] = self.native_typed.stats()
        return result

    def clear(self):
        try:
            self.native.clear()
        finally:
            self.native_typed.clear()

    def close(self):
        try:
            self.native.close()
        finally:
            self.native_typed.close()

    def __enter__(self):
        self.native.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __reduce_ex__(self, protocol):
        raise TypeError("transfer resources cannot be serialized")
