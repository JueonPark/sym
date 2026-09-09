"""Scoped eager Torch transfer observation."""

from __future__ import annotations

from contextlib import contextmanager
import weakref

from . import compat
from .records import TensorMetadata, TransferRecord


_HISTORY_LIMIT = 16


def _dimension(value):
    return value if type(value) is int else str(value)


def _metadata(tensor):
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
            capacity = tensor.untyped_storage().nbytes()
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
        is_subclass=not compat.is_plain_tensor_or_parameter(tensor),
        storage_capacity_bytes=capacity,
    )


def _first_tensor(value):
    if compat.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        return _first_tensor(tuple(value.values()))
    return None


class TransferObserver:
    def __init__(self):
        compat.check_version()
        self.records = []
        self._phases = ["input"]
        self._provenance = {}
        observer = self
        base = compat.torch_dispatch_mode_type()

        class Mode(base):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return observer._dispatch(func, args, {} if kwargs is None else kwargs)

        self._mode = Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._mode.__exit__(exc_type, exc_value, traceback)

    @contextmanager
    def phase(self, name):
        self._phases.append(str(name))
        try:
            yield self
        finally:
            self._phases.pop()

    def _history(self, tensor):
        entry = self._provenance.get(id(tensor))
        return entry[1] if entry is not None and entry[0]() is tensor else ()

    def _set_history(self, tensor, history):
        identity = id(tensor)

        def remove(reference, *, identity=identity):
            current = self._provenance.get(identity)
            if current is not None and current[0] is reference:
                self._provenance.pop(identity, None)

        self._provenance[identity] = (
            weakref.ref(tensor, remove),
            tuple(history[-_HISTORY_LIMIT:]),
        )

    def _dispatch(self, func, args, kwargs):
        name = compat.operator_name(func)
        bound = compat.bound_schema_arguments(func, args, kwargs)
        mutable_tensors = [
            value for _, value, write in bound if write and compat.is_tensor(value)
        ]
        input_tensors = [
            value for _, value, write in bound if not write and compat.is_tensor(value)
        ]
        all_tensors = [value for _, value, _ in bound if compat.is_tensor(value)]
        source = (input_tensors or all_tensors or [_first_tensor(kwargs)])[0]
        mutates = bool(mutable_tensors)
        options = dict((key, value) for key, value, _ in bound)
        source_metadata = _metadata(source) if source is not None else None
        history = self._history(source) if source is not None else ()
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            if source_metadata is not None:
                self.records.append(
                    TransferRecord(
                        name,
                        self._phases[-1],
                        source_metadata,
                        source_metadata,
                        bool(options.get("non_blocking", False)),
                        mutates,
                        True,
                        history,
                        type(error).__name__,
                        str(error),
                    )
                )
            raise
        destination = mutable_tensors[0] if mutable_tensors else _first_tensor(result)
        if source is None or destination is None:
            return result
        destination_metadata = _metadata(destination)
        aliases = compat.tensors_alias(source, destination)
        if mutates:
            new_history = ()
            self._provenance.clear()
        elif compat.is_layout_operator(func):
            new_history = history + (name,)
        elif name == "aten._to_copy.default":
            new_history = history
        else:
            new_history = ()
        self._set_history(destination, new_history)
        self.records.append(
            TransferRecord(
                name,
                self._phases[-1],
                source_metadata,
                destination_metadata,
                bool(options.get("non_blocking", False)),
                mutates,
                aliases,
                tuple(new_history[-_HISTORY_LIMIT:]),
            )
        )
        return result


def observe_transfers() -> TransferObserver:
    return TransferObserver()
