"""Scoped eager Torch transfer observation."""

from __future__ import annotations

from contextlib import contextmanager
import weakref

from . import compat
from .records import TransferRecord


_HISTORY_LIMIT = 16


_metadata = compat.tensor_metadata


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


def _tensor_leaves(value):
    if compat.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_leaves(item)]
    if isinstance(value, dict):
        return _tensor_leaves(tuple(value.values()))
    return []


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
            tensor
            for _, value, write in bound
            if write
            for tensor in _tensor_leaves(value)
        ]
        input_tensors = [
            tensor
            for _, value, write in bound
            if not write
            for tensor in _tensor_leaves(value)
        ]
        all_tensors = [
            tensor for _, value, _ in bound for tensor in _tensor_leaves(value)
        ]
        mutates = bool(mutable_tensors)
        options = dict((key, value) for key, value, _ in bound)
        if mutates:
            if compat.is_foreach_copy_operator(func):
                pairs = [
                    (
                        input_tensors[index]
                        if index < len(input_tensors)
                        else destination,
                        destination,
                    )
                    for index, destination in enumerate(mutable_tensors)
                ]
            else:
                fallback = input_tensors[0] if input_tensors else None
                pairs = [
                    (
                        input_tensors[index]
                        if len(input_tensors) == len(mutable_tensors)
                        else (fallback if fallback is not None else destination),
                        destination,
                    )
                    for index, destination in enumerate(mutable_tensors)
                ]
        else:
            pairs = [(source, None) for source in (input_tensors or all_tensors)[:1]]
        before = [(_metadata(source), self._history(source)) for source, _ in pairs]
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            for (source_metadata, history), _ in zip(before, pairs):
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
        if not mutates:
            outputs = _tensor_leaves(result)
            if not outputs:
                return result
            sources = input_tensors or all_tensors
            if not sources:
                return result
            fallback = sources[0] if sources else None
            pairs = [
                (
                    sources[index] if len(sources) == len(outputs) else fallback,
                    destination,
                )
                for index, destination in enumerate(outputs)
                if sources or fallback is not None
            ]
            initial = before[0]
            before = [initial for _ in pairs]
        if not pairs:
            return result
        if mutates:
            self._provenance.clear()
        for (source_metadata, history), (source, destination) in zip(before, pairs):
            destination_metadata = _metadata(destination)
            aliases = compat.tensors_alias(source, destination)
            if mutates:
                new_history = ()
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
