"""Callable ``torch.compile`` backend: guarded replacement of accepted FX regions.

The backend never modifies the captured ``GraphModule``. For each T2 candidate
it compiles (or reuses) an artifact, registers an execution entry, inserts the
functional custom op with explicit symbolic output metadata, redirects the
region's tail uses, and erases only the verified member nodes. Rejected regions
keep their original nodes. The returned callable owns the rewritten graph and
its handle registrations; a backend ``close()`` releases them explicitly.
"""

from __future__ import annotations

import copy
import operator
import threading
import weakref

from . import compat
from .artifact import UnsupportedRecipe
from .cache import DEFAULT_CAPACITY, REGISTRY, ArtifactCache, artifact_key
from .diagnostics import Diagnostics
from .recipe import Recipe, TensorSpec
from .runtime import ExecutionEntry, TransportAdapter
from .symbolic import Add, Const, FloorDiv, Mod, Mul, Symbol, dense_strides, expression


def _tensor_arguments(values):
    import torch
    from torch.utils._pytree import tree_flatten

    return [value for value in tree_flatten(values)[0] if isinstance(value, torch.Tensor)]


def _needs_autograd(values):
    import torch

    return torch.is_grad_enabled() and any(t.requires_grad for t in _tensor_arguments(values))


def identity_recipe(rank, dtype, direction):
    """Symbolic identity recipe shared by all eager transfers of one rank/dtype/direction."""
    if rank < 1:
        raise UnsupportedRecipe("unsupported_rank", "identity recipe requires rank >= 1")
    shape = tuple(Symbol(f"s{axis}") for axis in range(rank))
    descriptor = TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    return Recipe(descriptor, (), descriptor, direction)


class GraphCallable:
    """Executes the rewritten graph, or the untouched original when autograd is live."""

    def __init__(self, original, rewritten, registrations):
        self.original = original
        self.rewritten = rewritten
        self._registrations = list(registrations)

    @property
    def handles(self):
        return tuple(registration.handle for registration in self._registrations)

    def __call__(self, *args, **kwargs):
        if (
            self.rewritten is None
            or self.rewritten is self.original
            or _needs_autograd((args, kwargs))
        ):
            return self.original(*args, **kwargs)
        return self.rewritten(*args, **kwargs)

    def release(self):
        registrations, self._registrations = self._registrations, []
        for registration in registrations:
            registration.release()


class RelocBackend:
    """``RelocBackend(compiler=..., runtime=...)`` is a Dynamo backend callable.

    Default construction resolves the R1 exporter from ``SYM_RELOC_EXPORT`` /
    ``SYM_OPT`` and the R2 transport bridge lazily. ``stats()`` returns a plain
    snapshot; ``close()`` invalidates every live handle and later use.
    """

    def __init__(
        self,
        *,
        compiler=None,
        runtime=None,
        cache_capacity=DEFAULT_CAPACITY,
        importer=None,
        registry=REGISTRY,
    ):
        self._compiler = compiler
        self._runtime = runtime
        self._importer = importer
        self._registry = registry
        self.diagnostics = Diagnostics()
        self._cache = ArtifactCache(cache_capacity)
        self._live = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        self._replaced = 0
        self._closed = False

    @property
    def compiler(self):
        if self._compiler is None:
            from .compiler import CompilerClient

            self._compiler = CompilerClient.from_environment()
        return self._compiler

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = TransportAdapter()
        return self._runtime

    @property
    def importer(self):
        if self._importer is None:
            from .fx_import import import_graph

            self._importer = import_graph
        return self._importer

    @property
    def closed(self):
        return self._closed

    def _require_open(self):
        if self._closed:
            raise RuntimeError("RelocBackend is closed")

    def stats(self):
        result = self.diagnostics.snapshot()
        with self._lock:
            result["replaced_regions"] = self._replaced
            result["live_handles"] = len(self._live)
        result["cache_entries"] = len(self._cache)
        result["cache_rejections"] = self._cache.rejections
        result["closed"] = self._closed
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            live = list(self._live.items())
            self._live.clear()
        for registration, entry in live:
            entry.close()
            registration.release()

    def compile_recipe(self, recipe):
        """Compile through the bounded cache; rejections raise ``UnsupportedRecipe``."""
        key = artifact_key(
            recipe,
            compiler_identity=self.compiler.identity,
            runtime_capability=self.runtime.capability_identity,
        )
        return self._cache.get_or_compile(key, lambda: self.compiler.compile(recipe), self.diagnostics)

    def eager_entry(self, src, device, original, direction=None):
        """Identity entry for an eligible eager transfer of the current source descriptor."""
        self._require_open()
        if direction is None:
            direction = "h2d" if src.device.type == "cpu" else "d2h"
        dtype = compat.dtype_name(src.dtype)
        compiled = self.compile_recipe(identity_recipe(src.dim(), dtype, direction))
        entry = ExecutionEntry(
            compiled=compiled,
            original=original,
            runtime=self.runtime,
            diagnostics=self.diagnostics,
        )
        # Eager entries are never registered; give diagnostics a stable label.
        entry.handle = f"eager/{direction}/{dtype}/rank{src.dim()}"
        return entry

    def __call__(self, gm, example_inputs):
        self._require_open()
        compat.check_version()
        self.diagnostics.increment("dynamo_compiles")
        if _needs_autograd(tuple(example_inputs or ())):
            self.diagnostics.record_exclusion("requires_grad")
            return GraphCallable(gm, None, ())
        report = self.importer(gm, example_inputs)
        for exclusion in report.exclusions:
            self.diagnostics.record_exclusion(exclusion.reason)
        rewritten, registrations = self._rewrite(gm, report.candidates)
        return GraphCallable(gm, rewritten, registrations)

    def _rewrite(self, gm, candidates):
        from torch.fx import GraphModule

        if not candidates:
            return gm, ()
        originals = {node.name: node for node in gm.graph.nodes}
        graph = None
        nodes = None
        registrations = []
        replaced = 0
        for candidate in candidates:
            # Cheap structural checks first, on the untouched graph; compile
            # and copy only for a region that can actually be replaced.
            reason = _region_reason(originals, candidate)
            if reason is not None:
                self.diagnostics.record_exclusion(reason)
                continue
            try:
                compiled = self.compile_recipe(candidate.recipe)
            except UnsupportedRecipe as error:
                self.diagnostics.record_exclusion(error.reason)
                continue
            if graph is None:
                graph = copy.deepcopy(gm.graph)
                nodes = {node.name: node for node in graph.nodes}
            reason = _region_reason(nodes, candidate)
            if reason is not None:
                self.diagnostics.record_exclusion(reason)
                continue
            root = nodes[candidate.source]
            members = [nodes[name] for name in candidate.members]
            tail = members[-1]
            device = candidate.device
            if device is None:
                value = compat.graph_value(tail)
                if not compat.is_tensor(value):
                    self.diagnostics.record_exclusion("metadata_unavailable")
                    continue
                device = value.device
            entry = ExecutionEntry(
                compiled=compiled,
                original=candidate.original,
                runtime=self.runtime,
                diagnostics=self.diagnostics,
                symbolic_bindings=candidate.symbolic_bindings,
                extent_guards=candidate.extent_guards,
            )
            registration = self._registry.register(entry)
            with self._lock:
                self._live[registration] = entry
            with graph.inserting_before(tail):
                op_node = _insert_transfer(
                    graph, root, tail, compiled, registration.handle, device,
                    [nodes[name] for name in candidate.parameters],
                )
            tail.replace_all_uses_with(op_node)
            for member in reversed(members):
                if member.users:
                    raise RuntimeError(f"region member {member.name} is still used after redirection")
                graph.erase_node(member)
            registrations.append(registration)
            replaced += 1
        if not replaced:
            return gm, ()
        graph.lint()
        rewritten = GraphModule(gm, graph)
        with self._lock:
            self._replaced += replaced
        return rewritten, registrations


def _region_reason(nodes, candidate):
    """Structural exclusion for a candidate over `nodes`, or None when replaceable."""
    root = nodes.get(candidate.source)
    members = [nodes.get(name) for name in candidate.members]
    if root is None or any(member is None for member in members):
        return "graph_mismatch"
    member_set = set(members)
    if any(user not in member_set for member in members[:-1] for user in member.users):
        return "escaping_intermediate"
    return None


def _insert_transfer(graph, root, tail, compiled, handle, device, parameters=()):
    import torch
    from .ops import OP, TYPED_OP

    symbol_nodes = {}
    for source in compiled.symbol_sources:
        symbol_nodes[source.name] = graph.call_function(torch.ops.aten.sym_size.int, (root, source.axis))

    def binary(function, left, right):
        if isinstance(left, int) and isinstance(right, int):
            return function(left, right)
        return graph.call_function(function, (left, right))

    def emit(value):
        match expression(value):
            case Const(number):
                return number
            case Symbol(name):
                return symbol_nodes[name]
            case Add(lhs, rhs):
                return binary(operator.add, emit(lhs), emit(rhs))
            case Mul(lhs, rhs):
                return binary(operator.mul, emit(lhs), emit(rhs))
            case FloorDiv(lhs, divisor):
                return binary(operator.floordiv, emit(lhs), divisor)
            case Mod(lhs, divisor):
                return binary(operator.mod, emit(lhs), divisor)
        raise RuntimeError("unsupported destination expression")

    symbols = [symbol_nodes[name] for name in compiled.symbols]
    destination = compiled.logical_destination
    out_shape = [emit(dim) for dim in destination.shape]
    out_strides = [emit(dim) for dim in destination.strides]
    if compiled.typed:
        # C4: the typed op carries the destination dtype explicitly and its
        # runtime parameters as graph inputs (recipe declaration order).
        if len(parameters) != len(compiled.parameters):
            raise RuntimeError("typed candidate parameters do not match the compiled declarations")
        node = graph.call_function(
            TYPED_OP,
            (root, list(parameters), handle, symbols, out_shape, out_strides,
             torch.device(device), getattr(torch, destination.dtype)),
        )
    else:
        node = graph.call_function(
            OP, (root, handle, symbols, out_shape, out_strides, torch.device(device))
        )
    node.meta = dict(tail.meta)
    return node


__all__ = ("GraphCallable", "RelocBackend", "identity_recipe")
