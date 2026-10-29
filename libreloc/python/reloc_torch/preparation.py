"""Explicit recipe compilation and bounded, caller-requested transfer warming.

Warming executes exactly one completed transfer of the supplied example. It is
not a timing probe or a placement search. Examples and warm outputs are never
retained; each serving call validates live values and allocates its own output.
"""

import os
from copy import deepcopy
from threading import RLock
from time import perf_counter

from .compiler import CompilerClient
from .resources import TransferResources, _transfer_configuration


class PreparedTransfers:
    """Serial preparation/serving session for a finite mapping of named recipes.

    Construction compiles recipes without initializing CUDA. ``warm`` explicitly
    initializes the selected device and executes one representative transfer.
    Resource limits cover native scratch, streams and workers; max_output_bytes
    separately bounds each fresh output, not outputs retained by the caller or
    Torch's caching allocator. Preparation does not reserve resources forever:
    incompatible device/affinity/options or larger shapes can replace contexts.
    Use ``stats`` to observe this, and ``close`` to measure retirement.
    """

    def __init__(self, recipes, *, compiler=None, resource_options=None,
                 transfer_options=None, max_recipes=32, max_output_bytes=256 << 20):
        for name, value in (("max_recipes", max_recipes), ("max_output_bytes", max_output_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        recipes = dict(recipes)
        if not recipes or len(recipes) > max_recipes:
            raise ValueError("expected between one and max_recipes recipes")
        if any(not isinstance(name, str) or not name for name in recipes):
            raise ValueError("recipe names must be nonempty strings")
        self._options = _transfer_configuration(None, transfer_options)
        if self._options.get("gather_pool") is not None:
            raise ValueError("preparation owns its worker budget; external gather pools are unsupported")
        limits = dict(resource_options or {})
        # Unlike general-purpose TransferResources, preparation always has hard
        # live limits, including scratch that cannot be retained between calls.
        limits.setdefault("max_live_staging_bytes", 256 << 20)
        limits.setdefault("max_typed_live_bytes", 64 << 20)
        for name in ("max_live_staging_bytes", "max_typed_live_bytes"):
            if type(limits[name]) is not int or limits[name] <= 0:
                raise ValueError(f"{name} must be a positive integer for preparation")
        self._pid = os.getpid()
        self._lock = RLock()
        self._max_output_bytes = max_output_bytes
        self._closed = False
        self._devices = set()
        self._report = dict(compiler_preparation_ms=0., cuda_initialization_ms=0.,
                            resource_warming_ms=0., warm_calls=0, serving_calls=0,
                            first_execution_ms=None, subsequent_execution_ms=0.,
                            retirement_ms=0.)
        start = perf_counter()
        self._compiler = compiler if compiler is not None else CompilerClient.from_environment()
        self._compiled = {name: self._compiler.compile(recipe) for name, recipe in recipes.items()}
        self._report["compiler_preparation_ms"] = (perf_counter() - start) * 1000
        start = perf_counter()
        self._resources = TransferResources(**limits)
        self._report["resource_owner_creation_ms"] = (perf_counter() - start) * 1000

    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("process_mismatch: preparation belongs to another process")

    def _check_open(self):
        self._check_process()
        if self._closed:
            raise RuntimeError("preparation session is closed")
        self._resources.native  # Reject inherited use before touching CUDA.

    def initialize_cuda(self, device="cuda:0"):
        """Initialize a device/current stream explicitly; return elapsed ms.

        Already-initialized processes measure only this call's work. Process
        import and CUDA driver startup must be measured by the deployment caller.
        """
        self._check_process()
        import torch

        with self._lock:
            self._check_open()
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError("initialize_cuda requires a CUDA device")
            start = perf_counter()
            with torch.cuda.device(device):
                torch.cuda.init()
                torch.cuda.current_stream(device).synchronize()
                index = torch.cuda.current_device()
            elapsed = (perf_counter() - start) * 1000
            self._devices.add(index)
            self._report["cuda_initialization_ms"] += elapsed
            return elapsed

    def _execute(self, name, source, device, *, parameters=None, indices=None,
                 policy="auto", threads=8, implementation=""):
        from .dispatch import prepare_typed_transfer, execute_typed_transfer
        from .index_select import prepare_index_select_transfer
        from .transport import prepare_transfer, execute_transfer

        compiled = self._compiled[name]
        if compiled.indexed:
            if parameters or policy != "auto" or implementation:
                raise ValueError("indexed preparation does not accept typed placement/parameters")
            request = prepare_index_select_transfer(compiled, source, indices, device, threads=threads)
        elif compiled.typed:
            if indices is not None:
                raise ValueError("indices require an indexed recipe")
            request = prepare_typed_transfer(compiled, source, device, parameters=parameters,
                                             policy=policy, threads=threads, implementation=implementation)
        else:
            if indices is not None or parameters or policy != "auto" or implementation:
                raise ValueError("layout preparation does not accept typed/indexed arguments")
            request = prepare_transfer(compiled, source, device)
        descriptor = request.destination
        sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
        output_bytes = sizes[descriptor.dtype]
        for extent in descriptor.shape:
            output_bytes *= extent
        if output_bytes > self._max_output_bytes:
            raise ValueError("output exceeds preparation max_output_bytes")
        if compiled.typed or compiled.indexed:
            result = execute_typed_transfer(request, resources=self._resources, **self._options)
            return result.tensor
        return execute_transfer(request, resources=self._resources, **self._options)

    def warm(self, name, source, device, **arguments):
        """Run one explicit completed example, discard output, return stats.

        The reported warming time includes binding, output allocation, resource
        creation, real CPU/GPU work and completion. It is not allocation-only.
        No candidate is selected by timing this operation.
        """
        self._check_process()
        import torch

        with self._lock:
            self._check_open()
            destination = torch.device(device)
            cuda = destination if destination.type == "cuda" else source.device
            if cuda.index is None or cuda.index not in self._devices:
                self.initialize_cuda(cuda)
            start = perf_counter()
            output = self._execute(name, source, device, **arguments)
            del output
            self._report["resource_warming_ms"] += (perf_counter() - start) * 1000
            self._report["warm_calls"] += 1
            return self.stats()

    def transfer(self, name, source, device, **arguments):
        """Transfer fresh live inputs; return an independent completed tensor."""
        self._check_process()
        with self._lock:
            self._check_open()
            start = perf_counter()
            output = self._execute(name, source, device, **arguments)
            elapsed = (perf_counter() - start) * 1000
            if self._report["serving_calls"] == 0:
                self._report["first_execution_ms"] = elapsed
            else:
                self._report["subsequent_execution_ms"] += elapsed
            self._report["serving_calls"] += 1
            return output

    def stats(self):
        """Bounded scalar diagnostics; does not initialize CUDA."""
        self._check_process()
        with self._lock:
            result = deepcopy(self._report)
            result["compiler_cache"] = self._compiler.cache_stats()
            result["resources"] = self._resources.stats()
            result["recipe_count"] = len(self._compiled)
            result["max_output_bytes"] = self._max_output_bytes
            return result

    def close(self):
        self._check_process()
        with self._lock:
            if self._closed:
                return
            start = perf_counter()
            try:
                self._resources.close()
            finally:
                self._closed = True
                self._report["retirement_ms"] += (perf_counter() - start) * 1000

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *unused):
        self.close()

    def __reduce_ex__(self, protocol):
        raise TypeError("preparation sessions cannot be serialized")
