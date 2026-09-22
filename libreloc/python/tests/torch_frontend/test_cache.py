"""Artifact cache keys, eviction, ownership, registry lifetime and counters."""
import dataclasses
import gc
import threading
import time

import pytest
import torch

from conftest import CountingRuntime, make_entry


def api():
    from reloc_torch import cache

    return cache


def _key(recipe, **overrides):
    options = dict(compiler_identity="exporter@/a", runtime_capability="cpu-test-adapter/1")
    options.update(overrides)
    return api().artifact_key(recipe, **options)


class RecordingCompiler:
    def __init__(self, compiler, delay=0.0):
        self.compiler = compiler
        self.delay = delay
        self.calls = 0
        self.identity = compiler.identity

    def compile(self, recipe):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.compiler.compile(recipe)


def test_keys_exclude_dynamic_values_and_separate_semantic_identity(identity_recipe, split_transpose_recipe):
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec
    from reloc_torch.symbolic import Const, FloorDiv, Symbol

    assert _key(identity_recipe) == _key(identity_recipe)
    assert _key(identity_recipe) != _key(split_transpose_recipe)
    descriptor = TensorSpec((Symbol("s0"),), (Const(1),), Const(0), "float16")
    assert _key(identity_recipe) != _key(Recipe(descriptor, (), descriptor, "h2d"))
    assert _key(identity_recipe) != _key(dataclasses.replace(identity_recipe, direction="d2h"))
    assert _key(identity_recipe) != _key(identity_recipe, compiler_identity="exporter@/b")
    assert _key(identity_recipe) != _key(identity_recipe, runtime_capability="other/2")
    assert _key(identity_recipe) != _key(identity_recipe, wire_version=1)
    assert _key(identity_recipe) != _key(identity_recipe, frontend_identity="reloc_torch/other")
    n = Symbol("s0")
    split_32 = dataclasses.replace(
        split_transpose_recipe,
        operations=(Reshape((FloorDiv(n, 32), Const(32))), split_transpose_recipe.operations[1]),
        destination=TensorSpec((Const(32), FloorDiv(n, 32)), (FloorDiv(n, 32), Const(1)), Const(0), "float32"),
    )
    assert _key(split_transpose_recipe) != _key(split_32)
    key = _key(identity_recipe)
    assert "0x" not in repr(key)
    hash(key)


def test_cache_hits_evicts_and_counts(compiler, identity_recipe, split_transpose_recipe):
    from reloc_torch.diagnostics import Diagnostics

    recording = RecordingCompiler(compiler)
    cache = api().ArtifactCache(capacity=1)
    diagnostics = Diagnostics()
    first = cache.get_or_compile(_key(identity_recipe), lambda: recording.compile(identity_recipe), diagnostics)
    again = cache.get_or_compile(_key(identity_recipe), lambda: recording.compile(identity_recipe), diagnostics)
    assert again is first
    assert recording.calls == 1
    cache.get_or_compile(_key(split_transpose_recipe), lambda: recording.compile(split_transpose_recipe), diagnostics)
    assert recording.calls == 2
    assert len(cache) == 1
    cache.get_or_compile(_key(identity_recipe), lambda: recording.compile(identity_recipe), diagnostics)
    assert recording.calls == 3
    snapshot = diagnostics.snapshot()
    assert snapshot["plan_compiles"] == 3
    assert snapshot["cache_hits"] == 1
    with pytest.raises(ValueError):
        api().ArtifactCache(capacity=0)


def test_evicted_artifact_stays_usable_by_a_live_entry(compiler, identity_recipe, split_transpose_recipe, counting_runtime):
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.runtime import execute_or_fallback

    cache = api().ArtifactCache(capacity=1)
    compiled = cache.get_or_compile(_key(identity_recipe), lambda: compiler.compile(identity_recipe), Diagnostics())
    entry = make_entry(compiled, counting_runtime, lambda src, *s: src.clone())
    cache.get_or_compile(_key(split_transpose_recipe), lambda: compiler.compile(split_transpose_recipe), Diagnostics())
    assert len(cache) == 1
    x = torch.arange(6, dtype=torch.float32)
    assert torch.equal(execute_or_fallback(entry, x, None, torch.device("cpu")), x)
    assert counting_runtime.executions == 1


def test_concurrent_same_key_compilation_happens_once(compiler, identity_recipe):
    from reloc_torch.diagnostics import Diagnostics

    recording = RecordingCompiler(compiler, delay=0.2)
    cache = api().ArtifactCache(capacity=4)
    diagnostics = Diagnostics()
    results = []
    key = _key(identity_recipe)

    def worker():
        results.append(cache.get_or_compile(key, lambda: recording.compile(identity_recipe), diagnostics))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recording.calls == 1
    assert all(result is results[0] for result in results)
    snapshot = diagnostics.snapshot()
    assert snapshot["plan_compiles"] == 1
    assert snapshot["cache_hits"] == 7


def test_unsupported_recipes_are_cached_per_capability(compiler):
    from reloc_torch import UnsupportedRecipe
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const

    source = TensorSpec((Const(2), Const(3)), (Const(3), Const(1)), Const(0), "float32")
    destination = TensorSpec((Const(6),), (Const(1),), Const(0), "float32")
    bail = Recipe(source, (Transpose((1, 0)), Reshape((Const(6),))), destination, "h2d")
    recording = RecordingCompiler(compiler)
    cache = api().ArtifactCache(capacity=4)
    diagnostics = Diagnostics()
    for _ in range(2):
        with pytest.raises(UnsupportedRecipe) as failure:
            cache.get_or_compile(_key(bail), lambda: recording.compile(bail), diagnostics)
        assert failure.value.reason == "fold_unsupported"
    assert recording.calls == 1
    with pytest.raises(UnsupportedRecipe):
        cache.get_or_compile(_key(bail, runtime_capability="new-runtime/2"), lambda: recording.compile(bail), diagnostics)
    assert recording.calls == 2
    with pytest.raises(UnsupportedRecipe):
        cache.get_or_compile(_key(bail, compiler_identity="exporter@/new"), lambda: recording.compile(bail), diagnostics)
    assert recording.calls == 3
    assert diagnostics.snapshot()["plan_compiles"] == 3


def test_unsupported_cache_is_bounded(compiler):
    from reloc_torch import UnsupportedRecipe
    from reloc_torch.diagnostics import Diagnostics
    from reloc_torch.recipe import Recipe, Reshape, TensorSpec, Transpose
    from reloc_torch.symbolic import Const

    def bail(rows):
        source = TensorSpec((Const(rows), Const(3)), (Const(3), Const(1)), Const(0), "float32")
        destination = TensorSpec((Const(rows * 3),), (Const(1),), Const(0), "float32")
        return Recipe(source, (Transpose((1, 0)), Reshape((Const(rows * 3),))), destination, "h2d")

    recording = RecordingCompiler(compiler)
    cache = api().ArtifactCache(capacity=1)
    for rows in (2, 4, 2):
        with pytest.raises(UnsupportedRecipe):
            cache.get_or_compile(_key(bail(rows)), lambda: recording.compile(bail(rows)), Diagnostics())
    assert recording.calls == 3


def test_registry_handles_are_process_local_owned_and_released(compiler, identity_recipe, counting_runtime):
    cache = api()
    registry = cache.HandleRegistry()
    entry = make_entry(compiler.compile(identity_recipe), counting_runtime, lambda src, *s: src.clone())
    registration = registry.register(entry)
    handle = registration.handle
    assert isinstance(handle, str) and handle
    assert registry.lookup(handle) is entry
    assert entry.handle == handle
    other = registry.register(entry)
    assert other.handle != handle
    registration.release()
    registration.release()
    with pytest.raises(RuntimeError, match="unknown or closed"):
        registry.lookup(handle)
    assert registry.lookup(other.handle) is entry
    del other
    gc.collect()
    with pytest.raises(RuntimeError, match="unknown or closed"):
        registry.lookup(handle)
    assert len(registry) == 0
    with pytest.raises(RuntimeError, match="unknown or closed"):
        registry.lookup("reloc-not-a-handle")


def test_registry_is_thread_safe(compiler, identity_recipe, counting_runtime):
    registry = api().HandleRegistry()
    entry = make_entry(compiler.compile(identity_recipe), counting_runtime, lambda src, *s: src.clone())
    registrations = []
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            registration = registry.register(entry)
            assert registry.lookup(registration.handle) is entry
            with lock:
                registrations.append(registration)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({r.handle for r in registrations}) == 400
    assert len(registry) == 400
    for registration in registrations:
        registration.release()
    assert len(registry) == 0


def test_diagnostics_snapshot_is_a_plain_copy_without_tensors():
    from reloc_torch.diagnostics import Diagnostics

    diagnostics = Diagnostics()
    diagnostics.increment("dynamo_compiles")
    diagnostics.increment("runtime_executions", 2)
    diagnostics.record_fallback("unsupported_layout")
    diagnostics.record_exclusion("mutation")
    snapshot = diagnostics.snapshot()
    assert snapshot["dynamo_compiles"] == 1
    assert snapshot["runtime_executions"] == 2
    assert snapshot["fallbacks"] == {"unsupported_layout": 1}
    assert snapshot["exclusions"] == {"mutation": 1}
    assert set(snapshot) >= {"dynamo_compiles", "plan_compiles", "symbol_binds", "cache_hits", "runtime_executions", "fallbacks"}
    snapshot["fallbacks"]["unsupported_layout"] = 99
    assert diagnostics.fallbacks["unsupported_layout"] == 1

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(v) for v in value.values())
        return False

    assert not contains_tensor(snapshot)
    with pytest.raises(KeyError):
        diagnostics.increment("not_a_counter")
