"""Persistent compiler cache admission, invalidation and process sharing."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from reloc_torch import CompilerClient


def client(compiler, directory, **options):
    return CompilerClient(compiler.executable, cache_dir=directory, **options)


def test_cache_fresh_client_and_no_exporter_launch(compiler, identity_recipe, tmp_path, monkeypatch):
    first = client(compiler, tmp_path)
    artifact = first.compile(identity_recipe)
    second = client(compiler, tmp_path)
    monkeypatch.setattr(second, "_compile", lambda r: pytest.fail("cache hit launched exporter"))
    assert second.compile(identity_recipe).to_bytes() == artifact.to_bytes()
    assert second.cache_stats()["hits"] == 1
    assert first.cache_stats()["misses"] == 1


@pytest.mark.parametrize("damage", [b"not JSON", b"null", b'[]', b'{"key": 0}', b'{}'])
def test_corrupt_cache_recompiles(compiler, identity_recipe, tmp_path, damage):
    c = client(compiler, tmp_path)
    expected = c.compile(identity_recipe).to_bytes()
    next(tmp_path.glob("*.artifact")).write_bytes(damage)
    assert c.compile(identity_recipe).to_bytes() == expected
    assert c.cache_stats()["invalid"] == 1
    assert c.cache_stats()["misses"] == 2


def test_corrupt_plan_revalidated(compiler, identity_recipe, tmp_path):
    c = client(compiler, tmp_path)
    expected = c.compile(identity_recipe).to_bytes()
    path = next(tmp_path.glob("*.artifact"))
    envelope = json.loads(path.read_bytes())
    artifact = json.loads(envelope["artifact"])
    artifact["plan_base64"] = "AAAA"
    envelope["artifact"] = json.dumps(artifact)
    path.write_text(json.dumps(envelope))
    assert c.compile(identity_recipe).to_bytes() == expected
    assert c.cache_stats()["invalid"] == 1


def test_binary_change_even_with_preserved_mtime(compiler, identity_recipe, tmp_path):
    binary = tmp_path / "exporter"
    shutil.copy2(compiler.executable, binary)
    c = CompilerClient(binary, cache_dir=tmp_path / "cache")
    c.compile(identity_recipe)
    stat = binary.stat()
    # Appending ELF padding preserves behavior but changes compiler content.
    with binary.open("ab") as stream:
        stream.write(b"new build")
    os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    c.compile(identity_recipe)
    assert c.cache_stats()["misses"] == 2


def test_namespace_schema_recipe_and_direction_invalidations(compiler, identity_recipe, split_transpose_recipe, tmp_path):
    from dataclasses import replace
    c = client(compiler, tmp_path)
    c.compile(identity_recipe)
    c.compile(split_transpose_recipe)
    c.compile(replace(identity_recipe, direction="d2h"))
    other = client(compiler, tmp_path, cache_namespace="new-configuration")
    other.compile(identity_recipe)
    other._store.frontend = "new-schema"
    other.compile(identity_recipe)
    assert c.cache_stats()["misses"] == 3
    assert other.cache_stats()["misses"] == 2
    assert other.cache_stats()["entries"] == 5


def test_cache_bounds_and_abandoned_write_cleanup(compiler, identity_recipe, split_transpose_recipe, tmp_path):
    c = client(compiler, tmp_path, cache_max_entries=1)
    c.compile(identity_recipe)
    c.compile(split_transpose_recipe)
    (tmp_path / ".write-abandoned").write_bytes(b"partial")
    stats = c.cache_stats()
    assert stats["entries"] == 1 and stats["evictions"] == 1
    assert not (tmp_path / ".write-abandoned").exists()
    small = client(compiler, tmp_path, cache_max_bytes=1)
    small.compile(identity_recipe)
    assert small.cache_stats()["bytes"] == 0
    assert small.cache_stats()["entries"] == 0


def test_compile_failure_not_cached(compiler, identity_recipe, tmp_path, monkeypatch):
    c = client(compiler, tmp_path)
    def fail(recipe):
        raise RuntimeError("compiler failed")
    monkeypatch.setattr(c, "_compile", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="compiler failed"):
            c.compile(identity_recipe)
    assert c.cache_stats()["entries"] == 0 and c.cache_stats()["misses"] == 2


def test_concurrent_processes_share_one_complete_artifact(compiler, identity_recipe, tmp_path):
    # Import only the compiler frontend: this must not import Torch/init CUDA.
    script = '''
import sys
from reloc_torch import CompilerClient, PreparedTransfers
from reloc_torch.artifact import CompiledRecipe
assert 'torch' not in sys.modules
recipe = CompiledRecipe.load(sys.argv[1]).recipe
c = CompilerClient(sys.argv[2], cache_dir=sys.argv[3])
with PreparedTransfers({'recipe': recipe}, compiler=c) as prepared:
    assert prepared.stats()['resources']['contexts'] == 0
    assert 'torch' not in sys.modules
print(c.cache_stats()['misses'])
'''
    artifact = tmp_path / "input.json"
    compiler.compile(identity_recipe).save(artifact)
    processes = [subprocess.Popen([sys.executable, "-c", script, str(artifact),
                                  str(compiler.executable), str(tmp_path / "cache")],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(3)]
    misses = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stderr
        misses.append(int(stdout))
    assert sorted(misses) == [0, 0, 1]
