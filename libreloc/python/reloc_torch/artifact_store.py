"""Opt-in, bounded on-disk recipe artifacts (local filesystem, POSIX locks).

Only validated portable plans are persisted, never tensors, executable requests,
placement decisions or CUDA state. The directory must be trusted, as compiler
artifacts are executable metadata. One advisory lock serializes cache access
across processes; compilation happens inside it to avoid startup stampedes.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .artifact import CompiledRecipe, _encode_recipe


CACHE_VERSION = 1


def frontend_digest():
    # Include admission, schemas, recipe serialization, emission and binding.
    # Hash source bytes rather than timestamps so installed copies share keys.
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, directory, *, max_bytes=64 << 20, max_entries=128,
                 namespace="default"):
        for name, value in (("max_bytes", max_bytes), ("max_entries", max_entries)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(namespace, str):
            raise TypeError("cache namespace must be a string")
        self.directory = Path(directory)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.namespace = namespace
        self.frontend = frontend_digest()
        self.hits = self.misses = self.invalid = self.evictions = 0

    @contextmanager
    def _locked(self):
        import fcntl

        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                for abandoned in self.directory.glob(".write-*"):
                    abandoned.unlink()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _files(self):
        return sorted(self.directory.glob("*.artifact"), key=lambda p: p.stat().st_mtime_ns)

    def _trim(self, incoming=0, slots=0):
        files = self._files()
        sizes = {p: p.stat().st_size for p in files}
        total = sum(sizes.values())
        count = len(files)
        for path in files:
            if total + incoming <= self.max_bytes and count + slots <= self.max_entries:
                break
            path.unlink()
            self.evictions += 1
            total -= sizes[path]
            count -= 1

    def get_or_compile(self, recipe, compiler_digest, compile):
        identity = {
            "version": CACHE_VERSION, "frontend": self.frontend,
            "compiler": compiler_digest, "namespace": self.namespace,
            "recipe": _encode_recipe(recipe),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + ".artifact")
        with self._locked():
            self._trim()
            if path.exists():
                try:
                    # Bound reads even for a corrupted or externally replaced file.
                    with path.open("rb") as stream:
                        data = stream.read(self.max_bytes + 1)
                    if len(data) > self.max_bytes:
                        raise ValueError("oversize artifact")
                    envelope = json.loads(data)
                    if (not isinstance(envelope, dict) or envelope.get("key") != key
                            or not isinstance(envelope.get("artifact"), str)):
                        raise ValueError("cache identity mismatch")
                    compiled = CompiledRecipe.from_bytes(envelope["artifact"].encode())
                    if compiled.recipe != recipe:
                        raise ValueError("recipe mismatch")
                except (RuntimeError, ValueError, TypeError, KeyError, UnicodeError):
                    self.invalid += 1
                    path.unlink()
                else:
                    self.hits += 1
                    os.utime(path, None)
                    return compiled
            self.misses += 1
            compiled = compile()  # Exceptions/rejections are never persisted.
            data = json.dumps({"key": key, "artifact": compiled.to_bytes().decode()},
                              separators=(",", ":")).encode()
            if len(data) <= self.max_bytes:
                self._trim(len(data), 1)
                # Atomic replacement: a killed writer cannot publish a partial plan.
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".write-",
                                                     delete=False) as stream:
                        temporary = stream.name
                        stream.write(data)
                    os.replace(temporary, path)
                finally:
                    if temporary is not None and os.path.exists(temporary):
                        os.unlink(temporary)
            return compiled

    def stats(self):
        with self._locked():
            self._trim()
            files = self._files()
            return dict(hits=self.hits, misses=self.misses, invalid=self.invalid,
                        evictions=self.evictions, entries=len(files),
                        bytes=sum(p.stat().st_size for p in files),
                        max_entries=self.max_entries, max_bytes=self.max_bytes)
