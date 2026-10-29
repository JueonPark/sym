"""Subprocess bridge for the supported R1/C3 relocation-plan exporter."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from .artifact import (
    CompiledRecipe,
    UnsupportedRecipe,
    _admit,
    _read_manifest,
    _validate_unsupported_manifest,
)
from .mlir_emit import emit_mlir


class CompilerClient:
    """Invoke the supported exporter and admit only associated artifacts."""

    def __init__(self, executable, *, cache_dir=None, cache_max_bytes=64 << 20,
                 cache_max_entries=128, cache_namespace="default", canonicalize=True):
        self.executable = Path(executable)
        # canonicalize=False exports the raw fold output (sym-reloc-export
        # --no-canonicalize): a measurement mode for compiler attribution,
        # never the default. The manifest does not record it, so every
        # identity this client derives carries the mode itself.
        self.canonicalize = bool(canonicalize)
        self._identity = None
        self._binary_signature = self._binary_digest = None
        self._store = None
        if cache_dir is not None:
            from .artifact_store import ArtifactStore
            self._store = ArtifactStore(cache_dir, max_bytes=cache_max_bytes,
                                        max_entries=cache_max_entries,
                                        namespace=cache_namespace)

    @classmethod
    def from_environment(cls, environ=None, **options):
        """Resolve explicit overrides first, then this installation's exporter."""
        environ = os.environ if environ is None else environ
        configured = environ.get("SYM_RELOC_EXPORT")
        if configured is None:
            sym_opt = environ.get("SYM_OPT")
            if sym_opt is not None:
                configured = Path(sym_opt).with_name("sym-reloc-export")
            else:
                try:
                    from sym_reloc.tools import native_tool
                except ImportError as error:
                    raise RuntimeError(
                        "SYM_RELOC_EXPORT or SYM_OPT must select the R1 exporter; "
                        "no bundled compiler is installed"
                    ) from error
                configured = native_tool("sym-reloc-export")
        client = cls(configured, **options)
        if not client.executable.is_file() or not os.access(client.executable, os.X_OK):
            raise RuntimeError(f"configured R1 exporter is absent or not executable: {client.executable}")
        return client

    @property
    def identity(self):
        """Pre-compilation compiler identity for cache keys: the resolved
        exporter path plus the binary's size and modification time, so a
        rebuilt exporter never reuses artifacts or rejections produced by the
        previous binary. Never an object id."""
        if self._identity is None:
            self._identity = f"sym-reloc-export@{self.executable.resolve()}"
            if not self.canonicalize:
                self._identity += "+no-canonicalize"
        try:
            status = os.stat(self.executable)
        except OSError:
            return self._identity
        return f"{self._identity}#{status.st_size}-{status.st_mtime_ns}-{status.st_ctime_ns}"

    def cache_stats(self):
        """Disk cache counters, or None when caching is disabled (the default)."""
        return None if self._store is None else self._store.stats()

    def _content_identity(self):
        status = self.executable.stat()
        signature = (str(self.executable.resolve()), status.st_dev, status.st_ino,
                     status.st_size, status.st_mtime_ns, status.st_ctime_ns)
        if signature != self._binary_signature:
            with self.executable.open("rb") as stream:
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
            self._binary_signature, self._binary_digest = signature, digest
        return self._binary_digest if self.canonicalize else self._binary_digest + "+no-canonicalize"

    def compile(self, recipe):
        if self._store is None:
            return self._compile(recipe)
        identity = self._content_identity()

        def compile_stable():
            result = self._compile(recipe)
            if identity != self._content_identity():
                raise RuntimeError("compiler changed during compilation; retry preparation")
            return result

        result = self._store.get_or_compile(recipe, identity, compile_stable)
        if identity != self._content_identity():
            raise RuntimeError("compiler changed during cache lookup; retry preparation")
        return result

    def _compile(self, recipe):
        mlir = emit_mlir(recipe).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="reloc-compile-") as directory:
            root = Path(directory)
            input_path = root / "recipe.mlir"
            plan_path = root / "plan.bin"
            manifest_path = root / "manifest.json"
            input_path.write_bytes(mlir)
            arguments = [
                str(self.executable),
                str(input_path),
                "--output",
                str(plan_path),
                "--manifest",
                str(manifest_path),
            ]
            if recipe.typed:
                # Value transforms are opt-in, including a cast after selection.
                arguments.append("--typed")
            if recipe.indexed:
                arguments.append('--indexed')
            if not self.canonicalize:
                arguments.append('--no-canonicalize')
            try:
                result = subprocess.run(arguments, check=False, capture_output=True)
            except OSError as error:
                raise RuntimeError(f"unable to invoke compiler: {self.executable}") from error
            if result.returncode == 2:
                if plan_path.exists():
                    raise RuntimeError("compiler unsupported response unexpectedly published a plan")
                manifest = _read_manifest(manifest_path)
                _validate_unsupported_manifest(manifest)
                raise UnsupportedRecipe(manifest["reason"], manifest["detail"])
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", "replace").strip()
                message = f"compiler exited with status {result.returncode}"
                if stderr:
                    message += f": {stderr}"
                raise RuntimeError(message)
            if not plan_path.is_file() or not manifest_path.is_file():
                raise RuntimeError("compiler succeeded without a complete plan/manifest pair")
            plan = plan_path.read_bytes()
            manifest = _read_manifest(manifest_path)
        return _admit(recipe, mlir, plan, manifest)


__all__ = ("CompilerClient", "CompiledRecipe", "UnsupportedRecipe")
