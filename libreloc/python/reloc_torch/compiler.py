"""Subprocess bridge for the supported R1 relocation-plan exporter."""
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
    """Invoke the supported R1 exporter and admit only associated artifacts."""

    def __init__(self, executable):
        self.executable = Path(executable)
        self._identity = None

    @classmethod
    def from_environment(cls, environ=None):
        """Resolve the documented exporter configuration: ``SYM_RELOC_EXPORT``,
        else the ``sym-reloc-export`` sibling of ``SYM_OPT``."""
        environ = os.environ if environ is None else environ
        configured = environ.get("SYM_RELOC_EXPORT")
        if configured is None:
            sym_opt = environ.get("SYM_OPT")
            if sym_opt is None:
                raise RuntimeError(
                    "SYM_RELOC_EXPORT or SYM_OPT must select the R1 exporter"
                )
            configured = Path(sym_opt).with_name("sym-reloc-export")
        client = cls(configured)
        if not client.executable.is_file():
            raise RuntimeError(f"configured R1 exporter is absent: {client.executable}")
        return client

    @property
    def identity(self):
        """Pre-compilation compiler identity for cache keys: the resolved
        exporter path plus the binary's size and modification time, so a
        rebuilt exporter never reuses artifacts or rejections produced by the
        previous binary. Never an object id."""
        if self._identity is None:
            self._identity = f"sym-reloc-export@{self.executable.resolve()}"
        try:
            status = os.stat(self.executable)
        except OSError:
            return self._identity
        return f"{self._identity}#{status.st_size}-{status.st_mtime_ns}"

    def compile(self, recipe):
        mlir = emit_mlir(recipe).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="reloc-compile-") as directory:
            root = Path(directory)
            input_path = root / "recipe.mlir"
            plan_path = root / "plan.bin"
            manifest_path = root / "manifest.json"
            input_path.write_bytes(mlir)
            try:
                result = subprocess.run(
                    [
                        str(self.executable),
                        str(input_path),
                        "--output",
                        str(plan_path),
                        "--manifest",
                        str(manifest_path),
                    ],
                    check=False,
                    capture_output=True,
                )
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
