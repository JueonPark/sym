"""Export, bind and execute one symbolic layout for three input sizes."""

from importlib.resources import files
import subprocess
import tempfile
from pathlib import Path


def run(device):
    import numpy as np
    import pyreloc
    from sym_reloc.tools import native_tool

    with tempfile.TemporaryDirectory(prefix="sym-demo-") as directory:
        root = Path(directory)
        recipe = root / "recipe.mlir"
        recipe.write_text(
            files(__package__).joinpath("split_transpose.mlir").read_text()
        )
        subprocess.run(
            [
                str(native_tool("sym-reloc-export")),
                str(recipe),
                "--output",
                str(root / "plan.bin"),
                "--manifest",
                str(root / "manifest.json"),
            ],
            check=True,
            capture_output=True,
        )
        plan = pyreloc.load_plan((root / "plan.bin").read_bytes())
        for n in (128, 192, 256):
            source = np.arange(n, dtype=np.float32)
            output = np.empty(n, dtype=np.float32)
            bound = pyreloc.bind(plan, {"s0": n})
            pyreloc.relocate(
                bound,
                source.ctypes.data,
                source.nbytes,
                output.ctypes.data,
                output.nbytes,
            )
            np.testing.assert_array_equal(
                output.reshape(64, -1), source.reshape(-1, 64).T
            )
    return dict(sizes=[128, 192, 256], runtime_executions=3, direction="host")
