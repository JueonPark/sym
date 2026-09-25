"""Exercise installed native tools and an independent CMake consumer."""

import os
from pathlib import Path
import struct
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_relocated_sdk(tmp_path):
    build = os.environ.get("SYM_PACKAGING_BUILD")
    if not build:
        pytest.skip("set SYM_PACKAGING_BUILD to a built install configuration")
    original = tmp_path / "original"
    subprocess.run(
        ["cmake", "--install", build, "--prefix", str(original)],
        check=True,
        capture_output=True,
    )
    assert (original / "bin/sym-reloc-export").is_file(), (
        "exporter missing from install"
    )
    relocated = tmp_path / "moved SDK"
    original.rename(relocated)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LD_LIBRARY_PATH", "PYTHONPATH"}
    }
    subprocess.run(
        [str(relocated / "bin/sym-opt"), "--version"],
        check=True,
        env=env,
        capture_output=True,
    )
    recipe = ROOT / "libreloc/examples/recipes/split_transpose.mlir"
    plan, manifest = tmp_path / "plan.bin", tmp_path / "manifest.json"
    subprocess.run(
        [
            str(relocated / "bin/sym-reloc-export"),
            str(recipe),
            "--output",
            str(plan),
            "--manifest",
            str(manifest),
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    source, output = tmp_path / "input.bin", tmp_path / "output.bin"
    source.write_bytes(struct.pack("<128f", *range(128)))
    subprocess.run(
        [
            str(relocated / "bin/reloc-run-artifact"),
            str(plan),
            "--symbols",
            "s0=128",
            "--input",
            str(source),
            "--output",
            str(output),
            "--direction",
            "host",
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    assert struct.unpack("<128f", output.read_bytes()) == tuple(
        float(r * 64 + c) for c in range(64) for r in range(2)
    )
    consumer = tmp_path / "consumer"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "tests/packaging/cpp_consumer"),
            "-B",
            str(consumer),
            f"-DCMAKE_PREFIX_PATH={relocated}",
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(consumer)], check=True, env=env, capture_output=True
    )
    subprocess.run(
        [str(consumer / "consumer")], check=True, env=env, capture_output=True
    )
