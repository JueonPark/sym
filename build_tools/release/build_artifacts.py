"""Build local candidate wheels/SDKs and the self-contained installer helper.

Candidates are not portable release artifacts until separately audited and
qualified. This command neither retags wheels nor publishes anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipapp

from .make_manifest import asset_record

ROOT = Path(__file__).resolve().parents[2]


def build_helper(output: Path):
    with tempfile.TemporaryDirectory(prefix="sym-helper-") as directory:
        stage = Path(directory)
        for name in ("build_tools", "build_tools/release"):
            (stage / name).mkdir(exist_ok=True)
            (stage / name / "__init__.py").write_text("")
        for name in (
            "build_tools/install_sym.py",
            "build_tools/release/make_manifest.py",
        ):
            shutil.copy2(ROOT / name, stage / name)
        zipapp.create_archive(
            stage, output, main="build_tools.install_sym:main", compressed=True
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--variant", choices=["cpu", "cu126"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlir-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    import re

    if (
        not re.fullmatch(
            r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", args.version
        )
        or args.jobs < 1
    ):
        parser.error("use a three-part version and positive job count")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty to avoid mixing candidates")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    # Build from a private source snapshot so variant/version generation cannot
    # modify a user's checkout or race a second variant build.
    with tempfile.TemporaryDirectory(prefix="sym-candidate-") as temporary:
        stage = Path(temporary) / "source"
        shutil.copytree(
            ROOT,
            stage,
            ignore=shutil.ignore_patterns(
                ".git",
                "build",
                "dist",
                ".venv",
                ".superpowers",
                ".claude",
                "__pycache__",
                "*.egg-info",
            ),
        )
        (stage / "libreloc/python/sym_reloc/_version.py").write_text(
            f'__version__ = "{args.version}+{args.variant}"\n'
        )
        env = {
            **os.environ,
            "MLIR_DIR": str(args.mlir_dir.resolve()),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
        }
        cuda = "ON" if args.variant == "cu126" else "OFF"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
                f"-Ccmake.define.RELOC_ENABLE_CUDA={cuda}",
                f"-Ccmake.define.SYM_SOURCE_REVISION={revision}",
            ],
            cwd=stage,
            env=env,
            check=True,
        )
        native = Path(temporary) / "native"
        sdk = Path(temporary) / f"sym-sdk-{args.version}-{args.variant}"
        subprocess.run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                str(stage),
                "-B",
                str(native),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DMLIR_DIR={args.mlir_dir.resolve()}",
                f"-DRELOC_ENABLE_CUDA={cuda}",
                f"-DSYM_SOURCE_REVISION={revision}",
                "-DSYM_BUILD_TESTS=OFF",
                "-DSYM_BUILD_BENCHMARKS=OFF",
                "-DSYM_BUILD_PYTHON=OFF",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(native),
                "--target",
                "sym-opt",
                "sym-reloc-export",
                "reloc-run-artifact",
                "-j",
                str(args.jobs),
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--install", str(native), "--prefix", str(sdk)], check=True
        )
        with tarfile.open(output / f"{sdk.name}.tar.gz", "w:gz") as archive:
            archive.add(sdk, arcname=sdk.name)
    build_helper(output / "sym-installer.pyz")
    shutil.copy2(ROOT / "install.sh", output / "install.sh")
    shutil.copy2(
        ROOT / f"release/locks/{args.variant}.txt", output / f"{args.variant}.txt"
    )
    inventory = dict(
        release_version=args.version,
        variant=args.variant,
        source_revision=revision,
        source_dirty=dirty,
        qualified=False,
        assets=[asset_record(path, args.version) for path in sorted(output.iterdir())],
    )
    (output / "candidate.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        f"Candidate saved to {output}; portability, clean-host and GPU qualification are separate gates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
