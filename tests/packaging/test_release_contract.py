"""Release admission must reject a corrupt or mismatched executable payload."""

import copy
import hashlib
import importlib

import pytest


def contract():
    try:
        return importlib.import_module("build_tools.release.make_manifest")
    except ModuleNotFoundError:
        pytest.fail("release manifest validator is not implemented")


@pytest.fixture
def manifest(tmp_path):
    asset = tmp_path / "sym_reloc-0.1.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl"
    asset.write_bytes(b"candidate bytes")
    record = dict(
        filename=asset.name,
        url=f"https://github.com/JueonPark/sym/releases/download/v0.1.0/{asset.name}",
        size=asset.stat().st_size,
        sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
    )
    return dict(
        schema_version=1,
        installer_protocol=1,
        release_version="0.1.0",
        source_revision="a" * 40,
        python_version="3.14.7",
        python_abi="cp314-cp314",
        glibc_min="2.28",
        llvm_revision="2078da43e25a4623cab2d0d60decddf709aaea28",
        variants={
            "cpu": dict(
                package_version="0.1.0+cpu",
                torch_version="2.14.0+cpu",
                cuda=None,
                assets={
                    k: copy.deepcopy(record) for k in ("wheel", "sdk", "lock", "helper")
                },
            )
        },
    )


def test_admit_cpu_and_verify_real_download(manifest, tmp_path):
    api = contract()
    assert api.validate_manifest(manifest) == manifest
    record = manifest["variants"]["cpu"]["assets"]["wheel"]
    api.verify_asset(tmp_path / record["filename"], record)
    (tmp_path / record["filename"]).write_bytes(b"corrupted bytes")
    with pytest.raises(ValueError, match="size|checksum"):
        api.verify_asset(tmp_path / record["filename"], record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("installer_protocol", 2),
        ("python_version", "3.14.6"),
        ("python_abi", "cp314-cp314t"),
        ("source_revision", "bad"),
    ],
)
def test_reject_incompatible_identity(manifest, field, value):
    manifest[field] = value
    with pytest.raises(ValueError):
        contract().validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch_version", "2.14.0+cu130"),
        ("package_version", "0.2.0+cpu"),
        ("cuda", "12.6"),
    ],
)
def test_reject_wrong_variant(manifest, field, value):
    manifest["variants"]["cpu"][field] = value
    with pytest.raises(ValueError):
        contract().validate_manifest(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("filename", "../escape"),
        ("sha256", "x" * 64),
        ("size", True),
        ("url", "http://github.com/payload"),
        ("url", "https://evil.example/payload"),
        ("url", "https://github.com/other/project/payload"),
    ],
)
def test_reject_invalid_asset(manifest, field, value):
    manifest["variants"]["cpu"]["assets"]["wheel"][field] = value
    with pytest.raises(ValueError):
        contract().validate_manifest(manifest)


def test_absent_cuda_is_not_cpu_fallback(manifest):
    with pytest.raises(ValueError, match="variant"):
        contract().select_variant(manifest, "cu126")


def test_declared_baseline_is_accepted_by_frontend():
    import sys
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "libreloc/python"))
    from reloc_torch.compat import _validate_version

    baseline = tomllib.loads((root / "release/compatibility.toml").read_text())
    assert (
        baseline["llvm_revision"]
        == (root / "build_tools/llvm_version.txt").read_text().strip()
    )
    for variant in baseline["variants"].values():
        _validate_version(
            python_version=tuple(map(int, baseline["python_version"].split("."))),
            python_releaselevel="final",
            implementation="CPython",
            gil_enabled=True,
            soabi="cpython-314-x86_64-linux-gnu",
            torch_version=variant["torch_version"],
            torch_cuda=variant.get("cuda"),
        )
