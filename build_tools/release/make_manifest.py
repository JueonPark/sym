"""Standard-library release admission shared by builders and the installer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

RELEASE_ROOT = "https://github.com/JueonPark/sym/releases/download/"
HEX = re.compile(r"[0-9a-f]{64}")
VERSION = re.compile(r"0|[1-9][0-9]*")


def validate_asset(asset: dict, version: str) -> None:
    if not isinstance(asset, dict):
        raise ValueError("asset must be an object")
    name = asset.get("filename")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("invalid asset filename")
    if type(asset.get("size")) is not int or asset["size"] <= 0:
        raise ValueError("invalid asset size")
    if not isinstance(asset.get("sha256"), str) or not HEX.fullmatch(asset["sha256"]):
        raise ValueError("invalid asset checksum")
    url = asset.get("url")
    if not isinstance(url, str) or not url.startswith(f"{RELEASE_ROOT}v{version}/"):
        raise ValueError("asset URL must belong to the selected official release")
    parsed = urlsplit(url)
    if (
        parsed.query
        or parsed.fragment
        or unquote(parsed.path.rsplit("/", 1)[-1]) != name
    ):
        raise ValueError("asset URL/filename mismatch")


def validate_manifest(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    for name in ("schema_version", "installer_protocol"):
        if type(data.get(name)) is not int or data[name] != 1:
            raise ValueError(f"unsupported {name}")
    version = data.get("release_version")
    if (
        not isinstance(version, str)
        or len(version.split(".")) != 3
        or not all(VERSION.fullmatch(part) for part in version.split("."))
    ):
        raise ValueError("invalid release_version")
    for name, expected in [
        ("python_version", "3.14.7"),
        ("python_abi", "cp314-cp314"),
        ("glibc_min", "2.28"),
        ("llvm_revision", "2078da43e25a4623cab2d0d60decddf709aaea28"),
    ]:
        if data.get(name) != expected:
            raise ValueError(f"unsupported {name}")
    if not isinstance(data.get("source_revision"), str) or not re.fullmatch(
        "[0-9a-f]{40}", data["source_revision"]
    ):
        raise ValueError("invalid source_revision")
    variants = data.get("variants")
    if (
        not isinstance(variants, dict)
        or not variants
        or set(variants) - {"cpu", "cu126"}
    ):
        raise ValueError("invalid variants")
    for name, variant in variants.items():
        if not isinstance(variant, dict):
            raise ValueError("variant must be an object")
        expected = dict(
            package_version=f"{version}+{name}",
            torch_version=f"2.14.0+{name}",
            cuda="12.6" if name == "cu126" else None,
        )
        for key, value in expected.items():
            if key not in variant or variant[key] != value:
                raise ValueError(f"variant {name}: wrong {key}")
        assets = variant.get("assets")
        if (
            not isinstance(assets, dict)
            or not {"wheel", "sdk", "lock", "helper"} <= assets.keys()
        ):
            raise ValueError("missing variant assets")
        for asset in assets.values():
            validate_asset(asset, version)
        wheel = assets["wheel"]["filename"]
        if not wheel.startswith(
            f"sym_reloc-{version}+{name}-cp314-cp314-"
        ) or not wheel.endswith(".whl"):
            raise ValueError("wheel version/variant/ABI mismatch")
    return data


def select_variant(manifest: dict, variant: str) -> dict:
    validate_manifest(manifest)
    if variant not in manifest["variants"]:
        raise ValueError(f"release does not contain requested variant {variant}")
    return manifest["variants"][variant]


def verify_asset(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["size"]:
        raise ValueError(f"asset size mismatch: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != asset["sha256"]:
        raise ValueError(f"asset checksum mismatch: {path.name}")


def asset_record(path: Path, version: str) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        filename=path.name,
        url=f"{RELEASE_ROOT}v{version}/{path.name}",
        size=path.stat().st_size,
        sha256=digest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print("Release manifest valid")


if __name__ == "__main__":
    main()
