"""Atomic activation and failure behavior, using real directories and symlinks."""

import importlib
import json
from pathlib import Path
import sys

import pytest


def installer_module():
    try:
        return importlib.import_module("build_tools.install_sym")
    except ModuleNotFoundError:
        pytest.fail("installer is not implemented")


@pytest.fixture
def installation(tmp_path, monkeypatch):
    api = installer_module()
    prefix = tmp_path / "prefix with spaces"
    manifest = {
        "release_version": "0.1.0",
        "source_revision": "a" * 40,
        "variants": {
            "cpu": {
                "package_version": "0.1.0+cpu",
                "assets": {"wheel": {"sha256": "b" * 64}},
            }
        },
    }
    monkeypatch.setattr(api, "select_variant", lambda m, v: m["variants"][v])
    monkeypatch.setattr(api, "validate_platform", lambda m: None)

    def construct(env, *args):
        (env / "bin").mkdir()
        (env / "bin/python").symlink_to(sys.executable)

    monkeypatch.setattr(api, "install_environment", construct)
    monkeypatch.setattr(
        api,
        "validate_environment",
        lambda *args: {"doctor": {"status": "ok"}, "demo": {"status": "ok"}},
    )
    return api, prefix, manifest


def test_install_idempotent_and_upgrade_retains_previous(installation):
    api, prefix, manifest = installation
    first = api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
    assert (prefix / "current").resolve() == first
    assert (
        api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
        == first
    )
    manifest["release_version"] = "0.2.0"
    manifest["variants"]["cpu"]["package_version"] = "0.2.0+cpu"
    with pytest.raises(ValueError, match="upgrade"):
        api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
    second = api.install(
        prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable), upgrade=True
    )
    assert second != first and first.is_dir()
    assert (prefix / "previous").resolve() == first
    assert (prefix / "current").resolve() == second


def test_failed_upgrade_never_activates_candidate(installation, monkeypatch):
    api, prefix, manifest = installation
    first = api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
    manifest["release_version"] = "0.2.0"

    def failed(*args):
        raise RuntimeError("typed demo mismatch")

    monkeypatch.setattr(api, "validate_environment", failed)
    with pytest.raises(RuntimeError, match="typed demo mismatch"):
        api.install(
            prefix,
            manifest,
            "cpu",
            Path("/fake/uv"),
            Path(sys.executable),
            upgrade=True,
        )
    assert (prefix / "current").resolve() == first
    assert list((prefix / "envs").iterdir()) == [first]


def test_unmanaged_prefix_entries_remain_untouched(installation):
    api, prefix, manifest = installation
    prefix.mkdir()
    (prefix / "current").write_text("user data")
    with pytest.raises(ValueError, match="managed|symlink"):
        api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
    assert (prefix / "current").read_text() == "user data"


def test_concurrent_install_rejected(installation):
    api, prefix, manifest = installation
    prefix.mkdir()
    with api.prefix_lock(prefix):
        with pytest.raises(RuntimeError, match="another installer"):
            api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))


def test_symlinked_storage_rejected(installation, tmp_path):
    api, prefix, manifest = installation
    prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (prefix / "envs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        api.install(prefix, manifest, "cpu", Path("/fake/uv"), Path(sys.executable))
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--cpu", "--cuda", "cu126"],
        ["--cuda", "cu130"],
        ["--cpu", "--version", "../bad"],
    ],
)
def test_bootstrap_rejects_invalid_cli_before_download(arguments):
    import subprocess

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "install.sh"), *arguments], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Sym installation failed" in result.stderr


def test_validation_rejects_wrong_release_identity(tmp_path, monkeypatch):
    api = installer_module()
    import subprocess

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            0,
            json.dumps(
                {
                    "status": "ok",
                    "variant": "cpu",
                    "versions": {"sym": "9.9.9+cpu"},
                    "checks": {"build": {"detail": {"source_revision": "c" * 40}}},
                }
            ),
            "",
        )

    monkeypatch.setattr(api.subprocess, "run", run)
    manifest = {
        "variants": {"cpu": {"package_version": "0.1.0+cpu"}},
        "source_revision": "a" * 40,
    }
    with pytest.raises(ValueError, match="identity"):
        api.validate_environment(tmp_path / "environment", "cpu", manifest)


def test_real_environment_from_candidate(tmp_path, monkeypatch):
    """Run real uv/venv/wheel/doctor/demo; substitute only unpublished asset hosting."""
    import os
    import shutil
    import tomllib

    artifact_dir = os.environ.get("SYM_INSTALLER_SMOKE_ARTIFACTS")
    if not artifact_dir:
        pytest.skip("set SYM_INSTALLER_SMOKE_ARTIFACTS to a CPU candidate directory")
    api = installer_module()
    from build_tools.release.make_manifest import asset_record, verify_asset

    root = Path(__file__).resolve().parents[2]
    assets = Path(artifact_dir)
    baseline = tomllib.loads((root / "release/compatibility.toml").read_text())
    inventory = json.loads((assets / "candidate.json").read_text())
    manifest = {
        key: baseline[key]
        for key in (
            "release_version",
            "python_version",
            "python_abi",
            "glibc_min",
            "llvm_revision",
        )
    }
    manifest.update(
        schema_version=1,
        installer_protocol=1,
        source_revision=inventory["source_revision"],
    )
    paths = {
        "wheel": next(assets.glob("*.whl")),
        "sdk": next(assets.glob("*.tar.gz")),
        "lock": assets / "cpu.txt",
        "helper": assets / "sym-installer.pyz",
    }
    manifest["variants"] = {
        "cpu": dict(
            package_version="0.1.0+cpu",
            torch_version="2.14.0+cpu",
            cuda=None,
            assets={k: asset_record(p, "0.1.0") for k, p in paths.items()},
        )
    }

    def local_release(asset, destination):
        shutil.copy2(assets / asset["filename"], destination)
        verify_asset(destination, asset)
        return destination

    monkeypatch.setattr(api, "download_asset", local_release)
    prefix = tmp_path / "real installation"
    cache_seed = os.environ.get("SYM_INSTALLER_SMOKE_CACHE")
    if cache_seed:
        # Prewarm a private cache using immutable downloaded files; installation
        # still runs the actual uv resolver/hash checks and all numerical gates.
        shutil.copytree(
            cache_seed,
            prefix / "cache",
            copy_function=os.link,
            symlinks=True,
            ignore=shutil.ignore_patterns(".tmp*"),
        )
    installed = api.install(
        prefix, manifest, "cpu", Path(shutil.which("uv")), Path(sys.executable)
    )
    evidence = json.loads((installed / "installation.json").read_text())["validation"]
    assert evidence["demo"]["status"] == "ok"
    assert (prefix / "current").resolve() == installed
    assert (
        api.install(
            prefix, manifest, "cpu", Path(shutil.which("uv")), Path(sys.executable)
        )
        == installed
    )


def test_bootstrap_ignores_inherited_uv_configuration(tmp_path):
    import os
    import subprocess

    script = Path(__file__).resolve().parents[2] / "install.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; sanitize_uv_environment; env',
            "test",
            str(script),
        ],
        env={
            **os.environ,
            "UV_CONFIG_FILE": "/unrelated/config",
            "UV_PYTHON_INSTALL_DIR": "/unrelated/python",
            "UV_NO_CONFIG": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "UV_CONFIG_FILE=" not in result.stdout
    assert "UV_PYTHON_INSTALL_DIR=" not in result.stdout


def test_failed_activation_preserves_both_rollback_links(tmp_path, monkeypatch):
    api = installer_module()
    prefix = tmp_path
    older = prefix / "A"
    old = prefix / "B"
    candidate = prefix / "C"
    for p in (older, old, candidate):
        p.mkdir()
    (prefix / "previous").symlink_to(older)
    (prefix / "current").symlink_to(old)
    replace = Path.replace

    def fail_current(self, target):
        if Path(target) == prefix / "current":
            raise OSError("activation failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_current)
    with pytest.raises(OSError, match="activation failure"):
        api.activate_environment(prefix, candidate, old)
    assert (prefix / "current").resolve() == old
    assert (prefix / "previous").resolve() == older
