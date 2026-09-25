"""Install a validated release into a private prefix without moving its venv."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from urllib.parse import urlsplit
import uuid

from build_tools.release.make_manifest import select_variant, verify_asset


@contextmanager
def prefix_lock(prefix):
    lock = prefix / ".install.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another installer is using this prefix") from error
        yield
    finally:
        os.close(fd)


def validate_platform(manifest):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("Sym binaries require glibc Linux x86_64")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split("."))) < (2, 28):
        raise ValueError("Sym binaries require glibc >= 2.28")
    import sysconfig

    if (
        platform.python_version() != manifest["python_version"]
        or platform.python_implementation() != "CPython"
        or sysconfig.get_config_var("SOABI") != "cpython-314-x86_64-linux-gnu"
        or not getattr(sys, "_is_gil_enabled", lambda: False)()
    ):
        raise ValueError("installer helper requires regular-GIL CPython 3.14.7")


class ReleaseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {
            "github.com",
            "release-assets.githubusercontent.com",
            "objects.githubusercontent.com",
        }:
            raise ValueError("release download redirected to an unsupported host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_asset(asset, destination):
    # URL admission happens in select_variant; redirects are checked before use.
    opener = urllib.request.build_opener(ReleaseRedirects())
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with (
            opener.open(asset["url"], timeout=60) as source,
            temporary.open("xb") as output,
        ):
            remaining = asset["size"]
            while chunk := source.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError("download exceeds declared asset size")
                output.write(chunk)
        verify_asset(temporary, asset)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def subprocess_environment(prefix):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("UV_", "PIP_", "PYTHON"))
        and k not in {"LD_LIBRARY_PATH", "SYM_OPT", "SYM_RELOC_EXPORT", "VIRTUAL_ENV"}
    }
    env.update(
        UV_NO_CONFIG="1",
        UV_CACHE_DIR=str(prefix / "cache"),
        UV_PYTHON_INSTALL_DIR=str(prefix / "python"),
    )
    return env


def install_environment(environment, manifest, variant, uv, python, prefix):
    selected = select_variant(manifest, variant)
    env = subprocess_environment(prefix)
    subprocess.run(
        [str(uv), "venv", "--python", str(python), str(environment)],
        check=True,
        env=env,
        cwd=prefix,
    )
    with tempfile.TemporaryDirectory(
        prefix="downloads-", dir=prefix / "cache"
    ) as temporary:
        root = Path(temporary)
        lock = download_asset(
            selected["assets"]["lock"], root / selected["assets"]["lock"]["filename"]
        )
        wheel = download_asset(
            selected["assets"]["wheel"], root / selected["assets"]["wheel"]["filename"]
        )
        target = environment / "bin/python"
        subprocess.run(
            [
                str(uv),
                "pip",
                "sync",
                "--python",
                str(target),
                "--require-hashes",
                "--only-binary",
                ":all:",
                str(lock),
            ],
            check=True,
            env=env,
            cwd=prefix,
        )
        subprocess.run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(target),
                "--no-deps",
                str(wheel),
            ],
            check=True,
            env=env,
            cwd=prefix,
        )


def validate_environment(environment, variant, manifest):
    env = subprocess_environment(environment.parent.parent)
    reports = {}
    for name, args in [
        (
            "doctor",
            ["--require-torch"] + (["--require-cuda"] if variant == "cu126" else []),
        ),
        ("demo", ["--device", "cuda" if variant == "cu126" else "cpu"]),
    ]:
        result = subprocess.run(
            [str(environment / "bin" / f"sym-{name}"), *args, "--json"],
            cwd=environment,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stdout}\n{result.stderr}")
        data = json.loads(result.stdout)
        if data.get("status") != "ok" or data.get("variant") != variant:
            raise ValueError(f"{name} did not validate requested variant")
        identity = data.get("checks", {}).get("build", {}).get("detail", {})
        if (
            data.get("versions", {}).get("sym")
            != manifest["variants"][variant]["package_version"]
            or identity.get("source_revision") != manifest["source_revision"]
        ):
            raise ValueError(
                f"{name} installed release identity does not match manifest"
            )
        reports[name] = data
    return reports


def managed_current(prefix):
    current = prefix / "current"
    if not current.exists() and not current.is_symlink():
        return None
    if not current.is_symlink():
        raise ValueError("current must be a managed symlink, not an existing user file")
    target = current.resolve()
    if (
        target.parent != (prefix / "envs").resolve()
        or not (target / "installation.json").is_file()
    ):
        raise ValueError("current does not point to a managed environment")
    return target


def activate_environment(prefix, environment, previous):
    def replace_link(name, target):
        path = prefix / name
        if path.exists() and not path.is_symlink():
            raise ValueError(f"{name} is an unmanaged entry")
        temporary = prefix / f".{name}-{uuid.uuid4().hex}"
        try:
            temporary.symlink_to(target)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    rollback = prefix / "previous"
    original_rollback = rollback.readlink() if rollback.is_symlink() else None
    if previous is not None:
        replace_link("previous", previous)
    try:
        replace_link("current", environment)
    except BaseException:
        if previous is not None:
            if original_rollback is None:
                rollback.unlink(missing_ok=True)
            else:
                replace_link("previous", original_rollback)
        raise


def install(prefix, manifest, variant, uv, python, upgrade=False):
    selected = select_variant(manifest, variant)
    validate_platform(manifest)
    prefix = Path(prefix).absolute()
    if prefix.is_symlink():
        raise ValueError("installation prefix must not be a symlink")
    prefix.mkdir(parents=True, exist_ok=True)
    prefix = prefix.resolve()
    with prefix_lock(prefix):
        for name in ("envs", "cache", "logs"):
            path = prefix / name
            if path.is_symlink():
                raise ValueError(f"{name} must not be a symlink")
            path.mkdir(exist_ok=True)
        previous = managed_current(prefix)
        identity = dict(
            release_version=manifest["release_version"],
            variant=variant,
            source_revision=manifest["source_revision"],
            assets=selected["assets"],
        )
        if previous:
            old = json.loads((previous / "installation.json").read_text())
            if all(old.get(k) == v for k, v in identity.items()):
                validate_environment(previous, variant, manifest)
                return previous
            if not upgrade:
                raise ValueError(
                    "a different release is active; use --upgrade to replace it"
                )
        environment = (
            prefix
            / "envs"
            / f"{manifest['release_version']}-{variant}-{uuid.uuid4().hex}"
        )
        environment.mkdir()
        # This path is final: venv shebangs and interpreter links must never move.
        try:
            install_environment(environment, manifest, variant, uv, python, prefix)
            evidence = validate_environment(environment, variant, manifest)
            (environment / "installation.json").write_text(
                json.dumps(identity | {"validation": evidence}, indent=2) + "\n"
            )
            activate_environment(prefix, environment, previous)
        except BaseException as error:
            # Only this invocation's candidate is removed, never the active target.
            if (
                not (prefix / "current").is_symlink()
                or (prefix / "current").resolve() != environment
            ):
                shutil.rmtree(environment)
            (prefix / "logs" / f"failed-{environment.name}.log").write_text(
                str(error) + "\n"
            )
            raise
        return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--variant", choices=["cpu", "cu126"], required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        path = install(
            args.prefix,
            manifest,
            args.variant,
            args.uv,
            Path(sys.executable),
            args.upgrade,
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Sym installation failed: {error}", file=sys.stderr)
        return 1
    import shlex

    print("Sym installed and validated.")
    print(
        "Activate: source "
        + shlex.quote(str(args.prefix.absolute() / "current/bin/activate"))
    )
    print("Python: " + shlex.quote(str(path / "bin/python")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
