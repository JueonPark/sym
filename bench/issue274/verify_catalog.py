"""Check frozen repository evidence and optional downloaded archives without network access."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


CATALOG = Path(__file__).with_name("historical-manifest.json")
ROOT = Path(__file__).resolve().parents[2]
CHUNK_BYTES = 1 << 20


def verify_file(path, record):
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            count += len(block)
            digest.update(block)
    if (count, digest.hexdigest()) != (record["bytes"], record["sha256"]):
        raise ValueError(f"size or SHA-256 differs for {path}")


def verify_archive(path, record):
    """Check original split-file boundaries in an already assembled archive."""
    whole = hashlib.sha256()
    with path.open("rb") as source:
        for part in record["parts"]:
            remaining = part["bytes"]
            digest = hashlib.sha256()
            while remaining:
                block = source.read(min(CHUNK_BYTES, remaining))
                if not block:
                    raise ValueError(f"truncated archive part {part['name']}")
                remaining -= len(block)
                digest.update(block)
                whole.update(block)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError(f"SHA-256 differs for {part['name']}")
        if source.read(1) or whole.hexdigest() != record["sha256"]:
            raise ValueError(f"size or SHA-256 differs for complete archive {path}")
    members = {member["name"]: member for member in record.get("members", [])}
    if not members:
        return
    with tarfile.open(path, "r|gz") as archive:
        for entry in archive:
            if entry.name not in members:
                continue
            expected = members.pop(entry.name)
            if not entry.isfile() or entry.size != expected["bytes"]:
                raise ValueError(f"size or type differs for {entry.name} in {path}")
            with archive.extractfile(entry) as source:
                digest = hashlib.sha256()
                for block in iter(lambda: source.read(CHUNK_BYTES), b""):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA-256 differs for {entry.name} in {path}")
            if not members:
                return
    raise ValueError(f"missing archive members: {', '.join(members)} in {path}")


def validate_catalog(catalog):
    if catalog["schema"] != 1:
        raise ValueError("unsupported catalog schema")
    archives = catalog["archives"]
    files = catalog["repository_files"]
    if set(catalog["campaigns"]) != {f"E{n}" for n in range(1, 6)} | {
        str(n) for n in range(263, 273)
    }:
        raise ValueError("incomplete historical campaign inventory")
    if set(archives) != {"original"} | {str(n) for n in range(263, 270)}:
        raise ValueError("incomplete historical Wiki archive inventory")
    revision = catalog["wiki_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Wiki revision must be a Git commit SHA")
    for key, archive in archives.items():
        if not archive["wiki_revision_url"].endswith("/" + revision):
            raise ValueError(f"missing Wiki revision pin for {key}")
        if archive["name"] != Path(archive["name"]).name:
            raise ValueError(f"invalid download name for {key}")
        parts = archive["parts"]
        if (not parts or sum(part["bytes"] for part in parts) != archive["bytes"] or
                len({part["name"] for part in parts}) != len(parts) or
                any(not 0 < part["bytes"] <= 32 << 20 for part in parts)):
            raise ValueError(f"inconsistent split-file inventory for {key}")
    records = [*files.values(), catalog["wiki_downloader"]]
    for archive in archives.values():
        records.extend([archive, *archive["parts"], *archive.get("members", []),
                        archive["download_manifest"], *archive["checksums"]])
    if any(not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) for record in records):
        raise ValueError("invalid SHA-256 field")
    for key, campaign in catalog["campaigns"].items():
        if campaign.get("archive") is not None and campaign["archive"] not in archives:
            raise ValueError(f"unknown Wiki archive for {key}")
        if set(campaign.get("repo_paths", ())) - files.keys():
            raise ValueError(f"unverified repository path for {key}")
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid repository path {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, help="directory of assembled Wiki downloads")
    parser.add_argument("--archive", action="append", help="verify only this archive ID (repeatable)")
    args = parser.parse_args()
    try:
        catalog = json.loads(CATALOG.read_text())
        validate_catalog(catalog)
        for name, record in catalog["repository_files"].items():
            verify_file(ROOT / name, record)
        print(f"Verified {len(catalog['repository_files'])} repository files across "
              f"{len(catalog['campaigns'])} campaigns")
        if args.archive and args.archive_dir is None:
            raise ValueError("--archive requires --archive-dir")
        if args.archive_dir is not None:
            for name in args.archive or catalog["archives"]:
                archive = catalog["archives"][name]
                verify_archive(args.archive_dir / archive["name"], archive)
                print(f"Verified {name}: {len(archive['parts'])} splits, whole archive, "
                      f"pinned members: {len(archive.get('members', []))}")
    except (KeyError, ValueError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"Catalog verification failed: {error}\n")


if __name__ == "__main__":
    main()
