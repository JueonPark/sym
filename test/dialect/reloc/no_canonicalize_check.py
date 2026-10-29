"""sym-reloc-export --no-canonicalize: same contract, different plan bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TOOL = shutil.which(sys.argv[1])
assert TOOL, "supported sym-reloc-export executable is missing"
SOURCE = Path(sys.argv[2]).read_text()


def run(root, *flags):
    root.mkdir()
    inp, plan, manifest = (root / n for n in ("input.mlir", "plan.bin", "manifest.json"))
    inp.write_text(SOURCE)
    proc = subprocess.run([TOOL, str(inp), "--output", str(plan), "--manifest", str(manifest), *flags],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    meta = json.loads(manifest.read_text())
    blob = plan.read_bytes()
    assert meta["status"] == "ok" and meta["plan_sha256"] == hashlib.sha256(blob).hexdigest()
    return blob, meta


with tempfile.TemporaryDirectory() as tmp:
    canonical, canonical_meta = run(Path(tmp) / "canonical")
    raw, raw_meta = run(Path(tmp) / "raw", "--no-canonicalize")
    again, _ = run(Path(tmp) / "raw-again", "--no-canonicalize")
    assert raw == again, "the raw export is deterministic"
    assert raw != canonical, "the raw export keeps the unmerged axes"
    assert len(raw) > len(canonical)
    # Same interface: the manifest records the same contract (schema, wire
    # version, descriptors, symbols); only the plan digest differs, and the
    # manifest does not record the mode.
    for key in ("schema_version", "wire_version", "symbols", "logical_source", "logical_destination",
                "constraints", "compiler", "plan_count", "input_sha256"):
        assert canonical_meta[key] == raw_meta[key], key
    assert canonical_meta["plan_sha256"] != raw_meta["plan_sha256"]
    assert set(canonical_meta) == set(raw_meta)
print("ok")
