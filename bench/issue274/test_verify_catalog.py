import hashlib
from io import BytesIO
import tarfile
import tempfile
import unittest
from pathlib import Path

from bench.issue274.verify_catalog import verify_archive, verify_file


def inventory(name, data):
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class CatalogIntegrityTests(unittest.TestCase):
    def test_archive_parts_and_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.tar.gz"
            data = b"original evidence"
            member = "experiment/sym-benefit-assessment.md"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, BytesIO(data))
            blob = path.read_bytes()
            first, second = blob[:47], blob[47:]
            record = inventory(path.name, blob) | {
                "parts": [inventory("part-001", first), inventory("part-002", second)],
                "members": [inventory(member, data)],
            }
            verify_archive(path, record)

            changed_part = inventory("part-001", first[::-1])
            with self.assertRaisesRegex(ValueError, "part-001"):
                verify_archive(path, record | {"parts": [changed_part, record["parts"][1]]})
            with self.assertRaisesRegex(ValueError, "complete archive"):
                verify_archive(path, record | {"sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "sym-benefit-assessment"):
                verify_archive(path, record | {"members": [inventory(member, b"modified evidence")]})

    def test_repository_file_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.md"
            record = inventory(path.name, b"correct")
            path.write_bytes(b"correct")
            verify_file(path, record)
            path.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "report.md"):
                verify_file(path, record)


if __name__ == "__main__":
    unittest.main()
