import tempfile
import unittest
from pathlib import Path

from tools.titan_local import (
    CONCH_REQUIRED_FILES,
    TITAN_REQUIRED_FILES,
    validate_titan_snapshot,
)


class TitanSnapshotValidationTest(unittest.TestCase):
    def test_requires_local_text_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in TITAN_REQUIRED_FILES:
                (root / name).touch()
            self.assertEqual(validate_titan_snapshot(root), root.resolve())

    def test_conch_assets_are_checked_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in TITAN_REQUIRED_FILES:
                (root / name).touch()
            with self.assertRaisesRegex(FileNotFoundError, "conch_v1_5"):
                validate_titan_snapshot(root, require_conch=True)
            for name in CONCH_REQUIRED_FILES:
                (root / name).touch()
            self.assertEqual(
                validate_titan_snapshot(root, require_conch=True), root.resolve()
            )


if __name__ == "__main__":
    unittest.main()
