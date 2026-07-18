import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset.dataset_survival import SurvivalDataset


class DatasetEventAlignmentTest(unittest.TestCase):
    def _dataset(self, directory):
        dataset = SurvivalDataset.__new__(SurvivalDataset)
        dataset.conch_patch_feature_dir = str(directory)
        dataset.reuse_slot_features_as_conch = False
        dataset.require_conch_alignment = True
        return dataset

    def _save(self, directory, verified):
        path = Path(directory) / "TCGA-TEST-01Z.pt"
        torch.save(
            {
                "slide_id": "TCGA-TEST-01Z",
                "feature_space": "conch_contrastive",
                "normalized": True,
                "alignment_verified": verified,
                "patch_order_sha256": "example",
                "conch_patch_embeddings": F.normalize(torch.randn(6, 8), dim=-1),
            },
            path,
        )

    def test_verified_features_load_in_slide_order(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save(directory, True)
            result = self._dataset(directory).load_conch_features(
                ["TCGA-TEST-01Z.svs"], [6]
            )
            self.assertEqual(result.shape, (6, 8))

    def test_unverified_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save(directory, False)
            with self.assertRaisesRegex(ValueError, "not verified"):
                self._dataset(directory).load_conch_features(
                    ["TCGA-TEST-01Z.svs"], [6]
                )


if __name__ == "__main__":
    unittest.main()
