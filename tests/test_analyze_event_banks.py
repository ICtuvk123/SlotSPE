import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.analyze_event_banks import (
    activation_comparison,
    embedding_comparison,
    load_event_bank,
    patient_id_from_filename,
)


class EventBankAnalysisTest(unittest.TestCase):
    def _save_bank(self, path, event_ids, embeddings, version):
        torch.save(
            {
                "event_ids": event_ids,
                "event_names": [event_id.replace("_", " ") for event_id in event_ids],
                "event_embeddings": F.normalize(embeddings.float(), dim=-1),
                "ontology_version": version,
                "review_status": "machine_curated_pending_pathologist_review",
            },
            path,
        )

    def test_embedding_and_streaming_activation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v0_path = root / "v0.pt"
            v2_path = root / "v2.pt"
            feature_dir = root / "patches"
            feature_dir.mkdir()
            self._save_bank(
                v0_path,
                ["v0_x", "v0_y"],
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "v0",
            )
            self._save_bank(
                v2_path,
                ["v2_x", "v2_diag"],
                torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
                "v2",
            )
            torch.save(
                F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1),
                feature_dir / "TCGA-AA-0001-01Z.slide_a.pt",
            )
            torch.save(
                F.normalize(torch.tensor([[1.0, 1.0], [1.0, 0.0]]), dim=-1),
                feature_dir / "TCGA-AA-0001-01Z.slide_b.pt",
            )

            v0 = load_event_bank(v0_path)
            v2 = load_event_bank(v2_path)
            embedding = embedding_comparison(v0, v2, threshold=0.9)
            self.assertEqual(embedding["v2_nearest_v0"][0]["v2_event_id"], "v2_x")
            self.assertAlmostEqual(
                embedding["v2_nearest_v0"][0]["cosine_similarity"], 1.0
            )

            activation = activation_comparison(
                v0,
                v2,
                feature_dir=feature_dir,
                device=torch.device("cpu"),
                batch_size=2,
                score_threshold=0.7,
                min_slide_fraction=0.5,
                min_patient_fraction=0.5,
                max_files=None,
            )
            self.assertEqual(activation["num_slides"], 2)
            self.assertEqual(activation["num_patients"], 1)
            self.assertEqual(activation["num_patches"], 4)
            record = next(
                item for item in activation["events"] if item["event_id"] == "v2_x"
            )
            self.assertEqual(record["activated_patches"], 3)
            self.assertAlmostEqual(record["patch_activation_frequency"], 0.75)
            self.assertEqual(record["covered_patients"], 1)

    def test_tcga_patient_id_is_derived_from_first_three_barcode_fields(self):
        path = Path("TCGA-BP-1234-01Z-00-DX1.uuid.pt")
        self.assertEqual(patient_id_from_filename(path), "TCGA-BP-1234")


if __name__ == "__main__":
    unittest.main()
