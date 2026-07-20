import unittest

import torch
import torch.nn.functional as F

from models.event_grounding import ConchPatchEventEvidence, FeatureSpaceMismatchError


class ConchEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.z = F.normalize(torch.randn(2, 32, 16), dim=-1)
        self.e = F.normalize(torch.randn(10, 16), dim=-1)

    def test_shape_and_calibrated_support(self):
        module = ConchPatchEventEvidence("calibrated_sigmoid")
        raw, support = module(
            self.z,
            self.e,
            z_feature_space="conch_contrastive",
            event_feature_space="conch_contrastive",
        )
        self.assertEqual(raw.shape, (2, 10, 32))
        self.assertEqual(support.shape, (2, 10, 32))
        self.assertTrue(((support >= 0) & (support <= 1)).all())

    def test_softmax_is_over_events(self):
        module = ConchPatchEventEvidence("softmax")
        _, support = module(
            self.z,
            self.e,
            z_feature_space="conch_contrastive",
            event_feature_space="conch_contrastive",
        )
        self.assertTrue(torch.allclose(support.sum(dim=1), torch.ones(2, 32), atol=1e-6))

    def test_mask_allows_zero_padding(self):
        z = self.z.clone()
        z[:, -2:] = 0
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, -2:] = False
        raw, support = ConchPatchEventEvidence()(z, self.e,
            z_feature_space="conch_contrastive",
            event_feature_space="conch_contrastive", mask=mask)
        self.assertTrue((raw[:, :, -2:] == 0).all())
        self.assertTrue((support[:, :, -2:] == 0).all())

    def test_uni_to_conch_comparison_is_rejected(self):
        with self.assertRaises(FeatureSpaceMismatchError):
            ConchPatchEventEvidence()(self.z, self.e,
                z_feature_space="uni", event_feature_space="conch_contrastive")

    def test_conch_v15_matching_space_is_accepted(self):
        raw, _ = ConchPatchEventEvidence()(self.z, self.e,
            z_feature_space="conch_v1_5_contrastive",
            event_feature_space="conch_v1_5_contrastive")
        self.assertEqual(raw.shape, (2, 10, 32))

    def test_conch_versions_cannot_be_mixed(self):
        with self.assertRaises(FeatureSpaceMismatchError):
            ConchPatchEventEvidence()(self.z, self.e,
                z_feature_space="conch_v1_5_contrastive",
                event_feature_space="conch_contrastive")


if __name__ == "__main__":
    unittest.main()
