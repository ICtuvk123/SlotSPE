import unittest

import torch

from models.event_grounding import KnownEventGate


class KnownEventGateTest(unittest.TestCase):
    def setUp(self):
        self.module = KnownEventGate(
            delta_sem=0.2,
            beta_sem=0.05,
            delta_vis=0.3,
            beta_vis=0.05,
            use_agreement_gate=False,
        )

    def _compute(self, similarity, support):
        routing = torch.softmax(similarity / 0.1, dim=-1)
        return self.module(similarity, routing, support)

    def test_open_slot(self):
        result = self._compute(
            torch.full((1, 1, 3), -0.8), torch.full((1, 1, 3), 0.01)
        )
        self.assertLess(float(result["event_gate"]), 1e-5)

    def test_known_event_slot(self):
        result = self._compute(
            torch.tensor([[[0.9, -0.5, -0.4]]]),
            torch.tensor([[[0.9, 0.1, 0.1]]]),
        )
        self.assertGreater(float(result["semantic_gate"]), 0.99)
        self.assertGreater(float(result["visual_gate"]), 0.99)
        self.assertGreater(float(result["event_gate"]), 0.98)

    def test_semantic_high_visual_low(self):
        result = self._compute(
            torch.tensor([[[0.9, -0.5]]]), torch.tensor([[[0.01, 0.9]]])
        )
        self.assertLess(float(result["event_gate"]), 0.01)

    def test_visual_high_semantic_low(self):
        result = self._compute(
            torch.tensor([[[-0.8, -0.9]]]), torch.tensor([[[0.9, 0.1]]])
        )
        self.assertLess(float(result["event_gate"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
