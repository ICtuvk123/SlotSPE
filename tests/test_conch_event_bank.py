import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from models.event_grounding import FrozenEventBank
from tools.build_conch_event_bank import encode_prompt_ensemble
from tools.conch_utils import conch_tokenize


class ConchImportTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("conch"), "official CONCH is not installed")
    def test_official_api_import(self):
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

        self.assertTrue(callable(create_model_from_pretrained))
        self.assertTrue(callable(get_tokenizer))
        self.assertTrue(callable(tokenize))

    def test_tokenizer_compatibility_with_transformers_5(self):
        from conch.open_clip_custom import get_tokenizer, tokenize

        tokens = conch_tokenize(tokenize, get_tokenizer(), ["H&E renal histopathology"])
        self.assertEqual(tokens.shape, (1, 128))
        self.assertEqual(tokens[0, -1].item(), 0)


class PromptEnsembleTest(unittest.TestCase):
    def test_four_prompts_are_encoded_then_ensembled(self):
        prompts = [[f"event {event} prompt {prompt}" for prompt in range(4)] for event in range(3)]
        seen = []

        def fake_encode(texts):
            seen.extend(texts)
            rows = []
            for text in texts:
                value = float(sum(text.encode("utf-8")) % 13 + 1)
                rows.append(torch.tensor([value, value + 1, value + 2]))
            return torch.stack(rows)

        prompt_embeddings, event_embeddings = encode_prompt_ensemble(
            prompts, encode_batch=fake_encode, batch_size=5
        )
        self.assertEqual(prompt_embeddings.shape, (3, 4, 3))
        self.assertEqual(event_embeddings.shape, (3, 3))
        self.assertEqual(seen, [prompt for group in prompts for prompt in group])
        self.assertTrue(
            torch.allclose(event_embeddings.norm(dim=-1), torch.ones(3), atol=1e-6)
        )


class FrozenEventBankTest(unittest.TestCase):
    def _artifact(self):
        embeddings = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
        return {
            "model_name": "conch_ViT-B-16",
            "encoder_type": "CONCH text encoder",
            "feature_space": "conch_contrastive",
            "review_status": "machine_checked_pending_expert_review",
            "event_ids": [f"event_{i}" for i in range(5)],
            "event_names": [f"Event {i}" for i in range(5)],
            "event_embeddings": embeddings,
        }

    def test_loads_normalized_embeddings_as_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            torch.save(self._artifact(), path)
            bank = FrozenEventBank(path)
            self.assertEqual(bank.num_events, 5)
            self.assertEqual(bank.embedding_dim, 7)
            self.assertFalse(any(name == "_embeddings" for name, _ in bank.named_parameters()))
            self.assertTrue(any(name == "_embeddings" for name, _ in bank.named_buffers()))

    def test_trainable_mode_uses_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            torch.save(self._artifact(), path)
            bank = FrozenEventBank(path, trainable=True)
            self.assertTrue(isinstance(bank._embeddings, torch.nn.Parameter))
            self.assertTrue(
                torch.allclose(bank.embeddings.norm(dim=-1), torch.ones(5), atol=1e-6)
            )

    def test_loads_conch_v15_space(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact()
            artifact["feature_space"] = "conch_v1_5_contrastive"
            artifact["event_embeddings"] = torch.nn.functional.normalize(
                torch.randn(5, 768), dim=-1
            )
            path = Path(directory) / "bank_v15.pt"
            torch.save(artifact, path)
            bank = FrozenEventBank(path)
            self.assertEqual(bank.embedding_dim, 768)
            self.assertEqual(bank.feature_space, "conch_v1_5_contrastive")

    def test_loads_titan_text_space(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact()
            artifact["feature_space"] = "titan_text_768"
            artifact["event_embeddings"] = torch.nn.functional.normalize(
                torch.randn(5, 768), dim=-1
            )
            path = Path(directory) / "titan_bank.pt"
            torch.save(artifact, path)
            bank = FrozenEventBank(path)
            self.assertEqual(bank.embedding_dim, 768)
            self.assertEqual(bank.feature_space, "titan_text_768")


if __name__ == "__main__":
    unittest.main()
