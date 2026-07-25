import unittest

import torch
from torch import nn

from tools.conch_qv_lora import (
    GeneConditionedQVLoRAQKVLinear,
    QVLoRAQKVLinear,
    conch_qv_lora_gene_context,
    inject_conch_qv_lora,
)


class ToyAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        return self.qkv(x)


class ToyBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attn = ToyAttention(dim)

    def forward(self, x):
        return self.attn(x)


class ToyConch(nn.Module):
    def __init__(self, depth=4, dim=8):
        super().__init__()
        self.blocks = nn.Sequential(*[ToyBlock(dim) for _ in range(depth)])

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))
        return outputs


class ConchQvLoraTest(unittest.TestCase):
    def test_static_qv_lora_replaces_last_layers_and_starts_as_noop(self):
        torch.manual_seed(7)
        model = ToyConch(depth=4, dim=8)
        inputs = torch.randn(2, 5, 8)
        expected = model.blocks[-1].attn.qkv(inputs)

        summary = inject_conch_qv_lora(model, layers=2, rank=3)

        self.assertEqual(summary.replaced, 2)
        self.assertEqual(
            summary.target_names,
            ("blocks.2.attn.qkv", "blocks.3.attn.qkv"),
        )
        self.assertIsInstance(model.blocks[2].attn.qkv, QVLoRAQKVLinear)
        self.assertIsInstance(model.blocks[3].attn.qkv, QVLoRAQKVLinear)
        self.assertIsInstance(model.blocks[1].attn.qkv, nn.Linear)
        self.assertTrue(torch.equal(model.blocks[-1].attn.qkv(inputs), expected))

        loss = model.blocks[-1].attn.qkv(inputs).sum()
        loss.backward()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(".q_" in name or ".v_" in name for name in trainable))
        self.assertTrue(any("q_up.weight" in name for name in trainable))
        self.assertTrue(any("v_up.weight" in name for name in trainable))

    def test_gene_conditioned_qv_lora_requires_context(self):
        model = ToyConch(depth=2, dim=8)
        inject_conch_qv_lora(model, layers=1, rank=2, gene_dim=6)
        wrapped = model.blocks[-1].attn.qkv
        self.assertIsInstance(wrapped, GeneConditionedQVLoRAQKVLinear)
        inputs = torch.randn(3, 4, 8)

        with self.assertRaisesRegex(RuntimeError, "gene_context"):
            wrapped(inputs)

        with conch_qv_lora_gene_context(model, torch.randn(3, 6)):
            output = wrapped(inputs)
        self.assertEqual(output.shape, (3, 4, 24))


if __name__ == "__main__":
    unittest.main()
