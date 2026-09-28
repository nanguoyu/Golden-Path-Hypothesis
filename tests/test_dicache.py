from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from flux.dicache_native import FluxDiCacheAdapter, FluxDiCacheConfig
from lib.dicache import aligned_residual


class _Block(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **_kwargs):
        return encoder_hidden_states + 1.0, hidden_states + 1.0


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.single_transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, hidden_states, encoder_hidden_states):
        for block in [*self.transformer_blocks, *self.single_transformer_blocks]:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states


def test_aligned_residual_uses_two_anchor_trajectory() -> None:
    residual, gamma = aligned_residual(
        torch.tensor([3.0]),
        [torch.tensor([2.0]), torch.tensor([4.0])],
        [torch.tensor([1.0]), torch.tensor([2.0])],
    )
    assert gamma == 1.5
    assert torch.equal(residual, torch.tensor([5.0]))


def test_flux_dicache_native_follows_gate_and_keeps_probe_compute() -> None:
    transformer = _Transformer()
    adapter = FluxDiCacheAdapter(
        SimpleNamespace(transformer=transformer),
        FluxDiCacheConfig(
            num_steps=7,
            threshold=100.0,
            ret_ratio=0.0,
            probe_depth=1,
        ),
    )
    adapter.install()
    for step in range(7):
        transformer(
            torch.full((1, 2, 2), float(step + 1)),
            torch.zeros((1, 1, 2)),
        )

    assert [row["action"] for row in adapter.records].count("cache") == 5
    assert adapter.records[0]["action"] == "full"
    assert adapter.records[-1]["action"] == "full"
    assert all(
        row["original_block_calls"] == 1
        for row in adapter.records
        if row["action"] == "cache"
    )
    decisions = adapter.decisions()
    assert decisions["summary"]["n_cached"] == 5
    assert "target_cache_count" not in decisions
    assert all("closure_intervened" not in row for row in adapter.records)

    adapter.restore()
    assert all("forward" not in block.__dict__ for block in transformer.transformer_blocks)
