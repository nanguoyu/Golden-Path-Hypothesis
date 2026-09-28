from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from flux.dpcache_exact import FluxDPCacheAdapter
from lib.dpcache import (
    calibration_cost_tensor,
    derivatives_from_history,
    predict_derivatives,
    select_full_steps,
    update_derivatives,
)


def test_nonuniform_taylor_history_matches_official_discrete_update() -> None:
    history = {}
    previous_step = None
    for step in (0, 1, 2):
        value = torch.tensor([float(step * step)])
        history = update_derivatives(
            history,
            value,
            step_gap=1 if previous_step is None else step - previous_step,
            order=2,
        )
        previous_step = step
    predicted = predict_derivatives(history, step_offset=1, order=2)
    # DPCache recursively differences already-normalized derivatives, then
    # applies the Taylor factorial at prediction time.
    assert torch.allclose(predicted, torch.tensor([8.0]))


def test_calibration_cost_tensor_has_official_reachable_edges() -> None:
    features = [
        torch.full((1, 2, 2), float(step * step))
        for step in range(6)
    ]
    derivatives = derivatives_from_history(
        [(0, features[0]), (2, features[2]), (4, features[4])],
        order=2,
    )
    assert set(derivatives) == {0, 1, 2}

    costs = calibration_cost_tensor(features, order=2)
    assert costs.shape == (6, 7, 7)
    assert np.isfinite(costs[1, 2, 6])
    assert np.isinf(costs[0, 2, 6])


def test_path_dp_returns_exact_mandatory_full_steps() -> None:
    costs = np.ones((6, 7, 7), dtype=np.float64)
    full = select_full_steps(
        costs,
        total_steps=6,
        full_count=4,
        first_full_steps=2,
        last_full_steps=1,
        max_jump_fraction=1.0,
    )
    assert len(full) == 4
    assert {0, 1, 5}.issubset(full)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, **_kwargs):
        self.calls += 1
        return encoder_hidden_states + 1.0, hidden_states + 1.0


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])
        self.single_transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, hidden_states, encoder_hidden_states):
        for block in [*self.transformer_blocks, *self.single_transformer_blocks]:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states


def test_dpcache_adapter_skips_all_blocks_on_fixed_cache_step() -> None:
    transformer = _Transformer()
    adapter = FluxDPCacheAdapter(
        SimpleNamespace(transformer=transformer),
        cache_steps=(3,),
        num_steps=5,
        order=2,
    )
    adapter.install()
    for step in range(5):
        transformer(
            torch.full((1, 2, 2), float(step)),
            torch.zeros((1, 1, 2)),
        )

    assert [row["action"] for row in adapter.records] == [
        "full",
        "full",
        "full",
        "cache",
        "full",
    ]
    assert [row["original_block_calls"] for row in adapter.records] == [2, 2, 2, 0, 2]
    assert set(adapter.histories) == {
        "double_encoder",
        "double_hidden",
        "single_encoder",
        "single_hidden",
    }
    assert all(history for history in adapter.histories.values())
    assert adapter.decisions()["summary"]["n_cached"] == 1
    adapter.restore()
    assert all("forward" not in block.__dict__ for block in transformer.transformer_blocks)
    assert all("forward" not in block.__dict__ for block in transformer.single_transformer_blocks)
