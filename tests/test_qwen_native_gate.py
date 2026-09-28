from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from qwen_image.coarse_cache import QwenCoarseConfig, _decide_step, _new_state
from qwen_image.runner import _validate_fixed_cache_count


class _Block:
    def img_mod(self, temb: torch.Tensor) -> torch.Tensor:
        return torch.cat((temb, temb), dim=-1)

    def img_norm1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        _params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, torch.ones_like(hidden_states[:, 0])


def test_qwen_dynamic_gate_uses_native_threshold_decisions() -> None:
    state = _new_state(
        QwenCoarseConfig(
            mode="TeaCache",
            num_steps=6,
            first_enhance=1,
            teacache_thresh=1.0,
        ),
        num_layers=2,
        scheduler=None,
    )
    hidden = torch.ones((1, 3, 4))
    temb = torch.ones((1, 4))
    block = _Block()
    actions = []
    rows = []

    for step in range(6):
        state["current_step"] = step
        state["current_branch"] = "cond"
        row = _decide_step(
            state,
            hidden_states=hidden * (1.0 + step * 0.01),
            temb=temb,
            first_block=block,
        )
        rows.append(row)
        actions.append(row["action"])
        if row["action"] == "full":
            for branch in ("cond", "uncond"):
                state["branches"][branch]["previous_residual"] = torch.zeros_like(hidden)

    assert actions.count("cache") == 4
    assert actions[0] == "full"
    assert actions[-1] == "full"
    assert "budget" not in state
    assert all("closure_intervened" not in row["gate"] for row in rows)


def test_qwen_fixed_schedule_cache_count_is_checked_after_generation() -> None:
    args = SimpleNamespace(exact_cache_count=3)
    decisions = {"summary": {"n_cached": 3}}
    _validate_fixed_cache_count(args, decisions, prompt_idx=7)

    with pytest.raises(RuntimeError, match="3 != 4"):
        _validate_fixed_cache_count(
            SimpleNamespace(exact_cache_count=4),
            decisions,
            prompt_idx=7,
        )
