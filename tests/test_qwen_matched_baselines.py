from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lib.sencache import SenCacheSensitivityTable
from qwen_image.coarse_cache import QwenCoarseConfig, _decide_step, _new_state
from qwen_image.dicache import (
    QwenDiCacheConfig,
    install_qwen_dicache,
    qwen_dicache_decisions,
    restore_qwen_dicache,
)
from qwen_image.dpcache import (
    install_qwen_dpcache,
    qwen_dpcache_decisions,
    restore_qwen_dpcache,
)
from qwen_image.fine_scaffold import QwenFineConfig, _decide_step as _fine_decide
from qwen_image.meancache import (
    guided_velocity,
    install_qwen_meancache,
    qwen_meancache_decisions,
    restore_qwen_meancache,
)
from qwen_image.meancache_calibrate import trajectory_costs
from qwen_image.runner import (
    _decode_latents_to_pil,
    _fixed_cache_steps,
    _meancache_jvp_spans,
    parse_args,
    validate_args,
)
from qwen_image.sencache_calibrate import QwenSensitivityCollector


class _Block(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, hidden_states, encoder_hidden_states, **_kwargs):
        return encoder_hidden_states + self.scale, hidden_states * self.scale


class _BlockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(1.5), _Block(1.25)])

    def forward(self, hidden_states, encoder_hidden_states):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states


class _OutputTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states):
        self.calls += 1
        return hidden_states * 2.0 + encoder_hidden_states.mean()


def test_qwen_split_timing_decode_matches_pipeline_formula() -> None:
    class Vae:
        dtype = torch.float32
        config = SimpleNamespace(
            latents_mean=[0.5],
            latents_std=[2.0],
            z_dim=1,
        )

        @staticmethod
        def decode(latents, return_dict=False):
            assert return_dict is False
            return (latents + 1.0,)

    class Processor:
        @staticmethod
        def postprocess(image, output_type):
            assert output_type == "pil"
            return [image]

    pipe = SimpleNamespace(
        vae=Vae(),
        vae_scale_factor=1,
        image_processor=Processor(),
        _unpack_latents=lambda latents, *_args: latents,
    )
    latent = torch.ones((1, 1, 1, 1, 1))
    image = _decode_latents_to_pil(pipe, latent, 16, 16)[0]
    assert torch.equal(image, torch.full((1, 1, 1, 1), 3.5))


def _run_cfg_steps(transformer: nn.Module, num_steps: int) -> list[torch.Tensor]:
    outputs = []
    for step in range(num_steps):
        outputs.append(
            transformer(
                torch.full((1, 2, 2), float(step + 1)),
                torch.zeros((1, 1, 2)),
            )
        )
        outputs.append(
            transformer(
                torch.full((1, 2, 2), float(step + 11)),
                torch.ones((1, 1, 2)),
            )
        )
    return outputs


def test_qwen_fine_and_coarse_reuse_accept_explicit_fixed_schedules() -> None:
    fine_state = {
        "config": QwenFineConfig(
            mode="TaylorSeer_fine",
            num_steps=6,
            first_enhance=3,
            cache_steps=(3, 4),
        ),
        "activated_steps": [],
        "cache_counter": 0,
    }
    assert [_fine_decide(fine_state, step)["action"] for step in range(6)] == [
        "full",
        "full",
        "full",
        "cache",
        "cache",
        "full",
    ]

    coarse_state = _new_state(
        QwenCoarseConfig(
            mode="BudCache",
            num_steps=6,
            fixed_cache_steps=(2, 4),
        ),
        num_layers=2,
        scheduler=None,
    )
    assert coarse_state["locked_actions"] == {
        0: "full",
        1: "full",
        2: "cache",
        3: "full",
        4: "cache",
        5: "full",
    }
    reuse_state = _new_state(
        QwenCoarseConfig(
            mode="SeaCachePayload",
            payload_mode="reuse",
            num_steps=6,
            fixed_cache_steps=(2, 4),
        ),
        num_layers=2,
        scheduler=None,
    )
    assert reuse_state["locked_actions"] == coarse_state["locked_actions"]


def test_qwen_runner_accepts_fixed_reuse_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    schedule = tmp_path / "reuse.json"
    schedule.write_text(
        json.dumps(
            {
                "num_steps": 50,
                "cache_count": 2,
                "cache_steps": [3, 4],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-runner",
            "--mode",
            "SeaCachePayload",
            "--payload_mode",
            "reuse",
            "--output_dir",
            str(tmp_path / "out"),
            "--schedule_file",
            str(schedule),
        ],
    )
    args = parse_args()
    validate_args(args)
    assert _fixed_cache_steps(args) == (3, 4)
    assert args.exact_cache_count == 2


def test_qwen_sencache_uses_native_score_without_budget_closure() -> None:
    table = SenCacheSensitivityTable(
        path="memory",
        sha256="test",
        timesteps=np.asarray([1.0, 0.5]),
        j_x_norm=np.asarray([1.0, 1.0]),
        j_t_norm=np.asarray([1.0, 1.0]),
        metadata={},
    )
    state = _new_state(
        QwenCoarseConfig(
            mode="SenCache",
            num_steps=4,
            first_enhance=1,
            sencache_sensitivity_path="unused-in-unit-test",
            sencache_threshold_main=1.0,
            sencache_threshold_start=1.0,
            sencache_threshold_scale=1.0,
        ),
        num_layers=2,
        scheduler=None,
        sencache_table=table,
    )
    latent = torch.ones((1, 2, 2))
    for branch in ("cond", "uncond"):
        state["branches"][branch]["previous_residual"] = torch.zeros_like(latent)
        state["branches"][branch]["sencache_anchor_latent"] = latent.clone()
        state["branches"][branch]["sencache_anchor_timestep"] = 1.0
        state["branches"][branch]["sencache_anchor_step"] = 0
    state["current_step"] = 1
    state["current_branch"] = "cond"
    state["gate_latent"] = latent
    state["gate_timestep"] = 1.0
    row = _decide_step(
        state,
        hidden_states=latent,
        temb=torch.ones((1, 2)),
        first_block=object(),
    )
    assert row["action"] == "cache"
    assert row["gate"]["score"] == 0.0
    assert "closure_intervened" not in row["gate"]


def test_qwen_dicache_shares_action_and_isolates_branch_payloads() -> None:
    transformer = _BlockTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    install_qwen_dicache(
        pipe,
        QwenDiCacheConfig(
            num_steps=4,
            threshold=100.0,
            ret_ratio=0.0,
            probe_depth=1,
            true_cfg=True,
        ),
    )
    outputs = _run_cfg_steps(transformer, 4)
    decisions = qwen_dicache_decisions(pipe)

    assert [row["action"] for row in decisions["steps"]] == [
        "full",
        "cache",
        "cache",
        "full",
    ]
    assert not torch.equal(outputs[2], outputs[3])
    assert all(set(row["branches"]) == {"cond", "uncond"} for row in decisions["steps"])
    restore_qwen_dicache(pipe)
    assert all(
        "forward" not in block.__dict__ for block in transformer.transformer_blocks
    )


def test_qwen_dpcache_exact_schedule_and_cfg_histories() -> None:
    transformer = _BlockTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    install_qwen_dpcache(
        pipe,
        cache_steps=(3,),
        num_steps=5,
        order=2,
        true_cfg=True,
    )
    _run_cfg_steps(transformer, 5)
    decisions = qwen_dpcache_decisions(pipe)

    assert [row["action"] for row in decisions["steps"]] == [
        "full",
        "full",
        "full",
        "cache",
        "full",
    ]
    assert decisions["summary"]["n_cached"] == 1
    assert set(decisions["steps"][3]["branches"]) == {"cond", "uncond"}
    restore_qwen_dpcache(pipe)
    assert all(
        "forward" not in block.__dict__ for block in transformer.transformer_blocks
    )


def test_qwen_meancache_skips_both_cfg_forwards_on_fixed_cache_step() -> None:
    transformer = _OutputTransformer()
    scheduler = SimpleNamespace(sigmas=torch.linspace(1.0, 0.0, 4))
    pipe = SimpleNamespace(transformer=transformer, scheduler=scheduler)
    install_qwen_meancache(
        pipe,
        cache_steps=(1,),
        num_steps=3,
        jvp_span=2,
        true_cfg=True,
    )
    outputs = _run_cfg_steps(transformer, 3)
    decisions = qwen_meancache_decisions(pipe)

    assert transformer.calls == 4
    assert torch.equal(outputs[2], outputs[3])
    assert [row["action"] for row in decisions["steps"]] == [
        "full",
        "cache",
        "full",
    ]
    assert decisions["summary"]["n_cached"] == 1
    assert decisions["prediction_target"] == "post_true_cfg_guided_velocity"
    restore_qwen_meancache(pipe)
    assert "forward" not in transformer.__dict__


def test_qwen_runner_accepts_fixed_k_only_for_fixed_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-runner",
            "--mode",
            "BudCache",
            "--output_dir",
            str(tmp_path / "bud"),
            "--cache_steps",
            "1,2",
            "--exact_cache_count",
            "2",
        ],
    )
    validate_args(parse_args())

    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-runner",
            "--mode",
            "SeaCache",
            "--output_dir",
            str(tmp_path / "sea"),
            "--exact_cache_count",
            "2",
        ],
    )
    with pytest.raises(SystemExit, match="explicit fixed schedule"):
        validate_args(parse_args())


def test_qwen_runner_loads_complete_schedule_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    schedule = tmp_path / "mean.json"
    schedule.write_text(
        json.dumps(
            {
                "num_steps": 50,
                "cache_count": 2,
                "cache_steps": [3, 4],
                "jvp_spans": {"3": 2, "4": 3},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-runner",
            "--mode",
            "MeanCache",
            "--output_dir",
            str(tmp_path / "out"),
            "--schedule_file",
            str(schedule),
        ],
    )
    args = parse_args()
    validate_args(args)
    assert _fixed_cache_steps(args) == (3, 4)
    assert args.exact_cache_count == 2
    assert _meancache_jvp_spans(args) == {3: 2, 4: 3}


def test_qwen_runner_rejects_conflicting_schedule_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps({"cache_count": 2, "cache_steps": [3, 4]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-runner",
            "--mode",
            "DPCache",
            "--output_dir",
            str(tmp_path / "out"),
            "--schedule_file",
            str(schedule),
            "--cache_steps",
            "3,5",
        ],
    )
    with pytest.raises(SystemExit, match="disagree"):
        validate_args(parse_args())


def test_qwen_guided_velocity_matches_true_cfg_norm_rescale() -> None:
    cond = torch.tensor([[[3.0, 4.0]]])
    uncond = torch.tensor([[[1.0, 2.0]]])
    result = guided_velocity(cond, uncond, true_cfg_scale=4.0)
    combined = uncond + 4.0 * (cond - uncond)
    expected = combined * (
        torch.linalg.vector_norm(cond, dim=-1, keepdim=True)
        / torch.linalg.vector_norm(combined, dim=-1, keepdim=True)
    )
    assert torch.allclose(result, expected)


def test_qwen_meancache_calibration_cost_is_zero_for_linear_trajectory() -> None:
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    latents = [
        torch.full((1, 2, 2), float(sigma))
        for sigma in sigmas
    ]
    velocities = [torch.ones((1, 2, 2)) for _ in range(4)]
    sums, counts = trajectory_costs(
        latents,
        velocities,
        sigmas,
        (2,),
        max_edge_gap=4,
    )
    assert counts[0, 2, 4] == 1
    assert sums[0, 2, 4] == pytest.approx(0.0)


class _SensitivityTransformer(nn.Module):
    def forward(
        self,
        *,
        hidden_states,
        timestep,
        scale=2.0,
        return_dict=False,
    ):
        del return_dict
        return (hidden_states * scale + timestep.reshape(-1, 1, 1),)


def test_qwen_sencache_collector_produces_adjacent_directional_rows() -> None:
    transformer = _SensitivityTransformer()
    collector = QwenSensitivityCollector(transformer, num_steps=3)
    collector.install()
    try:
        for step in range(3):
            latent = torch.full((1, 2, 2), float(step + 1))
            timestep = torch.tensor([1.0 - 0.2 * step])
            transformer(
                hidden_states=latent,
                timestep=timestep,
                return_dict=False,
            )
            transformer(
                hidden_states=latent + 10.0,
                timestep=timestep,
                return_dict=False,
            )
        rows = collector.rows(prompt_id=7, seed=11)
    finally:
        collector.restore()
    assert len(rows) == 3
    assert [row["step_index"] for row in rows] == [0, 1, 2]
    assert all(row["J_x_directional"] == pytest.approx(2.0) for row in rows)
    assert all(row["J_t_directional"] > 0.0 for row in rows)
