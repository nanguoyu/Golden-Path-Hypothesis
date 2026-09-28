from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flux.meancache_exact import FluxMeanCacheAdapter
from flux.sp_cross_runner import (
    ORACLE_CACHE_MODE,
    FluxSPCrossDiCacheAdapter,
    build_adapter as flux_build_adapter,
)
from lib.dicache import aligned_residual
from lib.hermite import hermite_update, hicache_predict
from lib.taylor import taylor_predict
from qwen_image.meancache import QwenMeanCacheAdapter, guided_velocity
from qwen_image.sp_cross_runner import (
    QwenSPCrossDiCacheAdapter,
    QwenSPCrossResidualAdapter,
    build_adapter as qwen_build_adapter,
)


# The fake blocks apply h -> 2h + 1, so a stack of three turns h into 8h + 7:
# the whole-transformer residual is 7h + 7 and the depth-1 probe residual h + 1.
def _residual(hidden: torch.Tensor) -> torch.Tensor:
    return 7.0 * hidden + 7.0


def _probe_residual(hidden: torch.Tensor) -> torch.Tensor:
    return hidden + 1.0


class _Block(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **_kwargs):
        return encoder_hidden_states, hidden_states * 2.0 + 1.0


class _QwenTransformer(nn.Module):
    def __init__(self, n_blocks: int = 3) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])

    def forward(self, hidden_states, encoder_hidden_states):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states


class _FluxTransformer(nn.Module):
    def __init__(self) -> None:
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


def _drive(transformer: nn.Module, num_steps: int, width: int = 2):
    """Run one synthetic trajectory; return per-step inputs and payloads."""

    inputs: dict[int, torch.Tensor] = {}
    payloads: dict[int, torch.Tensor] = {}
    for step in range(num_steps):
        hidden = torch.full((1, width), float(step) + 1.0)
        inputs[step] = hidden
        output = transformer(hidden, torch.zeros((1, 1)))
        payloads[step] = output - hidden
    return inputs, payloads


def _reference_history(
    inputs: dict[int, torch.Tensor], full_steps: tuple[int, ...], max_order: int = 2
):
    """Rebuild the residual history the way the locked updater does."""

    history: dict[int, torch.Tensor] = {}
    last: int | None = None
    for step in full_steps:
        gap = 1 if last is None else step - last
        history = hermite_update(
            history, _residual(inputs[step]), step_gap=gap, max_order=max_order
        )
        last = step
    return history


def _residual_adapter(payload: str, cache_steps, num_steps: int, true_cfg: bool = False):
    transformer = _QwenTransformer()
    adapter = QwenSPCrossResidualAdapter(
        SimpleNamespace(transformer=transformer),
        cache_steps=cache_steps,
        payload=payload,
        num_steps=num_steps,
        true_cfg=true_cfg,
    )
    adapter.install()
    adapter.reset(prompt_idx=0, seed=7)
    return transformer, adapter


@pytest.mark.parametrize("payload", ["reuse", "taylor_o1", "hermite_o2"])
def test_residual_payloads_match_the_locked_predictors(payload: str) -> None:
    transformer, adapter = _residual_adapter(payload, (3, 4), 6)
    inputs, payloads = _drive(transformer, 6)
    adapter.restore()

    history = _reference_history(inputs, (0, 1, 2))
    for step in (3, 4):
        offset = step - 2
        if payload == "reuse":
            expected = history[0]
        elif payload == "taylor_o1":
            expected = taylor_predict(history, step_offset=offset, max_order=1)
        else:
            expected = hicache_predict(
                history, step_offset=offset, sigma=0.5, max_order=2
            )
        assert torch.allclose(payloads[step], expected)

    decisions = adapter.decisions()
    assert decisions["summary"] == {
        "n_total": 6,
        "n_full": 4,
        "n_cached": 2,
        "cache_ratio": 2 / 6,
    }
    assert decisions["steps"][3]["branches"]["cond"]["step_offset"] == 1
    assert decisions["steps"][4]["branches"]["cond"]["step_offset"] == 2


def test_residual_payload_full_steps_are_bit_exact() -> None:
    transformer, adapter = _residual_adapter("hermite_o2", (3,), 5)
    inputs, payloads = _drive(transformer, 5)
    adapter.restore()
    for step in (0, 1, 2, 4):
        assert torch.allclose(payloads[step], _residual(inputs[step]))


def test_residual_history_uses_the_gap_between_full_steps() -> None:
    # Full steps 0, 1, 4 -> the step-4 update must divide by a gap of 3.
    transformer, adapter = _residual_adapter("taylor_o1", (2, 3, 5), 7)
    inputs, payloads = _drive(transformer, 7)
    adapter.restore()
    history = _reference_history(inputs, (0, 1, 4))
    assert torch.allclose(
        payloads[5], taylor_predict(history, step_offset=1, max_order=1)
    )


@pytest.mark.parametrize(
    ("cache_step", "expected_orders"),
    [
        (1, {"reuse": 0, "taylor_o1": 0, "hermite_o2": 0}),
        (2, {"reuse": 0, "taylor_o1": 1, "hermite_o2": 1}),
        (3, {"reuse": 0, "taylor_o1": 1, "hermite_o2": 2}),
    ],
)
def test_warmup_falls_back_to_the_highest_available_order(
    cache_step: int, expected_orders: dict[str, int]
) -> None:
    for payload, expected_order in expected_orders.items():
        transformer, adapter = _residual_adapter(payload, (cache_step,), 5)
        inputs, payloads = _drive(transformer, 5)
        adapter.restore()
        row = adapter.decisions()["steps"][cache_step]["branches"]["cond"]
        assert row["order_used"] == expected_order
        if expected_order == 0:
            assert torch.allclose(payloads[cache_step], _residual(inputs[cache_step - 1]))


def test_true_cfg_branches_keep_separate_histories() -> None:
    transformer = _QwenTransformer()
    adapter = QwenSPCrossResidualAdapter(
        SimpleNamespace(transformer=transformer),
        cache_steps=(2,),
        payload="reuse",
        num_steps=4,
        true_cfg=True,
    )
    adapter.install()
    adapter.reset(prompt_idx=0, seed=7)
    inputs: dict[tuple[int, str], torch.Tensor] = {}
    payloads: dict[tuple[int, str], torch.Tensor] = {}
    for step in range(4):
        for branch, bias in (("cond", 0.0), ("uncond", 100.0)):
            hidden = torch.full((1, 2), float(step) + 1.0 + bias)
            inputs[(step, branch)] = hidden
            payloads[(step, branch)] = (
                transformer(hidden, torch.zeros((1, 1))) - hidden
            )
    adapter.restore()

    assert torch.allclose(payloads[(2, "cond")], _residual(inputs[(1, "cond")]))
    assert torch.allclose(payloads[(2, "uncond")], _residual(inputs[(1, "uncond")]))
    row = adapter.decisions()["steps"][2]
    assert set(row["branches"]) == {"cond", "uncond"}
    assert row["u"] == 1


def test_residual_adapter_rejects_a_schedule_that_caches_step_zero() -> None:
    with pytest.raises(ValueError, match="forced-full"):
        QwenSPCrossResidualAdapter(
            SimpleNamespace(transformer=_QwenTransformer()),
            cache_steps=(0, 2),
            payload="reuse",
            num_steps=4,
        )


@pytest.mark.parametrize("backend", ["flux", "qwen"])
def test_di_two_anchor_matches_aligned_residual(backend: str) -> None:
    if backend == "flux":
        transformer = _FluxTransformer()
        adapter = FluxSPCrossDiCacheAdapter(
            SimpleNamespace(transformer=transformer),
            cache_steps=(2, 3),
            num_steps=5,
            probe_depth=1,
        )
        adapter.install()
    else:
        transformer = _QwenTransformer()
        adapter = QwenSPCrossDiCacheAdapter(
            SimpleNamespace(transformer=transformer),
            cache_steps=(2, 3),
            num_steps=5,
            probe_depth=1,
            true_cfg=False,
        )
        adapter.install()
        adapter.reset(prompt_idx=0, seed=7)

    inputs, payloads = _drive(transformer, 5)

    residual_history = [_residual(inputs[0]), _residual(inputs[1])]
    probe_history = [_probe_residual(inputs[0]), _probe_residual(inputs[1])]
    expected, gamma = aligned_residual(
        _probe_residual(inputs[2]), residual_history, probe_history
    )
    assert torch.allclose(payloads[2], expected)
    assert 1.0 <= gamma <= 1.5

    # A second consecutive cache step keeps the SAME two anchors: cache steps
    # never append to the histories, and the shallow probe of a cache step is
    # not an anchor either.
    expected_next, gamma_next = aligned_residual(
        _probe_residual(inputs[3]), residual_history, probe_history
    )
    assert torch.allclose(payloads[3], expected_next)

    if backend == "flux":
        records = {row["step"]: row for row in adapter.records}
        # A cached step still pays for the shallow probe, and only that.
        assert records[2]["original_block_calls"] == 1
        assert records[0]["original_block_calls"] == 3
        assert records[2]["gamma"] == pytest.approx(gamma)
        assert records[3]["original_block_calls"] == 1
        assert records[3]["gamma"] == pytest.approx(gamma_next)
        assert records[3]["n_anchors"] == 2
        # Steps 0, 1 and 4 are the full steps: the two anchors left at the end
        # are the last two of those, never anything produced at a cache step.
        assert [tensor.tolist() for tensor in adapter.residual_history] == [
            _residual(inputs[1]).tolist(),
            _residual(inputs[4]).tolist(),
        ]
    else:
        row = adapter.decisions()["steps"][2]["branches"]["cond"]
        assert row["original_block_calls"] == 1
        assert row["gamma"] == pytest.approx(gamma)
        next_row = adapter.decisions()["steps"][3]["branches"]["cond"]
        assert next_row["original_block_calls"] == 1
        assert next_row["gamma"] == pytest.approx(gamma_next)
        assert next_row["n_anchors"] == 2

    adapter.restore()
    assert all("forward" not in block.__dict__ for block in transformer.transformer_blocks)


@pytest.mark.parametrize("backend", ["flux", "qwen"])
def test_di_two_anchor_single_anchor_falls_back_to_reuse(backend: str) -> None:
    if backend == "flux":
        transformer = _FluxTransformer()
        adapter = FluxSPCrossDiCacheAdapter(
            SimpleNamespace(transformer=transformer),
            cache_steps=(1,),
            num_steps=3,
            probe_depth=1,
        )
        adapter.install()
        gamma_of = lambda: adapter.records[1]["gamma"]  # noqa: E731
    else:
        transformer = _QwenTransformer()
        adapter = QwenSPCrossDiCacheAdapter(
            SimpleNamespace(transformer=transformer),
            cache_steps=(1,),
            num_steps=3,
            probe_depth=1,
            true_cfg=False,
        )
        adapter.install()
        adapter.reset(prompt_idx=0, seed=7)
        gamma_of = lambda: adapter.decisions()["steps"][1]["branches"]["cond"]["gamma"]  # noqa: E731

    inputs, payloads = _drive(transformer, 3)
    assert torch.allclose(payloads[1], _residual(inputs[0]))
    assert gamma_of() is None
    adapter.restore()


def test_qwen_mean_avg_vel_uses_the_locked_meancache_jvp() -> None:
    class _VelocityTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.transformer_blocks = nn.ModuleList([nn.Identity()])

        def forward(self, hidden_states, **_kwargs):
            self.calls += 1
            return hidden_states * 0.25 + float(self.calls)

    sigmas = torch.linspace(1.0, 0.0, 7)
    transformer = _VelocityTransformer()
    pipe = SimpleNamespace(
        transformer=transformer, scheduler=SimpleNamespace(sigmas=sigmas)
    )
    args = argparse.Namespace(
        payload="mean_avg_vel",
        num_steps=6,
        meancache_jvp_span=2,
        true_cfg_scale=4.0,
        hicache_sigma=0.5,
        hicache_max_order=2,
        taylorseer_max_order=1,
        dicache_probe_depth=1,
    )
    adapter = qwen_build_adapter(pipe, args, (3, 4))
    assert isinstance(adapter, QwenMeanCacheAdapter)
    adapter.install()
    adapter.reset(prompt_idx=0, seed=7)

    latents: dict[tuple[int, str], torch.Tensor] = {}
    outputs: dict[tuple[int, str], torch.Tensor] = {}
    for step in range(6):
        for branch, bias in (("cond", 0.0), ("uncond", 0.5)):
            hidden = torch.full((1, 2, 2), float(step) + bias)
            latents[(step, branch)] = hidden
            outputs[(step, branch)] = transformer(hidden_states=hidden)

    velocities = [
        guided_velocity(
            outputs[(step, "cond")], outputs[(step, "uncond")], true_cfg_scale=4.0
        )
        for step in range(3)
    ]
    reference = len(velocities) - min(args.meancache_jvp_span, len(velocities))
    delta = sigmas[3].to(torch.float32) - sigmas[reference].to(torch.float32)
    average = (
        latents[(3, "cond")].to(torch.float32)
        - latents[(reference, "cond")].to(torch.float32)
    ) / delta
    jvp = (average - velocities[reference].to(torch.float32)) / delta
    expected = velocities[-1].to(torch.float32) + (
        sigmas[4] - sigmas[3]
    ).to(torch.float32) * jvp

    assert torch.allclose(outputs[(3, "cond")], expected.to(velocities[-1].dtype))
    assert torch.allclose(outputs[(3, "uncond")], outputs[(3, "cond")])
    # 4 full steps x 2 CFG branches; the 2 cached steps call nothing.
    assert transformer.calls == 8
    assert adapter.decisions()["summary"]["n_cached"] == 2
    adapter.restore()


def test_flux_build_adapter_routes_to_the_locked_meancache_adapter() -> None:
    args = argparse.Namespace(
        payload="mean_avg_vel",
        num_steps=5,
        meancache_jvp_span=4,
        hicache_sigma=0.5,
        hicache_max_order=2,
        taylorseer_max_order=1,
        dicache_probe_depth=1,
    )
    pipe = SimpleNamespace(
        transformer=_FluxTransformer(),
        scheduler=SimpleNamespace(sigmas=torch.linspace(1.0, 0.0, 6)),
    )
    adapter = flux_build_adapter(pipe, args, (3,))
    assert isinstance(adapter, FluxMeanCacheAdapter)
    assert adapter.cache_steps == (3,)
    assert adapter.jvp_span == 4

    args.payload = "di_two_anchor"
    assert isinstance(
        flux_build_adapter(pipe, args, (3,)), FluxSPCrossDiCacheAdapter
    )


def test_flux_residual_payloads_map_to_the_locked_oracle_modes() -> None:
    assert ORACLE_CACHE_MODE == {
        "reuse": "seacache",
        "taylor_o1": "taylorseer",
        "hermite_o2": "hicache",
    }


def test_flux_oracle_modes_exist_upstream() -> None:
    oracle_runner = pytest.importorskip(
        "flux.oracle_runner", reason="needs diffusers"
    )
    assert set(ORACLE_CACHE_MODE.values()) <= set(oracle_runner.CACHE_MODES)


def test_decisions_reject_a_cache_count_mismatch() -> None:
    transformer, adapter = _residual_adapter("reuse", (2,), 4)
    _drive(transformer, 2)  # the trajectory stops before the scheduled cache step
    adapter.restore()
    with pytest.raises(RuntimeError, match="expected K=1"):
        adapter.decisions()


# --------------------------------------------------------------------------
# per-edge jvp_spans plumbing (both runners)
# --------------------------------------------------------------------------

def _spans_file(tmp_path, payload):
    import json
    path = tmp_path / "meancache_solution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("load", [
    pytest.param(lambda: __import__("flux.sp_cross_runner", fromlist=["x"]).load_meancache_spans, id="flux"),
    pytest.param(lambda: __import__("qwen_image.sp_cross_runner", fromlist=["x"]).load_meancache_spans, id="qwen"),
])
class TestMeanCacheSpansLoader:
    """The frozen MeanCache solutions carry per-edge spans; a scalar span
    cannot express them, so without this loader the W1b homologous cell is not
    the frozen method's payload."""

    def test_spans_load_and_key_by_int_step(self, load, tmp_path):
        from argparse import Namespace
        path = _spans_file(tmp_path, {"cache_steps": [3, 7], "jvp_spans": {"3": 2, "7": 5}})
        assert load()(Namespace(meancache_jvp_spans=path), [7, 3]) == {3: 2, 7: 5}

    def test_no_file_means_the_scalar_span(self, load):
        from argparse import Namespace
        assert load()(Namespace(meancache_jvp_spans=None), [3, 7]) is None
        assert load()(Namespace(), [3, 7]) is None

    def test_spans_solved_for_another_schedule_are_refused(self, load, tmp_path):
        """Spans are half of the solved path; applied to different bits the
        cell is neither the method nor the payload -- the attribution ambiguity
        P4 forbids."""
        from argparse import Namespace
        path = _spans_file(tmp_path, {"cache_steps": [3, 7], "jvp_spans": {"3": 2}})
        with pytest.raises(SystemExit, match="do not transfer"):
            load()(Namespace(meancache_jvp_spans=path), [3, 8])

    def test_a_file_without_spans_is_refused(self, load, tmp_path):
        from argparse import Namespace
        path = _spans_file(tmp_path, {"cache_steps": [3, 7]})
        with pytest.raises(SystemExit, match="carries no jvp_spans"):
            load()(Namespace(meancache_jvp_spans=path), [3, 7])
