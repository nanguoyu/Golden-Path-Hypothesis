from __future__ import annotations

from types import SimpleNamespace

from flux import coarse_native
from flux.coarse_native import FluxNativeGateAdapter, FluxNativeGateConfig


def test_flux_native_wrapper_uses_locked_gate_and_preserves_decisions(
    monkeypatch,
) -> None:
    transformer = SimpleNamespace(teacache_decisions=[])
    pipe = SimpleNamespace(transformer=transformer)
    lifecycle = {"installed": False, "reset": False, "restored": False}

    def fake_install(
        installed_pipe,
        *,
        threshold,
        num_steps,
        first_enhance,
        backbone,
    ):
        assert installed_pipe is pipe
        assert (threshold, num_steps, first_enhance, backbone) == (
            0.38,
            3,
            1,
            "flux",
        )
        lifecycle["installed"] = True

        def teardown() -> None:
            lifecycle["restored"] = True

        return teardown

    def fake_reset(reset_pipe) -> None:
        assert reset_pipe is pipe
        lifecycle["reset"] = True
        transformer.teacache_decisions = []

    module = SimpleNamespace(
        install=fake_install,
        reset_per_image_state=fake_reset,
    )
    monkeypatch.setattr(coarse_native, "_load_native_module", lambda mode: module)

    adapter = FluxNativeGateAdapter(
        pipe,
        FluxNativeGateConfig(
            mode="teacache",
            num_steps=3,
            threshold=0.38,
        ),
    )
    adapter.install()
    adapter.reset(prompt_idx=7, seed=49)
    transformer.teacache_decisions = [
        {"step": 0, "u": 0, "force_full": True},
        {"step": 1, "u": 1, "force_full": False},
        {"step": 2, "u": 0, "force_full": True},
    ]

    decisions = adapter.decisions()
    assert lifecycle == {"installed": True, "reset": True, "restored": False}
    assert decisions["prompt_idx"] == 7
    assert decisions["seed"] == 49
    assert decisions["summary"]["n_cached"] == 1
    assert [row["action"] for row in decisions["per_step"]] == [
        "full",
        "cache",
        "full",
    ]
    assert "target_cache_count" not in decisions
    assert all("closure_intervened" not in row for row in decisions["per_step"])

    adapter.restore()
    assert lifecycle["restored"] is True
