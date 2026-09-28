"""Research-only fine-grained FoCa predictor for diffusers FLUX.

This is an auditable FoCa interpretation on top of the repo's existing
114-slot fine scaffold.  It is not an official FoCa reproduction because the
paper does not specify FLUX hook points or all derivative/calibration details.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from lib.flux_fine_scaffold import CacheHistory, install_fine_cache, reset_per_image_state_fine
from lib.foca import FoCaConfig, commit_prediction, predict_foca, update_full_history


class FoCaFinePredictor:
    """FoCa predictor adapter for ``lib.flux_fine_scaffold``."""

    max_order: int = 0

    def __init__(self, config: FoCaConfig, *, expected_slots: int = 114) -> None:
        self.config = config
        self.expected_slots = int(expected_slots)
        self.current_step = 0
        self._decision: Optional[dict[str, Any]] = None

    def begin_step(self, decision: dict[str, Any]) -> None:
        self._decision = decision
        self.current_step = int(decision.get("step", 0))
        decision.update({
            "expected_slots": int(self.expected_slots),
            "slots_ready": 0,
            "slots_updated": 0,
            "slots_predicted": 0,
            "slots_missing": 0,
            "slots_fallback": 0,
            "foca_target": "fine_pregate_114",
            "foca_heun_variant": self.config.heun_variant,
            "foca_history_policy": self.config.history_policy,
            "foca_derivative": self.config.derivative,
            "foca_h": float(self.config.h),
            "foca_bdf2_available_count": 0,
            "foca_heun_available_count": 0,
            "foca_nan_inf_count": 0,
            "foca_fallback_reasons": {},
        })

    def _record_update(self) -> None:
        if self._decision is not None:
            self._decision["slots_updated"] = int(self._decision.get("slots_updated", 0)) + 1
            self._decision["slots_ready"] = max(
                int(self._decision.get("slots_ready", 0)),
                int(self._decision.get("slots_updated", 0)),
            )

    def _record_prediction(self, fields: dict[str, Any]) -> None:
        if self._decision is None:
            return
        d = self._decision
        d["slots_predicted"] = int(d.get("slots_predicted", 0)) + 1
        if fields.get("foca_bdf2_available"):
            d["foca_bdf2_available_count"] = int(d.get("foca_bdf2_available_count", 0)) + 1
        if fields.get("foca_heun_available"):
            d["foca_heun_available_count"] = int(d.get("foca_heun_available_count", 0)) + 1
        d["foca_nan_inf_count"] = int(d.get("foca_nan_inf_count", 0)) + int(
            fields.get("foca_nan_inf_count", 0) or 0
        )
        reason = fields.get("foca_fallback_reason")
        if reason:
            d["slots_fallback"] = int(d.get("slots_fallback", 0)) + 1
            reasons = dict(d.get("foca_fallback_reasons") or {})
            reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
            d["foca_fallback_reasons"] = reasons
        # Keep one representative set of step-local fields for audit.
        for key in (
            "foca_prev_roll_step",
            "foca_latest_roll_step",
            "foca_target_step",
            "foca_latest_full_step",
            "foca_prev_full_step",
            "foca_dry_run_steps",
            "foca_prediction_kind",
        ):
            if key in fields:
                d[key] = fields[key]

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        self._record_update()
        return update_full_history(prev_history, feature, step=int(self.current_step))

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        pred, fields = predict_foca(
            history,
            current_step=int(self.current_step),
            config=self.config,
        )
        self._record_prediction(fields)
        return pred

    def commit_prediction(
        self,
        history: CacheHistory,
        step_offset: int,
        prediction: torch.Tensor,
    ) -> CacheHistory:
        if self.config.history_policy != "recursive":
            return dict(history)
        return commit_prediction(history, prediction, step=int(self.current_step))


def install(
    pipe,
    *,
    interval: int = 7,
    first_enhance: int = 3,
    num_steps: int,
    heun_variant: str = "paper_literal",
    history_policy: str = "recursive",
    derivative: str = "step_backward",
    h: float = 1.0,
    log_norms: bool = False,
) -> Callable[[], None]:
    """Install the fine 114-slot FoCa research variant."""
    config = FoCaConfig(
        heun_variant=str(heun_variant),
        history_policy=str(history_policy),
        derivative=str(derivative),
        h=float(h),
        log_norms=bool(log_norms),
    )
    predictor = FoCaFinePredictor(config)
    teardown_fine = install_fine_cache(
        pipe,
        predictor=predictor,
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
        method_tag=f"foca_fine_{config.heun_variant}_{config.history_policy}",
    )
    tr = pipe.transformer
    tr._foca_config = config
    tr._foca_target = "fine_pregate_114"

    def teardown() -> None:
        teardown_fine()
        for attr in ("_foca_config", "_foca_target"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass

    return teardown


def reset_per_image_state(
    pipe,
    *,
    action_steps: Optional[set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    reset_per_image_state_fine(pipe, action_steps=action_steps, prompt_idx=prompt_idx)

