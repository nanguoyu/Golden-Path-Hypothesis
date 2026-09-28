"""FoCa-style predictor helpers for cache experiments.

This module intentionally implements an auditable FoCa interpretation, not an
official reproduction.  The FoCa paper leaves the concrete FLUX hook points,
endpoint derivative, and Heun anchor semantics under-specified.  Callers must
record the selected variants in run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch


FOCA_HEUN_VARIANTS = ("paper_literal", "anchored_prose", "none")
FOCA_HISTORY_POLICIES = ("recursive", "full_refresh_only")
FOCA_DERIVATIVES = ("step_backward",)


History = Dict[str, Any]


@dataclass(frozen=True)
class FoCaConfig:
    heun_variant: str = "paper_literal"
    history_policy: str = "recursive"
    derivative: str = "step_backward"
    h: float = 1.0
    log_norms: bool = False

    def __post_init__(self) -> None:
        if self.heun_variant not in FOCA_HEUN_VARIANTS:
            raise ValueError(f"unknown FoCa Heun variant: {self.heun_variant}")
        if self.history_policy not in FOCA_HISTORY_POLICIES:
            raise ValueError(f"unknown FoCa history policy: {self.history_policy}")
        if self.derivative not in FOCA_DERIVATIVES:
            raise ValueError(f"unknown FoCa derivative: {self.derivative}")
        if float(self.h) <= 0.0:
            raise ValueError("FoCa h must be positive")


def _detach_feature(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def _empty_history() -> History:
    return {
        "latest_roll_step": None,
        "latest_roll_feature": None,
        "prev_roll_step": None,
        "prev_roll_feature": None,
        "latest_full_step": None,
        "latest_full_feature": None,
        "prev_full_step": None,
        "prev_full_feature": None,
        "last_prediction_kind": None,
        "last_fallback_reason": None,
    }


def update_full_history(
    prev_history: Optional[Mapping[str, Any]],
    feature: torch.Tensor,
    *,
    step: int,
) -> History:
    """Update FoCa per-slot state with a true full-compute feature."""
    old = _empty_history()
    if prev_history:
        old.update(dict(prev_history))

    out = dict(old)
    feature = _detach_feature(feature)
    out["prev_roll_step"] = old.get("latest_roll_step")
    out["prev_roll_feature"] = old.get("latest_roll_feature")
    out["latest_roll_step"] = int(step)
    out["latest_roll_feature"] = feature

    out["prev_full_step"] = old.get("latest_full_step")
    out["prev_full_feature"] = old.get("latest_full_feature")
    out["latest_full_step"] = int(step)
    out["latest_full_feature"] = feature
    out["last_prediction_kind"] = "full"
    out["last_fallback_reason"] = None
    return out


def commit_prediction(
    history: Mapping[str, Any],
    prediction: torch.Tensor,
    *,
    step: int,
) -> History:
    """Commit a cached-step prediction into the rolling history."""
    out = dict(history)
    out["prev_roll_step"] = out.get("latest_roll_step")
    out["prev_roll_feature"] = out.get("latest_roll_feature")
    out["latest_roll_step"] = int(step)
    out["latest_roll_feature"] = _detach_feature(prediction)
    out["last_prediction_kind"] = "predicted"
    out["last_fallback_reason"] = None
    return out


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).norm().cpu().item())


def _nan_inf_count(x: torch.Tensor) -> int:
    bad = torch.isnan(x).sum() + torch.isinf(x).sum()
    return int(bad.detach().cpu().item())


def _fallback(
    history: Mapping[str, Any],
    *,
    reason: str,
    log_norms: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    value = history.get("latest_roll_feature")
    if value is None:
        value = history.get("latest_full_feature")
    if value is None:
        raise ValueError("FoCa history has no feature to reuse")
    pred = value.detach().clone()
    fields: dict[str, Any] = {
        "foca_bdf2_available": False,
        "foca_heun_available": False,
        "foca_fallback_reason": reason,
        "foca_prediction_kind": "fallback_reuse",
        "foca_dry_run_steps": 0,
        "foca_nan_inf_count": _nan_inf_count(pred),
    }
    if log_norms:
        fields.update({
            "foca_prediction_norm": _norm(pred),
            "foca_reuse_norm": _norm(value),
            "foca_forecast_minus_reuse_norm": 0.0,
            "foca_calibration_delta_norm": 0.0,
            "foca_endpoint_slope_norm": None,
            "foca_full_anchor_slope_norm": None,
        })
    return pred, fields


def _full_anchor_slope(history: Mapping[str, Any], h: float) -> Optional[torch.Tensor]:
    latest_full_step = history.get("latest_full_step")
    prev_full_step = history.get("prev_full_step")
    latest_full = history.get("latest_full_feature")
    prev_full = history.get("prev_full_feature")
    if (
        latest_full_step is None
        or prev_full_step is None
        or latest_full is None
        or prev_full is None
    ):
        return None
    gap = int(latest_full_step) - int(prev_full_step)
    if gap <= 0:
        return None
    return (latest_full - prev_full) / (float(gap) * float(h))


def _forecast_one(
    *,
    prev_step: int,
    prev_feature: torch.Tensor,
    latest_step: int,
    latest_feature: torch.Tensor,
    target_step: int,
    history: Mapping[str, Any],
    config: FoCaConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gap = int(latest_step) - int(prev_step)
    if gap <= 0:
        return _fallback(history, reason="nonpositive_roll_gap", log_norms=config.log_norms)

    h = float(config.h)
    d_prev = (latest_feature - prev_feature) / (float(gap) * h)
    bdf2 = (
        latest_feature.mul(4.0 / 3.0)
        - prev_feature.mul(1.0 / 3.0)
        + d_prev.mul((2.0 * h) / 3.0)
    )
    d_hat = (bdf2.mul(3.0) - latest_feature.mul(4.0) + prev_feature) / (2.0 * h)

    full_slope = _full_anchor_slope(history, h)
    heun_available = full_slope is not None and config.heun_variant != "none"
    fallback_reason: Optional[str] = None
    if config.heun_variant == "none":
        pred = bdf2
        prediction_kind = "bdf2_only"
    elif full_slope is None:
        pred = bdf2
        prediction_kind = "bdf2_only"
        fallback_reason = "missing_full_anchor_slope"
    elif config.heun_variant == "paper_literal":
        pred = latest_feature + (full_slope + d_hat).mul(h / 2.0)
        prediction_kind = "paper_literal"
    elif config.heun_variant == "anchored_prose":
        anchor_step = int(history["latest_full_step"])
        anchor_feature = history["latest_full_feature"]
        horizon = (int(target_step) - anchor_step) * h
        pred = anchor_feature + (full_slope + d_hat).mul(horizon / 2.0)
        prediction_kind = "anchored_prose"
    else:  # guarded by FoCaConfig
        raise ValueError(f"unknown FoCa Heun variant: {config.heun_variant}")

    nan_count = _nan_inf_count(pred)
    if nan_count > 0:
        fallback = latest_feature.detach().clone()
        fields = {
            "foca_bdf2_available": True,
            "foca_heun_available": bool(heun_available),
            "foca_fallback_reason": "nonfinite_prediction",
            "foca_prediction_kind": "fallback_reuse",
            "foca_prev_roll_step": int(prev_step),
            "foca_latest_roll_step": int(latest_step),
            "foca_target_step": int(target_step),
            "foca_latest_full_step": (
                None if history.get("latest_full_step") is None else int(history["latest_full_step"])
            ),
            "foca_prev_full_step": (
                None if history.get("prev_full_step") is None else int(history["prev_full_step"])
            ),
            "foca_nan_inf_count": int(nan_count),
        }
        if config.log_norms:
            fields.update({
                "foca_prediction_norm": _norm(fallback),
                "foca_reuse_norm": _norm(latest_feature),
                "foca_forecast_minus_reuse_norm": None,
                "foca_calibration_delta_norm": None,
                "foca_endpoint_slope_norm": None,
                "foca_full_anchor_slope_norm": None if full_slope is None else _norm(full_slope),
            })
        return fallback, fields

    fields: dict[str, Any] = {
        "foca_bdf2_available": True,
        "foca_heun_available": bool(heun_available),
        "foca_fallback_reason": fallback_reason,
        "foca_prediction_kind": prediction_kind,
        "foca_prev_roll_step": int(prev_step),
        "foca_latest_roll_step": int(latest_step),
        "foca_target_step": int(target_step),
        "foca_latest_full_step": (
            None if history.get("latest_full_step") is None else int(history["latest_full_step"])
        ),
        "foca_prev_full_step": (
            None if history.get("prev_full_step") is None else int(history["prev_full_step"])
        ),
        "foca_nan_inf_count": 0,
    }
    if config.log_norms:
        fields.update({
            "foca_prediction_norm": _norm(pred),
            "foca_reuse_norm": _norm(latest_feature),
            "foca_forecast_minus_reuse_norm": _norm(bdf2 - latest_feature),
            "foca_calibration_delta_norm": _norm(pred - bdf2),
            "foca_endpoint_slope_norm": _norm(d_hat),
            "foca_full_anchor_slope_norm": None if full_slope is None else _norm(full_slope),
        })
    return pred, fields


def predict_foca(
    history: Mapping[str, Any],
    *,
    current_step: int,
    config: FoCaConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predict the feature for ``current_step`` and return diagnostics."""
    latest_step = history.get("latest_roll_step")
    prev_step = history.get("prev_roll_step")
    latest_feature = history.get("latest_roll_feature")
    prev_feature = history.get("prev_roll_feature")
    if latest_step is None or prev_step is None or latest_feature is None or prev_feature is None:
        return _fallback(history, reason="insufficient_roll_history", log_norms=config.log_norms)

    if int(current_step) <= int(latest_step):
        pred = latest_feature.detach().clone()
        fields: dict[str, Any] = {
            "foca_bdf2_available": False,
            "foca_heun_available": False,
            "foca_fallback_reason": "target_not_after_latest_roll",
            "foca_prediction_kind": "fallback_reuse",
            "foca_dry_run_steps": 0,
            "foca_nan_inf_count": _nan_inf_count(pred),
        }
        if config.log_norms:
            fields.update({
                "foca_prediction_norm": _norm(pred),
                "foca_reuse_norm": _norm(latest_feature),
                "foca_forecast_minus_reuse_norm": 0.0,
                "foca_calibration_delta_norm": 0.0,
                "foca_endpoint_slope_norm": None,
                "foca_full_anchor_slope_norm": None,
            })
        return pred, fields

    temp_prev_step = int(prev_step)
    temp_prev = prev_feature
    temp_latest_step = int(latest_step)
    temp_latest = latest_feature
    final_pred: Optional[torch.Tensor] = None
    final_fields: dict[str, Any] = {}
    dry_run_steps = 0
    for target in range(temp_latest_step + 1, int(current_step) + 1):
        final_pred, final_fields = _forecast_one(
            prev_step=temp_prev_step,
            prev_feature=temp_prev,
            latest_step=temp_latest_step,
            latest_feature=temp_latest,
            target_step=target,
            history=history,
            config=config,
        )
        temp_prev_step, temp_prev = temp_latest_step, temp_latest
        temp_latest_step, temp_latest = target, final_pred
        dry_run_steps += 1

    assert final_pred is not None
    final_fields["foca_dry_run_steps"] = int(dry_run_steps)
    return final_pred, final_fields
