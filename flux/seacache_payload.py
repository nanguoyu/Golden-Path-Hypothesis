"""Research-only SeaCache gate with forecast residual payloads.

This module keeps the locked SeaCache baseline untouched.  It reuses the
native SeaCache decision rule, but lets cached steps inject a residual forecast
computed only from prior full-refresh residual history.
"""

from __future__ import annotations

import time
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from lib.gates import rel_l1
from lib.history_fd_observer import (
    forecast_predictions,
    init_state as init_history_fd_state,
    online_fields as history_fd_online_fields,
    update_on_full as history_fd_update_on_full,
)
from lib import svdcache_payload
from lib.teacache_coeffs import get_coeffs
from lib.update_history_observer import (
    best_target as update_history_best_target,
    init_state as init_update_history_state,
    online_fields as update_history_online_fields,
    update_on_full as update_history_update_on_full,
)
from lib.wiener import apply_sea_with_scheduler

logger = logging.get_logger(__name__)

PAYLOAD_GATE_MODES = (
    "seacache",
    "teacache",
    "forecast_uncertainty",
    "seacache_forecast_intersection",
    "rfc_input_error",
)
FORECAST_GUARD_MODES = ("taylor_o1", "taylor_o2", "hicache_o2")

BASE_PAYLOAD_MODES = (
    "reuse",
    "taylor_o1",
    "taylor_o2",
    "hicache_o2",
    "ensemble_mean",
)

SELECTOR_PAYLOAD_MODES = (
    "residual_select",
    "output_select",
    "update_select",
)

PCA_CLEAN_PAYLOAD_SPECS = {
    "taylor_o1_pca_project_q1": {
        "raw_mode": "taylor_o1",
        "control": "pca_project",
        "q": 1,
        "gamma": 0.0,
    },
    "taylor_o1_pca_shrink_q1_g0.25": {
        "raw_mode": "taylor_o1",
        "control": "pca_shrink",
        "q": 1,
        "gamma": 0.25,
    },
    "ensemble_pca_project_q2": {
        "raw_mode": "ensemble_mean",
        "control": "pca_project",
        "q": 2,
        "gamma": 0.0,
    },
    "ensemble_pca_shrink_q2_g0.25": {
        "raw_mode": "ensemble_mean",
        "control": "pca_shrink",
        "q": 2,
        "gamma": 0.25,
    },
    "hicache_o2_pca_shrink_q2_g0.25": {
        "raw_mode": "hicache_o2",
        "control": "pca_shrink",
        "q": 2,
        "gamma": 0.25,
    },
}

PCA_CLEAN_PAYLOAD_MODES = tuple(PCA_CLEAN_PAYLOAD_SPECS.keys())
PCA_CLEAN_CONTROLS = {"pca_project", "pca_shrink"}

UPDATE_INVERSE_PAYLOAD_SPECS = {
    "update_inv_q1": {
        "q": 1,
        "target": "ensemble_mean",
        "ridge_rel": 1e-4,
        "alpha_abs_clip": 2.0,
        "delta_norm_clip_rel": 0.5,
    },
    "update_inv_q2": {
        "q": 2,
        "target": "ensemble_mean",
        "ridge_rel": 1e-4,
        "alpha_abs_clip": 2.0,
        "delta_norm_clip_rel": 0.5,
    },
}

UPDATE_INVERSE_PAYLOAD_MODES = tuple(UPDATE_INVERSE_PAYLOAD_SPECS.keys())

UPDATE_SCALAR_CALIB_PAYLOAD_SPECS = {
    "update_alpha_taylor_o1": {
        "raw_mode": "taylor_o1",
        "target": "taylor_o1",
        "alpha_min": 0.0,
        "alpha_max": 1.5,
    },
    "update_alpha_ensemble_mean": {
        "raw_mode": "ensemble_mean",
        "target": "ensemble_mean",
        "alpha_min": 0.0,
        "alpha_max": 1.5,
    },
}

UPDATE_SCALAR_CALIB_PAYLOAD_MODES = tuple(UPDATE_SCALAR_CALIB_PAYLOAD_SPECS.keys())

FORECAST_OPT_PAYLOAD_SPECS = {
    "time_sigma_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "time_forecast",
        "coord": "sigma",
    },
    "time_logsnr_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "time_forecast",
        "coord": "logsnr",
    },
    "curvlim_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "curvature_limited",
        "beta": 1.0,
        "lambda_min": 0.25,
    },
    "prebias_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "prequential_bias",
        "gamma": 0.5,
        "clip_rel": 0.5,
        "ema_beta": 0.1,
    },
    "state_transport_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "state_transport",
        "gamma": 0.5,
        "coef_clip": 1.5,
    },
    "continuous_state_transport_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "state_transport",
        "gamma": 0.5,
        "coef_clip": 1.5,
    },
    "state_gap_damp_taylor_o1_g0.25": {
        "raw_mode": "taylor_o1",
        "control": "state_gap_damped",
        "gamma": 0.25,
        "lambda_min": 0.25,
    },
    "state_gap_damp_taylor_o1_g0.5": {
        "raw_mode": "taylor_o1",
        "control": "state_gap_damped",
        "gamma": 0.5,
        "lambda_min": 0.25,
    },
    "state_lr_q1_g0.5_clip0.5_r1e-3": {
        "raw_mode": "taylor_o1",
        "control": "state_lowrank",
        "rank": 1,
        "gamma": 0.5,
        "clip_rel": 0.5,
        "ridge_rel": 1e-3,
        "residualize_time": False,
    },
    "state_lr_q2_g0.5_clip0.25_r1e-2": {
        "raw_mode": "taylor_o1",
        "control": "state_lowrank",
        "rank": 2,
        "gamma": 0.5,
        "clip_rel": 0.25,
        "ridge_rel": 1e-2,
        "residualize_time": False,
    },
    "state_lr_resid_q2_g0.5_clip0.15_r5e-2": {
        "raw_mode": "taylor_o1",
        "control": "state_lowrank",
        "rank": 2,
        "gamma": 0.5,
        "clip_rel": 0.15,
        "ridge_rel": 5e-2,
        "residualize_time": True,
    },
    "output_target_time_grid9": {
        "raw_mode": "taylor_o1",
        "control": "output_target_scalar",
        "rank": 0,
        "search": "grid",
        "grid_count": 9,
        "ridge_rel": 1e-3,
        "gamma": 0.0,
        "correction_clip_rel": 0.5,
        "shrink_only": False,
    },
    "output_target_q1_grid9": {
        "raw_mode": "taylor_o1",
        "control": "output_target_scalar",
        "rank": 1,
        "search": "grid",
        "grid_count": 9,
        "ridge_rel": 1e-3,
        "gamma": 0.5,
        "correction_clip_rel": 0.5,
        "shrink_only": False,
    },
    "output_target_q1_shrink_grid9": {
        "raw_mode": "taylor_o1",
        "control": "output_target_scalar",
        "rank": 1,
        "search": "grid",
        "grid_count": 9,
        "ridge_rel": 1e-3,
        "gamma": 0.5,
        "correction_clip_rel": 0.5,
        "shrink_only": True,
    },
    "output_target_q2_grid9": {
        "raw_mode": "taylor_o1",
        "control": "output_target_scalar",
        "rank": 2,
        "search": "grid",
        "grid_count": 9,
        "ridge_rel": 1e-3,
        "gamma": 0.5,
        "correction_clip_rel": 0.25,
        "shrink_only": False,
    },
    "output_target_q1_linear": {
        "raw_mode": "taylor_o1",
        "control": "output_target_scalar",
        "rank": 1,
        "search": "linear",
        "grid_count": 0,
        "ridge_rel": 1e-3,
        "gamma": 0.5,
        "correction_clip_rel": 0.5,
        "shrink_only": False,
    },
    "rfc_rfe_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "rfc_rfe",
        "order": 1,
    },
    "rfc_rfe_taylor_o2": {
        "raw_mode": "taylor_o2",
        "control": "rfc_rfe",
        "order": 2,
    },
    "input_gap_eta0_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 0.0,
    },
    "input_gap_eta0.25_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 0.25,
    },
    "input_gap_eta0.5_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 0.5,
    },
    "input_gap_eta0.75_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 0.75,
    },
    "input_gap_eta1_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 1.0,
    },
    "input_gap_eta1.25_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "input_gap_eta",
        "eta": 1.25,
    },
}

FORECAST_OPT_PAYLOAD_MODES = tuple(FORECAST_OPT_PAYLOAD_SPECS.keys())
SVD_CACHE_PAYLOAD_MODES = svdcache_payload.SVD_CACHE_PAYLOAD_MODES

POSTERIOR_ORACLE_PAYLOAD_SPECS = {
    "posterior_oracle_cy_taylor_o1": {
        "raw_mode": "taylor_o1",
        "control": "posterior_oracle_output_scalar",
        "grid_count": 17,
    },
}

POSTERIOR_ORACLE_PAYLOAD_MODES = tuple(POSTERIOR_ORACLE_PAYLOAD_SPECS.keys())

CONTROL_SUFFIXES = (
    "_shift_m1",
    "_shift_p1",
    "_mirror_step",
    "_norm_only",
    "_random_dir",
    "_delta_negative",
    "_delta_random_dir",
    "_delta_orthogonal_random",
    "_step_shuffle",
    "_wrong_prompt",
    "_wrong_seed",
    "_step_only",
    "_prompt_only",
)

BANK_CONTROLS = {
    "step_shuffle",
    "wrong_prompt",
    "wrong_seed",
    "step_only",
    "prompt_only",
}

PAYLOAD_MODES = BASE_PAYLOAD_MODES + tuple(
    f"{base}{suffix}"
    for base in BASE_PAYLOAD_MODES
    if base != "reuse"
    for suffix in CONTROL_SUFFIXES
) + SELECTOR_PAYLOAD_MODES + PCA_CLEAN_PAYLOAD_MODES
PAYLOAD_MODES = PAYLOAD_MODES + UPDATE_INVERSE_PAYLOAD_MODES
PAYLOAD_MODES = PAYLOAD_MODES + UPDATE_SCALAR_CALIB_PAYLOAD_MODES
PAYLOAD_MODES = PAYLOAD_MODES + FORECAST_OPT_PAYLOAD_MODES
PAYLOAD_MODES = PAYLOAD_MODES + SVD_CACHE_PAYLOAD_MODES
PAYLOAD_MODES = PAYLOAD_MODES + POSTERIOR_ORACLE_PAYLOAD_MODES

SHADOW_PAYLOADS = BASE_PAYLOAD_MODES
EPS = 1e-12


def _coefficients_hash(values: list[float]) -> str:
    text = ",".join(f"{float(v):.17g}" for v in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())


def _cos(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    af = a.detach().to(torch.float32)
    bf = b.detach().to(torch.float32)
    an = float(af.norm().item())
    bn = float(bf.norm().item())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.sum(af * bf).item()) / (an * bn + EPS)


def _maybe_sync(tensor: torch.Tensor) -> None:
    if tensor.is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(tensor.device)


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return float(value.detach().to("cpu").item())
        return float(value)
    except Exception:
        return None


def _scheduler_step_fields(self, step: int) -> Dict[str, Optional[float]]:
    fields: Dict[str, Optional[float]] = {
        "sigma_n": None,
        "sigma_np1": None,
        "step_size_H": None,
    }
    sigmas = getattr(getattr(self, "scheduler", None), "sigmas", None)
    if sigmas is None:
        return fields
    try:
        if int(step) + 1 >= len(sigmas):
            return fields
        sigma_n = _as_float(sigmas[int(step)])
        sigma_np1 = _as_float(sigmas[int(step) + 1])
    except Exception:
        return fields
    fields["sigma_n"] = sigma_n
    fields["sigma_np1"] = sigma_np1
    fields["step_size_H"] = (
        None if sigma_n is None or sigma_np1 is None else float(sigma_np1 - sigma_n)
    )
    return fields


def _project_output(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    return self.proj_out(self.norm_out(hidden_states, temb))


def init_forecast_guard_state() -> Dict[str, Any]:
    return {
        "nonconformity": [],
        "calib_count": 0,
        "last_lor_err_abs": None,
        "last_lor_err_rel": None,
        "last_lor_nonconformity": None,
    }


def _forecast_guard_quantile(state: Optional[Dict[str, Any]], q: float) -> Optional[float]:
    st = state or {}
    values = [
        float(v)
        for v in (st.get("nonconformity") or [])
        if v is not None and np.isfinite(float(v))
    ]
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _empty_forecast_guard_fields() -> Dict[str, Any]:
    return {
        "payload_gate_mode": "seacache",
        "forecast_guard_enabled": False,
        "forecast_guard_observe": False,
        "forecast_guard_active": False,
        "forecast_guard_mode": None,
        "forecast_guard_space": "update",
        "forecast_guard_available_pre": None,
        "forecast_guard_warmup_ready_pre": None,
        "forecast_guard_force_full_reason": None,
        "forecast_guard_anchor_step_pre": None,
        "forecast_guard_age_pre": None,
        "forecast_guard_age_scale_pre": None,
        "forecast_guard_calib_count_pre": None,
        "forecast_guard_quantile_level": None,
        "forecast_guard_quantile_pre": None,
        "forecast_guard_tau": None,
        "forecast_guard_scale_floor": None,
        "forecast_guard_self_scale_abs_pre": None,
        "forecast_guard_self_scale_rel_pre": None,
        "forecast_guard_score_pre": None,
        "forecast_guard_cache_allowed": None,
        "forecast_guard_decision_reason": None,
        "forecast_guard_reuse_update_norm_pre": None,
        "forecast_guard_forecast_update_norm_pre": None,
        "forecast_guard_elapsed_ms_pre": None,
        "forecast_guard_observation_valid": False,
        "forecast_guard_lor_err_abs": None,
        "forecast_guard_lor_err_rel": None,
        "forecast_guard_lor_nonconformity": None,
        "forecast_guard_calib_count_post": None,
        "forecast_guard_quantile_post": None,
    }


def _empty_rfc_gate_fields() -> Dict[str, Any]:
    return {
        "rfc_gate_enabled": False,
        "rfc_gate_tau": None,
        "rfc_gate_order_requested": None,
        "rfc_gate_order_used": None,
        "rfc_gate_history_count": None,
        "rfc_gate_anchor_step": None,
        "rfc_gate_prev_step": None,
        "rfc_gate_prev2_step": None,
        "rfc_gate_gap": None,
        "rfc_gate_prev_step_gap": None,
        "rfc_gate_prev2_step_gap": None,
        "rfc_gate_input_norm": None,
        "rfc_gate_input_delta_norm": None,
        "rfc_gate_prediction_norm": None,
        "rfc_gate_prediction_error_abs_l1": None,
        "rfc_gate_prediction_error_rel_l1": None,
        "rfc_gate_accumulator_before": None,
        "rfc_gate_accumulator_after_increment": None,
        "rfc_gate_cache_allowed": False,
        "rfc_gate_decision_reason": None,
        "rfc_gate_elapsed_ms": None,
    }


def _rfc_gate_order_from_payload(payload_mode: str) -> int:
    mode = str(payload_mode)
    if mode.endswith("_o2") or "taylor_o2" in mode or "hicache_o2" in mode:
        return 2
    return 1


def _rfc_input_error_gate_pre_fields(
    self,
    *,
    gate_input: torch.Tensor,
    opt_state: Optional[Dict[str, Any]],
    payload_mode: str,
    step: int,
    force_full_reason: Optional[str],
) -> Dict[str, Any]:
    fields = _empty_rfc_gate_fields()
    fields["rfc_gate_enabled"] = True
    tau = float(getattr(self, "seacache_payload_rfc_gate_tau", getattr(self, "seacache_thresh", 0.3)))
    order_requested = _rfc_gate_order_from_payload(payload_mode)
    accumulator_before = float(getattr(self, "seacache_payload_rfc_accumulated_error", 0.0))
    fields.update({
        "rfc_gate_tau": tau,
        "rfc_gate_order_requested": int(order_requested),
        "rfc_gate_accumulator_before": accumulator_before,
    })
    start = time.perf_counter()
    try:
        records = _forecast_opt_records(opt_state, "gate_input_records")
        fields["rfc_gate_history_count"] = int(len(records))
        if force_full_reason is not None:
            fields["rfc_gate_decision_reason"] = f"force_full:{force_full_reason}"
            return fields
        if len(records) < 2:
            fields["rfc_gate_decision_reason"] = "insufficient_gate_input_records"
            return fields
        anchor, prev = records[0], records[1]
        anchor_x = anchor.get("tensor")
        prev_x = prev.get("tensor")
        if (
            anchor_x is None
            or prev_x is None
            or tuple(anchor_x.shape) != tuple(gate_input.shape)
            or tuple(prev_x.shape) != tuple(gate_input.shape)
        ):
            fields["rfc_gate_decision_reason"] = "record_shape_mismatch"
            return fields
        anchor_step = int(anchor["step"])
        prev_step = int(prev["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = _record_step_gap(anchor, prev)
        x_anchor = anchor_x.detach().to(device=gate_input.device, dtype=torch.float32)
        x_prev = prev_x.detach().to(device=gate_input.device, dtype=torch.float32)
        x_current = gate_input.detach().to(device=gate_input.device, dtype=torch.float32)
        d1 = (x_anchor - x_prev) / float(prev_gap)
        pred = x_anchor + float(gap) * d1
        order_used = 1
        prev2_step = None
        prev2_gap = None
        if order_requested >= 2 and len(records) >= 3:
            prev2 = records[2]
            prev2_x = prev2.get("tensor")
            if prev2_x is not None and tuple(prev2_x.shape) == tuple(gate_input.shape):
                prev2_step = int(prev2["step"])
                prev2_gap = _record_step_gap(prev, prev2)
                x_prev2 = prev2_x.detach().to(device=gate_input.device, dtype=torch.float32)
                d1_prev = (x_prev - x_prev2) / float(prev2_gap)
                d2 = 2.0 * (d1 - d1_prev) / float(prev_gap + prev2_gap)
                pred = pred + 0.5 * float(gap * gap) * d2
                order_used = 2
        err = x_current - pred
        err_abs_l1 = float(torch.sum(torch.abs(err)).item())
        input_norm_l1 = float(torch.sum(torch.abs(x_current)).item())
        err_rel_l1 = float(err_abs_l1 / (input_norm_l1 + EPS))
        accumulator_after = accumulator_before + err_rel_l1
        cache_allowed = bool(accumulator_after <= tau)
        fields.update({
            "rfc_gate_order_used": int(order_used),
            "rfc_gate_anchor_step": anchor_step,
            "rfc_gate_prev_step": prev_step,
            "rfc_gate_prev2_step": prev2_step,
            "rfc_gate_gap": int(gap),
            "rfc_gate_prev_step_gap": int(prev_gap),
            "rfc_gate_prev2_step_gap": prev2_gap,
            "rfc_gate_input_norm": float(x_current.norm().item()),
            "rfc_gate_input_delta_norm": float((x_current - x_anchor).norm().item()),
            "rfc_gate_prediction_norm": float(pred.norm().item()),
            "rfc_gate_prediction_error_abs_l1": err_abs_l1,
            "rfc_gate_prediction_error_rel_l1": err_rel_l1,
            "rfc_gate_accumulator_after_increment": accumulator_after,
            "rfc_gate_cache_allowed": cache_allowed,
            "rfc_gate_decision_reason": "cache_allowed" if cache_allowed else "accumulated_error_exceeds_tau",
        })
        return fields
    finally:
        fields["rfc_gate_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)


def _resolve_forecast_guard_mode(self, payload_mode: str) -> Optional[str]:
    mode = str(getattr(self, "seacache_payload_forecast_guard_mode", "auto"))
    if mode == "auto":
        base_mode, control = _payload_spec(str(payload_mode))
        if control == "none" and base_mode in FORECAST_GUARD_MODES:
            return str(base_mode)
        return None
    return mode if mode in FORECAST_GUARD_MODES else None


def _single_forecast_payload(
    *,
    mode: str,
    reuse: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    sigma: float,
) -> Optional[torch.Tensor]:
    if mode not in FORECAST_GUARD_MODES:
        return None
    preds = forecast_predictions(history_state, step=int(step), sigma=float(sigma))
    pred = preds.get(str(mode))
    if pred is None or pred.shape != reuse.shape:
        return None
    return pred.to(dtype=reuse.dtype, device=reuse.device)


def _forecast_guard_pre_fields(
    self,
    *,
    payload_mode: str,
    reuse: Optional[torch.Tensor],
    history_state: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step: int,
    sigma: float,
    step_size: Optional[float],
    force_full_reason: Optional[str],
) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    fields = _empty_forecast_guard_fields()
    gate_mode = str(getattr(self, "seacache_payload_gate_mode", "seacache"))
    observe = bool(getattr(self, "seacache_payload_forecast_guard_observe", False))
    active = gate_mode in ("forecast_uncertainty", "seacache_forecast_intersection")
    enabled = bool(observe or active)
    guard_mode = _resolve_forecast_guard_mode(self, payload_mode)
    state = getattr(self, "seacache_payload_forecast_guard_state", None) or init_forecast_guard_state()
    q_level = float(getattr(self, "seacache_payload_forecast_guard_quantile", 0.8))
    tau = float(getattr(self, "seacache_payload_forecast_guard_tau", 0.05))
    warmup = int(getattr(self, "seacache_payload_forecast_guard_warmup_updates", 1))
    age_gamma = float(getattr(self, "seacache_payload_forecast_guard_age_gamma", 0.0))
    scale_floor = float(getattr(self, "seacache_payload_forecast_guard_scale_floor", 1e-4))
    calib_count = int(state.get("calib_count") or 0)
    q_pre = _forecast_guard_quantile(state, q_level)
    history = (history_state or {}).get("history") or {}
    anchor_step = (history_state or {}).get("anchor_step")
    age = None if anchor_step is None else max(int(step) - int(anchor_step), 0)
    age_scale = None if age is None else float(max(int(age), 1) ** age_gamma)

    fields.update({
        "payload_gate_mode": gate_mode,
        "forecast_guard_enabled": enabled,
        "forecast_guard_observe": observe,
        "forecast_guard_active": active,
        "forecast_guard_mode": guard_mode,
        "forecast_guard_force_full_reason": force_full_reason,
        "forecast_guard_anchor_step_pre": None if anchor_step is None else int(anchor_step),
        "forecast_guard_age_pre": age,
        "forecast_guard_age_scale_pre": age_scale,
        "forecast_guard_calib_count_pre": calib_count,
        "forecast_guard_quantile_level": q_level,
        "forecast_guard_quantile_pre": q_pre,
        "forecast_guard_tau": tau,
        "forecast_guard_scale_floor": scale_floor,
        "forecast_guard_warmup_ready_pre": bool(calib_count >= warmup),
        "forecast_guard_calib_count_post": calib_count,
        "forecast_guard_quantile_post": q_pre,
    })
    if not enabled:
        return fields, None, None, None

    start = time.perf_counter()
    reason = None
    forecast = None
    reuse_update = None
    forecast_update = None
    if guard_mode is None:
        reason = "mode_unavailable"
    elif reuse is None:
        reason = "missing_reuse"
    elif not history:
        reason = "missing_history"
    elif step_size is None:
        reason = "missing_step_size"
    else:
        forecast = _single_forecast_payload(
            mode=guard_mode,
            reuse=reuse,
            history_state=history_state,
            step=int(step),
            sigma=float(sigma),
        )
        if forecast is None:
            reason = "forecast_unavailable"
        else:
            with torch.inference_mode():
                reuse_output = _project_output(self, hidden_states + reuse, temb)
                forecast_output = _project_output(self, hidden_states + forecast, temb)
            reuse_update = reuse_output.detach().to(torch.float32) * float(step_size)
            forecast_update = forecast_output.detach().to(torch.float32) * float(step_size)
            diff = forecast_update - reuse_update
            self_scale_abs = float(diff.norm().item())
            reuse_update_norm = float(reuse_update.norm().item())
            self_scale_rel = float(self_scale_abs / (reuse_update_norm + EPS))
            forecast_update_norm = float(forecast_update.norm().item())
            score = None
            if q_pre is not None and age_scale is not None:
                score = float(q_pre * max(self_scale_rel, scale_floor) * age_scale)
            warmup_ready = bool(calib_count >= warmup)
            allowed = bool(warmup_ready and score is not None and score <= tau)
            if not warmup_ready:
                reason = "warmup"
            elif score is None:
                reason = "missing_score"
            elif not allowed:
                reason = "score_exceeds_tau"
            else:
                reason = "cache_allowed"
            fields.update({
                "forecast_guard_available_pre": True,
                "forecast_guard_self_scale_abs_pre": self_scale_abs,
                "forecast_guard_self_scale_rel_pre": self_scale_rel,
                "forecast_guard_score_pre": score,
                "forecast_guard_cache_allowed": allowed,
                "forecast_guard_decision_reason": reason,
                "forecast_guard_reuse_update_norm_pre": reuse_update_norm,
                "forecast_guard_forecast_update_norm_pre": forecast_update_norm,
            })

    if fields["forecast_guard_available_pre"] is None:
        fields["forecast_guard_available_pre"] = False
        fields["forecast_guard_cache_allowed"] = False
        fields["forecast_guard_decision_reason"] = reason
    fields["forecast_guard_elapsed_ms_pre"] = float((time.perf_counter() - start) * 1000.0)
    return fields, forecast, reuse_update, forecast_update


def _forecast_guard_update_on_full(
    state: Optional[Dict[str, Any]],
    *,
    forecast_update: Optional[torch.Tensor],
    actual_update: Optional[torch.Tensor],
    self_scale_rel: Optional[float],
    scale_floor: float,
    window: int,
    quantile: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    st = dict(state or init_forecast_guard_state())
    values = [
        float(v)
        for v in (st.get("nonconformity") or [])
        if v is not None and np.isfinite(float(v))
    ]
    fields = {
        "forecast_guard_observation_valid": False,
        "forecast_guard_lor_err_abs": None,
        "forecast_guard_lor_err_rel": None,
        "forecast_guard_lor_nonconformity": None,
        "forecast_guard_calib_count_post": int(st.get("calib_count") or 0),
        "forecast_guard_quantile_post": _forecast_guard_quantile(st, float(quantile)),
    }
    if (
        forecast_update is None
        or actual_update is None
        or forecast_update.shape != actual_update.shape
        or self_scale_rel is None
    ):
        return st, fields

    err_abs = float((forecast_update.detach().to(torch.float32) - actual_update.detach().to(torch.float32)).norm().item())
    actual_norm = float(actual_update.detach().to(torch.float32).norm().item())
    err_rel = float(err_abs / (actual_norm + EPS))
    nonconformity = float(err_rel / max(float(self_scale_rel), float(scale_floor)))
    if not np.isfinite(nonconformity):
        return st, fields

    values.append(nonconformity)
    if int(window) > 0 and len(values) > int(window):
        values = values[-int(window):]
    st["nonconformity"] = values
    st["calib_count"] = int(st.get("calib_count") or 0) + 1
    st["last_lor_err_abs"] = err_abs
    st["last_lor_err_rel"] = err_rel
    st["last_lor_nonconformity"] = nonconformity
    fields.update({
        "forecast_guard_observation_valid": True,
        "forecast_guard_lor_err_abs": err_abs,
        "forecast_guard_lor_err_rel": err_rel,
        "forecast_guard_lor_nonconformity": nonconformity,
        "forecast_guard_calib_count_post": int(st["calib_count"]),
        "forecast_guard_quantile_post": _forecast_guard_quantile(st, float(quantile)),
    })
    return st, fields


def _full_transformer_residual(
    self,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    controlnet_block_samples,
    controlnet_single_block_samples,
    controlnet_blocks_repeat: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the expensive transformer body and return updated context + hidden states."""

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            def _ckpt(module):
                def _fwd(hs, ehs, temb_, ire):
                    return module(
                        hidden_states=hs,
                        encoder_hidden_states=ehs,
                        temb=temb_,
                        image_rotary_emb=ire,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                return _fwd

            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                _ckpt(block), hidden_states, encoder_hidden_states, temb,
                image_rotary_emb, **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
            )
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            def _ckpt2(module):
                def _fwd(hs, ehs, temb_, ire):
                    return module(
                        hidden_states=hs,
                        encoder_hidden_states=ehs,
                        temb=temb_,
                        image_rotary_emb=ire,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                return _fwd

            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                _ckpt2(block), hidden_states, encoder_hidden_states, temb,
                image_rotary_emb, **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_single_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples))
            )
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    return encoder_hidden_states, hidden_states


def _candidate_payloads(
    *,
    reuse: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    sigma: float,
) -> Dict[str, torch.Tensor]:
    preds = forecast_predictions(history_state, step=int(step), sigma=float(sigma))
    out: Dict[str, torch.Tensor] = {"reuse": reuse}
    for name in ("taylor_o1", "taylor_o2", "hicache_o2"):
        pred = preds.get(name)
        if pred is not None and pred.shape == reuse.shape:
            out[name] = pred.to(dtype=reuse.dtype, device=reuse.device)
    parts = [
        out[name]
        for name in ("taylor_o1", "taylor_o2", "hicache_o2")
        if name in out
    ]
    if parts:
        out["ensemble_mean"] = torch.stack(parts, dim=0).mean(dim=0)
    return out


def _empty_scalar_family_fields() -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "shadow_scalar_family_available": False,
        "shadow_scalar_family_fallback_reason": None,
        "shadow_scalar_family_anchor_step": None,
        "shadow_scalar_family_raw_c": None,
        "shadow_scalar_family_grid_min_c": None,
        "shadow_scalar_family_grid_max_c": None,
        "shadow_scalar_family_grid_count": None,
        "shadow_scalar_family_anchor_norm": None,
        "shadow_scalar_family_direction_norm": None,
        "shadow_scalar_family_raw_delta_from_taylor_o1_norm": None,
    }
    for space in ("payload", "output", "update"):
        fields[f"shadow_scalar_family_{space}_reuse_err_abs"] = None
        fields[f"shadow_scalar_family_{space}_raw_err_abs"] = None
        fields[f"shadow_scalar_family_{space}_oracle_err_abs"] = None
        fields[f"shadow_scalar_family_{space}_oracle_c"] = None
        fields[f"shadow_scalar_family_{space}_oracle_improvement_vs_raw_abs"] = None
        fields[f"shadow_scalar_family_{space}_oracle_improvement_vs_raw_rel"] = None
        fields[f"shadow_scalar_family_{space}_oracle_improvement_vs_reuse_abs"] = None
        fields[f"shadow_scalar_family_{space}_oracle_improvement_vs_reuse_rel"] = None
        fields[f"shadow_scalar_family_{space}_oracle_boundary_hit"] = None
    fields["shadow_scalar_family_payload_oracle_c_unbounded"] = None
    fields["shadow_scalar_family_payload_oracle_c_bounded"] = None
    for control in ("negdir", "randdir"):
        fields[f"shadow_scalar_family_output_{control}_oracle_c"] = None
        fields[f"shadow_scalar_family_output_{control}_oracle_err_abs"] = None
        fields[f"shadow_scalar_family_output_{control}_oracle_improvement_vs_raw_abs"] = None
        fields[f"shadow_scalar_family_output_{control}_oracle_improvement_vs_raw_rel"] = None
        fields[f"shadow_scalar_family_output_{control}_oracle_boundary_hit"] = None
    return fields


def _unique_float_values(values: list[float], *, atol: float = 1e-8) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values if np.isfinite(float(v))):
        if not out or abs(float(value) - out[-1]) > atol:
            out.append(float(value))
    return out


def _scalar_family_grid(
    *,
    c_min: float,
    c_max: float,
    c_raw: float,
    c_res_bounded: Optional[float],
    count: int,
) -> list[float]:
    values = np.linspace(float(c_min), float(c_max), num=int(count), dtype=np.float64).tolist()
    values.append(float(c_raw))
    if c_res_bounded is not None and np.isfinite(float(c_res_bounded)):
        values.append(float(c_res_bounded))
    return _unique_float_values(values)


def _scalar_payload(anchor: torch.Tensor, direction: torch.Tensor, c_value: float) -> torch.Tensor:
    return (
        anchor.detach().to(device=direction.device, dtype=torch.float32)
        + float(c_value) * direction.detach().to(device=direction.device, dtype=torch.float32)
    )


def _err_norm(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.detach().to(torch.float32) - target.detach().to(torch.float32)).norm().item())


def _improvement_fields(
    fields: Dict[str, Any],
    *,
    prefix: str,
    reuse_err: Optional[float],
    raw_err: Optional[float],
    oracle_err: Optional[float],
) -> None:
    if raw_err is not None and oracle_err is not None:
        improvement = float(raw_err - oracle_err)
        fields[f"{prefix}_improvement_vs_raw_abs"] = improvement
        fields[f"{prefix}_improvement_vs_raw_rel"] = float(improvement / (float(raw_err) + EPS))
    if reuse_err is not None and oracle_err is not None:
        improvement = float(reuse_err - oracle_err)
        fields[f"{prefix}_improvement_vs_reuse_abs"] = improvement
        fields[f"{prefix}_improvement_vs_reuse_rel"] = float(improvement / (float(reuse_err) + EPS))


def _bounded_output_oracle(
    self,
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    shadow_output: torch.Tensor,
    anchor: torch.Tensor,
    direction: torch.Tensor,
    grid_values: list[float],
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    if not grid_values:
        return None, None, None
    best_c: Optional[float] = None
    best_err: Optional[float] = None
    with torch.inference_mode():
        for c_value in grid_values:
            payload = _scalar_payload(anchor, direction, float(c_value)).to(
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            candidate_output = _project_output(self, hidden_states + payload, temb)
            err_abs = _err_norm(candidate_output, shadow_output)
            if best_err is None or err_abs < best_err:
                best_err = float(err_abs)
                best_c = float(c_value)
    if best_c is None:
        return None, None, None
    boundary_hit = bool(
        abs(best_c - min(grid_values)) <= 1e-8
        or abs(best_c - max(grid_values)) <= 1e-8
    )
    return best_c, best_err, boundary_hit


def _scalar_family_oracle_fields(
    self,
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    shadow_residual: torch.Tensor,
    shadow_output: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    candidates: Dict[str, torch.Tensor],
    step: int,
    step_size_abs: Optional[float],
) -> Dict[str, Any]:
    fields = _empty_scalar_family_fields()
    st = history_state or {}
    history = st.get("history") or {}
    anchor_step = st.get("anchor_step")
    anchor = history.get(0) if isinstance(history, dict) else None
    direction = history.get(1) if isinstance(history, dict) else None
    if (
        anchor is None
        or direction is None
        or anchor_step is None
        or tuple(anchor.shape) != tuple(shadow_residual.shape)
        or tuple(direction.shape) != tuple(shadow_residual.shape)
    ):
        fields["shadow_scalar_family_fallback_reason"] = "missing_o1_history"
        return fields

    anchor_f = anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
    direction_f = direction.detach().to(device=hidden_states.device, dtype=torch.float32)
    shadow_residual_f = shadow_residual.detach().to(device=hidden_states.device, dtype=torch.float32)
    shadow_output_f = shadow_output.detach().to(device=hidden_states.device, dtype=torch.float32)
    denom = float(torch.sum(direction_f * direction_f).item())
    if denom <= EPS:
        fields["shadow_scalar_family_fallback_reason"] = "zero_direction"
        return fields

    c_raw = float(max(int(step) - int(anchor_step), 0))
    c_min = 0.0
    c_max = float(max(4.0, 2.0 * c_raw + 2.0))
    raw_payload = _scalar_payload(anchor_f, direction_f, c_raw)
    reuse_payload = anchor_f
    c_res_unbounded = float(
        torch.sum(direction_f * (shadow_residual_f - anchor_f)).item() / (denom + EPS)
    )
    c_res_bounded = float(np.clip(c_res_unbounded, c_min, c_max))
    residual_oracle_payload = _scalar_payload(anchor_f, direction_f, c_res_bounded)
    grid_values = _scalar_family_grid(
        c_min=c_min,
        c_max=c_max,
        c_raw=c_raw,
        c_res_bounded=c_res_bounded,
        count=17,
    )

    fields.update({
        "shadow_scalar_family_available": True,
        "shadow_scalar_family_anchor_step": int(anchor_step),
        "shadow_scalar_family_raw_c": float(c_raw),
        "shadow_scalar_family_grid_min_c": float(c_min),
        "shadow_scalar_family_grid_max_c": float(c_max),
        "shadow_scalar_family_grid_count": int(len(grid_values)),
        "shadow_scalar_family_anchor_norm": float(anchor_f.norm().item()),
        "shadow_scalar_family_direction_norm": float(direction_f.norm().item()),
        "shadow_scalar_family_payload_oracle_c_unbounded": float(c_res_unbounded),
        "shadow_scalar_family_payload_oracle_c_bounded": float(c_res_bounded),
        "shadow_scalar_family_payload_oracle_c": float(c_res_bounded),
        "shadow_scalar_family_payload_oracle_boundary_hit": bool(
            abs(c_res_bounded - c_min) <= 1e-8 or abs(c_res_bounded - c_max) <= 1e-8
        ),
    })
    taylor_o1 = candidates.get("taylor_o1")
    if taylor_o1 is not None and tuple(taylor_o1.shape) == tuple(raw_payload.shape):
        fields["shadow_scalar_family_raw_delta_from_taylor_o1_norm"] = _err_norm(raw_payload, taylor_o1)

    payload_reuse_err = _err_norm(reuse_payload, shadow_residual_f)
    payload_raw_err = _err_norm(raw_payload, shadow_residual_f)
    payload_oracle_err = _err_norm(residual_oracle_payload, shadow_residual_f)
    fields.update({
        "shadow_scalar_family_payload_reuse_err_abs": payload_reuse_err,
        "shadow_scalar_family_payload_raw_err_abs": payload_raw_err,
        "shadow_scalar_family_payload_oracle_err_abs": payload_oracle_err,
    })
    _improvement_fields(
        fields,
        prefix="shadow_scalar_family_payload_oracle",
        reuse_err=payload_reuse_err,
        raw_err=payload_raw_err,
        oracle_err=payload_oracle_err,
    )

    with torch.inference_mode():
        reuse_output = _project_output(self, hidden_states + reuse_payload.to(dtype=hidden_states.dtype), temb)
        raw_output = _project_output(self, hidden_states + raw_payload.to(dtype=hidden_states.dtype), temb)
    output_reuse_err = _err_norm(reuse_output, shadow_output_f)
    output_raw_err = _err_norm(raw_output, shadow_output_f)
    output_oracle_c, output_oracle_err, output_boundary = _bounded_output_oracle(
        self,
        hidden_states=hidden_states,
        temb=temb,
        shadow_output=shadow_output_f,
        anchor=anchor_f,
        direction=direction_f,
        grid_values=grid_values,
    )
    fields.update({
        "shadow_scalar_family_output_reuse_err_abs": output_reuse_err,
        "shadow_scalar_family_output_raw_err_abs": output_raw_err,
        "shadow_scalar_family_output_oracle_c": output_oracle_c,
        "shadow_scalar_family_output_oracle_err_abs": output_oracle_err,
        "shadow_scalar_family_output_oracle_boundary_hit": output_boundary,
    })
    _improvement_fields(
        fields,
        prefix="shadow_scalar_family_output_oracle",
        reuse_err=output_reuse_err,
        raw_err=output_raw_err,
        oracle_err=output_oracle_err,
    )

    if step_size_abs is not None:
        update_reuse_err = float(step_size_abs * output_reuse_err)
        update_raw_err = float(step_size_abs * output_raw_err)
        update_oracle_err = None if output_oracle_err is None else float(step_size_abs * output_oracle_err)
        fields.update({
            "shadow_scalar_family_update_reuse_err_abs": update_reuse_err,
            "shadow_scalar_family_update_raw_err_abs": update_raw_err,
            "shadow_scalar_family_update_oracle_c": output_oracle_c,
            "shadow_scalar_family_update_oracle_err_abs": update_oracle_err,
            "shadow_scalar_family_update_oracle_boundary_hit": output_boundary,
        })
        _improvement_fields(
            fields,
            prefix="shadow_scalar_family_update_oracle",
            reuse_err=update_reuse_err,
            raw_err=update_raw_err,
            oracle_err=update_oracle_err,
        )

    control_grid_values = _scalar_family_grid(
        c_min=c_min,
        c_max=c_max,
        c_raw=c_raw,
        c_res_bounded=None,
        count=9,
    )
    controls = {
        "negdir": -direction_f,
        "randdir": _random_direction_like(direction_f, mode="scalar_family_random_dir", step=int(step)).detach().to(
            device=hidden_states.device,
            dtype=torch.float32,
        ),
    }
    for control_name, control_direction in controls.items():
        control_c, control_err, control_boundary = _bounded_output_oracle(
            self,
            hidden_states=hidden_states,
            temb=temb,
            shadow_output=shadow_output_f,
            anchor=anchor_f,
            direction=control_direction,
            grid_values=control_grid_values,
        )
        fields.update({
            f"shadow_scalar_family_output_{control_name}_oracle_c": control_c,
            f"shadow_scalar_family_output_{control_name}_oracle_err_abs": control_err,
            f"shadow_scalar_family_output_{control_name}_oracle_boundary_hit": control_boundary,
        })
        if control_err is not None:
            improvement = float(output_raw_err - control_err)
            fields[f"shadow_scalar_family_output_{control_name}_oracle_improvement_vs_raw_abs"] = improvement
            fields[f"shadow_scalar_family_output_{control_name}_oracle_improvement_vs_raw_rel"] = (
                float(improvement / (output_raw_err + EPS))
            )

    return fields


def _payload_spec(mode: str) -> Tuple[str, str]:
    spec = POSTERIOR_ORACLE_PAYLOAD_SPECS.get(str(mode))
    if spec is not None:
        return str(spec["raw_mode"]), str(spec["control"])
    spec = PCA_CLEAN_PAYLOAD_SPECS.get(str(mode))
    if spec is not None:
        return str(spec["raw_mode"]), str(spec["control"])
    spec = UPDATE_SCALAR_CALIB_PAYLOAD_SPECS.get(str(mode))
    if spec is not None:
        return str(spec["raw_mode"]), "update_scalar_calib"
    spec = FORECAST_OPT_PAYLOAD_SPECS.get(str(mode))
    if spec is not None:
        return str(spec["raw_mode"]), str(spec["control"])
    if svdcache_payload.is_mode(str(mode)):
        return svdcache_payload.payload_spec(str(mode))
    if mode in UPDATE_INVERSE_PAYLOAD_MODES:
        return "update_target", "inverse_basis"
    if mode in SELECTOR_PAYLOAD_MODES:
        return mode, "selector"
    for suffix in CONTROL_SUFFIXES:
        if mode.endswith(suffix):
            base = mode[: -len(suffix)]
            if base in BASE_PAYLOAD_MODES and base != "reuse":
                return base, suffix[1:]
    return mode, "none"


def _stable_control_seed(mode: str, step: int, shape: Tuple[int, ...], salt: int = 0) -> int:
    payload = f"{mode}|{int(step)}|{int(salt)}|{','.join(str(int(x)) for x in shape)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _random_direction_like(
    reference: torch.Tensor,
    *,
    mode: str,
    step: int,
    salt: int = 0,
) -> torch.Tensor:
    ref = reference.detach().to(torch.float32)
    ref_norm = float(ref.norm().item())
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(_stable_control_seed(mode, int(step), tuple(reference.shape), int(salt)))
    noise = torch.randn(reference.shape, device=reference.device, generator=gen, dtype=torch.float32)
    noise_norm = float(noise.norm().item())
    if ref_norm <= 0.0 or noise_norm <= 0.0:
        return torch.zeros_like(reference)
    return (noise / (noise_norm + EPS) * ref_norm).to(dtype=reference.dtype, device=reference.device)


def _orthogonal_random_direction_like(
    reference: torch.Tensor,
    *,
    mode: str,
    step: int,
    salt: int = 0,
) -> torch.Tensor:
    ref = reference.detach().to(torch.float32)
    ref_norm = float(ref.norm().item())
    if ref_norm <= 0.0:
        return torch.zeros_like(reference)
    rand = _random_direction_like(reference, mode=mode, step=step, salt=int(salt)).detach().to(torch.float32)
    rand = rand - (torch.sum(rand * ref) / (ref_norm * ref_norm + EPS)) * ref
    rand_norm = float(rand.norm().item())
    if rand_norm <= 0.0:
        return torch.zeros_like(reference)
    return (rand / (rand_norm + EPS) * ref_norm).to(dtype=reference.dtype, device=reference.device)


def _pca_clean_payload(
    *,
    raw_forecast: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    q: int,
    gamma: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    fields = _empty_pca_fields()
    fields.update({
        "payload_pca_enabled": True,
        "payload_pca_q": int(q),
        "payload_pca_gamma": float(gamma),
    })
    st = history_state or {}
    history = st.get("history") or {}
    anchor = history.get(0)
    if anchor is None or not hasattr(anchor, "shape"):
        fields["payload_pca_fallback_reason"] = "missing_anchor"
        return None, fields
    if tuple(anchor.shape) != tuple(raw_forecast.shape):
        fields["payload_pca_fallback_reason"] = "anchor_shape_mismatch"
        return None, fields

    basis = []
    for key in sorted(int(k) for k in history.keys() if int(k) > 0):
        tensor = history.get(key)
        if tensor is not None and tuple(tensor.shape) == tuple(raw_forecast.shape):
            basis.append(tensor.detach().to(device=raw_forecast.device, dtype=torch.float32))
    fields["payload_pca_basis_count"] = int(len(basis))
    if not basis:
        fields["payload_pca_fallback_reason"] = "missing_basis"
        return None, fields

    anchor_f = anchor.detach().to(device=raw_forecast.device, dtype=torch.float32)
    raw_f = raw_forecast.detach().to(torch.float32)
    delta = raw_f - anchor_f
    delta_flat = delta.reshape(-1)
    delta_norm = float(delta_flat.norm().item())
    fields["payload_pca_anchor_norm"] = _norm(anchor)
    fields["payload_pca_raw_delta_norm"] = delta_norm
    if delta_norm <= 0.0:
        clean = anchor_f.to(dtype=raw_forecast.dtype, device=raw_forecast.device)
        fields.update({
            "payload_pca_rank_used": 0,
            "payload_pca_projected_delta_norm": 0.0,
            "payload_pca_residual_delta_norm": 0.0,
            "payload_pca_clean_delta_norm": 0.0,
            "payload_pca_energy_keep_rel": 0.0,
            "payload_pca_eigvals": "",
        })
        return clean, fields

    # Rows are basis vectors. Gram PCA avoids materializing full D x D operators.
    basis_mat = torch.stack([b.reshape(-1) for b in basis], dim=0)
    gram = basis_mat @ basis_mat.T
    try:
        eigvals, eigvecs = torch.linalg.eigh(gram)
    except RuntimeError as exc:
        fields["payload_pca_fallback_reason"] = f"eigh_failed:{type(exc).__name__}"
        return None, fields
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    valid = eigvals > (float(eigvals[0].item()) * 1e-6 + EPS) if eigvals.numel() else eigvals > 0
    rank = min(int(q), int(valid.sum().item()))
    fields["payload_pca_rank_used"] = int(rank)
    fields["payload_pca_eigvals"] = ",".join(
        f"{float(x):.6g}" for x in eigvals[: min(4, eigvals.numel())].detach().cpu().tolist()
    )
    if rank <= 0:
        fields["payload_pca_fallback_reason"] = "rank_zero"
        return None, fields

    vals_q = eigvals[:rank].clamp_min(EPS)
    vecs_q = eigvecs[:, :rank]
    coeff = basis_mat @ delta_flat
    weights = vecs_q @ ((vecs_q.T @ coeff) / vals_q)
    projected_flat = weights @ basis_mat
    projected = projected_flat.reshape_as(delta)
    residual = delta - projected
    clean_delta = projected + float(gamma) * residual
    clean = (anchor_f + clean_delta).to(dtype=raw_forecast.dtype, device=raw_forecast.device)

    projected_norm = float(projected_flat.norm().item())
    residual_norm = float(residual.reshape(-1).norm().item())
    clean_norm = float(clean_delta.reshape(-1).norm().item())
    fields.update({
        "payload_pca_projected_delta_norm": projected_norm,
        "payload_pca_residual_delta_norm": residual_norm,
        "payload_pca_clean_delta_norm": clean_norm,
        "payload_pca_energy_keep_rel": float((projected_norm * projected_norm) / (delta_norm * delta_norm + EPS)),
        "payload_pca_fallback_reason": None,
    })
    return clean, fields


def _empty_update_inverse_fields() -> Dict[str, Any]:
    return {
        "payload_update_inverse_enabled": False,
        "payload_update_inverse_target_name": None,
        "payload_update_inverse_target_available": None,
        "payload_update_inverse_target_norm": None,
        "payload_update_inverse_q_requested": None,
        "payload_update_inverse_basis_count": None,
        "payload_update_inverse_rank_used": None,
        "payload_update_inverse_basis_eigvals": None,
        "payload_update_inverse_condition": None,
        "payload_update_inverse_ridge_lambda": None,
        "payload_update_inverse_alpha": None,
        "payload_update_inverse_alpha_norm": None,
        "payload_update_inverse_alpha_clipped": None,
        "payload_update_inverse_delta_norm": None,
        "payload_update_inverse_delta_norm_clipped": None,
        "payload_update_inverse_delta_norm_clip_scale": None,
        "payload_update_inverse_eta": None,
        "payload_update_inverse_obj_reuse": None,
        "payload_update_inverse_obj_raw": None,
        "payload_update_inverse_obj_best": None,
        "payload_update_inverse_obj_improvement_rel": None,
        "payload_update_inverse_elapsed_ms": None,
        "payload_update_inverse_fallback_reason": None,
    }


def _empty_update_calib_fields() -> Dict[str, Any]:
    return {
        "payload_update_calib_enabled": False,
        "payload_update_calib_raw_mode": None,
        "payload_update_calib_target_name": None,
        "payload_update_calib_target_available": None,
        "payload_update_calib_target_norm": None,
        "payload_update_calib_alpha_unclipped": None,
        "payload_update_calib_alpha": None,
        "payload_update_calib_alpha_min": None,
        "payload_update_calib_alpha_max": None,
        "payload_update_calib_alpha_clipped": None,
        "payload_update_calib_direction_update_norm": None,
        "payload_update_calib_target_delta_norm": None,
        "payload_update_calib_denom": None,
        "payload_update_calib_cos_raw_target_delta": None,
        "payload_update_calib_obj_reuse": None,
        "payload_update_calib_obj_raw": None,
        "payload_update_calib_obj_calibrated": None,
        "payload_update_calib_obj_improvement_vs_reuse_rel": None,
        "payload_update_calib_obj_improvement_vs_raw_rel": None,
        "payload_update_calib_elapsed_ms": None,
        "payload_update_calib_fallback_reason": None,
    }


def _init_forecast_opt_state() -> Dict[str, Any]:
    return {
        "residual_records": [],
        "hidden_records": [],
        "gate_input_records": [],
        "output_records": [],
        "prebias": {
            "bias": {},
            "updates": {},
            "last_error_abs": {},
            "last_error_rel": {},
        },
    }


def _forecast_opt_records(state: Optional[Dict[str, Any]], key: str) -> list[Dict[str, Any]]:
    records = (state or {}).get(key) or []
    return [row for row in records if isinstance(row, dict)]


def _forecast_opt_needs_output_records(mode: str) -> bool:
    spec = FORECAST_OPT_PAYLOAD_SPECS.get(str(mode))
    return bool(spec is not None and spec.get("control") == "output_target_scalar")


def _record_step_gap(newer: Dict[str, Any], older: Dict[str, Any]) -> int:
    return max(int(newer.get("step")) - int(older.get("step")), 1)


def _forecast_opt_tau(self, step: int, coord: str) -> Optional[float]:
    sigmas = getattr(getattr(self, "scheduler", None), "sigmas", None)
    if sigmas is None:
        return None
    try:
        sigma = _as_float(sigmas[int(step)])
    except Exception:
        return None
    if sigma is None:
        return None
    if coord == "sigma":
        return float(sigma)
    if coord == "logsnr":
        s = float(np.clip(float(sigma), 1e-5, 1.0 - 1e-5))
        alpha = 1.0 - s
        return float(np.log((alpha * alpha + EPS) / (s * s + EPS)))
    return None


def _empty_forecast_opt_fields() -> Dict[str, Any]:
    return {
        "payload_forecast_opt_enabled": False,
        "payload_forecast_opt_mode": None,
        "payload_forecast_opt_raw_mode": None,
        "payload_forecast_opt_fallback_reason": None,
        "payload_forecast_opt_history_updated": False,
        "payload_forecast_opt_history_update_source": None,
        "payload_time_enabled": False,
        "payload_time_coord": None,
        "payload_time_tau_current": None,
        "payload_time_tau_anchor": None,
        "payload_time_tau_prev": None,
        "payload_time_tau_gap_current": None,
        "payload_time_tau_gap_prev": None,
        "payload_time_coef_o1": None,
        "payload_time_prev_step_gap": None,
        "payload_time_delta_norm": None,
        "payload_time_elapsed_ms": None,
        "payload_curv_enabled": False,
        "payload_curv_beta": None,
        "payload_curv_lambda_min": None,
        "payload_curv_anchor_step": None,
        "payload_curv_gap": None,
        "payload_curv_d0_norm": None,
        "payload_curv_d1_norm": None,
        "payload_curv_d2_norm": None,
        "payload_curv_o1_increment_norm": None,
        "payload_curv_o2_term_norm": None,
        "payload_curv_ratio": None,
        "payload_curv_d1_d2_cos": None,
        "payload_curv_lambda_unclipped": None,
        "payload_curv_lambda": None,
        "payload_curv_lambda_clipped": None,
        "payload_curv_elapsed_ms": None,
        "payload_prebias_enabled": False,
        "payload_prebias_raw_mode": None,
        "payload_prebias_updates_pre": None,
        "payload_prebias_gamma": None,
        "payload_prebias_clip_rel": None,
        "payload_prebias_bias_norm": None,
        "payload_prebias_raw_innovation_norm": None,
        "payload_prebias_correction_norm_raw": None,
        "payload_prebias_correction_norm": None,
        "payload_prebias_clip_scale": None,
        "payload_prebias_applied": False,
        "payload_prebias_elapsed_ms": None,
        "payload_prebias_update_source": None,
        "payload_prebias_updated_modes": None,
        "payload_prebias_last_error_abs_taylor_o1": None,
        "payload_prebias_last_error_rel_taylor_o1": None,
        "payload_state_transport_enabled": False,
        "payload_state_transport_gamma": None,
        "payload_state_transport_anchor_step": None,
        "payload_state_transport_prev_step": None,
        "payload_state_transport_gap": None,
        "payload_state_transport_prev_step_gap": None,
        "payload_state_transport_hidden_anchor_norm": None,
        "payload_state_transport_hidden_slope_norm": None,
        "payload_state_transport_current_delta_norm": None,
        "payload_state_transport_time_coef": None,
        "payload_state_transport_beta_unclipped": None,
        "payload_state_transport_beta": None,
        "payload_state_transport_beta_clipped": None,
        "payload_state_transport_coef_final": None,
        "payload_state_transport_coef_delta": None,
        "payload_state_transport_residual_slope_norm": None,
        "payload_state_transport_delta_from_time_norm": None,
        "payload_state_transport_elapsed_ms": None,
        "payload_state_gap_enabled": False,
        "payload_state_gap_gamma": None,
        "payload_state_gap_lambda_min": None,
        "payload_state_gap_anchor_step": None,
        "payload_state_gap_prev_step": None,
        "payload_state_gap_gap": None,
        "payload_state_gap_prev_step_gap": None,
        "payload_state_gap_time_coef": None,
        "payload_state_gap_hidden_anchor_norm": None,
        "payload_state_gap_hidden_slope_norm": None,
        "payload_state_gap_current_delta_norm": None,
        "payload_state_gap_expected_delta_norm": None,
        "payload_state_gap_state_delta_norm": None,
        "payload_state_gap_state_delta_rel": None,
        "payload_state_gap_parallel_coef": None,
        "payload_state_gap_parallel_rel": None,
        "payload_state_gap_orth_norm": None,
        "payload_state_gap_damping_unclipped": None,
        "payload_state_gap_damping": None,
        "payload_state_gap_damping_clipped": False,
        "payload_state_gap_coef_final": None,
        "payload_state_gap_coef_delta": None,
        "payload_state_gap_residual_slope_norm": None,
        "payload_state_gap_delta_from_time_norm": None,
        "payload_state_gap_elapsed_ms": None,
        "payload_lowrank_enabled": False,
        "payload_lowrank_rank_requested": None,
        "payload_lowrank_rank_used": None,
        "payload_lowrank_basis_count": None,
        "payload_lowrank_history_count": None,
        "payload_lowrank_gamma": None,
        "payload_lowrank_residualized": False,
        "payload_lowrank_residualized_applied": False,
        "payload_lowrank_anchor_step": None,
        "payload_lowrank_prev_step": None,
        "payload_lowrank_gap": None,
        "payload_lowrank_prev_step_gap": None,
        "payload_lowrank_time_coef": None,
        "payload_lowrank_hidden_slope_norm": None,
        "payload_lowrank_current_delta_norm": None,
        "payload_lowrank_state_delta_norm": None,
        "payload_lowrank_state_delta_resid_norm": None,
        "payload_lowrank_residual_slope_norm": None,
        "payload_lowrank_raw_innovation_norm": None,
        "payload_lowrank_gram_trace": None,
        "payload_lowrank_gram_min_eig": None,
        "payload_lowrank_gram_max_eig": None,
        "payload_lowrank_gram_eigvals": None,
        "payload_lowrank_condition": None,
        "payload_lowrank_ridge_lambda": None,
        "payload_lowrank_projection_norm": None,
        "payload_lowrank_projection_rel": None,
        "payload_lowrank_alpha_values": None,
        "payload_lowrank_alpha_norm": None,
        "payload_lowrank_alpha_max_abs": None,
        "payload_lowrank_correction_norm_raw": None,
        "payload_lowrank_correction_norm": None,
        "payload_lowrank_correction_raw_ratio": None,
        "payload_lowrank_clip_rel": None,
        "payload_lowrank_clip_scale": None,
        "payload_lowrank_clipped": False,
        "payload_lowrank_correction_time_slope_cos": None,
        "payload_lowrank_delta_from_time_norm": None,
        "payload_lowrank_elapsed_ms": None,
        "payload_output_target_enabled": False,
        "payload_output_target_rank_requested": None,
        "payload_output_target_rank_used": None,
        "payload_output_target_basis_count": None,
        "payload_output_target_history_count": None,
        "payload_output_target_search": None,
        "payload_output_target_grid_count": None,
        "payload_output_target_gamma": None,
        "payload_output_target_ridge_rel": None,
        "payload_output_target_correction_clip_rel": None,
        "payload_output_target_correction_clip_scale": None,
        "payload_output_target_correction_clipped": False,
        "payload_output_target_shrink_only": False,
        "payload_output_target_anchor_step": None,
        "payload_output_target_prev_step": None,
        "payload_output_target_gap": None,
        "payload_output_target_prev_step_gap": None,
        "payload_output_target_time_coef": None,
        "payload_output_target_alpha_min": None,
        "payload_output_target_alpha_max": None,
        "payload_output_target_alpha_time": None,
        "payload_output_target_alpha_unclipped": None,
        "payload_output_target_alpha": None,
        "payload_output_target_alpha_boundary_hit": None,
        "payload_output_target_c_raw_normalized": None,
        "payload_output_target_c_normalized": None,
        "payload_output_target_c_delta_raw_normalized": None,
        "payload_output_target_alpha_values": None,
        "payload_output_target_residual_delta_norm": None,
        "payload_output_target_output_delta_norm": None,
        "payload_output_target_hidden_slope_norm": None,
        "payload_output_target_current_delta_norm": None,
        "payload_output_target_state_delta_norm": None,
        "payload_output_target_time_target_norm": None,
        "payload_output_target_correction_norm": None,
        "payload_output_target_correction_rel": None,
        "payload_output_target_target_norm": None,
        "payload_output_target_gram_trace": None,
        "payload_output_target_gram_min_eig": None,
        "payload_output_target_gram_max_eig": None,
        "payload_output_target_gram_eigvals": None,
        "payload_output_target_condition": None,
        "payload_output_target_ridge_lambda": None,
        "payload_output_target_solve_alpha_values": None,
        "payload_output_target_solve_alpha_norm": None,
        "payload_output_target_projection_norm": None,
        "payload_output_target_projection_rel": None,
        "payload_output_target_obj_reuse": None,
        "payload_output_target_obj_raw": None,
        "payload_output_target_obj_chosen": None,
        "payload_output_target_obj_improvement_vs_reuse_rel": None,
        "payload_output_target_obj_improvement_vs_raw_rel": None,
        "payload_output_target_linear_direction_norm": None,
        "payload_output_target_elapsed_ms": None,
        "payload_input_gap_enabled": False,
        "payload_input_gap_eta": None,
        "payload_input_gap_raw_mode": None,
        "payload_input_gap_anchor_step": None,
        "payload_input_gap_prev_step": None,
        "payload_input_gap_gap": None,
        "payload_input_gap_prev_step_gap": None,
        "payload_input_gap_time_coef": None,
        "payload_input_gap_hidden_anchor_norm": None,
        "payload_input_gap_hidden_slope_norm": None,
        "payload_input_gap_pred_input_norm": None,
        "payload_input_gap_actual_input_norm": None,
        "payload_input_gap_norm": None,
        "payload_input_gap_rel_to_pred_input": None,
        "payload_input_gap_rel_to_actual_input": None,
        "payload_input_gap_residual_anchor_norm": None,
        "payload_input_gap_residual_slope_norm": None,
        "payload_input_gap_raw_payload_norm": None,
        "payload_input_gap_direct_output_payload_norm": None,
        "payload_input_gap_chosen_payload_norm": None,
        "payload_input_gap_correction_norm": None,
        "payload_input_gap_correction_rel_to_raw_payload": None,
        "payload_input_gap_delta_from_raw_norm": None,
        "payload_input_gap_raw_vs_time_residual_norm": None,
        "payload_input_gap_output_forecast_norm": None,
        "payload_input_gap_eta0_equiv_error_norm": None,
        "payload_input_gap_elapsed_ms": None,
        "payload_rfc_rfe_enabled": False,
        "payload_rfc_rfe_raw_mode": None,
        "payload_rfc_rfe_raw_mode_used": None,
        "payload_rfc_rfe_order": None,
        "payload_rfc_rfe_anchor_step": None,
        "payload_rfc_rfe_prev_step": None,
        "payload_rfc_rfe_gap": None,
        "payload_rfc_rfe_prev_step_gap": None,
        "payload_rfc_rfe_input_delta_norm": None,
        "payload_rfc_rfe_hist_input_delta_norm": None,
        "payload_rfc_rfe_hist_output_delta_norm": None,
        "payload_rfc_rfe_s_ratio": None,
        "payload_rfc_rfe_taylor_direction_norm": None,
        "payload_rfc_rfe_magnitude": None,
        "payload_rfc_rfe_delta_from_raw_norm": None,
        "payload_rfc_rfe_delta_from_reuse_norm": None,
        "payload_rfc_rfe_elapsed_ms": None,
    }


def _empty_posterior_oracle_fields() -> Dict[str, Any]:
    return {
        "posterior_oracle_enabled": False,
        "posterior_oracle_family": None,
        "posterior_oracle_ran": False,
        "posterior_oracle_fail_reason": None,
        "posterior_oracle_uses_shadow_full_compute": False,
        "posterior_oracle_updates_full_history": None,
        "posterior_oracle_closed_loop_committed": False,
        "posterior_oracle_anchor_step": None,
        "posterior_oracle_c_raw": None,
        "posterior_oracle_c_star": None,
        "posterior_oracle_c_min": None,
        "posterior_oracle_c_max": None,
        "posterior_oracle_grid_count": None,
        "posterior_oracle_boundary_hit": None,
        "posterior_oracle_payload_norm": None,
        "posterior_oracle_delta_from_reuse_norm": None,
        "posterior_oracle_shadow_residual_norm": None,
        "posterior_oracle_shadow_output_norm": None,
        "posterior_oracle_output_err_abs": None,
        "posterior_oracle_update_err_abs": None,
        "posterior_oracle_c_matches_shadow_output_oracle": None,
        "posterior_oracle_elapsed_ms": None,
    }


def _append_forecast_opt_record(
    state: Optional[Dict[str, Any]],
    *,
    key: str,
    step: int,
    sigma: float,
    tensor: torch.Tensor,
    max_records: int,
) -> Dict[str, Any]:
    st = state if isinstance(state, dict) else _init_forecast_opt_state()
    records = _forecast_opt_records(st, key)
    records.insert(0, {
        "step": int(step),
        "sigma": float(sigma),
        "tensor": tensor.detach().clone(),
    })
    st[key] = records[: int(max_records)]
    return st


def _forecast_opt_update_on_full(
    state: Optional[Dict[str, Any]],
    *,
    residual_history_state: Optional[Dict[str, Any]],
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_input: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    step: int,
    sigma: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    st = state if isinstance(state, dict) else _init_forecast_opt_state()
    fields = {
        "payload_forecast_opt_history_updated": True,
        "payload_forecast_opt_history_update_source": "full",
        "payload_prebias_update_source": "full",
        "payload_prebias_updated_modes": "",
        "payload_prebias_last_error_abs_taylor_o1": None,
        "payload_prebias_last_error_rel_taylor_o1": None,
    }
    preds = forecast_predictions(residual_history_state, step=int(step), sigma=float(sigma))
    actual = residual.detach()
    actual_norm = float(actual.to(torch.float32).norm().item())
    prebias = st.setdefault("prebias", {
        "bias": {},
        "updates": {},
        "last_error_abs": {},
        "last_error_rel": {},
    })
    bias = prebias.setdefault("bias", {})
    updates = prebias.setdefault("updates", {})
    last_error_abs = prebias.setdefault("last_error_abs", {})
    last_error_rel = prebias.setdefault("last_error_rel", {})
    updated_modes = []
    beta = float(FORECAST_OPT_PAYLOAD_SPECS["prebias_taylor_o1"]["ema_beta"])
    for name in ("taylor_o1",):
        pred = preds.get(name)
        if pred is None or tuple(pred.shape) != tuple(actual.shape):
            continue
        err = actual.to(torch.float32) - pred.to(device=actual.device, dtype=torch.float32)
        err_abs = float(err.norm().item())
        err_rel = float(err_abs / (actual_norm + EPS))
        prev = bias.get(name)
        if prev is None or tuple(prev.shape) != tuple(actual.shape):
            new_bias = err.detach().to(dtype=actual.dtype, device=actual.device)
        else:
            new_bias = (
                (1.0 - beta) * prev.detach().to(device=actual.device, dtype=torch.float32)
                + beta * err
            ).to(dtype=actual.dtype, device=actual.device)
        bias[name] = new_bias.detach().clone()
        updates[name] = int(updates.get(name, 0) or 0) + 1
        last_error_abs[name] = err_abs
        last_error_rel[name] = err_rel
        updated_modes.append(name)
    fields["payload_prebias_updated_modes"] = ",".join(updated_modes)
    fields["payload_prebias_last_error_abs_taylor_o1"] = last_error_abs.get("taylor_o1")
    fields["payload_prebias_last_error_rel_taylor_o1"] = last_error_rel.get("taylor_o1")

    st = _append_forecast_opt_record(
        st,
        key="residual_records",
        step=int(step),
        sigma=float(sigma),
        tensor=residual,
        max_records=5,
    )
    st = _append_forecast_opt_record(
        st,
        key="hidden_records",
        step=int(step),
        sigma=float(sigma),
        tensor=hidden_states,
        max_records=5,
    )
    if gate_input is not None:
        st = _append_forecast_opt_record(
            st,
            key="gate_input_records",
            step=int(step),
            sigma=float(sigma),
            tensor=gate_input,
            max_records=5,
        )
    if output is not None:
        st = _append_forecast_opt_record(
            st,
            key="output_records",
            step=int(step),
            sigma=float(sigma),
            tensor=output,
            max_records=5,
        )
    return st, fields


def _forecast_opt_payload(
    self,
    *,
    mode: str,
    reuse: torch.Tensor,
    residual_history_state: Optional[Dict[str, Any]],
    opt_state: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step: int,
    sigma: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    start = time.perf_counter()
    spec = FORECAST_OPT_PAYLOAD_SPECS[str(mode)]
    raw_mode = str(spec["raw_mode"])
    control = str(spec["control"])
    fields = _empty_forecast_opt_fields()
    fields.update(_empty_bank_fields())
    fields.update(_empty_pca_fields())
    fields.update({
        "payload_mode": str(mode),
        "payload_base_mode": raw_mode,
        "payload_control": control,
        "payload_used": "reuse",
        "payload_available": False,
        "payload_fallback": True,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": None,
        "payload_chosen_norm": _norm(reuse),
        "payload_delta_from_reuse_norm": 0.0,
        "payload_forecast_opt_enabled": True,
        "payload_forecast_opt_mode": str(mode),
        "payload_forecast_opt_raw_mode": raw_mode,
    })

    def _finish_fallback(reason: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        fields["payload_forecast_opt_fallback_reason"] = reason
        elapsed = float((time.perf_counter() - start) * 1000.0)
        if control == "time_forecast":
            fields["payload_time_elapsed_ms"] = elapsed
        elif control == "curvature_limited":
            fields["payload_curv_elapsed_ms"] = elapsed
        elif control == "prequential_bias":
            fields["payload_prebias_elapsed_ms"] = elapsed
        elif control == "state_transport":
            fields["payload_state_transport_elapsed_ms"] = elapsed
        elif control == "state_gap_damped":
            fields["payload_state_gap_elapsed_ms"] = elapsed
        elif control == "state_lowrank":
            fields["payload_lowrank_elapsed_ms"] = elapsed
        elif control == "output_target_scalar":
            fields["payload_output_target_elapsed_ms"] = elapsed
        elif control == "input_gap_eta":
            fields["payload_input_gap_elapsed_ms"] = elapsed
        elif control == "rfc_rfe":
            fields["payload_rfc_rfe_elapsed_ms"] = elapsed
        return reuse, fields

    chosen: Optional[torch.Tensor] = None
    raw_forecast: Optional[torch.Tensor] = None

    if control == "time_forecast":
        fields["payload_time_enabled"] = True
        coord = str(spec["coord"])
        fields["payload_time_coord"] = coord
        records = _forecast_opt_records(opt_state, "residual_records")
        if len(records) < 2:
            return _finish_fallback("insufficient_residual_records")
        anchor, prev = records[0], records[1]
        anchor_tensor = anchor.get("tensor")
        prev_tensor = prev.get("tensor")
        if (
            anchor_tensor is None
            or prev_tensor is None
            or tuple(anchor_tensor.shape) != tuple(reuse.shape)
            or tuple(prev_tensor.shape) != tuple(reuse.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        tau_cur = _forecast_opt_tau(self, int(step), coord)
        tau_anchor = _forecast_opt_tau(self, int(anchor["step"]), coord)
        tau_prev = _forecast_opt_tau(self, int(prev["step"]), coord)
        fields.update({
            "payload_time_tau_current": tau_cur,
            "payload_time_tau_anchor": tau_anchor,
            "payload_time_tau_prev": tau_prev,
        })
        if tau_cur is None or tau_anchor is None or tau_prev is None:
            return _finish_fallback("missing_time_coordinate")
        denom = float(tau_anchor) - float(tau_prev)
        if abs(denom) <= EPS:
            return _finish_fallback("zero_time_gap")
        coef = (float(tau_cur) - float(tau_anchor)) / denom
        delta = (
            anchor_tensor.detach().to(device=reuse.device, dtype=torch.float32)
            - prev_tensor.detach().to(device=reuse.device, dtype=torch.float32)
        )
        fields.update({
            "payload_time_tau_gap_current": float(tau_cur) - float(tau_anchor),
            "payload_time_tau_gap_prev": denom,
            "payload_time_coef_o1": float(coef),
            "payload_time_prev_step_gap": int(anchor["step"]) - int(prev["step"]),
            "payload_time_delta_norm": float(delta.norm().item()),
        })
        chosen = (
            anchor_tensor.detach().to(device=reuse.device, dtype=torch.float32)
            + float(coef) * delta
        ).to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = chosen

    elif control == "curvature_limited":
        fields["payload_curv_enabled"] = True
        st = residual_history_state or {}
        history = st.get("history") or {}
        anchor_step = st.get("anchor_step")
        d0 = history.get(0)
        d1 = history.get(1)
        d2 = history.get(2)
        if d0 is None or d1 is None or tuple(d0.shape) != tuple(reuse.shape) or tuple(d1.shape) != tuple(reuse.shape):
            return _finish_fallback("missing_o1_history")
        gap = 0 if anchor_step is None else max(int(step) - int(anchor_step), 0)
        beta = float(spec["beta"])
        lambda_min = float(spec["lambda_min"])
        d1_f = d1.detach().to(device=reuse.device, dtype=torch.float32)
        increment_o1 = float(gap) * d1_f
        inc_norm = float(increment_o1.norm().item())
        curv_term_norm = None
        ratio = 0.0
        if d2 is not None and tuple(d2.shape) == tuple(reuse.shape):
            d2_f = d2.detach().to(device=reuse.device, dtype=torch.float32)
            curv_term = 0.5 * float(gap * gap) * d2_f
            curv_term_norm = float(curv_term.norm().item())
            ratio = float(curv_term_norm / (inc_norm + EPS))
        lambda_unclipped = 1.0 / (1.0 + beta * ratio)
        lambda_value = float(np.clip(lambda_unclipped, lambda_min, 1.0))
        fields.update({
            "payload_curv_beta": beta,
            "payload_curv_lambda_min": lambda_min,
            "payload_curv_anchor_step": None if anchor_step is None else int(anchor_step),
            "payload_curv_gap": int(gap),
            "payload_curv_d0_norm": _norm(d0),
            "payload_curv_d1_norm": _norm(d1),
            "payload_curv_d2_norm": _norm(d2),
            "payload_curv_o1_increment_norm": inc_norm,
            "payload_curv_o2_term_norm": curv_term_norm,
            "payload_curv_ratio": ratio,
            "payload_curv_d1_d2_cos": _cos(d1, d2),
            "payload_curv_lambda_unclipped": float(lambda_unclipped),
            "payload_curv_lambda": lambda_value,
            "payload_curv_lambda_clipped": bool(abs(lambda_value - lambda_unclipped) > 1e-12),
        })
        chosen = (
            d0.detach().to(device=reuse.device, dtype=torch.float32)
            + lambda_value * increment_o1
        ).to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = (
            d0.detach().to(device=reuse.device, dtype=torch.float32)
            + increment_o1
        ).to(dtype=reuse.dtype, device=reuse.device)

    elif control == "prequential_bias":
        fields["payload_prebias_enabled"] = True
        fields["payload_prebias_raw_mode"] = raw_mode
        fields["payload_prebias_gamma"] = float(spec["gamma"])
        fields["payload_prebias_clip_rel"] = float(spec["clip_rel"])
        candidates = _candidate_payloads(
            reuse=reuse,
            history_state=residual_history_state,
            step=int(step),
            sigma=float(sigma),
        )
        raw_forecast = candidates.get(raw_mode)
        if raw_forecast is None or tuple(raw_forecast.shape) != tuple(reuse.shape):
            return _finish_fallback("raw_forecast_unavailable")
        raw_forecast = raw_forecast.to(dtype=reuse.dtype, device=reuse.device)
        prebias = (opt_state or {}).get("prebias") or {}
        bias_map = prebias.get("bias") or {}
        update_map = prebias.get("updates") or {}
        fields["payload_prebias_updates_pre"] = int(update_map.get(raw_mode, 0) or 0)
        bias = bias_map.get(raw_mode)
        if bias is None or tuple(bias.shape) != tuple(reuse.shape):
            chosen = raw_forecast
            fields["payload_prebias_applied"] = False
            fields["payload_prebias_clip_scale"] = 1.0
        else:
            bias_f = bias.detach().to(device=reuse.device, dtype=torch.float32)
            gamma = float(spec["gamma"])
            correction = gamma * bias_f
            correction_norm_raw = float(correction.norm().item())
            innovation_norm = float(
                (raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)).norm().item()
            )
            clip_limit = float(spec["clip_rel"]) * (innovation_norm + EPS)
            clip_scale = 1.0
            if correction_norm_raw > clip_limit > 0.0:
                clip_scale = clip_limit / (correction_norm_raw + EPS)
                correction = correction * clip_scale
            fields.update({
                "payload_prebias_bias_norm": float(bias_f.norm().item()),
                "payload_prebias_raw_innovation_norm": innovation_norm,
                "payload_prebias_correction_norm_raw": correction_norm_raw,
                "payload_prebias_correction_norm": float(correction.norm().item()),
                "payload_prebias_clip_scale": float(clip_scale),
                "payload_prebias_applied": True,
            })
            chosen = (
                raw_forecast.detach().to(torch.float32) + correction
            ).to(dtype=reuse.dtype, device=reuse.device)

    elif control == "state_transport":
        fields["payload_state_transport_enabled"] = True
        fields["payload_state_transport_gamma"] = float(spec["gamma"])
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        if len(records_r) < 2 or len(records_h) < 2:
            return _finish_fallback("insufficient_transport_records")
        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        if int(anchor_r.get("step")) != int(anchor_h.get("step")) or int(prev_r.get("step")) != int(prev_h.get("step")):
            return _finish_fallback("record_step_mismatch")
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = max(anchor_step - prev_step, 1)
        time_coef = float(gap) / float(prev_gap)
        h_delta = (
            h_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
            - h_prev.detach().to(device=hidden_states.device, dtype=torch.float32)
        )
        cur_delta = hidden_states.detach().to(torch.float32) - h_anchor.detach().to(
            device=hidden_states.device, dtype=torch.float32
        )
        denom = float(torch.sum(h_delta * h_delta).item())
        if denom <= EPS:
            return _finish_fallback("zero_hidden_slope")
        beta_unclipped = float(torch.sum(cur_delta * h_delta).item() / (denom + EPS))
        beta_max = max(float(spec["coef_clip"]) * max(time_coef, 1.0), 0.0)
        beta = float(np.clip(beta_unclipped, 0.0, beta_max))
        gamma = float(spec["gamma"])
        coef_final = (1.0 - gamma) * time_coef + gamma * beta
        r_delta = (
            r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
            - r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        )
        time_payload = r_anchor.detach().to(device=reuse.device, dtype=torch.float32) + time_coef * r_delta
        chosen_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32) + coef_final * r_delta
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = time_payload.to(dtype=reuse.dtype, device=reuse.device)
        fields.update({
            "payload_state_transport_anchor_step": anchor_step,
            "payload_state_transport_prev_step": prev_step,
            "payload_state_transport_gap": int(gap),
            "payload_state_transport_prev_step_gap": int(prev_gap),
            "payload_state_transport_hidden_anchor_norm": _norm(h_anchor),
            "payload_state_transport_hidden_slope_norm": float(h_delta.norm().item()),
            "payload_state_transport_current_delta_norm": float(cur_delta.norm().item()),
            "payload_state_transport_time_coef": float(time_coef),
            "payload_state_transport_beta_unclipped": beta_unclipped,
            "payload_state_transport_beta": beta,
            "payload_state_transport_beta_clipped": bool(abs(beta - beta_unclipped) > 1e-12),
            "payload_state_transport_coef_final": float(coef_final),
            "payload_state_transport_coef_delta": float(coef_final - time_coef),
            "payload_state_transport_residual_slope_norm": float(r_delta.norm().item()),
            "payload_state_transport_delta_from_time_norm": float((chosen_f - time_payload).norm().item()),
        })

    elif control == "state_gap_damped":
        fields["payload_state_gap_enabled"] = True
        gamma = float(spec["gamma"])
        lambda_min = float(spec["lambda_min"])
        fields["payload_state_gap_gamma"] = gamma
        fields["payload_state_gap_lambda_min"] = lambda_min
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        if len(records_r) < 2 or len(records_h) < 2:
            return _finish_fallback("insufficient_state_gap_records")
        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        if int(anchor_r.get("step")) != int(anchor_h.get("step")) or int(prev_r.get("step")) != int(prev_h.get("step")):
            return _finish_fallback("record_step_mismatch")
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = max(anchor_step - prev_step, 1)
        time_coef = float(gap) / float(prev_gap)
        h_anchor_f = h_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
        h_prev_f = h_prev.detach().to(device=hidden_states.device, dtype=torch.float32)
        hidden_f = hidden_states.detach().to(device=hidden_states.device, dtype=torch.float32)
        r_anchor_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        r_prev_f = r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        h_delta = h_anchor_f - h_prev_f
        cur_delta = hidden_f - h_anchor_f
        expected_delta = float(time_coef) * h_delta
        state_delta = cur_delta - expected_delta
        expected_norm = float(expected_delta.norm().item())
        state_delta_norm = float(state_delta.norm().item())
        state_delta_rel = float(state_delta_norm / (expected_norm + EPS))
        denom = float(torch.sum(h_delta * h_delta).item())
        parallel_coef = None
        parallel_rel = None
        orth_norm = None
        if denom > EPS:
            parallel_coef = float(torch.sum(state_delta * h_delta).item() / (denom + EPS))
            parallel = parallel_coef * h_delta
            orth = state_delta - parallel
            orth_norm = float(orth.norm().item())
            parallel_rel = float(parallel_coef / (time_coef + EPS)) if abs(time_coef) > EPS else None
        damping_unclipped = 1.0 / (1.0 + gamma * state_delta_rel)
        damping = float(np.clip(damping_unclipped, lambda_min, 1.0))
        coef_final = float(time_coef) * damping
        r_delta = r_anchor_f - r_prev_f
        time_payload = r_anchor_f + time_coef * r_delta
        chosen_f = r_anchor_f + coef_final * r_delta
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = time_payload.to(dtype=reuse.dtype, device=reuse.device)
        fields.update({
            "payload_state_gap_anchor_step": anchor_step,
            "payload_state_gap_prev_step": prev_step,
            "payload_state_gap_gap": int(gap),
            "payload_state_gap_prev_step_gap": int(prev_gap),
            "payload_state_gap_time_coef": float(time_coef),
            "payload_state_gap_hidden_anchor_norm": _norm(h_anchor),
            "payload_state_gap_hidden_slope_norm": float(h_delta.norm().item()),
            "payload_state_gap_current_delta_norm": float(cur_delta.norm().item()),
            "payload_state_gap_expected_delta_norm": expected_norm,
            "payload_state_gap_state_delta_norm": state_delta_norm,
            "payload_state_gap_state_delta_rel": state_delta_rel,
            "payload_state_gap_parallel_coef": parallel_coef,
            "payload_state_gap_parallel_rel": parallel_rel,
            "payload_state_gap_orth_norm": orth_norm,
            "payload_state_gap_damping_unclipped": float(damping_unclipped),
            "payload_state_gap_damping": damping,
            "payload_state_gap_damping_clipped": bool(abs(damping - damping_unclipped) > 1e-12),
            "payload_state_gap_coef_final": coef_final,
            "payload_state_gap_coef_delta": float(coef_final - time_coef),
            "payload_state_gap_residual_slope_norm": float(r_delta.norm().item()),
            "payload_state_gap_delta_from_time_norm": float((chosen_f - time_payload).norm().item()),
        })

    elif control == "state_lowrank":
        fields["payload_lowrank_enabled"] = True
        rank_requested = int(spec["rank"])
        gamma = float(spec["gamma"])
        clip_rel = float(spec["clip_rel"])
        ridge_rel = float(spec["ridge_rel"])
        residualize_time = bool(spec.get("residualize_time", False))
        fields.update({
            "payload_lowrank_rank_requested": rank_requested,
            "payload_lowrank_gamma": gamma,
            "payload_lowrank_clip_rel": clip_rel,
            "payload_lowrank_residualized": residualize_time,
        })
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        history_count = min(len(records_r), len(records_h))
        fields["payload_lowrank_history_count"] = int(history_count)
        if history_count < 2:
            return _finish_fallback("insufficient_lowrank_records")
        for idx in range(history_count):
            if int(records_r[idx].get("step")) != int(records_h[idx].get("step")):
                return _finish_fallback("record_step_mismatch")
        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = max(anchor_step - prev_step, 1)
        time_coef = float(gap) / float(prev_gap)
        h_anchor_f = h_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
        h_prev_f = h_prev.detach().to(device=hidden_states.device, dtype=torch.float32)
        r_anchor_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        r_prev_f = r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        hidden_f = hidden_states.detach().to(device=hidden_states.device, dtype=torch.float32)
        h_delta0 = h_anchor_f - h_prev_f
        r_delta0 = r_anchor_f - r_prev_f
        cur_delta = hidden_f - h_anchor_f
        state_delta = cur_delta - time_coef * h_delta0
        time_payload = r_anchor_f + time_coef * r_delta0
        raw_innovation_norm = float((time_payload - r_anchor_f).norm().item())

        max_pairs = min(rank_requested, history_count - 1)
        h_pairs: list[torch.Tensor] = []
        r_pairs: list[torch.Tensor] = []
        for pair_idx in range(max_pairs):
            newer_r = records_r[pair_idx].get("tensor")
            older_r = records_r[pair_idx + 1].get("tensor")
            newer_h = records_h[pair_idx].get("tensor")
            older_h = records_h[pair_idx + 1].get("tensor")
            if (
                newer_r is None
                or older_r is None
                or newer_h is None
                or older_h is None
                or tuple(newer_r.shape) != tuple(reuse.shape)
                or tuple(older_r.shape) != tuple(reuse.shape)
                or tuple(newer_h.shape) != tuple(hidden_states.shape)
                or tuple(older_h.shape) != tuple(hidden_states.shape)
            ):
                return _finish_fallback("record_shape_mismatch")
            h_pairs.append(
                newer_h.detach().to(device=hidden_states.device, dtype=torch.float32)
                - older_h.detach().to(device=hidden_states.device, dtype=torch.float32)
            )
            r_pairs.append(
                newer_r.detach().to(device=reuse.device, dtype=torch.float32)
                - older_r.detach().to(device=reuse.device, dtype=torch.float32)
            )

        solve_state_delta = state_delta
        solve_h_pairs = h_pairs
        solve_r_pairs = r_pairs
        residualized_applied = False
        if residualize_time:
            denom0 = float(torch.sum(h_delta0 * h_delta0).item())
            if denom0 > EPS and len(h_pairs) > 1:
                beta_x = float(torch.sum(state_delta * h_delta0).item() / (denom0 + EPS))
                solve_state_delta = state_delta - beta_x * h_delta0
                residualized_h: list[torch.Tensor] = []
                residualized_r: list[torch.Tensor] = []
                for dh_i, dr_i in zip(h_pairs[1:], r_pairs[1:]):
                    beta_i = float(torch.sum(dh_i * h_delta0).item() / (denom0 + EPS))
                    residualized_h.append(dh_i - beta_i * h_delta0)
                    residualized_r.append(dr_i - beta_i * r_delta0)
                solve_h_pairs = residualized_h
                solve_r_pairs = residualized_r
                residualized_applied = True
            else:
                solve_state_delta = torch.zeros_like(state_delta)
                solve_h_pairs = []
                solve_r_pairs = []

        rank_used = len(solve_h_pairs)
        correction_raw = torch.zeros_like(r_anchor_f)
        correction = torch.zeros_like(r_anchor_f)
        projection = torch.zeros_like(solve_state_delta)
        eigvals_list = None
        gram_trace = 0.0
        gram_min = 0.0
        gram_max = 0.0
        condition = 0.0
        ridge_lambda = 0.0
        alpha_values: list[float] = []
        alpha_norm = 0.0
        alpha_max_abs = 0.0
        projection_norm = 0.0
        projection_rel = 0.0
        if rank_used > 0:
            q = int(rank_used)
            scale = float(max(int(solve_state_delta.numel()), 1))
            gram = torch.empty((q, q), device=hidden_states.device, dtype=torch.float32)
            rhs = torch.empty((q,), device=hidden_states.device, dtype=torch.float32)
            for i in range(q):
                rhs[i] = torch.sum(solve_state_delta * solve_h_pairs[i]) / scale
                for j in range(q):
                    gram[i, j] = torch.sum(solve_h_pairs[i] * solve_h_pairs[j]) / scale
            eigvals = torch.linalg.eigvalsh(gram).detach().to(torch.float32)
            eigvals_list = [float(x) for x in eigvals.detach().cpu().tolist()]
            gram_trace = float(torch.trace(gram).item())
            gram_min = float(eigvals[0].item()) if eigvals.numel() else 0.0
            gram_max = float(eigvals[-1].item()) if eigvals.numel() else 0.0
            condition = float(gram_max / (gram_min + EPS)) if gram_max > 0.0 else 0.0
            ridge_lambda = float(ridge_rel * (gram_trace / max(q, 1) + EPS))
            system = gram + ridge_lambda * torch.eye(q, device=hidden_states.device, dtype=torch.float32)
            try:
                alpha = torch.linalg.solve(system, rhs)
            except RuntimeError:
                alpha = torch.linalg.pinv(system) @ rhs
            alpha_values = [float(x) for x in alpha.detach().cpu().tolist()]
            alpha_norm = float(alpha.norm().item())
            alpha_max_abs = float(alpha.abs().max().item()) if alpha.numel() else 0.0
            for i in range(q):
                correction_raw = correction_raw + alpha[i] * solve_r_pairs[i]
                projection = projection + alpha[i] * solve_h_pairs[i]
            projection_norm = float(projection.norm().item())
            projection_rel = float(projection_norm / (float(solve_state_delta.norm().item()) + EPS))
            correction = gamma * correction_raw

        correction_norm_raw = float(correction_raw.norm().item())
        correction_norm_preclip = float(correction.norm().item())
        clip_limit = clip_rel * (raw_innovation_norm + EPS)
        clip_scale = 1.0
        if correction_norm_preclip > clip_limit > 0.0:
            clip_scale = float(clip_limit / (correction_norm_preclip + EPS))
            correction = correction * clip_scale
        correction_norm = float(correction.norm().item())
        chosen_f = time_payload + correction
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = time_payload.to(dtype=reuse.dtype, device=reuse.device)
        fields.update({
            "payload_lowrank_rank_used": int(rank_used),
            "payload_lowrank_basis_count": int(rank_used),
            "payload_lowrank_residualized_applied": bool(residualized_applied),
            "payload_lowrank_anchor_step": anchor_step,
            "payload_lowrank_prev_step": prev_step,
            "payload_lowrank_gap": int(gap),
            "payload_lowrank_prev_step_gap": int(prev_gap),
            "payload_lowrank_time_coef": float(time_coef),
            "payload_lowrank_hidden_slope_norm": float(h_delta0.norm().item()),
            "payload_lowrank_current_delta_norm": float(cur_delta.norm().item()),
            "payload_lowrank_state_delta_norm": float(state_delta.norm().item()),
            "payload_lowrank_state_delta_resid_norm": float(solve_state_delta.norm().item()),
            "payload_lowrank_residual_slope_norm": float(r_delta0.norm().item()),
            "payload_lowrank_raw_innovation_norm": raw_innovation_norm,
            "payload_lowrank_gram_trace": float(gram_trace),
            "payload_lowrank_gram_min_eig": float(gram_min),
            "payload_lowrank_gram_max_eig": float(gram_max),
            "payload_lowrank_gram_eigvals": (
                None if eigvals_list is None else ",".join(f"{x:.8g}" for x in eigvals_list)
            ),
            "payload_lowrank_condition": float(condition),
            "payload_lowrank_ridge_lambda": float(ridge_lambda),
            "payload_lowrank_projection_norm": float(projection_norm),
            "payload_lowrank_projection_rel": float(projection_rel),
            "payload_lowrank_alpha_values": ",".join(f"{x:.8g}" for x in alpha_values),
            "payload_lowrank_alpha_norm": float(alpha_norm),
            "payload_lowrank_alpha_max_abs": float(alpha_max_abs),
            "payload_lowrank_correction_norm_raw": correction_norm_raw,
            "payload_lowrank_correction_norm": correction_norm,
            "payload_lowrank_correction_raw_ratio": float(correction_norm / (raw_innovation_norm + EPS)),
            "payload_lowrank_clip_scale": float(clip_scale),
            "payload_lowrank_clipped": bool(clip_scale < 1.0),
            "payload_lowrank_correction_time_slope_cos": _cos(correction, r_delta0),
            "payload_lowrank_delta_from_time_norm": float((chosen_f - time_payload).norm().item()),
        })

    elif control == "rfc_rfe":
        fields["payload_rfc_rfe_enabled"] = True
        fields["payload_rfc_rfe_raw_mode"] = raw_mode
        fields["payload_rfc_rfe_order"] = int(spec.get("order", 1))
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        if len(records_r) < 2 or len(records_h) < 2:
            return _finish_fallback("insufficient_rfc_records")
        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        if int(anchor_r.get("step")) != int(anchor_h.get("step")) or int(prev_r.get("step")) != int(prev_h.get("step")):
            return _finish_fallback("record_step_mismatch")
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        candidates = _candidate_payloads(
            reuse=reuse,
            history_state=residual_history_state,
            step=int(step),
            sigma=float(sigma),
        )
        raw_forecast = candidates.get(raw_mode)
        raw_mode_used = raw_mode
        if (raw_forecast is None or tuple(raw_forecast.shape) != tuple(reuse.shape)) and raw_mode == "taylor_o2":
            raw_forecast = candidates.get("taylor_o1")
            raw_mode_used = "taylor_o1"
        if raw_forecast is None or tuple(raw_forecast.shape) != tuple(reuse.shape):
            return _finish_fallback("raw_forecast_unavailable")
        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = _record_step_gap(anchor_r, prev_r)
        r_anchor_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        r_prev_f = r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        h_anchor_f = h_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
        h_prev_f = h_prev.detach().to(device=hidden_states.device, dtype=torch.float32)
        hidden_f = hidden_states.detach().to(device=hidden_states.device, dtype=torch.float32)
        raw_forecast_f = raw_forecast.detach().to(device=reuse.device, dtype=torch.float32)
        input_delta = hidden_f - h_anchor_f
        hist_input_delta = h_anchor_f - h_prev_f
        hist_output_delta = r_anchor_f - r_prev_f
        input_delta_norm = float(input_delta.norm().item())
        hist_input_delta_norm = float(hist_input_delta.norm().item())
        hist_output_delta_norm = float(hist_output_delta.norm().item())
        if hist_input_delta_norm <= EPS:
            return _finish_fallback("zero_hist_input_delta")
        direction = raw_forecast_f - r_anchor_f
        direction_norm = float(direction.norm().item())
        if direction_norm <= EPS:
            return _finish_fallback("zero_taylor_direction")
        s_ratio = float(hist_output_delta_norm / (hist_input_delta_norm + EPS))
        magnitude = float(s_ratio * input_delta_norm)
        chosen_f = r_anchor_f + magnitude * direction / (direction_norm + EPS)
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = raw_forecast_f.to(dtype=reuse.dtype, device=reuse.device)
        fields.update({
            "payload_rfc_rfe_anchor_step": anchor_step,
            "payload_rfc_rfe_raw_mode_used": raw_mode_used,
            "payload_rfc_rfe_prev_step": prev_step,
            "payload_rfc_rfe_gap": int(gap),
            "payload_rfc_rfe_prev_step_gap": int(prev_gap),
            "payload_rfc_rfe_input_delta_norm": input_delta_norm,
            "payload_rfc_rfe_hist_input_delta_norm": hist_input_delta_norm,
            "payload_rfc_rfe_hist_output_delta_norm": hist_output_delta_norm,
            "payload_rfc_rfe_s_ratio": s_ratio,
            "payload_rfc_rfe_taylor_direction_norm": direction_norm,
            "payload_rfc_rfe_magnitude": magnitude,
            "payload_rfc_rfe_delta_from_raw_norm": float((chosen_f - raw_forecast_f).norm().item()),
            "payload_rfc_rfe_delta_from_reuse_norm": float((chosen_f - reuse.detach().to(torch.float32)).norm().item()),
        })

    elif control == "input_gap_eta":
        fields["payload_input_gap_enabled"] = True
        eta = float(spec.get("eta", 1.0))
        fields["payload_input_gap_eta"] = eta
        fields["payload_input_gap_raw_mode"] = raw_mode
        if raw_mode != "taylor_o1":
            return _finish_fallback(f"unsupported_input_gap_raw_mode:{raw_mode}")
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        if len(records_r) < 2 or len(records_h) < 2:
            return _finish_fallback("insufficient_input_gap_records")
        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        if int(anchor_r.get("step")) != int(anchor_h.get("step")) or int(prev_r.get("step")) != int(prev_h.get("step")):
            return _finish_fallback("record_step_mismatch")
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
            or tuple(hidden_states.shape) != tuple(reuse.shape)
        ):
            return _finish_fallback("record_shape_mismatch")
        candidates = _candidate_payloads(
            reuse=reuse,
            history_state=residual_history_state,
            step=int(step),
            sigma=float(sigma),
        )
        raw_forecast = candidates.get(raw_mode)
        if raw_forecast is None or tuple(raw_forecast.shape) != tuple(reuse.shape):
            return _finish_fallback("raw_forecast_unavailable")
        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = _record_step_gap(anchor_r, prev_r)
        time_coef = float(gap) / float(prev_gap)

        r_anchor_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        r_prev_f = r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        h_anchor_f = h_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        h_prev_f = h_prev.detach().to(device=reuse.device, dtype=torch.float32)
        hidden_f = hidden_states.detach().to(device=reuse.device, dtype=torch.float32)
        raw_forecast_f = raw_forecast.detach().to(device=reuse.device, dtype=torch.float32)

        h_slope = (h_anchor_f - h_prev_f) / float(prev_gap)
        r_slope = (r_anchor_f - r_prev_f) / float(prev_gap)
        pred_input = h_anchor_f + float(gap) * h_slope
        input_gap = hidden_f - pred_input
        direct_output_payload = raw_forecast_f - input_gap
        correction = -eta * input_gap
        chosen_f = raw_forecast_f + correction
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = raw_forecast_f.to(dtype=reuse.dtype, device=reuse.device)
        time_residual = r_anchor_f + float(gap) * r_slope
        output_forecast = pred_input + raw_forecast_f

        input_gap_norm = float(input_gap.norm().item())
        raw_payload_norm = float(raw_forecast_f.norm().item())
        correction_norm = float(correction.norm().item())
        fields.update({
            "payload_input_gap_anchor_step": anchor_step,
            "payload_input_gap_prev_step": prev_step,
            "payload_input_gap_gap": int(gap),
            "payload_input_gap_prev_step_gap": int(prev_gap),
            "payload_input_gap_time_coef": float(time_coef),
            "payload_input_gap_hidden_anchor_norm": float(h_anchor_f.norm().item()),
            "payload_input_gap_hidden_slope_norm": float(h_slope.norm().item()),
            "payload_input_gap_pred_input_norm": float(pred_input.norm().item()),
            "payload_input_gap_actual_input_norm": float(hidden_f.norm().item()),
            "payload_input_gap_norm": input_gap_norm,
            "payload_input_gap_rel_to_pred_input": float(input_gap_norm / (float(pred_input.norm().item()) + EPS)),
            "payload_input_gap_rel_to_actual_input": float(input_gap_norm / (float(hidden_f.norm().item()) + EPS)),
            "payload_input_gap_residual_anchor_norm": float(r_anchor_f.norm().item()),
            "payload_input_gap_residual_slope_norm": float(r_slope.norm().item()),
            "payload_input_gap_raw_payload_norm": raw_payload_norm,
            "payload_input_gap_direct_output_payload_norm": float(direct_output_payload.norm().item()),
            "payload_input_gap_chosen_payload_norm": float(chosen_f.norm().item()),
            "payload_input_gap_correction_norm": correction_norm,
            "payload_input_gap_correction_rel_to_raw_payload": float(correction_norm / (raw_payload_norm + EPS)),
            "payload_input_gap_delta_from_raw_norm": float((chosen_f - raw_forecast_f).norm().item()),
            "payload_input_gap_raw_vs_time_residual_norm": float((raw_forecast_f - time_residual).norm().item()),
            "payload_input_gap_output_forecast_norm": float(output_forecast.norm().item()),
            "payload_input_gap_eta0_equiv_error_norm": (
                float((chosen_f - raw_forecast_f).norm().item()) if abs(eta) <= 1e-12 else None
            ),
        })

    elif control == "output_target_scalar":
        fields["payload_output_target_enabled"] = True
        rank_requested = int(spec["rank"])
        search = str(spec["search"])
        grid_count = int(spec.get("grid_count", 0) or 0)
        gamma = float(spec.get("gamma", 0.0))
        ridge_rel = float(spec.get("ridge_rel", 1e-3))
        correction_clip_rel = float(spec.get("correction_clip_rel", 0.5))
        shrink_only = bool(spec.get("shrink_only", False))
        fields.update({
            "payload_output_target_rank_requested": rank_requested,
            "payload_output_target_search": search,
            "payload_output_target_grid_count": grid_count,
            "payload_output_target_gamma": gamma,
            "payload_output_target_ridge_rel": ridge_rel,
            "payload_output_target_correction_clip_rel": correction_clip_rel,
            "payload_output_target_shrink_only": shrink_only,
        })
        records_r = _forecast_opt_records(opt_state, "residual_records")
        records_h = _forecast_opt_records(opt_state, "hidden_records")
        records_y = _forecast_opt_records(opt_state, "output_records")
        history_count = min(len(records_r), len(records_h), len(records_y))
        fields["payload_output_target_history_count"] = int(history_count)
        if history_count < 2:
            return _finish_fallback("insufficient_output_target_records")
        for idx in range(history_count):
            step_r = int(records_r[idx].get("step"))
            if step_r != int(records_h[idx].get("step")) or step_r != int(records_y[idx].get("step")):
                return _finish_fallback("record_step_mismatch")

        anchor_r, prev_r = records_r[0], records_r[1]
        anchor_h, prev_h = records_h[0], records_h[1]
        anchor_y, prev_y = records_y[0], records_y[1]
        r_anchor = anchor_r.get("tensor")
        r_prev = prev_r.get("tensor")
        h_anchor = anchor_h.get("tensor")
        h_prev = prev_h.get("tensor")
        y_anchor = anchor_y.get("tensor")
        y_prev = prev_y.get("tensor")
        if (
            r_anchor is None
            or r_prev is None
            or h_anchor is None
            or h_prev is None
            or y_anchor is None
            or y_prev is None
            or tuple(r_anchor.shape) != tuple(reuse.shape)
            or tuple(r_prev.shape) != tuple(reuse.shape)
            or tuple(h_anchor.shape) != tuple(hidden_states.shape)
            or tuple(h_prev.shape) != tuple(hidden_states.shape)
            or tuple(y_anchor.shape) != tuple(y_prev.shape)
        ):
            return _finish_fallback("record_shape_mismatch")

        anchor_step = int(anchor_r["step"])
        prev_step = int(prev_r["step"])
        gap = max(int(step) - anchor_step, 0)
        prev_gap = _record_step_gap(anchor_r, prev_r)
        time_coef = float(gap) / float(prev_gap)
        r_anchor_f = r_anchor.detach().to(device=reuse.device, dtype=torch.float32)
        r_prev_f = r_prev.detach().to(device=reuse.device, dtype=torch.float32)
        h_anchor_f = h_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
        h_prev_f = h_prev.detach().to(device=hidden_states.device, dtype=torch.float32)
        hidden_f = hidden_states.detach().to(device=hidden_states.device, dtype=torch.float32)
        y_anchor_f = y_anchor.detach().to(device=hidden_states.device, dtype=torch.float32)
        y_prev_f = y_prev.detach().to(device=hidden_states.device, dtype=torch.float32)

        r_delta0 = r_anchor_f - r_prev_f
        h_delta0 = h_anchor_f - h_prev_f
        y_delta0 = y_anchor_f - y_prev_f
        cur_delta = hidden_f - h_anchor_f
        state_delta = cur_delta - time_coef * h_delta0
        time_payload = r_anchor_f + time_coef * r_delta0
        output_time_delta = time_coef * y_delta0
        output_time_target = y_anchor_f + output_time_delta
        correction_raw = torch.zeros_like(output_time_target)
        correction = torch.zeros_like(output_time_target)
        projection = torch.zeros_like(state_delta)
        rank_used = 0
        basis_count = 0
        gram_trace = 0.0
        gram_min = 0.0
        gram_max = 0.0
        condition = 0.0
        ridge_lambda = 0.0
        eigvals_list = None
        solve_alpha_values: list[float] = []
        solve_alpha_norm = 0.0
        projection_norm = 0.0
        projection_rel = 0.0
        if rank_requested > 0:
            max_pairs = min(rank_requested, history_count - 1)
            h_pairs: list[torch.Tensor] = []
            y_pairs: list[torch.Tensor] = []
            for pair_idx in range(max_pairs):
                newer_h = records_h[pair_idx].get("tensor")
                older_h = records_h[pair_idx + 1].get("tensor")
                newer_y = records_y[pair_idx].get("tensor")
                older_y = records_y[pair_idx + 1].get("tensor")
                if (
                    newer_h is None
                    or older_h is None
                    or newer_y is None
                    or older_y is None
                    or tuple(newer_h.shape) != tuple(hidden_states.shape)
                    or tuple(older_h.shape) != tuple(hidden_states.shape)
                    or tuple(newer_y.shape) != tuple(output_time_target.shape)
                    or tuple(older_y.shape) != tuple(output_time_target.shape)
                ):
                    return _finish_fallback("record_shape_mismatch")
                pair_gap = _record_step_gap(records_h[pair_idx], records_h[pair_idx + 1])
                h_pairs.append(
                    (
                        newer_h.detach().to(device=hidden_states.device, dtype=torch.float32)
                        - older_h.detach().to(device=hidden_states.device, dtype=torch.float32)
                    ) / float(pair_gap)
                )
                y_pairs.append(
                    (
                        newer_y.detach().to(device=hidden_states.device, dtype=torch.float32)
                        - older_y.detach().to(device=hidden_states.device, dtype=torch.float32)
                    ) / float(pair_gap)
                )

            basis_count = len(h_pairs)
            rank_used = len(h_pairs)
            if rank_used > 0:
                q = int(rank_used)
                scale = float(max(int(state_delta.numel()), 1))
                gram = torch.empty((q, q), device=hidden_states.device, dtype=torch.float32)
                rhs = torch.empty((q,), device=hidden_states.device, dtype=torch.float32)
                for i in range(q):
                    rhs[i] = torch.sum(state_delta * h_pairs[i]) / scale
                    for j in range(q):
                        gram[i, j] = torch.sum(h_pairs[i] * h_pairs[j]) / scale
                eigvals = torch.linalg.eigvalsh(gram).detach().to(torch.float32)
                eigvals_list = [float(x) for x in eigvals.detach().cpu().tolist()]
                gram_trace = float(torch.trace(gram).item())
                gram_min = float(eigvals[0].item()) if eigvals.numel() else 0.0
                gram_max = float(eigvals[-1].item()) if eigvals.numel() else 0.0
                condition = float(gram_max / (gram_min + EPS)) if gram_max > 0.0 else 0.0
                ridge_lambda = float(ridge_rel * (gram_trace / max(q, 1) + EPS))
                system = gram + ridge_lambda * torch.eye(q, device=hidden_states.device, dtype=torch.float32)
                try:
                    solve_alpha = torch.linalg.solve(system, rhs)
                except RuntimeError:
                    solve_alpha = torch.linalg.pinv(system) @ rhs
                solve_alpha_values = [float(x) for x in solve_alpha.detach().cpu().tolist()]
                solve_alpha_norm = float(solve_alpha.norm().item())
                for i in range(q):
                    correction_raw = correction_raw + solve_alpha[i] * y_pairs[i]
                    projection = projection + solve_alpha[i] * h_pairs[i]
                projection_norm = float(projection.norm().item())
                projection_rel = float(projection_norm / (float(state_delta.norm().item()) + EPS))
                correction = gamma * correction_raw

        correction_norm_preclip = float(correction.norm().item())
        output_time_delta_norm = float(output_time_delta.norm().item())
        correction_clip_scale = 1.0
        clip_limit = correction_clip_rel * (output_time_delta_norm + EPS)
        if correction_norm_preclip > clip_limit > 0.0:
            correction_clip_scale = float(clip_limit / (correction_norm_preclip + EPS))
            correction = correction * correction_clip_scale
        target_output = output_time_target + correction

        c_raw_normalized = float(gap)
        c_max_normalized = c_raw_normalized if shrink_only else float(max(4.0, 2.0 * c_raw_normalized + 2.0))
        alpha_min = 0.0
        alpha_max = max(float(c_max_normalized) / float(prev_gap), float(time_coef), 0.0)
        if shrink_only:
            alpha_max = max(float(time_coef), 0.0)
        fields.update({
            "payload_output_target_rank_used": int(rank_used),
            "payload_output_target_basis_count": int(basis_count),
            "payload_output_target_anchor_step": anchor_step,
            "payload_output_target_prev_step": prev_step,
            "payload_output_target_gap": int(gap),
            "payload_output_target_prev_step_gap": int(prev_gap),
            "payload_output_target_time_coef": float(time_coef),
            "payload_output_target_alpha_min": float(alpha_min),
            "payload_output_target_alpha_max": float(alpha_max),
            "payload_output_target_alpha_time": float(time_coef),
            "payload_output_target_c_raw_normalized": c_raw_normalized,
            "payload_output_target_residual_delta_norm": float(r_delta0.norm().item()),
            "payload_output_target_output_delta_norm": float(y_delta0.norm().item()),
            "payload_output_target_hidden_slope_norm": float(h_delta0.norm().item()),
            "payload_output_target_current_delta_norm": float(cur_delta.norm().item()),
            "payload_output_target_state_delta_norm": float(state_delta.norm().item()),
            "payload_output_target_time_target_norm": float(output_time_target.norm().item()),
            "payload_output_target_correction_norm": float(correction.norm().item()),
            "payload_output_target_correction_rel": float(correction.norm().item() / (output_time_delta_norm + EPS)),
            "payload_output_target_correction_clip_scale": float(correction_clip_scale),
            "payload_output_target_correction_clipped": bool(correction_clip_scale < 1.0),
            "payload_output_target_target_norm": float(target_output.norm().item()),
            "payload_output_target_gram_trace": float(gram_trace),
            "payload_output_target_gram_min_eig": float(gram_min),
            "payload_output_target_gram_max_eig": float(gram_max),
            "payload_output_target_gram_eigvals": (
                None if eigvals_list is None else ",".join(f"{x:.8g}" for x in eigvals_list)
            ),
            "payload_output_target_condition": float(condition),
            "payload_output_target_ridge_lambda": float(ridge_lambda),
            "payload_output_target_solve_alpha_values": ",".join(f"{x:.8g}" for x in solve_alpha_values),
            "payload_output_target_solve_alpha_norm": float(solve_alpha_norm),
            "payload_output_target_projection_norm": float(projection_norm),
            "payload_output_target_projection_rel": float(projection_rel),
        })

        def _payload_from_alpha(alpha_value: float) -> torch.Tensor:
            return (r_anchor_f + float(alpha_value) * r_delta0).to(dtype=reuse.dtype, device=reuse.device)

        target_f = target_output.detach().to(device=hidden_states.device, dtype=torch.float32)
        with torch.inference_mode():
            reuse_output = _project_output(self, hidden_states + _payload_from_alpha(0.0), temb)
            raw_output = _project_output(self, hidden_states + _payload_from_alpha(time_coef), temb)
        if tuple(reuse_output.shape) != tuple(target_f.shape) or tuple(raw_output.shape) != tuple(target_f.shape):
            return _finish_fallback("output_shape_mismatch")
        obj_reuse = _err_norm(reuse_output, target_f)
        obj_raw = _err_norm(raw_output, target_f)
        best_alpha = 0.0
        best_obj = obj_reuse
        alpha_unclipped = None
        alpha_values: list[float] = []
        if obj_raw < best_obj:
            best_alpha = float(time_coef)
            best_obj = obj_raw
        if search == "linear":
            output_direction = raw_output.detach().to(torch.float32) - reuse_output.detach().to(torch.float32)
            target_delta = target_f - reuse_output.detach().to(torch.float32)
            denom = float(torch.sum(output_direction * output_direction).item())
            fields["payload_output_target_linear_direction_norm"] = float(output_direction.norm().item())
            if denom > EPS and abs(float(time_coef)) > EPS:
                raw_fraction = float(torch.sum(output_direction * target_delta).item() / (denom + EPS))
                alpha_unclipped = float(raw_fraction * float(time_coef))
            else:
                alpha_unclipped = float(time_coef)
            alpha_candidate = float(np.clip(alpha_unclipped, alpha_min, alpha_max))
            alpha_values = _unique_float_values([0.0, time_coef, alpha_candidate, alpha_max])
        elif search == "grid":
            alpha_values = np.linspace(alpha_min, alpha_max, num=max(int(grid_count), 2), dtype=np.float64).tolist()
            alpha_values.extend([0.0, time_coef])
            alpha_values = _unique_float_values(alpha_values)
        else:
            return _finish_fallback(f"unknown_output_target_search:{search}")

        with torch.inference_mode():
            for alpha_value in alpha_values:
                candidate_payload = _payload_from_alpha(float(alpha_value))
                candidate_output = _project_output(self, hidden_states + candidate_payload, temb)
                obj = _err_norm(candidate_output, target_f)
                if obj < best_obj:
                    best_alpha = float(alpha_value)
                    best_obj = float(obj)

        chosen_f = r_anchor_f + best_alpha * r_delta0
        chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
        raw_forecast = time_payload.to(dtype=reuse.dtype, device=reuse.device)
        c_normalized = float(best_alpha * float(prev_gap))
        fields.update({
            "payload_output_target_alpha_unclipped": alpha_unclipped,
            "payload_output_target_alpha": float(best_alpha),
            "payload_output_target_alpha_boundary_hit": bool(
                abs(best_alpha - alpha_min) <= 1e-8 or abs(best_alpha - alpha_max) <= 1e-8
            ),
            "payload_output_target_c_normalized": c_normalized,
            "payload_output_target_c_delta_raw_normalized": float(c_normalized - c_raw_normalized),
            "payload_output_target_alpha_values": ",".join(f"{x:.8g}" for x in alpha_values),
            "payload_output_target_obj_reuse": obj_reuse,
            "payload_output_target_obj_raw": obj_raw,
            "payload_output_target_obj_chosen": float(best_obj),
            "payload_output_target_obj_improvement_vs_reuse_rel": float((obj_reuse - best_obj) / (obj_reuse + EPS)),
            "payload_output_target_obj_improvement_vs_raw_rel": float((obj_raw - best_obj) / (obj_raw + EPS)),
        })

    else:
        return _finish_fallback(f"unknown_forecast_opt_control:{control}")

    if chosen is None or tuple(chosen.shape) != tuple(reuse.shape):
        return _finish_fallback("chosen_unavailable")
    fields.update({
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_forecast_norm": _norm(raw_forecast),
        "payload_chosen_norm": _norm(chosen),
        "payload_delta_from_reuse_norm": _norm(
            chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)
        ),
        "payload_forecast_opt_fallback_reason": None,
    })
    elapsed = float((time.perf_counter() - start) * 1000.0)
    if control == "time_forecast":
        fields["payload_time_elapsed_ms"] = elapsed
    elif control == "curvature_limited":
        fields["payload_curv_elapsed_ms"] = elapsed
    elif control == "prequential_bias":
        fields["payload_prebias_elapsed_ms"] = elapsed
    elif control == "state_transport":
        fields["payload_state_transport_elapsed_ms"] = elapsed
    elif control == "state_gap_damped":
        fields["payload_state_gap_elapsed_ms"] = elapsed
    elif control == "state_lowrank":
        fields["payload_lowrank_elapsed_ms"] = elapsed
    elif control == "output_target_scalar":
        fields["payload_output_target_elapsed_ms"] = elapsed
    elif control == "input_gap_eta":
        fields["payload_input_gap_elapsed_ms"] = elapsed
    elif control == "rfc_rfe":
        fields["payload_rfc_rfe_elapsed_ms"] = elapsed
    return chosen, fields


def _basis_rank_fields(basis: list[torch.Tensor]) -> Dict[str, Any]:
    if not basis:
        return {
            "payload_update_inverse_rank_used": 0,
            "payload_update_inverse_basis_eigvals": "",
            "payload_update_inverse_condition": None,
        }
    basis_mat = torch.stack([b.detach().to(torch.float32).reshape(-1) for b in basis], dim=0)
    gram = basis_mat @ basis_mat.T
    eigvals = torch.linalg.eigvalsh(gram).detach().to(torch.float32)
    eigvals = torch.sort(eigvals, descending=True).values
    if eigvals.numel() == 0:
        rank = 0
        cond = None
    else:
        max_eig = float(eigvals[0].item())
        valid = eigvals > (max_eig * 1e-6 + EPS)
        rank = int(valid.sum().item())
        if rank <= 0:
            cond = None
        else:
            min_eig = float(eigvals[rank - 1].item())
            cond = float(max_eig / (min_eig + EPS))
    return {
        "payload_update_inverse_rank_used": int(rank),
        "payload_update_inverse_basis_eigvals": ",".join(
            f"{float(x):.6g}" for x in eigvals[: min(4, eigvals.numel())].cpu().tolist()
        ),
        "payload_update_inverse_condition": cond,
    }


def _update_inverse_payload(
    self,
    *,
    mode: str,
    reuse: torch.Tensor,
    residual_history_state: Optional[Dict[str, Any]],
    update_history_state: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step: int,
    sigma: float,
    step_size: Optional[float],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Optional[torch.Tensor]]]:
    fields = _empty_update_inverse_fields()
    fields.update(_empty_bank_fields())
    fields.update({
        "payload_mode": str(mode),
        "payload_base_mode": "update_target",
        "payload_control": "inverse_basis",
        "payload_used": "reuse",
        "payload_available": False,
        "payload_fallback": True,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": None,
        "payload_chosen_norm": _norm(reuse),
        "payload_delta_from_reuse_norm": 0.0,
        "payload_update_inverse_enabled": True,
    })
    aux: Dict[str, Optional[torch.Tensor]] = {
        "target_payload": None,
        "target_output": None,
        "target_update": None,
    }
    _maybe_sync(hidden_states)
    start = time.perf_counter()

    spec = UPDATE_INVERSE_PAYLOAD_SPECS[str(mode)]
    q = int(spec["q"])
    fields["payload_update_inverse_q_requested"] = int(q)
    target_name, target_update = update_history_best_target(
        update_history_state,
        step=int(step),
        sigma=float(sigma),
        preferred=str(spec["target"]),
    )
    fields.update({
        "payload_update_inverse_target_name": target_name,
        "payload_update_inverse_target_available": target_update is not None,
        "payload_update_inverse_target_norm": _norm(target_update),
    })
    aux["target_update"] = target_update
    if target_update is None:
        fields["payload_update_inverse_fallback_reason"] = "target_unavailable"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux
    if step_size is None:
        fields["payload_update_inverse_fallback_reason"] = "missing_step_size"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    st = residual_history_state or {}
    history = st.get("history") or {}
    basis: list[torch.Tensor] = []
    for key in sorted(int(k) for k in history.keys() if int(k) > 0):
        tensor = history.get(key)
        if tensor is not None and tuple(tensor.shape) == tuple(reuse.shape):
            basis.append(tensor.detach().to(device=reuse.device, dtype=torch.float32))
        if len(basis) >= q:
            break
    fields["payload_update_inverse_basis_count"] = int(len(basis))
    try:
        fields.update(_basis_rank_fields(basis))
    except RuntimeError as exc:
        fields["payload_update_inverse_fallback_reason"] = f"basis_eigh_failed:{type(exc).__name__}"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux
    if len(basis) < q:
        fields["payload_update_inverse_fallback_reason"] = "insufficient_basis"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux
    if int(fields.get("payload_update_inverse_rank_used") or 0) < q:
        fields["payload_update_inverse_fallback_reason"] = "rank_deficient"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    target_f = target_update.detach().to(device=reuse.device, dtype=torch.float32)
    if tuple(target_f.shape) != tuple(_project_output(self, hidden_states + reuse, temb).shape):
        fields["payload_update_inverse_fallback_reason"] = "target_shape_mismatch"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    step_size_f = float(step_size)
    with torch.inference_mode():
        reuse_output = _project_output(self, hidden_states + reuse, temb)
        reuse_update = reuse_output.detach().to(torch.float32) * step_size_f
        cols = []
        for b in basis:
            cand_output = _project_output(
                self,
                hidden_states + reuse + b.to(device=reuse.device, dtype=reuse.dtype),
                temb,
            )
            cols.append(cand_output.detach().to(torch.float32) * step_size_f - reuse_update)
    target_delta = (target_f - reuse_update).reshape(-1)
    obj_reuse = float(target_delta.norm().item())
    fields["payload_update_inverse_obj_reuse"] = obj_reuse
    if not cols:
        fields["payload_update_inverse_fallback_reason"] = "missing_response_columns"
        fields["payload_update_inverse_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    response = torch.stack([c.reshape(-1) for c in cols], dim=1)
    gram = response.T @ response
    rhs = response.T @ target_delta
    trace = float(torch.trace(gram).item())
    ridge = float(spec["ridge_rel"]) * (trace / max(int(response.shape[1]), 1) + EPS)
    fields["payload_update_inverse_ridge_lambda"] = ridge
    try:
        alpha = torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
            rhs,
        )
    except RuntimeError:
        alpha = torch.linalg.lstsq(
            gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype),
            rhs,
        ).solution

    alpha_clip = float(spec["alpha_abs_clip"])
    alpha_clamped = alpha.clamp(min=-alpha_clip, max=alpha_clip)
    fields["payload_update_inverse_alpha_clipped"] = bool(
        torch.max(torch.abs(alpha - alpha_clamped)).item() > 0.0
    )
    alpha = alpha_clamped
    fields["payload_update_inverse_alpha"] = ",".join(
        f"{float(x):.6g}" for x in alpha.detach().cpu().tolist()
    )
    fields["payload_update_inverse_alpha_norm"] = float(alpha.norm().item())

    delta = torch.zeros_like(reuse, dtype=torch.float32)
    for coeff, b in zip(alpha, basis):
        delta = delta + coeff * b.to(device=reuse.device, dtype=torch.float32)
    delta_norm = float(delta.norm().item())
    fields["payload_update_inverse_delta_norm"] = delta_norm
    reuse_norm = float(reuse.detach().to(torch.float32).norm().item())
    clip_limit = float(spec["delta_norm_clip_rel"]) * (reuse_norm + EPS)
    clip_scale = 1.0
    if delta_norm > clip_limit > 0.0:
        clip_scale = clip_limit / (delta_norm + EPS)
        delta = delta * clip_scale
        delta_norm = float(delta.norm().item())
    fields["payload_update_inverse_delta_norm_clipped"] = delta_norm
    fields["payload_update_inverse_delta_norm_clip_scale"] = float(clip_scale)

    best_eta = 0.0
    best_obj = obj_reuse
    best_payload = reuse
    raw_obj = None
    with torch.inference_mode():
        for eta in (0.0, 0.25, 0.5, 1.0):
            cand = reuse.detach().to(torch.float32) + float(eta) * delta
            cand_t = cand.to(dtype=reuse.dtype, device=reuse.device)
            cand_output = _project_output(self, hidden_states + cand_t, temb)
            cand_update = cand_output.detach().to(torch.float32) * step_size_f
            obj = float((cand_update - target_f).norm().item())
            if eta == 1.0:
                raw_obj = obj
            if obj < best_obj:
                best_obj = obj
                best_eta = float(eta)
                best_payload = cand_t
    fields.update({
        "payload_update_inverse_eta": float(best_eta),
        "payload_update_inverse_obj_raw": raw_obj,
        "payload_update_inverse_obj_best": float(best_obj),
        "payload_update_inverse_obj_improvement_rel": float((obj_reuse - best_obj) / (obj_reuse + EPS)),
        "payload_update_inverse_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        "payload_update_inverse_fallback_reason": None,
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_forecast_norm": _norm(best_payload),
        "payload_chosen_norm": _norm(best_payload),
        "payload_delta_from_reuse_norm": _norm(best_payload.detach().to(torch.float32) - reuse.detach().to(torch.float32)),
    })
    _maybe_sync(hidden_states)
    return best_payload, fields, aux


def _update_scalar_calib_payload(
    self,
    *,
    mode: str,
    reuse: torch.Tensor,
    residual_history_state: Optional[Dict[str, Any]],
    update_history_state: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step: int,
    sigma: float,
    step_size: Optional[float],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Optional[torch.Tensor]]]:
    fields = _empty_update_calib_fields()
    fields.update(_empty_bank_fields())
    fields.update({
        "payload_mode": str(mode),
        "payload_base_mode": _payload_spec(str(mode))[0],
        "payload_control": "update_scalar_calib",
        "payload_used": "reuse",
        "payload_available": False,
        "payload_fallback": True,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": None,
        "payload_chosen_norm": _norm(reuse),
        "payload_delta_from_reuse_norm": 0.0,
        "payload_update_calib_enabled": True,
    })
    aux: Dict[str, Optional[torch.Tensor]] = {
        "target_payload": None,
        "target_output": None,
        "target_update": None,
    }
    _maybe_sync(hidden_states)
    start = time.perf_counter()

    spec = UPDATE_SCALAR_CALIB_PAYLOAD_SPECS[str(mode)]
    raw_mode = str(spec["raw_mode"])
    target_preferred = str(spec["target"])
    alpha_min = float(spec["alpha_min"])
    alpha_max = float(spec["alpha_max"])
    fields.update({
        "payload_update_calib_raw_mode": raw_mode,
        "payload_update_calib_alpha_min": alpha_min,
        "payload_update_calib_alpha_max": alpha_max,
    })

    candidates = _candidate_payloads(
        reuse=reuse,
        history_state=residual_history_state,
        step=int(step),
        sigma=float(sigma),
    )
    raw_forecast = candidates.get(raw_mode)
    if raw_forecast is None or tuple(raw_forecast.shape) != tuple(reuse.shape):
        fields["payload_update_calib_fallback_reason"] = "raw_unavailable"
        fields["payload_update_calib_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux
    raw_forecast = raw_forecast.to(dtype=reuse.dtype, device=reuse.device)
    fields["payload_forecast_norm"] = _norm(raw_forecast)

    target_name, target_update = update_history_best_target(
        update_history_state,
        step=int(step),
        sigma=float(sigma),
        preferred=target_preferred,
    )
    fields.update({
        "payload_update_calib_target_name": target_name,
        "payload_update_calib_target_available": target_update is not None,
        "payload_update_calib_target_norm": _norm(target_update),
    })
    aux["target_update"] = target_update
    if target_update is None:
        fields.update({
            "payload_used": raw_mode,
            "payload_available": True,
            "payload_chosen_norm": _norm(raw_forecast),
            "payload_delta_from_reuse_norm": _norm(
                raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
            "payload_update_calib_fallback_reason": "target_unavailable",
            "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        })
        return raw_forecast, fields, aux
    if step_size is None:
        fields.update({
            "payload_used": raw_mode,
            "payload_available": True,
            "payload_chosen_norm": _norm(raw_forecast),
            "payload_delta_from_reuse_norm": _norm(
                raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
            "payload_update_calib_fallback_reason": "missing_step_size",
            "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        })
        return raw_forecast, fields, aux

    step_size_f = float(step_size)
    target_f = target_update.detach().to(device=reuse.device, dtype=torch.float32)
    with torch.inference_mode():
        reuse_output = _project_output(self, hidden_states + reuse, temb)
        raw_output = _project_output(self, hidden_states + raw_forecast, temb)
    if tuple(target_f.shape) != tuple(reuse_output.shape):
        fields.update({
            "payload_used": raw_mode,
            "payload_available": True,
            "payload_chosen_norm": _norm(raw_forecast),
            "payload_delta_from_reuse_norm": _norm(
                raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
            "payload_update_calib_fallback_reason": "target_shape_mismatch",
            "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        })
        return raw_forecast, fields, aux

    reuse_update = reuse_output.detach().to(torch.float32) * step_size_f
    raw_update = raw_output.detach().to(torch.float32) * step_size_f
    update_direction = raw_update - reuse_update
    target_delta = target_f - reuse_update
    direction_norm = float(update_direction.norm().item())
    target_delta_norm = float(target_delta.norm().item())
    denom = float(torch.sum(update_direction * update_direction).item())
    fields.update({
        "payload_update_calib_direction_update_norm": direction_norm,
        "payload_update_calib_target_delta_norm": target_delta_norm,
        "payload_update_calib_denom": denom,
        "payload_update_calib_cos_raw_target_delta": (
            None if direction_norm <= 0.0 or target_delta_norm <= 0.0
            else float(torch.sum(update_direction * target_delta).item()) / (
                direction_norm * target_delta_norm + EPS
            )
        ),
    })

    obj_reuse = target_delta_norm
    obj_raw = float((raw_update - target_f).norm().item())
    fields.update({
        "payload_update_calib_obj_reuse": obj_reuse,
        "payload_update_calib_obj_raw": obj_raw,
    })
    if denom <= EPS:
        fields.update({
            "payload_used": raw_mode,
            "payload_available": True,
            "payload_chosen_norm": _norm(raw_forecast),
            "payload_delta_from_reuse_norm": _norm(
                raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
            "payload_update_calib_fallback_reason": "zero_update_direction",
            "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        })
        return raw_forecast, fields, aux

    alpha_raw = float(torch.sum(update_direction * target_delta).item() / (denom + EPS))
    alpha = min(max(alpha_raw, alpha_min), alpha_max)
    fields.update({
        "payload_update_calib_alpha_unclipped": alpha_raw,
        "payload_update_calib_alpha": alpha,
        "payload_update_calib_alpha_clipped": bool(alpha != alpha_raw),
    })

    chosen_f = reuse.detach().to(torch.float32) + float(alpha) * (
        raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
    )
    chosen = chosen_f.to(dtype=reuse.dtype, device=reuse.device)
    try:
        with torch.inference_mode():
            calib_output = _project_output(self, hidden_states + chosen, temb)
        calib_update = calib_output.detach().to(torch.float32) * step_size_f
        obj_calib = float((calib_update - target_f).norm().item())
    except RuntimeError as exc:
        fields.update({
            "payload_used": raw_mode,
            "payload_available": True,
            "payload_chosen_norm": _norm(raw_forecast),
            "payload_delta_from_reuse_norm": _norm(
                raw_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
            "payload_update_calib_fallback_reason": f"calibrated_eval_failed:{type(exc).__name__}",
            "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        })
        return raw_forecast, fields, aux

    fields.update({
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_chosen_norm": _norm(chosen),
        "payload_delta_from_reuse_norm": _norm(
            chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)
        ),
        "payload_update_calib_obj_calibrated": obj_calib,
        "payload_update_calib_obj_improvement_vs_reuse_rel": float(
            (obj_reuse - obj_calib) / (obj_reuse + EPS)
        ),
        "payload_update_calib_obj_improvement_vs_raw_rel": float(
            (obj_raw - obj_calib) / (obj_raw + EPS)
        ),
        "payload_update_calib_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        "payload_update_calib_fallback_reason": None,
    })
    _maybe_sync(hidden_states)
    return chosen, fields, aux


def _empty_bank_fields() -> Dict[str, Any]:
    return {
        "payload_bank_dir": None,
        "payload_wrong_seed_bank_dir": None,
        "payload_bank_source_kind": None,
        "payload_bank_source_prompt_idx": None,
        "payload_bank_source_step": None,
        "payload_bank_source_path": None,
        "payload_bank_loaded": False,
        "payload_bank_missing": False,
        "payload_bank_error": None,
        "payload_bank_write_dir": None,
        "payload_bank_write_path": None,
        "payload_bank_write_ok": None,
        "payload_bank_write_error": None,
    }


def _empty_selector_fields() -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "payload_selector_enabled": False,
        "payload_selector_mode": None,
        "payload_selector_target_space": None,
        "payload_selector_target_available": None,
        "payload_selector_target_name": None,
        "payload_selector_target_norm": None,
        "payload_selector_anchor_step": None,
        "payload_selector_anchor_gap": None,
        "payload_selector_order_avail": None,
        "payload_selector_candidates": None,
        "payload_selector_candidate_count": None,
        "payload_selector_scored_count": None,
        "payload_selector_chosen": None,
        "payload_selector_chosen_score_abs": None,
        "payload_selector_chosen_score_rel": None,
        "payload_selector_fallback_reason": None,
        "payload_selector_elapsed_ms": None,
    }
    for name in BASE_PAYLOAD_MODES:
        fields.update({
            f"candidate_{name}_available": None,
            f"candidate_{name}_payload_norm": None,
            f"candidate_{name}_output_norm": None,
            f"candidate_{name}_update_norm": None,
            f"candidate_{name}_score_to_target_abs": None,
            f"candidate_{name}_score_to_target_rel": None,
            f"candidate_{name}_rank": None,
        })
    return fields


def _empty_pca_fields() -> Dict[str, Any]:
    return {
        "payload_pca_enabled": False,
        "payload_pca_raw_mode": None,
        "payload_pca_q": None,
        "payload_pca_gamma": None,
        "payload_pca_basis_count": None,
        "payload_pca_rank_used": None,
        "payload_pca_anchor_norm": None,
        "payload_pca_raw_delta_norm": None,
        "payload_pca_projected_delta_norm": None,
        "payload_pca_residual_delta_norm": None,
        "payload_pca_clean_delta_norm": None,
        "payload_pca_energy_keep_rel": None,
        "payload_pca_eigvals": None,
        "payload_pca_fallback_reason": None,
    }


def _selector_history_digest(state: Optional[Dict[str, Any]], step: int) -> Dict[str, Any]:
    st = state or {}
    history = st.get("history") or {}
    anchor_step = st.get("anchor_step")
    return {
        "anchor_step": None if anchor_step is None else int(anchor_step),
        "anchor_gap": None if anchor_step is None else int(step) - int(anchor_step),
        "order_avail": max(history.keys()) if history else -1,
    }


def _bank_prompt_dir(bank_dir: Path, prompt_idx: int) -> Path:
    return bank_dir / f"prompt_{int(prompt_idx):05d}"


def _bank_step_path(bank_dir: Path, prompt_idx: int, step: int) -> Path:
    return _bank_prompt_dir(bank_dir, int(prompt_idx)) / f"step_{int(step):03d}.pt"


def _bank_prompt_mean_path(bank_dir: Path, prompt_idx: int) -> Path:
    return _bank_prompt_dir(bank_dir, int(prompt_idx)) / "prompt_mean.pt"


def _bank_step_mean_path(bank_dir: Path, step: int) -> Path:
    return bank_dir / "step_mean" / f"step_{int(step):03d}.pt"


@lru_cache(maxsize=16)
def _read_bank_index(bank_dir_str: str) -> Dict[str, Any]:
    path = Path(bank_dir_str) / "index.json"
    if not path.is_file():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _lookup_bank_index(index: Dict[str, Any], key: str, prompt_idx: int, step: int) -> Optional[int]:
    block = index.get(key)
    if not isinstance(block, dict):
        return None
    prompt_block = block.get(str(int(prompt_idx)))
    if not isinstance(prompt_block, dict):
        return None
    value = prompt_block.get(str(int(step)))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _load_bank_tensor(path: Path, reference: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = torch.load(path, map_location="cpu")
        tensor = payload.get("tensor") if isinstance(payload, dict) else payload
        if tensor is None or not hasattr(tensor, "shape"):
            return None, "no_tensor"
        if tuple(tensor.shape) != tuple(reference.shape):
            return None, f"shape_mismatch:{tuple(tensor.shape)}!={tuple(reference.shape)}"
        return tensor.to(dtype=reference.dtype, device=reference.device), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _bank_control_tensor(
    *,
    control: str,
    bank_dir: Optional[Path],
    wrong_seed_bank_dir: Optional[Path],
    prompt_idx: Optional[int],
    step: int,
    reuse: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    fields = _empty_bank_fields()
    fields["payload_bank_dir"] = None if bank_dir is None else str(bank_dir)
    fields["payload_wrong_seed_bank_dir"] = (
        None if wrong_seed_bank_dir is None else str(wrong_seed_bank_dir)
    )
    if prompt_idx is None:
        fields["payload_bank_missing"] = True
        fields["payload_bank_error"] = "missing_prompt_idx"
        return None, fields

    source_bank = wrong_seed_bank_dir if control == "wrong_seed" else bank_dir
    if source_bank is None:
        fields["payload_bank_missing"] = True
        fields["payload_bank_error"] = "missing_bank_dir"
        return None, fields

    prompt_idx_i = int(prompt_idx)
    step_i = int(step)
    source_prompt = prompt_idx_i
    source_step = step_i
    source_kind = control
    index = _read_bank_index(str(source_bank))

    if control == "step_shuffle":
        mapped = _lookup_bank_index(index, "step_shuffle_source_step", prompt_idx_i, step_i)
        if mapped is None:
            fields["payload_bank_missing"] = True
            fields["payload_bank_error"] = "missing_step_shuffle_index"
            return None, fields
        source_step = mapped
        path = _bank_step_path(source_bank, source_prompt, source_step)
    elif control == "wrong_prompt":
        mapped = _lookup_bank_index(index, "wrong_prompt_source_prompt", prompt_idx_i, step_i)
        if mapped is None:
            fields["payload_bank_missing"] = True
            fields["payload_bank_error"] = "missing_wrong_prompt_index"
            return None, fields
        source_prompt = mapped
        path = _bank_step_path(source_bank, source_prompt, source_step)
    elif control == "wrong_seed":
        path = _bank_step_path(source_bank, source_prompt, source_step)
    elif control == "step_only":
        source_prompt = None  # type: ignore[assignment]
        path = _bank_step_mean_path(source_bank, source_step)
    elif control == "prompt_only":
        source_step = None  # type: ignore[assignment]
        path = _bank_prompt_mean_path(source_bank, source_prompt)
    else:
        fields["payload_bank_missing"] = True
        fields["payload_bank_error"] = f"unknown_bank_control:{control}"
        return None, fields

    tensor, error = _load_bank_tensor(path, reuse)
    fields.update({
        "payload_bank_source_kind": source_kind,
        "payload_bank_source_prompt_idx": source_prompt,
        "payload_bank_source_step": source_step,
        "payload_bank_source_path": str(path),
        "payload_bank_loaded": tensor is not None,
        "payload_bank_missing": tensor is None,
        "payload_bank_error": error,
    })
    return tensor, fields


def _write_bank_tensor(
    *,
    write_dir: Optional[Path],
    prompt_idx: Optional[int],
    step: int,
    tensor: torch.Tensor,
    mode: str,
    sigma: float,
) -> Dict[str, Any]:
    fields = _empty_bank_fields()
    fields["payload_bank_write_dir"] = None if write_dir is None else str(write_dir)
    if write_dir is None:
        return fields
    if prompt_idx is None:
        fields["payload_bank_write_ok"] = False
        fields["payload_bank_write_error"] = "missing_prompt_idx"
        return fields

    path = _bank_step_path(write_dir, int(prompt_idx), int(step))
    fields["payload_bank_write_path"] = str(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tensor": tensor.detach().to(device="cpu", dtype=torch.float16),
            "prompt_idx": int(prompt_idx),
            "step": int(step),
            "payload_mode": str(mode),
            "payload_sigma": float(sigma),
        }
        torch.save(payload, path)
        fields["payload_bank_write_ok"] = True
    except Exception as exc:
        fields["payload_bank_write_ok"] = False
        fields["payload_bank_write_error"] = f"{type(exc).__name__}: {exc}"
    return fields


def _reuse_direction_with_norm(reuse: torch.Tensor, reference: torch.Tensor) -> Optional[torch.Tensor]:
    reuse_f = reuse.detach().to(torch.float32)
    ref_norm = float(reference.detach().to(torch.float32).norm().item())
    reuse_norm = float(reuse_f.norm().item())
    if reuse_norm <= 0.0 or ref_norm <= 0.0:
        return None
    return (reuse_f / (reuse_norm + EPS) * ref_norm).to(dtype=reuse.dtype, device=reuse.device)


def _selector_payload(
    self,
    *,
    mode: str,
    reuse: torch.Tensor,
    residual_history_state: Optional[Dict[str, Any]],
    output_history_state: Optional[Dict[str, Any]],
    update_history_state: Optional[Dict[str, Any]],
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step: int,
    sigma: float,
    step_size: Optional[float],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Optional[torch.Tensor]]]:
    fields = _empty_selector_fields()
    fields.update(_empty_bank_fields())
    fields.update({
        "payload_mode": str(mode),
        "payload_base_mode": str(mode),
        "payload_control": "selector",
        "payload_used": "reuse",
        "payload_available": False,
        "payload_fallback": True,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": None,
        "payload_chosen_norm": _norm(reuse),
        "payload_delta_from_reuse_norm": 0.0,
        "payload_selector_enabled": True,
        "payload_selector_mode": str(mode),
    })
    aux: Dict[str, Optional[torch.Tensor]] = {
        "target_payload": None,
        "target_output": None,
        "target_update": None,
    }
    _maybe_sync(hidden_states)
    start = time.perf_counter()

    if mode == "residual_select":
        target_space = "payload"
        target_name, target = update_history_best_target(
            residual_history_state, step=int(step), sigma=float(sigma)
        )
        digest = _selector_history_digest(residual_history_state, int(step))
        aux["target_payload"] = target
    elif mode == "output_select":
        target_space = "output"
        target_name, target = update_history_best_target(
            output_history_state, step=int(step), sigma=float(sigma)
        )
        digest = _selector_history_digest(output_history_state, int(step))
        aux["target_output"] = target
    elif mode == "update_select":
        target_space = "update"
        target_name, target = update_history_best_target(
            update_history_state, step=int(step), sigma=float(sigma)
        )
        digest = _selector_history_digest(update_history_state, int(step))
        aux["target_update"] = target
    else:
        target_space = None
        target_name = None
        target = None
        digest = {"anchor_step": None, "anchor_gap": None, "order_avail": None}

    fields.update({
        "payload_selector_target_space": target_space,
        "payload_selector_target_available": target is not None,
        "payload_selector_target_name": target_name,
        "payload_selector_target_norm": _norm(target),
        "payload_selector_anchor_step": digest.get("anchor_step"),
        "payload_selector_anchor_gap": digest.get("anchor_gap"),
        "payload_selector_order_avail": digest.get("order_avail"),
    })
    if target is None:
        fields["payload_selector_fallback_reason"] = "target_unavailable"
        fields["payload_selector_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    candidates = _candidate_payloads(
        reuse=reuse,
        history_state=residual_history_state,
        step=int(step),
        sigma=float(sigma),
    )
    fields["payload_selector_candidates"] = ",".join(candidates.keys())
    fields["payload_selector_candidate_count"] = int(len(candidates))

    target_f = target.detach().to(torch.float32)
    target_norm = float(target_f.norm().item())
    step_size_f = None if step_size is None else float(step_size)
    scores: Dict[str, float] = {}
    for name in BASE_PAYLOAD_MODES:
        pred = candidates.get(name)
        fields[f"candidate_{name}_available"] = bool(pred is not None and pred.shape == reuse.shape)
        if pred is None or pred.shape != reuse.shape:
            continue
        pred = pred.to(dtype=reuse.dtype, device=reuse.device)
        fields[f"candidate_{name}_payload_norm"] = _norm(pred)
        score: Optional[float] = None
        if target_space == "payload":
            if pred.shape == target.shape:
                score = float((pred.detach().to(torch.float32) - target_f).norm().item())
        else:
            with torch.inference_mode():
                cand_output = _project_output(self, hidden_states + pred, temb)
            fields[f"candidate_{name}_output_norm"] = _norm(cand_output)
            if step_size_f is not None:
                fields[f"candidate_{name}_update_norm"] = _norm(cand_output * step_size_f)
            if target_space == "output" and cand_output.shape == target.shape:
                score = float((cand_output.detach().to(torch.float32) - target_f).norm().item())
            elif (
                target_space == "update"
                and step_size_f is not None
                and cand_output.shape == target.shape
            ):
                score = float(
                    (cand_output.detach().to(torch.float32) * step_size_f - target_f).norm().item()
                )
        if score is None:
            continue
        scores[name] = score
        fields[f"candidate_{name}_score_to_target_abs"] = score
        fields[f"candidate_{name}_score_to_target_rel"] = float(score / (target_norm + EPS))

    fields["payload_selector_scored_count"] = int(len(scores))
    if not scores:
        fields["payload_selector_fallback_reason"] = "no_scored_candidates"
        fields["payload_selector_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return reuse, fields, aux

    ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    for rank, (name, _score) in enumerate(ranked, start=1):
        fields[f"candidate_{name}_rank"] = int(rank)

    chosen_name, chosen_score = ranked[0]
    chosen = candidates[chosen_name].to(dtype=reuse.dtype, device=reuse.device)
    fields.update({
        "payload_used": chosen_name,
        "payload_available": True,
        "payload_fallback": False,
        "payload_forecast_norm": None if chosen_name == "reuse" else _norm(chosen),
        "payload_chosen_norm": _norm(chosen),
        "payload_delta_from_reuse_norm": _norm(chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)),
        "payload_selector_chosen": chosen_name,
        "payload_selector_chosen_score_abs": float(chosen_score),
        "payload_selector_chosen_score_rel": float(chosen_score / (target_norm + EPS)),
        "payload_selector_fallback_reason": None,
        "payload_selector_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
    })
    _maybe_sync(hidden_states)
    return chosen, fields, aux


def _empty_shadow_fields() -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "shadow_observer_enabled": False,
        "shadow_observer_ran": False,
        "shadow_observer_elapsed_ms": None,
        "shadow_full_residual_norm": None,
        "shadow_full_output_norm": None,
        "shadow_full_update_norm": None,
        "sigma_n": None,
        "sigma_np1": None,
        "step_size_H": None,
        "payload_error_improvement_vs_reuse_abs": None,
        "payload_error_improvement_vs_reuse_rel": None,
        "output_error_improvement_vs_reuse_abs": None,
        "output_error_improvement_vs_reuse_rel": None,
        "update_error_improvement_vs_reuse_abs": None,
        "update_error_improvement_vs_reuse_rel": None,
        "payload_selector_target_output_err_abs": None,
        "payload_selector_target_output_err_rel": None,
        "payload_selector_target_update_err_abs": None,
        "payload_selector_target_update_err_rel": None,
        "payload_selector_target_payload_err_abs": None,
        "payload_selector_target_payload_err_rel": None,
        "shadow_oracle_payload_chosen": None,
        "shadow_oracle_payload_score_abs": None,
        "shadow_oracle_output_chosen": None,
        "shadow_oracle_output_score_abs": None,
        "shadow_oracle_update_chosen": None,
        "shadow_oracle_update_score_abs": None,
        "payload_selector_matches_oracle_payload": None,
        "payload_selector_matches_oracle_output": None,
        "payload_selector_matches_oracle_update": None,
    }
    for name in (*SHADOW_PAYLOADS, "chosen"):
        fields[f"payload_err_{name}_available"] = False
        fields[f"payload_err_{name}_abs"] = None
        fields[f"payload_err_{name}_rel"] = None
        fields[f"payload_err_{name}_cos"] = None
        fields[f"output_err_{name}_available"] = False
        fields[f"output_err_{name}_abs"] = None
        fields[f"output_err_{name}_rel"] = None
        fields[f"output_err_{name}_cos"] = None
        fields[f"update_err_{name}_available"] = False
        fields[f"update_err_{name}_abs"] = None
        fields[f"update_err_{name}_rel"] = None
    fields.update(_empty_scalar_family_fields())
    return fields


def _shadow_fields_from_residual(
    self,
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    reuse_payload: torch.Tensor,
    chosen_payload: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    shadow_residual: torch.Tensor,
    shadow_output: torch.Tensor,
    step: int,
    sigma: float,
    elapsed_ms: Optional[float] = None,
    selector_chosen: Optional[str] = None,
    selector_target_payload: Optional[torch.Tensor] = None,
    selector_target_output: Optional[torch.Tensor] = None,
    selector_target_update: Optional[torch.Tensor] = None,
    precomputed_scalar_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fields = _empty_shadow_fields()
    fields["shadow_observer_enabled"] = True
    fields.update(_scheduler_step_fields(self, int(step)))
    step_size = fields.get("step_size_H")
    step_size_abs = None if step_size is None else abs(float(step_size))

    fields["shadow_full_residual_norm"] = _norm(shadow_residual)
    fields["shadow_full_output_norm"] = _norm(shadow_output)
    fields["shadow_full_update_norm"] = (
        None if step_size_abs is None or fields["shadow_full_output_norm"] is None
        else float(step_size_abs * fields["shadow_full_output_norm"])
    )
    if selector_target_output is not None and selector_target_output.shape == shadow_output.shape:
        target_output_diff = (
            selector_target_output.detach().to(torch.float32)
            - shadow_output.detach().to(torch.float32)
        )
        target_output_err = float(target_output_diff.norm().item())
        fields["payload_selector_target_output_err_abs"] = target_output_err
        fields["payload_selector_target_output_err_rel"] = (
            None if fields["shadow_full_output_norm"] is None
            else float(target_output_err / (fields["shadow_full_output_norm"] + EPS))
        )
    if (
        selector_target_update is not None
        and selector_target_update.shape == shadow_output.shape
        and step_size is not None
    ):
        shadow_update = shadow_output.detach().to(torch.float32) * float(step_size)
        target_update_diff = selector_target_update.detach().to(torch.float32) - shadow_update
        target_update_err = float(target_update_diff.norm().item())
        fields["payload_selector_target_update_err_abs"] = target_update_err
        fields["payload_selector_target_update_err_rel"] = (
            None if fields["shadow_full_update_norm"] is None
            else float(target_update_err / (fields["shadow_full_update_norm"] + EPS))
        )

    residual_actual_norm = _norm(shadow_residual)
    output_actual_norm = _norm(shadow_output)
    update_actual_norm = fields["shadow_full_update_norm"]
    if selector_target_payload is not None and selector_target_payload.shape == shadow_residual.shape:
        target_payload_diff = (
            selector_target_payload.detach().to(torch.float32)
            - shadow_residual.detach().to(torch.float32)
        )
        target_payload_err = float(target_payload_diff.norm().item())
        fields["payload_selector_target_payload_err_abs"] = target_payload_err
        fields["payload_selector_target_payload_err_rel"] = (
            None if residual_actual_norm is None
            else float(target_payload_err / (residual_actual_norm + EPS))
        )
    candidates = _candidate_payloads(
        reuse=reuse_payload,
        history_state=history_state,
        step=int(step),
        sigma=float(sigma),
    )
    candidates["chosen"] = chosen_payload
    if precomputed_scalar_fields is None:
        fields.update(_scalar_family_oracle_fields(
            self,
            hidden_states=hidden_states,
            temb=temb,
            shadow_residual=shadow_residual,
            shadow_output=shadow_output,
            history_state=history_state,
            candidates=candidates,
            step=int(step),
            step_size_abs=step_size_abs,
        ))
    else:
        fields.update(precomputed_scalar_fields)

    residual_errors: Dict[str, Optional[float]] = {}
    output_errors: Dict[str, Optional[float]] = {}
    update_errors: Dict[str, Optional[float]] = {}
    for name in (*SHADOW_PAYLOADS, "chosen"):
        pred = candidates.get(name)
        if pred is None or pred.shape != shadow_residual.shape:
            continue
        diff = pred.detach().to(torch.float32) - shadow_residual.detach().to(torch.float32)
        err_abs = float(diff.norm().item())
        residual_errors[name] = err_abs
        fields[f"payload_err_{name}_available"] = True
        fields[f"payload_err_{name}_abs"] = err_abs
        fields[f"payload_err_{name}_rel"] = (
            None if residual_actual_norm is None else float(err_abs / (residual_actual_norm + EPS))
        )
        fields[f"payload_err_{name}_cos"] = _cos(pred, shadow_residual)

        with torch.inference_mode():
            candidate_output = _project_output(self, hidden_states + pred, temb)
        output_diff = candidate_output.detach().to(torch.float32) - shadow_output.detach().to(torch.float32)
        output_err_abs = float(output_diff.norm().item())
        output_errors[name] = output_err_abs
        fields[f"output_err_{name}_available"] = True
        fields[f"output_err_{name}_abs"] = output_err_abs
        fields[f"output_err_{name}_rel"] = (
            None if output_actual_norm is None else float(output_err_abs / (output_actual_norm + EPS))
        )
        fields[f"output_err_{name}_cos"] = _cos(candidate_output, shadow_output)

        if step_size_abs is not None:
            update_err_abs = float(step_size_abs * output_err_abs)
            update_errors[name] = update_err_abs
            fields[f"update_err_{name}_available"] = True
            fields[f"update_err_{name}_abs"] = update_err_abs
            fields[f"update_err_{name}_rel"] = (
                None if update_actual_norm is None else float(update_err_abs / (update_actual_norm + EPS))
            )

    if residual_errors:
        chosen_name, score = min(residual_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_payload_chosen"] = chosen_name
        fields["shadow_oracle_payload_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_payload"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )
    if output_errors:
        chosen_name, score = min(output_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_output_chosen"] = chosen_name
        fields["shadow_oracle_output_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_output"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )
    if update_errors:
        chosen_name, score = min(update_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_update_chosen"] = chosen_name
        fields["shadow_oracle_update_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_update"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )

    reuse_err = residual_errors.get("reuse")
    chosen_err = residual_errors.get("chosen")
    if reuse_err is not None and chosen_err is not None:
        improvement = float(reuse_err - chosen_err)
        fields["payload_error_improvement_vs_reuse_abs"] = improvement
        fields["payload_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_err + EPS))

    reuse_output_err = output_errors.get("reuse")
    chosen_output_err = output_errors.get("chosen")
    if reuse_output_err is not None and chosen_output_err is not None:
        improvement = float(reuse_output_err - chosen_output_err)
        fields["output_error_improvement_vs_reuse_abs"] = improvement
        fields["output_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_output_err + EPS))

    reuse_update_err = update_errors.get("reuse")
    chosen_update_err = update_errors.get("chosen")
    if reuse_update_err is not None and chosen_update_err is not None:
        improvement = float(reuse_update_err - chosen_update_err)
        fields["update_error_improvement_vs_reuse_abs"] = improvement
        fields["update_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_update_err + EPS))

    fields["shadow_observer_elapsed_ms"] = elapsed_ms
    fields["shadow_observer_ran"] = True
    return fields


def _shadow_full_residual_fields(
    self,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    controlnet_block_samples,
    controlnet_single_block_samples,
    controlnet_blocks_repeat: bool,
    reuse_payload: torch.Tensor,
    chosen_payload: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    sigma: float,
    selector_chosen: Optional[str] = None,
    selector_target_payload: Optional[torch.Tensor] = None,
    selector_target_output: Optional[torch.Tensor] = None,
    selector_target_update: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    fields = _empty_shadow_fields()
    fields["shadow_observer_enabled"] = True
    fields.update(_scheduler_step_fields(self, int(step)))
    step_size = fields.get("step_size_H")
    step_size_abs = None if step_size is None else abs(float(step_size))

    _maybe_sync(hidden_states)
    start = time.perf_counter()
    with torch.inference_mode():
        _, shadow_hidden_states = _full_transformer_residual(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        shadow_residual = shadow_hidden_states - hidden_states
        shadow_output = _project_output(self, shadow_hidden_states, temb)
    fields["shadow_full_residual_norm"] = _norm(shadow_residual)
    fields["shadow_full_output_norm"] = _norm(shadow_output)
    fields["shadow_full_update_norm"] = (
        None if step_size_abs is None or fields["shadow_full_output_norm"] is None
        else float(step_size_abs * fields["shadow_full_output_norm"])
    )
    if selector_target_output is not None and selector_target_output.shape == shadow_output.shape:
        target_output_diff = (
            selector_target_output.detach().to(torch.float32)
            - shadow_output.detach().to(torch.float32)
        )
        target_output_err = float(target_output_diff.norm().item())
        fields["payload_selector_target_output_err_abs"] = target_output_err
        fields["payload_selector_target_output_err_rel"] = (
            None if fields["shadow_full_output_norm"] is None
            else float(target_output_err / (fields["shadow_full_output_norm"] + EPS))
        )
    if (
        selector_target_update is not None
        and selector_target_update.shape == shadow_output.shape
        and step_size is not None
    ):
        shadow_update = shadow_output.detach().to(torch.float32) * float(step_size)
        target_update_diff = selector_target_update.detach().to(torch.float32) - shadow_update
        target_update_err = float(target_update_diff.norm().item())
        fields["payload_selector_target_update_err_abs"] = target_update_err
        fields["payload_selector_target_update_err_rel"] = (
            None if fields["shadow_full_update_norm"] is None
            else float(target_update_err / (fields["shadow_full_update_norm"] + EPS))
        )

    residual_actual_norm = _norm(shadow_residual)
    output_actual_norm = _norm(shadow_output)
    update_actual_norm = fields["shadow_full_update_norm"]
    if selector_target_payload is not None and selector_target_payload.shape == shadow_residual.shape:
        target_payload_diff = (
            selector_target_payload.detach().to(torch.float32)
            - shadow_residual.detach().to(torch.float32)
        )
        target_payload_err = float(target_payload_diff.norm().item())
        fields["payload_selector_target_payload_err_abs"] = target_payload_err
        fields["payload_selector_target_payload_err_rel"] = (
            None if residual_actual_norm is None
            else float(target_payload_err / (residual_actual_norm + EPS))
        )
    candidates = _candidate_payloads(
        reuse=reuse_payload,
        history_state=history_state,
        step=int(step),
        sigma=float(sigma),
    )
    candidates["chosen"] = chosen_payload
    fields.update(_scalar_family_oracle_fields(
        self,
        hidden_states=hidden_states,
        temb=temb,
        shadow_residual=shadow_residual,
        shadow_output=shadow_output,
        history_state=history_state,
        candidates=candidates,
        step=int(step),
        step_size_abs=step_size_abs,
    ))

    residual_errors: Dict[str, Optional[float]] = {}
    output_errors: Dict[str, Optional[float]] = {}
    update_errors: Dict[str, Optional[float]] = {}
    for name in (*SHADOW_PAYLOADS, "chosen"):
        pred = candidates.get(name)
        if pred is None or pred.shape != shadow_residual.shape:
            continue
        diff = pred.detach().to(torch.float32) - shadow_residual.detach().to(torch.float32)
        err_abs = float(diff.norm().item())
        residual_errors[name] = err_abs
        fields[f"payload_err_{name}_available"] = True
        fields[f"payload_err_{name}_abs"] = err_abs
        fields[f"payload_err_{name}_rel"] = (
            None if residual_actual_norm is None else float(err_abs / (residual_actual_norm + EPS))
        )
        fields[f"payload_err_{name}_cos"] = _cos(pred, shadow_residual)

        with torch.inference_mode():
            candidate_output = _project_output(self, hidden_states + pred, temb)
        output_diff = candidate_output.detach().to(torch.float32) - shadow_output.detach().to(torch.float32)
        output_err_abs = float(output_diff.norm().item())
        output_errors[name] = output_err_abs
        fields[f"output_err_{name}_available"] = True
        fields[f"output_err_{name}_abs"] = output_err_abs
        fields[f"output_err_{name}_rel"] = (
            None if output_actual_norm is None else float(output_err_abs / (output_actual_norm + EPS))
        )
        fields[f"output_err_{name}_cos"] = _cos(candidate_output, shadow_output)

        if step_size_abs is not None:
            update_err_abs = float(step_size_abs * output_err_abs)
            update_errors[name] = update_err_abs
            fields[f"update_err_{name}_available"] = True
            fields[f"update_err_{name}_abs"] = update_err_abs
            fields[f"update_err_{name}_rel"] = (
                None if update_actual_norm is None else float(update_err_abs / (update_actual_norm + EPS))
            )

    if residual_errors:
        chosen_name, score = min(residual_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_payload_chosen"] = chosen_name
        fields["shadow_oracle_payload_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_payload"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )
    if output_errors:
        chosen_name, score = min(output_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_output_chosen"] = chosen_name
        fields["shadow_oracle_output_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_output"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )
    if update_errors:
        chosen_name, score = min(update_errors.items(), key=lambda kv: (kv[1], kv[0]))
        fields["shadow_oracle_update_chosen"] = chosen_name
        fields["shadow_oracle_update_score_abs"] = float(score)
        fields["payload_selector_matches_oracle_update"] = (
            None if selector_chosen is None else bool(str(selector_chosen) == chosen_name)
        )

    reuse_err = residual_errors.get("reuse")
    chosen_err = residual_errors.get("chosen")
    if reuse_err is not None and chosen_err is not None:
        improvement = float(reuse_err - chosen_err)
        fields["payload_error_improvement_vs_reuse_abs"] = improvement
        fields["payload_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_err + EPS))

    reuse_output_err = output_errors.get("reuse")
    chosen_output_err = output_errors.get("chosen")
    if reuse_output_err is not None and chosen_output_err is not None:
        improvement = float(reuse_output_err - chosen_output_err)
        fields["output_error_improvement_vs_reuse_abs"] = improvement
        fields["output_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_output_err + EPS))

    reuse_update_err = update_errors.get("reuse")
    chosen_update_err = update_errors.get("chosen")
    if reuse_update_err is not None and chosen_update_err is not None:
        improvement = float(reuse_update_err - chosen_update_err)
        fields["update_error_improvement_vs_reuse_abs"] = improvement
        fields["update_error_improvement_vs_reuse_rel"] = float(improvement / (reuse_update_err + EPS))

    _maybe_sync(hidden_states)
    fields["shadow_observer_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
    fields["shadow_observer_ran"] = True

    return fields


def _posterior_oracle_cy_payload(
    self,
    *,
    mode: str,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    controlnet_block_samples,
    controlnet_single_block_samples,
    controlnet_blocks_repeat: bool,
    reuse_payload: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    sigma: float,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    spec = POSTERIOR_ORACLE_PAYLOAD_SPECS[str(mode)]
    raw_mode = str(spec["raw_mode"])
    control = str(spec["control"])
    payload_fields = _empty_posterior_oracle_fields()
    payload_fields.update(_empty_bank_fields())
    payload_fields.update(_empty_pca_fields())
    payload_fields.update({
        "payload_mode": str(mode),
        "payload_base_mode": raw_mode,
        "payload_control": control,
        "payload_used": "reuse",
        "payload_available": False,
        "payload_fallback": True,
        "payload_reuse_norm": _norm(reuse_payload),
        "payload_forecast_norm": None,
        "payload_chosen_norm": _norm(reuse_payload),
        "payload_delta_from_reuse_norm": 0.0,
        "posterior_oracle_enabled": True,
        "posterior_oracle_family": raw_mode,
        "posterior_oracle_uses_shadow_full_compute": True,
        "posterior_oracle_updates_full_history": False,
        "posterior_oracle_closed_loop_committed": False,
    })

    _maybe_sync(hidden_states)
    start = time.perf_counter()
    with torch.inference_mode():
        _, shadow_hidden_states = _full_transformer_residual(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        shadow_residual = shadow_hidden_states - hidden_states
        shadow_output = _project_output(self, shadow_hidden_states, temb)
    _maybe_sync(hidden_states)
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)

    step_fields = _scheduler_step_fields(self, int(step))
    step_size = step_fields.get("step_size_H")
    step_size_abs = None if step_size is None else abs(float(step_size))
    candidates = _candidate_payloads(
        reuse=reuse_payload,
        history_state=history_state,
        step=int(step),
        sigma=float(sigma),
    )
    scalar_fields = _scalar_family_oracle_fields(
        self,
        hidden_states=hidden_states,
        temb=temb,
        shadow_residual=shadow_residual,
        shadow_output=shadow_output,
        history_state=history_state,
        candidates=candidates,
        step=int(step),
        step_size_abs=step_size_abs,
    )
    c_star = scalar_fields.get("shadow_scalar_family_output_oracle_c")

    def _fallback(reason: str) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        payload_fields.update({
            "posterior_oracle_ran": True,
            "posterior_oracle_fail_reason": str(reason),
            "posterior_oracle_anchor_step": scalar_fields.get("shadow_scalar_family_anchor_step"),
            "posterior_oracle_c_raw": scalar_fields.get("shadow_scalar_family_raw_c"),
            "posterior_oracle_c_star": c_star,
            "posterior_oracle_c_min": scalar_fields.get("shadow_scalar_family_grid_min_c"),
            "posterior_oracle_c_max": scalar_fields.get("shadow_scalar_family_grid_max_c"),
            "posterior_oracle_grid_count": scalar_fields.get("shadow_scalar_family_grid_count"),
            "posterior_oracle_boundary_hit": scalar_fields.get(
                "shadow_scalar_family_output_oracle_boundary_hit"
            ),
            "posterior_oracle_payload_norm": _norm(reuse_payload),
            "posterior_oracle_delta_from_reuse_norm": 0.0,
            "posterior_oracle_shadow_residual_norm": _norm(shadow_residual),
            "posterior_oracle_shadow_output_norm": _norm(shadow_output),
            "posterior_oracle_output_err_abs": scalar_fields.get(
                "shadow_scalar_family_output_reuse_err_abs"
            ),
            "posterior_oracle_update_err_abs": scalar_fields.get(
                "shadow_scalar_family_update_reuse_err_abs"
            ),
            "posterior_oracle_elapsed_ms": elapsed_ms,
        })
        shadow_fields = _shadow_fields_from_residual(
            self,
            hidden_states=hidden_states,
            temb=temb,
            reuse_payload=reuse_payload,
            chosen_payload=reuse_payload,
            history_state=history_state,
            shadow_residual=shadow_residual,
            shadow_output=shadow_output,
            step=int(step),
            sigma=float(sigma),
            elapsed_ms=elapsed_ms,
            precomputed_scalar_fields=scalar_fields,
        )
        return reuse_payload, payload_fields, shadow_fields

    if not scalar_fields.get("shadow_scalar_family_available"):
        return _fallback(str(scalar_fields.get("shadow_scalar_family_fallback_reason") or "scalar_family_unavailable"))
    if c_star is None or not np.isfinite(float(c_star)):
        return _fallback("output_oracle_c_unavailable")

    st = history_state or {}
    history = st.get("history") or {}
    anchor = history.get(0) if isinstance(history, dict) else None
    direction = history.get(1) if isinstance(history, dict) else None
    if (
        anchor is None
        or direction is None
        or tuple(anchor.shape) != tuple(reuse_payload.shape)
        or tuple(direction.shape) != tuple(reuse_payload.shape)
    ):
        return _fallback("history_shape_mismatch")

    oracle_payload = _scalar_payload(
        anchor.detach().to(device=reuse_payload.device, dtype=torch.float32),
        direction.detach().to(device=reuse_payload.device, dtype=torch.float32),
        float(c_star),
    ).to(dtype=reuse_payload.dtype, device=reuse_payload.device)
    shadow_fields = _shadow_fields_from_residual(
        self,
        hidden_states=hidden_states,
        temb=temb,
        reuse_payload=reuse_payload,
        chosen_payload=oracle_payload,
        history_state=history_state,
        shadow_residual=shadow_residual,
        shadow_output=shadow_output,
        step=int(step),
        sigma=float(sigma),
        elapsed_ms=elapsed_ms,
        precomputed_scalar_fields=scalar_fields,
    )
    c_matches = (
        shadow_fields.get("shadow_scalar_family_output_oracle_c") is not None
        and abs(float(shadow_fields["shadow_scalar_family_output_oracle_c"]) - float(c_star)) <= 1e-9
    )
    payload_fields.update({
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_forecast_norm": _norm(oracle_payload),
        "payload_chosen_norm": _norm(oracle_payload),
        "payload_delta_from_reuse_norm": _norm(
            oracle_payload.detach().to(torch.float32) - reuse_payload.detach().to(torch.float32)
        ),
        "posterior_oracle_ran": True,
        "posterior_oracle_fail_reason": None,
        "posterior_oracle_closed_loop_committed": True,
        "posterior_oracle_anchor_step": scalar_fields.get("shadow_scalar_family_anchor_step"),
        "posterior_oracle_c_raw": scalar_fields.get("shadow_scalar_family_raw_c"),
        "posterior_oracle_c_star": float(c_star),
        "posterior_oracle_c_min": scalar_fields.get("shadow_scalar_family_grid_min_c"),
        "posterior_oracle_c_max": scalar_fields.get("shadow_scalar_family_grid_max_c"),
        "posterior_oracle_grid_count": scalar_fields.get("shadow_scalar_family_grid_count"),
        "posterior_oracle_boundary_hit": scalar_fields.get(
            "shadow_scalar_family_output_oracle_boundary_hit"
        ),
        "posterior_oracle_payload_norm": _norm(oracle_payload),
        "posterior_oracle_delta_from_reuse_norm": _norm(
            oracle_payload.detach().to(torch.float32) - reuse_payload.detach().to(torch.float32)
        ),
        "posterior_oracle_shadow_residual_norm": _norm(shadow_residual),
        "posterior_oracle_shadow_output_norm": _norm(shadow_output),
        "posterior_oracle_output_err_abs": shadow_fields.get("output_err_chosen_abs"),
        "posterior_oracle_update_err_abs": shadow_fields.get("update_err_chosen_abs"),
        "posterior_oracle_c_matches_shadow_output_oracle": bool(c_matches),
        "posterior_oracle_elapsed_ms": elapsed_ms,
    })
    return oracle_payload, payload_fields, shadow_fields


def _forecast_payload(
    mode: str,
    *,
    reuse: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    svd_state: Optional[Dict[str, Any]] = None,
    step: int,
    sigma: float,
    num_steps: Optional[int] = None,
    prompt_idx: Optional[int] = None,
    payload_bank_dir: Optional[Path] = None,
    payload_wrong_seed_bank_dir: Optional[Path] = None,
    payload_bank_write_dir: Optional[Path] = None,
    payload_control_seed_salt: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    base_mode, control = _payload_spec(str(mode))
    if mode == "reuse":
        fields = {
            "payload_mode": "reuse",
            "payload_base_mode": "reuse",
            "payload_control": "none",
            "payload_control_seed_salt": int(payload_control_seed_salt),
            "payload_used": "reuse",
            "payload_available": True,
            "payload_fallback": False,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": None,
            "payload_chosen_norm": _norm(reuse),
            "payload_delta_from_reuse_norm": 0.0,
        }
        fields.update(_empty_bank_fields())
        fields.update(_empty_pca_fields())
        return reuse, fields

    if control in BANK_CONTROLS:
        bank_forecast, bank_fields = _bank_control_tensor(
            control=control,
            bank_dir=payload_bank_dir,
            wrong_seed_bank_dir=payload_wrong_seed_bank_dir,
            prompt_idx=prompt_idx,
            step=int(step),
            reuse=reuse,
        )
        if bank_forecast is None:
            fields = {
                "payload_mode": str(mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": None,
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
            fields.update(bank_fields)
            fields.update(_empty_pca_fields())
            return reuse, fields

        fields = {
            "payload_mode": str(mode),
            "payload_base_mode": str(base_mode),
            "payload_control": str(control),
            "payload_control_seed_salt": int(payload_control_seed_salt),
            "payload_used": str(mode),
            "payload_available": True,
            "payload_fallback": False,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": _norm(bank_forecast),
            "payload_chosen_norm": _norm(bank_forecast),
            "payload_delta_from_reuse_norm": _norm(
                bank_forecast.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
        }
        fields.update(bank_fields)
        fields.update(_empty_pca_fields())
        return bank_forecast, fields

    control_step = int(step)
    if control == "shift_m1":
        control_step = max(int(step) - 1, 0)
    elif control == "shift_p1":
        control_step = int(step) + 1
    elif control == "mirror_step" and num_steps is not None:
        control_step = max(int(num_steps) - 1 - int(step), 0)

    preds = forecast_predictions(history_state, step=control_step, sigma=float(sigma))
    if base_mode == "ensemble_mean":
        parts = [
            preds.get(name)
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            if preds.get(name) is not None and preds.get(name).shape == reuse.shape
        ]
        forecast = None if not parts else torch.stack(
            [part.to(dtype=reuse.dtype, device=reuse.device) for part in parts], dim=0
        ).mean(dim=0)
    else:
        forecast = preds.get(base_mode)

    if forecast is None or forecast.shape != reuse.shape:
        if control != "svdcache_ema":
            fields = {
                "payload_mode": str(mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": None,
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
            fields.update(_empty_bank_fields())
            fields.update(_empty_pca_fields())
            fields.update(svdcache_payload.empty_fields())
            return reuse, fields
    elif forecast is not None:
        forecast = forecast.to(dtype=reuse.dtype, device=reuse.device)

    if control in svdcache_payload.SVD_CACHE_CONTROLS:
        svd_chosen, svd_fields = svdcache_payload.payload(
            svd_state,
            mode=str(mode),
            reuse=reuse,
            forecast=forecast,
        )
        if svd_chosen is None:
            fields = {
                "payload_mode": str(mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": _norm(forecast),
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
            fields.update(_empty_bank_fields())
            fields.update(_empty_pca_fields())
            fields.update(svd_fields)
            return reuse, fields
        write_fields = _write_bank_tensor(
            write_dir=payload_bank_write_dir,
            prompt_idx=prompt_idx,
            step=int(step),
            tensor=svd_chosen,
            mode=str(mode),
            sigma=float(sigma),
        )
        fields = {
            "payload_mode": str(mode),
            "payload_base_mode": str(base_mode),
            "payload_control": str(control),
            "payload_control_seed_salt": int(payload_control_seed_salt),
            "payload_used": str(mode),
            "payload_available": True,
            "payload_fallback": False,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": _norm(forecast),
            "payload_chosen_norm": _norm(svd_chosen),
            "payload_delta_from_reuse_norm": _norm(
                svd_chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)
            ),
        }
        fields.update(write_fields)
        fields.update(_empty_pca_fields())
        fields.update(svd_fields)
        return svd_chosen, fields

    write_fields = _write_bank_tensor(
        write_dir=payload_bank_write_dir,
        prompt_idx=prompt_idx,
        step=int(step),
        tensor=forecast,
        mode=str(base_mode),
        sigma=float(sigma),
    )
    chosen = forecast
    pca_fields = _empty_pca_fields()
    if control in PCA_CLEAN_CONTROLS:
        spec = PCA_CLEAN_PAYLOAD_SPECS.get(str(mode))
        q = int(spec["q"]) if spec is not None else 1
        gamma = float(spec["gamma"]) if spec is not None else 0.0
        pca_chosen, pca_fields = _pca_clean_payload(
            raw_forecast=forecast,
            history_state=history_state,
            q=q,
            gamma=gamma,
        )
        pca_fields["payload_pca_raw_mode"] = str(base_mode)
        if pca_chosen is None:
            fields = {
                "payload_mode": str(mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": _norm(forecast),
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
            fields.update(write_fields)
            fields.update(pca_fields)
            return reuse, fields
        chosen = pca_chosen.to(dtype=reuse.dtype, device=reuse.device)
    if control == "norm_only":
        norm_only = _reuse_direction_with_norm(reuse, forecast)
        if norm_only is None:
            fields = {
                "payload_mode": str(mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": _norm(forecast),
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
            fields.update(write_fields)
            fields.update(_empty_pca_fields())
            return reuse, fields
        chosen = norm_only
    elif control == "random_dir":
        chosen = _random_direction_like(
            forecast,
            mode=str(mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )
    elif control == "delta_negative":
        chosen = reuse - (forecast - reuse)
    elif control == "delta_random_dir":
        chosen = reuse + _random_direction_like(
            forecast - reuse,
            mode=str(mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )
    elif control == "delta_orthogonal_random":
        chosen = reuse + _orthogonal_random_direction_like(
            forecast - reuse,
            mode=str(mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )

    fields = {
        "payload_mode": str(mode),
        "payload_base_mode": str(base_mode),
        "payload_control": str(control),
        "payload_control_seed_salt": int(payload_control_seed_salt),
        "payload_used": str(mode),
        "payload_available": True,
        "payload_fallback": False,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": _norm(forecast),
        "payload_chosen_norm": _norm(chosen),
        "payload_delta_from_reuse_norm": _norm(
            chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)
        ),
    }
    fields.update(write_fields)
    fields.update(pca_fields)
    return chosen, fields


def _seacache_payload_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids is not None and txt_ids.ndim == 3:
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        img_ids = img_ids[0]

    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    cnt = int(getattr(self, "cnt", 0))
    threshold = float(getattr(self, "seacache_thresh", 0.3))
    gate_mode = str(getattr(self, "seacache_payload_gate_mode", "seacache"))
    native_gate_kind = "teacache" if gate_mode == "teacache" else "seacache"
    payload_mode = str(getattr(self, "seacache_payload_mode", "reuse"))
    selector_mode = payload_mode in SELECTOR_PAYLOAD_MODES
    update_inverse_mode = payload_mode in UPDATE_INVERSE_PAYLOAD_MODES
    update_scalar_calib_mode = payload_mode in UPDATE_SCALAR_CALIB_PAYLOAD_MODES
    forecast_opt_mode = payload_mode in FORECAST_OPT_PAYLOAD_MODES or gate_mode == "rfc_input_error"
    svd_cache_mode = payload_mode in SVD_CACHE_PAYLOAD_MODES
    posterior_oracle_mode = payload_mode in POSTERIOR_ORACLE_PAYLOAD_MODES
    update_history_mode = bool(selector_mode or update_inverse_mode or update_scalar_calib_mode)
    history_state_pre = getattr(self, "seacache_payload_history_fd_state", None)
    output_history_state_pre = getattr(self, "seacache_payload_output_history_state", None)
    update_history_state_pre = getattr(self, "seacache_payload_update_history_state", None)
    forecast_opt_state_pre = getattr(self, "seacache_payload_forecast_opt_state", None)
    history_fields = history_fd_online_fields(
        history_state_pre,
        step=cnt,
        sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
    )
    output_history_fields = (
        update_history_online_fields(output_history_state_pre, step=cnt, prefix="online_output_fd")
        if update_history_mode else {}
    )
    update_history_fields = (
        update_history_online_fields(update_history_state_pre, step=cnt, prefix="online_update_fd")
        if update_history_mode else {}
    )
    scheduler_fields = _scheduler_step_fields(self, int(cnt))
    step_size = scheduler_fields.get("step_size_H")

    inp = hidden_states
    first_block = self.transformer_blocks[0]
    modulated_inp, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = first_block.norm1(inp, emb=temb)

    force_full_reason = None
    if cnt < int(getattr(self, "first_enhance", 1)):
        force_full_reason = "first_enhance"
    elif cnt == 0:
        force_full_reason = "step0"
    elif cnt == self.num_steps - 1:
        force_full_reason = "final_step"
    elif self.previous_modulated_input is None:
        force_full_reason = "no_previous_modulated_input"
    elif self.previous_residual is None:
        force_full_reason = "no_previous_residual"

    accumulator_before = float(getattr(self, "accumulated_rel_l1_distance", 0.0))
    sea_increment = None
    gate_increment_raw = None
    gate_increment_rescaled = None
    accumulator_after_increment = accumulator_before
    native_should_calc = True

    if force_full_reason is None:
        if native_gate_kind == "teacache":
            gate_increment_raw = rel_l1(modulated_inp, self.previous_modulated_input)
            teacache_rescale = getattr(self, "teacache_rescale", None)
            if teacache_rescale is None:
                raise RuntimeError("TeaCachePayload gate requested without teacache_rescale")
            gate_increment_rescaled = float(teacache_rescale(float(gate_increment_raw)))
            modulated_for_state = modulated_inp
        else:
            modulated_for_distance = modulated_inp.reshape(
                modulated_inp.shape[0],
                int(img_ids[:, 1].max().item() + 1),
                int(img_ids[:, 2].max().item() + 1),
                modulated_inp.shape[-1],
            )
            modulated_for_distance = apply_sea_with_scheduler(
                modulated_for_distance,
                self.scheduler,
                cnt,
                power_exp=2.0,
                dims=(-2, -3),
                norm_mode="mean",
            )
            modulated_for_distance = modulated_for_distance.reshape(
                modulated_for_distance.shape[0], -1, modulated_for_distance.shape[-1]
            )
            sea_increment = rel_l1(modulated_for_distance, self.previous_modulated_input)
            gate_increment_raw = float(sea_increment)
            gate_increment_rescaled = float(sea_increment)
            modulated_for_state = modulated_for_distance
        accumulator_after_increment = accumulator_before + float(gate_increment_rescaled)
        native_should_calc = not (accumulator_after_increment < threshold)
    else:
        modulated_for_state = modulated_inp

    reuse_payload_pre = self.previous_residual
    guard_fields, guard_forecast_pre, guard_reuse_update_pre, guard_forecast_update_pre = (
        _forecast_guard_pre_fields(
            self,
            payload_mode=payload_mode,
            reuse=reuse_payload_pre,
            history_state=history_state_pre,
            hidden_states=hidden_states,
            temb=temb,
            step=cnt,
            sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
            step_size=step_size,
            force_full_reason=force_full_reason,
        )
    )
    rfc_gate_fields = _empty_rfc_gate_fields()
    if gate_mode == "rfc_input_error":
        rfc_gate_fields = _rfc_input_error_gate_pre_fields(
            self,
            gate_input=modulated_inp,
            opt_state=forecast_opt_state_pre,
            payload_mode=payload_mode,
            step=cnt,
            force_full_reason=force_full_reason,
        )

    action_steps: Optional[Set[int]] = getattr(self, "seacache_payload_action_steps", None)
    schedule_locked = action_steps is not None
    schedule_u = None if action_steps is None else int(cnt in action_steps)
    seacache_gate_cache_allowed = bool(force_full_reason is None and not native_should_calc)
    forecast_guard_cache_allowed = bool(guard_fields.get("forecast_guard_cache_allowed"))
    rfc_gate_cache_allowed = bool(rfc_gate_fields.get("rfc_gate_cache_allowed"))
    combined_gate_cache_allowed = None
    combined_gate_veto_reason = None
    if gate_mode == "seacache_forecast_intersection":
        combined_gate_cache_allowed = bool(seacache_gate_cache_allowed and forecast_guard_cache_allowed)
        if force_full_reason is not None:
            combined_gate_veto_reason = f"force_full:{force_full_reason}"
        elif not seacache_gate_cache_allowed:
            combined_gate_veto_reason = "seacache_full"
        elif not forecast_guard_cache_allowed:
            combined_gate_veto_reason = (
                "forecast_guard:"
                f"{guard_fields.get('forecast_guard_decision_reason') or 'not_allowed'}"
            )
        else:
            combined_gate_veto_reason = "cache_allowed"

    if force_full_reason is not None:
        should_calc = True
    elif gate_mode == "forecast_uncertainty":
        should_calc = not forecast_guard_cache_allowed
    elif gate_mode == "seacache_forecast_intersection":
        should_calc = not bool(combined_gate_cache_allowed)
    elif gate_mode == "rfc_input_error":
        should_calc = not rfc_gate_cache_allowed
    elif action_steps is not None:
        should_calc = not bool(schedule_u)
    else:
        should_calc = bool(native_should_calc)

    self.previous_modulated_input = modulated_for_state.detach()
    if gate_mode == "rfc_input_error":
        self.seacache_payload_rfc_accumulated_error = (
            0.0 if should_calc else float(rfc_gate_fields.get("rfc_gate_accumulator_after_increment") or 0.0)
        )
        self.accumulated_rel_l1_distance = 0.0 if should_calc else float(accumulator_after_increment)
    else:
        self.accumulated_rel_l1_distance = 0.0 if should_calc else float(accumulator_after_increment)

    payload_fields: Dict[str, Any] = {
        "payload_mode": payload_mode,
        "payload_base_mode": _payload_spec(payload_mode)[0],
        "payload_control": _payload_spec(payload_mode)[1],
        "payload_control_seed_salt": int(getattr(self, "seacache_payload_control_seed_salt", 0)),
        "payload_blend": float(getattr(self, "seacache_payload_blend", 1.0)),
        "payload_used": None,
        "payload_available": None,
        "payload_fallback": None,
        "payload_reuse_norm": None,
        "payload_forecast_norm": None,
        "payload_chosen_norm": None,
        "payload_delta_from_reuse_norm": None,
    }
    payload_fields.update(_empty_bank_fields())
    payload_fields.update(_empty_selector_fields())
    payload_fields.update(_empty_pca_fields())
    payload_fields.update(_empty_update_inverse_fields())
    payload_fields.update(_empty_update_calib_fields())
    payload_fields.update(_empty_forecast_opt_fields())
    payload_fields.update(_empty_posterior_oracle_fields())
    payload_fields.update(svdcache_payload.empty_fields())
    payload_fields.update(output_history_fields)
    payload_fields.update(update_history_fields)
    full_residual_norm = None
    full_output_norm = None
    full_update_norm = None
    update_history_updated = False
    history_updated = False
    shadow_fields = _empty_shadow_fields()
    shadow_fields["shadow_observer_enabled"] = bool(
        getattr(self, "seacache_payload_shadow_full_residual", False)
    )
    shadow_fields.update(scheduler_fields)

    if (
        getattr(self, "enable_seacache_payload", False)
        and not should_calc
        and self.previous_residual is not None
    ):
        reuse_payload = self.previous_residual
        selector_aux: Dict[str, Optional[torch.Tensor]] = {
            "target_payload": None,
            "target_output": None,
            "target_update": None,
        }
        if posterior_oracle_mode:
            cache_payload, oracle_fields, shadow_fields = _posterior_oracle_cy_payload(
                self,
                mode=payload_mode,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                reuse_payload=reuse_payload,
                history_state=history_state_pre,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
            )
            payload_fields.update(oracle_fields)
        elif selector_mode:
            cache_payload, selector_fields, selector_aux = _selector_payload(
                self,
                mode=payload_mode,
                reuse=reuse_payload,
                residual_history_state=history_state_pre,
                output_history_state=output_history_state_pre,
                update_history_state=update_history_state_pre,
                hidden_states=hidden_states,
                temb=temb,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                step_size=step_size,
            )
            payload_fields.update(selector_fields)
        elif update_scalar_calib_mode:
            cache_payload, calib_fields, selector_aux = _update_scalar_calib_payload(
                self,
                mode=payload_mode,
                reuse=reuse_payload,
                residual_history_state=history_state_pre,
                update_history_state=update_history_state_pre,
                hidden_states=hidden_states,
                temb=temb,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                step_size=step_size,
            )
            payload_fields.update(calib_fields)
        elif update_inverse_mode:
            cache_payload, inverse_fields, selector_aux = _update_inverse_payload(
                self,
                mode=payload_mode,
                reuse=reuse_payload,
                residual_history_state=history_state_pre,
                update_history_state=update_history_state_pre,
                hidden_states=hidden_states,
                temb=temb,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                step_size=step_size,
            )
            payload_fields.update(inverse_fields)
        elif forecast_opt_mode:
            forecast_payload, forecast_fields = _forecast_opt_payload(
                self,
                mode=payload_mode,
                reuse=reuse_payload,
                residual_history_state=history_state_pre,
                opt_state=forecast_opt_state_pre,
                hidden_states=hidden_states,
                temb=temb,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
            )
            blend = float(getattr(self, "seacache_payload_blend", 1.0))
            if forecast_fields["payload_used"] == "reuse":
                cache_payload = reuse_payload
            else:
                cache_payload = (1.0 - blend) * reuse_payload + blend * forecast_payload
            payload_fields.update(forecast_fields)
            payload_fields["payload_blend"] = blend
            payload_fields["payload_chosen_norm"] = _norm(cache_payload)
            if forecast_fields["payload_used"] != "reuse":
                payload_fields["payload_raw_delta_from_reuse_norm"] = forecast_fields.get(
                    "payload_delta_from_reuse_norm"
                )
            if forecast_fields["payload_used"] != "reuse" and blend != 1.0:
                payload_fields["payload_used"] = f"blend_{forecast_fields['payload_used']}"
                payload_fields["payload_delta_from_reuse_norm"] = _norm(
                    cache_payload.detach().to(torch.float32) - reuse_payload.detach().to(torch.float32)
                )
        else:
            forecast_payload, forecast_fields = _forecast_payload(
                payload_mode,
                reuse=reuse_payload,
                history_state=history_state_pre,
                svd_state=getattr(self, "seacache_payload_svd_state", None),
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                num_steps=int(getattr(self, "num_steps", 0)),
                prompt_idx=getattr(self, "seacache_payload_prompt_idx", None),
                payload_bank_dir=getattr(self, "seacache_payload_bank_dir", None),
                payload_wrong_seed_bank_dir=getattr(self, "seacache_payload_wrong_seed_bank_dir", None),
                payload_bank_write_dir=getattr(self, "seacache_payload_bank_write_dir", None),
                payload_control_seed_salt=int(getattr(self, "seacache_payload_control_seed_salt", 0)),
            )
            blend = float(getattr(self, "seacache_payload_blend", 1.0))
            if (
                forecast_fields["payload_used"] == "reuse"
                or payload_mode == "reuse"
            ):
                cache_payload = reuse_payload
            else:
                cache_payload = (1.0 - blend) * reuse_payload + blend * forecast_payload
            payload_fields.update(forecast_fields)
            payload_fields["payload_blend"] = blend
            payload_fields["payload_chosen_norm"] = _norm(cache_payload)
            if forecast_fields["payload_used"] != "reuse":
                payload_fields["payload_raw_delta_from_reuse_norm"] = forecast_fields.get(
                    "payload_delta_from_reuse_norm"
                )
            if forecast_fields["payload_used"] != "reuse" and blend != 1.0:
                payload_fields["payload_used"] = f"blend_{forecast_fields['payload_used']}"
                payload_fields["payload_delta_from_reuse_norm"] = _norm(
                    cache_payload.detach().to(torch.float32) - reuse_payload.detach().to(torch.float32)
                )
        if getattr(self, "seacache_payload_shadow_full_residual", False) and not posterior_oracle_mode:
            shadow_fields = _shadow_full_residual_fields(
                self,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                reuse_payload=reuse_payload,
                chosen_payload=cache_payload,
                history_state=history_state_pre,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                selector_chosen=payload_fields.get("payload_selector_chosen"),
                selector_target_payload=selector_aux.get("target_payload"),
                selector_target_output=selector_aux.get("target_output"),
                selector_target_update=selector_aux.get("target_update"),
            )
        hidden_states = hidden_states + cache_payload
    else:
        ori_hidden_states = hidden_states
        encoder_hidden_states, hidden_states = _full_transformer_residual(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        self.previous_residual = hidden_states - ori_hidden_states
        full_residual_norm = _norm(self.previous_residual)
        full_output = None
        full_update = None
        if (
            update_history_mode
            or bool(guard_fields.get("forecast_guard_enabled"))
            or _forecast_opt_needs_output_records(payload_mode)
        ):
            with torch.inference_mode():
                full_output = _project_output(self, hidden_states, temb)
            full_output_norm = _norm(full_output)
            full_update = None if step_size is None else full_output * float(step_size)
            full_update_norm = _norm(full_update)
        if update_history_mode:
            self.seacache_payload_output_history_state = update_history_update_on_full(
                output_history_state_pre,
                tensor=full_output,
                step=cnt,
                max_order=2,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                ema_beta=float(getattr(self, "seacache_payload_ema_beta", 0.2)),
            )
            if full_update is not None:
                self.seacache_payload_update_history_state = update_history_update_on_full(
                    update_history_state_pre,
                    tensor=full_update,
                    step=cnt,
                    max_order=2,
                    sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
                    ema_beta=float(getattr(self, "seacache_payload_ema_beta", 0.2)),
                )
            update_history_updated = True
        if bool(guard_fields.get("forecast_guard_enabled")):
            guard_state_post, guard_obs_fields = _forecast_guard_update_on_full(
                getattr(self, "seacache_payload_forecast_guard_state", None),
                forecast_update=guard_forecast_update_pre,
                actual_update=full_update,
                self_scale_rel=guard_fields.get("forecast_guard_self_scale_rel_pre"),
                scale_floor=float(getattr(self, "seacache_payload_forecast_guard_scale_floor", 1e-4)),
                window=int(getattr(self, "seacache_payload_forecast_guard_window", 16)),
                quantile=float(getattr(self, "seacache_payload_forecast_guard_quantile", 0.8)),
            )
            self.seacache_payload_forecast_guard_state = guard_state_post
            guard_fields.update(guard_obs_fields)
        if forecast_opt_mode:
            self.seacache_payload_forecast_opt_state, forecast_opt_update_fields = _forecast_opt_update_on_full(
                forecast_opt_state_pre,
                residual_history_state=history_state_pre,
                residual=self.previous_residual,
                hidden_states=ori_hidden_states,
                gate_input=modulated_inp,
                output=full_output,
                step=cnt,
                sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
            )
            payload_fields.update(forecast_opt_update_fields)
        if svd_cache_mode:
            self.seacache_payload_svd_state, svd_update_fields = svdcache_payload.update_on_full(
                getattr(self, "seacache_payload_svd_state", None),
                residual=self.previous_residual,
                step=cnt,
                mode=payload_mode,
            )
            payload_fields.update(svd_update_fields)
        self.seacache_payload_history_fd_state = history_fd_update_on_full(
            history_state_pre,
            residual=self.previous_residual,
            step=cnt,
            max_order=2,
            sigma=float(getattr(self, "seacache_payload_sigma", 0.5)),
            ema_beta=float(getattr(self, "seacache_payload_ema_beta", 0.2)),
        )
        history_updated = True

    if hasattr(self, "seacache_payload_decisions"):
        self.seacache_payload_decisions.append({
            "step": int(cnt),
            "u": int(not should_calc),
            "native_u": int(not native_should_calc) if force_full_reason is None else 0,
            "native_gate_kind": native_gate_kind,
            "native_gate_cache_allowed": bool(seacache_gate_cache_allowed),
            "seacache_gate_cache_allowed": bool(seacache_gate_cache_allowed),
            "combined_gate_cache_allowed": combined_gate_cache_allowed,
            "combined_gate_veto_reason": combined_gate_veto_reason,
            "schedule_locked": bool(schedule_locked),
            "schedule_u": schedule_u,
            "threshold": threshold,
            "teacache_backbone": getattr(self, "teacache_backbone", None),
            "teacache_variant": getattr(self, "teacache_variant", None),
            "teacache_coefficients_hash": getattr(self, "teacache_coefficients_hash", None),
            "force_full": bool(force_full_reason is not None),
            "force_full_reason": force_full_reason,
            "accumulator_before": accumulator_before,
            "sea_increment_rel_l1": sea_increment,
            "gate_increment_raw_rel_l1": gate_increment_raw,
            "gate_increment_rescaled": gate_increment_rescaled,
            "accumulator_after_increment": float(accumulator_after_increment),
            "accumulator_after_commit": float(self.accumulated_rel_l1_distance),
            "accumulator_reset": bool(should_calc),
            "previous_modulated_input_present": True,
            "previous_residual_present": self.previous_residual is not None,
            "previous_residual_norm": _norm(self.previous_residual),
            "cnt_pre": int(cnt),
            "cnt_post": int((cnt + 1) % int(self.num_steps)),
            "history_updated": bool(history_updated),
            "full_residual_norm": full_residual_norm,
            "full_output_norm": full_output_norm,
            "full_update_norm": full_update_norm,
            "update_history_updated": bool(update_history_updated),
            "update_history_update_source": "full" if update_history_updated else None,
            "history_fd_sigma": float(getattr(self, "seacache_payload_sigma", 0.5)),
            **history_fields,
            **payload_fields,
            **guard_fields,
            **rfc_gate_fields,
            **shadow_fields,
        })

    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install(
    pipe,
    *,
    threshold: float,
    num_steps: int,
    first_enhance: int = 1,
    payload_mode: str = "reuse",
    payload_blend: float = 1.0,
    payload_sigma: float = 0.5,
    payload_control_seed_salt: int = 0,
    shadow_full_residual: bool = False,
    payload_bank_dir: Optional[Path] = None,
    payload_wrong_seed_bank_dir: Optional[Path] = None,
    payload_bank_write_dir: Optional[Path] = None,
    payload_gate_mode: str = "seacache",
    forecast_guard_observe: bool = False,
    forecast_guard_mode: str = "auto",
    forecast_guard_tau: float = 0.05,
    forecast_guard_quantile: float = 0.8,
    forecast_guard_window: int = 16,
    forecast_guard_warmup_updates: int = 1,
    forecast_guard_age_gamma: float = 0.0,
    forecast_guard_scale_floor: float = 1e-4,
    rfc_gate_tau: Optional[float] = None,
    teacache_backbone: str = "flux",
    teacache_variant: Optional[str] = None,
) -> Callable[[], None]:
    if payload_mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown payload_mode: {payload_mode!r}")
    if payload_gate_mode not in PAYLOAD_GATE_MODES:
        raise ValueError(f"unknown payload_gate_mode: {payload_gate_mode!r}")
    if forecast_guard_mode != "auto" and forecast_guard_mode not in FORECAST_GUARD_MODES:
        raise ValueError(f"unknown forecast_guard_mode: {forecast_guard_mode!r}")
    if not (0.0 <= float(payload_blend) <= 2.0):
        raise ValueError(f"payload_blend must be in [0, 2], got {payload_blend}")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _seacache_payload_forward

    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_seacache_payload = True
    tr.seacache_thresh = float(threshold)
    tr.num_steps = int(num_steps)
    tr.first_enhance = int(first_enhance)
    tr.seacache_payload_mode = str(payload_mode)
    tr.seacache_payload_blend = float(payload_blend)
    tr.seacache_payload_sigma = float(payload_sigma)
    tr.seacache_payload_control_seed_salt = int(payload_control_seed_salt)
    tr.seacache_payload_ema_beta = 0.2
    tr.seacache_payload_shadow_full_residual = bool(shadow_full_residual)
    tr.seacache_payload_bank_dir = None if payload_bank_dir is None else Path(payload_bank_dir)
    tr.seacache_payload_wrong_seed_bank_dir = (
        None if payload_wrong_seed_bank_dir is None else Path(payload_wrong_seed_bank_dir)
    )
    tr.seacache_payload_bank_write_dir = (
        None if payload_bank_write_dir is None else Path(payload_bank_write_dir)
    )
    tr.seacache_payload_gate_mode = str(payload_gate_mode)
    tr.seacache_payload_forecast_guard_observe = bool(forecast_guard_observe)
    tr.seacache_payload_forecast_guard_mode = str(forecast_guard_mode)
    tr.seacache_payload_forecast_guard_tau = float(forecast_guard_tau)
    tr.seacache_payload_forecast_guard_quantile = float(forecast_guard_quantile)
    tr.seacache_payload_forecast_guard_window = int(forecast_guard_window)
    tr.seacache_payload_forecast_guard_warmup_updates = int(forecast_guard_warmup_updates)
    tr.seacache_payload_forecast_guard_age_gamma = float(forecast_guard_age_gamma)
    tr.seacache_payload_forecast_guard_scale_floor = float(forecast_guard_scale_floor)
    tr.seacache_payload_rfc_gate_tau = float(threshold if rfc_gate_tau is None else rfc_gate_tau)
    if str(payload_gate_mode) == "teacache":
        coeffs = get_coeffs(str(teacache_backbone), teacache_variant)
        tr.teacache_rescale = np.poly1d(coeffs)
        tr.teacache_backbone = str(teacache_backbone)
        tr.teacache_variant = teacache_variant
        tr.teacache_coefficients_hash = _coefficients_hash(coeffs)
    else:
        tr.teacache_rescale = None
        tr.teacache_backbone = None
        tr.teacache_variant = None
        tr.teacache_coefficients_hash = None
    tr.seacache_payload_prompt_idx = None
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.seacache_payload_rfc_accumulated_error = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    tr.seacache_payload_history_fd_state = init_history_fd_state()
    tr.seacache_payload_output_history_state = init_update_history_state()
    tr.seacache_payload_update_history_state = init_update_history_state()
    tr.seacache_payload_forecast_guard_state = init_forecast_guard_state()
    tr.seacache_payload_forecast_opt_state = _init_forecast_opt_state()
    tr.seacache_payload_svd_state = svdcache_payload.init_state()
    tr.seacache_payload_action_steps = None
    tr.seacache_payload_decisions = []

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_seacache_payload", "seacache_thresh", "num_steps", "first_enhance",
            "seacache_payload_mode", "seacache_payload_blend", "seacache_payload_sigma",
            "seacache_payload_control_seed_salt",
            "seacache_payload_ema_beta", "seacache_payload_shadow_full_residual",
            "seacache_payload_bank_dir", "seacache_payload_wrong_seed_bank_dir",
            "seacache_payload_bank_write_dir", "seacache_payload_gate_mode",
            "seacache_payload_forecast_guard_observe",
            "seacache_payload_forecast_guard_mode",
            "seacache_payload_forecast_guard_tau",
            "seacache_payload_forecast_guard_quantile",
            "seacache_payload_forecast_guard_window",
            "seacache_payload_forecast_guard_warmup_updates",
            "seacache_payload_forecast_guard_age_gamma",
            "seacache_payload_forecast_guard_scale_floor",
            "seacache_payload_rfc_gate_tau",
            "teacache_rescale", "teacache_backbone", "teacache_variant",
            "teacache_coefficients_hash",
            "seacache_payload_prompt_idx",
            "cnt", "accumulated_rel_l1_distance",
            "seacache_payload_rfc_accumulated_error",
            "previous_modulated_input", "previous_residual",
            "seacache_payload_history_fd_state",
            "seacache_payload_output_history_state", "seacache_payload_update_history_state",
            "seacache_payload_forecast_guard_state",
            "seacache_payload_forecast_opt_state",
            "seacache_payload_svd_state",
            "seacache_payload_action_steps",
            "seacache_payload_decisions", "scheduler",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(
    pipe,
    action_steps: Optional[Set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    tr = pipe.transformer
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.seacache_payload_rfc_accumulated_error = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    tr.seacache_payload_history_fd_state = init_history_fd_state()
    tr.seacache_payload_output_history_state = init_update_history_state()
    tr.seacache_payload_update_history_state = init_update_history_state()
    tr.seacache_payload_forecast_guard_state = init_forecast_guard_state()
    tr.seacache_payload_forecast_opt_state = _init_forecast_opt_state()
    tr.seacache_payload_svd_state = svdcache_payload.init_state()
    tr.seacache_payload_action_steps = None if action_steps is None else set(int(x) for x in action_steps)
    tr.seacache_payload_prompt_idx = None if prompt_idx is None else int(prompt_idx)
    tr.seacache_payload_decisions = []
