"""Finite-difference history observables for online cache-risk probes.

This module intentionally stays outside the locked baseline implementations.
It turns TaylorSeer/HiCache-style activation histories into scalar online
features and delayed self-supervised prediction errors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from lib.hermite import hermite_update, hicache_predict
from lib.taylor import taylor_predict

EPS = 1e-12

History = Dict[int, torch.Tensor]


def clone_history(history: Optional[History]) -> History:
    if not history:
        return {}
    return {int(k): v.detach().clone() for k, v in history.items()}


def norm_optional(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.detach().to(torch.float32).norm().item())


def cos_optional(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    af = a.detach().to(torch.float32)
    bf = b.detach().to(torch.float32)
    an = float(af.norm().item())
    bn = float(bf.norm().item())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.sum(af * bf).item()) / (an * bn + EPS)


def init_state() -> Dict[str, Any]:
    return {
        "history": {},
        "anchor_step": None,
        "error_ema": {},
        "error_updates": 0,
    }


def clone_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = state or init_state()
    return {
        "history": clone_history(src.get("history")),
        "anchor_step": src.get("anchor_step"),
        "error_ema": dict(src.get("error_ema") or {}),
        "error_updates": int(src.get("error_updates") or 0),
    }


def _predictions(history: History, step_gap: int, sigma: float) -> Dict[str, torch.Tensor]:
    if not history:
        return {}
    out: Dict[str, torch.Tensor] = {"reuse": history[0]}
    order_avail = max(int(k) for k in history.keys())
    if order_avail >= 1:
        out["taylor_o1"] = taylor_predict(history, step_gap, max_order=1)
    if order_avail >= 2:
        out["taylor_o2"] = taylor_predict(history, step_gap, max_order=2)
        out["hicache_o2"] = hicache_predict(history, step_gap, sigma=sigma, max_order=2)
    return out


def forecast_predictions(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    sigma: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Return residual forecasts computed only from past full residual history."""

    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    gap = 0 if anchor_step is None else max(int(step) - int(anchor_step), 0)
    return {
        name: value.detach()
        for name, value in _predictions(history, gap, sigma=sigma).items()
    }


def ffro_residual_fields(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    sigma: float = 0.5,
    prefix: str = "online_ffro",
) -> Dict[str, Any]:
    """Forecasted-full residual observer fields.

    These fields compare the committed reuse residual against TaylorSeer /
    HiCache-style forecasts of the current residual. They are decision-time
    observables because both the reuse residual and the forecast history come
    only from prior full-refresh steps.
    """

    st = clone_state(state)
    preds = forecast_predictions(st, step=step, sigma=sigma)
    reuse = preds.get("reuse")

    def dist(name: str) -> Optional[float]:
        pred = preds.get(name)
        if reuse is None or pred is None or reuse.shape != pred.shape:
            return None
        return float((reuse.to(torch.float32) - pred.to(torch.float32)).norm().item())

    reuse_norm = norm_optional(reuse)

    def rel(value: Optional[float]) -> Optional[float]:
        if value is None or reuse_norm is None:
            return None
        return float(value) / (float(reuse_norm) + EPS)

    t1 = dist("taylor_o1")
    t2 = dist("taylor_o2")
    h2 = dist("hicache_o2")
    t2_h2 = None
    t1_h2 = None
    t1_pred = preds.get("taylor_o1")
    t2_pred = preds.get("taylor_o2")
    h2_pred = preds.get("hicache_o2")
    if t2_pred is not None and h2_pred is not None and t2_pred.shape == h2_pred.shape:
        t2_h2 = float((t2_pred.to(torch.float32) - h2_pred.to(torch.float32)).norm().item())
    if t1_pred is not None and h2_pred is not None and t1_pred.shape == h2_pred.shape:
        t1_h2 = float((t1_pred.to(torch.float32) - h2_pred.to(torch.float32)).norm().item())

    available = [x for x in (t1, t2, h2) if x is not None]
    ensemble_min = min(available) if available else None
    ensemble_max = max(available) if available else None
    ensemble_mean = float(sum(available) / len(available)) if available else None
    return {
        f"{prefix}_available_pre": bool(available),
        f"{prefix}_reuse_residual_norm_pre": reuse_norm,
        f"{prefix}_taylor_o1_res_norm_pre": t1,
        f"{prefix}_taylor_o1_res_rel_pre": rel(t1),
        f"{prefix}_taylor_o2_res_norm_pre": t2,
        f"{prefix}_taylor_o2_res_rel_pre": rel(t2),
        f"{prefix}_hicache_o2_res_norm_pre": h2,
        f"{prefix}_hicache_o2_res_rel_pre": rel(h2),
        f"{prefix}_taylor_o2_hicache_o2_spread_norm_pre": t2_h2,
        f"{prefix}_taylor_o1_hicache_o2_spread_norm_pre": t1_h2,
        f"{prefix}_ensemble_res_min_pre": ensemble_min,
        f"{prefix}_ensemble_res_mean_pre": ensemble_mean,
        f"{prefix}_ensemble_res_max_pre": ensemble_max,
    }


def online_fields(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    sigma: float = 0.5,
    prefix: str = "online_fd",
) -> Dict[str, Any]:
    """Return decision-time fields computed only from past full activations."""

    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    gap = None if anchor_step is None else int(step) - int(anchor_step)
    order_avail = max(history.keys()) if history else -1
    d0 = history.get(0)
    d1 = history.get(1)
    d2 = history.get(2)
    d0_norm = norm_optional(d0)
    d1_norm = norm_optional(d1)
    d2_norm = norm_optional(d2)
    gap_for_pred = max(int(gap or 0), 0)
    preds = _predictions(history, gap_for_pred, sigma=sigma)
    reuse = preds.get("reuse")
    t1 = preds.get("taylor_o1")
    t2 = preds.get("taylor_o2")
    h2 = preds.get("hicache_o2")

    def rel(x: Optional[float], base: Optional[float] = d0_norm) -> Optional[float]:
        if x is None or base is None:
            return None
        return float(x) / (float(base) + EPS)

    def dist(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
        if a is None or b is None or a.shape != b.shape:
            return None
        return float((a.detach().to(torch.float32) - b.detach().to(torch.float32)).norm().item())

    t1_drift = dist(t1, reuse)
    t2_drift = dist(t2, reuse)
    t2_minus_t1 = dist(t2, t1)
    h2_minus_t2 = dist(h2, t2)
    h2_minus_t1 = dist(h2, t1)

    ema = st.get("error_ema") or {}
    return {
        f"{prefix}_anchor_present_pre": bool(history),
        f"{prefix}_anchor_step_pre": None if anchor_step is None else int(anchor_step),
        f"{prefix}_anchor_gap_pre": gap,
        f"{prefix}_order_avail_pre": int(order_avail),
        f"{prefix}_d0_norm_pre": d0_norm,
        f"{prefix}_d1_norm_pre": d1_norm,
        f"{prefix}_d1_rel_pre": rel(d1_norm),
        f"{prefix}_d2_norm_pre": d2_norm,
        f"{prefix}_d2_rel_pre": rel(d2_norm),
        f"{prefix}_d1_d2_cos_pre": cos_optional(d1, d2),
        f"{prefix}_taylor_o1_drift_norm_pre": t1_drift,
        f"{prefix}_taylor_o1_drift_rel_pre": rel(t1_drift),
        f"{prefix}_taylor_o2_drift_norm_pre": t2_drift,
        f"{prefix}_taylor_o2_drift_rel_pre": rel(t2_drift),
        f"{prefix}_taylor_o2_minus_o1_norm_pre": t2_minus_t1,
        f"{prefix}_taylor_o2_minus_o1_rel_pre": rel(t2_minus_t1),
        f"{prefix}_hicache_o2_minus_taylor_o2_norm_pre": h2_minus_t2,
        f"{prefix}_hicache_o2_minus_taylor_o2_rel_pre": rel(h2_minus_t2),
        f"{prefix}_hicache_o2_minus_o1_norm_pre": h2_minus_t1,
        f"{prefix}_hicache_o2_minus_o1_rel_pre": rel(h2_minus_t1),
        f"{prefix}_taylor_o1_innovation_ema_pre": ema.get("taylor_o1"),
        f"{prefix}_taylor_o2_innovation_ema_pre": ema.get("taylor_o2"),
        f"{prefix}_hicache_o2_innovation_ema_pre": ema.get("hicache_o2"),
        f"{prefix}_innovation_update_count_pre": int(st.get("error_updates") or 0),
    }


def update_on_full(
    state: Optional[Dict[str, Any]],
    *,
    residual: torch.Tensor,
    step: int,
    max_order: int = 2,
    sigma: float = 0.5,
    ema_beta: float = 0.2,
) -> Dict[str, Any]:
    """Update history after a full forward and record delayed prediction error."""

    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    step_gap = 0 if anchor_step is None else max(int(step) - int(anchor_step), 1)
    preds = _predictions(history, step_gap, sigma=sigma) if history else {}
    actual = residual.detach()
    actual_norm = float(actual.to(torch.float32).norm().item())
    err_ema = dict(st.get("error_ema") or {})
    updated_any = False
    for name in ("taylor_o1", "taylor_o2", "hicache_o2"):
        pred = preds.get(name)
        if pred is None or pred.shape != actual.shape:
            continue
        err = float((pred.to(torch.float32) - actual.to(torch.float32)).norm().item())
        err_rel = err / (actual_norm + EPS)
        prev = err_ema.get(name)
        err_ema[name] = err_rel if prev is None else (
            (1.0 - float(ema_beta)) * float(prev) + float(ema_beta) * err_rel
        )
        updated_any = True
    new_history = hermite_update(history, actual.detach(), step_gap, max_order=max_order)
    st["history"] = clone_history(new_history)
    st["anchor_step"] = int(step)
    st["error_ema"] = err_ema
    if updated_any:
        st["error_updates"] = int(st.get("error_updates") or 0) + 1
    return st


def digest_payload(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    st = clone_state(state)
    history = st["history"]
    return {
        "anchor_step": st.get("anchor_step"),
        "order_keys": sorted(int(k) for k in history.keys()),
        "history_norms": {
            str(k): norm_optional(v)
            for k, v in sorted(history.items())
        },
        "error_ema": dict(st.get("error_ema") or {}),
        "error_updates": int(st.get("error_updates") or 0),
    }
