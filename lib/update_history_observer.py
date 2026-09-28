"""History-only output/update forecast helpers for payload experiments.

The helpers mirror ``history_fd_observer`` but keep the naming generic enough
for FLUX output and solver-update tensors.  They are research-only utilities:
all histories must be updated from committed full steps, never from shadow full
observer calls.
"""

from __future__ import annotations

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


def norm_optional(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    return float(tensor.detach().to(torch.float32).norm().item())


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
    parts = [
        out[name]
        for name in ("taylor_o1", "taylor_o2", "hicache_o2")
        if name in out
    ]
    if parts:
        out["ensemble_mean"] = torch.stack(parts, dim=0).mean(dim=0)
    return out


def forecast_predictions(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    sigma: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Return forecasts computed only from prior committed full-step tensors."""

    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    gap = 0 if anchor_step is None else max(int(step) - int(anchor_step), 0)
    return {
        name: value.detach()
        for name, value in _predictions(history, gap, sigma=sigma).items()
    }


def best_target(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    sigma: float = 0.5,
    preferred: str = "ensemble_mean",
) -> tuple[Optional[str], Optional[torch.Tensor]]:
    """Return the preferred forecast target, falling back to the best available one."""

    preds = forecast_predictions(state, step=step, sigma=sigma)
    for name in (preferred, "taylor_o2", "hicache_o2", "taylor_o1", "reuse"):
        tensor = preds.get(name)
        if tensor is not None:
            return name, tensor
    return None, None


def online_fields(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    prefix: str,
) -> Dict[str, Any]:
    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    return {
        f"{prefix}_anchor_present_pre": bool(history),
        f"{prefix}_anchor_step_pre": None if anchor_step is None else int(anchor_step),
        f"{prefix}_anchor_gap_pre": None if anchor_step is None else int(step) - int(anchor_step),
        f"{prefix}_order_avail_pre": max(history.keys()) if history else -1,
        f"{prefix}_d0_norm_pre": norm_optional(history.get(0)),
        f"{prefix}_d1_norm_pre": norm_optional(history.get(1)),
        f"{prefix}_d2_norm_pre": norm_optional(history.get(2)),
        f"{prefix}_innovation_update_count_pre": int(st.get("error_updates") or 0),
    }


def update_on_full(
    state: Optional[Dict[str, Any]],
    *,
    tensor: torch.Tensor,
    step: int,
    max_order: int = 2,
    sigma: float = 0.5,
    ema_beta: float = 0.2,
) -> Dict[str, Any]:
    """Update history after a committed full step and record delayed errors."""

    st = clone_state(state)
    history: History = st["history"]
    anchor_step = st.get("anchor_step")
    step_gap = 0 if anchor_step is None else max(int(step) - int(anchor_step), 1)
    preds = _predictions(history, step_gap, sigma=sigma) if history else {}
    actual = tensor.detach()
    actual_norm = float(actual.to(torch.float32).norm().item())
    err_ema = dict(st.get("error_ema") or {})
    updated_any = False
    for name in ("taylor_o1", "taylor_o2", "hicache_o2", "ensemble_mean"):
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

