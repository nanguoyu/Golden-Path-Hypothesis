"""Coarse residual payloads for Wan2.1 fixed-schedule cache experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from lib.history_fd_observer import (
    forecast_predictions,
    init_state as init_history_state,
    update_on_full,
)


PAYLOAD_MODES = ("reuse", "taylor_o1", "taylor_o2", "hicache_o2", "ensemble_mean")


def init_branch_payload_state() -> Dict[str, Any]:
    return init_history_state()


def update_full(
    state: Optional[Dict[str, Any]],
    *,
    residual: torch.Tensor,
    step: int,
    sigma: float = 0.5,
) -> Dict[str, Any]:
    return update_on_full(
        state,
        residual=residual.detach(),
        step=int(step),
        max_order=2,
        sigma=float(sigma),
    )


def _norm(x: Optional[torch.Tensor]) -> Optional[float]:
    if x is None:
        return None
    return float(x.detach().to(torch.float32).norm().item())


def choose_payload(
    state: Optional[Dict[str, Any]],
    *,
    step: int,
    mode: str,
    sigma: float,
    reuse: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown Wan2.1 payload mode: {mode}")

    preds = forecast_predictions(state, step=int(step), sigma=float(sigma))
    available = sorted(preds.keys())
    selected = mode
    fallback = False

    if mode == "ensemble_mean":
        members = [
            preds[name]
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            if name in preds and preds[name].shape == reuse.shape
        ]
        if members:
            payload = torch.stack([m.to(reuse.dtype) for m in members], dim=0).mean(dim=0)
        else:
            payload = reuse
            fallback = True
            selected = "reuse"
    elif mode == "reuse":
        payload = reuse
    else:
        pred = preds.get(mode)
        if pred is None or pred.shape != reuse.shape:
            payload = reuse
            fallback = True
            selected = "reuse"
        else:
            payload = pred.to(reuse.dtype)

    return payload, {
        "payload_requested": mode,
        "payload_selected": selected,
        "payload_available": available,
        "payload_fallback": bool(fallback),
        "payload_norm": _norm(payload),
        "reuse_residual_norm": _norm(reuse),
    }
