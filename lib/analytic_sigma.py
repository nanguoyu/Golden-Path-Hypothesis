"""HiCache-Analytic: per-layer adaptive sigma based on online delta statistics.

Maintains an EMA (or quantile) estimate `q_l` of the magnitude of Delta^1 F per
layer l, and computes

    sigma_l = min(sigma_max, alpha / (sqrt(2) * q_l + eps))

Optional log-domain smoothing applies an EMA to sigma_l across steps to avoid
chattering. Lifted from the HiCache reference fork's `taylor_utils` and made
backbone-agnostic (no `cache_dic` coupling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class AnalyticSigmaConfig:
    """Configuration for analytic adaptive sigma.

    Defaults match the HiCache fork's defaults: alpha=1.28 gives initial
    sigma ≈ 0.9 when q ≈ 1.
    """
    alpha: float = 1.28
    sigma_max: float = 1.0
    beta: float = 0.01            # EMA coefficient; beta<=0 disables online update
    eps: float = 1e-6
    q_quantile: Optional[float] = None  # in (0,1) → use quantile of |delta| instead of mean
    sigma_smooth: float = 0.0     # gamma in log-sigma EMA; 0 disables smoothing


@dataclass
class AnalyticSigmaState:
    """Per-layer running state. `q[layer]` is the magnitude estimate."""
    q: Dict[int, float] = field(default_factory=dict)
    sigma_smoothed: Dict[int, float] = field(default_factory=dict)


def update_q(
    state: AnalyticSigmaState,
    layer_key: int,
    delta_tensor: torch.Tensor,
    cfg: AnalyticSigmaConfig,
) -> None:
    """Update q_l from a fresh Delta^1 F sample. Skips silently when beta<=0."""
    if cfg.beta <= 0:
        return
    with torch.no_grad():
        norms = delta_tensor.detach().float().norm(dim=-1)
        if cfg.q_quantile is not None and 0.0 < cfg.q_quantile < 1.0:
            d_val = float(torch.quantile(norms, cfg.q_quantile))
        else:
            d_val = float(norms.mean())
    q_old = state.q.get(layer_key)
    if q_old is None:
        state.q[layer_key] = d_val
    else:
        # Square-domain EMA: q_new = sqrt((1-beta) * q_old^2 + beta * d_val^2)
        q_sq_new = (1.0 - cfg.beta) * (q_old**2) + cfg.beta * (d_val**2)
        state.q[layer_key] = math.sqrt(q_sq_new)


def compute_sigma(
    state: AnalyticSigmaState,
    layer_key: int,
    cfg: AnalyticSigmaConfig,
) -> float:
    """Return the per-layer sigma. Applies optional log-domain smoothing."""
    q = state.q.get(layer_key, 1.0)
    raw = cfg.alpha / (math.sqrt(2.0) * q + cfg.eps)
    sigma = min(cfg.sigma_max, raw)
    if cfg.sigma_smooth and cfg.sigma_smooth > 0:
        prev = state.sigma_smoothed.get(layer_key, sigma)
        gamma = float(cfg.sigma_smooth)
        log_prev = math.log(max(prev, cfg.eps))
        log_new = math.log(max(sigma, cfg.eps))
        sigma = math.exp((1.0 - gamma) * log_prev + gamma * log_new)
        sigma = min(cfg.sigma_max, sigma)
        state.sigma_smoothed[layer_key] = sigma
    return sigma
