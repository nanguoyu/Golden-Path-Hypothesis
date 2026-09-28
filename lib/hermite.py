"""Physicist's Hermite polynomial + HiCache cache-then-forecast math.

Paper: HiCache (arXiv:2508.16984). The predictor is

    F_pred(x) = F_0 + sum_{k=1..O} [ H_k(sigma*x) / k! ] * sigma^k * Delta^k F

which is the dual-scaled form `tilde_H_n(x) = sigma^n * H_n(sigma*x)` applied to
the standard Taylor cache, with O = max_order, sigma in (0, 1] (paper uses 0.5).

`Delta^k F` are finite differences computed at activation steps; see
`hermite_update` for the recurrence.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def hermite_poly(x: torch.Tensor, n: int) -> torch.Tensor:
    """Physicist's Hermite polynomial H_n(x) via the recurrence
    H_{k+1}(x) = 2x H_k(x) - 2k H_{k-1}(x), H_0=1, H_1=2x.

    Vectorized over x. Stable for the small `n` we use (typically 1..3).
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2 * x
    h_prev = torch.ones_like(x)
    h_curr = 2 * x
    for k in range(2, n + 1):
        h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def hicache_predict(
    history: Dict[int, torch.Tensor],
    step_offset: int,
    sigma: float,
    max_order: int,
) -> torch.Tensor:
    """Predict feature at (last_activation_step + step_offset) using HiCache.

    Args:
        history:     {0: F_0, 1: Delta^1 F, ..., k: Delta^k F}. Keys 0..K present.
        step_offset: integer step distance from the last activation step.
        sigma:       HiCache scaling factor (paper default 0.5).
        max_order:   paper O. Effective order is `min(max_order, len(history)-1)`.

    Returns:
        Predicted feature tensor of the same shape/dtype/device as F_0.
    """
    order_avail = len(history) - 1
    order = min(max_order, order_avail)
    if order < 1:
        return history[0]

    F0 = history[0]
    x = torch.tensor(float(step_offset), dtype=F0.dtype, device=F0.device)
    x_scaled = x * sigma

    pred = F0.clone()
    for k in range(1, order + 1):
        Hk = hermite_poly(x_scaled, k)
        alpha = float(Hk / math.factorial(k)) * (sigma**k)
        pred.add_(history[k], alpha=alpha)
    return pred


def hermite_update(
    prev_history: Optional[Dict[int, torch.Tensor]],
    feature: torch.Tensor,
    step_gap: int,
    max_order: int,
) -> Dict[int, torch.Tensor]:
    """Update the Delta^k history with a fresh activation feature.

    Computes higher-order differences from `feature` (new F_0) and the previous
    history's Delta^{0..k-1}, divided by `step_gap` (the integer step distance
    between the previous and current activation). Returns a NEW dict (does not
    mutate `prev_history`).

    Args:
        prev_history: history from the previous activation, or None on first
                      activation. None / empty / {0: ...} only → returns {0: F}.
        feature:      F at this activation step.
        step_gap:     positive integer step distance since previous activation.
        max_order:    highest k to compute. Output may have keys up to max_order.

    Returns:
        new_history: {0: F, 1: Delta^1 F, ..., k: Delta^k F} for k <= max_order
                     where higher-k entries stop as soon as prev_history runs out.
    """
    new: Dict[int, torch.Tensor] = {0: feature}
    if not prev_history or step_gap <= 0:
        return new
    for k in range(max_order):
        prev_k = prev_history.get(k)
        if prev_k is None:
            break
        new[k + 1] = (new[k] - prev_k) / step_gap
    return new
