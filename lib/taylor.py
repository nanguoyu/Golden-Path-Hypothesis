"""TaylorSeer-style Taylor expansion predictor.

The predictor is

    F_pred(x) = F_0 + sum_{k=1..O} (x^k / k!) * Delta^k F

The Delta^k recurrence is the same as HiCache (see `lib.hermite.hermite_update`),
so this module only provides the prediction formula and a thin re-export of the
shared updater.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from .hermite import hermite_update as taylor_update  # noqa: F401  re-export


def taylor_predict(
    history: Dict[int, torch.Tensor],
    step_offset: int,
    max_order: int,
) -> torch.Tensor:
    """Predict feature at (last_activation_step + step_offset) via Taylor.

    Args:
        history:     {0: F_0, 1: Delta^1 F, ..., k: Delta^k F}.
        step_offset: integer step distance from last activation step.
        max_order:   highest order used. Effective order is min(max_order, len-1).
    """
    order_avail = len(history) - 1
    order = min(max_order, order_avail)
    if order < 1:
        return history[0]

    F0 = history[0]
    x = float(step_offset)
    out = F0.clone()
    for k in range(1, order + 1):
        coef = (x**k) / math.factorial(k)
        out.add_(history[k], alpha=coef)
    return out


def taylor_scaled_predict(
    history: Dict[int, torch.Tensor],
    step_offset: int,
    sigma: float,
    max_order: int,
) -> torch.Tensor:
    """Hi-Taylor: Taylor predictor with HiCache's dual-scaling on a monomial basis.

    On monomials, `tilde_H_n(x) = sigma^n * H_n(sigma*x)` collapses to a single
    sigma^n factor since (sigma*x)^n = sigma^n * x^n. Equivalent to evaluating
    Taylor at x' = sigma * x. Provided for the HiCache paper's Sec. 4.3 ablation.
    """
    order_avail = len(history) - 1
    order = min(max_order, order_avail)
    if order < 1:
        return history[0]
    F0 = history[0]
    x = float(step_offset)
    out = F0.clone()
    for k in range(1, order + 1):
        coef = (x**k) / math.factorial(k) * (sigma**k)
        out.add_(history[k], alpha=coef)
    return out
