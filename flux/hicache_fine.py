"""HiCache (fine granularity) on diffusers FLUX.

Per-(block_idx, stream, sub_module) Hermite extrapolation. 114 cache slots per
FLUX trajectory: 19 dual-stream blocks × {img_attn, img_mlp, txt_attn, txt_mlp}
+ 38 single-stream blocks × {combined}.

This is the paper-faithful re-implementation of fenglang918/HiCache's flux
backbone, adapted from the BFL fork to diffusers 0.38.0 via the shared
scaffold in `lib/flux_fine_scaffold.py`.

For the coarse (whole-transformer-residual) variant — same Hermite math but
single cache slot — see `flux/hicache.py`. The coarse variant is faster but
less paper-faithful; this fine variant is closer to paper Tab 1 numbers.

## Operating points

The same `(interval, max_order, sigma, first_enhance)` tuple as coarse, but
the resulting speedup tier differs at the same N because fine has more
per-block framework overhead. Refer to `docs/experiments.md` §5 for the
matched-tier table.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from lib.flux_fine_scaffold import (
    CacheHistory,
    FineCachePredictor,
    install_fine_cache,
    reset_per_image_state_fine,
)
from lib.hermite import hermite_update, hicache_predict


class _HermiteFinePredictor(FineCachePredictor):
    """Adapter to the FineCachePredictor protocol for HiCache."""

    def __init__(self, max_order: int, sigma: float):
        self.max_order = int(max_order)
        self.sigma = float(sigma)

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        return hermite_update(prev_history, feature, step_gap, effective_max_order)

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        return hicache_predict(history, step_offset, self.sigma, self.max_order)


def install(
    pipe,
    *,
    interval: int = 7,
    max_order: int = 2,
    sigma: float = 0.5,
    first_enhance: int = 3,
    num_steps: int,
) -> Callable[[], None]:
    """Patch FLUX's transformer + both block classes' forwards with fine HiCache.

    Args:
        pipe:          loaded `DiffusionPipeline` / `FluxPipeline`.
        interval:      refresh every N steps after warmup (paper Table 1: 7).
        max_order:     Hermite truncation order O (paper Table 1: 2).
        sigma:         dual-scaling factor in `H_k(sigma*x)` (paper: 0.5).
        first_enhance: count of leading full-forward steps (paper: 3 = O+1, math min).
        num_steps:     total sampling steps; the last step is force-full.

    Returns:
        teardown: zero-arg callable restoring all 3 forwards and clearing state.
    """
    predictor = _HermiteFinePredictor(max_order=max_order, sigma=sigma)
    return install_fine_cache(
        pipe,
        predictor=predictor,
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
        method_tag="hicache_fine",
    )


def reset_per_image_state(pipe) -> None:
    """Reset HiCache fine trajectory state before each new prompt."""
    reset_per_image_state_fine(pipe)
