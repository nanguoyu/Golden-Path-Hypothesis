"""TaylorSeer (fine granularity) on diffusers FLUX.

Per-(block_idx, stream, sub_module) Taylor extrapolation. Same 114-slot
structure as `flux/hicache_fine.py`, only the predictor differs (Taylor
monomial basis instead of Hermite).

This is the paper-faithful re-implementation of Shenyi-Z/TaylorSeer's flux
backbone, adapted to diffusers 0.38.0 via the shared scaffold in
`lib/flux_fine_scaffold.py`.

For the coarse variant — same Taylor math but single cache slot — see
`flux/taylorseer.py`. The coarse variant is faster but less paper-faithful;
this fine variant is closer to paper Tab 1 numbers.

## Operating points

SeaCache paper convention for FLUX uses TaylorSeer with O=1 (expansion order).
TaylorSeer's own paper uses O=2 for FLUX (per `cache_init.py`). We default to
O=1 to match SeaCache paper Tab 1 reproduction; pass `--max_order 2` to use
TaylorSeer's own default.
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
from lib.taylor import taylor_predict, taylor_update


class _TaylorFinePredictor(FineCachePredictor):
    """Adapter to the FineCachePredictor protocol for TaylorSeer."""

    def __init__(self, max_order: int):
        self.max_order = int(max_order)

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        return taylor_update(prev_history, feature, step_gap, effective_max_order)

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        return taylor_predict(history, step_offset, self.max_order)


def install(
    pipe,
    *,
    interval: int = 7,
    max_order: int = 1,
    first_enhance: int = 3,
    num_steps: int,
) -> Callable[[], None]:
    """Patch FLUX's transformer + both block classes' forwards with fine TaylorSeer.

    Args:
        pipe:          loaded `DiffusionPipeline` / `FluxPipeline`.
        interval:      refresh every N steps after warmup (paper / SeaCache convention: 3 or 5).
        max_order:     Taylor truncation order O (SeaCache paper FLUX convention: 1;
                       TaylorSeer's own paper FLUX convention: 2).
        first_enhance: count of leading full-forward steps (TaylorSeer official: 3).
        num_steps:     total sampling steps; the last step is force-full.

    Returns:
        teardown: zero-arg callable restoring all 3 forwards and clearing state.
    """
    predictor = _TaylorFinePredictor(max_order=max_order)
    return install_fine_cache(
        pipe,
        predictor=predictor,
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
        method_tag="taylorseer_fine",
    )


def reset_per_image_state(pipe) -> None:
    """Reset TaylorSeer fine trajectory state before each new prompt."""
    reset_per_image_state_fine(pipe)
