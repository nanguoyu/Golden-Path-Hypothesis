from __future__ import annotations

from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from lib.hermite import hermite_update, hicache_predict


class HiCacheMethod:
    """Fine-grained HiCache predictor for Tencent HunyuanVideo blocks."""

    name = "hicache"

    def __init__(self, *, action: Any, max_order: int = 2, sigma: float = 0.5,
                 first_enhance: int = 3):
        if int(max_order) < 1:
            raise ValueError("HiCache max_order must be positive")
        if not 0.0 < float(sigma) <= 1.0:
            raise ValueError("HiCache sigma must be in (0, 1]")
        if int(first_enhance) < 1:
            raise ValueError("HiCache first_enhance must be positive")
        self.action = action
        self.max_order = int(max_order)
        self.sigma = float(sigma)
        self.first_enhance = int(first_enhance)
        self.reset()

    def reset(self) -> None:
        self.action.reset()
        self.step = 0
        self.last_full_step: int | None = None
        self.current_gap = 0
        self.histories: dict[str, dict[int, torch.Tensor]] = {}

    def decide(self, **_kwargs: Any) -> MethodDecision:
        full, reason = self.action.decide_full()
        if full:
            self.current_gap = 0 if self.last_full_step is None else self.step - self.last_full_step
        decision = MethodDecision(full=full, reason=reason)
        self.step += 1
        return decision

    @property
    def current_step(self) -> int:
        return self.step - 1

    @property
    def step_offset(self) -> int:
        if self.last_full_step is None:
            return 0
        return self.current_step - self.last_full_step

    @property
    def effective_max_order(self) -> int:
        """0 during the warmup steps, `max_order` afterwards.

        Every authority applies this clamp -- the official HiCache
        (`derivative_approximation` builds Delta^k for k>=1 only when
        `step > first_enhance - 2`), the official TaylorSeer-HunyuanVideo, and
        both FLUX ports (`flux/hicache.py:219`, `lib/flux_fine_scaffold.py:237`,
        whose comment says exactly why: without it the predictor has a richer
        history than upstream at the first skip step and silently produces
        different numbers). `decide()` post-increments `self.step`, so during
        `update_slot` it equals current_step + 1, which is the FLUX scaffold's
        `gate.cnt` convention: with first_enhance=3, steps 0-1 store only F_0
        and step 2 starts building differences, so the first cached step is
        anchored on an order-1 history, not an order-2 one no authority builds.
        """
        return 0 if self.step < self.first_enhance else self.max_order

    def update_slot(self, slot: str, feature: torch.Tensor) -> None:
        self.histories[slot] = hermite_update(
            self.histories.get(slot),
            feature.detach(),
            self.current_gap,
            self.effective_max_order,
        )

    def predict_slot(self, slot: str) -> torch.Tensor:
        if slot not in self.histories:
            raise RuntimeError(f"HiCache slot has no full history: {slot}")
        return hicache_predict(
            self.histories[slot],
            self.step_offset,
            self.sigma,
            self.max_order,
        )

    def finish_full_step(self) -> None:
        self.last_full_step = self.current_step
