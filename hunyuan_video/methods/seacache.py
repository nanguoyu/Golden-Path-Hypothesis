from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from hunyuan_video.methods.teacache import modulated_input
from lib.gates import rel_l1
from lib.wiener import apply_sea_with_scheduler


class SeaCacheMethod:
    name = "seacache"

    def __init__(
        self,
        *,
        num_steps: int,
        threshold: float,
        scheduler_provider: Callable[[], Any],
        first_enhance: int = 1,
        power_exp: float = 3.0,
    ):
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.scheduler_provider = scheduler_provider
        self.first_enhance = int(first_enhance)
        self.power_exp = float(power_exp)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.accumulated = 0.0
        self.previous_modulated_input: torch.Tensor | None = None

    def decide(
        self,
        *,
        img: torch.Tensor,
        vec: torch.Tensor,
        first_block: Any,
        grid_shape: tuple[int, int, int],
        **_kwargs: Any,
    ) -> MethodDecision:
        current = modulated_input(img, vec, first_block)
        tt, th, tw = grid_shape
        if current.shape[1] != tt * th * tw:
            raise ValueError(f"SEA grid {grid_shape} does not match {current.shape[1]} tokens")
        current = current.reshape(current.shape[0], tt, th, tw, current.shape[-1])
        current = apply_sea_with_scheduler(
            current,
            self.scheduler_provider(),
            self.step,
            power_exp=self.power_exp,
            dims=(-2, -3, -4),
            norm_mode="mean",
        ).reshape(current.shape[0], -1, current.shape[-1])
        force = (
            self.step < self.first_enhance
            or self.step == self.num_steps - 1
            or self.previous_modulated_input is None
        )
        scalar: float | None = None
        if force:
            full = True
            reason = "forced_boundary"
            self.accumulated = 0.0
        else:
            scalar = rel_l1(current, self.previous_modulated_input)
            self.accumulated += scalar
            full = self.accumulated >= self.threshold
            reason = "threshold_full" if full else "threshold_cache"
            if full:
                self.accumulated = 0.0
        self.previous_modulated_input = current.detach()
        self.step += 1
        return MethodDecision(
            full=full,
            reason=reason,
            gate_scalar=scalar,
            accumulated=self.accumulated,
            threshold=self.threshold,
        )
