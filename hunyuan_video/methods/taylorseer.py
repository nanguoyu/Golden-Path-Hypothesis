from __future__ import annotations

from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from lib.taylor import taylor_predict, taylor_update


class TaylorSeerMethod:
    name = "taylorseer"

    def __init__(self, *, action: Any, max_order: int = 1, first_enhance: int = 3):
        if max_order != 1:
            raise ValueError("Stage B Core source lane freezes TaylorSeer at O1")
        if int(first_enhance) < 1:
            raise ValueError("TaylorSeer first_enhance must be positive")
        self.action = action
        self.max_order = max_order
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

        The official TaylorSeer-HunyuanVideo builds differences only when
        `step > first_enhance - 2`, and the FLUX scaffold's warmup guard
        (`lib/flux_fine_scaffold.py:237`) is the same rule under its
        post-incremented counter; see HiCacheMethod.effective_max_order.
        With first_enhance=3 both conventions leave an order-1 history at the
        first post-warmup anchor, which for O1 happens to be indistinguishable
        from the unclamped update -- but only for schedules that keep steps 0-2
        full, and nothing in this class should depend on the schedule.
        """
        return 0 if self.step < self.first_enhance else self.max_order

    def update_slot(self, slot: str, feature: torch.Tensor) -> None:
        previous = self.histories.get(slot)
        self.histories[slot] = taylor_update(
            previous,
            feature.detach(),
            self.current_gap,
            self.effective_max_order,
        )

    def predict_slot(self, slot: str) -> torch.Tensor:
        if slot not in self.histories:
            raise RuntimeError(f"TaylorSeer slot has no full history: {slot}")
        return taylor_predict(self.histories[slot], self.step_offset, self.max_order)

    def finish_full_step(self) -> None:
        self.last_full_step = self.current_step
