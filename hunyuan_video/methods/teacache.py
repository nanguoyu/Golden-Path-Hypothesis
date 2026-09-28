from __future__ import annotations

from typing import Any, Sequence

import torch

from hunyuan_video.actions import MethodDecision
from lib.gates import rel_l1


HUNYUAN_VIDEO_COEFFICIENTS = (
    7.33226126e02,
    -4.01131952e02,
    6.75869174e01,
    -3.14987800e00,
    9.61237896e-02,
)


def polynomial(value: float, coefficients: Sequence[float]) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def modulated_input(img: torch.Tensor, vec: torch.Tensor, first_block: Any) -> torch.Tensor:
    shift, scale, *_ = first_block.img_mod(vec).chunk(6, dim=-1)
    normed = first_block.img_norm1(img)
    return normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TeaCacheMethod:
    name = "teacache"

    def __init__(
        self,
        *,
        num_steps: int,
        threshold: float,
        coefficients: Sequence[float] = HUNYUAN_VIDEO_COEFFICIENTS,
        first_enhance: int = 1,
    ):
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.coefficients = tuple(float(value) for value in coefficients)
        self.first_enhance = int(first_enhance)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.accumulated = 0.0
        self.previous_modulated_input: torch.Tensor | None = None

    def decide(self, *, img: torch.Tensor, vec: torch.Tensor, first_block: Any, **_kwargs: Any) -> MethodDecision:
        current = modulated_input(img, vec, first_block)
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
            raw = rel_l1(current, self.previous_modulated_input)
            scalar = polynomial(raw, self.coefficients)
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
