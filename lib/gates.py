"""Gating primitives for cache-acceleration methods.

A "gate" decides at each diffusion step whether to run the full transformer
forward or substitute a cached/predicted output. Two gate families are common:

  * **Interval gate**: skip every step except the first `first_enhance` steps
    and every Nth step thereafter. Used by HiCache, TaylorSeer.
  * **Threshold gate**: accumulate a rescaled rel_L1 distance between modulated
    inputs and skip when the sum stays under a threshold. This helper is only a
    generic primitive; the paper-faithful FLUX SeaCache/TeaCache forwards in
    `flux/seacache.py` and `flux/teacache.py` update their previous modulated
    input every step, so exact method-native experiments should follow those
    files rather than assuming this helper's state semantics.

Both are tiny state machines independent of the backbone, so they live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


def rel_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> float:
    """Scalar mean-abs relative L1: mean(|a - b|) / (mean(|b|) + eps)."""
    num = (a - b).abs().mean()
    den = b.abs().mean() + eps
    return float((num / den).detach().cpu())


# ----- Interval-style gate ---------------------------------------------------


@dataclass
class IntervalGate:
    """Stateful interval gate: full-forward for the first `first_enhance` steps,
    then every `interval` steps; skip in between.

    If `num_steps` is set, the LAST step of the trajectory (`cnt == num_steps - 1`)
    is also forced full. This mirrors the BFL `cal_type` rule
    `step >= num_steps - 1 -> full` and is important: skipping the final denoising
    step degrades image quality noticeably.

    The gate tracks the global step counter `cnt` and the index of the most
    recent full-forward step in `last_activated`. Callers can read
    `step_offset = (cnt - 1) - last_activated` immediately after `decide()`.

    Invariants after `decide()` returns:
      - If `should_skip` is False, `last_activated == cnt - 1`.
      - On the very first call, `should_skip` is False.
      - If `num_steps` is set, the call where `cnt == num_steps - 1` is False.
    """
    interval: int = 7
    first_enhance: int = 3
    num_steps: Optional[int] = None
    cnt: int = 0
    last_activated: int = -10**9
    activated_steps: List[int] = field(default_factory=list)

    def reset(self) -> None:
        self.cnt = 0
        self.last_activated = -(10**9)
        self.activated_steps.clear()

    def decide(self) -> bool:
        """Advance the counter by 1; return `should_skip` for the just-entered step."""
        if self.cnt < self.first_enhance:
            should_skip = False
        elif self.num_steps is not None and self.cnt >= self.num_steps - 1:
            should_skip = False
        elif (self.cnt - self.last_activated) >= self.interval:
            should_skip = False
        else:
            should_skip = True
        if not should_skip:
            self.last_activated = self.cnt
            self.activated_steps.append(self.cnt)
        self.cnt += 1
        return should_skip

    @property
    def step_offset(self) -> int:
        """Distance from the just-entered step to the last activation step.

        Useful immediately after `decide()` returns: `cnt-1` is the step that
        was just decided on. Returns 0 right after a full-forward step.
        """
        return (self.cnt - 1) - self.last_activated


# ----- Threshold-style gate --------------------------------------------------


@dataclass
class ThresholdGate:
    """Stateful accumulated-rel_L1 threshold gate (SeaCache/TeaCache-style).

    `feed(x)` is called with the current step's modulated/input tensor `x`. The
    gate compares `x` against the cached `x_prev` via rel_L1, accumulates the
    distance with an optional rescaling, and reports `should_skip = True` when
    the running sum stays below `threshold`. The internal counter also implements
    an early-warmup region: the first `first_enhance` steps are always full.

    If `num_steps` is set, the LAST step (`cnt == num_steps - 1`) is also forced
    full. This matches the SeaCache and TeaCache official FLUX code, both of
    which gate on `cnt == 0 or cnt == num_steps - 1`. Skipping the final
    denoising step degrades image quality noticeably, hence the safeguard.

    Default rescaling is identity (`rescale=None`); SeaCache uses a polynomial
    rescaling like `f(d) = c0 + c1*d + c2*d^2 + ...`. Pass a callable that takes
    a float and returns a float.
    """
    threshold: float = 0.3
    first_enhance: int = 1
    num_steps: Optional[int] = None
    rescale: Optional[Callable[[float], float]] = None
    cnt: int = 0
    accumulated: float = 0.0
    x_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.cnt = 0
        self.accumulated = 0.0
        self.x_prev = None

    def feed(self, x: torch.Tensor) -> bool:
        """Advance one step. Returns `should_skip`.

        Updates `x_prev` only on full-forward steps. This is a generic threshold
        state machine, not the exact state update used by the current FLUX
        SeaCache/TeaCache forwards, which refresh their previous modulated input
        every step while separately resetting the accumulator on full steps.
        """
        if self.cnt < self.first_enhance or self.x_prev is None:
            should_skip = False
        elif self.num_steps is not None and self.cnt >= self.num_steps - 1:
            should_skip = False
        else:
            d = rel_l1(x, self.x_prev)
            d = self.rescale(d) if self.rescale is not None else d
            self.accumulated += d
            if self.accumulated < self.threshold:
                should_skip = True
            else:
                should_skip = False
                self.accumulated = 0.0
        if not should_skip:
            self.x_prev = x.detach().clone()
        self.cnt += 1
        return should_skip
