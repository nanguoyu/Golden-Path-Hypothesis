from __future__ import annotations

from typing import Any

import torch

from hunyuan_video.actions import MethodDecision
from lib.sencache import (
    load_sensitivity_table,
    online_fields,
    threshold_scale_from_latent,
)


def _scalar_timestep(timestep: Any) -> float:
    if isinstance(timestep, torch.Tensor):
        return float(timestep.detach().to(torch.float32).reshape(-1)[0].item())
    return float(timestep)


class SenCacheMethod:
    """Sensitivity-gated whole-residual reuse, ported from ``flux/sencache.py``.

    The score is measured against the anchor step ``a`` — the last step whose
    blocks actually ran:

        sigma_k = J_x(t_a) * ||z_k - z_a||_2 + J_t(t_a) * |t_k - t_a|

    ``J_x`` / ``J_t`` come from a frozen HunyuanVideo sensitivity table, looked
    up by nearest anchor-timestep *value* (``lib/sencache.py:48``).  ``z`` is the
    latent before ``img_in`` and ``t`` the transformer timestep in its native
    range(0, 1000) units, neither of which reaches double block 0, so the
    adapter hands both to ``observe_input`` before ``decide()`` runs.

    Cache when ``sigma_k < delta_k``; the payload is the whole-transformer
    residual the coarse adapter already keeps.
    """

    name = "sencache"

    def __init__(
        self,
        *,
        num_steps: int,
        sensitivity_path: str,
        threshold_start: float,
        threshold_main: float,
        first_enhance: int = 3,
        max_skip: int = 10,
        threshold_scale: str | float | int | None = "auto",
        switch_ratio: float = 0.2,
        ret_steps: int = 0,
        cutoff_steps: int = -1,
    ):
        self.num_steps = int(num_steps)
        self.table = load_sensitivity_table(sensitivity_path)
        self.threshold_start = float(threshold_start)
        self.threshold_main = float(threshold_main)
        self.first_enhance = int(first_enhance)
        self.max_skip = int(max_skip)
        self.switch_ratio = float(switch_ratio)
        self.ret_steps = int(ret_steps)
        self.cutoff_steps = int(cutoff_steps)
        self.threshold_scale_arg = threshold_scale
        # sqrt(d) multiplies the raw threshold rather than dividing the score,
        # is measured once from the first observed latent, and survives reset()
        # (flux/sencache.py:128-145).
        self.threshold_scale: float | None = None
        self.reset()

    @property
    def cutoff_step(self) -> int:
        return (self.num_steps - 1) if self.cutoff_steps < 0 else self.cutoff_steps

    @property
    def switch_step(self) -> int:
        return int(round(self.num_steps * self.switch_ratio))

    def reset(self) -> None:
        self.step = 0
        self.consecutive_skips = 0
        self.anchor_latent: torch.Tensor | None = None
        self.anchor_timestep: float | None = None
        self.anchor_step: int | None = None
        self.latent: torch.Tensor | None = None
        self.timestep: float | None = None

    def observe_input(self, latent: torch.Tensor, timestep: Any) -> None:
        """Take this step's gate inputs from the transformer pre-hook."""

        self.latent = latent.detach()
        self.timestep = _scalar_timestep(timestep)

    def _forced_full_reason(self) -> str | None:
        if self.step < self.first_enhance or self.step == 0:
            return "warmup"
        if self.step >= max(min(self.cutoff_step, self.num_steps), 0):
            return "cutoff"
        if self.anchor_latent is None or self.anchor_timestep is None:
            return "no_anchor"
        return None

    def decide(self, **_kwargs: Any) -> MethodDecision:
        if self.latent is None or self.timestep is None:
            raise RuntimeError("SenCache gate did not receive the transformer input")
        if self.threshold_scale is None:
            self.threshold_scale = threshold_scale_from_latent(
                self.latent, self.threshold_scale_arg
            )
        threshold_raw = (
            self.threshold_start if self.step < self.switch_step else self.threshold_main
        )
        threshold = float(threshold_raw * self.threshold_scale)

        # Scored before the forced-full branch, as flux/sencache.py:132 does, so
        # warmup and cutoff steps still report what the gate would have seen.
        # Only step 0 has no anchor to score against.
        score: float | None = online_fields(
            table=self.table,
            current_latent=self.latent,
            current_timestep=self.timestep,
            anchor_latent=self.anchor_latent,
            anchor_timestep=self.anchor_timestep,
            anchor_step=self.anchor_step,
        )["online_sencache_score_pre"]

        reason = self._forced_full_reason()
        if reason is not None:
            full = True
        else:
            cache = (
                self.step >= self.ret_steps
                and score is not None
                and float(score) < threshold
                and self.consecutive_skips < self.max_skip
            )
            full = not cache
            reason = "threshold_full" if full else "threshold_cache"

        if full:
            self.anchor_latent = self.latent.clone()
            self.anchor_timestep = self.timestep
            self.anchor_step = self.step
            self.consecutive_skips = 0
        else:
            self.consecutive_skips += 1
        self.step += 1
        return MethodDecision(
            full=full,
            reason=reason,
            gate_scalar=score,
            accumulated=float(self.consecutive_skips),
            threshold=threshold,
        )
