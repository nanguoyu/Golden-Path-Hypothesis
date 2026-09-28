from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from hunyuan_video.actions import MethodDecision


class MeanCacheMethod:
    """MeanCache average-velocity payload on a fixed HunyuanVideo schedule.

    On a cached step k the substituted velocity comes from a JVP proxy that is
    estimated on the *current* trajectory (`flux/meancache_exact.py:111-137`):

        vbar   = (z_k - z_r) / (sigma_k - sigma_r)
        jhat   = (vbar - v_r) / (sigma_k - sigma_r)
        vhat   = v_latest + (sigma_{k+1} - sigma_k) * jhat

    `r` is a sliding lookback `k - span`, not the start of the cached interval,
    and `span` is chosen per edge by the offline search, so it is read from
    `jvp_spans` per step and only falls back to `jvp_span`.
    """

    name = "meancache"

    def __init__(
        self,
        *,
        action: Any,
        num_steps: int,
        scheduler_provider: Callable[[], Any],
        jvp_span: int = 4,
        jvp_spans: dict[int, int] | None = None,
    ) -> None:
        self.action = action
        self.num_steps = int(num_steps)
        self.scheduler_provider = scheduler_provider
        self.jvp_span = int(jvp_span)
        self.jvp_spans = {int(step): int(span) for step, span in (jvp_spans or {}).items()}
        if self.jvp_span < 1 or any(span < 1 for span in self.jvp_spans.values()):
            raise ValueError("MeanCache JVP spans must be positive")
        self.reset()

    def reset(self) -> None:
        self.action.reset()
        self.step = 0
        self.current_step = -1
        self.current_full = True
        self.current_latent: torch.Tensor | None = None
        self.current_sigma: torch.Tensor | None = None
        self.current_next_sigma: torch.Tensor | None = None
        self.latents: list[torch.Tensor] = []
        self.sigmas: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.last_payload_fields: dict[str, Any] = {}

    def span_for(self, step: int) -> int:
        return self.jvp_spans.get(int(step), self.jvp_span)

    def _scheduler_sigmas(self, device: torch.device) -> torch.Tensor:
        scheduler = self.scheduler_provider()
        sigmas = getattr(scheduler, "sigmas", None)
        # Exactly the inference schedule, not merely long enough: `predict()` builds
        # a fresh `FlowMatchDiscreteScheduler` per call
        # (`hyvideo/inference.py:611-616`) whose `__init__` leaves `sigmas` at the
        # 1001-entry training grid until `set_timesteps` runs
        # (`scheduling_flow_match_discrete.py:78-84`). Accepting that grid would
        # silently hand out sigma_t=1.0 and a 20x-too-small solver step.
        if sigmas is None or len(sigmas) != self.num_steps + 1:
            raise RuntimeError(
                f"MeanCache requires the {self.num_steps}-step inference sigmas, "
                f"got {'none' if sigmas is None else len(sigmas)}"
            )
        return torch.as_tensor(sigmas, device=device, dtype=torch.float32)

    def decide(self, *, latent: torch.Tensor, **_kwargs: Any) -> MethodDecision:
        full, reason = self.action.decide_full()
        self.current_step = self.step
        self.step += 1
        if not full and not self.velocities:
            # Downgrading to a full step here would quietly spend K-1 of the K
            # cached steps the fixed schedule promises, so it is an error, as on
            # FLUX (`flux/meancache_exact.py:152-153`).
            raise RuntimeError("MeanCache reached a cache step before its first full velocity")
        sigmas = self._scheduler_sigmas(latent.device)
        self.current_sigma = sigmas[self.current_step].detach()
        self.current_next_sigma = sigmas[self.current_step + 1].detach()
        self.current_latent = latent.detach()
        self.current_full = bool(full)
        self.last_payload_fields = {}
        return MethodDecision(full=bool(full), reason=str(reason))

    def _predict(self) -> tuple[torch.Tensor, int, bool]:
        available = len(self.velocities)
        actual = min(self.span_for(self.current_step), available)
        latest = self.velocities[-1]
        if actual <= 1:
            return latest.detach().clone(), actual, False

        reference = available - actual
        sigma_t = self.current_sigma.to(torch.float32)
        sigma_s = self.current_next_sigma.to(torch.float32)
        sigma_r = self.sigmas[reference].to(torch.float32)
        interval = sigma_t - sigma_r
        if float(interval.abs().item()) < 1e-12:
            return latest.detach().clone(), actual, False

        z_r = self.latents[reference].to(torch.float32)
        z_t = self.current_latent.to(torch.float32)
        v_r = self.velocities[reference].to(torch.float32)
        average_velocity = (z_t - z_r) / interval
        jvp = (average_velocity - v_r) / interval
        predicted = latest.to(torch.float32) + (sigma_s - sigma_t) * jvp
        return predicted.to(dtype=latest.dtype), actual, True

    def final_output(self, output: torch.Tensor | None = None) -> torch.Tensor:
        if self.current_step < 0 or self.current_latent is None:
            raise RuntimeError("MeanCache velocity requested before a step decision")
        requested = self.span_for(self.current_step)
        if self.current_full:
            if output is None:
                raise RuntimeError("full MeanCache step requires the real final-layer output")
            velocity = output
            actual_span, jvp_used = 0, False
        else:
            velocity, actual_span, jvp_used = self._predict()
        self.last_payload_fields = {
            "sigma_t": float(self.current_sigma.item()),
            "sigma_s": float(self.current_next_sigma.item()),
            "requested_jvp_span": int(requested),
            "actual_jvp_span": int(actual_span),
            "jvp_correction_used": bool(jvp_used),
        }
        self.latents.append(self.current_latent)
        self.sigmas.append(self.current_sigma)
        self.velocities.append(velocity.detach())
        return velocity
