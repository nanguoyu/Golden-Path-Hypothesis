"""MeanCache average-velocity payload on a fixed FLUX schedule."""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from lib.fixed_schedule import validate_cache_steps


def _set_forward(module: Any, function: Callable[..., Any]) -> tuple[bool, Any]:
    had_instance = "forward" in module.__dict__
    original = module.forward
    module.forward = types.MethodType(function, module)
    return had_instance, original


def _restore_forward(module: Any, state: tuple[bool, Any]) -> None:
    had_instance, original = state
    if had_instance:
        module.forward = original
    else:
        delattr(module, "forward")


def _sample(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if sample is None:
        raise TypeError(f"unsupported FLUX transformer output: {type(output).__name__}")
    return sample


def _replace_sample(template: Any, sample: torch.Tensor) -> Any:
    if isinstance(template, tuple):
        return (sample, *template[1:])
    if isinstance(template, torch.Tensor):
        return sample
    return type(template)(sample=sample)


class FluxMeanCacheAdapter:
    """Skip selected model evaluations and use MeanCache's cached-JVP velocity."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        jvp_span: int = 4,
        jvp_spans: dict[int, int] | None = None,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self.jvp_span = int(jvp_span)
        self.jvp_spans = {
            int(step): int(span) for step, span in (jvp_spans or {}).items()
        }
        if self.jvp_span < 1 or any(span < 1 for span in self.jvp_spans.values()):
            raise ValueError("MeanCache JVP spans must be positive")
        self._patch: tuple[bool, Any] | None = None
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.start_latents: list[torch.Tensor] = []
        self.start_sigmas: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.output_template: Any = None
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX MeanCache adapter already installed")
        original = self.transformer.forward

        def wrapped(module: Any, *args: Any, **kwargs: Any):
            return self._forward(original, *args, **kwargs)

        self._patch = _set_forward(self.transformer, wrapped)
        self._installed = True

    def restore(self) -> None:
        if self._patch is not None:
            _restore_forward(self.transformer, self._patch)
        self._patch = None
        self._installed = False

    def _sigmas(self, device: torch.device) -> torch.Tensor:
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        if sigmas is None or len(sigmas) < self.num_steps + 1:
            raise RuntimeError("MeanCache requires scheduler.sigmas for every FLUX step")
        return torch.as_tensor(sigmas, device=device, dtype=torch.float32)

    def _predict(
        self,
        current_latent: torch.Tensor,
        *,
        sigma_t: torch.Tensor,
        sigma_s: torch.Tensor,
        span: int,
    ) -> tuple[torch.Tensor, int, bool]:
        available = len(self.velocities)
        actual = min(int(span), available)
        latest = self.velocities[-1]
        if actual <= 1:
            return latest.detach().clone(), actual, False

        reference = available - actual
        z_r = self.start_latents[reference].to(torch.float32)
        sigma_r = self.start_sigmas[reference].to(torch.float32)
        z_t = current_latent.to(torch.float32)
        dt_rt = sigma_t.to(torch.float32) - sigma_r
        if float(dt_rt.abs().item()) < 1e-12:
            return latest.detach().clone(), actual, False

        average_velocity = (z_t - z_r) / dt_rt
        v_r = self.velocities[reference].to(torch.float32)
        jvp = (average_velocity - v_r) / dt_rt
        predicted = latest.to(torch.float32) + (sigma_s - sigma_t).to(torch.float32) * jvp
        return predicted.to(dtype=latest.dtype), actual, True

    def _forward(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        latent = kwargs.get("hidden_states", args[0] if args else None)
        if latent is None:
            raise RuntimeError("MeanCache could not find FLUX latent input")
        sigmas = self._sigmas(latent.device)
        sigma_t = sigmas[self.step]
        sigma_s = sigmas[self.step + 1]
        cache = self.step in self._cache_steps

        span = self.jvp_spans.get(self.step, self.jvp_span)
        jvp_used = False
        actual_span = 0
        if cache:
            if not self.velocities or self.output_template is None:
                raise RuntimeError("MeanCache reached a cache step before its first full velocity")
            velocity, actual_span, jvp_used = self._predict(
                latent,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                span=span,
            )
            output = _replace_sample(self.output_template, velocity)
        else:
            output = original(*args, **kwargs)
            velocity = _sample(output)
            if self.output_template is None:
                self.output_template = output

        self.start_latents.append(latent.detach())
        self.start_sigmas.append(sigma_t.detach())
        self.velocities.append(velocity.detach())
        self.records.append(
            {
                "step": int(self.step),
                "action": "cache" if cache else "full",
                "u": int(cache),
                "sigma_t": float(sigma_t.item()),
                "sigma_s": float(sigma_s.item()),
                "requested_jvp_span": int(span),
                "actual_jvp_span": int(actual_span),
                "jvp_correction_used": bool(jvp_used),
            }
        )
        self.step += 1
        return output

    def decisions(self) -> dict[str, Any]:
        cached = sum(row["u"] for row in self.records)
        if len(self.records) != self.num_steps or cached != len(self.cache_steps):
            raise RuntimeError(
                f"MeanCache recorded {len(self.records)} steps and K={cached}; "
                f"expected {self.num_steps} and K={len(self.cache_steps)}"
            )
        return {
            "schema": "flux_meancache_exact_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "meancache_exact",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "default_jvp_span": self.jvp_span,
            "jvp_spans": dict(self.jvp_spans),
            "per_step": list(self.records),
            "summary": {
                "n_total": len(self.records),
                "n_full": len(self.records) - cached,
                "n_cached": cached,
                "cache_ratio": cached / len(self.records),
                "jvp_corrected_steps": sum(
                    int(row["jvp_correction_used"]) for row in self.records
                ),
            },
        }
