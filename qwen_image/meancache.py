"""Fixed-schedule MeanCache adapter for diffusers Qwen-Image."""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from lib.fixed_schedule import validate_cache_steps


BRANCHES = ("cond", "uncond")


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
        raise TypeError(f"unsupported Qwen transformer output: {type(output).__name__}")
    return sample


def _replace_sample(template: Any, sample: torch.Tensor) -> Any:
    if isinstance(template, tuple):
        return (sample, *template[1:])
    if isinstance(template, torch.Tensor):
        return sample
    return type(template)(sample=sample)


def guided_velocity(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    *,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Match Qwen-Image's post-true-CFG norm-rescaled velocity."""

    combined = uncond + float(true_cfg_scale) * (cond - uncond)
    cond_norm = torch.linalg.vector_norm(cond, dim=-1, keepdim=True)
    combined_norm = torch.linalg.vector_norm(combined, dim=-1, keepdim=True)
    eps = torch.finfo(torch.float32).eps
    return combined * (cond_norm / combined_norm.clamp_min(eps))


class QwenMeanCacheAdapter:
    """Predict the guided velocity produced after Qwen's true-CFG rescaling."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        jvp_span: int = 4,
        jvp_spans: dict[int, int] | None = None,
        true_cfg: bool = True,
        true_cfg_scale: float = 4.0,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.true_cfg = bool(true_cfg)
        self.true_cfg_scale = float(true_cfg_scale)
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
            raise ValueError("Qwen MeanCache JVP spans must be positive")
        self._patch: tuple[bool, Any] | None = None
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.start_latents: list[torch.Tensor] = []
        self.start_sigmas: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.output_templates: dict[str, Any] = {branch: None for branch in BRANCHES}
        self.pending_cond: torch.Tensor | None = None
        self.pending_prediction: torch.Tensor | None = None
        self.pending_actual_span = 0
        self.pending_jvp_used = False
        self.pending_latent: torch.Tensor | None = None
        self.pending_sigma: torch.Tensor | None = None
        self.steps: dict[int, dict[str, Any]] = {}

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Qwen MeanCache adapter already installed")
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
            raise RuntimeError(
                "Qwen MeanCache requires scheduler.sigmas for every step"
            )
        return torch.as_tensor(sigmas, device=device, dtype=torch.float32)

    @staticmethod
    def _latent(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        latent = kwargs.get("hidden_states", args[0] if args else None)
        if latent is None:
            raise RuntimeError("Qwen MeanCache could not find transformer input")
        return latent

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
        delta_sigma = sigma_t.to(torch.float32) - sigma_r
        if float(delta_sigma.abs().item()) < 1e-12:
            return latest.detach().clone(), actual, False

        average_velocity = (z_t - z_r) / delta_sigma
        v_r = self.velocities[reference].to(torch.float32)
        jvp = (average_velocity - v_r) / delta_sigma
        predicted = (
            latest.to(torch.float32) + (sigma_s - sigma_t).to(torch.float32) * jvp
        )
        return predicted.to(dtype=latest.dtype), actual, True

    def _guided_velocity(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        if not self.true_cfg:
            return cond
        return guided_velocity(
            cond,
            uncond,
            true_cfg_scale=self.true_cfg_scale,
        )

    def _append_history(
        self,
        latent: torch.Tensor,
        sigma_t: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        self.start_latents.append(latent.detach())
        self.start_sigmas.append(sigma_t.detach())
        self.velocities.append(velocity.detach())

    def _forward(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        branches_per_step = 2 if self.true_cfg else 1
        call = int(self.forward_call_count)
        step = min(call // branches_per_step, self.num_steps - 1)
        branch = "cond" if branches_per_step == 1 or call % 2 == 0 else "uncond"
        latent = self._latent(args, kwargs)
        sigmas = self._sigmas(latent.device)
        sigma_t = sigmas[step]
        sigma_s = sigmas[step + 1]
        cache = step in self._cache_steps
        span = self.jvp_spans.get(step, self.jvp_span)

        if cache:
            if not self.velocities or self.output_templates[branch] is None:
                raise RuntimeError("Qwen MeanCache reached cache before a full output")
            if branch == "cond":
                velocity, actual_span, jvp_used = self._predict(
                    latent,
                    sigma_t=sigma_t,
                    sigma_s=sigma_s,
                    span=span,
                )
                self.pending_prediction = velocity
                self.pending_actual_span = int(actual_span)
                self.pending_jvp_used = bool(jvp_used)
                self.pending_latent = latent.detach()
                self.pending_sigma = sigma_t.detach()
            else:
                if self.pending_prediction is None:
                    raise RuntimeError(
                        "Qwen MeanCache uncond call has no cond prediction"
                    )
                velocity = self.pending_prediction
                actual_span = int(self.pending_actual_span)
                jvp_used = bool(self.pending_jvp_used)
            output = _replace_sample(self.output_templates[branch], velocity)
        else:
            output = original(*args, **kwargs)
            velocity = _sample(output)
            actual_span = 0
            jvp_used = False
            if self.output_templates[branch] is None:
                self.output_templates[branch] = output
            if branch == "cond":
                self.pending_cond = velocity.detach()
                self.pending_latent = latent.detach()
                self.pending_sigma = sigma_t.detach()

        completes_step = not self.true_cfg or branch == "uncond"
        if completes_step:
            if self.pending_latent is None or self.pending_sigma is None:
                raise RuntimeError("Qwen MeanCache step is missing latent history")
            if cache:
                assert self.pending_prediction is not None
                guided = self.pending_prediction
            else:
                if self.pending_cond is None:
                    raise RuntimeError(
                        "Qwen MeanCache full step is missing cond output"
                    )
                guided = self._guided_velocity(self.pending_cond, velocity)
            self._append_history(
                self.pending_latent,
                self.pending_sigma,
                guided,
            )
            self.pending_cond = None
            self.pending_prediction = None
            self.pending_actual_span = 0
            self.pending_jvp_used = False
            self.pending_latent = None
            self.pending_sigma = None
        row = self.steps.setdefault(
            step,
            {
                "step": step,
                "action": "cache" if cache else "full",
                "reason": "fixed_schedule_cache" if cache else "fixed_schedule_full",
                "branches": {},
            },
        )
        action = "cache" if cache else "full"
        if row["action"] != action:
            raise RuntimeError("Qwen CFG branches produced different MeanCache actions")
        row["branches"][branch] = {
            "action": action,
            "sigma_t": float(sigma_t.item()),
            "sigma_s": float(sigma_s.item()),
            "requested_jvp_span": int(span),
            "actual_jvp_span": int(actual_span),
            "jvp_correction_used": bool(jvp_used),
        }
        self.forward_call_count += 1
        return output

    def decisions(self) -> dict[str, Any]:
        rows = [
            self.steps.get(
                step,
                {
                    "step": step,
                    "action": "missing",
                    "reason": "missing",
                    "branches": {},
                },
            )
            for step in range(self.num_steps)
        ]
        cached = sum(row["action"] == "cache" for row in rows)
        if cached != len(self.cache_steps):
            raise RuntimeError(
                f"Qwen MeanCache recorded K={cached}, expected K={len(self.cache_steps)}"
            )
        return {
            "schema": "qwen_image_meancache_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "MeanCache",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "shared_step_action": True,
            "prediction_target": "post_true_cfg_guided_velocity",
            "true_cfg_scale": self.true_cfg_scale,
            "default_jvp_span": self.jvp_span,
            "jvp_spans": dict(self.jvp_spans),
            "steps": rows,
            "summary": {
                "n_total": self.num_steps,
                "n_full": self.num_steps - cached,
                "n_cached": cached,
                "cache_ratio": cached / self.num_steps,
                "jvp_corrected_steps": sum(
                    int(
                        any(
                            branch_row["jvp_correction_used"]
                            for branch_row in row["branches"].values()
                        )
                    )
                    for row in rows
                ),
            },
        }


def install_qwen_meancache(
    pipe: Any,
    *,
    cache_steps: tuple[int, ...],
    num_steps: int = 50,
    jvp_span: int = 4,
    jvp_spans: dict[int, int] | None = None,
    true_cfg: bool = True,
    true_cfg_scale: float = 4.0,
) -> QwenMeanCacheAdapter:
    adapter = QwenMeanCacheAdapter(
        pipe,
        cache_steps=cache_steps,
        num_steps=num_steps,
        jvp_span=jvp_span,
        jvp_spans=jvp_spans,
        true_cfg=true_cfg,
        true_cfg_scale=true_cfg_scale,
    )
    adapter.install()
    pipe.transformer._qwen_meancache_adapter = adapter
    return adapter


def restore_qwen_meancache(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    adapter = getattr(transformer, "_qwen_meancache_adapter", None)
    if adapter is not None:
        adapter.restore()
        delattr(transformer, "_qwen_meancache_adapter")


def reset_qwen_meancache(pipe: Any, *, prompt_idx: int, seed: int) -> None:
    pipe.transformer._qwen_meancache_adapter.reset(
        prompt_idx=prompt_idx,
        seed=seed,
    )


def qwen_meancache_decisions(pipe: Any) -> dict[str, Any]:
    return pipe.transformer._qwen_meancache_adapter.decisions()
