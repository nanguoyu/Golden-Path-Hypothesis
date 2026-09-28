"""Token-wise partial caching for diffusers FLUX.

The official ToCa FLUX code reuses full attention outputs on cache steps and
recomputes only sensitivity-selected MLP tokens. This adapter preserves that
compute pattern on the repository's diffusers backbone. Diffusers' fused SDPA
path does not expose the full attention matrix, so token sensitivity is ranked
by cached attention-output magnitude plus ToCa's cache-frequency term.
"""

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


def _value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    default: Any = None,
) -> Any:
    return kwargs.get(name, args[position] if len(args) > position else default)


def _normalized_score(value: torch.Tensor) -> torch.Tensor:
    score = value.detach().to(torch.float32).norm(dim=-1)
    return torch.nn.functional.normalize(score, dim=1, p=2)


def _select_fresh_indices(
    score: torch.Tensor,
    age: torch.Tensor,
    *,
    fresh_ratio: float,
    soft_fresh_weight: float,
    age_scale: float,
) -> torch.Tensor:
    if score.ndim != 2 or age.shape != score.shape:
        raise ValueError("ToCa score and age must have shape [batch,tokens]")
    tokens = int(score.shape[1])
    fresh = max(0, min(tokens, int(float(fresh_ratio) * tokens)))
    priority = score + float(soft_fresh_weight) * age.to(score.dtype) / float(age_scale)
    return priority.topk(fresh, dim=1, largest=True, sorted=False).indices


def _gather_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, value.shape[-1])
    return torch.gather(value, dim=1, index=expanded)


def _scatter_tokens(
    cached: torch.Tensor,
    indices: torch.Tensor,
    fresh: torch.Tensor,
) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, cached.shape[-1])
    cached.scatter_(dim=1, index=expanded, src=fresh.detach())
    return cached


class FluxToCaAdapter:
    """Run exact cache steps with ToCa-style token-wise partial recomputation."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        fresh_ratio: float = 0.1,
        soft_fresh_weight: float = 0.25,
        fresh_threshold: int = 4,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0, 1, 2, self.num_steps - 1},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self.fresh_ratio = float(fresh_ratio)
        self.soft_fresh_weight = float(soft_fresh_weight)
        self.fresh_threshold = int(fresh_threshold)
        if not 0.0 < self.fresh_ratio <= 1.0:
            raise ValueError("ToCa fresh_ratio must be in (0,1]")
        if self.fresh_threshold < 1:
            raise ValueError("ToCa fresh_threshold must be positive")
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.current_action = "full"
        self.current_cache_fresh = 0
        self.current_cache_tokens = 0
        self.current_block_calls = 0
        self.cache: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX ToCa adapter already installed")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.transformer_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._double_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_forward(block, wrapped)))
        offset = len(self._patches)
        for index, block in enumerate(self.transformer.single_transformer_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._single_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_forward(block, wrapped)))
        self._single_patch_offset = offset
        self._installed = True

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _pre(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        cache = self.step in self._cache_steps
        if cache and not self.cache:
            raise RuntimeError("ToCa reached a cache step before full token caches existed")
        self.current_action = "cache" if cache else "full"
        self.current_cache_fresh = 0
        self.current_cache_tokens = 0
        self.current_block_calls = 0

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        fresh_fraction = (
            self.current_cache_fresh / self.current_cache_tokens
            if self.current_cache_tokens
            else None
        )
        self.records.append(
            {
                "step": int(self.step),
                "action": self.current_action,
                "u": int(self.current_action == "cache"),
                "fresh_tokens": int(self.current_cache_fresh),
                "eligible_tokens": int(self.current_cache_tokens),
                "fresh_token_fraction": fresh_fraction,
                "reused_token_fraction": (
                    None if fresh_fraction is None else 1.0 - fresh_fraction
                ),
                "original_block_calls": int(self.current_block_calls),
            }
        )
        self.step += 1
        return output

    def _ratio(self, *, stream: str, layer: int) -> float:
        layer_factor = 1.5 - float(layer) / 27.0
        stream_factor = 0.4 if stream == "double" else 1.6
        return max(0.0, min(1.0, self.fresh_ratio * layer_factor * stream_factor))

    def _fresh(
        self,
        state: dict[str, torch.Tensor],
        *,
        score_key: str,
        age_key: str,
        stream: str,
        layer: int,
    ) -> torch.Tensor:
        score = state[score_key]
        age = state[age_key]
        indices = _select_fresh_indices(
            score,
            age,
            fresh_ratio=self._ratio(stream=stream, layer=layer),
            soft_fresh_weight=self.soft_fresh_weight,
            age_scale=float(self.fresh_threshold),
        )
        age.add_(1)
        age.scatter_(1, indices, 0)
        self.current_cache_fresh += int(indices.numel())
        self.current_cache_tokens += int(score.numel())
        return indices

    def _double_forward(
        self,
        index: int,
        module: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = _value(args, kwargs, "hidden_states", 0)
        encoder = _value(args, kwargs, "encoder_hidden_states", 1)
        temb = _value(args, kwargs, "temb", 2)
        image_rotary_emb = _value(args, kwargs, "image_rotary_emb", 3)
        joint = _value(args, kwargs, "joint_attention_kwargs", 4, None) or {}
        if hidden is None or encoder is None or temb is None:
            raise RuntimeError("ToCa double block received incomplete inputs")

        (
            norm_hidden,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = module.norm1(hidden, emb=temb)
        (
            norm_encoder,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = module.norm1_context(encoder, emb=temb)
        key = ("double", index)

        if self.current_action == "full":
            attention = module.attn(
                hidden_states=norm_hidden,
                encoder_hidden_states=norm_encoder,
                image_rotary_emb=image_rotary_emb,
                **joint,
            )
            attn_hidden, attn_encoder = attention[:2]
            ip_output = attention[2] if len(attention) == 3 else None
            state = {
                "attn_hidden": attn_hidden.detach(),
                "attn_encoder": attn_encoder.detach(),
                "score_hidden": _normalized_score(attn_hidden),
                "score_encoder": _normalized_score(attn_encoder),
                "age_hidden": torch.zeros(
                    attn_hidden.shape[:2], device=attn_hidden.device, dtype=torch.int32
                ),
                "age_encoder": torch.zeros(
                    attn_encoder.shape[:2], device=attn_encoder.device, dtype=torch.int32
                ),
            }

            hidden = hidden + gate_msa.unsqueeze(1) * attn_hidden
            norm_hidden = module.norm2(hidden)
            norm_hidden = norm_hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_hidden = module.ff(norm_hidden)
            state["ff_hidden"] = ff_hidden.detach()
            hidden = hidden + gate_mlp.unsqueeze(1) * ff_hidden
            if ip_output is not None:
                hidden = hidden + ip_output

            encoder = encoder + c_gate_msa.unsqueeze(1) * attn_encoder
            norm_encoder = module.norm2_context(encoder)
            norm_encoder = (
                norm_encoder * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            ff_encoder = module.ff_context(norm_encoder)
            state["ff_encoder"] = ff_encoder.detach()
            encoder = encoder + c_gate_mlp.unsqueeze(1) * ff_encoder
            self.cache[key] = state
            self.current_block_calls += 1
        else:
            if joint.get("ip_hidden_states") is not None:
                raise NotImplementedError("ToCa partial steps do not support IP-Adapter")
            state = self.cache[key]
            hidden = hidden + gate_msa.unsqueeze(1) * state["attn_hidden"]
            hidden_indices = self._fresh(
                state,
                score_key="score_hidden",
                age_key="age_hidden",
                stream="double",
                layer=index,
            )
            fresh_hidden = _gather_tokens(hidden, hidden_indices)
            fresh_norm = module.norm2(fresh_hidden)
            fresh_norm = (
                fresh_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )
            fresh_ff = module.ff(fresh_norm)
            _scatter_tokens(state["ff_hidden"], hidden_indices, fresh_ff)
            hidden = hidden + gate_mlp.unsqueeze(1) * state["ff_hidden"]

            encoder = encoder + c_gate_msa.unsqueeze(1) * state["attn_encoder"]
            encoder_indices = self._fresh(
                state,
                score_key="score_encoder",
                age_key="age_encoder",
                stream="double",
                layer=index,
            )
            fresh_encoder = _gather_tokens(encoder, encoder_indices)
            fresh_norm_encoder = module.norm2_context(fresh_encoder)
            fresh_norm_encoder = (
                fresh_norm_encoder * (1 + c_scale_mlp[:, None])
                + c_shift_mlp[:, None]
            )
            fresh_ff_encoder = module.ff_context(fresh_norm_encoder)
            _scatter_tokens(state["ff_encoder"], encoder_indices, fresh_ff_encoder)
            encoder = encoder + c_gate_mlp.unsqueeze(1) * state["ff_encoder"]

        if encoder.dtype == torch.float16:
            encoder = encoder.clip(-65504, 65504)
        return encoder, hidden

    def _single_forward(
        self,
        index: int,
        module: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = _value(args, kwargs, "hidden_states", 0)
        encoder = _value(args, kwargs, "encoder_hidden_states", 1)
        temb = _value(args, kwargs, "temb", 2)
        image_rotary_emb = _value(args, kwargs, "image_rotary_emb", 3)
        joint = _value(args, kwargs, "joint_attention_kwargs", 4, None) or {}
        if hidden is None or encoder is None or temb is None:
            raise RuntimeError("ToCa single block received incomplete inputs")

        text_tokens = encoder.shape[1]
        combined = torch.cat([encoder, hidden], dim=1)
        residual = combined
        norm_combined, gate = module.norm(combined, emb=temb)
        key = ("single", index)

        if self.current_action == "full":
            mlp = module.act_mlp(module.proj_mlp(norm_combined))
            attn = module.attn(
                hidden_states=norm_combined,
                image_rotary_emb=image_rotary_emb,
                **joint,
            )
            projected = module.proj_out(torch.cat([attn, mlp], dim=2))
            self.cache[key] = {
                "attn": attn.detach(),
                "projected": projected.detach(),
                "score": _normalized_score(attn),
                "age": torch.zeros(
                    attn.shape[:2], device=attn.device, dtype=torch.int32
                ),
            }
            self.current_block_calls += 1
        else:
            if joint.get("ip_hidden_states") is not None:
                raise NotImplementedError("ToCa partial steps do not support IP-Adapter")
            state = self.cache[key]
            indices = self._fresh(
                state,
                score_key="score",
                age_key="age",
                stream="single",
                layer=index,
            )
            fresh_norm = _gather_tokens(norm_combined, indices)
            fresh_mlp = module.act_mlp(module.proj_mlp(fresh_norm))
            fresh_attn = _gather_tokens(state["attn"], indices)
            fresh_projected = module.proj_out(
                torch.cat([fresh_attn, fresh_mlp], dim=2)
            )
            _scatter_tokens(state["projected"], indices, fresh_projected)
            projected = state["projected"]

        combined = residual + gate.unsqueeze(1) * projected
        if combined.dtype == torch.float16:
            combined = combined.clip(-65504, 65504)
        return combined[:, :text_tokens], combined[:, text_tokens:]

    def decisions(self) -> dict[str, Any]:
        cached = sum(row["u"] for row in self.records)
        if len(self.records) != self.num_steps or cached != len(self.cache_steps):
            raise RuntimeError(
                f"ToCa recorded {len(self.records)} steps and K={cached}; "
                f"expected {self.num_steps} and K={len(self.cache_steps)}"
            )
        fresh = sum(row["fresh_tokens"] for row in self.records)
        eligible = sum(row["eligible_tokens"] for row in self.records)
        return {
            "schema": "flux_toca_exact_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "toca_exact",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "fresh_ratio_base": self.fresh_ratio,
            "soft_fresh_weight": self.soft_fresh_weight,
            "fresh_threshold": self.fresh_threshold,
            "token_score": "attention_output_norm_plus_cache_frequency",
            "per_step": list(self.records),
            "summary": {
                "n_total": len(self.records),
                "n_full": len(self.records) - cached,
                "n_cached": cached,
                "cache_ratio": cached / len(self.records),
                "partial_fresh_token_fraction": fresh / eligible if eligible else 0.0,
                "partial_reused_token_fraction": (
                    1.0 - fresh / eligible if eligible else 0.0
                ),
            },
        }
