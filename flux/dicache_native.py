"""Official-equation native DiCache adapter for diffusers FLUX."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable

import torch

from lib.dicache import aligned_residual, append_anchor, relative_l1


@dataclass(frozen=True)
class FluxDiCacheConfig:
    num_steps: int = 50
    threshold: float = 0.12
    error_choice: str = "delta_y"
    ret_ratio: float = 0.2
    probe_depth: int = 1


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


class FluxDiCacheAdapter:
    """Preserve DiCache's shallow probe and trajectory-alignment payload."""

    def __init__(self, pipe: Any, config: FluxDiCacheConfig):
        if config.error_choice not in {"delta_y", "delta_minus"}:
            raise ValueError("DiCache error_choice must be delta_y or delta_minus")
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.config = config
        if not 1 <= int(config.probe_depth) <= len(self.transformer.transformer_blocks):
            raise ValueError("DiCache probe_depth is outside the FLUX double-block stack")
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.accumulated = 0.0
        self.previous_input: torch.Tensor | None = None
        self.previous_probe: torch.Tensor | None = None
        self.residual_history: list[torch.Tensor] = []
        self.probe_history: list[torch.Tensor] = []
        self.records: list[dict[str, Any]] = []
        self._current_action = "full"
        self._current_reason = "init"
        self._current_fields: dict[str, Any] = {}
        self._initial_hidden: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._probe_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._block_calls = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX DiCache adapter already installed")
        blocks = [
            *self.transformer.transformer_blocks,
            *self.transformer.single_transformer_blocks,
        ]
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )
        for index, block in enumerate(blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._block_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_forward(block, wrapped)))
        self._installed = True

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _pre(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._block_calls = 0
        self._probe_outputs = []

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        self.records.append(
            {
                "step": int(self.step),
                "action": self._current_action,
                "u": int(self._current_action == "cache"),
                "reason": self._current_reason,
                "original_block_calls": int(self._block_calls),
                **self._current_fields,
            }
        )
        self.step += 1
        return output

    def _run_probe(
        self,
        hidden: torch.Tensor,
        encoder: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_hidden = hidden.clone()
        probe_encoder = encoder.clone()
        self._probe_outputs = []
        for index in range(int(self.config.probe_depth)):
            original = self._patches[index][1][1]
            probe_kwargs = dict(kwargs)
            probe_args = list(args)
            if "hidden_states" in probe_kwargs:
                probe_kwargs["hidden_states"] = probe_hidden
                probe_kwargs["encoder_hidden_states"] = probe_encoder
            else:
                probe_args[0] = probe_hidden
                probe_args[1] = probe_encoder
            probe_encoder, probe_hidden = original(*probe_args, **probe_kwargs)
            self._probe_outputs.append((probe_encoder, probe_hidden))
            self._block_calls += 1
        return probe_encoder, probe_hidden

    def _decide(
        self,
        hidden: torch.Tensor,
        encoder: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        cfg = self.config
        self._initial_hidden = hidden.clone()
        self._current_probe = None
        self._current_fields = {}
        warmup_last = int(float(cfg.ret_ratio) * int(cfg.num_steps))
        hard_full = (
            self.step <= warmup_last
            or self.step == int(cfg.num_steps) - 1
            or self.previous_input is None
            or self.previous_probe is None
            or not self.residual_history
        )
        native_cache = False
        native_reason = "forced_boundary"
        proposed = self.accumulated
        delta_x: float | None = None
        delta_y: float | None = None
        error: float | None = None
        if not hard_full:
            _probe_encoder, probe_hidden = self._run_probe(
                hidden, encoder, args, kwargs
            )
            self._current_probe = probe_hidden
            delta_x = relative_l1(hidden, self.previous_input)
            delta_y = relative_l1(probe_hidden, self.previous_probe)
            error = (
                delta_y
                if cfg.error_choice == "delta_y"
                else abs(delta_y - delta_x)
            )
            proposed += error
            native_cache = proposed < float(cfg.threshold)
            native_reason = (
                "threshold_cache" if native_cache else "threshold_full"
            )

        self._current_action = "cache" if native_cache else "full"
        self._current_reason = native_reason
        self.accumulated = float(proposed) if native_cache else 0.0
        gamma: float | None = None
        payload: torch.Tensor | None = None
        if native_cache:
            assert self._current_probe is not None
            probe_residual = self._current_probe - hidden
            payload, gamma = aligned_residual(
                probe_residual,
                self.residual_history,
                self.probe_history,
            )
            self.previous_probe = self._current_probe.detach()
            self.previous_input = hidden.detach()

        self._current_fields = {
            "native_action": "cache" if native_cache else "full",
            "delta_x": delta_x,
            "delta_y": delta_y,
            "error": error,
            "accumulated": float(self.accumulated),
            "threshold": float(cfg.threshold),
            "gamma": gamma,
            "probe_depth": int(cfg.probe_depth),
        }
        return payload

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            encoder = kwargs.get(
                "encoder_hidden_states",
                args[1] if len(args) > 1 else None,
            )
            if hidden is None or encoder is None:
                raise RuntimeError("FLUX DiCache could not find first-block inputs")
            payload = self._decide(hidden, encoder, args, kwargs)
            if self._current_action == "cache":
                assert payload is not None
                return encoder, hidden + payload

        if self._current_action == "cache":
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            encoder = kwargs.get(
                "encoder_hidden_states",
                args[1] if len(args) > 1 else None,
            )
            return encoder, hidden

        if index < len(self._probe_outputs):
            output = self._probe_outputs[index]
        else:
            output = original(*args, **kwargs)
            self._block_calls += 1

        if index == int(self.config.probe_depth) - 1:
            self._current_probe = output[1].detach()
        if index == len(self._patches) - 1:
            if self._initial_hidden is None or self._current_probe is None:
                raise RuntimeError("FLUX DiCache full step is missing an anchor")
            residual = output[1] - self._initial_hidden
            probe_residual = self._current_probe - self._initial_hidden
            append_anchor(self.residual_history, residual)
            append_anchor(self.probe_history, probe_residual)
            self.previous_input = self._initial_hidden.detach()
            self.previous_probe = self._current_probe.detach()
        return output

    def decisions(self) -> dict[str, Any]:
        cached = sum(row["action"] == "cache" for row in self.records)
        return {
            "schema": "flux_dicache_native_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "dicache_native",
            "num_steps": int(self.config.num_steps),
            "per_step": list(self.records),
            "summary": {
                "n_total": len(self.records),
                "n_full": len(self.records) - cached,
                "n_cached": cached,
                "cache_ratio": float(cached / len(self.records)) if self.records else 0.0,
            },
        }
