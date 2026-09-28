"""Official-target L2P adapter for diffusers FLUX.

The July 2026 L2P release predicts the projected transformer output.  This
module keeps that path separate from the older in-repo final-hidden L2P mode.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Callable

import torch

from lib.l2p import append_history, load_l2p_weight_file, predict_l2p


def _set_instance_forward(module: Any, function: Callable[..., Any]) -> tuple[bool, Any]:
    had_instance = "forward" in module.__dict__
    original = module.forward
    module.forward = types.MethodType(function, module)
    return had_instance, original


def _restore_instance_forward(module: Any, state: tuple[bool, Any]) -> None:
    had_instance, original = state
    if had_instance:
        module.forward = original
    else:
        delattr(module, "forward")


class FluxL2POutputAdapter:
    """Skip FLUX blocks on fixed cache steps and predict `proj_out` output."""

    def __init__(
        self,
        pipe: Any,
        *,
        weights_path: str | Path,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        min_abs_weight: float = 0.0,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.cache_steps = tuple(int(step) for step in cache_steps)
        self._cache_steps = frozenset(self.cache_steps)
        self.min_abs_weight = float(min_abs_weight)
        if self.cache_steps != tuple(sorted(set(self.cache_steps))):
            raise ValueError("FLUX L2P cache_steps must be sorted and unique")
        if any(step < 0 or step >= self.num_steps for step in self.cache_steps):
            raise ValueError("FLUX L2P cache_steps are outside the trajectory")
        if 0 in self._cache_steps:
            raise ValueError("FLUX L2P cannot cache step 0")
        metadata = load_l2p_weight_file(weights_path, num_steps=self.num_steps)
        target = str(metadata.get("target", "final_output"))
        if target not in {"final_output", "projected_output"}:
            raise ValueError(f"FLUX output L2P requires final-output weights, got {target!r}")
        self.weights = metadata["weights"]
        self.weights_metadata = {
            key: value for key, value in metadata.items() if key != "weights"
        }
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.current_action = "full"
        self.current_reason = "init"
        self.current_block_calls = 0
        self.current_prediction_fields: dict[str, Any] = {}
        self.history: dict[int, torch.Tensor] = {}
        self.step_records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX L2P adapter already installed")
        blocks = [
            *self.transformer.transformer_blocks,
            *self.transformer.single_transformer_blocks,
        ]
        if not blocks:
            raise ValueError("FLUX L2P requires transformer blocks")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        for index, block in enumerate(blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._block_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))

        proj_out = self.transformer.proj_out

        def projected(module: Any, *args: Any, **kwargs: Any):
            return self._projected_forward(module, *args, **kwargs)

        self._patches.append((proj_out, _set_instance_forward(proj_out, projected)))
        self._installed = True

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_instance_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _transformer_pre(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        current = min(int(self.step), self.num_steps - 1)
        cache = current in self._cache_steps
        if cache and not self.history:
            cache = False
            reason = "history_unready"
        else:
            reason = "fixed_cache" if cache else "fixed_full"
        self.current_action = "cache" if cache else "full"
        self.current_reason = reason
        self.current_block_calls = 0
        self.current_prediction_fields = {}

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        self.step_records.append(
            {
                "step": int(self.step),
                "action": self.current_action,
                "u": 1 if self.current_action == "cache" else 0,
                "reason": self.current_reason,
                "original_block_calls": int(self.current_block_calls),
                **self.current_prediction_fields,
            }
        )
        self.step += 1
        return output

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if self.current_action == "full":
            self.current_block_calls += 1
            return original(*args, **kwargs)
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        encoder = kwargs.get(
            "encoder_hidden_states",
            args[1] if len(args) > 1 else None,
        )
        if hidden is None or encoder is None:
            raise RuntimeError("FLUX L2P block wrapper could not find hidden states")
        return encoder, hidden

    def _projected_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.current_action == "full":
            original = self._patches[-1][1][1]
            value = original(*args, **kwargs)
            fields = {
                "l2p_target": "final_output",
                "l2p_weights_used": 0,
                "l2p_fallback_latest": False,
            }
        else:
            value, prediction = predict_l2p(
                self.history,
                self.weights,
                current_step=self.step,
                min_abs_weight=self.min_abs_weight,
            )
            fields = {"l2p_target": "final_output", **prediction}
        self.history = append_history(self.history, self.step, value)
        self.current_prediction_fields = fields
        return value

    def decisions(self) -> dict[str, Any]:
        cached = sum(record["action"] == "cache" for record in self.step_records)
        return {
            "schema": "flux_l2p_output_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "L2P_output",
            "target": "final_output",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "weights": dict(self.weights_metadata),
            "per_step": list(self.step_records),
            "summary": {
                "n_total": len(self.step_records),
                "n_full": len(self.step_records) - cached,
                "n_cached": cached,
                "cache_ratio": (
                    float(cached / len(self.step_records))
                    if self.step_records
                    else 0.0
                ),
            },
        }
