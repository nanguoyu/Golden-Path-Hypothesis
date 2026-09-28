"""Exact-schedule DPCache adapter for diffusers FLUX."""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from lib.dpcache import predict_derivatives, update_derivatives
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


class FluxDPCacheAdapter:
    """Use a path-aware fixed schedule with DPCache's order-2 feature forecast."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        order: int = 2,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.order = int(order)
        if self.order != 2:
            raise ValueError("the DPCache baseline freezes the paper's order-2 predictor")
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0, 1, 2, self.num_steps - 1},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self.double_count = len(self.transformer.transformer_blocks)
        self.single_count = len(self.transformer.single_transformer_blocks)
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.step = 0
        self.current_action = "full"
        self.current_block_calls = 0
        self.last_full_step: int | None = None
        self.histories: dict[str, dict[int, torch.Tensor]] = {
            "double_encoder": {},
            "double_hidden": {},
            "single_encoder": {},
            "single_hidden": {},
        }
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("FLUX DPCache adapter already installed")
        blocks = [
            *self.transformer.transformer_blocks,
            *self.transformer.single_transformer_blocks,
        ]
        if not blocks:
            raise ValueError("FLUX DPCache requires transformer blocks")
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

    def _pre(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        cache = self.step in self._cache_steps
        if cache and self.last_full_step is None:
            raise RuntimeError("DPCache reached a cache step before its first full anchor")
        self.current_action = "cache" if cache else "full"
        self.current_block_calls = 0

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
                "action": self.current_action,
                "u": int(self.current_action == "cache"),
                "last_full_step": self.last_full_step,
                "prediction_order": self.order,
                "original_block_calls": int(self.current_block_calls),
            }
        )
        self.step += 1
        return output

    @staticmethod
    def _states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        encoder = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        if hidden is None or encoder is None:
            raise RuntimeError("FLUX DPCache block wrapper could not find hidden states")
        return hidden, encoder

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        double_stream = index < self.double_count
        stream = "double" if double_stream else "single"
        stream_index = index if double_stream else index - self.double_count
        stream_count = self.double_count if double_stream else self.single_count
        last = stream_index == stream_count - 1
        if self.current_action == "full":
            output = original(*args, **kwargs)
            self.current_block_calls += 1
            if last:
                encoder, hidden = output
                gap = 1 if self.last_full_step is None else self.step - self.last_full_step
                encoder_key = f"{stream}_encoder"
                hidden_key = f"{stream}_hidden"
                self.histories[encoder_key] = update_derivatives(
                    self.histories[encoder_key],
                    encoder,
                    step_gap=gap,
                    order=self.order,
                )
                self.histories[hidden_key] = update_derivatives(
                    self.histories[hidden_key],
                    hidden,
                    step_gap=gap,
                    order=self.order,
                )
                if not double_stream:
                    self.last_full_step = int(self.step)
            return output

        hidden, encoder = self._states(args, kwargs)
        if not last:
            return encoder, hidden
        assert self.last_full_step is not None
        offset = self.step - self.last_full_step
        encoder_key = f"{stream}_encoder"
        hidden_key = f"{stream}_hidden"
        predicted_encoder = predict_derivatives(
            self.histories[encoder_key],
            step_offset=offset,
            order=self.order,
        )
        predicted_hidden = predict_derivatives(
            self.histories[hidden_key],
            step_offset=offset,
            order=self.order,
        )
        return predicted_encoder, predicted_hidden

    def decisions(self) -> dict[str, Any]:
        cached = sum(record["u"] for record in self.records)
        if len(self.records) != self.num_steps or cached != len(self.cache_steps):
            raise RuntimeError(
                f"DPCache recorded {len(self.records)} steps and K={cached}; "
                f"expected {self.num_steps} and K={len(self.cache_steps)}"
            )
        return {
            "schema": "flux_dpcache_exact_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "dpcache_exact",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "full_steps": [
                step for step in range(self.num_steps) if step not in self._cache_steps
            ],
            "order": self.order,
            "per_step": list(self.records),
            "summary": {
                "n_total": len(self.records),
                "n_full": len(self.records) - cached,
                "n_cached": cached,
                "cache_ratio": cached / len(self.records),
            },
        }
