"""Fixed-schedule DPCache adapter for diffusers Qwen-Image."""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from lib.dpcache import predict_derivatives, update_derivatives
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


class QwenDPCacheAdapter:
    """Use DPCache's order-2 prediction with one action shared by CFG branches."""

    def __init__(
        self,
        pipe: Any,
        *,
        cache_steps: tuple[int, ...],
        num_steps: int = 50,
        order: int = 2,
        true_cfg: bool = True,
    ) -> None:
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_steps = int(num_steps)
        self.order = int(order)
        self.true_cfg = bool(true_cfg)
        if self.order != 2:
            raise ValueError("Qwen DPCache freezes the paper's order-2 predictor")
        self.cache_steps = validate_cache_steps(
            cache_steps,
            num_steps=self.num_steps,
            forced_full_steps={0, 1, 2, self.num_steps - 1},
        )
        self._cache_steps = frozenset(self.cache_steps)
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.histories: dict[str, dict[str, dict[int, torch.Tensor]]] = {
            branch: {"encoder": {}, "hidden": {}} for branch in BRANCHES
        }
        self.last_full_step: dict[str, int | None] = {
            branch: None for branch in BRANCHES
        }
        self.steps: dict[int, dict[str, Any]] = {}
        self.current_step = 0
        self.current_branch = "cond"
        self.current_action = "full"
        self.current_reason = "init"
        self.current_block_calls = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Qwen DPCache adapter already installed")
        blocks = list(self.transformer.transformer_blocks)
        if not blocks:
            raise ValueError("Qwen DPCache requires transformer_blocks")
        self._handles.append(
            self.transformer.register_forward_pre_hook(
                self._transformer_pre,
                with_kwargs=True,
            )
        )
        self._handles.append(
            self.transformer.register_forward_hook(
                self._transformer_post,
                with_kwargs=True,
            )
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

    def _transformer_pre(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        branches_per_step = 2 if self.true_cfg else 1
        call = int(self.forward_call_count)
        self.current_step = min(call // branches_per_step, self.num_steps - 1)
        self.current_branch = (
            "cond" if branches_per_step == 1 or call % 2 == 0 else "uncond"
        )
        cache = self.current_step in self._cache_steps
        if cache and self.last_full_step[self.current_branch] is None:
            raise RuntimeError(
                "Qwen DPCache reached cache before its first full anchor"
            )
        self.current_action = "cache" if cache else "full"
        self.current_reason = "fixed_schedule_cache" if cache else "fixed_schedule_full"
        self.current_block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        row = self.steps.setdefault(
            self.current_step,
            {
                "step": int(self.current_step),
                "action": self.current_action,
                "reason": self.current_reason,
                "branches": {},
            },
        )
        if row["action"] != self.current_action:
            raise RuntimeError("Qwen CFG branches produced different DPCache actions")
        row["branches"][self.current_branch] = {
            "action": self.current_action,
            "last_full_step": self.last_full_step[self.current_branch],
            "prediction_order": self.order,
            "original_block_calls": int(self.current_block_calls),
        }
        self.forward_call_count += 1
        return output

    @staticmethod
    def _states(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        encoder = kwargs.get(
            "encoder_hidden_states",
            args[1] if len(args) > 1 else None,
        )
        if hidden is None or encoder is None:
            raise RuntimeError("Qwen DPCache block wrapper could not find states")
        return hidden, encoder

    def _block_forward(
        self, index: int, _module: Any, *args: Any, **kwargs: Any
    ) -> Any:
        original = self._patches[index][1][1]
        last = index == len(self._patches) - 1
        branch = self.current_branch
        if self.current_action == "full":
            output = original(*args, **kwargs)
            self.current_block_calls += 1
            if last:
                encoder, hidden = output
                previous = self.last_full_step[branch]
                gap = 1 if previous is None else self.current_step - previous
                self.histories[branch]["encoder"] = update_derivatives(
                    self.histories[branch]["encoder"],
                    encoder,
                    step_gap=gap,
                    order=self.order,
                )
                self.histories[branch]["hidden"] = update_derivatives(
                    self.histories[branch]["hidden"],
                    hidden,
                    step_gap=gap,
                    order=self.order,
                )
                self.last_full_step[branch] = int(self.current_step)
            return output

        hidden, encoder = self._states(args, kwargs)
        if not last:
            return encoder, hidden
        previous = self.last_full_step[branch]
        assert previous is not None
        offset = self.current_step - previous
        return (
            predict_derivatives(
                self.histories[branch]["encoder"],
                step_offset=offset,
                order=self.order,
            ),
            predict_derivatives(
                self.histories[branch]["hidden"],
                step_offset=offset,
                order=self.order,
            ),
        )

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
                f"Qwen DPCache recorded K={cached}, expected K={len(self.cache_steps)}"
            )
        return {
            "schema": "qwen_image_dpcache_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "DPCache",
            "num_steps": self.num_steps,
            "cache_steps": list(self.cache_steps),
            "order": self.order,
            "shared_step_action": True,
            "steps": rows,
            "summary": {
                "n_total": self.num_steps,
                "n_full": self.num_steps - cached,
                "n_cached": cached,
                "cache_ratio": cached / self.num_steps,
            },
        }


def install_qwen_dpcache(
    pipe: Any,
    *,
    cache_steps: tuple[int, ...],
    num_steps: int = 50,
    order: int = 2,
    true_cfg: bool = True,
) -> QwenDPCacheAdapter:
    adapter = QwenDPCacheAdapter(
        pipe,
        cache_steps=cache_steps,
        num_steps=num_steps,
        order=order,
        true_cfg=true_cfg,
    )
    adapter.install()
    pipe.transformer._qwen_dpcache_adapter = adapter
    return adapter


def restore_qwen_dpcache(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    adapter = getattr(transformer, "_qwen_dpcache_adapter", None)
    if adapter is not None:
        adapter.restore()
        delattr(transformer, "_qwen_dpcache_adapter")


def reset_qwen_dpcache(pipe: Any, *, prompt_idx: int, seed: int) -> None:
    pipe.transformer._qwen_dpcache_adapter.reset(prompt_idx=prompt_idx, seed=seed)


def qwen_dpcache_decisions(pipe: Any) -> dict[str, Any]:
    return pipe.transformer._qwen_dpcache_adapter.decisions()
