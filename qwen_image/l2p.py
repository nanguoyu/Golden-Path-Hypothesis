"""Output-level L2P adapter for diffusers Qwen-Image."""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import torch

from lib.l2p import append_history, load_l2p_weight_file, predict_l2p


BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class QwenL2PConfig:
    weights_path: str | Path
    cache_steps: tuple[int, ...]
    num_steps: int = 50
    min_abs_weight: float = 0.0
    true_cfg: bool = True


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


class QwenL2PAdapter:
    """Skip Qwen blocks and predict the projected transformer output."""

    def __init__(self, pipe: Any, config: QwenL2PConfig):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.config = config
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        metadata = load_l2p_weight_file(
            config.weights_path,
            num_steps=int(config.num_steps),
        )
        target = str(metadata.get("target", "final_output"))
        if target not in {"final_output", "projected_output"}:
            raise ValueError(f"Qwen L2P requires final-output weights, got {target!r}")
        self.weights = metadata["weights"]
        self.weights_metadata = {
            key: value for key, value in metadata.items() if key != "weights"
        }
        self._validate_config()
        self._cache_steps = frozenset(int(step) for step in config.cache_steps)
        self.reset()

    def _validate_config(self) -> None:
        steps = tuple(int(step) for step in self.config.cache_steps)
        if steps != tuple(sorted(set(steps))):
            raise ValueError("Qwen L2P cache_steps must be sorted and unique")
        if any(step < 0 or step >= int(self.config.num_steps) for step in steps):
            raise ValueError("Qwen L2P cache_steps are outside the trajectory")
        if 0 in steps:
            raise ValueError("Qwen L2P cannot cache step 0")

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.histories: dict[str, dict[int, torch.Tensor]] = {
            branch: {} for branch in BRANCHES
        }
        self.steps: dict[int, dict[str, Any]] = {}
        self.current_step = 0
        self.current_branch = "cond"
        self.current_action = "full"
        self.current_reason = "init"
        self.current_block_calls = 0
        self.current_prediction_fields: dict[str, Any] = {}

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Qwen L2P adapter already installed")
        blocks = list(self.transformer.transformer_blocks)
        if not blocks:
            raise ValueError("Qwen L2P requires transformer_blocks")
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
        branches_per_step = 2 if self.config.true_cfg else 1
        call = int(self.forward_call_count)
        step = min(call // branches_per_step, int(self.config.num_steps) - 1)
        branch = "cond" if branches_per_step == 1 or call % 2 == 0 else "uncond"
        cache = step in self._cache_steps
        if cache and not self.histories[branch]:
            cache = False
            reason = "history_unready"
        else:
            reason = "fixed_cache" if cache else "fixed_full"
        self.current_step = int(step)
        self.current_branch = branch
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
        step = self.steps.setdefault(
            self.current_step,
            {
                "step": self.current_step,
                "action": self.current_action,
                "reason": self.current_reason,
                "branches": {},
            },
        )
        if step["action"] != self.current_action:
            raise RuntimeError("Qwen true-CFG branches produced different L2P actions")
        step["branches"][self.current_branch] = {
            "action": self.current_action,
            "reason": self.current_reason,
            "original_block_calls": int(self.current_block_calls),
            **self.current_prediction_fields,
        }
        self.forward_call_count += 1
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
            raise RuntimeError("Qwen L2P block wrapper could not find hidden states")
        return encoder, hidden

    def _projected_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        history = self.histories[self.current_branch]
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
                history,
                self.weights,
                current_step=self.current_step,
                min_abs_weight=float(self.config.min_abs_weight),
            )
            fields = {"l2p_target": "final_output", **prediction}
        self.histories[self.current_branch] = append_history(
            history,
            self.current_step,
            value,
        )
        self.current_prediction_fields = fields
        return value

    def decisions(self) -> Dict[str, Any]:
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
            for step in range(int(self.config.num_steps))
        ]
        cached = sum(row["action"] == "cache" for row in rows)
        full = sum(row["action"] == "full" for row in rows)
        return {
            "schema": "qwen_image_l2p_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "L2P_output",
            "target": "final_output",
            "shared_step_action": True,
            "num_steps": int(self.config.num_steps),
            "cache_steps": list(self.config.cache_steps),
            "weights": dict(self.weights_metadata),
            "steps": rows,
            "summary": {
                "n_total": int(self.config.num_steps),
                "n_full": int(full),
                "n_cached": int(cached),
                "cache_ratio": float(cached / int(self.config.num_steps)),
            },
        }


def install_qwen_l2p(pipe: Any, config: QwenL2PConfig) -> QwenL2PAdapter:
    adapter = QwenL2PAdapter(pipe, config)
    adapter.install()
    pipe.transformer._qwen_l2p_adapter = adapter
    return adapter


def restore_qwen_l2p(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    adapter = getattr(transformer, "_qwen_l2p_adapter", None)
    if adapter is not None:
        adapter.restore()
        delattr(transformer, "_qwen_l2p_adapter")


def reset_qwen_l2p(pipe: Any, *, prompt_idx: int, seed: int) -> None:
    pipe.transformer._qwen_l2p_adapter.reset(prompt_idx=prompt_idx, seed=seed)


def qwen_l2p_decisions(pipe: Any) -> Dict[str, Any]:
    return pipe.transformer._qwen_l2p_adapter.decisions()


def qwen_l2p_histories(pipe: Any) -> Iterable[dict[int, torch.Tensor]]:
    return pipe.transformer._qwen_l2p_adapter.histories.values()
