"""Native-gate DiCache adapter for diffusers Qwen-Image."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable

import torch

from lib.dicache import aligned_residual, append_anchor, relative_l1


BRANCHES = ("cond", "uncond")


@dataclass(frozen=True)
class QwenDiCacheConfig:
    num_steps: int = 50
    threshold: float = 0.12
    error_choice: str = "delta_y"
    ret_ratio: float = 0.2
    probe_depth: int = 1
    true_cfg: bool = True


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


class QwenDiCacheAdapter:
    """Share DiCache's gate action while keeping CFG payload histories separate."""

    def __init__(self, pipe: Any, config: QwenDiCacheConfig):
        if config.error_choice not in {"delta_y", "delta_minus"}:
            raise ValueError("Qwen DiCache error_choice must be delta_y or delta_minus")
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.config = config
        blocks = list(self.transformer.transformer_blocks)
        if not 1 <= int(config.probe_depth) <= len(blocks):
            raise ValueError("Qwen DiCache probe_depth is outside the block stack")
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed
        self.forward_call_count = 0
        self.accumulated = 0.0
        self.branches: dict[str, dict[str, Any]] = {
            branch: {
                "previous_input": None,
                "previous_probe": None,
                "residual_history": [],
                "probe_history": [],
            }
            for branch in BRANCHES
        }
        self.step_meta: dict[int, dict[str, Any]] = {}
        self.steps: dict[int, dict[str, Any]] = {}
        self.current_step = 0
        self.current_branch = "cond"
        self.current_action = "full"
        self.current_reason = "init"
        self.current_fields: dict[str, Any] = {}
        self.initial_hidden: torch.Tensor | None = None
        self.current_probe: torch.Tensor | None = None
        self.probe_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.current_block_calls = 0

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("Qwen DiCache adapter already installed")
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
        for index, block in enumerate(self.transformer.transformer_blocks):

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
        branches_per_step = 2 if self.config.true_cfg else 1
        call = int(self.forward_call_count)
        self.current_step = min(
            call // branches_per_step,
            int(self.config.num_steps) - 1,
        )
        self.current_branch = (
            "cond" if branches_per_step == 1 or call % 2 == 0 else "uncond"
        )
        meta = self.step_meta.get(self.current_step)
        self.current_action = "full" if meta is None else str(meta["action"])
        self.current_reason = "pre_decision" if meta is None else str(meta["reason"])
        self.current_fields = {}
        self.initial_hidden = None
        self.current_probe = None
        self.probe_outputs = []
        self.current_block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        meta = self.step_meta[self.current_step]
        row = self.steps.setdefault(
            self.current_step,
            {
                "step": int(self.current_step),
                "action": str(meta["action"]),
                "reason": str(meta["reason"]),
                "gate": dict(meta.get("gate", {})),
                "branches": {},
            },
        )
        if row["action"] != self.current_action:
            raise RuntimeError("Qwen CFG branches produced different DiCache actions")
        row["branches"][self.current_branch] = {
            "action": self.current_action,
            "reason": self.current_reason,
            "original_block_calls": int(self.current_block_calls),
            **self.current_fields,
        }
        self.forward_call_count += 1
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
        self.probe_outputs = []
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
            self.probe_outputs.append((probe_encoder, probe_hidden))
            self.current_block_calls += 1
        return probe_encoder, probe_hidden

    def _all_branches_ready(self) -> bool:
        return all(
            self.branches[branch]["previous_input"] is not None
            and self.branches[branch]["previous_probe"] is not None
            and bool(self.branches[branch]["residual_history"])
            for branch in BRANCHES
        )

    def _decide_cond(
        self,
        hidden: torch.Tensor,
        encoder: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        cfg = self.config
        branch_state = self.branches["cond"]
        warmup_last = int(float(cfg.ret_ratio) * int(cfg.num_steps))
        hard_full = (
            self.current_step <= warmup_last
            or self.current_step == int(cfg.num_steps) - 1
            or not self._all_branches_ready()
        )
        delta_x: float | None = None
        delta_y: float | None = None
        error: float | None = None
        proposed = float(self.accumulated)
        if hard_full:
            action = "full"
            reason = "forced_boundary_or_unready"
        else:
            _probe_encoder, probe_hidden = self._run_probe(
                hidden,
                encoder,
                args,
                kwargs,
            )
            self.current_probe = probe_hidden
            delta_x = relative_l1(hidden, branch_state["previous_input"])
            delta_y = relative_l1(probe_hidden, branch_state["previous_probe"])
            error = delta_y if cfg.error_choice == "delta_y" else abs(delta_y - delta_x)
            proposed += float(error)
            action = "cache" if proposed < float(cfg.threshold) else "full"
            reason = "threshold_cache" if action == "cache" else "threshold_full"
        self.accumulated = proposed if action == "cache" else 0.0
        self.current_action = action
        self.current_reason = reason
        gate = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "error": error,
            "accumulated": float(self.accumulated),
            "threshold": float(cfg.threshold),
            "error_choice": cfg.error_choice,
            "ret_ratio": float(cfg.ret_ratio),
            "probe_depth": int(cfg.probe_depth),
        }
        self.step_meta[self.current_step] = {
            "action": action,
            "reason": reason,
            "gate": gate,
        }

    def _cache_payload(
        self,
        hidden: torch.Tensor,
        encoder: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        branch_state = self.branches[self.current_branch]
        if self.current_probe is None:
            _probe_encoder, probe_hidden = self._run_probe(
                hidden,
                encoder,
                args,
                kwargs,
            )
            self.current_probe = probe_hidden
        probe_residual = self.current_probe - hidden
        payload, gamma = aligned_residual(
            probe_residual,
            branch_state["residual_history"],
            branch_state["probe_history"],
        )
        branch_state["previous_input"] = hidden.detach()
        branch_state["previous_probe"] = self.current_probe.detach()
        self.current_fields = {"gamma": gamma}
        return payload

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
            raise RuntimeError("Qwen DiCache block wrapper could not find states")
        return hidden, encoder

    def _block_forward(
        self, index: int, _module: Any, *args: Any, **kwargs: Any
    ) -> Any:
        original = self._patches[index][1][1]
        hidden, encoder = self._states(args, kwargs)
        if index == 0:
            self.initial_hidden = hidden.detach()
            if (
                self.current_branch == "cond"
                and self.current_step not in self.step_meta
            ):
                self._decide_cond(hidden, encoder, args, kwargs)
            elif self.current_step not in self.step_meta:
                raise RuntimeError("Qwen DiCache expected cond before uncond")
            else:
                meta = self.step_meta[self.current_step]
                self.current_action = str(meta["action"])
                self.current_reason = str(meta["reason"])
            if self.current_action == "cache":
                payload = self._cache_payload(hidden, encoder, args, kwargs)
                return encoder, hidden + payload

        if self.current_action == "cache":
            return encoder, hidden

        if index < len(self.probe_outputs):
            output = self.probe_outputs[index]
        else:
            output = original(*args, **kwargs)
            self.current_block_calls += 1
        if index == int(self.config.probe_depth) - 1:
            self.current_probe = output[1].detach()
        if index == len(self._patches) - 1:
            if self.initial_hidden is None or self.current_probe is None:
                raise RuntimeError("Qwen DiCache full step is missing an anchor")
            branch_state = self.branches[self.current_branch]
            residual = output[1] - self.initial_hidden
            probe_residual = self.current_probe - self.initial_hidden
            append_anchor(branch_state["residual_history"], residual)
            append_anchor(branch_state["probe_history"], probe_residual)
            branch_state["previous_input"] = self.initial_hidden.detach()
            branch_state["previous_probe"] = self.current_probe.detach()
        return output

    def decisions(self) -> dict[str, Any]:
        rows = [
            self.steps.get(
                step,
                {
                    "step": step,
                    "action": "missing",
                    "reason": "missing",
                    "gate": {},
                    "branches": {},
                },
            )
            for step in range(int(self.config.num_steps))
        ]
        cached = sum(row["action"] == "cache" for row in rows)
        return {
            "schema": "qwen_image_dicache_native_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "DiCache",
            "num_steps": int(self.config.num_steps),
            "shared_step_action": True,
            "decision_source_branch": "cond",
            "config": {
                "threshold": float(self.config.threshold),
                "error_choice": self.config.error_choice,
                "ret_ratio": float(self.config.ret_ratio),
                "probe_depth": int(self.config.probe_depth),
            },
            "steps": rows,
            "summary": {
                "n_total": int(self.config.num_steps),
                "n_full": int(self.config.num_steps) - cached,
                "n_cached": cached,
                "cache_ratio": cached / int(self.config.num_steps),
            },
        }


def install_qwen_dicache(
    pipe: Any,
    config: QwenDiCacheConfig,
) -> QwenDiCacheAdapter:
    adapter = QwenDiCacheAdapter(pipe, config)
    adapter.install()
    pipe.transformer._qwen_dicache_adapter = adapter
    return adapter


def restore_qwen_dicache(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    adapter = getattr(transformer, "_qwen_dicache_adapter", None)
    if adapter is not None:
        adapter.restore()
        delattr(transformer, "_qwen_dicache_adapter")


def reset_qwen_dicache(pipe: Any, *, prompt_idx: int, seed: int) -> None:
    pipe.transformer._qwen_dicache_adapter.reset(prompt_idx=prompt_idx, seed=seed)


def qwen_dicache_decisions(pipe: Any) -> dict[str, Any]:
    return pipe.transformer._qwen_dicache_adapter.decisions()
