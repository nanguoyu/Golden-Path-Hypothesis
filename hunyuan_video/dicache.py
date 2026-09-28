"""DiCache adapter for the pinned Tencent HunyuanVideo backend."""

from __future__ import annotations

import types
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from lib.dicache import aligned_residual, append_anchor, relative_l1


@dataclass(frozen=True)
class HunyuanDiCacheConfig:
    num_steps: int = 50
    threshold: float = 0.1
    ret_ratio: float = 0.2
    probe_depth: int = 1
    #: When set, the action comes from this table instead of from the gate: the
    #: video SPX `di_two_anchor` payload column, which scores DiCache's payload
    #: on a schedule DiCache did not choose. The payload is untouched -- the
    #: probe still runs on every cached step, gamma is still estimated and
    #: clamped, full steps still refresh both anchors -- only the decision is
    #: read off the table, so `threshold` and `ret_ratio` stop being inputs.
    cache_steps: tuple[int, ...] | None = None


@dataclass(frozen=True)
class DiCacheDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    delta_x: float | None
    delta_y: float | None
    accumulated: float
    threshold: float
    gamma: float | None
    probe_depth: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class HunyuanDiCacheAdapter:
    """Run the official shallow probe and trajectory-alignment equations."""

    def __init__(self, transformer: Any, config: HunyuanDiCacheConfig):
        self.transformer = transformer
        self.config = config
        if not 1 <= int(config.probe_depth) <= len(transformer.double_blocks):
            raise ValueError("DiCache probe_depth is outside the Hunyuan double-block stack")
        if config.cache_steps is not None:
            steps = list(config.cache_steps)
            if steps != sorted(set(steps)):
                raise ValueError("DiCache cache_steps must be sorted and unique")
            if steps and (steps[0] < 2 or steps[-1] >= int(config.num_steps) - 1):
                raise ValueError(
                    "a fixed-schedule DiCache keeps steps 0, 1 and the terminal step full")
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.accumulated = 0.0
        self.previous_input: torch.Tensor | None = None
        self.previous_probe: torch.Tensor | None = None
        self.residual_history: list[torch.Tensor] = []
        self.probe_history: list[torch.Tensor] = []
        self.decisions: list[DiCacheDecisionRecord] = []
        self._current_action = "full"
        self._current_reason = "init"
        self._current_fields: dict[str, Any] = {}
        self._initial_img: torch.Tensor | None = None
        self._current_probe: torch.Tensor | None = None
        self._probe_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._block_calls = 0

    def __enter__(self) -> "HunyuanDiCacheAdapter":
        if self._installed:
            raise RuntimeError("Hunyuan DiCache adapter already installed")
        blocks = [*self.transformer.double_blocks, *self.transformer.single_blocks]
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
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
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
        self.decisions.append(
            DiCacheDecisionRecord(
                step=int(self.step),
                action=self._current_action,
                reason=self._current_reason,
                original_block_calls=int(self._block_calls),
                **self._current_fields,
            )
        )
        self.step += 1
        return output

    def _run_probe(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_img = img.clone()
        probe_txt = txt.clone()
        self._probe_outputs = []
        for index in range(int(self.config.probe_depth)):
            original = self._patches[index][1][1]
            probe_args = list(args)
            probe_kwargs = dict(kwargs)
            if probe_kwargs:
                if "img" in probe_kwargs:
                    probe_kwargs["img"] = probe_img
                if "txt" in probe_kwargs:
                    probe_kwargs["txt"] = probe_txt
            else:
                probe_args[0] = probe_img
                probe_args[1] = probe_txt
            probe_img, probe_txt = original(*probe_args, **probe_kwargs)
            self._probe_outputs.append((probe_img, probe_txt))
            self._block_calls += 1
        return probe_img, probe_txt

    def _decide(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        cfg = self.config
        self._initial_img = img.clone()
        self._current_probe = None
        # Gate boundaries follow the image-side implementation
        # (flux/dicache_native.py:166-173), NOT Tencent's official HunyuanVideo
        # variant: warmup is `step <= int(ratio * N)` (11 forced steps at
        # 0.2/50, not 10), the terminal step is forced full, and the threshold
        # comparison below is strict. The official per-backbone variants differ
        # on all three; the matrix compares backbones, so the gate must be the
        # same policy on both, and the image side is the locked baseline.
        warmup_last = int(float(cfg.ret_ratio) * int(cfg.num_steps))
        cache = False
        reason = "forced_boundary"
        proposed = self.accumulated
        delta_x: float | None = None
        delta_y: float | None = None
        if cfg.cache_steps is not None:
            # Fixed-schedule payload column: the table decides, and a cached
            # step it names before the anchors exist is an error rather than a
            # silent demotion, which would spend one of the K cached steps the
            # schedule promises.
            cache = self.step in cfg.cache_steps
            reason = "fixed_cache" if cache else "fixed_full"
            if cache:
                if self.previous_input is None or len(self.residual_history) < 2:
                    raise RuntimeError(
                        f"fixed-schedule DiCache caches step {self.step} before two full "
                        f"steps have laid down anchors")
                probe_img, _probe_txt = self._run_probe(img, txt, args, kwargs)
                self._current_probe = probe_img
                delta_x = relative_l1(img, self.previous_input)
                delta_y = relative_l1(probe_img, self.previous_probe)
                # recorded, never compared: the schedule already decided
                proposed = self.accumulated + delta_y
        else:
            hard_full = (
                self.step <= warmup_last
                or self.step == int(cfg.num_steps) - 1
                or self.previous_input is None
                or self.previous_probe is None
                or not self.residual_history
            )
            if not hard_full:
                probe_img, _probe_txt = self._run_probe(img, txt, args, kwargs)
                self._current_probe = probe_img
                delta_x = relative_l1(img, self.previous_input)
                delta_y = relative_l1(probe_img, self.previous_probe)
                # delta_y hardcoded: the image-side matrix ran error_choice=delta_y
                # in every DiCache cell, so the option is not carried
                proposed += delta_y
                cache = proposed < float(cfg.threshold)
                reason = "threshold_cache" if cache else "threshold_full"

        self._current_action = "cache" if cache else "full"
        self._current_reason = reason
        self.accumulated = float(proposed) if cache else 0.0
        gamma: float | None = None
        payload: torch.Tensor | None = None
        if cache:
            assert self._current_probe is not None
            probe_residual = self._current_probe - img
            payload, gamma = aligned_residual(
                probe_residual,
                self.residual_history,
                self.probe_history,
            )
            self.previous_input = img.detach()
            self.previous_probe = self._current_probe.detach()

        self._current_fields = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "accumulated": float(self.accumulated),
            "threshold": float(cfg.threshold),
            "gamma": gamma,
            "probe_depth": int(cfg.probe_depth),
        }
        return payload

    def _block_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        double_count = len(self.transformer.double_blocks)
        if index == 0:
            img = args[0] if args else kwargs.get("img")
            txt = args[1] if len(args) > 1 else kwargs.get("txt")
            if img is None or txt is None:
                raise RuntimeError("Hunyuan DiCache could not find first-block inputs")
            payload = self._decide(img, txt, args, kwargs)
            if self._current_action == "cache":
                assert payload is not None
                return img + payload, txt

        if self._current_action == "cache":
            return args[0] if index >= double_count else (args[0], args[1])

        if index < len(self._probe_outputs):
            output = self._probe_outputs[index]
        else:
            output = original(*args, **kwargs)
            self._block_calls += 1

        if index == int(self.config.probe_depth) - 1:
            self._current_probe = output[0].detach()
        if index == len(self._patches) - 1:
            if self._initial_img is None or self._current_probe is None:
                raise RuntimeError("Hunyuan DiCache full step is missing an anchor")
            img_len = self._initial_img.shape[1]
            final_img = output[:, :img_len]
            append_anchor(self.residual_history, final_img - self._initial_img)
            append_anchor(self.probe_history, self._current_probe - self._initial_img)
            self.previous_input = self._initial_img.detach()
            self.previous_probe = self._current_probe.detach()
        return output
