from __future__ import annotations

import types
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from hunyuan_video.actions import MethodDecision
from hunyuan_video.records import DecisionRecord
from hunyuan_video.methods.l2p import L2POutputMethod
from hunyuan_video.methods.taylorseer import TaylorSeerMethod


def _apply_gate(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    return value if gate is None else value * gate.unsqueeze(1)


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


class CoarseBackboneAdapter:
    """Whole-backbone hidden-residual cache without replacing Tencent top-level forward."""

    def __init__(self, transformer: Any, method: Any):
        self.transformer = transformer
        self.method = method
        self.decisions: list[DecisionRecord] = []
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.method.reset()
        self.decisions.clear()
        self.previous_residual: torch.Tensor | None = None
        self._current_full = True
        self._current_reason = "uninitialized"
        self._current_decision: MethodDecision | None = None
        self._initial_img: torch.Tensor | None = None
        self._grid_shape: tuple[int, int, int] | None = None
        self._block_calls = 0

    def __enter__(self) -> "CoarseBackboneAdapter":
        if self._installed:
            raise RuntimeError("adapter already installed")
        if not self.transformer.double_blocks or not self.transformer.single_blocks:
            raise ValueError("HunyuanVideo coarse adapter requires double and single blocks")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.double_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._double_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))
        for index, block in enumerate(self.transformer.single_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._single_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))
        self._installed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_instance_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _transformer_pre(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        x = args[0] if args else kwargs["x"]
        _, _, ot, oh, ow = x.shape
        pt, ph, pw = self.transformer.patch_size
        self._grid_shape = (ot // pt, oh // ph, ow // pw)
        self._block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        decision = self._current_decision
        if decision is None:
            raise RuntimeError("coarse adapter did not enter the first double block")
        self.decisions.append(
            DecisionRecord(
                step=len(self.decisions),
                action="full" if self._current_full else "cache",
                reason=self._current_reason,
                gate_scalar=decision.gate_scalar,
                accumulated=decision.accumulated,
                threshold=decision.threshold,
                original_block_calls=self._block_calls,
            )
        )
        return output

    def _double_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            img, txt, vec = args[:3]
            if self._grid_shape is None:
                raise RuntimeError("transformer pre-hook did not capture the latent grid")
            decision = self.method.decide(
                img=img,
                txt=txt,
                vec=vec,
                first_block=module,
                grid_shape=self._grid_shape,
            )
            if not decision.full and self.previous_residual is None:
                decision = MethodDecision(
                    full=True,
                    reason="missing_payload_forced_full",
                    gate_scalar=decision.gate_scalar,
                    accumulated=decision.accumulated,
                    threshold=decision.threshold,
                )
            self._current_decision = decision
            self._current_full = decision.full
            self._current_reason = decision.reason
            if not decision.full:
                return img + self.previous_residual, txt
            self._initial_img = img.clone()
        if not self._current_full:
            return args[0], args[1]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _single_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        offset = len(self.transformer.double_blocks) + index
        original = self._patches[offset][1][1]
        if not self._current_full:
            return args[0]
        self._block_calls += 1
        output = original(*args, **kwargs)
        if index == len(self.transformer.single_blocks) - 1:
            if self._initial_img is None:
                raise RuntimeError("missing coarse residual anchor")
            img_len = self._initial_img.shape[1]
            self.previous_residual = (output[:, :img_len] - self._initial_img).detach()
        return output


class TaylorSeerFineAdapter:
    """Component-level O1 TaylorSeer adapter matching the released Hunyuan slots."""

    def __init__(
        self,
        transformer: Any,
        method: TaylorSeerMethod,
        *,
        source_history_compat: bool = False,
    ):
        self.transformer = transformer
        self.method = method
        self.decisions: list[DecisionRecord] = []
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._current_full = True
        self._current_decision: MethodDecision | None = None
        self._block_calls = 0
        self._installed = False
        self.source_history_compat = source_history_compat

    @property
    def slot_count(self) -> int:
        single_slots = 2 if self.source_history_compat else 1
        return (
            len(self.transformer.double_blocks) * 4
            + len(self.transformer.single_blocks) * single_slots
        )

    def reset(self) -> None:
        self.method.reset()
        self.decisions.clear()
        self._current_full = True
        self._current_decision = None
        self._block_calls = 0

    def __enter__(self) -> "TaylorSeerFineAdapter":
        if self._installed:
            raise RuntimeError("adapter already installed")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.double_blocks):
            for name, module in (
                ("img_attn", block.img_attn_proj),
                ("img_mlp", block.img_mlp),
                ("txt_attn", block.txt_attn_proj),
                ("txt_mlp", block.txt_mlp),
            ):
                slot = f"double.{index}.{name}"
                self._handles.append(module.register_forward_hook(self._slot_hook(slot)))
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._double_forward(_index, module, *args, **kwargs)
            self._patches.append((block, _set_instance_forward(block, wrapped)))
        for index, block in enumerate(self.transformer.single_blocks):
            slot = f"single.{index}.total"
            self._handles.append(block.linear2.register_forward_hook(self._slot_hook(slot)))
            if self.source_history_compat:
                attn_slot = f"single.{index}.attn"
                self._handles.append(
                    block.linear2.register_forward_pre_hook(
                        self._single_attn_history_hook(attn_slot, block.hidden_size)
                    )
                )
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._single_forward(_index, module, *args, **kwargs)
            self._patches.append((block, _set_instance_forward(block, wrapped)))
        self._installed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        for module, state in reversed(self._patches):
            _restore_instance_forward(module, state)
        self._handles.clear()
        self._patches.clear()
        self._installed = False

    def _slot_hook(self, slot: str) -> Callable[..., None]:
        def hook(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            if self._current_full:
                self.method.update_slot(slot, output)
        return hook

    def _single_attn_history_hook(
        self,
        slot: str,
        hidden_size: int,
    ) -> Callable[..., None]:
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if self._current_full:
                # The slice is a view into cat(attn, mlp). Clone so one history
                # retains only the source-compatible attention tensor storage.
                self.method.update_slot(slot, args[0][..., :hidden_size].clone())

        return hook

    def _transformer_pre(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        self._block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        decision = self._current_decision
        if decision is None:
            raise RuntimeError("Taylor adapter did not enter the first double block")
        self.decisions.append(
            DecisionRecord(
                step=len(self.decisions),
                action="full" if decision.full else "cache",
                reason=decision.reason,
                original_block_calls=self._block_calls,
            )
        )
        return output

    def _double_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            self._current_decision = self.method.decide()
            self._current_full = self._current_decision.full
        if self._current_full:
            self._block_calls += 1
            return original(*args, **kwargs)

        img, txt, vec = args[:3]
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = module.img_mod(vec).chunk(6, dim=-1)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = module.txt_mod(vec).chunk(6, dim=-1)
        del img_shift1, img_scale1, img_shift2, img_scale2, txt_shift1, txt_scale1, txt_shift2, txt_scale2
        img = img + _apply_gate(self.method.predict_slot(f"double.{index}.img_attn"), img_gate1)
        img = img + _apply_gate(self.method.predict_slot(f"double.{index}.img_mlp"), img_gate2)
        txt = txt + _apply_gate(self.method.predict_slot(f"double.{index}.txt_attn"), txt_gate1)
        txt = txt + _apply_gate(self.method.predict_slot(f"double.{index}.txt_mlp"), txt_gate2)
        return img, txt

    def _single_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        offset = len(self.transformer.double_blocks) + index
        original = self._patches[offset][1][1]
        if self._current_full:
            self._block_calls += 1
            output = original(*args, **kwargs)
            if index == len(self.transformer.single_blocks) - 1:
                self.method.finish_full_step()
            return output
        x, vec = args[:2]
        _, _, gate = module.modulation(vec).chunk(3, dim=-1)
        return x + _apply_gate(self.method.predict_slot(f"single.{index}.total"), gate)


class HiCacheFineAdapter(TaylorSeerFineAdapter):
    """The Hunyuan fine slot layout shared by HiCache and TaylorSeer."""


@dataclass(frozen=True)
class L2PDecisionRecord:
    step: int
    action: str
    reason: str
    original_block_calls: int
    l2p_target: str
    l2p_weights_used: int
    l2p_current_step: int | None = None
    l2p_history_steps: list[int] | None = None
    l2p_weight_l1: float | None = None
    l2p_weight_l2: float | None = None
    l2p_weight_sum: float | None = None
    l2p_fallback_latest: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class L2POutputAdapter:
    """Skip Tencent blocks and predict the official final-layer output."""

    def __init__(self, transformer: Any, method: L2POutputMethod):
        self.transformer = transformer
        self.method = method
        self.decisions: list[L2PDecisionRecord] = []
        self._patches: list[tuple[Any, tuple[bool, Any]]] = []
        self._handles: list[Any] = []
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.method.reset()
        self.decisions.clear()
        self._current_decision: MethodDecision | None = None
        self._current_full = True
        self._block_calls = 0

    def __enter__(self) -> "L2POutputAdapter":
        if self._installed:
            raise RuntimeError("adapter already installed")
        if not self.transformer.double_blocks or not self.transformer.single_blocks:
            raise ValueError("HunyuanVideo L2P requires double and single blocks")
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._transformer_post, with_kwargs=True)
        )
        for index, block in enumerate(self.transformer.double_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._double_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))
        for index, block in enumerate(self.transformer.single_blocks):
            def wrapped(module: Any, *args: Any, _index: int = index, **kwargs: Any):
                return self._single_forward(_index, module, *args, **kwargs)

            self._patches.append((block, _set_instance_forward(block, wrapped)))

        final_layer = self.transformer.final_layer

        def final_forward(module: Any, *args: Any, **kwargs: Any):
            return self._final_forward(module, *args, **kwargs)

        self._patches.append((final_layer, _set_instance_forward(final_layer, final_forward)))
        self._installed = True
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
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
        self._block_calls = 0

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        decision = self._current_decision
        if decision is None:
            raise RuntimeError("L2P adapter did not enter the first double block")
        fields = dict(self.method.last_prediction_fields)
        self.decisions.append(
            L2PDecisionRecord(
                step=len(self.decisions),
                action="full" if self._current_full else "cache",
                reason=decision.reason,
                original_block_calls=self._block_calls,
                l2p_target=str(fields.pop("l2p_target")),
                l2p_weights_used=int(fields.pop("l2p_weights_used")),
                **fields,
            )
        )
        return output

    def _double_forward(self, index: int, module: Any, *args: Any, **kwargs: Any) -> Any:
        original = self._patches[index][1][1]
        if index == 0:
            self._current_decision = self.method.decide()
            self._current_full = self._current_decision.full
        if not self._current_full:
            return args[0], args[1]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _single_forward(self, index: int, _module: Any, *args: Any, **kwargs: Any) -> Any:
        offset = len(self.transformer.double_blocks) + index
        original = self._patches[offset][1][1]
        if not self._current_full:
            return args[0]
        self._block_calls += 1
        return original(*args, **kwargs)

    def _final_forward(self, _module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        final_index = len(self._patches) - 1
        original = self._patches[final_index][1][1]
        if self._current_full:
            return self.method.final_output(original(*args, **kwargs))
        return self.method.final_output()
