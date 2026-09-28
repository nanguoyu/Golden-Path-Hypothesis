"""L2P final-hidden predictor for diffusers FLUX.

This is the paper-faithful coarse L2P target: the cached object is the final
image-token hidden feature after all transformer blocks and before
``norm_out + proj_out``.  It is intentionally not a residual-payload variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from lib.gates import IntervalGate
from lib.l2p import append_history, load_l2p_weight_file, predict_l2p

logger = logging.get_logger(__name__)


def _decide_l2p_step(self, history: Dict[int, torch.Tensor]) -> dict[str, Any]:
    gate: IntervalGate = self._l2p_gate
    action_steps: Optional[set[int]] = self._l2p_action_steps
    schedule_u: Optional[int] = None
    force_full_reason: Optional[str] = None

    if action_steps is None:
        should_skip = gate.decide()
        step = gate.cnt - 1
        if should_skip and not history:
            should_skip = False
            force_full_reason = "history_unready"
            gate.last_activated = step
            gate.activated_steps.append(step)
    else:
        step = gate.cnt
        schedule_u = int(step in action_steps)
        should_skip = bool(schedule_u and history)
        if schedule_u and not history:
            force_full_reason = "history_unready"
        prev_full = gate.last_activated if gate.activated_steps else None
        if not should_skip:
            gate.last_activated = step
            gate.activated_steps.append(step)
        gate.cnt += 1
        if not should_skip and prev_full is not None:
            self._l2p_step_gap = int(step - prev_full)
        else:
            self._l2p_step_gap = 1

    if action_steps is None:
        if not should_skip and len(gate.activated_steps) >= 2:
            self._l2p_step_gap = int(gate.activated_steps[-1] - gate.activated_steps[-2])
        else:
            self._l2p_step_gap = 1

    step_offset = int(step - gate.last_activated)
    return {
        "step": int(step),
        "u": int(should_skip),
        "schedule_u": schedule_u,
        "force_full": bool(not should_skip),
        "force_full_reason": force_full_reason,
        "history_ready": bool(history),
        "history_size": int(len(history)),
        "step_offset": int(step_offset),
        "step_gap": int(getattr(self, "_l2p_step_gap", 1)),
        "interval": int(gate.interval),
        "first_enhance": int(gate.first_enhance),
        "schedule_locked": action_steps is not None,
    }


def _l2p_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """Drop-in replacement for ``FluxTransformer2DModel.forward`` with L2P."""

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids is not None and txt_ids.ndim == 3:
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        img_ids = img_ids[0]

    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    history: Dict[int, torch.Tensor] = self._l2p_final_hidden_history
    decision = _decide_l2p_step(self, history)

    if int(decision["u"]) == 1:
        hidden_states, fields = predict_l2p(
            history,
            self._l2p_weights,
            current_step=int(decision["step"]),
            min_abs_weight=float(self._l2p_min_abs_weight),
        )
        self._l2p_final_hidden_history = append_history(
            history,
            int(decision["step"]),
            hidden_states,
        )
        decision.update(fields)
        decision["l2p_target"] = "final_hidden"
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_block_samples is not None:
                interval_control = int(
                    np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
                )
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt2(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt2(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = int(
                    np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples))
                )
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        self._l2p_final_hidden_history = append_history(
            history,
            int(decision["step"]),
            hidden_states,
        )
        decision["l2p_target"] = "final_hidden"

    self.l2p_decisions.append(decision)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install(
    pipe,
    *,
    weights_path: str | Path,
    interval: int = 7,
    first_enhance: int = 3,
    num_steps: int,
    min_abs_weight: float = 0.0,
) -> Callable[[], None]:
    """Patch ``FluxTransformer2DModel.forward`` and attach L2P state."""
    meta = load_l2p_weight_file(weights_path, num_steps=num_steps)
    target = str(meta.get("target", "final_hidden"))
    if target not in ("final_hidden", "final_layer", "hidden_states"):
        raise ValueError(f"L2P coarse expects final-hidden weights, got target={target!r}")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _l2p_forward

    tr = pipe.transformer
    tr._l2p_gate = IntervalGate(
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
    )
    tr._l2p_final_hidden_history = {}
    tr._l2p_weights = meta["weights"]
    tr._l2p_weights_meta = {k: v for k, v in meta.items() if k != "weights"}
    tr._l2p_min_abs_weight = float(min_abs_weight)
    tr._l2p_action_steps = None
    tr._l2p_step_gap = 1
    tr.l2p_decisions = []

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "_l2p_gate",
            "_l2p_final_hidden_history",
            "_l2p_weights",
            "_l2p_weights_meta",
            "_l2p_min_abs_weight",
            "_l2p_action_steps",
            "_l2p_step_gap",
            "l2p_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(
    pipe,
    *,
    action_steps: Optional[set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    """Reset L2P trajectory state before each new prompt."""
    tr = pipe.transformer
    tr._l2p_gate.reset()
    tr._l2p_final_hidden_history = {}
    tr._l2p_action_steps = None if action_steps is None else set(int(s) for s in action_steps)
    tr._l2p_step_gap = 1
    tr.l2p_decisions = []
