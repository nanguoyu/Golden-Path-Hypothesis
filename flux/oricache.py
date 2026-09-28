"""OriCache on diffusers `FluxTransformer2DModel`.

Research-only reproduction of "OriCache: Orientation-Guided Feature Caching
for DiT Acceleration" on the same coarse whole-transformer residual surface as
the locked FLUX SeaCache / TeaCache baselines in this repo.

OriCache keeps the TeaCache-style accumulated threshold gate, but replaces the
per-step feature-distance increment with a normalized local-curvature score:

    s_t = rho_t^2 + 1 - 2 rho_t cos(theta_t)

where rho_t = ||dz_t|| / ||dz_{t-1}|| and cos(theta_t) measures alignment
between consecutive block-input updates. Up to numerical eps terms, this is
||dz_t - dz_{t-1}||^2 / ||dz_{t-1}||^2.
"""

from __future__ import annotations

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

logger = logging.get_logger(__name__)

EPS = 1e-12
ORICACHE_SIGNALS = ("modulated", "raw")


def _vector_stats(delta: torch.Tensor, prev_delta: torch.Tensor) -> Dict[str, float]:
    cur = delta.detach().to(torch.float32)
    prev = prev_delta.detach().to(torch.float32)
    cur_flat = cur.reshape(-1)
    prev_flat = prev.reshape(-1)
    cur_norm = torch.linalg.vector_norm(cur_flat)
    prev_norm = torch.linalg.vector_norm(prev_flat)
    diff_norm = torch.linalg.vector_norm(cur_flat - prev_flat)
    denom = cur_norm * prev_norm + EPS
    cos_theta = torch.dot(cur_flat, prev_flat) / denom
    rho = cur_norm / (prev_norm + EPS)
    score = rho * rho + 1.0 - 2.0 * rho * cos_theta
    return {
        "oricache_score": float(score.item()),
        "oricache_rho": float(rho.item()),
        "oricache_cos_theta": float(cos_theta.item()),
        "oricache_delta_norm": float(cur_norm.item()),
        "oricache_prev_delta_norm": float(prev_norm.item()),
        "oricache_update_diff_norm": float(diff_norm.item()),
        "oricache_score_alt": float(((diff_norm * diff_norm) / (prev_norm * prev_norm + EPS)).item()),
    }


def _oricache_forward(
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
    """Drop-in replacement for `FluxTransformer2DModel.forward` with OriCache."""
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

    should_calc = True
    force_full_reason: Optional[str] = None
    score_fields: Dict[str, Optional[float]] = {
        "oricache_score": None,
        "oricache_rho": None,
        "oricache_cos_theta": None,
        "oricache_delta_norm": None,
        "oricache_prev_delta_norm": None,
        "oricache_update_diff_norm": None,
        "oricache_score_alt": None,
    }
    accumulator_before = float(getattr(self, "accumulated_oricache_score", 0.0))
    accumulator_after_increment = accumulator_before

    if getattr(self, "enable_oricache", False):
        signal_kind = str(getattr(self, "oricache_signal", "modulated"))
        if signal_kind == "modulated":
            first_block = self.transformer_blocks[0]
            current_signal, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = first_block.norm1(
                hidden_states, emb=temb
            )
        elif signal_kind == "raw":
            current_signal = hidden_states
        else:
            raise RuntimeError(f"unknown OriCache signal: {signal_kind}")

        current_signal = current_signal.detach()
        previous_signal = getattr(self, "previous_oricache_signal", None)
        previous_update = getattr(self, "previous_oricache_update", None)
        current_update = None if previous_signal is None else current_signal - previous_signal

        if self.cnt < int(getattr(self, "first_enhance", 2)):
            force_full_reason = "first_enhance"
        elif self.cnt == 0:
            force_full_reason = "step0"
        elif self.cnt == self.num_steps - 1:
            force_full_reason = "final_step"
        elif previous_signal is None:
            force_full_reason = "no_previous_signal"
        elif previous_update is None or current_update is None:
            force_full_reason = "no_previous_update"
        elif self.previous_residual is None:
            force_full_reason = "no_previous_residual"

        if force_full_reason is not None:
            should_calc = True
            self.accumulated_oricache_score = 0.0
            accumulator_after_increment = 0.0
        else:
            score_fields = _vector_stats(current_update, previous_update)
            accumulator_after_increment = accumulator_before + float(score_fields["oricache_score"])
            if accumulator_after_increment < float(self.oricache_thresh):
                should_calc = False
                self.accumulated_oricache_score = float(accumulator_after_increment)
            else:
                should_calc = True
                self.accumulated_oricache_score = 0.0

        self.previous_oricache_signal = current_signal
        if current_update is not None:
            self.previous_oricache_update = current_update.detach()

        if hasattr(self, "oricache_decisions"):
            self.oricache_decisions.append({
                "step": int(self.cnt),
                "u": int(not should_calc),
                "threshold": float(self.oricache_thresh),
                "signal": signal_kind,
                "force_full": bool(force_full_reason is not None),
                "force_full_reason": force_full_reason,
                "accumulator_before": float(accumulator_before),
                "accumulator_after_increment": float(accumulator_after_increment),
                "accumulator_after_commit": float(self.accumulated_oricache_score),
                **score_fields,
                "previous_signal_present": previous_signal is not None,
                "previous_update_present": previous_update is not None,
                "previous_residual_present": self.previous_residual is not None,
            })
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    if (
        getattr(self, "enable_oricache", False)
        and not should_calc
        and (self.previous_residual is not None)
    ):
        hidden_states = hidden_states + self.previous_residual
    else:
        ori_hidden_states = hidden_states
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

        if getattr(self, "enable_oricache", False):
            self.previous_residual = hidden_states - ori_hidden_states

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
    threshold: float,
    num_steps: int,
    first_enhance: int = 2,
    signal: str = "modulated",
) -> Callable[[], None]:
    """Patch `FluxTransformer2DModel.forward` and attach OriCache state."""
    if signal not in ORICACHE_SIGNALS:
        raise ValueError(f"signal must be one of {ORICACHE_SIGNALS}; got {signal!r}")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _oricache_forward

    tr = pipe.transformer
    tr.enable_oricache = True
    tr.oricache_thresh = float(threshold)
    tr.oricache_signal = str(signal)
    tr.num_steps = int(num_steps)
    tr.first_enhance = int(first_enhance)
    tr.cnt = 0
    tr.accumulated_oricache_score = 0.0
    tr.previous_oricache_signal = None
    tr.previous_oricache_update = None
    tr.previous_residual = None
    tr.oricache_decisions = []

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_oricache", "oricache_thresh", "oricache_signal",
            "num_steps", "first_enhance", "cnt",
            "accumulated_oricache_score", "previous_oricache_signal",
            "previous_oricache_update", "previous_residual",
            "oricache_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(pipe) -> None:
    """Reset OriCache trajectory state before each prompt."""
    tr = pipe.transformer
    tr.cnt = 0
    tr.accumulated_oricache_score = 0.0
    tr.previous_oricache_signal = None
    tr.previous_oricache_update = None
    tr.previous_residual = None
    tr.oricache_decisions = []
