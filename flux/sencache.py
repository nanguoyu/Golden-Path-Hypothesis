"""SenCache-style whole-transformer residual cache for diffusers FLUX.

This is a FLUX adaptation of SenCache's sensitivity-aware gate.  The official
SenCache code uses the model input latent ``z_t`` and a frozen sensitivity table:

    Lambda = J_x(anchor_t) * ||z_t - z_anchor||_2
           + J_t(anchor_t) * |t - anchor_t|.

FLUX diffusers exposes packed latent tokens as ``hidden_states`` before
``x_embedder``.  We use that tensor as the FLUX analogue of ``z_t`` and keep the
cached quantity consistent with SeaCache/TeaCache in this repo: the
whole-transformer residual after ``x_embedder``.
"""

from __future__ import annotations

import math
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

from lib.sencache import load_sensitivity_table, online_fields, threshold_scale_from_latent

logger = logging.get_logger(__name__)


def _sencache_forward(
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

    latent_for_gate = hidden_states.detach()
    timestep_for_gate = float(timestep.detach().to(torch.float32).reshape(-1)[0].item() * 1000.0)

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
    force_full = False
    force_full_reason = None
    score = None
    threshold = None
    threshold_raw = None
    cache_allowed = False
    online = {}
    if getattr(self, "enable_sencache", False):
        cnt = int(getattr(self, "cnt", 0))
        n_steps = int(getattr(self, "num_steps", 0))
        cutoff_arg = int(getattr(self, "sencache_cutoff_steps", -1))
        cutoff_step = (n_steps - 1) if cutoff_arg < 0 else cutoff_arg
        force_full = (
            cnt < int(getattr(self, "first_enhance", 1))
            or cnt == 0
            or cnt >= max(min(cutoff_step, n_steps), 0)
            or getattr(self, "sencache_anchor_latent", None) is None
            or getattr(self, "previous_residual", None) is None
        )
        if cnt < int(getattr(self, "first_enhance", 1)) or cnt == 0:
            force_full_reason = "warmup"
        elif cnt >= max(min(cutoff_step, n_steps), 0):
            force_full_reason = "cutoff"
        elif getattr(self, "sencache_anchor_latent", None) is None:
            force_full_reason = "no_anchor"
        elif getattr(self, "previous_residual", None) is None:
            force_full_reason = "no_previous_residual"

        if getattr(self, "sencache_threshold_scale", None) is None:
            self.sencache_threshold_scale = threshold_scale_from_latent(
                latent_for_gate, getattr(self, "sencache_threshold_scale_arg", "auto")
            )
        online = online_fields(
            table=getattr(self, "sencache_table", None),
            current_latent=latent_for_gate,
            current_timestep=timestep_for_gate,
            anchor_latent=getattr(self, "sencache_anchor_latent", None),
            anchor_timestep=getattr(self, "sencache_anchor_timestep", None),
            anchor_step=getattr(self, "sencache_anchor_step", None),
        )
        threshold_raw = (
            float(getattr(self, "sencache_thresh_start", 0.0))
            if cnt < int(round(n_steps * float(getattr(self, "sencache_switch_ratio", 0.2))))
            else float(getattr(self, "sencache_thresh_main", 0.0))
        )
        threshold = float(threshold_raw * float(getattr(self, "sencache_threshold_scale", 1.0)))
        score = online.get("online_sencache_score_pre")
        if force_full:
            should_calc = True
            self.sencache_accumulated_skips = 0
        else:
            cache_allowed = (
                cnt >= int(getattr(self, "sencache_ret_steps", 0))
                and score is not None
                and float(score) < threshold
                and int(getattr(self, "sencache_accumulated_skips", 0)) < int(getattr(self, "sencache_K", 10))
            )
            should_calc = not cache_allowed

    if (
        getattr(self, "enable_sencache", False)
        and not should_calc
        and (self.previous_residual is not None)
    ):
        hidden_states = hidden_states + self.previous_residual
        self.sencache_accumulated_skips = int(getattr(self, "sencache_accumulated_skips", 0)) + 1
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
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
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
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        if getattr(self, "enable_sencache", False):
            self.previous_residual = hidden_states - ori_hidden_states
            self.sencache_anchor_latent = latent_for_gate.detach().clone()
            self.sencache_anchor_timestep = float(timestep_for_gate)
            self.sencache_anchor_step = int(getattr(self, "cnt", 0))
            self.sencache_accumulated_skips = 0

    if getattr(self, "enable_sencache", False):
        cnt = int(getattr(self, "cnt", 0))
        if hasattr(self, "sencache_decisions"):
            self.sencache_decisions.append({
                "step": cnt,
                "u": int(not should_calc),
                "score": None if score is None else float(score),
                "threshold_raw": None if threshold_raw is None else float(threshold_raw),
                "threshold_scale": float(getattr(self, "sencache_threshold_scale", 1.0)),
                "threshold": None if threshold is None else float(threshold),
                "cache_allowed": bool(cache_allowed),
                "force_full": bool(force_full),
                "force_full_reason": force_full_reason,
                "sencache_sensitivity_sha256": getattr(self, "sencache_table_sha256", None),
                "cutoff_step": int((int(getattr(self, "num_steps", 0)) - 1)
                                   if int(getattr(self, "sencache_cutoff_steps", -1)) < 0
                                   else int(getattr(self, "sencache_cutoff_steps", -1))),
                "consecutive_skips": int(getattr(self, "sencache_accumulated_skips", 0)),
                **online,
            })
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

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
    sensitivity_path: str,
    threshold_start: float,
    threshold_main: float,
    num_steps: int,
    first_enhance: int = 1,
    max_skip: int = 10,
    threshold_scale: str | float | int | None = "auto",
    switch_ratio: float = 0.2,
    ret_steps: int = 0,
    cutoff_steps: int = -1,
) -> Callable[[], None]:
    table = load_sensitivity_table(sensitivity_path)
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _sencache_forward

    tr = pipe.transformer
    tr.enable_sencache = True
    tr.sencache_table = table
    tr.sencache_table_sha256 = table.sha256
    tr.sencache_table_metadata = dict(table.metadata)
    tr.sencache_thresh_start = float(threshold_start)
    tr.sencache_thresh_main = float(threshold_main)
    tr.sencache_K = int(max_skip)
    tr.sencache_threshold_scale_arg = threshold_scale
    tr.sencache_threshold_scale = None
    tr.sencache_switch_ratio = float(switch_ratio)
    tr.sencache_ret_steps = int(ret_steps)
    tr.sencache_cutoff_steps = int(cutoff_steps)
    tr.num_steps = int(num_steps)
    tr.first_enhance = int(first_enhance)
    tr.cnt = 0
    tr.previous_residual = None
    tr.sencache_anchor_latent = None
    tr.sencache_anchor_timestep = None
    tr.sencache_anchor_step = None
    tr.sencache_accumulated_skips = 0
    tr.sencache_decisions = []

    done = {"v": False}

    def teardown() -> None:
        if done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_sencache", "sencache_table", "sencache_thresh_start",
            "sencache_table_sha256", "sencache_table_metadata",
            "sencache_thresh_main", "sencache_K", "sencache_threshold_scale_arg",
            "sencache_threshold_scale", "sencache_switch_ratio", "sencache_ret_steps",
            "sencache_cutoff_steps", "num_steps", "first_enhance", "cnt",
            "previous_residual", "sencache_anchor_latent", "sencache_anchor_timestep",
            "sencache_anchor_step", "sencache_accumulated_skips", "sencache_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        done["v"] = True

    return teardown


def reset_per_image_state(pipe) -> None:
    tr = pipe.transformer
    tr.cnt = 0
    tr.previous_residual = None
    tr.sencache_anchor_latent = None
    tr.sencache_anchor_timestep = None
    tr.sencache_anchor_step = None
    tr.sencache_accumulated_skips = 0
    tr.sencache_decisions = []
