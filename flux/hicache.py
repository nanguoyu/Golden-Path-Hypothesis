"""HiCache on diffusers `FluxTransformer2DModel` (transformer-residual variant).

Clean diffusers re-implementation of HiCache (arXiv:2508.16984). The
predictor caches the **whole-transformer residual** at activation steps and,
on skip steps, extrapolates it via the dual-scaled Hermite formula
(paper Def. 2):

    F_pred(x) = F_0 + sum_{k=1..O} (H_k(sigma * x) / k!) * sigma^k * Delta^k F

where the Delta^k history is over RESIDUALs `hidden_states_out - hidden_states_in`.

## Granularity note

The HiCache paper caches features at a finer granularity (per (block_index,
sub_module, stream)) by editing inside each transformer block. The diffusers
`FluxTransformer2DModel.forward` signature does not naturally admit a
per-block cache dict, so this file uses the **same** granularity as SeaCache
(whole-transformer residual). That makes the implementation small,
monkey-patch-only, and reversible, at the cost of slightly weaker quality at
a matched speedup tier vs. the per-block scheme. For paper-exact reproduction,
run the upstream HiCache code directly from `reference/hicache/code/` (git
submodule); do not import from it.

## Operating points

  - `interval=7, max_order=2, sigma=0.5, first_enhance=3` matches the HiCache
    paper Table 1 setting and gives roughly the same speedup (~5×).
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

from lib.gates import IntervalGate
from lib.hermite import hermite_update, hicache_predict

logger = logging.get_logger(__name__)


def _hicache_forward(
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
    """Drop-in replacement for `FluxTransformer2DModel.forward` with HiCache."""

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

    # ---- IP-Adapter: project image_embeds → hidden_states (mirrors diffusers 0.38.0
    # FluxTransformer2DModel.forward and TeaCache official patch). Without this the
    # full-forward path silently drops IP-Adapter image conditioning.
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ---- HiCache gating: IntervalGate decides should_skip for THIS step ------
    gate: IntervalGate = self._hicache_gate
    should_skip = gate.decide()
    history: Dict[int, torch.Tensor] = self._hicache_residual_history

    if should_skip and history:
        # Predict residual via Hermite (degenerates to history[0] if order_avail < 1)
        residual = hicache_predict(
            history,
            step_offset=gate.step_offset,
            sigma=float(self._hicache_sigma),
            max_order=int(self._hicache_max_order),
        )
        hidden_states = hidden_states + residual
    else:
        # Full forward through all transformer blocks
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

        # Update Delta^k residual history. step_gap is the integer step distance
        # between the previous and current activation; for the very first
        # activation it doesn't matter (hermite_update returns {0: r}).
        residual = hidden_states - ori_hidden_states
        if len(gate.activated_steps) >= 2:
            step_gap = gate.activated_steps[-1] - gate.activated_steps[-2]
        else:
            step_gap = 1

        # Warmup guard: during the `first_enhance` warmup steps we only store
        # the zeroth-order term (F_0). This matches the upstream HiCache
        # behavior (`derivative_approximation` only computes Delta^k for k>=1
        # when `step > first_enhance - 2`). Without this guard our predictor
        # would have a richer history than upstream at the first skip step,
        # silently producing different numbers.
        if gate.cnt < gate.first_enhance:
            effective_max_order = 0
        else:
            effective_max_order = int(self._hicache_max_order)

        self._hicache_residual_history = hermite_update(
            history, residual, step_gap=step_gap, max_order=effective_max_order
        )

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----- install / teardown / reset --------------------------------------------


def install(
    pipe,
    *,
    interval: int = 7,
    max_order: int = 2,
    sigma: float = 0.5,
    first_enhance: int = 3,
    num_steps: int,
) -> Callable[[], None]:
    """Patch `FluxTransformer2DModel.forward` and attach HiCache state.

    Args:
        pipe:          a loaded `DiffusionPipeline` / `FluxPipeline`.
        interval:      refresh every N steps after warmup (paper Table 1: 7).
        max_order:     Hermite truncation order O (paper Table 1: 2).
        sigma:         dual-scaling factor in `H_k(sigma*x)` (paper: 0.5).
        first_enhance: count of leading full-forward steps (paper: 3).
        num_steps:     total sampling steps; the last step is force-full (mirrors
                       BFL `cal_type`'s `step >= num_steps - 1 -> full` rule).

    Returns:
        teardown: zero-arg callable restoring the original forward and clearing state.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _hicache_forward

    tr = pipe.transformer
    tr._hicache_gate = IntervalGate(
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
    )
    tr._hicache_residual_history = {}
    tr._hicache_sigma = float(sigma)
    tr._hicache_max_order = int(max_order)

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "_hicache_gate",
            "_hicache_residual_history",
            "_hicache_sigma",
            "_hicache_max_order",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(pipe) -> None:
    """Reset HiCache trajectory state before each new prompt."""
    tr = pipe.transformer
    tr._hicache_gate.reset()
    tr._hicache_residual_history = {}
