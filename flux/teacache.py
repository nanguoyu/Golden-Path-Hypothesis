"""TeaCache on diffusers `FluxTransformer2DModel`.

Clean re-implementation of TeaCache (arXiv:2411.19108) for FLUX. Structurally
near-identical to `flux/seacache.py`; the only mathematical difference is the
**rescaling function** between the per-step relative-L1 distance and the
accumulator:

  - SeaCache: spectral Wiener filter on the **tensor** modulated input
              before computing rel_L1.
  - TeaCache: degree-4 **polynomial** on the **scalar** rel_L1 value.

The polynomial is per-backbone (fit on ~70 calibration prompts upstream).
Coefficients live in `lib/teacache_coeffs.py`; the default here is FLUX.1-dev.

State protocol mirrors SeaCache for ease of comparison: same attribute names
(`enable_teacache`, `teacache_thresh`, `cnt`, `accumulated_rel_l1_distance`,
`previous_modulated_input`, `previous_residual`, `num_steps`, `first_enhance`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

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

from lib.gates import rel_l1
from lib.teacache_coeffs import get_coeffs

logger = logging.get_logger(__name__)


def _teacache_forward(
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
    """Drop-in replacement for `FluxTransformer2DModel.forward` with TeaCache.

    State read from `self` (attached by `install`):
        teacache_thresh   (float)     accumulated-distance trigger (paper δ)
        teacache_rescale  (np.poly1d) per-backbone scalar rescaler
        num_steps         (int)       total sampling steps in this trajectory
        first_enhance     (int)       force-full for the first N steps
        cnt               (int)       step counter, advanced inside forward
        accumulated_rel_l1_distance (float)
        previous_modulated_input (Tensor or None)
        previous_residual           (Tensor or None)
    """
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

    # ---- TeaCache gating ------------------------------------------------------
    should_calc = True
    if getattr(self, "enable_teacache", False):
        inp = hidden_states
        first_block = self.transformer_blocks[0]
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = first_block.norm1(inp, emb=temb)

        force_full = (
            self.cnt < int(getattr(self, "first_enhance", 1))
            or self.cnt == 0
            or self.cnt == self.num_steps - 1
            or self.previous_modulated_input is None
        )
        if force_full:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            # TeaCache rescaling: polynomial f(L1_rel) on the scalar value.
            # Upstream computes rel_L1 as mean|delta| / mean|prev| (no eps);
            # our lib.gates.rel_l1 has a small eps for numerical safety, which
            # is irrelevant in practice (|prev| is never numerically zero in
            # FLUX) and matches upstream values to machine precision.
            d = rel_l1(modulated_inp, self.previous_modulated_input)
            self.accumulated_rel_l1_distance += float(self.teacache_rescale(d))
            if self.accumulated_rel_l1_distance < float(self.teacache_thresh):
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0

        self.previous_modulated_input = modulated_inp
        if hasattr(self, "teacache_decisions"):
            self.teacache_decisions.append({
                "step": int(self.cnt),
                "u": int(not should_calc),
                "accumulated_rel_l1_distance": float(self.accumulated_rel_l1_distance),
                "threshold": float(self.teacache_thresh),
                "force_full": bool(force_full),
            })
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0  # ready for the next trajectory

    # ---- Main block compute / skip --------------------------------------------
    if (
        getattr(self, "enable_teacache", False)
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

        if getattr(self, "enable_teacache", False):
            self.previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----- install / teardown ----------------------------------------------------


def install(
    pipe,
    *,
    threshold: float,
    num_steps: int,
    first_enhance: int = 1,
    backbone: str = "flux",
    variant: Optional[str] = None,
    coefficients: Optional[Sequence[float]] = None,
) -> Callable[[], None]:
    """Patch `FluxTransformer2DModel.forward` and attach per-instance TeaCache state.

    Args:
        pipe:          a loaded `DiffusionPipeline` / `FluxPipeline`.
        threshold:     TeaCache distance threshold delta. **Backbone-specific scale!**
                       For FLUX: upstream README says 0.25→1.5×, 0.4→1.8×, 0.6→2×,
                       0.8→2.25×. SeaCache paper Table 1 uses 0.3 / 0.6 for the
                       two budget tiers. The video-task 0.1 / 0.2 thresholds in
                       the TeaCache paper text DO NOT transfer to FLUX (the FLUX
                       polynomial's intercept f(0)≈0.26 means thresh<=0.2
                       triggers every-step refresh → zero speedup).
        num_steps:     total sampling steps in this trajectory.
        first_enhance: count of leading full-forward steps. Upstream uses 1
                       (only `cnt==0` forced); paper TeaCache experiments use
                       no warmup region beyond that. Set higher to match
                       HiCache's `first_enhance >= 3` convention.
        backbone:      key into `lib.teacache_coeffs._COEFFS`. Default `"flux"`.
                       Ignored if `coefficients` is given.
        variant:       optional variant key (e.g. `"1.3b"` for wan21). Ignored
                       if `coefficients` is given.
        coefficients:  explicit polynomial coefficients (numpy `poly1d` order,
                       highest-degree first). When provided, overrides the
                       lookup in `lib.teacache_coeffs`.

    Returns:
        teardown: a zero-arg callable that restores the original forward and
                  clears the per-instance state. Idempotent.
    """
    if coefficients is None:
        coefficients = get_coeffs(backbone, variant=variant)
    rescale_fn = np.poly1d(list(coefficients))

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _teacache_forward

    tr = pipe.transformer
    tr.enable_teacache = True
    tr.teacache_thresh = float(threshold)
    tr.teacache_rescale = rescale_fn
    tr.num_steps = int(num_steps)
    tr.first_enhance = int(first_enhance)
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    tr.teacache_decisions = []

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_teacache", "teacache_thresh", "teacache_rescale",
            "num_steps", "first_enhance",
            "cnt", "accumulated_rel_l1_distance",
            "previous_modulated_input", "previous_residual",
            "teacache_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(pipe) -> None:
    """Reset TeaCache trajectory state before each new prompt. Call between
    `pipe()` invocations to avoid carrying counters / residuals across images."""
    tr = pipe.transformer
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    tr.teacache_decisions = []
