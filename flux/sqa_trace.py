"""Trace logger for the SQA E1 stale-gap experiment.

Runs a single full no-cache FLUX forward and records, per step `k`:

    psi_raw[k]:      raw modulated_inp from the first block's norm1, BEFORE
                     any SEA / Wiener filter is applied. This is the value
                     that flux/seacache.py:122 computes
                     (`modulated_inp = first_block.norm1(inp, emb=temb)[0]`),
                     unmodified.

    psi_filtered[k]: SEA / Wiener-filtered version of psi_raw[k], computed
                     on the fly using the same call signature flux/seacache.py
                     uses at lines 135-149 (apply_sea_with_scheduler with
                     power_exp=2.0, dims=(-2,-3), norm_mode='mean'). The
                     SeaCache hook itself is NOT installed; this module
                     replicates the filter math without enabling any caching.

    residual[k]:     whole-transformer residual r_k = h_out - h_in_post_xembed
                     at step k, exactly what flux/seacache.py:249 stores into
                     `previous_residual` on a full step. This is the object
                     the SQA E1 synthetic stale-memory action uses as R_a
                     when forking branch A at a future step n.

Plus the final packed latent `z_N` (returned from the outer `pipe(...)` call,
not captured by the hook).

Tensors are detached and moved to CPU (bf16) inside the hook so a 50-step
trace on FLUX 1024^2 occupies ~3.6 GB CPU RAM per prompt and does not
accumulate GPU memory across steps. Callers process one prompt at a time
and discard the trace before moving on.

Public surface:
    install_trace(pipe, *, num_steps) -> teardown
        Patches FluxTransformer2DModel.forward to record into
        `pipe.transformer.sqa_trace` (dict of lists). Returns a teardown
        callable. Idempotent.

    pop_trace(pipe) -> dict
        Detaches `pipe.transformer.sqa_trace`, replaces the on-instance
        attribute with empty buffers, and returns the captured dict.

Side-effects: this module replaces `FluxTransformer2DModel.forward` with a
trace-aware variant while installed. Teardown restores the original. The
recorded forward is otherwise byte-equivalent to the diffusers stock
forward (modulo the SEA filter being recomputed for storage only — it
does NOT feed back into the live forward path).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

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

from lib.wiener import apply_sea_with_scheduler

logger = logging.get_logger(__name__)


def _trace_forward(
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
    """Drop-in replacement for `FluxTransformer2DModel.forward` that records
    (psi_raw, psi_filtered, residual) per step into `self.sqa_trace`.

    The live forward path is unchanged: this is a recording wrapper, not a
    cache gate. All recorded tensors are detached + moved to CPU bf16 so the
    transformer state does not accumulate across steps.
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

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ---- Trace recording (BEFORE block stack runs) --------------------------
    # psi_raw at this step. Matches the SeaCache modulated-input derivation
    # in flux/seacache.py byte-for-byte (first_block.norm1 on the
    # post-x_embedder hidden state).
    record_step = -1
    if getattr(self, "enable_sqa_trace", False):
        record_step = int(self.cnt)
        first_block = self.transformer_blocks[0]
        psi_raw, _gm, _sm, _sm2, _gm2 = first_block.norm1(hidden_states, emb=temb)

        # psi_filtered: reshape → apply SEA filter → reshape back.
        # Matches the non-force-full SEA-filter call in
        # flux/seacache.py:_seacache_forward (SEA reshape → apply_sea_with_scheduler
        # → reshape-back, same kwargs).
        # NOTE: flux/seacache.py only runs this on the non-force-full branch
        # (cnt >= first_enhance and cnt > 0 and cnt < num_steps-1). For the
        # trace, we always compute it: callers in lib/sqa_replay.py decide
        # which step's filtered value to use (anchor uses RAW, window uses
        # FILTERED — see lib/sqa_replay.py docstring).
        # NOTE 2: at k=0, lib/wiener.py:ab_from_scheduler clamps sigma → ~1,
        # making the filter ≈ 0 numerically. psi_filtered[0] is therefore
        # degenerate; the E1 action grid (docs/research_plan_method_native_sqa.md
        # §6 E1) only accesses psi_filtered[a+1:n+1] with a >= 0, so
        # psi_filtered[0] is never read by E1. Stored for completeness only.
        psi_for_filter = psi_raw.reshape(
            psi_raw.shape[0],
            int(img_ids[:, 1].max().item() + 1),
            int(img_ids[:, 2].max().item() + 1),
            psi_raw.shape[-1],
        )
        psi_for_filter = apply_sea_with_scheduler(
            psi_for_filter,
            self.scheduler,
            int(self.cnt),
            power_exp=2.0,
            dims=(-2, -3),
            norm_mode="mean",
        )
        psi_filtered = psi_for_filter.reshape(
            psi_for_filter.shape[0], -1, psi_for_filter.shape[-1]
        )

        self.sqa_trace["psi_raw"].append(
            psi_raw.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        )
        self.sqa_trace["psi_filtered"].append(
            psi_filtered.detach().to("cpu", dtype=torch.bfloat16).contiguous()
        )

        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for next trajectory; pop_trace before reuse

    # ---- Block stack (verbatim from flux/seacache.py:170-247) ---------------
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

    # ---- Trace recording (AFTER block stack: residual r_k) ------------------
    if record_step >= 0:
        # residual = whole-transformer output minus pre-block hidden state.
        # Matches the previous_residual update in flux/seacache.py
        # (`previous_residual = hidden_states - ori_hidden_states` on full
        # steps).
        residual = (hidden_states - ori_hidden_states).detach()
        self.sqa_trace["residual"].append(
            residual.to("cpu", dtype=torch.bfloat16).contiguous()
        )

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----- install / teardown / pop ----------------------------------------------


def install_trace(
    pipe,
    *,
    num_steps: int,
) -> Callable[[], None]:
    """Patch `FluxTransformer2DModel.forward` so that every step records into
    `pipe.transformer.sqa_trace`. Returns a teardown callable.

    The pipeline `pipe.scheduler` must be set (used by apply_sea_with_scheduler
    to derive the SEA filter coefficients per step).

    Args:
        pipe: a loaded `DiffusionPipeline` / `FluxPipeline`.
        num_steps: total sampling steps in this trajectory. Used to reset the
            step counter at end-of-trajectory so the same hook can be reused
            across prompts. **It does NOT reset the recorded buffers.**

    Returns:
        teardown: a zero-arg callable that restores the original forward
        and clears the per-instance state. Idempotent.

    **Caller contract — important:** between successive `pipe(...)` calls,
    the caller MUST drain the captured trace via either `pop_trace(pipe)`
    (preferred, returns the dict and re-initializes the buffer) or
    `reset_trace(pipe)` (drops the dict). Otherwise the per-key lists grow
    unbounded across prompts (each prompt appends `num_steps` entries, so
    e.g. 10 prompts × 50 steps × 3 keys × ~25 MB/tensor ≈ 36 GB CPU RAM).
    The end-of-trajectory counter wraparound at the end of `_trace_forward`
    is only for counter sanity; it does not free buffers.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _trace_forward

    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_sqa_trace = True
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.sqa_trace = _empty_trace()

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_sqa_trace", "num_steps", "cnt", "sqa_trace", "scheduler",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_trace(pipe) -> None:
    """Reset the trace counter + buffers before each new prompt. Call between
    `pipe()` invocations if you reuse the same install across prompts."""
    tr = pipe.transformer
    tr.cnt = 0
    tr.sqa_trace = _empty_trace()


def pop_trace(pipe) -> Dict[str, List[torch.Tensor]]:
    """Detach and return the captured trace, then re-initialize the buffer.

    Returns:
        Dict with keys `psi_raw`, `psi_filtered`, `residual`. Each is a list
        of length `num_steps`, indexed by step `k` (0-based), holding CPU
        bf16 tensors with the same shape as the corresponding per-step
        embedded tensor (typically `(1, S_img, hidden_dim)`).
    """
    tr = pipe.transformer
    captured = tr.sqa_trace
    tr.sqa_trace = _empty_trace()
    tr.cnt = 0
    return captured


def _empty_trace() -> Dict[str, List[torch.Tensor]]:
    return {
        "psi_raw": [],
        "psi_filtered": [],
        "residual": [],
    }
