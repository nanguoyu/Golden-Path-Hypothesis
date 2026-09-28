"""Fine-grained cache scaffold for diffusers FLUX (per-block × per-sub-module).

Shared infrastructure for `flux/hicache_fine.py` and `flux/taylorseer_fine.py`.
Monkey-patches three things:

  * `FluxTransformer2DModel.forward`     — gate decision + per-step state setup
  * `FluxTransformerBlock.forward`       — 4 hooks per block, full vs skip branch
  * `FluxSingleTransformerBlock.forward` — 1 hook per block, full vs skip branch

Each "hook" caches the **pre-gate raw output** of one heavy compute (attention
or MLP), keyed by `(block_idx, stream, sub_module)`. On skip steps the hook
reads a predicted value from the cache and applies the *current step's*
modulation gate to it. This matches HiCache's upstream BFL pattern: cache the
underlying smooth signal, apply per-step modulation on top.

Total cache slots on FLUX = 19 × 4 (dual-stream) + 38 × 1 (single-stream) = 114.

Compatibility: tested against diffusers 0.38.0. A warning is emitted at
install time if running an untested version.

Inference-only: gradient_checkpointing is NOT supported on the patched paths.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import diffusers
import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from lib.gates import IntervalGate

logger = logging.get_logger(__name__)


_TESTED_DIFFUSERS_VERSIONS = {"0.38.0"}


def _check_diffusers_version() -> None:
    if diffusers.__version__ not in _TESTED_DIFFUSERS_VERSIONS:
        warnings.warn(
            f"flux fine cache scaffold tested against diffusers "
            f"{sorted(_TESTED_DIFFUSERS_VERSIONS)}; running on "
            f"{diffusers.__version__} is untested. Block forward signatures "
            f"or sub-module layout may have shifted. If output looks wrong, "
            f"diff `FluxTransformerBlock.forward` and "
            f"`FluxSingleTransformerBlock.forward` against the patched "
            f"versions in this file.",
            stacklevel=3,
        )


# ----------------------------------------------------------------------------
# Predictor abstraction
# ----------------------------------------------------------------------------

CacheHistory = Dict[int, torch.Tensor]


class FineCachePredictor(Protocol):
    """Two-method protocol that hicache_fine and taylorseer_fine implement."""

    max_order: int

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        """Compute new {0: F, 1: Δ¹F, ..., k: Δ^k F} from prev history + new feature."""
        ...

    def predict(
        self,
        history: CacheHistory,
        step_offset: int,
    ) -> torch.Tensor:
        """Predict feature at (last_activation_step + step_offset)."""
        ...


# ----------------------------------------------------------------------------
# Per-trajectory state (attached to FluxTransformer2DModel instance)
# ----------------------------------------------------------------------------


@dataclass
class FineCacheState:
    """Mutable per-trajectory state. Blocks read this via `_cache_state_ref`."""
    gate: IntervalGate
    predictor: FineCachePredictor
    method_tag: str
    cache_dic: Dict[Tuple[int, str, str], CacheHistory] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    action_steps: Optional[set[int]] = None
    prompt_idx: Optional[int] = None
    # Set per-step by patched transformer.forward, read by patched block forwards:
    should_skip: bool = False
    current_step: int = 0
    step_offset: int = 0
    step_gap: int = 1
    effective_max_order: int = 0

    def reset_trajectory(
        self,
        *,
        action_steps: Optional[set[int]] = None,
        prompt_idx: Optional[int] = None,
    ) -> None:
        self.gate.reset()
        self.cache_dic.clear()
        self.decisions.clear()
        self.action_steps = None if action_steps is None else set(int(s) for s in action_steps)
        self.prompt_idx = None if prompt_idx is None else int(prompt_idx)
        self.should_skip = False
        self.current_step = 0
        self.step_offset = 0
        self.step_gap = 1
        self.effective_max_order = 0

    def decide_step(self) -> tuple[int, Optional[int], Optional[str]]:
        """Advance one denoising step and return (step, schedule_u, force_full_reason)."""
        gate = self.gate
        schedule_u: Optional[int] = None
        force_full_reason: Optional[str] = None
        if self.action_steps is None:
            self.should_skip = gate.decide()
            step = gate.cnt - 1
            self.step_offset = gate.step_offset
            if not self.should_skip and len(gate.activated_steps) >= 2:
                self.step_gap = gate.activated_steps[-1] - gate.activated_steps[-2]
            else:
                self.step_gap = 1
        else:
            step = gate.cnt
            raw_skip = int(step in self.action_steps)
            schedule_u = raw_skip
            cache_ready = bool(self.cache_dic)
            self.should_skip = bool(raw_skip and cache_ready)
            if raw_skip and not cache_ready:
                force_full_reason = "cache_unready"

            prev_full = gate.last_activated if gate.activated_steps else None
            if not self.should_skip:
                gate.last_activated = step
                gate.activated_steps.append(step)
            gate.cnt += 1

            self.step_offset = step - gate.last_activated
            if not self.should_skip and prev_full is not None:
                self.step_gap = step - prev_full
            else:
                self.step_gap = 1

        self.current_step = int(step)
        decision = {
            "step": int(step),
            "u": int(self.should_skip),
            "schedule_u": schedule_u,
            "force_full": bool(not self.should_skip),
            "force_full_reason": force_full_reason,
            "history_ready": bool(self.cache_dic),
            "step_offset": int(self.step_offset),
            "step_gap": int(self.step_gap),
            "interval": int(gate.interval),
            "first_enhance": int(gate.first_enhance),
            "method_tag": str(self.method_tag),
            "schedule_locked": self.action_steps is not None,
        }
        self.decisions.append(decision)
        begin_step = getattr(self.predictor, "begin_step", None)
        if begin_step is not None:
            begin_step(decision)
        return int(step), schedule_u, force_full_reason


# ----------------------------------------------------------------------------
# Patched FluxTransformer2DModel.forward
# ----------------------------------------------------------------------------

def _patched_transformer_forward(
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
    """Replace FluxTransformer2DModel.forward. Decides skip/full once per step,
    then runs blocks (which read state via `_cache_state_ref`).
    """
    # Defensive: another model instance using the same class without install gets original behavior
    state: Optional[FineCacheState] = getattr(self, "_fine_state", None)
    if state is None:
        return _original_transformer_forward(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
    # --- gate decision (one per step) ---
    state.decide_step()

    # Warmup guard: during the first `first_enhance` steps we only store F_0.
    # `state.gate.cnt` is post-incremented inside decide(), so it equals "step we just
    # processed + 1". The check `gate.cnt < first_enhance` is True for the first
    # `first_enhance` calls, mirroring coarse `flux/hicache.py`'s convention.
    if state.gate.cnt < state.gate.first_enhance:
        state.effective_max_order = 0
    else:
        state.effective_max_order = state.predictor.max_order

    # --- run the original forward body (with patched blocks doing the work) ---
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
        logger.warning("`txt_ids` 3D input; dropping batch dim.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` 3D input; dropping batch dim.")
        img_ids = img_ids[0]
    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    # IP-Adapter: project image_embeds → hidden_states (mirrors diffusers 0.38.0
    # FluxTransformer2DModel.forward). The skip path further raises in
    # _patched_double_block_forward if `ip_hidden_states` is present, since
    # ip_attn_output is not cached.
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # --- block loops; patched block.forward reads self._cache_state_ref ---
    for index_block, block in enumerate(self.transformer_blocks):
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
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
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

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----------------------------------------------------------------------------
# Patched FluxTransformerBlock.forward (dual-stream, 4 hooks)
# ----------------------------------------------------------------------------

def _patched_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace FluxTransformerBlock.forward. 4 hooks (img_attn, img_mlp, txt_attn, txt_mlp).

    Always runs `norm1` and `norm1_context` (cheap modulation params).
    On full step: runs self.attn / self.ff / self.ff_context, caches their PRE-gate outputs.
    On skip step: skips heavy compute, predicts each PRE-gate output from cache,
    then applies the *current step's* modulation gate.
    """
    # Defensive: untagged blocks (e.g. another model in same process) get original behavior
    state: Optional[FineCacheState] = getattr(self, "_cache_state_ref", None)
    if state is None:
        return _original_double_forward(
            self, hidden_states, encoder_hidden_states, temb,
            image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
        )

    blk_idx: int = self._cache_block_idx
    cache_dic = state.cache_dic

    # Always run modulation (cheap, gives gates + scale + shift)
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    # ip_adapter is not cached; if used together with fine cache it'd silently
    # drop the IP contribution on skip steps. Refuse explicitly.
    if joint_attention_kwargs.get("ip_hidden_states") is not None and state.should_skip:
        raise NotImplementedError(
            "fine cache + ip_adapter is not supported: ip_attn_output is not "
            "cached and would be silently dropped on skip steps. Use the "
            "coarse `flux/hicache.py` or `flux/seacache.py` instead."
        )

    if state.should_skip:
        # === Skip path: predict raw outputs, apply current-step gates ===
        key_img_attn = (blk_idx, "double", "img_attn")
        key_txt_attn = (blk_idx, "double", "txt_attn")
        key_img_mlp = (blk_idx, "double", "img_mlp")
        key_txt_mlp = (blk_idx, "double", "txt_mlp")
        attn_output = state.predictor.predict(cache_dic[key_img_attn], state.step_offset)
        context_attn_output = state.predictor.predict(cache_dic[key_txt_attn], state.step_offset)
        commit_prediction = getattr(state.predictor, "commit_prediction", None)
        if commit_prediction is not None:
            cache_dic[key_img_attn] = commit_prediction(cache_dic[key_img_attn], state.step_offset, attn_output)
            cache_dic[key_txt_attn] = commit_prediction(cache_dic[key_txt_attn], state.step_offset, context_attn_output)

        # img branch: gate * attn, residual add
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # img mlp: predict (don't run norm2/ff), gate, residual add
        ff_output = state.predictor.predict(cache_dic[key_img_mlp], state.step_offset)
        if commit_prediction is not None:
            cache_dic[key_img_mlp] = commit_prediction(cache_dic[key_img_mlp], state.step_offset, ff_output)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        # NOTE: ip_adapter not supported on skip path (no cache for ip_attn_output).

        # txt branch: gate * attn, residual add
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # txt mlp: predict, gate, residual add
        context_ff_output = state.predictor.predict(cache_dic[key_txt_mlp], state.step_offset)
        if commit_prediction is not None:
            cache_dic[key_txt_mlp] = commit_prediction(cache_dic[key_txt_mlp], state.step_offset, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states

    # === Full path: compute attn + MLPs, cache PRE-gate outputs ===
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
        ip_attn_output = None
    else:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # Cache img_attn + txt_attn (BEFORE gating)
    cache_dic[(blk_idx, "double", "img_attn")] = state.predictor.update(
        cache_dic.get((blk_idx, "double", "img_attn")),
        attn_output, state.step_gap, state.effective_max_order,
    )
    cache_dic[(blk_idx, "double", "txt_attn")] = state.predictor.update(
        cache_dic.get((blk_idx, "double", "txt_attn")),
        context_attn_output, state.step_gap, state.effective_max_order,
    )

    # img branch: gate * attn, residual add
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    # img mlp: norm2 + ff, then cache, then gate + residual add
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_output = self.ff(norm_hidden_states)
    cache_dic[(blk_idx, "double", "img_mlp")] = state.predictor.update(
        cache_dic.get((blk_idx, "double", "img_mlp")),
        ff_output, state.step_gap, state.effective_max_order,
    )
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output

    # txt branch: gate * attn, residual add
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    # txt mlp: norm2_context + ff_context, then cache, then gate + residual add
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    cache_dic[(blk_idx, "double", "txt_mlp")] = state.predictor.update(
        cache_dic.get((blk_idx, "double", "txt_mlp")),
        context_ff_output, state.step_gap, state.effective_max_order,
    )
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


# ----------------------------------------------------------------------------
# Patched FluxSingleTransformerBlock.forward (single-stream, 1 hook)
# ----------------------------------------------------------------------------

def _patched_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace FluxSingleTransformerBlock.forward. 1 hook ('combined' = pre-gate proj_out)."""
    state: Optional[FineCacheState] = getattr(self, "_cache_state_ref", None)
    if state is None:
        return _original_single_forward(
            self, hidden_states, encoder_hidden_states, temb,
            image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
        )

    blk_idx: int = self._cache_block_idx
    cache_dic = state.cache_dic

    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = hidden_states

    # Always run norm (cheap, gives gate)
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    joint_attention_kwargs = joint_attention_kwargs or {}

    if state.should_skip:
        # === Skip: predict pre-gate combined output, apply current-step gate ===
        key_combined = (blk_idx, "single", "combined")
        proj_combined = state.predictor.predict(cache_dic[key_combined], state.step_offset)
        commit_prediction = getattr(state.predictor, "commit_prediction", None)
        if commit_prediction is not None:
            cache_dic[key_combined] = commit_prediction(cache_dic[key_combined], state.step_offset, proj_combined)
    else:
        # === Full: compute mlp + attn + proj_out, cache, then gate + residual ===
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        proj_combined = self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
        cache_dic[(blk_idx, "single", "combined")] = state.predictor.update(
            cache_dic.get((blk_idx, "single", "combined")),
            proj_combined, state.step_gap, state.effective_max_order,
        )

    hidden_states = residual + gate.unsqueeze(1) * proj_combined
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states


# ----------------------------------------------------------------------------
# Originals captured at install time so hasattr fallback in patched forwards works
# ----------------------------------------------------------------------------

_original_transformer_forward: Optional[Callable] = None
_original_double_forward: Optional[Callable] = None
_original_single_forward: Optional[Callable] = None


# ----------------------------------------------------------------------------
# install / teardown / reset
# ----------------------------------------------------------------------------

def install_fine_cache(
    pipe,
    *,
    predictor: FineCachePredictor,
    interval: int,
    first_enhance: int,
    num_steps: int,
    method_tag: str,
) -> Callable[[], None]:
    """Patch FluxTransformer + both block classes' forwards; attach state.

    Returns a teardown callable that restores all 3 forwards and clears state.
    Safe to call once per pipe; do NOT call twice without teardown in between.
    """
    global _original_transformer_forward, _original_double_forward, _original_single_forward

    _check_diffusers_version()

    # Capture originals on first install (subsequent installs reuse them)
    if _original_transformer_forward is None:
        _original_transformer_forward = FluxTransformer2DModel.forward
    if _original_double_forward is None:
        _original_double_forward = FluxTransformerBlock.forward
    if _original_single_forward is None:
        _original_single_forward = FluxSingleTransformerBlock.forward

    # Replace class-level forwards
    FluxTransformer2DModel.forward = _patched_transformer_forward
    FluxTransformerBlock.forward = _patched_double_block_forward
    FluxSingleTransformerBlock.forward = _patched_single_block_forward

    # Build state
    tr = pipe.transformer
    state = FineCacheState(
        gate=IntervalGate(
            interval=int(interval),
            first_enhance=int(first_enhance),
            num_steps=int(num_steps),
        ),
        predictor=predictor,
        method_tag=method_tag,
    )
    tr._fine_state = state
    tr.fine_cache_decisions = state.decisions

    # Tag every block instance with idx + stream + back-ref to state
    for idx, block in enumerate(tr.transformer_blocks):
        block._cache_block_idx = idx
        block._cache_stream = "double"
        block._cache_state_ref = state
    for idx, block in enumerate(tr.single_transformer_blocks):
        block._cache_block_idx = idx
        block._cache_stream = "single"
        block._cache_state_ref = state

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        # Restore class-level forwards
        FluxTransformer2DModel.forward = _original_transformer_forward
        FluxTransformerBlock.forward = _original_double_forward
        FluxSingleTransformerBlock.forward = _original_single_forward

        # Strip per-block tags
        for block in list(tr.transformer_blocks) + list(tr.single_transformer_blocks):
            for attr in ("_cache_block_idx", "_cache_stream", "_cache_state_ref"):
                if hasattr(block, attr):
                    try:
                        delattr(block, attr)
                    except AttributeError:
                        pass

        # Strip transformer state
        if hasattr(tr, "_fine_state"):
            try:
                delattr(tr, "_fine_state")
            except AttributeError:
                pass
        if hasattr(tr, "fine_cache_decisions"):
            try:
                delattr(tr, "fine_cache_decisions")
            except AttributeError:
                pass

        _torn_down["done"] = True

    return teardown


def reset_per_image_state_fine(
    pipe,
    *,
    action_steps: Optional[set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    """Reset the per-trajectory state (gate counter + cache_dic) before each new prompt."""
    tr = pipe.transformer
    if hasattr(tr, "_fine_state"):
        tr._fine_state.reset_trajectory(action_steps=action_steps, prompt_idx=prompt_idx)
        tr.fine_cache_decisions = tr._fine_state.decisions
