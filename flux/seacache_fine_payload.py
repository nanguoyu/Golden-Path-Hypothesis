"""Research-only SeaCache gate with fine-grained feature forecast payloads.

This mode is intentionally separate from the locked baseline implementations.
It combines SeaCache's threshold/schedule decision rule with Taylor/HiCache
forecasts over FLUX's 114 fine cache slots:

  * 19 dual-stream blocks × {img_attn, img_mlp, txt_attn, txt_mlp}
  * 38 single-stream blocks × {combined}

The cached object is the pre-gate raw submodule output used by
``lib.flux_fine_scaffold``.  Cached steps skip the heavy attention/MLP modules,
forecast each slot from histories collected only on full-refresh steps, then
apply the current timestep's modulation gates.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

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

from lib.gates import rel_l1
from lib.hermite import hermite_update, hicache_predict
from lib.taylor import taylor_predict, taylor_update
from lib.wiener import apply_sea_with_scheduler

logger = logging.get_logger(__name__)

_TESTED_DIFFUSERS_VERSIONS = {"0.38.0"}
EPS = 1e-12

PAYLOAD_MODES = (
    "fine_reuse",
    "fine_taylor_o1",
    "fine_taylor_o2",
    "fine_hicache_o2",
    "fine_rfc_rfe_taylor_o1",
    "fine_rfc_rfe_taylor_o2",
)
PAYLOAD_GATE_MODES = ("seacache", "rfc_input_error")

_DOUBLE_SLOT_NAMES = ("img_attn", "txt_attn", "img_mlp", "txt_mlp")
_SINGLE_SLOT_NAMES = ("combined",)

CacheKey = Tuple[int, str, str]
CacheHistory = Dict[int, torch.Tensor]


def _check_diffusers_version() -> None:
    if diffusers.__version__ not in _TESTED_DIFFUSERS_VERSIONS:
        warnings.warn(
            "SeaCacheFinePayload was tested against diffusers "
            f"{sorted(_TESTED_DIFFUSERS_VERSIONS)}; running on "
            f"{diffusers.__version__} is untested.",
            stacklevel=3,
        )


def _norm(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())


def _cos(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None or tuple(a.shape) != tuple(b.shape):
        return None
    af = a.detach().to(torch.float32)
    bf = b.detach().to(torch.float32)
    an = float(af.norm().item())
    bn = float(bf.norm().item())
    if an <= 0.0 or bn <= 0.0:
        return None
    return float(torch.sum(af * bf).item()) / (an * bn + EPS)


def _sqrt_tensor_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    scalar = float(value.detach().to("cpu").item())
    if scalar <= 0.0:
        return 0.0
    if not math.isfinite(scalar):
        return None
    return float(math.sqrt(scalar))


class _FinePayloadPredictor:
    def __init__(self, payload_mode: str, sigma: float):
        if payload_mode not in PAYLOAD_MODES:
            raise ValueError(f"unknown fine payload mode: {payload_mode!r}")
        self.payload_mode = str(payload_mode)
        self.sigma = float(sigma)
        if payload_mode == "fine_taylor_o1":
            self.kind = "taylor"
            self.max_order = 1
        elif payload_mode == "fine_taylor_o2":
            self.kind = "taylor"
            self.max_order = 2
        elif payload_mode == "fine_hicache_o2":
            self.kind = "hicache"
            self.max_order = 2
        elif payload_mode == "fine_rfc_rfe_taylor_o1":
            self.kind = "rfc_rfe"
            self.max_order = 1
        elif payload_mode == "fine_rfc_rfe_taylor_o2":
            self.kind = "rfc_rfe"
            self.max_order = 2
        else:
            self.kind = "reuse"
            self.max_order = 0

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        order = min(int(effective_max_order), int(self.max_order))
        if self.kind in {"taylor", "rfc_rfe"}:
            return taylor_update(prev_history, feature, step_gap, order)
        if self.kind == "hicache":
            return hermite_update(prev_history, feature, step_gap, order)
        return {0: feature}

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        if self.kind in {"taylor", "rfc_rfe"}:
            return taylor_predict(history, step_offset, self.max_order)
        if self.kind == "hicache":
            return hicache_predict(history, step_offset, self.sigma, self.max_order)
        return history[0]


@dataclass
class FineSeaPayloadState:
    predictor: _FinePayloadPredictor
    threshold: float
    num_steps: int
    first_enhance: int
    expected_slots: int
    log_payload_norms: bool = False
    shadow_full_velocity: bool = False
    cache_dic: Dict[CacheKey, CacheHistory] = field(default_factory=dict)
    input_cache_dic: Dict[CacheKey, CacheHistory] = field(default_factory=dict)
    cnt: int = 0
    accumulated_rel_l1_distance: float = 0.0
    previous_modulated_input: Optional[torch.Tensor] = None
    previous_rfc_gate_input: Optional[torch.Tensor] = None
    rfc_gate_dic: CacheHistory = field(default_factory=dict)
    rfc_accumulated_error: float = 0.0
    rfc_gate_tau: Optional[float] = None
    payload_gate_mode: str = "seacache"
    last_full_step: Optional[int] = None
    previous_full_step: Optional[int] = None
    full_steps_seen: int = 0
    action_steps: Optional[Set[int]] = None
    prompt_idx: Optional[int] = None
    decisions: list[Dict[str, Any]] = field(default_factory=list)
    observer: Optional[Any] = None

    # Per-step fields read by patched block forwards.
    should_skip: bool = False
    step_offset: int = 0
    step_gap: int = 1
    effective_max_order: int = 0

    # Per-step health/accounting counters.
    slots_ready_pre: int = 0
    slots_missing_pre: int = 0
    predicted_slots: int = 0
    updated_slots: int = 0
    payload_reuse_norm_sq: Optional[torch.Tensor] = None
    payload_forecast_norm_sq: Optional[torch.Tensor] = None
    payload_delta_norm_sq: Optional[torch.Tensor] = None
    rfc_payload_input_delta_norm_sq: Optional[torch.Tensor] = None
    rfc_payload_hist_input_delta_norm_sq: Optional[torch.Tensor] = None
    rfc_payload_hist_output_delta_norm_sq: Optional[torch.Tensor] = None
    rfc_payload_magnitude_sq: Optional[torch.Tensor] = None
    rfc_payload_slots_with_ratio: int = 0
    rfc_payload_slots_degraded_to_taylor: int = 0
    current_slot_inputs: Dict[CacheKey, torch.Tensor] = field(default_factory=dict)

    def reset_trajectory(
        self,
        action_steps: Optional[Set[int]] = None,
        prompt_idx: Optional[int] = None,
    ) -> None:
        self.cache_dic.clear()
        self.input_cache_dic.clear()
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_rfc_gate_input = None
        self.rfc_gate_dic.clear()
        self.rfc_accumulated_error = 0.0
        self.last_full_step = None
        self.previous_full_step = None
        self.full_steps_seen = 0
        self.action_steps = None if action_steps is None else set(int(x) for x in action_steps)
        self.prompt_idx = None if prompt_idx is None else int(prompt_idx)
        self.decisions.clear()
        self.should_skip = False
        self.step_offset = 0
        self.step_gap = 1
        self.effective_max_order = 0
        self.reset_step_accounting()

    def reset_step_accounting(self) -> None:
        self.slots_ready_pre = self.count_ready_slots()
        self.slots_missing_pre = max(0, int(self.expected_slots) - int(self.slots_ready_pre))
        self.predicted_slots = 0
        self.updated_slots = 0
        self.payload_reuse_norm_sq = None
        self.payload_forecast_norm_sq = None
        self.payload_delta_norm_sq = None
        self.rfc_payload_input_delta_norm_sq = None
        self.rfc_payload_hist_input_delta_norm_sq = None
        self.rfc_payload_hist_output_delta_norm_sq = None
        self.rfc_payload_magnitude_sq = None
        self.rfc_payload_slots_with_ratio = 0
        self.rfc_payload_slots_degraded_to_taylor = 0
        self.current_slot_inputs.clear()

    def count_ready_slots(self) -> int:
        return sum(1 for hist in self.cache_dic.values() if isinstance(hist, dict) and 0 in hist)

    def available_order_range(self) -> tuple[Optional[int], Optional[int]]:
        orders = [
            max(int(k) for k in hist)
            for hist in self.cache_dic.values()
            if isinstance(hist, dict) and 0 in hist
        ]
        if not orders:
            return None, None
        return min(orders), max(orders)

    @property
    def cache_ready(self) -> bool:
        return self.count_ready_slots() >= int(self.expected_slots)


def _predict_slot(state: FineSeaPayloadState, key: CacheKey) -> torch.Tensor:
    history = state.cache_dic[key]
    if state.predictor.kind == "rfc_rfe":
        pred = _predict_rfc_rfe_slot(state, key)
    else:
        pred = state.predictor.predict(history, state.step_offset)
    reuse = history[0]
    state.predicted_slots += 1
    if state.log_payload_norms:
        reuse_sq = reuse.detach().to(torch.float32).pow(2).sum()
        pred_sq = pred.detach().to(torch.float32).pow(2).sum()
        delta_sq = (pred.detach().to(torch.float32) - reuse.detach().to(torch.float32)).pow(2).sum()
        state.payload_reuse_norm_sq = (
            reuse_sq if state.payload_reuse_norm_sq is None else state.payload_reuse_norm_sq + reuse_sq
        )
        state.payload_forecast_norm_sq = (
            pred_sq if state.payload_forecast_norm_sq is None else state.payload_forecast_norm_sq + pred_sq
        )
        state.payload_delta_norm_sq = (
            delta_sq if state.payload_delta_norm_sq is None else state.payload_delta_norm_sq + delta_sq
        )
    return pred


def _add_sq(accum: Optional[torch.Tensor], value: torch.Tensor) -> torch.Tensor:
    return value if accum is None else accum + value


def _predict_rfc_rfe_slot(state: FineSeaPayloadState, key: CacheKey) -> torch.Tensor:
    output_history = state.cache_dic[key]
    input_history = state.input_cache_dic.get(key)
    reuse = output_history[0]
    raw = state.predictor.predict(output_history, state.step_offset)
    if (
        not isinstance(input_history, dict)
        or 0 not in input_history
        or key not in getattr(state, "current_slot_inputs", {})
        or int(state.step_offset) <= 0
        or 1 not in output_history
        or 1 not in input_history
    ):
        state.rfc_payload_slots_degraded_to_taylor += 1
        return raw

    current_input = state.current_slot_inputs[key]
    if tuple(current_input.shape) != tuple(input_history[0].shape):
        state.rfc_payload_slots_degraded_to_taylor += 1
        return raw

    raw_f = raw.detach().to(torch.float32)
    reuse_f = reuse.detach().to(torch.float32)
    direction = raw_f - reuse_f
    direction_norm = direction.norm()
    if float(direction_norm.item()) <= 0.0:
        state.rfc_payload_slots_degraded_to_taylor += 1
        return raw

    input_delta = current_input.detach().to(torch.float32) - input_history[0].detach().to(torch.float32)
    input_delta_norm = input_delta.norm()
    hist_input_delta_norm = input_history[1].detach().to(torch.float32).norm()
    hist_output_delta_norm = output_history[1].detach().to(torch.float32).norm()
    s_ratio = hist_output_delta_norm / (hist_input_delta_norm + EPS)
    magnitude = s_ratio * input_delta_norm
    chosen_f = reuse_f + direction / (direction_norm + EPS) * magnitude

    state.rfc_payload_slots_with_ratio += 1
    state.rfc_payload_input_delta_norm_sq = _add_sq(
        state.rfc_payload_input_delta_norm_sq,
        input_delta_norm.pow(2),
    )
    state.rfc_payload_hist_input_delta_norm_sq = _add_sq(
        state.rfc_payload_hist_input_delta_norm_sq,
        hist_input_delta_norm.pow(2),
    )
    state.rfc_payload_hist_output_delta_norm_sq = _add_sq(
        state.rfc_payload_hist_output_delta_norm_sq,
        hist_output_delta_norm.pow(2),
    )
    state.rfc_payload_magnitude_sq = _add_sq(
        state.rfc_payload_magnitude_sq,
        magnitude.pow(2),
    )
    return chosen_f.to(dtype=reuse.dtype, device=reuse.device)


def _update_slot(
    state: FineSeaPayloadState,
    key: CacheKey,
    feature: torch.Tensor,
    input_feature: Optional[torch.Tensor] = None,
) -> None:
    observer = getattr(state, "observer", None)
    if observer is not None and input_feature is not None:
        observer.observe_slot(
            state=state,
            key=key,
            input_feature=input_feature,
            output_feature=feature,
            prev_input_history=state.input_cache_dic.get(key),
            prev_output_history=state.cache_dic.get(key),
        )
    state.cache_dic[key] = state.predictor.update(
        state.cache_dic.get(key),
        feature,
        state.step_gap,
        state.effective_max_order,
    )
    if input_feature is not None:
        state.input_cache_dic[key] = taylor_update(
            state.input_cache_dic.get(key),
            input_feature,
            state.step_gap,
            state.effective_max_order,
        )
    state.updated_slots += 1


def _payload_base_mode(payload_mode: str) -> str:
    return {
        "fine_reuse": "fine_reuse",
        "fine_taylor_o1": "fine_taylor",
        "fine_taylor_o2": "fine_taylor",
        "fine_hicache_o2": "fine_hicache",
        "fine_rfc_rfe_taylor_o1": "fine_rfc_rfe",
        "fine_rfc_rfe_taylor_o2": "fine_rfc_rfe",
    }[payload_mode]


def _scheduler_step_fields(state: FineSeaPayloadState, step: int) -> Dict[str, Any]:
    scheduler = getattr(state, "scheduler", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or int(step) + 1 >= len(sigmas):
        return {
            "sigma_n": None,
            "sigma_np1": None,
            "step_size_H": None,
            "abs_step_size_H": None,
        }
    sigma_n = float(sigmas[int(step)])
    sigma_np1 = float(sigmas[int(step) + 1])
    step_size = sigma_np1 - sigma_n
    return {
        "sigma_n": sigma_n,
        "sigma_np1": sigma_np1,
        "step_size_H": float(step_size),
        "abs_step_size_H": abs(float(step_size)),
    }


def _empty_rfc_gate_fields(enabled: bool = False) -> Dict[str, Any]:
    return {
        "fine_rfc_gate_enabled": bool(enabled),
        "fine_rfc_gate_tau": None,
        "fine_rfc_gate_order_requested": None,
        "fine_rfc_gate_order_used": None,
        "fine_rfc_gate_history_count": None,
        "fine_rfc_gate_prediction_error_rel_l1": None,
        "fine_rfc_gate_accumulator_before": None,
        "fine_rfc_gate_accumulator_after_increment": None,
        "fine_rfc_gate_cache_allowed": False,
        "fine_rfc_gate_decision_reason": None,
    }


def _rfc_gate_pre_fields(
    state: FineSeaPayloadState,
    current_input: torch.Tensor,
    *,
    force_full_reason: Optional[str],
    step_offset: int,
) -> Tuple[bool, Dict[str, Any]]:
    fields = _empty_rfc_gate_fields(enabled=True)
    tau = float(state.threshold if state.rfc_gate_tau is None else state.rfc_gate_tau)
    order_requested = int(state.predictor.max_order)
    accumulator_before = float(state.rfc_accumulated_error)
    fields.update({
        "fine_rfc_gate_tau": tau,
        "fine_rfc_gate_order_requested": order_requested,
        "fine_rfc_gate_history_count": int(len(state.rfc_gate_dic)),
        "fine_rfc_gate_accumulator_before": accumulator_before,
    })
    if force_full_reason is not None:
        fields["fine_rfc_gate_decision_reason"] = f"force_full:{force_full_reason}"
        return False, fields
    if 0 not in state.rfc_gate_dic or order_requested < 1:
        fields["fine_rfc_gate_decision_reason"] = "insufficient_gate_input_records"
        return False, fields
    max_history_order = max(int(k) for k in state.rfc_gate_dic)
    if max_history_order < 1:
        fields["fine_rfc_gate_decision_reason"] = "insufficient_gate_input_records"
        return False, fields
    order_used = min(order_requested, max_history_order)
    pred = taylor_predict(state.rfc_gate_dic, max(1, int(step_offset)), order_used)
    if tuple(pred.shape) != tuple(current_input.shape):
        fields["fine_rfc_gate_decision_reason"] = "record_shape_mismatch"
        return False, fields
    err_rel = rel_l1(current_input, pred)
    accumulator_after = accumulator_before + float(err_rel)
    cache_allowed = bool(accumulator_after < tau)
    fields.update({
        "fine_rfc_gate_order_used": int(order_used),
        "fine_rfc_gate_prediction_error_rel_l1": float(err_rel),
        "fine_rfc_gate_accumulator_after_increment": float(accumulator_after),
        "fine_rfc_gate_cache_allowed": bool(cache_allowed),
        "fine_rfc_gate_decision_reason": (
            "cache_allowed" if cache_allowed else "accumulated_error_exceeds_tau"
        ),
    })
    return cache_allowed, fields


def _empty_shadow_velocity_fields(enabled: bool = False) -> Dict[str, Any]:
    return {
        "shadow_observer_enabled": bool(enabled),
        "shadow_observer_ran": False,
        "shadow_observer_elapsed_ms": None,
        "shadow_full_velocity_norm": None,
        "velocity_chosen_norm": None,
        "velocity_err_chosen_abs": None,
        "velocity_err_chosen_rel": None,
        "velocity_err_chosen_cos": None,
        "update_err_chosen_abs": None,
        "update_err_chosen_rel": None,
    }


def _run_original_transformer_without_fine_state(
    self,
    *,
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
    return_dict: bool = False,
    controlnet_blocks_repeat: bool = False,
) -> torch.Tensor:
    if _original_transformer_forward is None:
        raise RuntimeError("SeaCacheFinePayload original transformer forward is not installed")
    saved_refs = []
    for block in list(self.transformer_blocks) + list(self.single_transformer_blocks):
        had_ref = hasattr(block, "_seacache_fine_payload_state_ref")
        ref = getattr(block, "_seacache_fine_payload_state_ref", None)
        saved_refs.append((block, had_ref, ref))
        if had_ref:
            delattr(block, "_seacache_fine_payload_state_ref")
    try:
        out = _original_transformer_forward(
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
    finally:
        for block, had_ref, ref in saved_refs:
            if had_ref:
                block._seacache_fine_payload_state_ref = ref
    if isinstance(out, tuple):
        return out[0]
    return out.sample


def _shadow_velocity_fields(
    self,
    *,
    state: FineSeaPayloadState,
    step: int,
    chosen_velocity: torch.Tensor,
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
    controlnet_blocks_repeat: bool = False,
) -> Dict[str, Any]:
    fields = _empty_shadow_velocity_fields(enabled=True)
    step_size = _scheduler_step_fields(state, int(step)).get("step_size_H")
    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_event is not None:
        start_event.record()
    with torch.inference_mode():
        shadow_velocity = _run_original_transformer_without_fine_state(
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
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
    if end_event is not None:
        end_event.record()
        torch.cuda.synchronize()
        fields["shadow_observer_elapsed_ms"] = float(start_event.elapsed_time(end_event))
    diff = chosen_velocity.detach().to(torch.float32) - shadow_velocity.detach().to(torch.float32)
    err_abs = float(diff.norm().item())
    shadow_norm = float(shadow_velocity.detach().to(torch.float32).norm().item())
    fields.update({
        "shadow_observer_ran": True,
        "shadow_full_velocity_norm": _norm(shadow_velocity),
        "velocity_chosen_norm": _norm(chosen_velocity),
        "velocity_err_chosen_abs": err_abs,
        "velocity_err_chosen_rel": float(err_abs / (shadow_norm + EPS)),
        "velocity_err_chosen_cos": _cos(chosen_velocity, shadow_velocity),
    })
    if step_size is not None:
        update_err = abs(float(step_size)) * err_abs
        shadow_update_norm = abs(float(step_size)) * shadow_norm
        fields.update({
            "update_err_chosen_abs": float(update_err),
            "update_err_chosen_rel": float(update_err / (shadow_update_norm + EPS)),
        })
    return fields


def _seacache_fine_payload_forward(
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
    state: Optional[FineSeaPayloadState] = getattr(self, "_seacache_fine_payload_state", None)
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

    raw_hidden_states = hidden_states
    raw_encoder_hidden_states = encoder_hidden_states
    raw_pooled_projections = pooled_projections
    raw_timestep = timestep
    raw_img_ids = img_ids
    raw_txt_ids = txt_ids
    raw_guidance = guidance
    raw_joint_attention_kwargs = None if joint_attention_kwargs is None else joint_attention_kwargs.copy()

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
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim.")
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

    cnt = int(state.cnt)
    threshold = float(state.threshold)
    payload_mode = str(state.predictor.payload_mode)
    state.reset_step_accounting()

    inp = hidden_states
    first_block = self.transformer_blocks[0]
    modulated_inp, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = first_block.norm1(inp, emb=temb)
    modulated_context_inp, *_ = first_block.norm1_context(encoder_hidden_states, emb=temb)
    rfc_gate_input = torch.cat((modulated_context_inp, modulated_inp), dim=1)

    force_full_reason = None
    if cnt < int(state.first_enhance):
        force_full_reason = "first_enhance"
    elif cnt == 0:
        force_full_reason = "step0"
    elif cnt == int(state.num_steps) - 1:
        force_full_reason = "final_step"
    elif state.previous_modulated_input is None:
        force_full_reason = "no_previous_modulated_input"

    accumulator_before = float(state.accumulated_rel_l1_distance)
    sea_increment = None
    accumulator_after_increment = accumulator_before
    native_should_calc = True
    native_force_full_reason = force_full_reason

    if force_full_reason is None:
        modulated_for_distance = modulated_inp.reshape(
            modulated_inp.shape[0],
            int(img_ids[:, 1].max().item() + 1),
            int(img_ids[:, 2].max().item() + 1),
            modulated_inp.shape[-1],
        )
        modulated_for_distance = apply_sea_with_scheduler(
            modulated_for_distance,
            self.scheduler,
            cnt,
            power_exp=2.0,
            dims=(-2, -3),
            norm_mode="mean",
        )
        modulated_for_distance = modulated_for_distance.reshape(
            modulated_for_distance.shape[0], -1, modulated_for_distance.shape[-1]
        )
        sea_increment = rel_l1(modulated_for_distance, state.previous_modulated_input)
        modulated_for_state = modulated_for_distance
        accumulator_after_increment = accumulator_before + float(sea_increment)
        native_should_calc = not (accumulator_after_increment < threshold)
    else:
        modulated_for_state = modulated_inp

    action_steps = state.action_steps
    schedule_locked = action_steps is not None
    schedule_u = None if action_steps is None else int(cnt in action_steps)
    seacache_gate_cache_allowed = bool(force_full_reason is None and not native_should_calc)
    rfc_gate_fields = _empty_rfc_gate_fields(
        enabled=bool(str(state.payload_gate_mode) == "rfc_input_error")
    )
    rfc_gate_cache_allowed = False
    rfc_gate_step_offset = cnt - int(state.last_full_step) if state.last_full_step is not None else 0
    if str(state.payload_gate_mode) == "rfc_input_error":
        rfc_gate_cache_allowed, rfc_gate_fields = _rfc_gate_pre_fields(
            state,
            rfc_gate_input,
            force_full_reason=force_full_reason,
            step_offset=rfc_gate_step_offset,
        )
    candidate_should_calc = True
    if force_full_reason is not None:
        candidate_should_calc = True
    elif action_steps is not None:
        candidate_should_calc = not bool(schedule_u)
    elif str(state.payload_gate_mode) == "rfc_input_error":
        candidate_should_calc = not bool(rfc_gate_cache_allowed)
    else:
        candidate_should_calc = bool(native_should_calc)

    cache_ready_pre = bool(state.cache_ready)
    if not candidate_should_calc and not cache_ready_pre:
        force_full_reason = "fine_cache_unready"
        candidate_should_calc = True

    should_calc = bool(candidate_should_calc)
    state.should_skip = not should_calc
    if should_calc:
        state.step_offset = 0
        state.step_gap = 1 if state.last_full_step is None else max(1, cnt - int(state.last_full_step))
    else:
        state.step_offset = cnt - int(state.last_full_step) if state.last_full_step is not None else 0
        state.step_gap = 0
    state.effective_max_order = (
        0 if cnt < int(state.first_enhance) else int(state.predictor.max_order)
    )
    last_full_step_pre = state.last_full_step
    previous_full_step_pre = state.previous_full_step
    available_order_min_pre, available_order_max_pre = state.available_order_range()
    effective_predict_order = (
        None if available_order_min_pre is None
        else min(int(state.predictor.max_order), int(available_order_min_pre))
    )

    state.previous_modulated_input = modulated_for_state.detach()
    state.accumulated_rel_l1_distance = 0.0 if should_calc else float(accumulator_after_increment)
    if str(state.payload_gate_mode) == "rfc_input_error":
        state.rfc_accumulated_error = (
            0.0 if should_calc
            else float(rfc_gate_fields.get("fine_rfc_gate_accumulator_after_increment") or 0.0)
        )
        if should_calc:
            state.rfc_gate_dic = taylor_update(
                state.rfc_gate_dic,
                rfc_gate_input.detach(),
                state.step_gap,
                state.effective_max_order,
            )
        state.previous_rfc_gate_input = rfc_gate_input.detach()

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

    if should_calc:
        state.previous_full_step = state.last_full_step
        state.last_full_step = cnt
        state.full_steps_seen += 1

    payload_available = None
    payload_fallback = None
    payload_used = None
    payload_fallback_reason = None
    if not should_calc:
        payload_available = bool(cache_ready_pre and state.predicted_slots == int(state.expected_slots))
        payload_fallback = False
        payload_used = payload_mode
    elif schedule_locked and schedule_u == 1 and should_calc:
        payload_available = False
        payload_fallback = True
        payload_fallback_reason = force_full_reason or "schedule_cache_forced_full"

    scheduler_fields = _scheduler_step_fields(state, cnt)
    shadow_velocity_fields = _empty_shadow_velocity_fields(
        enabled=bool(state.shadow_full_velocity)
    )
    state.decisions.append({
        "step": int(cnt),
        "u": int(not should_calc),
        "native_u": (
            int(rfc_gate_cache_allowed)
            if str(state.payload_gate_mode) == "rfc_input_error" and native_force_full_reason is None
            else int(not native_should_calc) if native_force_full_reason is None else 0
        ),
        "seacache_native_u": int(not native_should_calc) if native_force_full_reason is None else 0,
        "seacache_gate_cache_allowed": bool(seacache_gate_cache_allowed),
        "combined_gate_cache_allowed": (
            bool(rfc_gate_cache_allowed)
            if str(state.payload_gate_mode) == "rfc_input_error" else None
        ),
        "combined_gate_veto_reason": None,
        "schedule_locked": bool(schedule_locked),
        "schedule_u": schedule_u,
        "threshold": threshold,
        "payload_gate_mode": str(state.payload_gate_mode),
        "force_full": bool(force_full_reason is not None),
        "force_full_reason": force_full_reason,
        "native_force_full_reason": native_force_full_reason,
        "accumulator_before": accumulator_before,
        "sea_increment_rel_l1": sea_increment,
        "accumulator_after_increment": float(accumulator_after_increment),
        "accumulator_after_commit": float(state.accumulated_rel_l1_distance),
        "accumulator_reset": bool(should_calc),
        "previous_modulated_input_present": True,
        "previous_residual_present": None,
        "previous_residual_norm": None,
        "cnt_pre": int(cnt),
        "cnt_post": int((cnt + 1) % int(state.num_steps)),
        "history_updated": bool(should_calc),
        "full_residual_norm": None,
        "full_output_norm": None,
        "full_update_norm": None,
        "update_history_updated": False,
        "update_history_update_source": None,
        "payload_mode": payload_mode,
        "payload_base_mode": _payload_base_mode(payload_mode),
        "payload_control": None,
        "payload_blend": 1.0,
        "payload_used": payload_used,
        "payload_available": payload_available,
        "payload_fallback": payload_fallback,
        "payload_fallback_reason": payload_fallback_reason,
        "payload_reuse_norm": (
            _sqrt_tensor_or_none(state.payload_reuse_norm_sq)
            if (not should_calc and state.log_payload_norms) else None
        ),
        "payload_forecast_norm": (
            _sqrt_tensor_or_none(state.payload_forecast_norm_sq)
            if (not should_calc and state.log_payload_norms) else None
        ),
        "payload_chosen_norm": (
            _sqrt_tensor_or_none(state.payload_forecast_norm_sq)
            if (not should_calc and state.log_payload_norms) else None
        ),
        "payload_delta_from_reuse_norm": (
            _sqrt_tensor_or_none(state.payload_delta_norm_sq)
            if (not should_calc and state.log_payload_norms) else None
        ),
        "payload_norms_logged": bool(state.log_payload_norms),
        "fine_payload_enabled": True,
        "fine_payload_method": state.predictor.kind,
        "fine_payload_mode": payload_mode,
        "fine_payload_order": int(state.predictor.max_order),
        "fine_payload_sigma": float(state.predictor.sigma),
        "fine_payload_expected_slots": int(state.expected_slots),
        "fine_payload_slots_ready_pre": int(state.slots_ready_pre),
        "fine_payload_slots_missing_pre": int(state.slots_missing_pre),
        "fine_payload_cache_ready_pre": bool(cache_ready_pre),
        "fine_payload_predicted_slots": int(state.predicted_slots),
        "fine_payload_updated_slots": int(state.updated_slots),
        "fine_payload_step_offset": int(state.step_offset),
        "fine_payload_step_gap": int(state.step_gap),
        "fine_payload_requested_order": int(state.predictor.max_order),
        "fine_payload_available_order_min_pre": (
            None if available_order_min_pre is None else int(available_order_min_pre)
        ),
        "fine_payload_available_order_max_pre": (
            None if available_order_max_pre is None else int(available_order_max_pre)
        ),
        "fine_payload_effective_predict_order": (
            None if effective_predict_order is None else int(effective_predict_order)
        ),
        "fine_payload_prediction_order_degraded": (
            None if effective_predict_order is None
            else bool(int(effective_predict_order) < int(state.predictor.max_order))
        ),
        "fine_payload_last_full_step_pre": (
            None if last_full_step_pre is None else int(last_full_step_pre)
        ),
        "fine_payload_previous_full_step_pre": (
            None if previous_full_step_pre is None else int(previous_full_step_pre)
        ),
        "fine_payload_last_full_step_post": (
            None if state.last_full_step is None else int(state.last_full_step)
        ),
        "fine_payload_full_steps_seen": int(state.full_steps_seen),
        "fine_payload_effective_max_order": int(state.effective_max_order),
        "fine_payload_history_source": "full_refresh_only",
        "fine_payload_feature_space": "pre_gate_raw_submodule_output",
        "fine_payload_granularity": "fine_114",
        "fine_rfc_rfe_enabled": bool(state.predictor.kind == "rfc_rfe"),
        "fine_rfc_rfe_boundary": (
            "dual_per_module_single_fused_combined"
            if state.predictor.kind == "rfc_rfe" else None
        ),
        "fine_rfc_rfe_slots_with_ratio": int(state.rfc_payload_slots_with_ratio),
        "fine_rfc_rfe_slots_degraded_to_taylor": int(state.rfc_payload_slots_degraded_to_taylor),
        "fine_rfc_rfe_input_delta_norm": (
            _sqrt_tensor_or_none(state.rfc_payload_input_delta_norm_sq)
            if not should_calc else None
        ),
        "fine_rfc_rfe_hist_input_delta_norm": (
            _sqrt_tensor_or_none(state.rfc_payload_hist_input_delta_norm_sq)
            if not should_calc else None
        ),
        "fine_rfc_rfe_hist_output_delta_norm": (
            _sqrt_tensor_or_none(state.rfc_payload_hist_output_delta_norm_sq)
            if not should_calc else None
        ),
        "fine_rfc_rfe_magnitude_norm": (
            _sqrt_tensor_or_none(state.rfc_payload_magnitude_sq)
            if not should_calc else None
        ),
        **scheduler_fields,
        **rfc_gate_fields,
        **shadow_velocity_fields,
    })

    state.cnt += 1
    if state.cnt == int(state.num_steps):
        state.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if bool(state.shadow_full_velocity) and not should_calc:
        shadow_velocity_fields = _shadow_velocity_fields(
            self,
            state=state,
            step=cnt,
            chosen_velocity=output,
            hidden_states=raw_hidden_states,
            encoder_hidden_states=raw_encoder_hidden_states,
            pooled_projections=raw_pooled_projections,
            timestep=raw_timestep,
            img_ids=raw_img_ids,
            txt_ids=raw_txt_ids,
            guidance=raw_guidance,
            joint_attention_kwargs=raw_joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        state.decisions[-1].update(shadow_velocity_fields)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _seacache_fine_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state: Optional[FineSeaPayloadState] = getattr(self, "_seacache_fine_payload_state_ref", None)
    if state is None:
        return _original_double_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    blk_idx = int(self._seacache_fine_payload_block_idx)
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    img_attn_key = (blk_idx, "double", "img_attn")
    txt_attn_key = (blk_idx, "double", "txt_attn")
    img_mlp_key = (blk_idx, "double", "img_mlp")
    txt_mlp_key = (blk_idx, "double", "txt_mlp")
    if state.predictor.kind == "rfc_rfe":
        state.current_slot_inputs[img_attn_key] = norm_hidden_states.detach()
        state.current_slot_inputs[txt_attn_key] = norm_encoder_hidden_states.detach()
    joint_attention_kwargs = joint_attention_kwargs or {}

    if joint_attention_kwargs.get("ip_hidden_states") is not None and state.should_skip:
        raise NotImplementedError(
            "SeaCacheFinePayload + IP-Adapter is not supported because IP "
            "attention output is not cached on skip steps."
        )

    if state.should_skip:
        attn_output = _predict_slot(state, img_attn_key)
        context_attn_output = _predict_slot(state, txt_attn_key)

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        if state.predictor.kind == "rfc_rfe":
            norm_hidden_states_mlp = self.norm2(hidden_states)
            norm_hidden_states_mlp = norm_hidden_states_mlp * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            state.current_slot_inputs[img_mlp_key] = norm_hidden_states_mlp.detach()
        ff_output = _predict_slot(state, img_mlp_key)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        if state.predictor.kind == "rfc_rfe":
            norm_encoder_hidden_states_mlp = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states_mlp = (
                norm_encoder_hidden_states_mlp * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            state.current_slot_inputs[txt_mlp_key] = norm_encoder_hidden_states_mlp.detach()
        context_ff_output = _predict_slot(state, txt_mlp_key)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states

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

    _update_slot(
        state,
        img_attn_key,
        attn_output,
        norm_hidden_states.detach() if state.predictor.kind == "rfc_rfe" else None,
    )
    _update_slot(
        state,
        txt_attn_key,
        context_attn_output,
        norm_encoder_hidden_states.detach() if state.predictor.kind == "rfc_rfe" else None,
    )

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_output = self.ff(norm_hidden_states)
    _update_slot(
        state,
        img_mlp_key,
        ff_output,
        norm_hidden_states.detach() if state.predictor.kind == "rfc_rfe" else None,
    )
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output

    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    _update_slot(
        state,
        txt_mlp_key,
        context_ff_output,
        norm_encoder_hidden_states.detach() if state.predictor.kind == "rfc_rfe" else None,
    )
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _seacache_fine_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state: Optional[FineSeaPayloadState] = getattr(self, "_seacache_fine_payload_state_ref", None)
    if state is None:
        return _original_single_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    blk_idx = int(self._seacache_fine_payload_block_idx)
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    residual = hidden_states

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    combined_key = (blk_idx, "single", "combined")
    if state.predictor.kind == "rfc_rfe":
        state.current_slot_inputs[combined_key] = norm_hidden_states.detach()
    joint_attention_kwargs = joint_attention_kwargs or {}

    if state.should_skip:
        proj_combined = _predict_slot(state, combined_key)
    else:
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        proj_combined = self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
        _update_slot(
            state,
            combined_key,
            proj_combined,
            norm_hidden_states.detach() if state.predictor.kind == "rfc_rfe" else None,
        )

    hidden_states = residual + gate.unsqueeze(1) * proj_combined
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states


_original_transformer_forward: Optional[Callable] = None
_original_double_forward: Optional[Callable] = None
_original_single_forward: Optional[Callable] = None


def install(
    pipe,
    *,
    threshold: float,
    num_steps: int,
    first_enhance: int = 1,
    payload_mode: str = "fine_taylor_o1",
    payload_sigma: float = 0.5,
    log_payload_norms: bool = False,
    shadow_full_velocity: bool = False,
    payload_gate_mode: str = "seacache",
    rfc_gate_tau: Optional[float] = None,
) -> Callable[[], None]:
    """Install SeaCache-gated fine payload forecast hooks."""
    global _original_transformer_forward, _original_double_forward, _original_single_forward

    if payload_mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown fine payload mode: {payload_mode!r}")
    if payload_gate_mode not in PAYLOAD_GATE_MODES:
        raise ValueError(f"unknown fine payload gate mode: {payload_gate_mode!r}")

    _check_diffusers_version()
    if _original_transformer_forward is None:
        _original_transformer_forward = FluxTransformer2DModel.forward
    if _original_double_forward is None:
        _original_double_forward = FluxTransformerBlock.forward
    if _original_single_forward is None:
        _original_single_forward = FluxSingleTransformerBlock.forward

    FluxTransformer2DModel.forward = _seacache_fine_payload_forward
    FluxTransformerBlock.forward = _seacache_fine_double_block_forward
    FluxSingleTransformerBlock.forward = _seacache_fine_single_block_forward

    tr = pipe.transformer
    expected_slots = len(tr.transformer_blocks) * len(_DOUBLE_SLOT_NAMES)
    expected_slots += len(tr.single_transformer_blocks) * len(_SINGLE_SLOT_NAMES)
    state = FineSeaPayloadState(
        predictor=_FinePayloadPredictor(payload_mode, payload_sigma),
        threshold=float(threshold),
        num_steps=int(num_steps),
        first_enhance=int(first_enhance),
        expected_slots=int(expected_slots),
        log_payload_norms=bool(log_payload_norms),
        shadow_full_velocity=bool(shadow_full_velocity),
        payload_gate_mode=str(payload_gate_mode),
        rfc_gate_tau=(None if rfc_gate_tau is None else float(rfc_gate_tau)),
    )
    tr._seacache_fine_payload_state = state
    tr.scheduler = pipe.scheduler
    state.scheduler = pipe.scheduler
    tr.seacache_payload_decisions = state.decisions

    for idx, block in enumerate(tr.transformer_blocks):
        block._seacache_fine_payload_block_idx = idx
        block._seacache_fine_payload_state_ref = state
    for idx, block in enumerate(tr.single_transformer_blocks):
        block._seacache_fine_payload_block_idx = idx
        block._seacache_fine_payload_state_ref = state

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = _original_transformer_forward
        FluxTransformerBlock.forward = _original_double_forward
        FluxSingleTransformerBlock.forward = _original_single_forward

        for block in list(tr.transformer_blocks) + list(tr.single_transformer_blocks):
            for attr in (
                "_seacache_fine_payload_block_idx",
                "_seacache_fine_payload_state_ref",
            ):
                if hasattr(block, attr):
                    try:
                        delattr(block, attr)
                    except AttributeError:
                        pass
        for attr in (
            "_seacache_fine_payload_state",
            "scheduler",
            "seacache_payload_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def install_block_hooks(
    pipe,
    *,
    threshold: float,
    num_steps: int,
    first_enhance: int = 1,
    payload_mode: str = "fine_taylor_o1",
    payload_sigma: float = 0.5,
    log_payload_norms: bool = False,
    shadow_full_velocity: bool = False,
    payload_gate_mode: str = "seacache",
    rfc_gate_tau: Optional[float] = None,
) -> Callable[[], None]:
    """Install only SeaCacheFinePayload block hooks.

    This is used by trajectory-deviation runners that need to own the
    transformer-level forward pass while still reusing the 114-slot fine
    feature cache/forecast implementation.
    """
    global _original_double_forward, _original_single_forward

    if payload_mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown fine payload mode: {payload_mode!r}")
    if payload_gate_mode not in PAYLOAD_GATE_MODES:
        raise ValueError(f"unknown fine payload gate mode: {payload_gate_mode!r}")

    _check_diffusers_version()
    if _original_double_forward is None:
        _original_double_forward = FluxTransformerBlock.forward
    if _original_single_forward is None:
        _original_single_forward = FluxSingleTransformerBlock.forward

    FluxTransformerBlock.forward = _seacache_fine_double_block_forward
    FluxSingleTransformerBlock.forward = _seacache_fine_single_block_forward

    tr = pipe.transformer
    expected_slots = len(tr.transformer_blocks) * len(_DOUBLE_SLOT_NAMES)
    expected_slots += len(tr.single_transformer_blocks) * len(_SINGLE_SLOT_NAMES)
    state = FineSeaPayloadState(
        predictor=_FinePayloadPredictor(payload_mode, payload_sigma),
        threshold=float(threshold),
        num_steps=int(num_steps),
        first_enhance=int(first_enhance),
        expected_slots=int(expected_slots),
        log_payload_norms=bool(log_payload_norms),
        shadow_full_velocity=bool(shadow_full_velocity),
        payload_gate_mode=str(payload_gate_mode),
        rfc_gate_tau=(None if rfc_gate_tau is None else float(rfc_gate_tau)),
    )
    tr._seacache_fine_payload_state = state
    state.scheduler = pipe.scheduler
    tr.seacache_payload_decisions = state.decisions

    for idx, block in enumerate(tr.transformer_blocks):
        block._seacache_fine_payload_block_idx = idx
        block._seacache_fine_payload_state_ref = state
    for idx, block in enumerate(tr.single_transformer_blocks):
        block._seacache_fine_payload_block_idx = idx
        block._seacache_fine_payload_state_ref = state

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformerBlock.forward = _original_double_forward
        FluxSingleTransformerBlock.forward = _original_single_forward

        for block in list(tr.transformer_blocks) + list(tr.single_transformer_blocks):
            for attr in (
                "_seacache_fine_payload_block_idx",
                "_seacache_fine_payload_state_ref",
            ):
                if hasattr(block, attr):
                    try:
                        delattr(block, attr)
                    except AttributeError:
                        pass
        for attr in (
            "_seacache_fine_payload_state",
            "seacache_payload_decisions",
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
    action_steps: Optional[Set[int]] = None,
    prompt_idx: Optional[int] = None,
) -> None:
    tr = pipe.transformer
    state: FineSeaPayloadState = tr._seacache_fine_payload_state
    state.reset_trajectory(action_steps=action_steps, prompt_idx=prompt_idx)
    tr.seacache_payload_decisions = state.decisions
