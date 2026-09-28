"""Research-only SeaCache gate with segment-boundary residual payloads.

This mode is between coarse whole-transformer residual reuse and 114-slot
fine feature caching. It partitions the FLUX transformer body into contiguous
block segments. On full steps it stores each segment's boundary residual over
the concatenated text/image state. On cached steps it skips the segment blocks
and injects a forecasted segment residual.

Locked baseline implementations are intentionally not touched.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import diffusers
import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from lib.gates import rel_l1
from lib.history_fd_observer import (
    forecast_predictions,
    init_state as init_history_state,
    update_on_full as history_update_on_full,
)
from lib.wiener import apply_sea_with_scheduler

logger = logging.get_logger(__name__)

_TESTED_DIFFUSERS_VERSIONS = {"0.38.0"}
EPS = 1e-12

PAYLOAD_MODES = (
    "segment_reuse",
    "segment_taylor_o1",
    "segment_taylor_o2",
    "segment_hicache_o2",
    "segment_ensemble_mean",
)

LAYOUTS = (
    "seg2",
    "double2_single1",
    "double1_single2",
    "seg4",
    "double4_single1",
    "double1_single4",
    "double2_single4",
    "double4_single2",
    "seg6",
    "seg8",
    "seg16",
    "block57",
    "double_all_single_blocks",
    "double_blocks_single_all",
    "late_fine",
    "coarse_mid_fine",
)

SegmentKey = Tuple[str, int, int]


def _check_diffusers_version() -> None:
    if diffusers.__version__ not in _TESTED_DIFFUSERS_VERSIONS:
        warnings.warn(
            "SeaCacheSegmentPayload was tested against diffusers "
            f"{sorted(_TESTED_DIFFUSERS_VERSIONS)}; running on "
            f"{diffusers.__version__} is untested.",
            stacklevel=3,
        )


def _norm(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())


def _sqrt_tensor_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    scalar = float(value.detach().to("cpu").item())
    if scalar <= 0.0:
        return 0.0
    return float(scalar ** 0.5)


def _split_ranges(n: int, parts: int) -> list[Tuple[int, int]]:
    if n <= 0 or parts <= 0:
        return []
    out = []
    for i in range(parts):
        start = int(round(i * n / parts))
        end = int(round((i + 1) * n / parts))
        if start < end:
            out.append((start, end))
    return out


def _single_blocks(start: int, end: int) -> list[Tuple[int, int]]:
    return [(i, i + 1) for i in range(int(start), int(end))]


def build_segments(layout: str, *, n_double: int, n_single: int) -> list[Dict[str, Any]]:
    """Return ordered segment specs over FLUX double and single block loops."""
    layout = str(layout)
    if layout not in LAYOUTS:
        raise ValueError(f"unknown segment layout: {layout!r}")

    double: list[Tuple[int, int]]
    single: list[Tuple[int, int]]
    if layout == "seg2":
        double = [(0, n_double)]
        single = [(0, n_single)]
    elif layout == "double2_single1":
        double = _split_ranges(n_double, 2)
        single = [(0, n_single)]
    elif layout == "double1_single2":
        double = [(0, n_double)]
        single = _split_ranges(n_single, 2)
    elif layout == "seg4":
        double = _split_ranges(n_double, 2)
        single = _split_ranges(n_single, 2)
    elif layout == "double4_single1":
        double = _split_ranges(n_double, 4)
        single = [(0, n_single)]
    elif layout == "double1_single4":
        double = [(0, n_double)]
        single = _split_ranges(n_single, 4)
    elif layout == "double2_single4":
        double = _split_ranges(n_double, 2)
        single = _split_ranges(n_single, 4)
    elif layout == "double4_single2":
        double = _split_ranges(n_double, 4)
        single = _split_ranges(n_single, 2)
    elif layout == "seg6":
        double = _split_ranges(n_double, 3)
        single = _split_ranges(n_single, 3)
    elif layout == "seg8":
        double = _split_ranges(n_double, 4)
        single = _split_ranges(n_single, 4)
    elif layout == "seg16":
        double = _split_ranges(n_double, 8)
        single = _split_ranges(n_single, 8)
    elif layout == "block57":
        double = _single_blocks(0, n_double)
        single = _single_blocks(0, n_single)
    elif layout == "double_all_single_blocks":
        double = [(0, n_double)]
        single = _single_blocks(0, n_single)
    elif layout == "double_blocks_single_all":
        double = _single_blocks(0, n_double)
        single = [(0, n_single)]
    elif layout == "late_fine":
        double = [(0, n_double)]
        single = [(0, min(25, n_single))] + _single_blocks(min(25, n_single), n_single)
    elif layout == "coarse_mid_fine":
        d0 = min(6, n_double)
        d1 = min(13, n_double)
        s0 = min(13, n_single)
        s1 = min(26, n_single)
        double = []
        if d0 > 0:
            double.append((0, d0))
        if d1 > d0:
            double.append((d0, d1))
        double += _single_blocks(d1, n_double)
        single = []
        if s0 > 0:
            single.append((0, s0))
        if s1 > s0:
            single.append((s0, s1))
        single += _single_blocks(s1, n_single)
    else:
        raise AssertionError(layout)

    specs: list[Dict[str, Any]] = []
    for start, end in double:
        specs.append({"kind": "double", "start": int(start), "end": int(end)})
    for start, end in single:
        specs.append({"kind": "single", "start": int(start), "end": int(end)})
    return specs


def _segment_key(spec: Dict[str, Any]) -> SegmentKey:
    return (str(spec["kind"]), int(spec["start"]), int(spec["end"]))


def _segment_id(spec: Dict[str, Any]) -> str:
    kind, start, end = _segment_key(spec)
    return f"{kind}:{start}-{end}"


def _spec_hash(specs: list[Dict[str, Any]]) -> str:
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _base_payload_mode(payload_mode: str) -> str:
    return {
        "segment_reuse": "reuse",
        "segment_taylor_o1": "taylor_o1",
        "segment_taylor_o2": "taylor_o2",
        "segment_hicache_o2": "hicache_o2",
        "segment_ensemble_mean": "ensemble_mean",
    }[str(payload_mode)]


@dataclass
class SegmentSeaPayloadState:
    threshold: float
    num_steps: int
    first_enhance: int
    payload_mode: str
    payload_sigma: float
    segment_layout: str
    segment_specs: list[Dict[str, Any]]
    segment_spec_hash: str
    action_steps: Optional[Set[int]] = None
    prompt_idx: Optional[int] = None
    decisions: list[Dict[str, Any]] = field(default_factory=list)
    history_by_segment: Dict[SegmentKey, Dict[str, Any]] = field(default_factory=dict)
    cnt: int = 0
    accumulated_rel_l1_distance: float = 0.0
    previous_modulated_input: Optional[torch.Tensor] = None
    last_full_step: Optional[int] = None
    previous_full_step: Optional[int] = None
    full_steps_seen: int = 0
    predicted_segments: int = 0
    updated_segments: int = 0
    payload_reuse_norm_sq: Optional[torch.Tensor] = None
    payload_forecast_norm_sq: Optional[torch.Tensor] = None
    payload_delta_norm_sq: Optional[torch.Tensor] = None
    unavailable_segments: list[str] = field(default_factory=list)

    @property
    def expected_segments(self) -> int:
        return len(self.segment_specs)

    def reset_trajectory(
        self,
        action_steps: Optional[Set[int]] = None,
        prompt_idx: Optional[int] = None,
    ) -> None:
        self.history_by_segment.clear()
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.last_full_step = None
        self.previous_full_step = None
        self.full_steps_seen = 0
        self.action_steps = None if action_steps is None else set(int(x) for x in action_steps)
        self.prompt_idx = None if prompt_idx is None else int(prompt_idx)
        self.decisions.clear()
        self.reset_step_accounting()

    def reset_step_accounting(self) -> None:
        self.predicted_segments = 0
        self.updated_segments = 0
        self.payload_reuse_norm_sq = None
        self.payload_forecast_norm_sq = None
        self.payload_delta_norm_sq = None
        self.unavailable_segments = []

    def _history(self, key: SegmentKey) -> Optional[Dict[str, Any]]:
        return self.history_by_segment.get(key)

    def available_order_range(self) -> tuple[Optional[int], Optional[int]]:
        orders = []
        for st in self.history_by_segment.values():
            history = (st or {}).get("history") or {}
            if history:
                orders.append(max(int(k) for k in history.keys()))
        if not orders:
            return None, None
        return min(orders), max(orders)

    def ready_segments(self) -> int:
        return sum(
            1
            for st in self.history_by_segment.values()
            if isinstance((st or {}).get("history"), dict) and 0 in ((st or {}).get("history") or {})
        )


def _combine(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.cat((encoder_hidden_states, hidden_states), dim=1)


def _split_combined(
    combined: torch.Tensor,
    text_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return combined[:, :text_seq_len], combined[:, text_seq_len:]


def _candidate_predictions(
    state: SegmentSeaPayloadState,
    key: SegmentKey,
    step: int,
) -> Dict[str, torch.Tensor]:
    history_state = state._history(key)
    return forecast_predictions(history_state, step=int(step), sigma=float(state.payload_sigma))


def _required_order(payload_mode: str) -> int:
    base = _base_payload_mode(str(payload_mode))
    if base == "reuse":
        return 0
    if base == "taylor_o1":
        return 1
    return 2


def _choose_prediction(
    state: SegmentSeaPayloadState,
    key: SegmentKey,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    preds = _candidate_predictions(state, key, step)
    reuse = preds.get("reuse")
    base = _base_payload_mode(state.payload_mode)
    if base == "ensemble_mean":
        parts = [
            preds.get(name)
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            if preds.get(name) is not None
        ]
        forecast = None if not parts else torch.stack([p.to(dtype=dtype, device=device) for p in parts], dim=0).mean(dim=0)
    else:
        forecast = preds.get(base)
    if reuse is None or forecast is None or tuple(reuse.shape) != tuple(forecast.shape):
        return None, None
    return reuse.to(dtype=dtype, device=device), forecast.to(dtype=dtype, device=device)


def _all_segments_available(state: SegmentSeaPayloadState, step: int) -> bool:
    del step
    required_order = _required_order(state.payload_mode)
    for spec in state.segment_specs:
        key = _segment_key(spec)
        st = state._history(key)
        history = (st or {}).get("history") or {}
        if 0 not in history:
            return False
        if max(int(k) for k in history.keys()) < int(required_order):
            return False
    return True


def _run_double_segment_full(
    self,
    *,
    start: int,
    end: int,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs,
    controlnet_block_samples,
    controlnet_blocks_repeat: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    for index_block in range(int(start), int(end)):
        block = self.transformer_blocks[index_block]
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
    return encoder_hidden_states, hidden_states


def _run_single_segment_full(
    self,
    *,
    start: int,
    end: int,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs,
    controlnet_single_block_samples,
) -> tuple[torch.Tensor, torch.Tensor]:
    for index_block in range(int(start), int(end)):
        block = self.single_transformer_blocks[index_block]
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
    return encoder_hidden_states, hidden_states


def _execute_segments(
    self,
    *,
    state: SegmentSeaPayloadState,
    should_calc: bool,
    step: int,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs,
    controlnet_block_samples,
    controlnet_single_block_samples,
    controlnet_blocks_repeat: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    for spec in state.segment_specs:
        key = _segment_key(spec)
        text_len = int(encoder_hidden_states.shape[1])
        before = _combine(encoder_hidden_states, hidden_states)
        if should_calc:
            if spec["kind"] == "double":
                encoder_hidden_states, hidden_states = _run_double_segment_full(
                    self,
                    start=int(spec["start"]),
                    end=int(spec["end"]),
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )
            else:
                encoder_hidden_states, hidden_states = _run_single_segment_full(
                    self,
                    start=int(spec["start"]),
                    end=int(spec["end"]),
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                )
            after = _combine(encoder_hidden_states, hidden_states)
            residual = after.detach() - before.detach()
            state.history_by_segment[key] = history_update_on_full(
                state.history_by_segment.get(key) or init_history_state(),
                residual=residual,
                step=int(step),
                max_order=2,
                sigma=float(state.payload_sigma),
            )
            state.updated_segments += 1
        else:
            reuse, forecast = _choose_prediction(
                state,
                key,
                step,
                dtype=before.dtype,
                device=before.device,
            )
            if reuse is None or forecast is None:
                state.unavailable_segments.append(_segment_id(spec))
                forecast = torch.zeros_like(before)
                reuse = torch.zeros_like(before)
            combined = before + forecast
            encoder_hidden_states, hidden_states = _split_combined(combined, text_len)
            state.predicted_segments += 1
    return encoder_hidden_states, hidden_states


def _scheduler_step_fields(state: SegmentSeaPayloadState, step: int) -> Dict[str, Any]:
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


def _seacache_segment_payload_forward(
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
    state: Optional[SegmentSeaPayloadState] = getattr(self, "_seacache_segment_payload_state", None)
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

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
        logger.warning("Passing `scale` via `joint_attention_kwargs` when not using PEFT is ineffective.")

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
    state.reset_step_accounting()

    first_block = self.transformer_blocks[0]
    modulated_inp, _gate_msa, _shift_mlp, _scale_mlp, _gate_mlp = first_block.norm1(
        hidden_states,
        emb=temb,
    )

    force_full_reason = None
    if cnt < int(state.first_enhance):
        force_full_reason = "first_enhance"
    elif cnt == 0:
        force_full_reason = "step0"
    elif cnt == int(state.num_steps) - 1:
        force_full_reason = "final_step"
    elif state.previous_modulated_input is None:
        force_full_reason = "no_previous_modulated_input"
    elif controlnet_block_samples is not None or controlnet_single_block_samples is not None:
        force_full_reason = "controlnet_unsupported"

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

    schedule_locked = state.action_steps is not None
    schedule_u = None if state.action_steps is None else int(cnt in state.action_steps)
    seacache_gate_cache_allowed = bool(force_full_reason is None and not native_should_calc)
    if force_full_reason is not None:
        should_calc = True
    elif schedule_locked:
        should_calc = not bool(schedule_u)
    else:
        should_calc = bool(native_should_calc)

    ready_pre = int(state.ready_segments())
    available_order_min_pre, available_order_max_pre = state.available_order_range()
    cache_ready_pre = bool(ready_pre >= int(state.expected_segments))
    if not should_calc and not _all_segments_available(state, cnt):
        force_full_reason = "segment_cache_unready"
        should_calc = True

    last_full_step_pre = state.last_full_step
    previous_full_step_pre = state.previous_full_step
    state.previous_modulated_input = modulated_for_state.detach()
    state.accumulated_rel_l1_distance = 0.0 if should_calc else float(accumulator_after_increment)

    encoder_hidden_states, hidden_states = _execute_segments(
        self,
        state=state,
        should_calc=bool(should_calc),
        step=cnt,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
    )

    if should_calc:
        state.previous_full_step = state.last_full_step
        state.last_full_step = cnt
        state.full_steps_seen += 1

    payload_available = None
    payload_fallback = None
    payload_used = None
    payload_fallback_reason = None
    if not should_calc:
        payload_available = bool(
            state.predicted_segments == int(state.expected_segments)
            and not state.unavailable_segments
        )
        payload_fallback = False
        payload_used = str(state.payload_mode)
    elif schedule_locked and schedule_u == 1:
        payload_available = False
        payload_fallback = True
        payload_fallback_reason = force_full_reason or "schedule_cache_forced_full"

    base_mode = _base_payload_mode(state.payload_mode)
    state.decisions.append({
        "step": int(cnt),
        "u": int(not should_calc),
        "native_u": int(not native_should_calc) if native_force_full_reason is None else 0,
        "seacache_gate_cache_allowed": bool(seacache_gate_cache_allowed),
        "combined_gate_cache_allowed": None,
        "combined_gate_veto_reason": None,
        "schedule_locked": bool(schedule_locked),
        "schedule_u": schedule_u,
        "threshold": threshold,
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
        "payload_mode": str(state.payload_mode),
        "payload_base_mode": base_mode,
        "payload_control": "none",
        "payload_blend": 1.0,
        "payload_used": payload_used,
        "payload_available": payload_available,
        "payload_fallback": payload_fallback,
        "payload_fallback_reason": payload_fallback_reason,
        "payload_reuse_norm": _sqrt_tensor_or_none(state.payload_reuse_norm_sq) if not should_calc else None,
        "payload_forecast_norm": _sqrt_tensor_or_none(state.payload_forecast_norm_sq) if not should_calc else None,
        "payload_chosen_norm": _sqrt_tensor_or_none(state.payload_forecast_norm_sq) if not should_calc else None,
        "payload_delta_from_reuse_norm": _sqrt_tensor_or_none(state.payload_delta_norm_sq) if not should_calc else None,
        "payload_norms_logged": False,
        "segment_payload_enabled": True,
        "segment_payload_layout": str(state.segment_layout),
        "segment_payload_mode": str(state.payload_mode),
        "segment_payload_base_mode": base_mode,
        "segment_payload_sigma": float(state.payload_sigma),
        "segment_payload_expected_segments": int(state.expected_segments),
        "segment_payload_ready_segments_pre": int(ready_pre),
        "segment_payload_missing_segments_pre": max(0, int(state.expected_segments) - int(ready_pre)),
        "segment_payload_cache_ready_pre": bool(cache_ready_pre),
        "segment_payload_predicted_segments": int(state.predicted_segments),
        "segment_payload_updated_segments": int(state.updated_segments),
        "segment_payload_unavailable_segments": list(state.unavailable_segments),
        "segment_payload_available_order_min_pre": (
            None if available_order_min_pre is None else int(available_order_min_pre)
        ),
        "segment_payload_available_order_max_pre": (
            None if available_order_max_pre is None else int(available_order_max_pre)
        ),
        "segment_payload_last_full_step_pre": (
            None if last_full_step_pre is None else int(last_full_step_pre)
        ),
        "segment_payload_previous_full_step_pre": (
            None if previous_full_step_pre is None else int(previous_full_step_pre)
        ),
        "segment_payload_last_full_step_post": (
            None if state.last_full_step is None else int(state.last_full_step)
        ),
        "segment_payload_full_steps_seen": int(state.full_steps_seen),
        "segment_payload_history_source": "full_refresh_only",
        "segment_payload_feature_space": "segment_boundary_tuple_residual",
        "segment_payload_granularity": f"segment_{state.segment_layout}",
        "segment_payload_spec_hash": str(state.segment_spec_hash),
        **_scheduler_step_fields(state, cnt),
    })

    state.cnt += 1
    if state.cnt == int(state.num_steps):
        state.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


_original_transformer_forward: Optional[Callable] = None


def install(
    pipe,
    *,
    threshold: float,
    num_steps: int,
    first_enhance: int = 1,
    payload_mode: str = "segment_taylor_o1",
    payload_sigma: float = 0.5,
    segment_layout: str = "seg8",
) -> Callable[[], None]:
    """Install SeaCacheSegmentPayload transformer forward."""
    global _original_transformer_forward

    if payload_mode not in PAYLOAD_MODES:
        raise ValueError(f"unknown segment payload mode: {payload_mode!r}")
    if segment_layout not in LAYOUTS:
        raise ValueError(f"unknown segment layout: {segment_layout!r}")

    _check_diffusers_version()
    if _original_transformer_forward is None:
        _original_transformer_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _seacache_segment_payload_forward

    tr = pipe.transformer
    specs = build_segments(
        str(segment_layout),
        n_double=len(tr.transformer_blocks),
        n_single=len(tr.single_transformer_blocks),
    )
    state = SegmentSeaPayloadState(
        threshold=float(threshold),
        num_steps=int(num_steps),
        first_enhance=int(first_enhance),
        payload_mode=str(payload_mode),
        payload_sigma=float(payload_sigma),
        segment_layout=str(segment_layout),
        segment_specs=specs,
        segment_spec_hash=_spec_hash(specs),
    )
    tr._seacache_segment_payload_state = state
    tr.scheduler = pipe.scheduler
    state.scheduler = pipe.scheduler
    tr.seacache_segment_payload_decisions = state.decisions

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = _original_transformer_forward
        for attr in (
            "_seacache_segment_payload_state",
            "scheduler",
            "seacache_segment_payload_decisions",
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
    state: SegmentSeaPayloadState = tr._seacache_segment_payload_state
    state.reset_trajectory(action_steps=action_steps, prompt_idx=prompt_idx)
    tr.seacache_segment_payload_decisions = state.decisions


def segment_metadata(pipe) -> Dict[str, Any]:
    state: SegmentSeaPayloadState = pipe.transformer._seacache_segment_payload_state
    return {
        "segment_payload_layout": str(state.segment_layout),
        "segment_payload_expected_segments": int(state.expected_segments),
        "segment_payload_spec_hash": str(state.segment_spec_hash),
        "segment_payload_specs": list(state.segment_specs),
    }
