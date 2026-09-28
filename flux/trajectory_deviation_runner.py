#!/usr/bin/env python3
"""Closed-loop trajectory deviation audit for FLUX cache policies.

This is an offline analysis runner for
`docs/research_plan_trajectory_deviation.md`.  It keeps the native cached
rollout as the driver, while optionally computing a read-only counterfactual
full output on the same cached latent.

The runner intentionally lives outside the locked baseline implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

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

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.oracle_runner import _decode_to_pil  # noqa: E402
from flux import seacache_fine_payload as seacache_fine_payload_modes  # noqa: E402
from flux import seacache_segment_payload as seacache_segment_payload_modes  # noqa: E402
from lib.gates import rel_l1  # noqa: E402
from lib.history_fd_observer import (  # noqa: E402
    clone_state as clone_history_fd_state,
    digest_payload as history_fd_digest_payload,
    ffro_residual_fields,
    forecast_predictions as history_fd_forecast_predictions,
    init_state as init_history_fd_state,
    online_fields as history_fd_online_fields,
    update_on_full as history_fd_update_on_full,
)
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402
from lib.sencache import (  # noqa: E402
    load_sensitivity_table,
    online_fields as sencache_online_fields,
    threshold_scale_from_latent,
)
from lib.teacache_coeffs import get_coeffs  # noqa: E402
from lib.wiener import apply_sea_with_scheduler  # noqa: E402

logger = logging.get_logger(__name__)

EPS = 1e-12
NATIVE_ROW_COMPARE_EXACT_FIELDS = [
    "step_index",
    "scheduler_step_index",
    "decision_u",
    "is_cached",
    "decision_reason",
    "cache_state_digest_before",
    "cache_state_digest_after_native_before_cf",
    "scheduler_state_digest_before_cf",
    "rng_state_digest_before_cf",
]
NATIVE_ROW_COMPARE_FLOAT_FIELDS = [
    "gate_increment_raw",
    "gate_increment_rescaled",
    "gate_displacement_to_anchor",
    "gate_tortuosity_log",
    "threshold_margin_before",
    "threshold_margin_after",
    "gate_accumulator_before",
    "gate_accumulator_after",
    "latent_drift_pre",
    "latent_drift_post",
    "o_drv_norm",
    "output_drift",
    "output_drift_rel",
    "scheduler_delta_cached_norm",
]
STEP_FIELDS = [
    "run_name", "prompt_id", "prompt", "seed", "step_index", "timestep",
    "sigma_n", "sigma_np1", "step_size_H", "scheduler_step_index",
    "mode", "cache_threshold", "decision_u", "is_cached", "decision_reason",
    "gate_increment_raw", "gate_increment_rescaled",
    "threshold_margin_before", "threshold_margin_after",
    "gate_accumulator_before", "gate_accumulator_after",
    "cache_state_digest_before", "cache_state_digest_after",
    "cache_state_digest_after_native_before_cf", "cache_state_digest_after_cf",
    "scheduler_state_digest_before_cf", "scheduler_state_digest_after_cf",
    "rng_state_digest_before_cf", "rng_state_digest_after_cf",
    "z_full_norm", "z_cached_norm", "latent_drift_pre", "latent_drift_pre_rel",
    "latent_drift_post", "latent_drift_post_rel",
    "o_full_norm", "o_drv_norm", "o_cf_norm",
    "output_drift", "output_drift_rel",
    "action_defect", "action_defect_rel", "action_defect_rel_to_full",
    "state_gap", "state_gap_rel",
    "dot_action_gap", "cos_action_gap", "projection_action_on_total",
    "action_latent_step_defect", "action_latent_step_defect_rel",
    "latent_state_scheduler_gap",
    "decomposition_closure_abs", "decomposition_closure_rel",
    "latent_decomposition_closure_abs", "latent_decomposition_closure_rel",
    "cos_drv_full", "cos_drv_cf", "cos_cf_full",
    "scheduler_delta_full_norm", "scheduler_delta_cached_norm",
    "full_scheduler_closure_abs_fp32", "full_scheduler_closure_rel_fp32",
    "cached_scheduler_closure_abs_fp32", "cached_scheduler_closure_rel_fp32",
    "cf_scheduler_closure_abs_fp32", "cf_scheduler_closure_rel_fp32",
    "payload_mode", "payload_base_mode", "payload_control",
    "payload_blend", "payload_sigma", "payload_used",
    "payload_available", "payload_fallback", "payload_reuse_norm",
    "payload_forecast_norm", "payload_chosen_norm",
    "payload_delta_from_reuse_norm", "schedule_locked", "schedule_u",
    "native_u",
    "fine_payload_enabled", "fine_payload_method", "fine_payload_mode",
    "fine_payload_order", "fine_payload_sigma", "fine_payload_expected_slots",
    "fine_payload_slots_ready_pre", "fine_payload_slots_missing_pre",
    "fine_payload_cache_ready_pre", "fine_payload_predicted_slots",
    "fine_payload_updated_slots", "fine_payload_step_offset",
    "fine_payload_step_gap", "fine_payload_effective_predict_order",
    "fine_payload_prediction_order_degraded",
    "segment_payload_enabled", "segment_payload_layout", "segment_payload_mode",
    "segment_payload_base_mode", "segment_payload_sigma",
    "segment_payload_expected_segments", "segment_payload_ready_segments_pre",
    "segment_payload_missing_segments_pre", "segment_payload_cache_ready_pre",
    "segment_payload_predicted_segments", "segment_payload_updated_segments",
    "segment_payload_unavailable_segments",
    "segment_payload_available_order_min_pre", "segment_payload_available_order_max_pre",
    "segment_payload_last_full_step_pre", "segment_payload_previous_full_step_pre",
    "segment_payload_last_full_step_post", "segment_payload_full_steps_seen",
    "segment_payload_history_source", "segment_payload_feature_space",
    "segment_payload_granularity", "segment_payload_spec_hash",
]

BASE_PAYLOAD_MODES = (
    "reuse", "taylor_o1", "taylor_o2", "hicache_o2", "ensemble_mean",
)
PAYLOAD_CONTROL_SUFFIXES = (
    "_shift_m1",
    "_shift_p1",
    "_mirror_step",
    "_norm_only",
    "_random_dir",
    "_delta_negative",
    "_delta_random_dir",
    "_delta_orthogonal_random",
)
PAYLOAD_MODES = set(BASE_PAYLOAD_MODES) | {
    f"{base}{suffix}"
    for base in BASE_PAYLOAD_MODES
    if base != "reuse"
    for suffix in PAYLOAD_CONTROL_SUFFIXES
}
FINE_PAYLOAD_MODES = set(seacache_fine_payload_modes.PAYLOAD_MODES)
SEGMENT_PAYLOAD_MODES = set(seacache_segment_payload_modes.PAYLOAD_MODES)
SEGMENT_LAYOUTS = set(seacache_segment_payload_modes.LAYOUTS)


def _parse_int_list(text: Optional[str]) -> Tuple[int, ...]:
    if text is None:
        return ()
    vals: List[int] = []
    for token in str(text).replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"depths must be positive integers, got {value}")
        vals.append(value)
    return tuple(sorted(set(vals)))


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).norm().item())


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(torch.float32) * b.detach().to(torch.float32)).sum().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    an = _norm(a)
    bn = _norm(b)
    if an <= 0.0 or bn <= 0.0:
        return None
    return _dot(a, b) / (an * bn + EPS)


def _block_direction_sketch(t: Optional[torch.Tensor], dims: int) -> Dict[str, Any]:
    """Cheap deterministic direction sketch over an online-visible tensor.

    This is intentionally not a learned projection.  It projects the normalized
    flattened tensor onto block-constant basis vectors, which is enough to test
    whether coarse direction information carries held-out signal beyond scalar
    norms.  The sketch is only used by research audits and never by locked
    baseline methods.
    """
    if t is None or dims <= 0:
        return {"present": False}
    x = t.detach().to(torch.float32).flatten()
    if x.numel() == 0:
        return {"present": False}
    norm = float(x.norm().item())
    if norm <= 0.0:
        return {"present": False, "norm": 0.0}
    chunks = torch.chunk(x / (norm + EPS), int(dims))
    vals = []
    for chunk in chunks:
        vals.append(float(chunk.sum().item() / math.sqrt(max(int(chunk.numel()), 1))))
    while len(vals) < int(dims):
        vals.append(0.0)
    return {
        "present": True,
        "norm": norm,
        "sketch_l2": math.sqrt(sum(v * v for v in vals)),
        "values": vals[:int(dims)],
    }


def _online_direction_sketch_fields(
    *,
    gate_current: Optional[torch.Tensor],
    gate_previous: Optional[torch.Tensor],
    previous_residual: Optional[torch.Tensor],
    dims: int,
) -> Dict[str, Any]:
    if dims <= 0:
        return {}
    out: Dict[str, Any] = {"online_direction_sketch_dims": int(dims)}
    gate_diff = None
    if gate_current is not None and gate_previous is not None and gate_current.shape == gate_previous.shape:
        gate_diff = gate_current - gate_previous
        cos_res = (
            _cos(gate_diff, previous_residual)
            if previous_residual is not None and previous_residual.shape == gate_diff.shape
            else None
        )
        out["online_gate_diff_prev_residual_cos"] = cos_res
    else:
        out["online_gate_diff_prev_residual_cos"] = None

    for prefix, tensor in (
        ("online_gate_diff", gate_diff),
        ("online_prev_residual", previous_residual),
    ):
        sketch = _block_direction_sketch(tensor, dims)
        out[f"{prefix}_present"] = bool(sketch.get("present"))
        out[f"{prefix}_norm"] = sketch.get("norm")
        out[f"{prefix}_sketch_l2"] = sketch.get("sketch_l2")
        vals = sketch.get("values") or []
        for idx in range(int(dims)):
            out[f"{prefix}_sketch_{idx:02d}"] = vals[idx] if idx < len(vals) else None
    return out


def _ffro_velocity_head_fields(
    transformer,
    *,
    history_state: Optional[Dict[str, Any]],
    step: int,
    ori_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    step_size_abs: Optional[float],
) -> Dict[str, Any]:
    """Velocity-head FFRO scores without running transformer blocks."""

    preds = history_fd_forecast_predictions(
        history_state,
        step=int(step),
        sigma=float(getattr(transformer, "_td_history_fd_sigma", 0.5)),
    )
    reuse = preds.get("reuse")
    if reuse is None:
        return {
            "online_ffro_vel_available_pre": False,
            "online_ffro_vel_elapsed_ms_pre": 0.0,
        }
    start = time.perf_counter()
    with torch.no_grad():
        v_reuse = transformer.proj_out(transformer.norm_out(ori_hidden_states + reuse, temb))
        out: Dict[str, Any] = {
            "online_ffro_vel_available_pre": True,
            "online_ffro_reuse_velocity_norm_pre": _norm(v_reuse),
        }
        vals: List[float] = []
        for name, suffix in (
            ("taylor_o1", "taylor_o1"),
            ("taylor_o2", "taylor_o2"),
            ("hicache_o2", "hicache_o2"),
        ):
            pred = preds.get(name)
            if pred is None or pred.shape != reuse.shape:
                out[f"online_ffro_{suffix}_vel_norm_pre"] = None
                out[f"online_ffro_{suffix}_step_norm_pre"] = None
                continue
            v_pred = transformer.proj_out(transformer.norm_out(ori_hidden_states + pred, temb))
            defect = _norm(v_reuse.detach().to(torch.float32) - v_pred.detach().to(torch.float32))
            out[f"online_ffro_{suffix}_vel_norm_pre"] = defect
            out[f"online_ffro_{suffix}_step_norm_pre"] = (
                None if step_size_abs is None else float(step_size_abs) * defect
            )
            vals.append(defect)
        if vals:
            out["online_ffro_ensemble_vel_min_pre"] = min(vals)
            out["online_ffro_ensemble_vel_mean_pre"] = float(sum(vals) / len(vals))
            out["online_ffro_ensemble_vel_max_pre"] = max(vals)
            if step_size_abs is not None:
                out["online_ffro_ensemble_step_min_pre"] = float(step_size_abs) * min(vals)
                out["online_ffro_ensemble_step_mean_pre"] = float(step_size_abs) * float(sum(vals) / len(vals))
                out["online_ffro_ensemble_step_max_pre"] = float(step_size_abs) * max(vals)
        out["online_ffro_vel_elapsed_ms_pre"] = (time.perf_counter() - start) * 1000.0
        return out


def _flowmatch_euler_post(
    sample_f32: torch.Tensor,
    step_size: float,
    model_output_f32: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror diffusers FlowMatchEulerDiscreteScheduler.step() for this runner.

    The scheduler computes the Euler update in fp32 and then casts the previous
    sample back to the model output dtype.  FLUX bf16 runs therefore follow
    Q(z + h v), not an exact fp32 z + h v map.
    """
    native_update = float(step_size) * model_output_f32.to(output_dtype)
    return (sample_f32 + native_update).to(output_dtype).to(torch.float32)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tensor_digest(t: Optional[torch.Tensor]) -> Dict[str, Any]:
    if t is None:
        return {"present": False}
    td = t.detach()
    tf = td.to(torch.float32)
    raw = td.to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "present": True,
        "shape": list(td.shape),
        "dtype": str(td.dtype),
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
        "norm": float(tf.norm().item()),
        "mean": float(tf.mean().item()),
        "sum": float(tf.sum().item()),
    }


def _state_digest(tr) -> str:
    fine_state = getattr(tr, "_seacache_fine_payload_state", None)
    fine_payload = None
    if fine_state is not None:
        fine_payload = {
            "cnt": int(getattr(fine_state, "cnt", -1)),
            "slots": int(len(getattr(fine_state, "cache_dic", {}) or {})),
            "last_full_step": getattr(fine_state, "last_full_step", None),
            "previous_full_step": getattr(fine_state, "previous_full_step", None),
            "full_steps_seen": int(getattr(fine_state, "full_steps_seen", 0)),
            "should_skip": bool(getattr(fine_state, "should_skip", False)),
            "step_offset": int(getattr(fine_state, "step_offset", 0)),
            "step_gap": int(getattr(fine_state, "step_gap", 0)),
            "effective_max_order": int(getattr(fine_state, "effective_max_order", 0)),
        }
    segment_state = getattr(tr, "_seacache_segment_payload_state", None)
    segment_payload = None
    if segment_state is not None:
        segment_payload = {
            "cnt": int(getattr(segment_state, "cnt", -1)),
            "layout": str(getattr(segment_state, "segment_layout", "")),
            "expected_segments": int(getattr(segment_state, "expected_segments", 0)),
            "histories": int(len(getattr(segment_state, "history_by_segment", {}) or {})),
            "last_full_step": getattr(segment_state, "last_full_step", None),
            "previous_full_step": getattr(segment_state, "previous_full_step", None),
            "full_steps_seen": int(getattr(segment_state, "full_steps_seen", 0)),
            "spec_hash": str(getattr(segment_state, "segment_spec_hash", "")),
        }
    payload = {
        "cnt": int(getattr(tr, "cnt", -1)),
        "acc": float(getattr(tr, "accumulated_rel_l1_distance", 0.0)),
        "prev_mod": _tensor_digest(getattr(tr, "previous_modulated_input", None)),
        "prev_res": _tensor_digest(getattr(tr, "previous_residual", None)),
        "history_fd": history_fd_digest_payload(getattr(tr, "_td_history_fd_state", None)),
        "fine": fine_payload,
        "segment": segment_payload,
    }
    return _hash_text(json.dumps(payload, sort_keys=True))


def _scheduler_digest(scheduler) -> str:
    payload = {
        "step_index": getattr(scheduler, "_step_index", None),
        "timesteps_hash": _hash_text(str(getattr(scheduler, "timesteps", None))),
        "sigmas_hash": _hash_text(str(getattr(scheduler, "sigmas", None))),
    }
    return _hash_text(json.dumps(payload, sort_keys=True, default=str))


@contextmanager
def _fine_block_refs_disabled(transformer) -> Iterator[None]:
    """Temporarily force patched fine blocks to call their original forwards."""
    saved = []
    for block in list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks):
        had_ref = hasattr(block, "_seacache_fine_payload_state_ref")
        ref = getattr(block, "_seacache_fine_payload_state_ref", None)
        saved.append((block, had_ref, ref))
        if had_ref:
            delattr(block, "_seacache_fine_payload_state_ref")
    try:
        yield
    finally:
        for block, had_ref, ref in saved:
            if had_ref:
                block._seacache_fine_payload_state_ref = ref


def _load_payload_action_steps(
    schedule_dir: Optional[Path],
    prompt_id: int,
    *,
    expected_num_steps: Optional[int] = None,
    require_reuse_reference: bool = False,
) -> Optional[Set[int]]:
    if schedule_dir is None:
        return None
    candidates = [
        schedule_dir / f"decisions_{prompt_id:05d}.json",
        schedule_dir / f"prompt_{prompt_id:05d}" / "decisions.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        joined = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"missing payload schedule for prompt {prompt_id}: tried {joined}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        schedule_prompt_idx = payload.get("prompt_idx")
        if schedule_prompt_idx is not None and int(schedule_prompt_idx) != int(prompt_id):
            raise ValueError(
                f"schedule prompt_idx mismatch in {path}: expected {prompt_id}, got {schedule_prompt_idx}"
            )
        if require_reuse_reference:
            schedule_mode = payload.get("mode")
            if schedule_mode is not None and str(schedule_mode) != "SeaCachePayload":
                raise ValueError(
                    f"schedule {path} must come from SeaCachePayload(reuse); got mode={schedule_mode!r}"
                )
            schedule_payload_mode = payload.get("payload_mode")
            if schedule_payload_mode is not None and str(schedule_payload_mode) != "reuse":
                raise ValueError(
                    f"schedule {path} must come from SeaCachePayload(reuse); "
                    f"got payload_mode={schedule_payload_mode!r}"
                )
        rows = payload.get("per_step") or payload.get("decisions") or []
    else:
        raise ValueError(f"unsupported schedule JSON in {path}: {type(payload).__name__}")
    if not isinstance(rows, list):
        raise ValueError(f"schedule rows in {path} are not a list")
    if expected_num_steps is not None and int(expected_num_steps) > 0:
        steps = [
            int(row.get("step"))
            for row in rows
            if isinstance(row, dict) and row.get("step") is not None
        ]
        expected = list(range(int(expected_num_steps)))
        if steps != expected:
            raise ValueError(
                f"schedule step sequence mismatch in {path}: expected 0.."
                f"{int(expected_num_steps) - 1}, got {steps[:10]}... len={len(steps)}"
            )
    return {
        int(row["step"])
        for row in rows
        if isinstance(row, dict) and int(row.get("u", 0)) == 1
    }


def _empty_payload_fields() -> Dict[str, Any]:
    return {
        "payload_mode": None,
        "payload_base_mode": None,
        "payload_control": None,
        "payload_blend": None,
        "payload_sigma": None,
        "payload_used": None,
        "payload_available": None,
        "payload_fallback": None,
        "payload_reuse_norm": None,
        "payload_forecast_norm": None,
        "payload_chosen_norm": None,
        "payload_delta_from_reuse_norm": None,
        "schedule_locked": False,
        "schedule_u": None,
        "native_u": None,
    }


def _mode_uses_fixed_payload_schedule(mode: str) -> bool:
    return str(mode) in ("SeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload")


def _mode_uses_seacache_gate(mode: str) -> bool:
    return str(mode) in ("SeaCache", "SeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload")


def _payload_spec(mode: str) -> Tuple[str, str]:
    for suffix in PAYLOAD_CONTROL_SUFFIXES:
        if mode.endswith(suffix):
            base = mode[: -len(suffix)]
            if base in BASE_PAYLOAD_MODES and base != "reuse":
                return base, suffix[1:]
    return mode, "none"


def _payload_base_for_mode(mode: str, payload_mode: str) -> Tuple[str, str]:
    if str(mode) == "SeaCacheFinePayload":
        return seacache_fine_payload_modes._payload_base_mode(str(payload_mode)), "none"
    if str(mode) == "SeaCacheSegmentPayload":
        return seacache_segment_payload_modes._base_payload_mode(str(payload_mode)), "none"
    return _payload_spec(str(payload_mode))


def _sqrt_tensor_or_none(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    scalar = float(value.detach().to("cpu").item())
    if scalar <= 0.0:
        return 0.0
    if not math.isfinite(scalar):
        return None
    return float(math.sqrt(scalar))


def _stable_control_seed(mode: str, step: int, shape: Tuple[int, ...], salt: int = 0) -> int:
    payload = f"{mode}|{int(step)}|{int(salt)}|{','.join(str(int(x)) for x in shape)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _random_direction_like(
    reference: torch.Tensor,
    *,
    mode: str,
    step: int,
    salt: int = 0,
) -> torch.Tensor:
    ref = reference.detach().to(torch.float32)
    ref_norm = float(ref.norm().item())
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(_stable_control_seed(mode, int(step), tuple(reference.shape), int(salt)))
    noise = torch.randn(reference.shape, device=reference.device, generator=gen, dtype=torch.float32)
    noise_norm = float(noise.norm().item())
    if ref_norm <= 0.0 or noise_norm <= 0.0:
        return torch.zeros_like(reference)
    return (noise / (noise_norm + EPS) * ref_norm).to(dtype=reference.dtype, device=reference.device)


def _orthogonal_random_direction_like(
    reference: torch.Tensor,
    *,
    mode: str,
    step: int,
    salt: int = 0,
) -> torch.Tensor:
    ref = reference.detach().to(torch.float32)
    ref_norm = float(ref.norm().item())
    if ref_norm <= 0.0:
        return torch.zeros_like(reference)
    rand = _random_direction_like(reference, mode=mode, step=step, salt=int(salt)).detach().to(torch.float32)
    rand = rand - (torch.sum(rand * ref) / (ref_norm * ref_norm + EPS)) * ref
    rand_norm = float(rand.norm().item())
    if rand_norm <= 0.0:
        return torch.zeros_like(reference)
    return (rand / (rand_norm + EPS) * ref_norm).to(dtype=reference.dtype, device=reference.device)


def _reuse_direction_with_norm(reuse: torch.Tensor, reference: torch.Tensor) -> Optional[torch.Tensor]:
    reuse_f = reuse.detach().to(torch.float32)
    ref_norm = float(reference.detach().to(torch.float32).norm().item())
    reuse_norm = float(reuse_f.norm().item())
    if reuse_norm <= 0.0 or ref_norm <= 0.0:
        return None
    return (reuse_f / (reuse_norm + EPS) * ref_norm).to(dtype=reuse.dtype, device=reuse.device)


def _forecast_payload(
    *,
    payload_mode: str,
    payload_blend: float,
    payload_sigma: float,
    reuse: torch.Tensor,
    history_state: Optional[Dict[str, Any]],
    step: int,
    num_steps: Optional[int] = None,
    payload_control_seed_salt: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    base_mode, control = _payload_spec(str(payload_mode))
    if payload_mode == "reuse":
        return reuse, {
            "payload_mode": "reuse",
            "payload_base_mode": "reuse",
            "payload_control": "none",
            "payload_control_seed_salt": int(payload_control_seed_salt),
            "payload_blend": float(payload_blend),
            "payload_sigma": float(payload_sigma),
            "payload_used": "reuse",
            "payload_available": True,
            "payload_fallback": False,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": None,
            "payload_chosen_norm": _norm(reuse),
            "payload_delta_from_reuse_norm": 0.0,
        }

    control_step = int(step)
    if control == "shift_m1":
        control_step = max(int(step) - 1, 0)
    elif control == "shift_p1":
        control_step = int(step) + 1
    elif control == "mirror_step" and num_steps is not None:
        control_step = max(int(num_steps) - 1 - int(step), 0)

    preds = history_fd_forecast_predictions(
        history_state,
        step=int(control_step),
        sigma=float(payload_sigma),
    )
    if base_mode == "ensemble_mean":
        parts = [
            pred
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            for pred in [preds.get(name)]
            if pred is not None and pred.shape == reuse.shape
        ]
        forecast = None if not parts else torch.stack(
            [part.to(dtype=reuse.dtype, device=reuse.device) for part in parts], dim=0
        ).mean(dim=0)
    else:
        forecast = preds.get(base_mode)

    if forecast is None or forecast.shape != reuse.shape:
        return reuse, {
            "payload_mode": str(payload_mode),
            "payload_base_mode": str(base_mode),
            "payload_control": str(control),
            "payload_control_seed_salt": int(payload_control_seed_salt),
            "payload_blend": float(payload_blend),
            "payload_sigma": float(payload_sigma),
            "payload_used": "reuse",
            "payload_available": False,
            "payload_fallback": True,
            "payload_reuse_norm": _norm(reuse),
            "payload_forecast_norm": None,
            "payload_chosen_norm": _norm(reuse),
            "payload_delta_from_reuse_norm": 0.0,
        }

    forecast = forecast.to(dtype=reuse.dtype, device=reuse.device)
    controlled_forecast = forecast
    if control == "norm_only":
        norm_only = _reuse_direction_with_norm(reuse, forecast)
        if norm_only is None:
            return reuse, {
                "payload_mode": str(payload_mode),
                "payload_base_mode": str(base_mode),
                "payload_control": str(control),
                "payload_control_seed_salt": int(payload_control_seed_salt),
                "payload_blend": float(payload_blend),
                "payload_sigma": float(payload_sigma),
                "payload_used": "reuse",
                "payload_available": False,
                "payload_fallback": True,
                "payload_reuse_norm": _norm(reuse),
                "payload_forecast_norm": _norm(forecast),
                "payload_chosen_norm": _norm(reuse),
                "payload_delta_from_reuse_norm": 0.0,
            }
        controlled_forecast = norm_only
    elif control == "random_dir":
        controlled_forecast = _random_direction_like(
            forecast,
            mode=str(payload_mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )
    elif control == "delta_negative":
        controlled_forecast = reuse - (forecast - reuse)
    elif control == "delta_random_dir":
        controlled_forecast = reuse + _random_direction_like(
            forecast - reuse,
            mode=str(payload_mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )
    elif control == "delta_orthogonal_random":
        controlled_forecast = reuse + _orthogonal_random_direction_like(
            forecast - reuse,
            mode=str(payload_mode),
            step=int(step),
            salt=int(payload_control_seed_salt),
        )

    if float(payload_blend) >= 1.0:
        chosen = controlled_forecast
        payload_used = str(payload_mode)
    else:
        chosen = (1.0 - float(payload_blend)) * reuse + float(payload_blend) * controlled_forecast
        payload_used = f"blend_{payload_mode}"
    return chosen, {
        "payload_mode": str(payload_mode),
        "payload_base_mode": str(base_mode),
        "payload_control": str(control),
        "payload_control_seed_salt": int(payload_control_seed_salt),
        "payload_blend": float(payload_blend),
        "payload_sigma": float(payload_sigma),
        "payload_used": payload_used,
        "payload_available": True,
        "payload_fallback": False,
        "payload_reuse_norm": _norm(reuse),
        "payload_forecast_norm": _norm(forecast),
        "payload_chosen_norm": _norm(chosen),
        "payload_delta_from_reuse_norm": _norm(
            chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)
        ),
    }


def _rng_digest(device: torch.device) -> str:
    parts = [hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()]
    if device.type == "cuda":
        parts.append(hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest())
    return _hash_text("|".join(parts))


def _capture_rng_state(device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng_state(device: torch.device, state: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _git_commit() -> Tuple[Optional[str], Optional[bool]]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_PROJECT_ROOT, text=True
        ).strip())
        return sha, dirty
    except Exception:
        return None, None


def _full_blocks(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    controlnet_blocks_repeat: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

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

    return encoder_hidden_states, hidden_states


def _prefix_shadow_fields(
    self,
    *,
    ori_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    depths: Tuple[int, ...],
    update_anchor: bool,
    emit_metrics: bool,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    controlnet_blocks_repeat: bool = False,
) -> Dict[str, Any]:
    """Run a read-only transformer prefix and record scalar shadow metrics.

    Depth counts FLUX transformer blocks in execution order: first the
    dual-stream blocks, then the single-stream blocks.  The observer never
    mutates cache/gate/scheduler state; the only persistent state it updates is
    its own anchor dictionary when the native action is full.
    """
    if not depths:
        return {}
    max_depth = max(depths)
    n_double = len(self.transformer_blocks)
    n_single = len(self.single_transformer_blocks)
    max_allowed = n_double + n_single
    if max_depth > max_allowed:
        raise RuntimeError(f"shadow depth {max_depth} exceeds FLUX block count {max_allowed}")

    anchors: Dict[int, torch.Tensor] = getattr(self, "_td_shadow_anchor_residuals", {})
    next_anchors: Dict[int, torch.Tensor] = {}
    fields: Dict[str, Any] = {
        "shadow_depths": ",".join(str(d) for d in depths),
        "shadow_emit_metrics": bool(emit_metrics),
        "shadow_update_anchor": bool(update_anchor),
    }
    wanted = set(depths)
    hidden = ori_hidden_states.detach().clone()
    enc = encoder_hidden_states.detach().clone()
    t0 = time.perf_counter()

    def _record(depth: int, h: torch.Tensor) -> None:
        residual = (h - ori_hidden_states).detach()
        next_anchors[depth] = residual
        prefix = f"online_shadow_d{depth:02d}"
        anchor = anchors.get(depth)
        fields[f"{prefix}_residual_norm"] = _norm(residual)
        fields[f"{prefix}_anchor_present"] = bool(anchor is not None and anchor.shape == residual.shape)
        fields[f"{prefix}_elapsed_ms"] = float((time.perf_counter() - t0) * 1000.0)
        if emit_metrics and anchor is not None and anchor.shape == residual.shape:
            diff = residual - anchor
            fields[f"{prefix}_residual_to_anchor"] = _norm(diff)
            fields[f"{prefix}_residual_to_anchor_rel"] = _norm(diff) / (_norm(anchor) + EPS)
            fields[f"{prefix}_cos_anchor"] = _cos(residual, anchor)
            fields[f"{prefix}_anchor_residual_norm"] = _norm(anchor)
        else:
            fields[f"{prefix}_residual_to_anchor"] = None
            fields[f"{prefix}_residual_to_anchor_rel"] = None
            fields[f"{prefix}_cos_anchor"] = None
            fields[f"{prefix}_anchor_residual_norm"] = _norm(anchor) if anchor is not None else None

    for depth in range(1, max_depth + 1):
        if depth <= n_double:
            index_block = depth - 1
            block = self.transformer_blocks[index_block]
            enc, hidden = block(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(n_double / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden = hidden + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden = hidden + controlnet_block_samples[index_block // interval_control]
        else:
            index_block = depth - n_double - 1
            block = self.single_transformer_blocks[index_block]
            enc, hidden = block(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(n_single / len(controlnet_single_block_samples)))
                hidden = hidden + controlnet_single_block_samples[index_block // interval_control]
        if depth in wanted:
            _record(depth, hidden)

    if update_anchor:
        # Keep anchors on device for the next online comparison.  One tensor per
        # requested depth is acceptable for the small depth sweeps used here.
        self._td_shadow_anchor_residuals = {
            depth: tensor.detach()
            for depth, tensor in next_anchors.items()
            if depth in wanted
        }
    return fields


def _td_forward(
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

    sencache_latent_for_gate = hidden_states.detach()
    sencache_timestep_for_gate = float(timestep.detach().to(torch.float32).reshape(-1)[0].item() * 1000.0)
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
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
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

    run_kind = getattr(self, "_td_run_kind", "full")
    mode = getattr(self, "_td_mode", "SeaCache")
    with_cf = bool(getattr(self, "_td_with_cf", True))
    forced_action = getattr(self, "_td_force_action", None)
    if forced_action in ("", "none", "None"):
        forced_action = None
    if forced_action not in (None, "full", "cache"):
        raise RuntimeError(f"unsupported forced trajectory action: {forced_action!r}")
    if run_kind == "cached":
        cur_step = int(getattr(self, "cnt", 0))
    else:
        cur_step = int(getattr(self, "_td_step", 0))

    ori_hidden_states = hidden_states
    decision = "full"
    decision_reason = "full_trace"
    is_cached = False
    force_full = False
    gate_inc_raw = None
    gate_inc_rescaled = None
    gate_disp_anchor = None
    gate_tortuosity_log = None
    threshold_margin_before = None
    threshold_margin_after = None
    acc_before = float(getattr(self, "accumulated_rel_l1_distance", 0.0))
    acc_after = acc_before
    native_decision = "full"
    native_decision_reason = "full_trace"
    cache_digest_before = _state_digest(self)
    cache_digest_after_native_before_cf = None
    scheduler_digest_before_cf = None
    rng_digest_before_cf = None
    online_direction_fields: Dict[str, Any] = {}
    shadow_fields: Dict[str, Any] = {}
    sencache_fields: Dict[str, Any] = {}
    history_fd_fields: Dict[str, Any] = {}
    ffro_fields: Dict[str, Any] = {}
    payload_fields: Dict[str, Any] = _empty_payload_fields()
    fine_state = (
        getattr(self, "_seacache_fine_payload_state", None)
        if mode == "SeaCacheFinePayload"
        else None
    )
    segment_state = (
        getattr(self, "_seacache_segment_payload_state", None)
        if mode == "SeaCacheSegmentPayload"
        else None
    )
    fine_cache_ready_pre = None
    fine_last_full_step_pre = None
    fine_previous_full_step_pre = None
    fine_available_order_min_pre = None
    fine_available_order_max_pre = None
    fine_effective_predict_order = None
    fine_payload_fallback_reason = None
    segment_cache_ready_pre = None
    segment_ready_pre = None
    segment_available_order_min_pre = None
    segment_available_order_max_pre = None
    segment_last_full_step_pre = None
    segment_previous_full_step_pre = None
    segment_payload_fallback_reason = None

    if run_kind == "cached":
        first_block = self.transformer_blocks[0]
        modulated_inp, _, _, _, _ = first_block.norm1(ori_hidden_states, emb=temb)
        previous_modulated_input_for_gate = getattr(self, "previous_modulated_input", None)
        anchor_modulated_input_for_gate = getattr(self, "_td_anchor_modulated_input", None)
        previous_residual_for_gate = getattr(self, "previous_residual", None)
        gate_feature_current = modulated_inp
        history_fd_state_pre = clone_history_fd_state(getattr(self, "_td_history_fd_state", None))
        history_fd_fields = history_fd_online_fields(
            history_fd_state_pre,
            step=cur_step,
            sigma=float(getattr(self, "_td_history_fd_sigma", 0.5)),
        )
        step_size_abs = None
        sigmas = getattr(getattr(self, "scheduler", None), "sigmas", None)
        if sigmas is not None and cur_step + 1 < len(sigmas):
            step_size_abs = abs(float((sigmas[cur_step + 1] - sigmas[cur_step]).detach().to("cpu").item()))
        ffro_fields = ffro_residual_fields(
            history_fd_state_pre,
            step=cur_step,
            sigma=float(getattr(self, "_td_history_fd_sigma", 0.5)),
        )
        ffro_fields.update(_ffro_velocity_head_fields(
            self,
            history_state=history_fd_state_pre,
            step=cur_step,
            ori_hidden_states=ori_hidden_states,
            temb=temb,
            step_size_abs=step_size_abs,
        ))
        sencache_fields = sencache_online_fields(
            table=getattr(self, "_td_sencache_table", None),
            current_latent=sencache_latent_for_gate,
            current_timestep=sencache_timestep_for_gate,
            anchor_latent=getattr(self, "_td_sencache_anchor_latent", None),
            anchor_timestep=getattr(self, "_td_sencache_anchor_timestep", None),
            anchor_step=getattr(self, "_td_sencache_anchor_step", None),
        )
        force_full = (
            cur_step < int(getattr(self, "first_enhance", 1))
            or cur_step == 0
            or cur_step == int(getattr(self, "num_steps", 0)) - 1
            or (
                mode == "SenCache"
                and cur_step >= max(min(
                    (
                        int(getattr(self, "num_steps", 0)) - 1
                        if int(getattr(self, "_td_sencache_cutoff_steps", -1)) < 0
                        else int(getattr(self, "_td_sencache_cutoff_steps", -1))
                    ),
                    int(getattr(self, "num_steps", 0)),
                ), 0)
            )
            or previous_modulated_input_for_gate is None
            or (
                mode == "SenCache"
                and getattr(self, "_td_sencache_anchor_latent", None) is None
            )
        )
        should_calc = True
        native_should_calc = True
        native_acc_after = 0.0
        if force_full:
            self.accumulated_rel_l1_distance = 0.0
            decision_reason = "force_full"
            native_decision_reason = decision_reason
        else:
            if _mode_uses_seacache_gate(mode):
                mod_for_gate = modulated_inp.reshape(
                    modulated_inp.shape[0],
                    int(img_ids[:, 1].max().item() + 1),
                    int(img_ids[:, 2].max().item() + 1),
                    modulated_inp.shape[-1],
                )
                mod_for_gate = apply_sea_with_scheduler(
                    mod_for_gate,
                    self.scheduler,
                    cur_step,
                    power_exp=2.0,
                    dims=(-2, -3),
                    norm_mode="mean",
                )
                mod_for_gate = mod_for_gate.reshape(mod_for_gate.shape[0], -1, mod_for_gate.shape[-1])
                gate_inc_raw = rel_l1(mod_for_gate, self.previous_modulated_input)
                gate_inc_rescaled = gate_inc_raw
                modulated_inp = mod_for_gate
                gate_feature_current = modulated_inp
            elif mode == "TeaCache":
                gate_inc_raw = rel_l1(modulated_inp, self.previous_modulated_input)
                gate_inc_rescaled = float(self.teacache_rescale(gate_inc_raw))
                gate_feature_current = modulated_inp
            elif mode == "SenCache":
                gate_inc_raw = sencache_fields.get("online_sencache_score_pre")
                gate_inc_rescaled = gate_inc_raw
                gate_feature_current = modulated_inp
            else:
                raise RuntimeError(f"unsupported trajectory-deviation mode: {mode}")

            gate_disp_anchor = (
                rel_l1(gate_feature_current, anchor_modulated_input_for_gate)
                if (
                    anchor_modulated_input_for_gate is not None
                    and gate_feature_current.shape == anchor_modulated_input_for_gate.shape
                )
                else None
            )

            if mode == "SenCache":
                if getattr(self, "_td_sencache_threshold_scale", None) is None:
                    self._td_sencache_threshold_scale = threshold_scale_from_latent(
                        sencache_latent_for_gate,
                        getattr(self, "_td_sencache_threshold_scale_arg", "auto"),
                    )
                threshold_raw = (
                    float(getattr(self, "_td_sencache_thresh_start", 0.0))
                    if cur_step < int(round(int(getattr(self, "num_steps", 0)) * float(getattr(self, "_td_sencache_switch_ratio", 0.2))))
                    else float(getattr(self, "_td_sencache_thresh_main", 0.0))
                )
                threshold = float(threshold_raw * float(getattr(self, "_td_sencache_threshold_scale", 1.0)))
                self.accumulated_rel_l1_distance = 0.0
            else:
                threshold = float(getattr(self, "_td_threshold", 0.0))
            threshold_margin_before = threshold - float(self.accumulated_rel_l1_distance)
            acc_after_increment = (
                float(self.accumulated_rel_l1_distance) + float(gate_inc_rescaled)
                if gate_inc_rescaled is not None
                else None
            )
            gate_tortuosity_log = (
                math.log(float(acc_after_increment) + EPS) - math.log(float(gate_disp_anchor) + EPS)
                if acc_after_increment is not None and gate_disp_anchor is not None
                else None
            )
            threshold_margin_after = (
                threshold - float(acc_after_increment)
                if acc_after_increment is not None
                else None
            )
            if mode == "SenCache":
                can_cache = (
                    cur_step >= int(getattr(self, "_td_sencache_ret_steps", 0))
                    and gate_inc_rescaled is not None
                    and float(gate_inc_rescaled) < threshold
                    and int(getattr(self, "_td_sencache_accumulated_skips", 0)) < int(getattr(self, "_td_sencache_K", 10))
                )
                if can_cache:
                    native_should_calc = False
                    native_decision_reason = "sencache_threshold_cache"
                    native_acc_after = 0.0
                else:
                    native_should_calc = True
                    native_decision_reason = "sencache_threshold_full"
                    native_acc_after = 0.0
            elif acc_after_increment is not None and acc_after_increment < threshold:
                native_should_calc = False
                native_decision_reason = "threshold_cache"
                native_acc_after = acc_after_increment
            else:
                native_should_calc = True
                native_decision_reason = "threshold_full"
                native_acc_after = 0.0

            should_calc = native_should_calc
            decision_reason = native_decision_reason
            self.accumulated_rel_l1_distance = native_acc_after

        action_steps: Optional[Set[int]] = (
            getattr(self, "_td_payload_action_steps", None)
            if _mode_uses_fixed_payload_schedule(mode)
            else None
        )
        schedule_locked = action_steps is not None
        schedule_u = None if action_steps is None else int(cur_step in action_steps)
        if _mode_uses_fixed_payload_schedule(mode) and schedule_locked and not force_full:
            should_calc = not bool(schedule_u)
            decision_reason = (
                f"schedule_cache_native_{native_decision_reason}"
                if not should_calc
                else f"schedule_full_native_{native_decision_reason}"
            )
            self.accumulated_rel_l1_distance = (
                0.0 if should_calc else float(acc_after_increment or 0.0)
            )

        native_is_cached = (
            (not native_should_calc)
            and (
                mode in ("SeaCacheFinePayload", "SeaCacheSegmentPayload")
                or self.previous_residual is not None
            )
        )
        native_decision = "cache" if native_is_cached else "full"
        if forced_action == "full":
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
            decision_reason = f"forced_full_native_{native_decision_reason}"
        elif forced_action == "cache":
            if mode in ("SeaCacheFinePayload", "SeaCacheSegmentPayload"):
                if force_full:
                    raise RuntimeError(f"cannot force {mode} cache at step {cur_step}: force_full=True")
            elif force_full or self.previous_residual is None:
                raise RuntimeError(
                    f"cannot force cache at step {cur_step}: "
                    f"force_full={force_full}, previous_residual={self.previous_residual is not None}"
                )
            should_calc = False
            if not force_full and gate_inc_rescaled is not None:
                self.accumulated_rel_l1_distance = float(acc_after_increment)
            decision_reason = f"forced_cache_native_{native_decision_reason}"

        if mode == "SeaCacheFinePayload":
            if fine_state is None:
                raise RuntimeError("SeaCacheFinePayload trajectory mode missing fine payload state")
            fine_state.cnt = int(cur_step)
            fine_state.reset_step_accounting()
            fine_cache_ready_pre = bool(fine_state.cache_ready)
            fine_last_full_step_pre = fine_state.last_full_step
            fine_previous_full_step_pre = fine_state.previous_full_step
            fine_available_order_min_pre, fine_available_order_max_pre = fine_state.available_order_range()
            fine_effective_predict_order = (
                None if fine_available_order_min_pre is None
                else min(int(fine_state.predictor.max_order), int(fine_available_order_min_pre))
            )
            if (not should_calc) and not fine_cache_ready_pre:
                fine_payload_fallback_reason = "fine_cache_unready"
                should_calc = True
                decision_reason = f"fine_cache_unready_native_{native_decision_reason}"
                self.accumulated_rel_l1_distance = 0.0
            fine_state.should_skip = not bool(should_calc)
            if should_calc:
                fine_state.step_offset = 0
                fine_state.step_gap = (
                    1 if fine_state.last_full_step is None
                    else max(1, int(cur_step) - int(fine_state.last_full_step))
                )
            else:
                fine_state.step_offset = (
                    int(cur_step) - int(fine_state.last_full_step)
                    if fine_state.last_full_step is not None else 0
                )
                fine_state.step_gap = 0
            fine_state.effective_max_order = (
                0 if int(cur_step) < int(fine_state.first_enhance)
                else int(fine_state.predictor.max_order)
            )

        if mode == "SeaCacheSegmentPayload":
            if segment_state is None:
                raise RuntimeError("SeaCacheSegmentPayload trajectory mode missing segment payload state")
            segment_state.cnt = int(cur_step)
            segment_state.reset_step_accounting()
            segment_ready_pre = int(segment_state.ready_segments())
            segment_cache_ready_pre = bool(segment_ready_pre >= int(segment_state.expected_segments))
            segment_last_full_step_pre = segment_state.last_full_step
            segment_previous_full_step_pre = segment_state.previous_full_step
            (
                segment_available_order_min_pre,
                segment_available_order_max_pre,
            ) = segment_state.available_order_range()
            if (not should_calc) and not seacache_segment_payload_modes._all_segments_available(segment_state, cur_step):
                segment_payload_fallback_reason = "segment_cache_unready"
                should_calc = True
                decision_reason = f"segment_cache_unready_native_{native_decision_reason}"
                self.accumulated_rel_l1_distance = 0.0

        online_direction_fields = _online_direction_sketch_fields(
            gate_current=gate_feature_current,
            gate_previous=previous_modulated_input_for_gate,
            previous_residual=previous_residual_for_gate,
            dims=int(getattr(self, "_td_online_direction_sketch_dims", 0)),
        )
        shadow_depths = tuple(getattr(self, "_td_shadow_depths", ()) or ())
        if shadow_depths:
            will_cache = (
                (not should_calc)
                and (
                    mode in ("SeaCacheFinePayload", "SeaCacheSegmentPayload")
                    or previous_residual_for_gate is not None
                )
            )
            shadow_on = str(getattr(self, "_td_shadow_on", "all_steps"))
            emit_metrics = (shadow_on == "all_steps") or will_cache
            shadow_fields = _prefix_shadow_fields(
                self,
                ori_hidden_states=ori_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                depths=shadow_depths,
                update_anchor=not will_cache,
                emit_metrics=emit_metrics,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0
        acc_after = float(getattr(self, "accumulated_rel_l1_distance", 0.0))
        payload_fields.update({
            "schedule_locked": bool(schedule_locked),
            "schedule_u": schedule_u,
            "native_u": int(native_is_cached),
            "payload_mode": (
                str(getattr(self, "_td_payload_mode", "reuse"))
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
            "payload_base_mode": (
                _payload_base_for_mode(mode, str(getattr(self, "_td_payload_mode", "reuse")))[0]
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
            "payload_control": (
                _payload_base_for_mode(mode, str(getattr(self, "_td_payload_mode", "reuse")))[1]
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
            "payload_blend": (
                float(getattr(self, "_td_payload_blend", 1.0))
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
            "payload_control_seed_salt": (
                int(getattr(self, "_td_payload_control_seed_salt", 0))
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
            "payload_sigma": (
                float(getattr(self, "_td_history_fd_sigma", 0.5))
                if _mode_uses_fixed_payload_schedule(mode) else None
            ),
        })

        if mode == "SeaCacheSegmentPayload" and (not should_calc):
            if segment_state is None:
                raise RuntimeError("SeaCacheSegmentPayload trajectory mode missing segment payload state")
            is_cached = True
            decision = "cache"
            _, segment_hidden = seacache_segment_payload_modes._execute_segments(
                self,
                state=segment_state,
                should_calc=False,
                step=cur_step,
                hidden_states=ori_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )
            o_drv = self.proj_out(self.norm_out(segment_hidden, temb))
        elif mode == "SeaCacheFinePayload" and (not should_calc):
            if fine_state is None:
                raise RuntimeError("SeaCacheFinePayload trajectory mode missing fine payload state")
            is_cached = True
            decision = "cache"
            _, fine_hidden = _full_blocks(
                self, ori_hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                joint_attention_kwargs, controlnet_block_samples,
                controlnet_single_block_samples, controlnet_blocks_repeat,
            )
            o_drv = self.proj_out(self.norm_out(fine_hidden, temb))
        elif (not should_calc) and (self.previous_residual is not None):
            is_cached = True
            decision = "cache"
            cache_payload = self.previous_residual
            if mode == "SeaCachePayload":
                cache_payload, selected_fields = _forecast_payload(
                    payload_mode=str(getattr(self, "_td_payload_mode", "reuse")),
                    payload_blend=float(getattr(self, "_td_payload_blend", 1.0)),
                    payload_sigma=float(getattr(self, "_td_history_fd_sigma", 0.5)),
                    reuse=self.previous_residual,
                    history_state=history_fd_state_pre,
                    step=cur_step,
                    num_steps=int(getattr(self, "num_steps", 0)),
                    payload_control_seed_salt=int(getattr(self, "_td_payload_control_seed_salt", 0)),
                )
                payload_fields.update(selected_fields)
                payload_fields.update({
                    "schedule_locked": bool(schedule_locked),
                    "schedule_u": schedule_u,
                    "native_u": int(native_is_cached),
                })
            drv_hidden = ori_hidden_states + cache_payload
            o_drv = self.proj_out(self.norm_out(drv_hidden, temb))
        else:
            decision = "full"
            if mode == "SeaCacheSegmentPayload":
                if segment_state is None:
                    raise RuntimeError("SeaCacheSegmentPayload trajectory mode missing segment payload state")
                _, full_hidden = seacache_segment_payload_modes._execute_segments(
                    self,
                    state=segment_state,
                    should_calc=True,
                    step=cur_step,
                    hidden_states=ori_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )
                segment_state.previous_full_step = segment_state.last_full_step
                segment_state.last_full_step = int(cur_step)
                segment_state.full_steps_seen += 1
            else:
                _, full_hidden = _full_blocks(
                    self, ori_hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                    joint_attention_kwargs, controlnet_block_samples,
                    controlnet_single_block_samples, controlnet_blocks_repeat,
                )
            if mode == "SeaCacheFinePayload" and fine_state is not None:
                fine_state.previous_full_step = fine_state.last_full_step
                fine_state.last_full_step = int(cur_step)
                fine_state.full_steps_seen += 1
            self.previous_residual = full_hidden - ori_hidden_states
            self._td_history_fd_state = history_fd_update_on_full(
                history_fd_state_pre,
                residual=self.previous_residual,
                step=cur_step,
                sigma=float(getattr(self, "_td_history_fd_sigma", 0.5)),
                ema_beta=float(getattr(self, "_td_history_fd_ema_beta", 0.2)),
            )
            if mode == "SenCache" or getattr(self, "_td_sencache_table", None) is not None:
                self._td_sencache_anchor_latent = sencache_latent_for_gate.detach().clone()
                self._td_sencache_anchor_timestep = float(sencache_timestep_for_gate)
                self._td_sencache_anchor_step = int(cur_step)
                self._td_sencache_accumulated_skips = 0
            self._td_anchor_modulated_input = gate_feature_current.detach()
            o_drv = self.proj_out(self.norm_out(full_hidden, temb))
        if mode == "SeaCacheSegmentPayload" and segment_state is not None:
            payload_mode_value = str(getattr(self, "_td_payload_mode", "segment_taylor_o1"))
            payload_available = None
            payload_fallback = None
            payload_used = None
            if is_cached:
                payload_available = bool(
                    segment_state.predicted_segments == int(segment_state.expected_segments)
                    and not segment_state.unavailable_segments
                )
                payload_fallback = False
                payload_used = payload_mode_value
            elif schedule_locked and schedule_u == 1:
                payload_available = False
                payload_fallback = True
                payload_used = None
                if segment_payload_fallback_reason is None:
                    segment_payload_fallback_reason = "schedule_cache_forced_full"
            segment_base_mode = seacache_segment_payload_modes._base_payload_mode(payload_mode_value)
            payload_fields.update({
                "payload_used": payload_used,
                "payload_available": payload_available,
                "payload_fallback": payload_fallback,
                "payload_fallback_reason": segment_payload_fallback_reason,
                "payload_reuse_norm": None,
                "payload_forecast_norm": None,
                "payload_chosen_norm": None,
                "payload_delta_from_reuse_norm": None,
                "payload_norms_logged": False,
                "segment_payload_enabled": True,
                "segment_payload_layout": str(segment_state.segment_layout),
                "segment_payload_mode": payload_mode_value,
                "segment_payload_base_mode": segment_base_mode,
                "segment_payload_sigma": float(segment_state.payload_sigma),
                "segment_payload_expected_segments": int(segment_state.expected_segments),
                "segment_payload_ready_segments_pre": int(segment_ready_pre or 0),
                "segment_payload_missing_segments_pre": max(
                    0, int(segment_state.expected_segments) - int(segment_ready_pre or 0)
                ),
                "segment_payload_cache_ready_pre": bool(segment_cache_ready_pre),
                "segment_payload_predicted_segments": int(segment_state.predicted_segments),
                "segment_payload_updated_segments": int(segment_state.updated_segments),
                "segment_payload_unavailable_segments": ",".join(segment_state.unavailable_segments),
                "segment_payload_available_order_min_pre": (
                    None if segment_available_order_min_pre is None else int(segment_available_order_min_pre)
                ),
                "segment_payload_available_order_max_pre": (
                    None if segment_available_order_max_pre is None else int(segment_available_order_max_pre)
                ),
                "segment_payload_last_full_step_pre": (
                    None if segment_last_full_step_pre is None else int(segment_last_full_step_pre)
                ),
                "segment_payload_previous_full_step_pre": (
                    None if segment_previous_full_step_pre is None else int(segment_previous_full_step_pre)
                ),
                "segment_payload_last_full_step_post": (
                    None if segment_state.last_full_step is None else int(segment_state.last_full_step)
                ),
                "segment_payload_full_steps_seen": int(segment_state.full_steps_seen),
                "segment_payload_history_source": "full_refresh_only",
                "segment_payload_feature_space": "segment_boundary_tuple_residual",
                "segment_payload_granularity": f"segment_{segment_state.segment_layout}",
                "segment_payload_spec_hash": str(segment_state.segment_spec_hash),
            })
            segment_state.previous_modulated_input = gate_feature_current.detach()
            segment_state.cnt = int((int(cur_step) + 1) % int(segment_state.num_steps))
        if mode == "SeaCacheFinePayload" and fine_state is not None:
            payload_mode_value = str(getattr(self, "_td_payload_mode", "fine_taylor_o1"))
            payload_available = None
            payload_fallback = None
            payload_used = None
            if is_cached:
                payload_available = bool(
                    fine_cache_ready_pre
                    and int(fine_state.predicted_slots) == int(fine_state.expected_slots)
                )
                payload_fallback = False
                payload_used = payload_mode_value
            elif schedule_locked and schedule_u == 1:
                payload_available = False
                payload_fallback = True
                payload_used = None
                if fine_payload_fallback_reason is None:
                    fine_payload_fallback_reason = "schedule_cache_forced_full"
            payload_fields.update({
                "payload_used": payload_used,
                "payload_available": payload_available,
                "payload_fallback": payload_fallback,
                "payload_fallback_reason": fine_payload_fallback_reason,
                "payload_reuse_norm": (
                    _sqrt_tensor_or_none(fine_state.payload_reuse_norm_sq)
                    if is_cached and fine_state.log_payload_norms else None
                ),
                "payload_forecast_norm": (
                    _sqrt_tensor_or_none(fine_state.payload_forecast_norm_sq)
                    if is_cached and fine_state.log_payload_norms else None
                ),
                "payload_chosen_norm": (
                    _sqrt_tensor_or_none(fine_state.payload_forecast_norm_sq)
                    if is_cached and fine_state.log_payload_norms else None
                ),
                "payload_delta_from_reuse_norm": (
                    _sqrt_tensor_or_none(fine_state.payload_delta_norm_sq)
                    if is_cached and fine_state.log_payload_norms else None
                ),
                "payload_norms_logged": bool(fine_state.log_payload_norms),
                "fine_payload_enabled": True,
                "fine_payload_method": fine_state.predictor.kind,
                "fine_payload_mode": payload_mode_value,
                "fine_payload_order": int(fine_state.predictor.max_order),
                "fine_payload_sigma": float(fine_state.predictor.sigma),
                "fine_payload_expected_slots": int(fine_state.expected_slots),
                "fine_payload_slots_ready_pre": int(fine_state.slots_ready_pre),
                "fine_payload_slots_missing_pre": int(fine_state.slots_missing_pre),
                "fine_payload_cache_ready_pre": bool(fine_cache_ready_pre),
                "fine_payload_predicted_slots": int(fine_state.predicted_slots),
                "fine_payload_updated_slots": int(fine_state.updated_slots),
                "fine_payload_step_offset": int(fine_state.step_offset),
                "fine_payload_step_gap": int(fine_state.step_gap),
                "fine_payload_requested_order": int(fine_state.predictor.max_order),
                "fine_payload_available_order_min_pre": (
                    None if fine_available_order_min_pre is None else int(fine_available_order_min_pre)
                ),
                "fine_payload_available_order_max_pre": (
                    None if fine_available_order_max_pre is None else int(fine_available_order_max_pre)
                ),
                "fine_payload_effective_predict_order": (
                    None if fine_effective_predict_order is None else int(fine_effective_predict_order)
                ),
                "fine_payload_prediction_order_degraded": (
                    None if fine_effective_predict_order is None
                    else bool(int(fine_effective_predict_order) < int(fine_state.predictor.max_order))
                ),
                "fine_payload_last_full_step_pre": (
                    None if fine_last_full_step_pre is None else int(fine_last_full_step_pre)
                ),
                "fine_payload_previous_full_step_pre": (
                    None if fine_previous_full_step_pre is None else int(fine_previous_full_step_pre)
                ),
                "fine_payload_last_full_step_post": (
                    None if fine_state.last_full_step is None else int(fine_state.last_full_step)
                ),
                "fine_payload_full_steps_seen": int(fine_state.full_steps_seen),
                "fine_payload_effective_max_order": int(fine_state.effective_max_order),
                "fine_payload_history_source": "full_refresh_only",
                "fine_payload_feature_space": "pre_gate_raw_submodule_output",
                "fine_payload_granularity": "fine_114",
            })
            fine_state.cnt = int((int(cur_step) + 1) % int(fine_state.num_steps))
        if mode == "SenCache" and is_cached:
            self._td_sencache_accumulated_skips = int(getattr(self, "_td_sencache_accumulated_skips", 0)) + 1

        cache_digest_after_native_before_cf = _state_digest(self)
        scheduler_digest_before_cf = _scheduler_digest(self.scheduler)
        rng_digest_before_cf = _rng_digest(o_drv.device)

        if is_cached and with_cf:
            if mode == "SeaCacheFinePayload":
                with _fine_block_refs_disabled(self):
                    _, cf_hidden = _full_blocks(
                        self, ori_hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                        joint_attention_kwargs, controlnet_block_samples,
                        controlnet_single_block_samples, controlnet_blocks_repeat,
                    )
            else:
                _, cf_hidden = _full_blocks(
                    self, ori_hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                    joint_attention_kwargs, controlnet_block_samples,
                    controlnet_single_block_samples, controlnet_blocks_repeat,
                )
            o_cf = self.proj_out(self.norm_out(cf_hidden, temb))
        elif is_cached:
            o_cf = None
        else:
            o_cf = o_drv
    else:
        _, full_hidden = _full_blocks(
            self, ori_hidden_states, encoder_hidden_states, temb, image_rotary_emb,
            joint_attention_kwargs, controlnet_block_samples,
            controlnet_single_block_samples, controlnet_blocks_repeat,
        )
        o_drv = self.proj_out(self.norm_out(full_hidden, temb))
        o_cf = o_drv
        self._td_step = cur_step + 1
        if self._td_step == int(getattr(self, "num_steps", 0)):
            self._td_step = 0

    cache_digest_after_cf = _state_digest(self)
    scheduler_digest_after_cf = _scheduler_digest(getattr(self, "scheduler", None))
    rng_digest_after_cf = _rng_digest(o_drv.device)
    self._td_last = {
        "step": cur_step,
        "decision_u": decision,
        "is_cached": bool(is_cached),
        "decision_reason": decision_reason,
        "forced_action": forced_action,
        "native_decision_u": native_decision,
        "native_decision_reason": native_decision_reason,
        "force_full": bool(force_full),
        "gate_increment_raw": gate_inc_raw,
        "gate_increment_rescaled": gate_inc_rescaled,
        "gate_displacement_to_anchor": gate_disp_anchor,
        "gate_tortuosity_log": gate_tortuosity_log,
        "threshold_margin_before": threshold_margin_before,
        "threshold_margin_after": threshold_margin_after,
        "gate_accumulator_before": acc_before,
        "gate_accumulator_after": acc_after,
        "cache_state_digest_before": cache_digest_before,
        "cache_state_digest_after": cache_digest_after_cf,
        "cache_state_digest_after_native_before_cf": cache_digest_after_native_before_cf,
        "cache_state_digest_after_cf": cache_digest_after_cf,
        "scheduler_state_digest_before_cf": scheduler_digest_before_cf,
        "scheduler_state_digest_after_cf": scheduler_digest_after_cf,
        "rng_state_digest_before_cf": rng_digest_before_cf,
        "rng_state_digest_after_cf": rng_digest_after_cf,
        "online_p_disp_decision": gate_disp_anchor,
        "online_p_tortuosity_log": gate_tortuosity_log,
        "o_drv": o_drv,
        "o_cf": o_cf,
        **sencache_fields,
        **history_fd_fields,
        **ffro_fields,
        **payload_fields,
        **online_direction_fields,
        **shadow_fields,
    }

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (o_drv,)
    return Transformer2DModelOutput(sample=o_drv)


def install_trajectory_deviation(
    pipe,
    *,
    mode: str,
    threshold: float,
    num_steps: int,
    first_enhance: int,
    payload_mode: str = "reuse",
    payload_blend: float = 1.0,
    payload_sigma: float = 0.5,
    payload_control_seed_salt: int = 0,
    segment_layout: str = "seg8",
    teacache_backbone: str = "flux",
    teacache_variant: Optional[str] = None,
    sencache_sensitivity_path: Optional[str] = None,
    sencache_thresh_start: float = 0.005,
    sencache_thresh_main: float = 0.07,
    sencache_K: int = 10,
    sencache_threshold_scale: Union[str, float] = "auto",
    sencache_switch_ratio: float = 0.2,
    sencache_ret_steps: int = 0,
    sencache_cutoff_steps: int = -1,
    online_direction_sketch_dims: int = 0,
    shadow_depths: Tuple[int, ...] = (),
    shadow_on: str = "all_steps",
) -> Callable[[], None]:
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _td_forward
    fine_teardown: Optional[Callable[[], None]] = None

    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr._td_mode = str(mode)
    tr._td_threshold = float(threshold)
    tr._td_run_kind = "full"
    tr._td_with_cf = True
    tr._td_force_action = None
    tr._td_step = 0
    tr._td_last = None
    tr._td_online_direction_sketch_dims = int(online_direction_sketch_dims)
    tr._td_shadow_depths = tuple(int(d) for d in shadow_depths)
    tr._td_shadow_on = str(shadow_on)
    tr._td_shadow_anchor_residuals = {}
    tr._td_anchor_modulated_input = None
    tr._td_history_fd_state = init_history_fd_state()
    tr._td_history_fd_sigma = float(payload_sigma)
    tr._td_history_fd_ema_beta = 0.2
    tr._td_payload_mode = str(payload_mode)
    tr._td_payload_blend = float(payload_blend)
    tr._td_payload_control_seed_salt = int(payload_control_seed_salt)
    tr._td_payload_action_steps = None
    tr._td_sencache_table = (
        load_sensitivity_table(sencache_sensitivity_path)
        if sencache_sensitivity_path
        else None
    )
    if mode == "SenCache" and tr._td_sencache_table is None:
        raise ValueError("mode='SenCache' requires sencache_sensitivity_path")
    if mode == "SeaCachePayload":
        if payload_mode not in PAYLOAD_MODES:
            raise ValueError(f"unknown payload_mode: {payload_mode!r}")
        if not (0.0 <= float(payload_blend) <= 1.0):
            raise ValueError(f"payload_blend must be in [0, 1], got {payload_blend}")
    if mode == "SeaCacheFinePayload":
        if payload_mode not in FINE_PAYLOAD_MODES:
            raise ValueError(f"unknown fine payload_mode: {payload_mode!r}")
        if float(payload_blend) != 1.0:
            raise ValueError("SeaCacheFinePayload trajectory mode requires payload_blend=1.0")
        fine_teardown = seacache_fine_payload_modes.install_block_hooks(
            pipe,
            threshold=float(threshold),
            num_steps=int(num_steps),
            first_enhance=int(first_enhance),
            payload_mode=str(payload_mode),
            payload_sigma=float(payload_sigma),
            log_payload_norms=False,
            shadow_full_velocity=False,
        )
    if mode == "SeaCacheSegmentPayload":
        if payload_mode not in SEGMENT_PAYLOAD_MODES:
            raise ValueError(f"unknown segment payload_mode: {payload_mode!r}")
        if str(segment_layout) not in SEGMENT_LAYOUTS:
            raise ValueError(f"unknown segment_layout: {segment_layout!r}")
        if float(payload_blend) != 1.0:
            raise ValueError("SeaCacheSegmentPayload trajectory mode requires payload_blend=1.0")
        specs = seacache_segment_payload_modes.build_segments(
            str(segment_layout),
            n_double=len(tr.transformer_blocks),
            n_single=len(tr.single_transformer_blocks),
        )
        segment_state = seacache_segment_payload_modes.SegmentSeaPayloadState(
            threshold=float(threshold),
            num_steps=int(num_steps),
            first_enhance=int(first_enhance),
            payload_mode=str(payload_mode),
            payload_sigma=float(payload_sigma),
            segment_layout=str(segment_layout),
            segment_specs=specs,
            segment_spec_hash=seacache_segment_payload_modes._spec_hash(specs),
        )
        segment_state.scheduler = pipe.scheduler
        tr._seacache_segment_payload_state = segment_state
        tr.seacache_segment_payload_decisions = segment_state.decisions
    tr._td_sencache_thresh_start = float(sencache_thresh_start)
    tr._td_sencache_thresh_main = float(sencache_thresh_main)
    tr._td_sencache_K = int(sencache_K)
    tr._td_sencache_threshold_scale_arg = sencache_threshold_scale
    tr._td_sencache_threshold_scale = None
    tr._td_sencache_switch_ratio = float(sencache_switch_ratio)
    tr._td_sencache_ret_steps = int(sencache_ret_steps)
    tr._td_sencache_cutoff_steps = int(sencache_cutoff_steps)
    tr._td_sencache_anchor_latent = None
    tr._td_sencache_anchor_timestep = None
    tr._td_sencache_anchor_step = None
    tr._td_sencache_accumulated_skips = 0
    tr.num_steps = int(num_steps)
    tr.first_enhance = int(first_enhance)
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr.previous_residual = None
    if mode == "TeaCache":
        tr.teacache_rescale = np.poly1d(list(get_coeffs(teacache_backbone, variant=teacache_variant)))

    done = {"v": False}

    def teardown() -> None:
        if done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        if fine_teardown is not None:
            fine_teardown()
        for attr in (
            "scheduler", "_td_mode", "_td_threshold", "_td_run_kind", "_td_with_cf",
            "_td_force_action", "_td_step", "_td_last", "num_steps", "first_enhance", "cnt",
            "_td_online_direction_sketch_dims",
            "_td_shadow_depths", "_td_shadow_on", "_td_shadow_anchor_residuals",
            "_td_anchor_modulated_input",
            "_td_history_fd_state", "_td_history_fd_sigma", "_td_history_fd_ema_beta",
            "_td_payload_mode", "_td_payload_blend", "_td_payload_control_seed_salt",
            "_td_payload_action_steps",
            "_td_sencache_table", "_td_sencache_thresh_start", "_td_sencache_thresh_main",
            "_td_sencache_K", "_td_sencache_threshold_scale_arg", "_td_sencache_threshold_scale",
            "_td_sencache_switch_ratio", "_td_sencache_ret_steps", "_td_sencache_cutoff_steps",
            "_td_sencache_anchor_latent", "_td_sencache_anchor_timestep", "_td_sencache_anchor_step",
            "_td_sencache_accumulated_skips",
            "accumulated_rel_l1_distance", "previous_modulated_input",
            "previous_residual", "teacache_rescale",
            "_seacache_segment_payload_state", "seacache_segment_payload_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        done["v"] = True

    return teardown


def reset_td_state(pipe, action_steps: Optional[Set[int]] = None) -> None:
    tr = pipe.transformer
    tr._td_step = 0
    tr._td_force_action = None
    tr._td_last = None
    tr.cnt = 0
    tr.accumulated_rel_l1_distance = 0.0
    tr.previous_modulated_input = None
    tr._td_anchor_modulated_input = None
    tr.previous_residual = None
    tr._td_shadow_anchor_residuals = {}
    tr._td_history_fd_state = init_history_fd_state()
    tr._td_payload_action_steps = None if action_steps is None else set(int(x) for x in action_steps)
    tr._td_sencache_anchor_latent = None
    tr._td_sencache_anchor_timestep = None
    tr._td_sencache_anchor_step = None
    tr._td_sencache_accumulated_skips = 0
    fine_state = getattr(tr, "_seacache_fine_payload_state", None)
    if fine_state is not None:
        fine_state.reset_trajectory(action_steps=action_steps)
        tr.seacache_payload_decisions = fine_state.decisions
    segment_state = getattr(tr, "_seacache_segment_payload_state", None)
    if segment_state is not None:
        segment_state.reset_trajectory(action_steps=action_steps)
        tr.seacache_segment_payload_decisions = segment_state.decisions


def _ensure_step_index(scheduler, timestep) -> Optional[int]:
    if getattr(scheduler, "_step_index", None) is None and hasattr(scheduler, "_init_step_index"):
        scheduler._init_step_index(timestep)
    idx = getattr(scheduler, "_step_index", None)
    return int(idx) if idx is not None else None


def _run_full_trace(pipe, ctx: Dict[str, Any], num_steps: int) -> Dict[str, Any]:
    tr = pipe.transformer
    tr._td_run_kind = "full"
    tr._td_with_cf = False
    reset_td_state(pipe)
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    z_pre: List[torch.Tensor] = []
    z_post: List[torch.Tensor] = []
    outputs: List[torch.Tensor] = []
    with torch.no_grad(), _fine_block_refs_disabled(pipe.transformer):
        for i, timestep in enumerate(ctx["timesteps"]):
            z_pre.append(latents.detach().to("cpu", dtype=torch.float32))
            t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=t_expanded / 1000,
                guidance=ctx["guidance"],
                pooled_projections=ctx["pooled_prompt_embeds"],
                encoder_hidden_states=ctx["prompt_embeds"],
                txt_ids=ctx["text_ids"],
                img_ids=ctx["latent_image_ids"],
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if int(pipe.transformer._td_last["step"]) != i:
                raise RuntimeError(f"full trace step mismatch: record={pipe.transformer._td_last['step']} loop={i}")
            outputs.append(noise_pred.detach().to("cpu", dtype=torch.float32))
            _ensure_step_index(pipe.scheduler, timestep)
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            z_post.append(latents.detach().to("cpu", dtype=torch.float32))
    if len(outputs) != num_steps:
        raise RuntimeError(f"full trace collected {len(outputs)} outputs, expected {num_steps}")
    return {"z_pre": z_pre, "z_post": z_post, "outputs": outputs, "final": latents.detach()}


def _run_cached_trace(
    pipe,
    ctx: Dict[str, Any],
    full_trace: Dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt_id: int,
    prompt: str,
    seed: int,
    with_cf: bool,
) -> Dict[str, Any]:
    tr = pipe.transformer
    tr._td_run_kind = "cached"
    tr._td_with_cf = bool(with_cf)
    action_steps = _load_payload_action_steps(
        args.payload_schedule_dir,
        prompt_id,
        expected_num_steps=int(args.num_steps),
        require_reuse_reference=_mode_uses_fixed_payload_schedule(args.mode),
    )
    reset_td_state(pipe, action_steps=action_steps)
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    rows: List[Dict[str, Any]] = []
    tensor_dump: List[Dict[str, torch.Tensor]] = []
    sigmas = [float(x) for x in ctx["sigmas"]]

    with torch.no_grad():
        for i, timestep in enumerate(ctx["timesteps"]):
            z_cached_pre = latents
            z_full_pre = full_trace["z_pre"][i].to(device=latents.device, dtype=torch.float32)
            z_full_post = full_trace["z_post"][i].to(device=latents.device, dtype=torch.float32)
            o_full = full_trace["outputs"][i].to(device=latents.device, dtype=torch.float32)

            t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
            o_drv = pipe.transformer(
                hidden_states=latents,
                timestep=t_expanded / 1000,
                guidance=ctx["guidance"],
                pooled_projections=ctx["pooled_prompt_embeds"],
                encoder_hidden_states=ctx["prompt_embeds"],
                txt_ids=ctx["text_ids"],
                img_ids=ctx["latent_image_ids"],
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            rec = dict(pipe.transformer._td_last)
            if int(rec["step"]) != i:
                raise RuntimeError(f"cached trace step mismatch: record={rec['step']} loop={i}")
            o_drv_f = o_drv.detach().to(torch.float32)
            o_cf = rec.get("o_cf")
            o_cf_f = o_cf.detach().to(torch.float32) if o_cf is not None else None
            z_cached_pre_f = z_cached_pre.detach().to(torch.float32)
            scheduler_step_index = _ensure_step_index(pipe.scheduler, timestep)
            H_i = sigmas[i + 1] - sigmas[i]
            saved_step_index = getattr(pipe.scheduler, "_step_index", None)
            z_cf_next = None
            if o_cf is not None:
                rng_state = _capture_rng_state(o_cf.device)
                z_cf_native = pipe.scheduler.step(o_cf, timestep, z_cached_pre, return_dict=False)[0]
                z_cf_next = z_cf_native.detach().to(torch.float32)
                pipe.scheduler._step_index = saved_step_index
                _restore_rng_state(o_cf.device, rng_state)
            latents_next = pipe.scheduler.step(o_drv, timestep, latents, return_dict=False)[0]
            z_cached_post_f = latents_next.detach().to(torch.float32)
            d_full_scheduler = z_full_post - z_full_pre
            z_full_model_post = _flowmatch_euler_post(z_full_pre, H_i, o_full, o_drv.dtype)
            d_full_model = z_full_model_post - z_full_pre
            d_cached_scheduler = z_cached_post_f - z_cached_pre_f
            z_cached_model_post = _flowmatch_euler_post(z_cached_pre_f, H_i, o_drv_f, o_drv.dtype)
            d_cached_model = z_cached_model_post - z_cached_pre_f
            full_scheduler_residual = d_full_scheduler - d_full_model
            cached_scheduler_residual = d_cached_scheduler - d_cached_model
            full_scheduler_den = _norm(d_full_scheduler) + _norm(d_full_model) + EPS
            cached_scheduler_den = _norm(d_cached_scheduler) + _norm(d_cached_model) + EPS
            drv_minus_full = o_drv_f - o_full
            row: Dict[str, Any] = {
                "run_name": args.run_name,
                "prompt_id": int(prompt_id),
                "prompt": prompt,
                "seed": int(seed),
                "step_index": i,
                "timestep": float(timestep.detach().cpu().item()),
                "sigma_n": sigmas[i],
                "sigma_np1": sigmas[i + 1],
                "step_size_H": float(H_i),
                "scheduler_step_index": scheduler_step_index,
                "mode": args.mode,
                "cache_threshold": float(_threshold_for_args(args)),
                "decision_u": rec["decision_u"],
                "is_cached": bool(rec["is_cached"]),
                "decision_reason": rec["decision_reason"],
                "gate_increment_raw": rec["gate_increment_raw"],
                "gate_increment_rescaled": rec["gate_increment_rescaled"],
                "threshold_margin_before": rec["threshold_margin_before"],
                "threshold_margin_after": rec["threshold_margin_after"],
                "gate_accumulator_before": rec["gate_accumulator_before"],
                "gate_accumulator_after": rec["gate_accumulator_after"],
                "cache_state_digest_before": rec["cache_state_digest_before"],
                "cache_state_digest_after": rec["cache_state_digest_after"],
                "cache_state_digest_after_native_before_cf": rec["cache_state_digest_after_native_before_cf"],
                "cache_state_digest_after_cf": rec["cache_state_digest_after_cf"],
                "scheduler_state_digest_before_cf": rec["scheduler_state_digest_before_cf"],
                "scheduler_state_digest_after_cf": rec["scheduler_state_digest_after_cf"],
                "rng_state_digest_before_cf": rec["rng_state_digest_before_cf"],
                "rng_state_digest_after_cf": rec["rng_state_digest_after_cf"],
                "z_full_norm": _norm(z_full_pre),
                "z_cached_norm": _norm(z_cached_pre_f),
                "latent_drift_pre": _norm(z_cached_pre_f - z_full_pre),
                "latent_drift_pre_rel": _norm(z_cached_pre_f - z_full_pre) / (_norm(z_full_pre) + EPS),
                "latent_drift_post": _norm(z_cached_post_f - z_full_post),
                "latent_drift_post_rel": _norm(z_cached_post_f - z_full_post) / (_norm(z_full_post) + EPS),
                "o_full_norm": _norm(o_full),
                "o_drv_norm": _norm(o_drv_f),
                "o_cf_norm": _norm(o_cf_f) if o_cf_f is not None else None,
                "output_drift": _norm(drv_minus_full),
                "output_drift_rel": _norm(drv_minus_full) / (_norm(o_full) + EPS),
                "cos_drv_full": _cos(o_drv_f, o_full),
                "scheduler_delta_full_norm": _norm(float(H_i) * o_full),
                "scheduler_delta_cached_norm": _norm(z_cached_post_f - z_cached_pre_f),
                "full_scheduler_closure_abs_fp32": _norm(full_scheduler_residual),
                "full_scheduler_closure_rel_fp32": _norm(full_scheduler_residual) / full_scheduler_den,
                "cached_scheduler_closure_abs_fp32": _norm(cached_scheduler_residual),
                "cached_scheduler_closure_rel_fp32": _norm(cached_scheduler_residual) / cached_scheduler_den,
                "payload_mode": rec.get("payload_mode"),
                "payload_base_mode": rec.get("payload_base_mode"),
                "payload_control": rec.get("payload_control"),
                "payload_blend": rec.get("payload_blend"),
                "payload_sigma": rec.get("payload_sigma"),
                "payload_used": rec.get("payload_used"),
                "payload_available": rec.get("payload_available"),
                "payload_fallback": rec.get("payload_fallback"),
                "payload_reuse_norm": rec.get("payload_reuse_norm"),
                "payload_forecast_norm": rec.get("payload_forecast_norm"),
                "payload_chosen_norm": rec.get("payload_chosen_norm"),
                "payload_delta_from_reuse_norm": rec.get("payload_delta_from_reuse_norm"),
                "schedule_locked": rec.get("schedule_locked"),
                "schedule_u": rec.get("schedule_u"),
                "native_u": rec.get("native_u"),
            }
            for key, value in rec.items():
                if (
                    key.startswith(("online_", "shadow_", "fine_payload_", "segment_payload_"))
                    or key in ("payload_fallback_reason", "payload_norms_logged")
                ):
                    row[key] = value

            if o_cf_f is None:
                for key in (
                    "action_defect", "action_defect_rel", "action_defect_rel_to_full",
                    "state_gap", "state_gap_rel", "dot_action_gap", "cos_action_gap",
                    "projection_action_on_total", "action_latent_step_defect",
                    "action_latent_step_defect_rel", "latent_state_scheduler_gap",
                    "decomposition_closure_abs", "decomposition_closure_rel",
                    "latent_decomposition_closure_abs", "latent_decomposition_closure_rel",
                    "cos_drv_cf", "cos_cf_full",
                    "cf_scheduler_closure_abs_fp32", "cf_scheduler_closure_rel_fp32",
                ):
                    row[key] = None
            else:
                act = o_drv_f - o_cf_f
                gap = o_cf_f - o_full
                out_closure = drv_minus_full - (act + gap)
                action_latent = z_cached_post_f - z_cf_next
                latent_gap = z_cf_next - z_full_post
                latent_closure = (z_cached_post_f - z_full_post) - (action_latent + latent_gap)
                d_cf_scheduler = z_cf_next - z_cached_pre_f
                d_cf_model = _flowmatch_euler_post(z_cached_pre_f, H_i, o_cf_f, o_cf.dtype) - z_cached_pre_f
                cf_scheduler_residual = d_cf_scheduler - d_cf_model
                cf_scheduler_den = _norm(d_cf_scheduler) + _norm(d_cf_model) + EPS
                row.update({
                    "action_defect": _norm(act),
                    "action_defect_rel": _norm(act) / (_norm(o_cf_f) + EPS),
                    "action_defect_rel_to_full": _norm(act) / (_norm(o_full) + EPS),
                    "state_gap": _norm(gap),
                    "state_gap_rel": _norm(gap) / (_norm(o_full) + EPS),
                    "dot_action_gap": _dot(act, gap),
                    "cos_action_gap": _cos(act, gap),
                    "projection_action_on_total": _dot(act, drv_minus_full) / (_norm(drv_minus_full) ** 2 + EPS),
                    "action_latent_step_defect": _norm(action_latent),
                    "action_latent_step_defect_rel": _norm(action_latent) / (_norm(z_cf_next - z_cached_pre_f) + EPS),
                    "latent_state_scheduler_gap": _norm(latent_gap),
                    "decomposition_closure_abs": _norm(out_closure),
                    "decomposition_closure_rel": _norm(out_closure) / (_norm(drv_minus_full) + EPS),
                    "latent_decomposition_closure_abs": _norm(latent_closure),
                    "latent_decomposition_closure_rel": _norm(latent_closure) / (_norm(z_cached_post_f - z_full_post) + EPS),
                    "cos_drv_cf": _cos(o_drv_f, o_cf_f),
                    "cos_cf_full": _cos(o_cf_f, o_full),
                    "cf_scheduler_closure_abs_fp32": _norm(cf_scheduler_residual),
                    "cf_scheduler_closure_rel_fp32": _norm(cf_scheduler_residual) / cf_scheduler_den,
                })

            rows.append(row)
            if args.save_trace_tensors:
                tensor_dump.append({
                    "z_full_pre": z_full_pre.detach().to("cpu", dtype=torch.bfloat16),
                    "z_full_post": z_full_post.detach().to("cpu", dtype=torch.bfloat16),
                    "z_cached_pre": z_cached_pre_f.detach().to("cpu", dtype=torch.bfloat16),
                    "z_cached_post": z_cached_post_f.detach().to("cpu", dtype=torch.bfloat16),
                    "o_full": o_full.detach().to("cpu", dtype=torch.bfloat16),
                    "o_drv": o_drv_f.detach().to("cpu", dtype=torch.bfloat16),
                    "o_cf": o_cf_f.detach().to("cpu", dtype=torch.bfloat16) if o_cf_f is not None else None,
                    "z_cf_post": z_cf_next.detach().to("cpu", dtype=torch.bfloat16) if z_cf_next is not None else None,
                })

            latents = latents_next

    return {"rows": rows, "final": latents.detach(), "tensor_dump": tensor_dump}


def _threshold_for_args(args: argparse.Namespace) -> float:
    if args.mode in ("SeaCache", "SeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload"):
        return float(args.seacache_thresh)
    if args.mode == "TeaCache":
        return float(args.teacache_thresh)
    if args.mode == "SenCache":
        return float(args.sencache_thresh_main)
    raise ValueError(f"unsupported mode: {args.mode!r}")


def _cache_params_for_args(args: argparse.Namespace) -> Dict[str, Any]:
    coeffs = None
    coeffs_hash = None
    if args.mode == "TeaCache":
        coeffs = [float(c) for c in get_coeffs(args.teacache_backbone, variant=args.teacache_variant)]
        coeffs_hash = _hash_text(json.dumps(coeffs, sort_keys=True))
    table_meta = None
    if getattr(args, "sencache_sensitivity_path", None):
        table = load_sensitivity_table(args.sencache_sensitivity_path)
        table_meta = {
            "path": str(args.sencache_sensitivity_path),
            "sha256": table.sha256,
            "metadata": table.metadata,
        }
    return {
        "seacache_thresh": (
            float(args.seacache_thresh)
            if args.mode in ("SeaCache", "SeaCachePayload", "SeaCacheFinePayload", "SeaCacheSegmentPayload") else None
        ),
        "payload_mode": args.payload_mode if _mode_uses_fixed_payload_schedule(args.mode) else None,
        "payload_base_mode": (
            _payload_base_for_mode(args.mode, str(args.payload_mode))[0]
            if _mode_uses_fixed_payload_schedule(args.mode) else None
        ),
        "payload_control": (
            _payload_base_for_mode(args.mode, str(args.payload_mode))[1]
            if _mode_uses_fixed_payload_schedule(args.mode) else None
        ),
        "payload_blend": float(args.payload_blend) if _mode_uses_fixed_payload_schedule(args.mode) else None,
        "payload_sigma": float(args.payload_sigma) if _mode_uses_fixed_payload_schedule(args.mode) else None,
        "segment_layout": args.segment_layout if args.mode == "SeaCacheSegmentPayload" else None,
        "payload_schedule_dir": (
            str(args.payload_schedule_dir)
            if _mode_uses_fixed_payload_schedule(args.mode) and args.payload_schedule_dir is not None else None
        ),
        "teacache_thresh": float(args.teacache_thresh) if args.mode == "TeaCache" else None,
        "teacache_backbone": args.teacache_backbone if args.mode == "TeaCache" else None,
        "teacache_variant": args.teacache_variant if args.mode == "TeaCache" else None,
        "teacache_coefficients": coeffs,
        "teacache_coefficients_hash": coeffs_hash,
        "sencache_thresh_start": float(args.sencache_thresh_start) if args.mode == "SenCache" else None,
        "sencache_thresh_main": float(args.sencache_thresh_main) if args.mode == "SenCache" else None,
        "sencache_K": int(args.sencache_K) if args.mode == "SenCache" else None,
        "sencache_threshold_scale": str(args.sencache_threshold_scale) if args.mode == "SenCache" else None,
        "sencache_switch_ratio": float(args.sencache_switch_ratio) if args.mode == "SenCache" else None,
        "sencache_ret_steps": int(args.sencache_ret_steps) if args.mode == "SenCache" else None,
        "sencache_cutoff_steps": int(args.sencache_cutoff_steps) if args.mode == "SenCache" else None,
        "sencache_sensitivity": table_meta,
        "first_enhance": int(args.first_enhance),
    }


def _run_name_for_args(args: argparse.Namespace) -> str:
    threshold = _threshold_for_args(args)
    if args.mode == "TeaCache":
        tag = f"trajdev_teacache_t{threshold}_{args.teacache_backbone}"
        if args.teacache_variant:
            tag += f"-{args.teacache_variant}"
    elif args.mode == "SenCache":
        tag = f"trajdev_sencache_tm{threshold}"
    elif args.mode == "SeaCachePayload":
        blend_tag = str(args.payload_blend).replace(".", "p")
        tag = f"trajdev_seacache_payload_t{threshold}_{args.payload_mode}_b{blend_tag}"
    elif args.mode == "SeaCacheFinePayload":
        tag = f"trajdev_seacache_fine_payload_t{threshold}_{args.payload_mode}"
    elif args.mode == "SeaCacheSegmentPayload":
        tag = f"trajdev_seacache_segment_payload_t{threshold}_{args.segment_layout}_{args.payload_mode}"
    else:
        tag = f"trajdev_seacache_t{threshold}"
    selected_count = len(getattr(args, "selected_prompt_ids", []) or [])
    if selected_count:
        limit_base = f"n{args.limit}" if int(args.limit) > 0 else "nfull"
        limit_tag = f"{limit_base}_sel{selected_count}"
    else:
        limit_tag = f"n{args.limit}" if int(args.limit) > 0 else "nfull"
    return f"{tag}_{limit_tag}_s{args.seed}_{args.num_steps}"


def _parse_prompt_id_tokens(text: str) -> Iterable[int]:
    for token in text.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid prompt id {token!r}") from exc
        if value < 0:
            raise ValueError(f"prompt id must be nonnegative: {value}")
        yield value


def _selected_prompt_ids(args: argparse.Namespace) -> Optional[List[int]]:
    ids: List[int] = []
    if args.prompt_ids:
        ids.extend(_parse_prompt_id_tokens(args.prompt_ids))
    if args.prompt_id_file:
        text = args.prompt_id_file.read_text(encoding="utf-8")
        ids.extend(_parse_prompt_id_tokens(text))
    if not ids:
        return None
    seen = set()
    out: List[int] = []
    for prompt_id in ids:
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        out.append(prompt_id)
    return sorted(out)


def _same_optional_float(a: Any, b: Any, *, atol: float) -> Tuple[bool, float]:
    if a is None or a == "":
        ax = None
    else:
        ax = float(a)
    if b is None or b == "":
        bx = None
    else:
        bx = float(b)
    if ax is None or bx is None:
        return ax is None and bx is None, 0.0
    diff = abs(ax - bx)
    return diff <= atol, diff


def _compare_native_rows_without_cf(
    no_cf_rows: List[Dict[str, Any]],
    with_cf_rows: List[Dict[str, Any]],
    *,
    float_atol: float = 1e-5,
) -> Dict[str, Any]:
    n = min(len(no_cf_rows), len(with_cf_rows))
    mismatches: List[Dict[str, Any]] = []
    max_float_abs_diff = 0.0
    for i in range(n):
        lhs = no_cf_rows[i]
        rhs = with_cf_rows[i]
        for field in NATIVE_ROW_COMPARE_EXACT_FIELDS:
            if lhs.get(field) != rhs.get(field):
                mismatches.append({
                    "row": i,
                    "field": field,
                    "no_cf": lhs.get(field),
                    "with_cf": rhs.get(field),
                })
        for field in NATIVE_ROW_COMPARE_FLOAT_FIELDS:
            ok, diff = _same_optional_float(lhs.get(field), rhs.get(field), atol=float_atol)
            max_float_abs_diff = max(max_float_abs_diff, diff)
            if not ok:
                mismatches.append({
                    "row": i,
                    "field": field,
                    "no_cf": lhs.get(field),
                    "with_cf": rhs.get(field),
                    "abs_diff": diff,
                })
    if len(no_cf_rows) != len(with_cf_rows):
        mismatches.append({
            "field": "row_count",
            "no_cf": len(no_cf_rows),
            "with_cf": len(with_cf_rows),
        })
    return {
        "with_without_cf_native_rows_equal": len(mismatches) == 0,
        "with_without_cf_native_row_count_equal": len(no_cf_rows) == len(with_cf_rows),
        "with_without_cf_native_compared_rows": int(n),
        "with_without_cf_native_compare_float_atol": float(float_atol),
        "with_without_cf_native_max_float_abs_diff": float(max_float_abs_diff),
        "with_without_cf_native_mismatch_count": int(len(mismatches)),
        "with_without_cf_native_mismatch_examples": mismatches[:10],
    }


def _trapezoid_unit(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(sum(0.5 * (float(xs[i]) + float(xs[i + 1])) for i in range(len(xs) - 1)))


def _summarize_prompt(rows: List[Dict[str, Any]], *, prompt_id: int, seed: int,
                      args: argparse.Namespace, final_drift: float,
                      final_drift_rel: float, checks: Dict[str, Any],
                      prompt_dir: Path) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    latent_states = [float(rows[0]["latent_drift_pre"])] + [float(r["latent_drift_post"]) for r in rows]
    latent_rel_states = [float(rows[0]["latent_drift_pre_rel"])] + [float(r["latent_drift_post_rel"]) for r in rows]
    H_abs = [abs(float(r["step_size_H"])) for r in rows]
    rel_time_auc = 0.0
    denom = sum(H_abs)
    if denom > 0:
        rel_time_auc = sum(0.5 * (latent_rel_states[i] + latent_rel_states[i + 1]) * H_abs[i]
                           for i in range(len(rows))) / denom
    action_cached = [float(r["action_defect_rel"]) for r in rows
                     if r.get("is_cached") and r.get("action_defect_rel") is not None]
    gate_seq = "".join("C" if r.get("is_cached") else "F" for r in rows)
    return {
        "prompt_id": int(prompt_id),
        "seed": int(seed),
        "mode": args.mode,
        "cache_params": _cache_params_for_args(args),
        "num_steps": int(args.num_steps),
        "final_latent_drift": final_drift,
        "final_latent_drift_rel": final_drift_rel,
        "trajectory_auc_z": _trapezoid_unit(latent_states),
        "trajectory_auc_z_rel_time": float(rel_time_auc),
        "total_action_exposure": float(sum(vals("action_defect"))),
        "mean_action_defect_cached": float(sum(action_cached) / len(action_cached)) if action_cached else None,
        "total_state_gap_exposure": float(sum(vals("state_gap"))),
        "total_action_latent_exposure": float(sum(vals("action_latent_step_defect"))),
        "max_D_z_pre": float(max(float(r["latent_drift_pre"]) for r in rows)),
        "argmax_D_z_pre": int(max(rows, key=lambda r: float(r["latent_drift_pre"]))["step_index"]),
        "max_action_defect": float(max(vals("action_defect"))) if vals("action_defect") else None,
        "argmax_action_defect": int(max(
            [r for r in rows if r.get("action_defect") is not None],
            key=lambda r: float(r["action_defect"]),
            default={"step_index": -1},
        )["step_index"]),
        "max_state_gap": float(max(vals("state_gap"))) if vals("state_gap") else None,
        "argmax_state_gap": int(max(
            [r for r in rows if r.get("state_gap") is not None],
            key=lambda r: float(r["state_gap"]),
            default={"step_index": -1},
        )["step_index"]),
        "cache_rate": float(sum(1 for r in rows if r["is_cached"]) / max(len(rows), 1)),
        "num_cached_steps": int(sum(1 for r in rows if r["is_cached"])),
        "num_full_steps": int(sum(1 for r in rows if not r["is_cached"])),
        "gate_sequence": gate_seq,
        "gate_sequence_hash": _hash_text(gate_seq),
        "final_image_path": str(prompt_dir / "cached.png") if args.save_images else None,
        "full_image_path": str(prompt_dir / "baseline.png") if args.save_images else None,
        "cached_image_path": str(prompt_dir / "cached.png") if args.save_images else None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "clip_similarity": None,
        "image_reward_delta": None,
        "acceptance_checks": checks,
    }


def _write_jsonl(path: Path, rows: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(STEP_FIELDS)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_one_prompt(pipe, prompt: str, prompt_id: int, seed: int,
                    args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with torch.no_grad():
        ctx = _encode_and_prepare(pipe, prompt, seed, args)
        full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
        no_cf = None
        if args.compare_without_cf:
            no_cf = _run_cached_trace(
                pipe, ctx, full_trace, args=args, prompt_id=prompt_id,
                prompt=prompt, seed=seed, with_cf=False,
            )
        no_shadow = None
        if args.compare_without_shadow and args.shadow_depths_tuple:
            tr = pipe.transformer
            saved_depths = tuple(getattr(tr, "_td_shadow_depths", ()) or ())
            saved_shadow_on = str(getattr(tr, "_td_shadow_on", "all_steps"))
            tr._td_shadow_depths = ()
            try:
                no_shadow = _run_cached_trace(
                    pipe, ctx, full_trace, args=args, prompt_id=prompt_id,
                    prompt=prompt, seed=seed, with_cf=bool(args.with_cf),
                )
            finally:
                tr._td_shadow_depths = saved_depths
                tr._td_shadow_on = saved_shadow_on
        cached = _run_cached_trace(
            pipe, ctx, full_trace, args=args, prompt_id=prompt_id,
            prompt=prompt, seed=seed, with_cf=bool(args.with_cf),
        )

    final_full = full_trace["final"].to(cached["final"].device, dtype=torch.float32)
    final_cached = cached["final"].to(torch.float32)
    final_drift = _norm(final_cached - final_full)
    final_drift_rel = final_drift / (_norm(final_full) + EPS)
    checks: Dict[str, Any] = {}
    if no_cf is not None:
        seq_a = "".join("C" if r["is_cached"] else "F" for r in no_cf["rows"])
        seq_b = "".join("C" if r["is_cached"] else "F" for r in cached["rows"])
        checks["with_without_cf_gate_sequence_equal"] = (seq_a == seq_b)
        checks["with_without_cf_final_l2"] = _norm(
            no_cf["final"].to(final_cached.device, dtype=torch.float32) - final_cached
        )
        checks.update(_compare_native_rows_without_cf(no_cf["rows"], cached["rows"]))
    if no_shadow is not None:
        seq_a = "".join("C" if r["is_cached"] else "F" for r in no_shadow["rows"])
        seq_b = "".join("C" if r["is_cached"] else "F" for r in cached["rows"])
        checks["with_without_shadow_gate_sequence_equal"] = (seq_a == seq_b)
        checks["with_without_shadow_final_l2"] = _norm(
            no_shadow["final"].to(final_cached.device, dtype=torch.float32) - final_cached
        )
        shadow_compare = _compare_native_rows_without_cf(no_shadow["rows"], cached["rows"])
        checks.update({f"with_without_shadow_{k}": v for k, v in shadow_compare.items()})
    prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(full_trace["final"].detach().to("cpu", dtype=torch.bfloat16), prompt_dir / "baseline.pt")
    torch.save(cached["final"].detach().to("cpu", dtype=torch.bfloat16), prompt_dir / "cached.pt")
    if args.save_images:
        H = (args.height // 16) * 16
        W = (args.width // 16) * 16
        _decode_to_pil(pipe, full_trace["final"], H, W).save(prompt_dir / "baseline.png")
        _decode_to_pil(pipe, cached["final"], H, W).save(prompt_dir / "cached.png")
    if args.save_trace_tensors:
        trace_payload = {
            "schema": "trajectory_trace_tensors.v1",
            "trace_dtype": "bfloat16",
            "height": int(ctx["H"]),
            "width": int(ctx["W"]),
            "latent_shape": list(ctx["latents_init"].shape),
            "latent_image_ids": ctx["latent_image_ids"].detach().to("cpu"),
            "rows": cached["tensor_dump"],
        }
        torch.save(trace_payload, prompt_dir / "trace_tensors.pt")

    _write_jsonl(prompt_dir / "trajectory_rows.jsonl", cached["rows"])
    summary = _summarize_prompt(
        cached["rows"], prompt_id=prompt_id, seed=seed, args=args,
        final_drift=final_drift, final_drift_rel=final_drift_rel,
        checks=checks, prompt_dir=prompt_dir,
    )
    (prompt_dir / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prompt_manifest = {
        "prompt_idx": int(prompt_id),
        "prompt": prompt,
        "seed": int(seed),
        "num_steps": int(args.num_steps),
        "mode": args.mode,
        "cache_params": summary["cache_params"],
        "with_cf": bool(args.with_cf),
        "compare_without_cf": bool(args.compare_without_cf),
        "compare_without_shadow": bool(args.compare_without_shadow),
        "shadow_depths": [int(x) for x in args.shadow_depths_tuple],
        "shadow_on": str(args.shadow_on),
        "files": {
            "trajectory_rows": "trajectory_rows.jsonl",
            "trajectory_summary": "trajectory_summary.json",
            "baseline_latent": "baseline.pt",
            "cached_latent": "cached.pt",
            "baseline_image": "baseline.png" if args.save_images else None,
            "cached_image": "cached.png" if args.save_images else None,
            "trace_tensors": "trace_tensors.pt" if args.save_trace_tensors else None,
        },
        "complete": True,
    }
    (prompt_dir / "manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cached["rows"], summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX closed-loop trajectory deviation audit.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--mode",
        choices=[
            "SeaCache", "SeaCachePayload", "SeaCacheFinePayload",
            "SeaCacheSegmentPayload", "TeaCache", "SenCache",
        ],
        default="SeaCache",
    )
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--seacache_thresh", type=float, default=0.3)
    p.add_argument(
        "--payload_mode",
        choices=sorted(PAYLOAD_MODES | FINE_PAYLOAD_MODES | SEGMENT_PAYLOAD_MODES),
        default="reuse",
        help=(
            "SeaCachePayload: residual payload used on cached steps. "
            "SeaCacheFinePayload: fine_reuse|fine_taylor_o1|fine_taylor_o2|fine_hicache_o2. "
            "SeaCacheSegmentPayload: segment_reuse|segment_taylor_o1|segment_taylor_o2|"
            "segment_hicache_o2|segment_ensemble_mean."
        ),
    )
    p.add_argument("--payload_blend", type=float, default=1.0,
                   help="SeaCachePayload: payload = (1-w)*reuse + w*forecast.")
    p.add_argument("--payload_sigma", type=float, default=0.5,
                   help="SeaCachePayload: HiCache Hermite sigma for hicache_o2 forecasts.")
    p.add_argument("--payload_control_seed_salt", type=int, default=0,
                   help="SeaCachePayload random/orthogonal control direction salt.")
    p.add_argument("--segment_layout", choices=sorted(SEGMENT_LAYOUTS), default="seg8")
    p.add_argument("--payload_schedule_dir", type=Path, default=None,
                   help="SeaCachePayload: required baseline decision dir for schedule-locked replay.")
    p.add_argument("--teacache_thresh", type=float, default=0.3)
    p.add_argument("--teacache_backbone", default="flux")
    p.add_argument("--teacache_variant", default=None)
    p.add_argument("--sencache_sensitivity_path", type=Path, default=None)
    p.add_argument("--sencache_thresh", type=float, default=None,
                   help="Alias for --sencache_thresh_main.")
    p.add_argument("--sencache_thresh_start", type=float, default=0.005)
    p.add_argument("--sencache_thresh_main", type=float, default=None)
    p.add_argument("--sencache_K", type=int, default=10)
    p.add_argument("--sencache_threshold_scale", default="auto")
    p.add_argument("--sencache_switch_ratio", type=float, default=0.2)
    p.add_argument("--sencache_ret_steps", type=int, default=0)
    p.add_argument("--sencache_cutoff_steps", type=int, default=-1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--prompt_ids", default=None,
                   help="Comma/space separated original prompt IDs to run. Preserves prompt IDs and seeds.")
    p.add_argument("--prompt_id_file", type=Path, default=None,
                   help="File containing original prompt IDs to run, separated by commas or whitespace.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--with_cf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compare_without_cf", action="store_true")
    p.add_argument("--compare_without_shadow", action="store_true",
                   help="Run an extra cached rollout with shadow disabled and compare native outputs.")
    p.add_argument("--save_trace_tensors", action="store_true")
    p.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shadow_depths", default="",
                   help="Optional comma/space list of prefix block depths to record, e.g. '1,2,4,8'.")
    p.add_argument("--shadow_on", choices=["all_steps", "cache_candidates"], default="all_steps",
                   help=("When shadow_depths is set, emit metrics on all steps or only native cache "
                         "candidates; full steps are still used to update anchors."))
    p.add_argument("--online_direction_sketch_dims", type=int, default=0,
                   help=(
                       "Record deterministic online direction sketches with this many "
                       "block-projection dimensions. 0 disables sketches."
                   ))
    args = p.parse_args()
    if args.sencache_thresh_main is None:
        args.sencache_thresh_main = (
            float(args.sencache_thresh) if args.sencache_thresh is not None else 0.07
        )
    if args.mode == "SenCache" and args.sencache_sensitivity_path is None:
        raise SystemExit("--mode SenCache requires --sencache_sensitivity_path")
    if args.mode == "SeaCacheFinePayload" and args.payload_mode == "reuse":
        args.payload_mode = "fine_taylor_o1"
    if args.mode == "SeaCacheSegmentPayload" and args.payload_mode == "reuse":
        args.payload_mode = "segment_taylor_o1"
    if args.mode == "SeaCachePayload" and args.payload_mode not in PAYLOAD_MODES:
        raise SystemExit("--mode SeaCachePayload requires a coarse payload_mode")
    if args.mode == "SeaCacheFinePayload" and args.payload_mode not in FINE_PAYLOAD_MODES:
        raise SystemExit("--mode SeaCacheFinePayload requires payload_mode in fine_* modes")
    if args.mode == "SeaCacheSegmentPayload" and args.payload_mode not in SEGMENT_PAYLOAD_MODES:
        raise SystemExit("--mode SeaCacheSegmentPayload requires payload_mode in segment_* modes")
    if args.mode == "SeaCachePayload" and not (0.0 <= float(args.payload_blend) <= 1.0):
        raise SystemExit("--payload_blend must be in [0, 1]")
    if args.mode == "SeaCacheFinePayload" and float(args.payload_blend) != 1.0:
        raise SystemExit("--mode SeaCacheFinePayload requires --payload_blend 1.0")
    if args.mode == "SeaCacheSegmentPayload" and float(args.payload_blend) != 1.0:
        raise SystemExit("--mode SeaCacheSegmentPayload requires --payload_blend 1.0")
    if _mode_uses_fixed_payload_schedule(args.mode) and args.payload_schedule_dir is None:
        raise SystemExit(f"--mode {args.mode} requires --payload_schedule_dir")
    args.shadow_depths_tuple = _parse_int_list(args.shadow_depths)
    return args


def main() -> int:
    args = parse_args()
    selected_ids = _selected_prompt_ids(args)
    args.selected_prompt_ids = selected_ids or []
    if args.run_name is None:
        args.run_name = _run_name_for_args(args)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    if selected_ids is None:
        prompt_records = list(enumerate(prompts_all))
    else:
        max_id = len(prompts_all) - 1
        missing = [idx for idx in selected_ids if idx > max_id]
        if missing:
            raise ValueError(
                f"prompt IDs outside loaded prompt range 0..{max_id}: {missing[:10]}; "
                "increase --limit or omit it for selected trace reruns"
            )
        prompt_records = [(idx, prompts_all[idx]) for idx in selected_ids]
    start, end = split_shard(len(prompt_records), args.shard_count, args.shard_idx)
    shard_records = prompt_records[start:end]
    if not shard_records:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded in {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} has {len(shard_records)} prompts", flush=True)

    threshold = _threshold_for_args(args)
    teardown = install_trajectory_deviation(
        pipe,
        mode=args.mode,
        threshold=threshold,
        num_steps=args.num_steps,
        first_enhance=args.first_enhance,
        payload_mode=args.payload_mode,
        payload_blend=float(args.payload_blend),
        payload_sigma=float(args.payload_sigma),
        payload_control_seed_salt=int(args.payload_control_seed_salt),
        segment_layout=args.segment_layout,
        teacache_backbone=args.teacache_backbone,
        teacache_variant=args.teacache_variant,
        sencache_sensitivity_path=(None if args.sencache_sensitivity_path is None
                                   else str(args.sencache_sensitivity_path)),
        sencache_thresh_start=float(args.sencache_thresh_start),
        sencache_thresh_main=float(args.sencache_thresh_main),
        sencache_K=int(args.sencache_K),
        sencache_threshold_scale=args.sencache_threshold_scale,
        sencache_switch_ratio=float(args.sencache_switch_ratio),
        sencache_ret_steps=int(args.sencache_ret_steps),
        sencache_cutoff_steps=int(args.sencache_cutoff_steps),
        online_direction_sketch_dims=args.online_direction_sketch_dims,
        shadow_depths=args.shadow_depths_tuple,
        shadow_on=args.shadow_on,
    )

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    try:
        for prompt_id, prompt in shard_records:
            prompt_dir = args.output_dir / f"prompt_{prompt_id:05d}"
            if args.resume and (prompt_dir / "trajectory_summary.json").is_file():
                print(f"[shard {args.shard_idx}] prompt {prompt_id} already complete, skip", flush=True)
                continue
            seed = seed_for(args.seed, prompt_id)
            t_prompt = time.perf_counter()
            rows, summary = _run_one_prompt(pipe, prompt, prompt_id, seed, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_prompt
            all_rows.extend(rows)
            summaries.append(summary)
            per_image_records.append({"idx": prompt_id, "denoise_s": float(dt), "decode_s": 0.0})
            print(f"[shard {args.shard_idx}] prompt {prompt_id} done in {dt:.1f}s "
                  f"cache_rate={summary['cache_rate']:.3f} final={summary['final_latent_drift']:.4g}",
                  flush=True)
    finally:
        teardown()

    _write_csv(args.output_dir / f"aggregate_rows_shard{args.shard_idx}of{args.shard_count}.csv", all_rows)
    _write_jsonl(args.output_dir / f"per_prompt_shard{args.shard_idx}of{args.shard_count}.jsonl", [
        {"prompt_id": s["prompt_id"], "rows": [r for r in all_rows if r["prompt_id"] == s["prompt_id"]]}
        for s in summaries
    ])
    _write_jsonl(args.output_dir / f"prompt_summaries_shard{args.shard_idx}of{args.shard_count}.jsonl", summaries)

    git_sha, git_dirty = _git_commit()
    sigmas_hash = None
    timesteps_hash = None
    if summaries:
        # All prompts share schedule; hashes are available from the row columns.
        timesteps_hash = _hash_text(",".join(str(r["timestep"]) for r in all_rows[: int(args.num_steps)]))
        sigmas_hash = _hash_text(",".join(str(r["sigma_n"]) for r in all_rows[: int(args.num_steps)]))
    manifest = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "backbone": "flux",
        "model_id": args.model_id,
        "mode": args.mode,
        "cache_params": {"threshold": threshold, **_cache_params_for_args(args)},
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "prompt_file": str(args.prompt_file),
        "limit": int(args.limit),
        "selected_prompt_ids": [int(x) for x in args.selected_prompt_ids],
        "prompt_id_file": str(args.prompt_id_file) if args.prompt_id_file else None,
        "height": int((args.height // 16) * 16),
        "width": int((args.width // 16) * 16),
        "guidance": float(args.guidance),
        "max_sequence_length": 512 if args.model_name == "flux-dev" else 256,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "diffusers_version": getattr(sys.modules.get("diffusers"), "__version__", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "timesteps_hash": timesteps_hash,
        "sigmas_hash": sigmas_hash,
        "with_cf": bool(args.with_cf),
        "compare_without_cf": bool(args.compare_without_cf),
        "compare_without_shadow": bool(args.compare_without_shadow),
        "online_direction_sketch_dims": int(args.online_direction_sketch_dims),
        "shadow_depths": [int(x) for x in args.shadow_depths_tuple],
        "shadow_on": str(args.shadow_on),
        "non_pollution_checks_enabled": True,
        "code_paths": [
            p for p in (
                "flux/trajectory_deviation_runner.py",
                "lib/history_fd_observer.py" if args.mode == "SeaCachePayload" else None,
                "flux/seacache_fine_payload.py" if args.mode == "SeaCacheFinePayload" else None,
                "flux/seacache_segment_payload.py" if args.mode == "SeaCacheSegmentPayload" else None,
                "lib/teacache_coeffs.py" if args.mode == "TeaCache" else None,
            )
            if p is not None
        ],
    }
    (args.output_dir / f"manifest_shard{args.shard_idx}of{args.shard_count}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if per_image_records:
        process_end = time.perf_counter()
        write_timing_json(
            args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json",
            per_image=per_image_records,
            config={
                "cache_mode": f"trajectory_{args.mode}_t{threshold}",
                "mode_raw": args.mode,
                "num_steps": int(args.num_steps),
                "base_seed": int(args.seed),
                "shard_idx": int(args.shard_idx),
                "shard_count": int(args.shard_count),
                "model_id": args.model_id,
                "model_name": args.model_name,
                "payload_mode": args.payload_mode if _mode_uses_fixed_payload_schedule(args.mode) else None,
                "payload_sigma": float(args.payload_sigma) if _mode_uses_fixed_payload_schedule(args.mode) else None,
                "segment_layout": args.segment_layout if args.mode == "SeaCacheSegmentPayload" else None,
                "guidance": float(args.guidance),
                "width": int(args.width),
                "height": int(args.height),
                "dtype": args.dtype,
            },
            model_load_s=float(model_load_end - t0),
            wallclock_total_s=float(process_end - t0),
            device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        )
    print(f"[shard {args.shard_idx}] wrote trajectory audit outputs under {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
