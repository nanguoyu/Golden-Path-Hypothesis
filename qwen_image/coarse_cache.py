"""Coarse whole-transformer cache scaffold for Qwen-Image.

This module integrates SeaCache/TeaCache/SenCache whole-transformer residual
reuse and BudCache fixed schedules. TeaCache coefficients and SenCache
sensitivities remain backbone-specific calibration assets.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from lib.fixed_schedule import validate_cache_steps
from lib.gates import rel_l1
from lib.history_fd_observer import (
    forecast_predictions,
    init_state as init_history_fd_state,
    update_on_full as history_fd_update_on_full,
)
from lib.sencache import (
    SenCacheSensitivityTable,
    load_sensitivity_table,
    online_fields as sencache_online_fields,
    threshold_scale_from_latent,
)
from lib.wiener import apply_sea_with_scheduler
from qwen_image._helpers import decisions_filename


BRANCHES = ("cond", "uncond")
PAYLOAD_MODES = ("reuse", "taylor_o1", "ensemble_mean")


@dataclass(frozen=True)
class QwenCoarseConfig:
    mode: str
    num_steps: int = 50
    first_enhance: int = 1
    seacache_thresh: float = 0.3
    teacache_thresh: float = 0.3
    teacache_coeff_source: str = "identity"
    teacache_coefficients: tuple[float, ...] | None = None
    payload_mode: str = "reuse"
    payload_sigma: float = 0.5
    payload_schedule_dir: Optional[str] = None
    fixed_cache_steps: tuple[int, ...] = ()
    sencache_sensitivity_path: Optional[str] = None
    sencache_threshold_start: float = 0.005
    sencache_threshold_main: float = 0.08
    sencache_threshold_scale: str | float = "auto"
    sencache_switch_ratio: float = 0.2
    sencache_max_skip: int = 10
    sencache_ret_steps: int = 0
    sencache_cutoff_steps: int = -1
    true_cfg: bool = True


def install_qwen_coarse_forward(pipe: Any, config: QwenCoarseConfig) -> None:
    """Install instance-level Qwen coarse cache patches."""

    if config.mode not in {
        "SeaCache",
        "TeaCache",
        "SeaCachePayload",
        "SenCache",
        "BudCache",
    }:
        raise ValueError(f"unsupported Qwen coarse mode: {config.mode!r}")
    if config.payload_mode not in PAYLOAD_MODES:
        raise ValueError(
            f"unsupported Qwen coarse payload_mode: {config.payload_mode!r}"
        )
    if (
        config.mode == "SeaCachePayload"
        and config.payload_mode != "reuse"
        and not config.payload_schedule_dir
    ):
        raise ValueError(
            "Qwen SeaCachePayload forecast modes require --payload_schedule_dir"
        )
    fixed_reuse = (
        config.mode == "SeaCachePayload"
        and config.payload_mode == "reuse"
        and bool(config.fixed_cache_steps)
    )
    if config.mode == "BudCache" or fixed_reuse:
        validate_cache_steps(
            config.fixed_cache_steps,
            num_steps=int(config.num_steps),
            forced_full_steps={0},
        )
    elif config.fixed_cache_steps:
        raise ValueError(
            "fixed_cache_steps are only valid for Qwen BudCache or "
            "SeaCachePayload reuse"
        )
    if config.mode == "SenCache":
        if not config.sencache_sensitivity_path:
            raise ValueError("Qwen SenCache requires a frozen sensitivity table")
        if int(config.sencache_max_skip) < 1:
            raise ValueError("Qwen SenCache max_skip must be positive")
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not hasattr(transformer, "transformer_blocks"):
        raise TypeError(
            "pipe.transformer.transformer_blocks is required for Qwen coarse cache"
        )
    restore_qwen_coarse_forward(pipe)
    sencache_table = (
        load_sensitivity_table(config.sencache_sensitivity_path)
        if config.mode == "SenCache"
        else None
    )
    state = _new_state(
        config,
        num_layers=len(transformer.transformer_blocks),
        scheduler=getattr(pipe, "scheduler", None),
        sencache_table=sencache_table,
    )
    transformer._qwen_image_coarse_state = state
    transformer._qwen_image_coarse_original_forward = transformer.forward
    transformer.forward = types.MethodType(_transformer_forward, transformer)
    for layer, block in enumerate(transformer.transformer_blocks):
        block._qwen_image_coarse_layer_index = int(layer)
        block._qwen_image_coarse_parent_transformer = transformer
        block._qwen_image_coarse_original_forward = block.forward
        block.forward = types.MethodType(_block_forward, block)


def restore_qwen_coarse_forward(pipe: Any) -> None:
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return
    original = getattr(transformer, "_qwen_image_coarse_original_forward", None)
    if original is not None:
        transformer.forward = original
        delattr(transformer, "_qwen_image_coarse_original_forward")
    for block in getattr(transformer, "transformer_blocks", []):
        original_block = getattr(block, "_qwen_image_coarse_original_forward", None)
        if original_block is not None:
            block.forward = original_block
            delattr(block, "_qwen_image_coarse_original_forward")
        for attr in (
            "_qwen_image_coarse_layer_index",
            "_qwen_image_coarse_parent_transformer",
        ):
            if hasattr(block, attr):
                delattr(block, attr)
    if hasattr(transformer, "_qwen_image_coarse_state"):
        delattr(transformer, "_qwen_image_coarse_state")


def reset_qwen_coarse_state(
    pipe: Any,
    *,
    prompt_idx: int,
    seed: int,
    locked_action_bitstring: Optional[str] = None,
) -> None:
    """Reset per-image state, optionally selecting an in-memory fixed schedule."""

    transformer = getattr(pipe, "transformer", None)
    state = getattr(transformer, "_qwen_image_coarse_state", None)
    if state is None:
        return
    cfg: QwenCoarseConfig = state["config"]
    transformer._qwen_image_coarse_state = _new_state(
        cfg,
        num_layers=int(state["num_layers"]),
        scheduler=state.get("scheduler"),
        sencache_table=state.get("sencache_table"),
        prompt_idx=int(prompt_idx),
        seed=int(seed),
        locked_action_bitstring=locked_action_bitstring,
    )


def qwen_coarse_decisions(pipe: Any) -> Dict[str, Any]:
    transformer = getattr(pipe, "transformer", None)
    state = getattr(transformer, "_qwen_image_coarse_state", None)
    if state is None:
        return {}
    cfg: QwenCoarseConfig = state["config"]
    rows = []
    for step in range(int(cfg.num_steps)):
        meta = state["step_meta"].get(step, {"action": "missing", "reason": "missing"})
        branches = state["decisions"].get(step, {})
        rows.append(
            {
                "step": step,
                "action": meta.get("action"),
                "u": 1 if meta.get("action") == "cache" else 0,
                "reason": meta.get("reason"),
                "schedule_locked": bool(meta.get("schedule_locked", False)),
                "source_action": meta.get("source_action"),
                "schedule_u": (
                    None
                    if meta.get("source_action") is None
                    else (1 if meta.get("source_action") == "cache" else 0)
                ),
                "gate": meta.get("gate", {}),
                "branches": {branch: branches.get(branch) for branch in BRANCHES},
            }
        )
    return {
        "schema": "qwen_image_decisions.v1",
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "mode": cfg.mode,
        "granularity": "coarse_residual",
        "shared_step_action": True,
        "decision_source_branch": "cond",
        "num_steps": int(cfg.num_steps),
        "first_enhance": int(cfg.first_enhance),
        "seacache_thresh": float(cfg.seacache_thresh),
        "teacache_thresh": float(cfg.teacache_thresh),
        "teacache_coeff_source": str(cfg.teacache_coeff_source),
        "teacache_coefficients": (
            None
            if cfg.teacache_coefficients is None
            else list(cfg.teacache_coefficients)
        ),
        "payload_mode": str(cfg.payload_mode),
        "payload_sigma": float(cfg.payload_sigma),
        "payload_schedule_dir": cfg.payload_schedule_dir,
        "fixed_cache_steps": list(cfg.fixed_cache_steps),
        "sencache_sensitivity_path": cfg.sencache_sensitivity_path,
        "sencache_sensitivity_sha256": (
            None
            if state.get("sencache_table") is None
            else state["sencache_table"].sha256
        ),
        "sencache_threshold_start": float(cfg.sencache_threshold_start),
        "sencache_threshold_main": float(cfg.sencache_threshold_main),
        "sencache_threshold_scale": cfg.sencache_threshold_scale,
        "sencache_switch_ratio": float(cfg.sencache_switch_ratio),
        "sencache_max_skip": int(cfg.sencache_max_skip),
        "sencache_ret_steps": int(cfg.sencache_ret_steps),
        "sencache_cutoff_steps": int(cfg.sencache_cutoff_steps),
        "steps": rows,
        "summary": _summary(state),
    }


def flatten_decision_rows(decisions: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for row in decisions.get("steps", []):
        step = int(row.get("step", -1))
        for branch, entry in (row.get("branches") or {}).items():
            if entry is None:
                yield {"step": step, "branch": branch, "missing": True}
            else:
                payload = dict(entry)
                payload["step"] = step
                payload["branch"] = branch
                payload["action"] = entry.get("action")
                yield payload


def _new_state(
    config: QwenCoarseConfig,
    *,
    num_layers: int,
    scheduler: Any,
    sencache_table: Optional[SenCacheSensitivityTable] = None,
    prompt_idx: Optional[int] = None,
    seed: Optional[int] = None,
    locked_action_bitstring: Optional[str] = None,
) -> Dict[str, Any]:
    if locked_action_bitstring is not None:
        locked_actions = _locked_actions_from_bitstring(
            locked_action_bitstring,
            num_steps=int(config.num_steps),
            first_enhance=int(config.first_enhance),
        )
    elif config.mode == "BudCache" or (
        config.mode == "SeaCachePayload"
        and config.payload_mode == "reuse"
        and config.fixed_cache_steps
    ):
        fixed = frozenset(int(step) for step in config.fixed_cache_steps)
        locked_actions = {
            step: ("cache" if step in fixed else "full")
            for step in range(int(config.num_steps))
        }
    else:
        locked_actions = _load_locked_actions(
            config.payload_schedule_dir,
            prompt_idx,
            num_steps=int(config.num_steps),
            first_enhance=int(config.first_enhance),
        )
    return {
        "config": config,
        "num_layers": int(num_layers),
        "scheduler": scheduler,
        "sencache_table": sencache_table,
        "sencache_threshold_scale_value": None,
        "sencache_consecutive_skips": 0,
        "prompt_idx": prompt_idx,
        "seed": seed,
        "forward_call_count": 0,
        "step_meta": {},
        "decisions": {},
        "current_step": 0,
        "current_branch": "cond",
        "current_action": "full",
        "current_reason": "init",
        "current_cache_ready": False,
        "current_payload_available": False,
        "current_payload_mode_used": None,
        "current_payload_fallback_reason": None,
        "current_decided": False,
        "body_skip_applied": False,
        "body_input": None,
        "img_shapes": None,
        "gate_latent": None,
        "gate_timestep": None,
        "locked_actions": locked_actions,
        "branches": {
            branch: {
                "previous_modulated_input": None,
                "previous_residual": None,
                "accumulated_rel_l1_distance": 0.0,
                "history_fd_state": init_history_fd_state(),
                "sencache_anchor_latent": None,
                "sencache_anchor_timestep": None,
                "sencache_anchor_step": None,
            }
            for branch in BRANCHES
        },
        "stats": {branch: {"full_count": 0, "cache_count": 0} for branch in BRANCHES},
    }


def _action_from_schedule_row(row: Dict[str, Any]) -> str:
    action = row.get("action")
    if action in {"full", "cache"}:
        return str(action)
    u = row.get("u")
    if u in {0, 1, "0", "1"}:
        return "cache" if int(u) == 1 else "full"
    raise ValueError(f"bad locked schedule row: {row!r}")


def _locked_actions_from_bitstring(
    bitstring: str,
    *,
    num_steps: int,
    first_enhance: int,
) -> Dict[int, str]:
    bits = str(bitstring).strip()
    if len(bits) != int(num_steps) or set(bits) - {"0", "1"}:
        raise ValueError(
            f"locked action bitstring must contain exactly {num_steps} zero/one characters"
        )
    forced_full = set(range(max(1, int(first_enhance)))) | {int(num_steps) - 1}
    forbidden = sorted(step for step in forced_full if bits[step] == "1")
    if forbidden:
        raise ValueError(
            "locked action bitstring caches forced-full steps: "
            + ",".join(str(step) for step in forbidden)
        )
    return {step: ("cache" if bit == "1" else "full") for step, bit in enumerate(bits)}


def _load_locked_actions(
    schedule_dir: Optional[str],
    prompt_idx: Optional[int],
    *,
    num_steps: int,
    first_enhance: int,
) -> Optional[Dict[int, str]]:
    if schedule_dir is None or prompt_idx is None:
        return None
    path = Path(schedule_dir) / decisions_filename(int(prompt_idx))
    if not path.is_file():
        raise FileNotFoundError(f"Qwen locked schedule decision file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    actions: Dict[int, str] = {}
    rows = data.get("steps") or data.get("per_step")
    if isinstance(rows, list) and rows:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"bad locked schedule row in {path}: {row!r}")
            step = int(row.get("step"))
            if step in actions:
                raise ValueError(
                    f"locked schedule contains duplicate step {step}: {path}"
                )
            actions[step] = _action_from_schedule_row(row)
    elif data.get("action_bitstring"):
        bits = str(data["action_bitstring"]).strip()
        if any(bit not in "01" for bit in bits):
            raise ValueError(f"bad locked action_bitstring in {path}: {bits!r}")
        actions = {
            step: ("cache" if bit == "1" else "full") for step, bit in enumerate(bits)
        }
    else:
        raise ValueError(
            f"locked schedule has neither steps/per_step nor action_bitstring: {path}"
        )
    expected_steps = set(range(int(num_steps)))
    if set(actions) != expected_steps:
        raise ValueError(
            f"locked schedule must contain exactly steps 0..{num_steps - 1}: {path}"
        )
    forced_full = set(range(max(1, int(first_enhance)))) | {int(num_steps) - 1}
    forbidden = sorted(step for step in forced_full if actions[step] == "cache")
    if forbidden:
        raise ValueError(
            f"locked schedule caches forced-full steps {forbidden}: {path}"
        )
    return actions


def _kw_or_pos(
    args: tuple[Any, ...], kwargs: Dict[str, Any], name: str, index: int
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def _scalar_timestep(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().to(torch.float32).reshape(-1)[0].item())
    return float(value)


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg: QwenCoarseConfig = state["config"]
    step_actions = [
        state["step_meta"].get(step, {}).get("action")
        for step in range(int(cfg.num_steps))
    ]
    n_full = sum(1 for action in step_actions if action == "full")
    n_cached = sum(1 for action in step_actions if action == "cache")
    out: Dict[str, Any] = {
        "n_total": int(cfg.num_steps),
        "n_full": int(n_full),
        "n_cached": int(n_cached),
        "cache_ratio": (
            float(n_cached / int(cfg.num_steps)) if int(cfg.num_steps) > 0 else 0.0
        ),
        "branches": {},
    }
    for branch, stats in state["stats"].items():
        full = int(stats["full_count"])
        cache = int(stats["cache_count"])
        total = full + cache
        out["branches"][branch] = {
            "full_count": full,
            "cache_count": cache,
            "cache_ratio": float(cache / total) if total else 0.0,
        }
    return out


def _transformer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    state = self._qwen_image_coarse_state
    cfg: QwenCoarseConfig = state["config"]
    branches_per_step = 2 if bool(cfg.true_cfg) else 1
    call_idx = int(state["forward_call_count"])
    step = call_idx // branches_per_step
    branch = "cond" if (branches_per_step == 1 or call_idx % 2 == 0) else "uncond"
    if step >= int(cfg.num_steps):
        step = int(cfg.num_steps) - 1

    state["current_step"] = int(step)
    state["current_branch"] = branch
    state["current_action"] = "full"
    state["current_reason"] = "pre_decision"
    state["current_cache_ready"] = False
    state["current_payload_available"] = False
    state["current_payload_mode_used"] = None
    state["current_payload_fallback_reason"] = None
    state["current_decided"] = False
    state["body_skip_applied"] = False
    state["body_input"] = None
    state["img_shapes"] = _kw_or_pos(args, kwargs, "img_shapes", 4)
    latent = _kw_or_pos(args, kwargs, "hidden_states", 0)
    state["gate_latent"] = latent.detach() if isinstance(latent, torch.Tensor) else None
    state["gate_timestep"] = _scalar_timestep(_kw_or_pos(args, kwargs, "timestep", 3))
    try:
        return self._qwen_image_coarse_original_forward(*args, **kwargs)
    finally:
        _record_branch_decision(state, step, branch)
        state["forward_call_count"] = call_idx + 1


def _all_branches_cache_ready(state: Dict[str, Any]) -> bool:
    return all(
        state["branches"][branch]["previous_residual"] is not None
        for branch in BRANCHES
    )


def _teacache_rescale(state: Dict[str, Any], value: float) -> float:
    cfg: QwenCoarseConfig = state["config"]
    if cfg.teacache_coefficients is not None:
        return float(np.poly1d(tuple(cfg.teacache_coefficients))(float(value)))
    if cfg.teacache_coeff_source == "identity":
        return float(value)
    if cfg.teacache_coeff_source == "flux_transfer":
        from lib.teacache_coeffs import get_coeffs

        return float(np.poly1d(get_coeffs("flux"))(float(value)))
    raise ValueError(
        f"unknown Qwen teacache coeff source: {cfg.teacache_coeff_source!r}"
    )


def _decide_sencache_step(
    state: Dict[str, Any],
    *,
    cache_ready: bool,
) -> Dict[str, Any]:
    cfg: QwenCoarseConfig = state["config"]
    step = int(state["current_step"])
    branch = str(state["current_branch"])
    branch_state = state["branches"][branch]
    latent = state.get("gate_latent")
    timestep = state.get("gate_timestep")
    gate: Dict[str, Any] = {
        "cache_ready_all_branches": bool(cache_ready),
        "decision_source_branch": branch,
    }
    if not isinstance(latent, torch.Tensor) or timestep is None:
        gate["force_full_reason"] = "missing_gate_input"
        state["sencache_consecutive_skips"] = 0
        return {
            "action": "full",
            "reason": "missing_gate_input",
            "schedule_locked": False,
            "source_action": None,
            "gate": gate,
        }

    if state["sencache_threshold_scale_value"] is None:
        state["sencache_threshold_scale_value"] = threshold_scale_from_latent(
            latent,
            cfg.sencache_threshold_scale,
        )
    online = sencache_online_fields(
        table=state.get("sencache_table"),
        current_latent=latent,
        current_timestep=float(timestep),
        anchor_latent=branch_state.get("sencache_anchor_latent"),
        anchor_timestep=branch_state.get("sencache_anchor_timestep"),
        anchor_step=branch_state.get("sencache_anchor_step"),
    )
    gate.update(online)

    cutoff = (
        int(cfg.num_steps) - 1
        if int(cfg.sencache_cutoff_steps) < 0
        else min(int(cfg.sencache_cutoff_steps), int(cfg.num_steps))
    )
    force_reason: Optional[str] = None
    if step < int(cfg.first_enhance) or step == 0:
        force_reason = "warmup"
    elif step >= cutoff:
        force_reason = "cutoff"
    elif branch_state.get("sencache_anchor_latent") is None:
        force_reason = "no_anchor"
    elif not cache_ready:
        force_reason = "cache_not_ready_all_branches"

    switch_step = int(round(int(cfg.num_steps) * float(cfg.sencache_switch_ratio)))
    threshold_raw = (
        float(cfg.sencache_threshold_start)
        if step < switch_step
        else float(cfg.sencache_threshold_main)
    )
    threshold = threshold_raw * float(state["sencache_threshold_scale_value"])
    score = online.get("online_sencache_score_pre")
    cache_allowed = (
        force_reason is None
        and step >= int(cfg.sencache_ret_steps)
        and score is not None
        and float(score) < threshold
        and int(state["sencache_consecutive_skips"]) < int(cfg.sencache_max_skip)
    )
    if cache_allowed:
        action = "cache"
        reason = "score_below_threshold"
        state["sencache_consecutive_skips"] = (
            int(state["sencache_consecutive_skips"]) + 1
        )
    else:
        action = "full"
        reason = force_reason or (
            "max_skip_refresh"
            if int(state["sencache_consecutive_skips"]) >= int(cfg.sencache_max_skip)
            else "score_exceeds_threshold"
        )
        state["sencache_consecutive_skips"] = 0
    gate.update(
        {
            "threshold_raw": threshold_raw,
            "threshold_scale": float(state["sencache_threshold_scale_value"]),
            "threshold": threshold,
            "score": None if score is None else float(score),
            "cache_allowed": bool(cache_allowed),
            "force_full_reason": force_reason,
            "consecutive_skips": int(state["sencache_consecutive_skips"]),
        }
    )
    return {
        "action": action,
        "reason": reason,
        "schedule_locked": False,
        "source_action": None,
        "gate": gate,
    }


def _maybe_sea_filter(
    state: Dict[str, Any],
    modulated: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    fields: Dict[str, Any] = {"sea_filter_applied": False, "sea_filter_reason": None}
    img_shapes = state.get("img_shapes")
    scheduler = state.get("scheduler")
    if scheduler is None:
        fields["sea_filter_reason"] = "missing_scheduler"
        return modulated, fields
    if img_shapes is None:
        fields["sea_filter_reason"] = "missing_img_shapes"
        return modulated, fields
    if isinstance(img_shapes, (list, tuple)) and len(img_shapes) == 0:
        fields["sea_filter_reason"] = "missing_img_shapes"
        return modulated, fields
    shape = img_shapes[0] if isinstance(img_shapes, (list, tuple)) else img_shapes
    if isinstance(shape, torch.Tensor):
        if shape.ndim > 1:
            shape = shape[0]
        shape = shape.detach().cpu().tolist()
    if (
        isinstance(shape, (list, tuple))
        and shape
        and isinstance(shape[0], (list, tuple))
    ):
        shape = shape[0]
    try:
        frame, height, width = [int(x) for x in shape]
    except Exception:
        fields["sea_filter_reason"] = "bad_img_shapes"
        return modulated, fields
    if frame * height * width != int(modulated.shape[1]):
        fields["sea_filter_reason"] = "shape_mismatch"
        fields["sea_filter_shape"] = [frame, height, width, int(modulated.shape[1])]
        return modulated, fields
    reshaped = modulated.reshape(
        modulated.shape[0], frame, height, width, modulated.shape[-1]
    )
    filtered = apply_sea_with_scheduler(
        reshaped,
        scheduler,
        int(state["current_step"]),
        power_exp=2.0,
        dims=(-2, -3),
        norm_mode="mean",
    )
    fields["sea_filter_applied"] = True
    return filtered.reshape_as(modulated), fields


def _decide_step(
    state: Dict[str, Any],
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    first_block: Any,
) -> Dict[str, Any]:
    cfg: QwenCoarseConfig = state["config"]
    step = int(state["current_step"])
    branch = str(state["current_branch"])
    branch_state = state["branches"][branch]
    locked_actions = state.get("locked_actions")
    forced_action = state.get("forced_action")
    cache_ready = _all_branches_cache_ready(state)
    gate: Dict[str, Any] = {
        "cache_ready_all_branches": bool(cache_ready),
        "decision_source_branch": branch,
    }

    if forced_action is not None:
        if forced_action not in {"full", "cache"}:
            raise ValueError(f"bad forced Qwen action: {forced_action!r}")
        action = str(forced_action)
        reason = "forced_action"
        source_action = None
        if locked_actions is not None:
            source_action = locked_actions.get(step)
            gate["forced_source_action"] = source_action
        if action == "cache" and not cache_ready:
            action = "full"
            reason = "forced_cache_not_ready"
        return {
            "action": action,
            "reason": reason,
            "schedule_locked": locked_actions is not None,
            "source_action": source_action,
            "gate": gate,
        }

    if locked_actions is not None:
        source_action = locked_actions.get(step)
        if source_action is None:
            raise RuntimeError(f"locked schedule missing step={step}")
        action = str(source_action)
        reason = "locked_schedule"
        if action == "cache" and not cache_ready:
            action = "full"
            reason = "locked_cache_not_ready"
        return {
            "action": action,
            "reason": reason,
            "schedule_locked": True,
            "source_action": source_action,
            "gate": gate,
        }

    if cfg.mode == "SenCache":
        return _decide_sencache_step(state, cache_ready=cache_ready)

    force_full_reason = None
    if step < int(cfg.first_enhance):
        force_full_reason = "warmup"
    elif step == 0:
        force_full_reason = "first_step"
    elif step == int(cfg.num_steps) - 1:
        force_full_reason = "final_step"
    elif branch_state["previous_modulated_input"] is None:
        force_full_reason = "no_previous_modulated_input"
    elif not cache_ready:
        force_full_reason = "cache_not_ready_all_branches"

    img_mod_params = first_block.img_mod(temb)
    img_mod1, _img_mod2 = img_mod_params.chunk(2, dim=-1)
    img_normed = first_block.img_norm1(hidden_states)
    modulated, _gate1 = first_block._modulate(img_normed, img_mod1)
    if cfg.mode in {"SeaCache", "SeaCachePayload"}:
        modulated_for_distance, filter_fields = _maybe_sea_filter(state, modulated)
        gate.update(filter_fields)
    else:
        modulated_for_distance = modulated

    accumulator_before = float(branch_state["accumulated_rel_l1_distance"])
    distance: Optional[float] = None
    increment: Optional[float] = None
    threshold: Optional[float] = None
    accumulator_proposed = accumulator_before
    if force_full_reason is not None:
        native_action = "full"
        native_reason = force_full_reason
        gate["force_full_reason"] = force_full_reason
    else:
        distance = rel_l1(
            modulated_for_distance, branch_state["previous_modulated_input"]
        )
        if cfg.mode == "TeaCache":
            increment = _teacache_rescale(state, distance)
            threshold = float(cfg.teacache_thresh)
        else:
            increment = float(distance)
            threshold = float(cfg.seacache_thresh)
        accumulator_proposed = accumulator_before + float(increment)
        if accumulator_proposed < threshold:
            native_action = "cache"
            native_reason = "accum_below_threshold"
        else:
            native_action = "full"
            native_reason = "accum_exceeds_threshold"

    action = native_action
    reason = native_reason

    accumulator_after = 0.0 if action == "full" else accumulator_proposed
    branch_state["accumulated_rel_l1_distance"] = float(accumulator_after)
    branch_state["previous_modulated_input"] = modulated_for_distance.detach()
    gate.update(
        {
            "rel_l1": None if distance is None else float(distance),
            "increment": None if increment is None else float(increment),
            "accumulator_before": accumulator_before,
            "accumulator_after": float(accumulator_after),
            "threshold": threshold,
        }
    )
    return {
        "action": action,
        "reason": reason,
        "schedule_locked": False,
        "source_action": None,
        "gate": gate,
    }


def _payload_for_branch(
    state: Dict[str, Any], branch: str
) -> tuple[Optional[torch.Tensor], Optional[str], Optional[str]]:
    cfg: QwenCoarseConfig = state["config"]
    branch_state = state["branches"][branch]
    reuse = branch_state["previous_residual"]
    payload_mode = str(state.get("forced_payload_mode") or cfg.payload_mode)
    if reuse is None:
        return None, None, "missing_reuse"
    if cfg.mode != "SeaCachePayload" or payload_mode == "reuse":
        return reuse, "reuse", None
    preds = forecast_predictions(
        branch_state.get("history_fd_state"),
        step=int(state["current_step"]),
        sigma=float(cfg.payload_sigma),
    )
    if payload_mode == "taylor_o1":
        pred = preds.get("taylor_o1")
        if pred is None or pred.shape != reuse.shape:
            return reuse, "reuse", "taylor_o1_unavailable"
        return pred.to(dtype=reuse.dtype, device=reuse.device), "taylor_o1", None
    if payload_mode == "ensemble_mean":
        parts = [
            preds[name].to(dtype=reuse.dtype, device=reuse.device)
            for name in ("taylor_o1", "taylor_o2", "hicache_o2")
            if name in preds and preds[name].shape == reuse.shape
        ]
        if not parts:
            return reuse, "reuse", "ensemble_unavailable"
        return torch.stack(parts, dim=0).mean(dim=0), "ensemble_mean", None
    return reuse, "reuse", f"unsupported_payload:{payload_mode}"


def _record_branch_decision(state: Dict[str, Any], step: int, branch: str) -> None:
    action = str(state.get("current_action", "full"))
    reason = str(state.get("current_reason", "unknown"))
    state["stats"][branch][f"{action}_count"] += 1
    state["decisions"].setdefault(int(step), {})[branch] = {
        "action": action,
        "reason": reason,
        "cache_ready": bool(state.get("current_cache_ready", False)),
        "payload_available": bool(state.get("current_payload_available", False)),
        "payload_mode_requested": str(state["config"].payload_mode),
        "payload_mode_used": state.get("current_payload_mode_used"),
        "payload_fallback_reason": state.get("current_payload_fallback_reason"),
    }


def _block_forward(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent = getattr(self, "_qwen_image_coarse_parent_transformer")
    state = parent._qwen_image_coarse_state
    layer = int(getattr(self, "_qwen_image_coarse_layer_index"))
    branch = str(state["current_branch"])

    if layer == 0 and not bool(state["current_decided"]):
        temb = kwargs.get("temb")
        if temb is None and len(args) >= 2:
            temb = args[1]
        if temb is None:
            raise TypeError("Qwen coarse block wrapper requires temb")
        state["body_input"] = hidden_states.detach()
        step = int(state["current_step"])
        if step in state["step_meta"]:
            meta = state["step_meta"][step]
        else:
            meta = _decide_step(
                state, hidden_states=hidden_states, temb=temb, first_block=self
            )
            state["step_meta"][step] = meta
        state["current_action"] = str(meta["action"])
        state["current_reason"] = str(meta["reason"])
        state["current_cache_ready"] = bool(
            meta.get("gate", {}).get("cache_ready_all_branches", False)
        )
        state["current_decided"] = True

    if state["current_action"] == "cache":
        if layer == 0 and not bool(state["body_skip_applied"]):
            payload, mode_used, fallback_reason = _payload_for_branch(state, branch)
            if payload is None:
                state["current_action"] = "full"
                state["current_reason"] = "payload_unavailable_force_full"
                state["step_meta"][int(state["current_step"])]["action"] = "full"
                state["step_meta"][int(state["current_step"])]["reason"] = state[
                    "current_reason"
                ]
            else:
                hidden_states = hidden_states + payload
                state["body_skip_applied"] = True
                state["current_payload_available"] = True
                state["current_payload_mode_used"] = mode_used
                state["current_payload_fallback_reason"] = fallback_reason
                return encoder_hidden_states, hidden_states
        elif bool(state["body_skip_applied"]):
            return encoder_hidden_states, hidden_states

    encoder_hidden_states, hidden_states = self._qwen_image_coarse_original_forward(
        hidden_states,
        encoder_hidden_states,
        *args,
        **kwargs,
    )
    if layer == int(state["num_layers"]) - 1 and state["current_action"] == "full":
        body_input = state.get("body_input")
        if body_input is not None and tuple(body_input.shape) == tuple(
            hidden_states.shape
        ):
            residual = hidden_states.detach() - body_input
            branch_state = state["branches"][branch]
            branch_state["previous_residual"] = residual
            branch_state["history_fd_state"] = history_fd_update_on_full(
                branch_state.get("history_fd_state"),
                residual=residual,
                step=int(state["current_step"]),
                max_order=2,
                sigma=float(state["config"].payload_sigma),
            )
            if state["config"].mode == "SenCache":
                gate_latent = state.get("gate_latent")
                gate_timestep = state.get("gate_timestep")
                if isinstance(gate_latent, torch.Tensor) and gate_timestep is not None:
                    branch_state["sencache_anchor_latent"] = gate_latent.detach()
                    branch_state["sencache_anchor_timestep"] = float(gate_timestep)
                    branch_state["sencache_anchor_step"] = int(state["current_step"])
            state["current_payload_available"] = True
            state["current_payload_mode_used"] = "full"
            state["current_payload_fallback_reason"] = None
    return encoder_hidden_states, hidden_states
