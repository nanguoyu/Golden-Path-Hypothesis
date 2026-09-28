"""Wan2.1 SeaCache-gated segment residual payload forward."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.cuda.amp as amp

from wan21.seacache import (
    _gate_kind,
    _prepare_gate_fingerprint,
    restore_original_forwards,
)
from wan21.structured_payload import (
    SEGMENT_LAYOUTS,
    SEGMENT_PAYLOAD_MODES,
    History,
    available_order,
    build_segments,
    payload_base_mode,
    payload_max_order,
    predict_from_history,
    segment_id,
    segment_key,
    segment_spec_hash,
    update_history,
)


BRANCH_NAMES = {0: "cond", 1: "uncond"}


@dataclass
class SegmentPayloadConfig:
    mode: str
    num_steps: int
    first_enhance: int = 1
    seacache_thresh: float = 0.20
    seacache_power_exp: float = 3.0
    seacache_norm_mode: str = "mean"
    payload_mode: str = "segment_taylor_o1"
    payload_sigma: float = 0.5
    segment_layout: str = "seg8"
    require_locked_schedule: bool = False


def install_segment_payload_forward(model: Any, config: SegmentPayloadConfig) -> None:
    if config.payload_mode not in SEGMENT_PAYLOAD_MODES:
        raise ValueError(f"unknown Wan2.1 segment payload mode: {config.payload_mode!r}")
    if config.segment_layout not in SEGMENT_LAYOUTS:
        raise ValueError(f"unknown Wan2.1 segment layout: {config.segment_layout!r}")
    restore_original_forwards(model)
    model._wan21_segment_payload_state = _new_state(config, num_blocks=len(model.blocks))
    model.forward = types.MethodType(_segment_payload_forward, model)


def reset_segment_payload_state(
    model: Any,
    *,
    prompt_idx: int,
    seed: int,
    locked_schedule: Optional[Dict[int, Dict[str, str]]] = None,
) -> None:
    state = getattr(model, "_wan21_segment_payload_state", None)
    if state is None:
        return
    cfg: SegmentPayloadConfig = state["config"]
    model._wan21_segment_payload_state = _new_state(
        cfg,
        num_blocks=len(model.blocks),
        prompt_idx=int(prompt_idx),
        seed=int(seed),
        locked_schedule=locked_schedule,
    )


def decisions(model: Any) -> Dict[str, Any]:
    state = getattr(model, "_wan21_segment_payload_state", None)
    if state is None:
        return {}
    cfg: SegmentPayloadConfig = state["config"]
    rows = []
    for step in range(cfg.num_steps):
        rows.append({"step": step, "branches": state["decisions"].get(step, {})})
    return {
        "mode": cfg.mode,
        "num_steps": cfg.num_steps,
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "locked_schedule": state.get("locked_schedule") is not None,
        "payload_mode": cfg.payload_mode,
        "payload_granularity": "segment",
        "segment_layout": cfg.segment_layout,
        "segment_payload_expected_segments": int(len(state["segments"])),
        "segment_payload_spec_hash": state["segment_spec_hash"],
        "segment_payload_specs": list(state["segments"]),
        "steps": rows,
        "summary": _summary(state),
    }


def _new_branch() -> Dict[str, Any]:
    return {
        "previous_fingerprint": None,
        "accumulated": 0.0,
        "last_full_step": None,
        "previous_full_step": None,
        "full_steps_seen": 0,
        "cache_age": 0,
        "full_count": 0,
        "cache_count": 0,
        "segment_history": {},
    }


def _new_state(
    config: SegmentPayloadConfig,
    *,
    num_blocks: int,
    prompt_idx: Optional[int] = None,
    seed: Optional[int] = None,
    locked_schedule: Optional[Dict[int, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    segments = build_segments(config.segment_layout, num_blocks=int(num_blocks))
    return {
        "config": config,
        "cnt": 0,
        "prompt_idx": prompt_idx,
        "seed": seed,
        "locked_schedule": locked_schedule,
        "segments": segments,
        "segment_spec_hash": segment_spec_hash(segments),
        "branches": {"cond": _new_branch(), "uncond": _new_branch()},
        "decisions": {},
    }


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, branch in state["branches"].items():
        full = int(branch["full_count"])
        cache = int(branch["cache_count"])
        total = full + cache
        out[name] = {
            "full_count": full,
            "cache_count": cache,
            "cache_ratio": float(cache / total) if total else 0.0,
        }
    return out


def _record(state: Dict[str, Any], step: int, branch_name: str, row: Dict[str, Any]) -> None:
    state["decisions"].setdefault(int(step), {})[branch_name] = row


def _expected_segments(state: Dict[str, Any]) -> int:
    return int(len(state["segments"]))


def _segment_histories(branch: Dict[str, Any]) -> Dict[tuple[int, int], History]:
    return branch["segment_history"]


def _ready_segments(branch: Dict[str, Any]) -> int:
    return sum(1 for hist in _segment_histories(branch).values() if 0 in hist)


def _available_order_range(branch: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    orders = [available_order(hist) for hist in _segment_histories(branch).values() if 0 in hist]
    if not orders:
        return None, None
    return min(orders), max(orders)


def _run_segment_full(
    self: Any,
    x: torch.Tensor,
    *,
    spec: Dict[str, int],
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    for layer in range(int(spec["start"]), int(spec["end"])):
        x = self.blocks[layer](x, **kwargs)
    return x


def _predict_segment(
    branch: Dict[str, Any],
    spec: Dict[str, int],
    *,
    step_offset: int,
    cfg: SegmentPayloadConfig,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    hist = _segment_histories(branch)[segment_key(spec)]
    pred, fields = predict_from_history(
        hist,
        step_offset=int(step_offset),
        mode=cfg.payload_mode,
        sigma=float(cfg.payload_sigma),
    )
    fields.update({
        "segment": segment_id(spec),
    })
    return pred, fields


def _decide_action(
    *,
    state: Dict[str, Any],
    branch: Dict[str, Any],
    branch_name: str,
    step: int,
    fingerprint: torch.Tensor,
) -> Dict[str, Any]:
    cfg: SegmentPayloadConfig = state["config"]
    locked = state.get("locked_schedule")
    previous_fp = branch["previous_fingerprint"]
    forced_full = (
        step < int(cfg.first_enhance)
        or step >= int(cfg.num_steps) - 1
        or previous_fp is None
    )

    distance = None
    accumulated_pre = float(branch["accumulated"])
    accumulated_post = accumulated_pre
    gate_pass = False
    if previous_fp is not None:
        from lib.gates import rel_l1

        distance = rel_l1(fingerprint, previous_fp)

    ready_pre = _ready_segments(branch)
    cache_ready_pre = ready_pre >= _expected_segments(state)
    force_full_reason = None
    if forced_full:
        if step < int(cfg.first_enhance):
            force_full_reason = "first_enhance"
        elif step >= int(cfg.num_steps) - 1:
            force_full_reason = "final_step"
        else:
            force_full_reason = "no_previous_fingerprint"

    if locked is not None:
        desired = (locked.get(int(step)) or {}).get(branch_name)
        if desired not in {"full", "cache"}:
            raise RuntimeError(f"locked schedule missing step={step} branch={branch_name}")
        if desired == "cache" and (forced_full or not cache_ready_pre):
            raise RuntimeError(
                f"locked schedule asks cache before segment payload is ready at "
                f"step={step} branch={branch_name}"
            )
        action = desired
        gate_pass = action == "cache"
    elif forced_full:
        action = "full"
        branch["accumulated"] = 0.0
        accumulated_post = 0.0
    else:
        branch["accumulated"] = accumulated_pre + float(distance)
        gate_pass = bool(branch["accumulated"] < float(cfg.seacache_thresh))
        action = "cache" if gate_pass else "full"
        if action == "cache" and not cache_ready_pre:
            action = "full"
            force_full_reason = "segment_cache_unready"
            gate_pass = False
        if action == "full":
            branch["accumulated"] = 0.0
        accumulated_post = float(branch["accumulated"])

    branch["previous_fingerprint"] = fingerprint.detach().clone()
    return {
        "gate": "seacache",
        "action": action,
        "forced_full": bool(action == "full"),
        "force_full_reason": force_full_reason,
        "schedule_locked": bool(locked is not None),
        "gate_pass": bool(gate_pass),
        "distance": None if distance is None else float(distance),
        "gate_increment": None if distance is None else float(distance),
        "accumulated_pre": accumulated_pre,
        "accumulated_post": accumulated_post,
        "segment_payload_ready_segments_pre": int(ready_pre),
        "segment_payload_missing_segments_pre": max(0, _expected_segments(state) - int(ready_pre)),
        "segment_payload_cache_ready_pre": bool(cache_ready_pre),
    }


def _segment_payload_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    state = self._wan21_segment_payload_state
    cfg: SegmentPayloadConfig = state["config"]
    step = int(state["cnt"] // 2)
    branch_name = BRANCH_NAMES[int(state["cnt"] % 2)]
    branch = state["branches"][branch_name]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x]
    )
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    from wan.modules.model import sinusoidal_embedding_1d

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    with amp.autocast(dtype=torch.float32):
        e_first = (self.blocks[0].modulation + e0).chunk(6, dim=1)
        fingerprint = self.blocks[0].norm1(x).float() * (1 + e_first[1]) + e_first[0]
    fingerprint = _prepare_gate_fingerprint(
        modulated=fingerprint,
        grid_sizes=grid_sizes,
        scheduler=getattr(self, "scheduler", None),
        step=step,
        cfg=cfg,
        gate=_gate_kind("SeaCachePayload"),
    )

    row = _decide_action(
        state=state,
        branch=branch,
        branch_name=branch_name,
        step=step,
        fingerprint=fingerprint,
    )
    row.update({
        "model_forward_idx": int(state["cnt"]),
        "branch": branch_name,
        "cache_age_pre": int(branch["cache_age"]),
        "payload_requested": cfg.payload_mode,
        "payload_mode": cfg.payload_mode,
        "payload_base_mode": payload_base_mode(cfg.payload_mode),
        "payload_fallback": False,
        "payload_unavailable": False,
        "segment_payload_enabled": True,
        "segment_payload_mode": cfg.payload_mode,
        "segment_payload_sigma": float(cfg.payload_sigma),
        "segment_payload_layout": str(cfg.segment_layout),
        "segment_payload_expected_segments": int(_expected_segments(state)),
        "segment_payload_granularity": f"segment_{cfg.segment_layout}",
        "segment_payload_spec_hash": str(state["segment_spec_hash"]),
        "segment_payload_history_source": "full_refresh_only",
        "segment_payload_feature_space": "wan_block_segment_residual",
        "segment_payload_last_full_step_pre": (
            None if branch["last_full_step"] is None else int(branch["last_full_step"])
        ),
        "segment_payload_previous_full_step_pre": (
            None if branch["previous_full_step"] is None else int(branch["previous_full_step"])
        ),
        "segment_payload_full_steps_seen": int(branch["full_steps_seen"]),
    })
    order_min, order_max = _available_order_range(branch)
    row.update({
        "segment_payload_available_order_min_pre": None if order_min is None else int(order_min),
        "segment_payload_available_order_max_pre": None if order_max is None else int(order_max),
    })

    if row["action"] == "cache":
        step_offset = 0 if branch["last_full_step"] is None else int(step) - int(branch["last_full_step"])
        fields: list[Dict[str, Any]] = []
        for spec in state["segments"]:
            pred, pred_fields = _predict_segment(
                branch,
                spec,
                step_offset=step_offset,
                cfg=cfg,
            )
            x = x + pred.to(x.dtype)
            fields.append(pred_fields)
        branch["cache_count"] += 1
        branch["cache_age"] += 1
        degraded = [f for f in fields if f.get("payload_order_degraded")]
        row.update({
            "payload_selected": cfg.payload_mode,
            "payload_available": [cfg.payload_mode],
            "segment_payload_predicted_segments": int(len(fields)),
            "segment_payload_updated_segments": 0,
            "segment_payload_step_offset": int(step_offset),
            "segment_payload_effective_predict_order": (
                None if not fields else min(int(f["payload_effective_order"]) for f in fields)
            ),
            "segment_payload_prediction_order_degraded": bool(degraded),
            "segment_payload_degraded_segments": int(len(degraded)),
        })
    else:
        last_full_pre = branch["last_full_step"]
        step_gap = 1 if last_full_pre is None else max(1, int(step) - int(last_full_pre))
        max_order = payload_max_order(cfg.payload_mode)
        histories = _segment_histories(branch)
        for spec in state["segments"]:
            before = x
            after = _run_segment_full(self, x, spec=spec, kwargs=kwargs)
            residual = after.detach() - before.detach()
            histories[segment_key(spec)] = update_history(
                histories.get(segment_key(spec)),
                residual,
                step_gap=step_gap,
                max_order=max_order,
            )
            x = after
        branch["previous_full_step"] = branch["last_full_step"]
        branch["last_full_step"] = int(step)
        branch["full_steps_seen"] += 1
        branch["full_count"] += 1
        branch["cache_age"] = 0
        row.update({
            "payload_selected": "full",
            "payload_available": [],
            "segment_payload_predicted_segments": 0,
            "segment_payload_updated_segments": int(_expected_segments(state)),
            "segment_payload_step_offset": 0,
            "segment_payload_effective_predict_order": None,
            "segment_payload_prediction_order_degraded": None,
            "segment_payload_degraded_segments": 0,
        })

    row["cache_age_post"] = int(branch["cache_age"])
    row["segment_payload_last_full_step_post"] = (
        None if branch["last_full_step"] is None else int(branch["last_full_step"])
    )
    _record(state, step, branch_name, row)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    state["cnt"] += 1
    if state["cnt"] >= int(cfg.num_steps) * 2:
        state["cnt"] = 0
    return [u.float() for u in x]
