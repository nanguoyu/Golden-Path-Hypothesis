"""Wan2.1 SeaCache-gated fine feature payload forward."""

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
    FINE_PAYLOAD_MODES,
    History,
    available_order,
    payload_base_mode,
    payload_max_order,
    predict_from_history,
    update_history,
)


BRANCH_NAMES = {0: "cond", 1: "uncond"}
BRANCH_TO_STREAM = {"cond": "cond_stream", "uncond": "uncond_stream"}
MODULES = ("self-attention", "cross-attention", "ffn")


@dataclass
class FinePayloadConfig:
    mode: str
    num_steps: int
    first_enhance: int = 1
    seacache_thresh: float = 0.20
    seacache_power_exp: float = 3.0
    seacache_norm_mode: str = "mean"
    payload_mode: str = "fine_taylor_o1"
    payload_sigma: float = 0.5
    require_locked_schedule: bool = False


def install_fine_payload_forward(model: Any, config: FinePayloadConfig) -> None:
    if config.payload_mode not in FINE_PAYLOAD_MODES:
        raise ValueError(f"unknown Wan2.1 fine payload mode: {config.payload_mode!r}")
    restore_original_forwards(model)
    model._wan21_fine_payload_state = _new_state(config)
    model.forward = types.MethodType(_fine_payload_forward, model)


def reset_fine_payload_state(
    model: Any,
    *,
    prompt_idx: int,
    seed: int,
    locked_schedule: Optional[Dict[int, Dict[str, str]]] = None,
) -> None:
    state = getattr(model, "_wan21_fine_payload_state", None)
    if state is None:
        return
    cfg: FinePayloadConfig = state["config"]
    model._wan21_fine_payload_state = _new_state(
        cfg,
        prompt_idx=int(prompt_idx),
        seed=int(seed),
        locked_schedule=locked_schedule,
    )


def decisions(model: Any) -> Dict[str, Any]:
    state = getattr(model, "_wan21_fine_payload_state", None)
    if state is None:
        return {}
    cfg: FinePayloadConfig = state["config"]
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
        "payload_granularity": "fine",
        "fine_payload_expected_slots": int(len(state["layers"]) * len(MODULES)),
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
        "slot_history": {},
    }


def _new_state(
    config: FinePayloadConfig,
    *,
    prompt_idx: Optional[int] = None,
    seed: Optional[int] = None,
    locked_schedule: Optional[Dict[int, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "config": config,
        "cnt": 0,
        "prompt_idx": prompt_idx,
        "seed": seed,
        "locked_schedule": locked_schedule,
        "branches": {"cond": _new_branch(), "uncond": _new_branch()},
        "decisions": {},
        "layers": [],
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


def _slot_key(layer: int, module: str) -> tuple[int, str]:
    return int(layer), str(module)


def _expected_slots(state: Dict[str, Any]) -> int:
    return int(len(state["layers"]) * len(MODULES))


def _slot_histories(branch: Dict[str, Any]) -> Dict[tuple[int, str], History]:
    return branch["slot_history"]


def _ready_slots(branch: Dict[str, Any]) -> int:
    return sum(1 for hist in _slot_histories(branch).values() if 0 in hist)


def _available_order_range(branch: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    orders = [available_order(hist) for hist in _slot_histories(branch).values() if 0 in hist]
    if not orders:
        return None, None
    return min(orders), max(orders)


def _block_full(
    block: Any,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    context: torch.Tensor,
    context_lens: Any,
    state: Dict[str, Any],
    branch: Dict[str, Any],
    layer: int,
    step: int,
    step_gap: int,
) -> torch.Tensor:
    cfg: FinePayloadConfig = state["config"]
    max_order = payload_max_order(cfg.payload_mode)
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        chunks = (block.modulation + e).chunk(6, dim=1)
    assert chunks[0].dtype == torch.float32

    y = block.self_attn(
        block.norm1(x).float() * (1 + chunks[1]) + chunks[0],
        seq_lens,
        grid_sizes,
        freqs,
    )
    _update_slot(branch, layer, "self-attention", y, step_gap=step_gap, max_order=max_order)
    with amp.autocast(dtype=torch.float32):
        x = x + y * chunks[2]

    y = block.cross_attn(block.norm3(x), context, context_lens)
    _update_slot(branch, layer, "cross-attention", y, step_gap=step_gap, max_order=max_order)
    x = x + y

    y = block.ffn(block.norm2(x).float() * (1 + chunks[4]) + chunks[3])
    _update_slot(branch, layer, "ffn", y, step_gap=step_gap, max_order=max_order)
    with amp.autocast(dtype=torch.float32):
        x = x + y * chunks[5]
    return x


def _update_slot(
    branch: Dict[str, Any],
    layer: int,
    module: str,
    feature: torch.Tensor,
    *,
    step_gap: int,
    max_order: int,
) -> None:
    key = _slot_key(layer, module)
    histories = _slot_histories(branch)
    histories[key] = update_history(
        histories.get(key),
        feature,
        step_gap=step_gap,
        max_order=max_order,
    )


def _block_forecast(
    block: Any,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    state: Dict[str, Any],
    branch: Dict[str, Any],
    layer: int,
    step_offset: int,
) -> tuple[torch.Tensor, list[Dict[str, Any]]]:
    cfg: FinePayloadConfig = state["config"]
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        chunks = (block.modulation + e).chunk(6, dim=1)
    assert chunks[0].dtype == torch.float32

    fields: list[Dict[str, Any]] = []
    sa, sa_fields = _predict_slot(branch, layer, "self-attention", step_offset, cfg)
    ca, ca_fields = _predict_slot(branch, layer, "cross-attention", step_offset, cfg)
    ffn, ffn_fields = _predict_slot(branch, layer, "ffn", step_offset, cfg)
    fields.extend([sa_fields, ca_fields, ffn_fields])

    with amp.autocast(dtype=torch.float32):
        x = x + sa.to(x.dtype) * chunks[2]
    x = x + ca.to(x.dtype)
    with amp.autocast(dtype=torch.float32):
        x = x + ffn.to(x.dtype) * chunks[5]
    return x, fields


def _predict_slot(
    branch: Dict[str, Any],
    layer: int,
    module: str,
    step_offset: int,
    cfg: FinePayloadConfig,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    hist = _slot_histories(branch)[_slot_key(layer, module)]
    pred, fields = predict_from_history(
        hist,
        step_offset=int(step_offset),
        mode=cfg.payload_mode,
        sigma=float(cfg.payload_sigma),
    )
    fields.update({
        "layer": int(layer),
        "module": str(module),
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
    cfg: FinePayloadConfig = state["config"]
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

    ready_pre = _ready_slots(branch)
    cache_ready_pre = ready_pre >= _expected_slots(state)
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
                f"locked schedule asks cache before fine payload is ready at "
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
            force_full_reason = "fine_cache_unready"
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
        "fine_payload_slots_ready_pre": int(ready_pre),
        "fine_payload_slots_missing_pre": max(0, _expected_slots(state) - int(ready_pre)),
        "fine_payload_cache_ready_pre": bool(cache_ready_pre),
    }


def _fine_payload_forward(
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

    state = self._wan21_fine_payload_state
    cfg: FinePayloadConfig = state["config"]
    state["layers"] = list(range(len(self.blocks)))
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
        "stream": BRANCH_TO_STREAM[branch_name],
        "cache_age_pre": int(branch["cache_age"]),
        "payload_requested": cfg.payload_mode,
        "payload_mode": cfg.payload_mode,
        "payload_base_mode": payload_base_mode(cfg.payload_mode),
        "payload_fallback": False,
        "payload_unavailable": False,
        "fine_payload_enabled": True,
        "fine_payload_mode": cfg.payload_mode,
        "fine_payload_sigma": float(cfg.payload_sigma),
        "fine_payload_expected_slots": int(_expected_slots(state)),
        "fine_payload_granularity": "fine_96",
        "fine_payload_history_source": "full_refresh_only",
        "fine_payload_feature_space": "wan_block_submodule_output",
        "fine_payload_last_full_step_pre": (
            None if branch["last_full_step"] is None else int(branch["last_full_step"])
        ),
        "fine_payload_previous_full_step_pre": (
            None if branch["previous_full_step"] is None else int(branch["previous_full_step"])
        ),
        "fine_payload_full_steps_seen": int(branch["full_steps_seen"]),
    })

    order_min, order_max = _available_order_range(branch)
    row.update({
        "fine_payload_available_order_min_pre": None if order_min is None else int(order_min),
        "fine_payload_available_order_max_pre": None if order_max is None else int(order_max),
    })

    if row["action"] == "cache":
        step_offset = 0 if branch["last_full_step"] is None else int(step) - int(branch["last_full_step"])
        slot_fields: list[Dict[str, Any]] = []
        for layer, block in enumerate(self.blocks):
            x, fields = _block_forecast(
                block,
                x,
                e=e0,
                state=state,
                branch=branch,
                layer=layer,
                step_offset=step_offset,
            )
            slot_fields.extend(fields)
        branch["cache_count"] += 1
        branch["cache_age"] += 1
        degraded = [f for f in slot_fields if f.get("payload_order_degraded")]
        row.update({
            "payload_selected": cfg.payload_mode,
            "payload_available": [cfg.payload_mode],
            "fine_payload_predicted_slots": int(len(slot_fields)),
            "fine_payload_updated_slots": 0,
            "fine_payload_step_offset": int(step_offset),
            "fine_payload_effective_predict_order": (
                None if not slot_fields else min(int(f["payload_effective_order"]) for f in slot_fields)
            ),
            "fine_payload_prediction_order_degraded": bool(degraded),
            "fine_payload_degraded_slots": int(len(degraded)),
        })
    else:
        last_full_pre = branch["last_full_step"]
        step_gap = 1 if last_full_pre is None else max(1, int(step) - int(last_full_pre))
        for layer, block in enumerate(self.blocks):
            x = _block_full(
                block,
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                state=state,
                branch=branch,
                layer=layer,
                step=step,
                step_gap=step_gap,
            )
        branch["previous_full_step"] = branch["last_full_step"]
        branch["last_full_step"] = int(step)
        branch["full_steps_seen"] += 1
        branch["full_count"] += 1
        branch["cache_age"] = 0
        row.update({
            "payload_selected": "full",
            "payload_available": [],
            "fine_payload_predicted_slots": 0,
            "fine_payload_updated_slots": int(_expected_slots(state)),
            "fine_payload_step_offset": 0,
            "fine_payload_effective_predict_order": None,
            "fine_payload_prediction_order_degraded": None,
            "fine_payload_degraded_slots": 0,
        })

    row["cache_age_post"] = int(branch["cache_age"])
    row["fine_payload_last_full_step_post"] = (
        None if branch["last_full_step"] is None else int(branch["last_full_step"])
    )
    _record(state, step, branch_name, row)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    state["cnt"] += 1
    if state["cnt"] >= int(cfg.num_steps) * 2:
        state["cnt"] = 0
    return [u.float() for u in x]
