"""Wan2.1 coarse residual cache gates and fixed-schedule payload forward."""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.cuda.amp as amp

from lib.gates import rel_l1
from lib.teacache_coeffs import get_coeffs
from lib.wiener import apply_sea_with_scheduler
from wan21 import payload as payload_lib


BRANCH_NAMES = {0: "cond", 1: "uncond"}


@dataclass
class CacheForwardConfig:
    mode: str
    num_steps: int
    first_enhance: int = 1
    seacache_thresh: float = 0.20
    seacache_power_exp: float = 3.0
    seacache_norm_mode: str = "mean"
    teacache_thresh: float = 0.30
    teacache_variant: str = "1.3b"
    payload_mode: str = "reuse"
    payload_sigma: float = 0.5
    payload_blend: float = 1.0
    require_locked_schedule: bool = False


def restore_original_forwards(model: Any) -> None:
    """Undo TaylorSeer reference monkey patches on a WanModel instance."""

    from wan.modules.model import WanAttentionBlock, WanModel

    model.forward = types.MethodType(WanModel.forward, model)
    for block in model.blocks:
        block.forward = types.MethodType(WanAttentionBlock.forward, block)


def install_cache_forward(model: Any, config: CacheForwardConfig) -> None:
    restore_original_forwards(model)
    state = _new_state(config)
    model._wan21_cache_state = state
    model.forward = types.MethodType(_cache_forward, model)


def reset_cache_state(
    model: Any,
    *,
    prompt_idx: int,
    seed: int,
    locked_schedule: Optional[Dict[int, Dict[str, str]]] = None,
) -> None:
    state = getattr(model, "_wan21_cache_state", None)
    if state is None:
        return
    cfg: CacheForwardConfig = state["config"]
    model._wan21_cache_state = _new_state(
        cfg,
        prompt_idx=int(prompt_idx),
        seed=int(seed),
        locked_schedule=locked_schedule,
    )


def decisions(model: Any) -> Dict[str, Any]:
    state = getattr(model, "_wan21_cache_state", None)
    if state is None:
        return {}
    cfg: CacheForwardConfig = state["config"]
    rows = []
    for step in range(cfg.num_steps):
        rows.append({"step": step, "branches": state["decisions"].get(step, {})})
    return {
        "mode": cfg.mode,
        "num_steps": cfg.num_steps,
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "locked_schedule": state.get("locked_schedule") is not None,
        "steps": rows,
        "summary": _summary(state),
    }


def _new_branch() -> Dict[str, Any]:
    return {
        "previous_fingerprint": None,
        "previous_residual": None,
        "accumulated": 0.0,
        "history": payload_lib.init_branch_payload_state(),
        "cache_age": 0,
        "last_full_step": None,
        "full_count": 0,
        "cache_count": 0,
    }


def _new_state(
    config: CacheForwardConfig,
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
    }


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, branch in state["branches"].items():
        total = int(branch["full_count"]) + int(branch["cache_count"])
        out[name] = {
            "full_count": int(branch["full_count"]),
            "cache_count": int(branch["cache_count"]),
            "cache_ratio": float(branch["cache_count"] / total) if total else 0.0,
        }
    return out


def _record(state: Dict[str, Any], step: int, branch_name: str, row: Dict[str, Any]) -> None:
    state["decisions"].setdefault(int(step), {})[branch_name] = row


def _poly_eval(coeffs: list[float], x: float) -> float:
    acc = 0.0
    for c in coeffs:
        acc = acc * float(x) + float(c)
    return float(acc)


def _shape_grid_modulated(modulated: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
    return modulated.reshape(
        modulated.shape[0],
        int(grid_sizes[0, 0]),
        int(grid_sizes[0, 1]),
        int(grid_sizes[0, 2]),
        modulated.shape[-1],
    )


def _prepare_gate_fingerprint(
    *,
    modulated: torch.Tensor,
    grid_sizes: torch.Tensor,
    scheduler: Any,
    step: int,
    cfg: CacheForwardConfig,
    gate: str,
) -> torch.Tensor:
    if gate == "seacache":
        video_grid = _shape_grid_modulated(modulated, grid_sizes)
        filtered = apply_sea_with_scheduler(
            video_grid,
            scheduler,
            int(step),
            dims=(-2, -3, -4),
            mode="flow",
            power_exp=float(cfg.seacache_power_exp),
            norm_mode=cfg.seacache_norm_mode,
        )
        return filtered.reshape(modulated.shape[0], -1, modulated.shape[-1])
    return modulated


def _gate_kind(mode: str) -> str:
    if mode in {"SeaCache", "SeaCachePayload"}:
        return "seacache"
    if mode in {"TeaCache", "TeaCachePayload"}:
        return "teacache"
    raise ValueError(f"unsupported Wan2.1 cache mode: {mode}")


def _decide_action(
    *,
    state: Dict[str, Any],
    branch: Dict[str, Any],
    branch_name: str,
    step: int,
    fingerprint: torch.Tensor,
) -> Dict[str, Any]:
    cfg: CacheForwardConfig = state["config"]
    gate = _gate_kind(cfg.mode)
    locked = state.get("locked_schedule")
    previous_fp = branch["previous_fingerprint"]
    previous_residual = branch["previous_residual"]
    forced_full = (
        step < int(cfg.first_enhance)
        or step >= int(cfg.num_steps) - 1
        or previous_fp is None
        or previous_residual is None
    )

    distance = None
    increment = None
    accumulated_pre = float(branch["accumulated"])
    if previous_fp is not None:
        distance = rel_l1(fingerprint, previous_fp)
        if gate == "teacache":
            increment = _poly_eval(get_coeffs("wan21", cfg.teacache_variant), distance)
        else:
            increment = float(distance)

    if locked is not None:
        desired = (locked.get(int(step)) or {}).get(branch_name)
        if desired not in {"full", "cache"}:
            raise RuntimeError(
                f"locked schedule missing step={step} branch={branch_name} "
                f"for prompt {state.get('prompt_idx')}"
            )
        if forced_full and desired == "cache":
            raise RuntimeError(
                f"locked schedule asks cache without valid state at step={step} "
                f"branch={branch_name}"
            )
        action = desired
        accumulated_post = accumulated_pre
        gate_pass = action == "cache"
    elif forced_full:
        action = "full"
        branch["accumulated"] = 0.0
        accumulated_post = 0.0
        gate_pass = False
    else:
        branch["accumulated"] = accumulated_pre + float(increment)
        threshold = cfg.seacache_thresh if gate == "seacache" else cfg.teacache_thresh
        gate_pass = bool(branch["accumulated"] < float(threshold))
        action = "cache" if gate_pass else "full"
        if action == "full":
            branch["accumulated"] = 0.0
        accumulated_post = float(branch["accumulated"])

    branch["previous_fingerprint"] = fingerprint.detach().clone()
    return {
        "gate": gate,
        "action": action,
        "forced_full": bool(forced_full),
        "schedule_locked": bool(locked is not None),
        "gate_pass": bool(gate_pass),
        "distance": None if distance is None else float(distance),
        "gate_increment": None if increment is None else float(increment),
        "accumulated_pre": accumulated_pre,
        "accumulated_post": accumulated_post,
    }


def _cache_forward(
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

    state = self._wan21_cache_state
    cfg: CacheForwardConfig = state["config"]
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
        gate=_gate_kind(cfg.mode),
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
        "history_order_pre": max(branch["history"].get("history", {}).keys(), default=-1),
    })

    if row["action"] == "cache":
        reuse = branch["previous_residual"]
        payload_tensor, payload_fields = payload_lib.choose_payload(
            branch["history"],
            step=step,
            mode=cfg.payload_mode,
            sigma=cfg.payload_sigma,
            reuse=reuse,
        )
        if cfg.payload_blend != 1.0:
            payload_tensor = (1.0 - cfg.payload_blend) * reuse + cfg.payload_blend * payload_tensor
        x = x + payload_tensor.to(x.dtype)
        branch["cache_count"] += 1
        branch["cache_age"] += 1
        row.update(payload_fields)
    else:
        ori_x = x.clone()
        for block in self.blocks:
            x = block(x, **kwargs)
        residual = x - ori_x
        branch["previous_residual"] = residual.detach().clone()
        branch["history"] = payload_lib.update_full(
            branch["history"],
            residual=residual,
            step=step,
            sigma=cfg.payload_sigma,
        )
        branch["full_count"] += 1
        branch["cache_age"] = 0
        branch["last_full_step"] = int(step)
        row.update({
            "payload_requested": cfg.payload_mode,
            "payload_selected": "full",
            "payload_available": [],
            "payload_fallback": False,
            "payload_norm": None,
            "reuse_residual_norm": None,
            "full_residual_norm": float(residual.detach().to(torch.float32).norm().item()),
        })

    row["cache_age_post"] = int(branch["cache_age"])
    row["history_order_post"] = max(branch["history"].get("history", {}).keys(), default=-1)
    _record(state, step, branch_name, row)

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    state["cnt"] += 1
    if state["cnt"] >= cfg.num_steps * 2:
        state["cnt"] = 0
    return [u.float() for u in x]
