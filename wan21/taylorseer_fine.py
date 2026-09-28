"""Wan2.1 fine-grained Taylor/HiCache-style baseline forward.

This implements the official TaylorSeer-Wan2.1 schedule and cache coordinate
inside the local Wan2.1 runner.  HiCache_fine is a local Wan adaptation: same
fixed interval schedule and fine module coordinate, but scaled-Hermite forecast.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.cuda.amp as amp

from lib.hermite import hicache_predict


BRANCH_NAMES = {0: "cond", 1: "uncond"}
BRANCH_TO_STREAM = {"cond": "cond_stream", "uncond": "uncond_stream"}
MODULES = ("self-attention", "cross-attention", "ffn")


@dataclass
class TaylorSeerFineConfig:
    num_steps: int
    fresh_threshold: int = 5
    first_enhance: int = 1
    max_order: int = 1
    mode: str = "TaylorSeer_fine"
    basis: str = "taylor"
    hicache_sigma: float = 0.5


def restore_original_forwards(model: Any) -> None:
    from wan.modules.model import WanAttentionBlock, WanModel

    model.forward = types.MethodType(WanModel.forward, model)
    for block in model.blocks:
        block.forward = types.MethodType(WanAttentionBlock.forward, block)


def install_taylorseer_fine_forward(model: Any, config: TaylorSeerFineConfig) -> None:
    restore_original_forwards(model)
    model._wan21_fine_forecast_state = _new_state(config)
    model.forward = types.MethodType(_fine_forecast_forward, model)


def install_hicache_fine_forward(model: Any, config: TaylorSeerFineConfig) -> None:
    restore_original_forwards(model)
    model._wan21_fine_forecast_state = _new_state(config)
    model.forward = types.MethodType(_fine_forecast_forward, model)


def reset_taylorseer_fine_state(model: Any, *, prompt_idx: int, seed: int) -> None:
    state = getattr(model, "_wan21_fine_forecast_state", None)
    if state is None:
        return
    cfg: TaylorSeerFineConfig = state["config"]
    model._wan21_fine_forecast_state = _new_state(
        cfg,
        prompt_idx=int(prompt_idx),
        seed=int(seed),
    )


def reset_hicache_fine_state(model: Any, *, prompt_idx: int, seed: int) -> None:
    reset_taylorseer_fine_state(model, prompt_idx=prompt_idx, seed=seed)


def decisions(model: Any) -> Dict[str, Any]:
    state = getattr(model, "_wan21_fine_forecast_state", None)
    if state is None:
        return {}
    cfg: TaylorSeerFineConfig = state["config"]
    rows = []
    for step in range(cfg.num_steps):
        rows.append({"step": step, "branches": state["decisions"].get(step, {})})
    return {
        "mode": cfg.mode,
        "num_steps": cfg.num_steps,
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "locked_schedule": False,
        "fresh_threshold": int(cfg.fresh_threshold),
        "first_enhance": int(cfg.first_enhance),
        "max_order": int(cfg.max_order),
        "basis": cfg.basis,
        "hicache_sigma": float(cfg.hicache_sigma),
        "activated_steps": list(state["activated_steps"]),
        "steps": rows,
        "summary": _summary(state),
    }


def _new_state(
    config: TaylorSeerFineConfig,
    *,
    prompt_idx: int | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    return {
        "config": config,
        "cnt": 0,
        "prompt_idx": prompt_idx,
        "seed": seed,
        "cache_counter": 0,
        "cal_threshold": int(config.fresh_threshold),
        "activated_steps": [0],
        "current_type": None,
        "step_types": {},
        "module_cache": {"cond_stream": {}, "uncond_stream": {}},
        "decisions": {},
        "stats": {
            "cond": {"full_count": 0, "cache_count": 0},
            "uncond": {"full_count": 0, "cache_count": 0},
        },
    }


def _summary(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for branch, stats in state["stats"].items():
        full = int(stats["full_count"])
        cache = int(stats["cache_count"])
        total = full + cache
        out[branch] = {
            "full_count": full,
            "cache_count": cache,
            "cache_ratio": float(cache / total) if total else 0.0,
        }
    return out


def _record(state: Dict[str, Any], step: int, branch_name: str, row: Dict[str, Any]) -> None:
    state["decisions"].setdefault(int(step), {})[branch_name] = row


def _stream_cache(state: Dict[str, Any], stream: str, layer: int, module: str) -> Dict[int, torch.Tensor]:
    stream_cache = state["module_cache"].setdefault(stream, {})
    layer_cache = stream_cache.setdefault(int(layer), {})
    return layer_cache.setdefault(module, {})


def _module_cache_complete(state: Dict[str, Any], stream: str) -> bool:
    expected_layers = int(state.get("num_layers") or 0)
    stream_cache = state["module_cache"].get(stream, {})
    if expected_layers <= 0 or len(stream_cache) < expected_layers:
        return False
    for layer in range(expected_layers):
        layer_cache = stream_cache.get(layer, {})
        for module in MODULES:
            module_cache = layer_cache.get(module)
            if not module_cache or 0 not in module_cache:
                return False
    return True


def _schedule_cond_step(state: Dict[str, Any], step: int) -> str:
    cfg: TaylorSeerFineConfig = state["config"]
    first_step = step < int(cfg.first_enhance)
    fresh_interval = int(cfg.fresh_threshold) if first_step else int(state["cal_threshold"])
    if first_step or int(state["cache_counter"]) == fresh_interval - 1:
        current_type = "full"
        state["cache_counter"] = 0
        state["activated_steps"].append(int(step))
        state["cal_threshold"] = int(round(float(cfg.fresh_threshold)))
    else:
        state["cache_counter"] = int(state["cache_counter"]) + 1
        current_type = "Taylor" if cfg.basis == "taylor" else "HiCache"
    state["current_type"] = current_type
    state["step_types"][int(step)] = current_type
    return current_type


def _update_derivatives(
    state: Dict[str, Any],
    *,
    stream: str,
    layer: int,
    module: str,
    feature: torch.Tensor,
) -> None:
    cfg: TaylorSeerFineConfig = state["config"]
    cache = _stream_cache(state, stream, layer, module)
    distance = int(state["activated_steps"][-1]) - int(state["activated_steps"][-2])
    updated: Dict[int, torch.Tensor] = {0: feature.detach().clone()}
    for order in range(int(cfg.max_order)):
        old = cache.get(order)
        if old is None or int(state["step"]) <= int(cfg.first_enhance) - 2 or distance == 0:
            break
        updated[order + 1] = ((updated[order] - old) / float(distance)).detach().clone()
    state["module_cache"][stream].setdefault(int(layer), {})[module] = updated


def _taylor_formula(derivative_dict: Dict[int, torch.Tensor], distance: int) -> torch.Tensor:
    if not derivative_dict or 0 not in derivative_dict:
        raise RuntimeError("TaylorSeer fine cache is missing order-0 module output")
    out = None
    for order in sorted(derivative_dict):
        term = derivative_dict[order] * ((float(distance) ** int(order)) / math.factorial(int(order)))
        out = term if out is None else out + term
    assert out is not None
    return out


def _forecast_formula(
    derivative_dict: Dict[int, torch.Tensor],
    *,
    distance: int,
    cfg: TaylorSeerFineConfig,
) -> torch.Tensor:
    if cfg.basis == "taylor":
        return _taylor_formula(derivative_dict, distance)
    if cfg.basis == "hicache":
        if not derivative_dict or 0 not in derivative_dict:
            raise RuntimeError("HiCache fine cache is missing order-0 module output")
        return hicache_predict(
            derivative_dict,
            step_offset=int(distance),
            sigma=float(cfg.hicache_sigma),
            max_order=int(cfg.max_order),
        )
    raise ValueError(f"unsupported fine forecast basis: {cfg.basis!r}")


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
    stream: str,
    layer: int,
) -> torch.Tensor:
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
    _update_derivatives(state, stream=stream, layer=layer, module="self-attention", feature=y)
    with amp.autocast(dtype=torch.float32):
        x = x + y * chunks[2]

    y = block.cross_attn(block.norm3(x), context, context_lens)
    _update_derivatives(state, stream=stream, layer=layer, module="cross-attention", feature=y)
    x = x + y

    y = block.ffn(block.norm2(x).float() * (1 + chunks[4]) + chunks[3])
    _update_derivatives(state, stream=stream, layer=layer, module="ffn", feature=y)
    with amp.autocast(dtype=torch.float32):
        x = x + y * chunks[5]
    return x


def _block_forecast(
    block: Any,
    x: torch.Tensor,
    *,
    e: torch.Tensor,
    state: Dict[str, Any],
    stream: str,
    layer: int,
    distance: int,
) -> torch.Tensor:
    cfg: TaylorSeerFineConfig = state["config"]
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        chunks = (block.modulation + e).chunk(6, dim=1)
    assert chunks[0].dtype == torch.float32

    layer_cache = state["module_cache"][stream][int(layer)]
    sa = _forecast_formula(layer_cache["self-attention"], distance=distance, cfg=cfg)
    ca = _forecast_formula(layer_cache["cross-attention"], distance=distance, cfg=cfg)
    ffn = _forecast_formula(layer_cache["ffn"], distance=distance, cfg=cfg)

    with amp.autocast(dtype=torch.float32):
        x = x + sa * chunks[2]
    x = x + ca
    with amp.autocast(dtype=torch.float32):
        x = x + ffn * chunks[5]
    return x


def _fine_forecast_forward(
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

    state = self._wan21_fine_forecast_state
    cfg: TaylorSeerFineConfig = state["config"]
    step = int(state["cnt"] // 2)
    branch_name = BRANCH_NAMES[int(state["cnt"] % 2)]
    stream = BRANCH_TO_STREAM[branch_name]
    state["step"] = step
    state["num_layers"] = len(self.blocks)

    if branch_name == "cond":
        current_type = _schedule_cond_step(state, step)
    else:
        current_type = state.get("current_type")
        if current_type not in {"full", "Taylor", "HiCache"}:
            raise RuntimeError(f"{cfg.mode} uncond stream reached before cond at step {step}")

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

    action = "full" if current_type == "full" else "cache"
    cache_complete_pre = _module_cache_complete(state, stream)
    distance = int(step) - int(state["activated_steps"][-1])

    if current_type == "full":
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
                stream=stream,
                layer=layer,
            )
    elif current_type in {"Taylor", "HiCache"}:
        if not cache_complete_pre:
            raise RuntimeError(
                f"{cfg.mode} cache incomplete before forecast step={step} stream={stream}"
            )
        for layer, block in enumerate(self.blocks):
            x = _block_forecast(
                block,
                x,
                e=e0,
                state=state,
                stream=stream,
                layer=layer,
                distance=distance,
            )
    else:
        raise ValueError(f"unsupported {cfg.mode} type: {current_type}")

    cache_complete_post = _module_cache_complete(state, stream)
    stats = state["stats"][branch_name]
    if action == "full":
        stats["full_count"] += 1
    else:
        stats["cache_count"] += 1

    _record(
        state,
        step,
        branch_name,
        {
            "gate": cfg.mode.lower(),
            "action": action,
            "current_type": current_type,
            "model_forward_idx": int(state["cnt"]),
            "branch": branch_name,
            "stream": stream,
            "fresh_threshold": int(cfg.fresh_threshold),
            "first_enhance": int(cfg.first_enhance),
            "max_order": int(cfg.max_order),
            "basis": cfg.basis,
            "hicache_sigma": float(cfg.hicache_sigma),
            "cache_counter": int(state["cache_counter"]),
            "cal_threshold": int(state["cal_threshold"]),
            "activated_step": int(state["activated_steps"][-1]),
            "activated_steps_tail": [int(v) for v in state["activated_steps"][-3:]],
            "distance": int(distance),
            "module_count": int(len(self.blocks) * len(MODULES)),
            "module_cache_complete_pre": bool(cache_complete_pre),
            "module_cache_complete": bool(cache_complete_post),
            "payload_requested": cfg.mode,
            "payload_selected": (
                f"fine_{cfg.basis}_o{int(cfg.max_order)}" if action == "cache" else "full"
            ),
            "payload_available": (
                [f"fine_{cfg.basis}_o{int(cfg.max_order)}"] if action == "cache" else []
            ),
            "payload_fallback": False,
            "schedule_locked": False,
            "forced_full": bool(action == "full"),
        },
    )

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    state["cnt"] += 1
    if state["cnt"] >= int(cfg.num_steps) * 2:
        state["cnt"] = 0
    return [u.float() for u in x]
