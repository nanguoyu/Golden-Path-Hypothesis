#!/usr/bin/env python3
"""Qwen-Image same-prefix payload fork probe.

This is the Qwen counterpart of the FLUX payload causal fork probe.  It compares
reuse and forecast payloads at the same cached prefix state under a locked
Qwen SeaCachePayload schedule.  The terminal comparison is latent-space drift
against the same prompt's full 50-step Qwen trajectory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.history_fd_observer import clone_state as clone_history_fd_state  # noqa: E402
from lib.io_utils import read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402
from qwen_image._helpers import DEFAULT_PROMPT_FILE, git_sha  # noqa: E402
from qwen_image.coarse_cache import (  # noqa: E402
    QwenCoarseConfig,
    install_qwen_coarse_forward,
    reset_qwen_coarse_state,
)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT_FILE)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default="")
    p.add_argument("--payload_schedule_dir", type=Path, required=True)
    p.add_argument("--payload_mode", choices=["taylor_o1", "ensemble_mean"], default="taylor_o1")
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--tail_policies", default="locked-after")
    p.add_argument("--fork_steps", default="6,20,35,36,38,39,42,43,46")
    p.add_argument("--prompt_ids", default="")
    p.add_argument("--prompt_id_file", type=Path, default=None)
    p.add_argument("--model_id", default="Qwen/Qwen-Image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--width", type=int, default=1328)
    p.add_argument("--height", type=int, default=1328)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--seacache_thresh", type=float, default=0.38)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _parse_ints(text: str = "") -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(text).replace(",", " ").split():
        if not tok.strip():
            continue
        value = int(tok)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_tail_policies(text: str) -> List[str]:
    policies = [tok.strip() for tok in str(text).replace(",", " ").split() if tok.strip()]
    for policy in policies:
        if policy not in {"locked-after", "full-after"}:
            raise ValueError(f"unsupported Qwen tail policy: {policy!r}")
    return policies


def _selected_prompt_ids(args: argparse.Namespace, prompt_count: int) -> List[int]:
    ids = _parse_ints(args.prompt_ids)
    if args.prompt_id_file:
        ids.extend(_parse_ints(args.prompt_id_file.read_text(encoding="utf-8")))
    if not ids:
        return list(range(prompt_count))
    out: List[int] = []
    seen = set()
    for idx in ids:
        if idx < 0 or idx >= prompt_count:
            raise ValueError(f"prompt id {idx} outside 0..{prompt_count - 1}")
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return sorted(out)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(t: torch.Tensor) -> float:
    return float(t.detach().to(torch.float32).norm().item())


def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().clone()


def _clone_branch_state(branch_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "previous_modulated_input": _clone_tensor(branch_state.get("previous_modulated_input")),
        "previous_residual": _clone_tensor(branch_state.get("previous_residual")),
        "accumulated_rel_l1_distance": float(branch_state.get("accumulated_rel_l1_distance", 0.0)),
        "history_fd_state": clone_history_fd_state(branch_state.get("history_fd_state")),
    }


def _clone_coarse_state(state: Dict[str, Any]) -> Dict[str, Any]:
    cloned = {
        "config": state["config"],
        "num_layers": int(state["num_layers"]),
        "scheduler": state.get("scheduler"),
        "prompt_idx": state.get("prompt_idx"),
        "seed": state.get("seed"),
        "forward_call_count": int(state.get("forward_call_count", 0)),
        "step_meta": copy.deepcopy(state.get("step_meta", {})),
        "decisions": copy.deepcopy(state.get("decisions", {})),
        "current_step": int(state.get("current_step", 0)),
        "current_branch": state.get("current_branch", "cond"),
        "current_action": state.get("current_action", "full"),
        "current_reason": state.get("current_reason", "snapshot"),
        "current_cache_ready": bool(state.get("current_cache_ready", False)),
        "current_payload_available": bool(state.get("current_payload_available", False)),
        "current_payload_mode_used": state.get("current_payload_mode_used"),
        "current_payload_fallback_reason": state.get("current_payload_fallback_reason"),
        "current_decided": bool(state.get("current_decided", False)),
        "body_skip_applied": bool(state.get("body_skip_applied", False)),
        "body_input": _clone_tensor(state.get("body_input")),
        "img_shapes": copy.deepcopy(state.get("img_shapes")),
        "locked_actions": None if state.get("locked_actions") is None else dict(state["locked_actions"]),
        "branches": {
            branch: _clone_branch_state(branch_state)
            for branch, branch_state in state.get("branches", {}).items()
        },
        "stats": copy.deepcopy(state.get("stats", {})),
    }
    for key in ("forced_action", "forced_payload_mode"):
        if key in state:
            cloned[key] = state[key]
    return cloned


def _snapshot(pipe: Any, latents: torch.Tensor, step_index: int) -> Dict[str, Any]:
    state = pipe.transformer._qwen_image_coarse_state
    device = latents.device
    return {
        "step_index": int(step_index),
        "latents": latents.detach().clone(),
        "scheduler_step_index": getattr(pipe.scheduler, "_step_index", None),
        "rng_cpu": torch.get_rng_state().clone(),
        "rng_cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
        "coarse_state": _clone_coarse_state(state),
    }


def _restore(pipe: Any, snap: Dict[str, Any]) -> torch.Tensor:
    latents = snap["latents"].detach().clone()
    pipe.scheduler._step_index = snap["scheduler_step_index"]
    torch.set_rng_state(snap["rng_cpu"])
    if latents.device.type == "cuda" and snap.get("rng_cuda") is not None:
        torch.cuda.set_rng_state(snap["rng_cuda"], latents.device)
    pipe.transformer._qwen_image_coarse_state = _clone_coarse_state(snap["coarse_state"])
    return latents


def _load_pipeline(args: argparse.Namespace, dtype: torch.dtype, device: str) -> Any:
    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    return pipe.to(device)


def _prepare_context(pipe: Any, prompt: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

    height = int(args.height)
    width = int(args.width)
    device = pipe._execution_device
    true_cfg_scale = float(args.true_cfg_scale)
    negative_prompt = args.negative_prompt
    if true_cfg_scale <= 1.0 or negative_prompt is None:
        raise ValueError("Qwen fork requires true CFG: true_cfg_scale > 1 and negative_prompt")

    pipe._guidance_scale = args.guidance_scale
    pipe._attention_kwargs = {}
    pipe._current_timestep = None
    pipe._interrupt = False

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
        prompt=negative_prompt,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    img_shapes = [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]]
    sigmas = np.linspace(1.0, 1 / int(args.num_steps), int(args.num_steps))
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        int(args.num_steps),
        device,
        sigmas=sigmas,
        mu=mu,
    )
    if int(num_inference_steps) != int(args.num_steps):
        raise RuntimeError(f"unexpected Qwen num_inference_steps={num_inference_steps}")

    if pipe.transformer.config.guidance_embeds:
        if args.guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-embed Qwen variants")
        guidance = torch.full([1], float(args.guidance_scale), device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    pipe.scheduler.set_begin_index(0)
    pipe.scheduler._step_index = None
    return {
        "prompt": prompt,
        "latents_init": latents.detach(),
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        "img_shapes": img_shapes,
        "guidance": guidance,
        "true_cfg_scale": true_cfg_scale,
        "attention_kwargs": {},
    }


def _step_record(pipe: Any, step_index: int, noise_pred: torch.Tensor) -> Dict[str, Any]:
    state = pipe.transformer._qwen_image_coarse_state
    meta = state.get("step_meta", {}).get(int(step_index), {})
    branches = state.get("decisions", {}).get(int(step_index), {})
    cond = branches.get("cond") or {}
    uncond = branches.get("uncond") or {}
    return {
        "step_index": int(step_index),
        "action": meta.get("action"),
        "is_cached": bool(meta.get("action") == "cache"),
        "reason": meta.get("reason"),
        "schedule_locked": bool(meta.get("schedule_locked", False)),
        "source_action": meta.get("source_action"),
        "cond_payload_used": cond.get("payload_mode_used"),
        "uncond_payload_used": uncond.get("payload_mode_used"),
        "cond_payload_fallback": cond.get("payload_fallback_reason"),
        "uncond_payload_fallback": uncond.get("payload_fallback_reason"),
        "cond_payload_available": cond.get("payload_available"),
        "uncond_payload_available": uncond.get("payload_available"),
        "noise_pred": noise_pred.detach(),
    }


def _one_step(
    pipe: Any,
    ctx: Dict[str, Any],
    latents: torch.Tensor,
    step_index: int,
    *,
    forced_action: Optional[str],
    payload_mode: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    state = pipe.transformer._qwen_image_coarse_state
    if forced_action is None:
        state.pop("forced_action", None)
    else:
        state["forced_action"] = forced_action
    state["forced_payload_mode"] = str(payload_mode)

    timestep = ctx["timesteps"][step_index]
    pipe._current_timestep = timestep
    timestep_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
    try:
        with pipe.transformer.cache_context("cond"):
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_expanded / 1000,
                guidance=ctx["guidance"],
                encoder_hidden_states_mask=ctx["prompt_embeds_mask"],
                encoder_hidden_states=ctx["prompt_embeds"],
                img_shapes=ctx["img_shapes"],
                attention_kwargs=ctx["attention_kwargs"],
                return_dict=False,
            )[0]
        with pipe.transformer.cache_context("uncond"):
            neg_noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_expanded / 1000,
                guidance=ctx["guidance"],
                encoder_hidden_states_mask=ctx["negative_prompt_embeds_mask"],
                encoder_hidden_states=ctx["negative_prompt_embeds"],
                img_shapes=ctx["img_shapes"],
                attention_kwargs=ctx["attention_kwargs"],
                return_dict=False,
            )[0]
        comb_pred = neg_noise_pred + float(ctx["true_cfg_scale"]) * (noise_pred - neg_noise_pred)
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm.clamp_min(EPS))
        latents_dtype = latents.dtype
        latents_next = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents_next.dtype != latents_dtype:
            latents_next = latents_next.to(latents_dtype)
        rec = _step_record(pipe, step_index, noise_pred)
        return latents_next.detach(), rec
    finally:
        state = pipe.transformer._qwen_image_coarse_state
        state.pop("forced_action", None)
        state.pop("forced_payload_mode", None)


def _run_full(pipe: Any, ctx: Dict[str, Any], args: argparse.Namespace, prompt_id: int, seed: int) -> torch.Tensor:
    reset_qwen_coarse_state(pipe, prompt_idx=int(prompt_id), seed=int(seed))
    pipe.scheduler._step_index = None
    latents = ctx["latents_init"].clone()
    with torch.no_grad():
        for i in range(int(args.num_steps)):
            latents, _rec = _one_step(pipe, ctx, latents, i, forced_action="full", payload_mode="reuse")
    return latents.detach()


def _native_with_snapshots(
    pipe: Any,
    ctx: Dict[str, Any],
    args: argparse.Namespace,
    prompt_id: int,
    seed: int,
    fork_steps: Sequence[int],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    reset_qwen_coarse_state(pipe, prompt_idx=int(prompt_id), seed=int(seed))
    pipe.scheduler._step_index = None
    state = pipe.transformer._qwen_image_coarse_state
    locked_actions = state.get("locked_actions") or {}
    wanted = {int(step) for step in fork_steps}
    latents = ctx["latents_init"].clone()
    rows: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, Any]] = {}
    with torch.no_grad():
        for i in range(int(args.num_steps)):
            if i in wanted and locked_actions.get(i) == "cache":
                snap = _snapshot(pipe, latents, i)
                snap["native_source_action"] = locked_actions.get(i)
                snapshots[i] = snap
            latents, rec = _one_step(pipe, ctx, latents, i, forced_action=None, payload_mode="reuse")
            row = {k: v for k, v in rec.items() if k != "noise_pred"}
            rows.append(row)
    return latents.detach(), rows, snapshots


def _run_branch(
    pipe: Any,
    ctx: Dict[str, Any],
    snap: Dict[str, Any],
    args: argparse.Namespace,
    *,
    payload_mode: str,
    tail_policy: str,
) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    start = int(snap["step_index"])
    rows: List[Dict[str, Any]] = []
    first_post: Optional[torch.Tensor] = None
    first_noise: Optional[torch.Tensor] = None
    with torch.no_grad():
        for i in range(start, int(args.num_steps)):
            if i == start:
                forced_action = "cache"
            elif tail_policy == "full-after":
                forced_action = "full"
            else:
                forced_action = None
            latents, rec = _one_step(
                pipe,
                ctx,
                latents,
                i,
                forced_action=forced_action,
                payload_mode=payload_mode,
            )
            if i == start:
                first_post = latents.detach().clone()
                first_noise = rec["noise_pred"].detach().clone()
            rows.append({k: v for k, v in rec.items() if k != "noise_pred"})
    if first_post is None or first_noise is None:
        raise RuntimeError("empty branch")
    return {
        "final": latents.detach(),
        "first_post": first_post,
        "first_noise": first_noise,
        "rows": rows,
    }


def _run_forced_full_step(pipe: Any, ctx: Dict[str, Any], snap: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    latents = _restore(pipe, snap)
    start = int(snap["step_index"])
    with torch.no_grad():
        post, rec = _one_step(pipe, ctx, latents, start, forced_action="full", payload_mode="reuse")
    return {
        "first_post": post.detach(),
        "first_noise": rec["noise_pred"].detach(),
        "rec": {k: v for k, v in rec.items() if k != "noise_pred"},
    }


def _hamming(a: Sequence[str], b: Sequence[str]) -> int:
    return int(sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)))


def _paired_rows(
    pipe: Any,
    ctx: Dict[str, Any],
    args: argparse.Namespace,
    *,
    full_final: torch.Tensor,
    native_final: torch.Tensor,
    native_rows: Sequence[Dict[str, Any]],
    snapshots: Dict[int, Dict[str, Any]],
    prompt_id: int,
    seed: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    z_full = full_final.to(torch.float32)
    z_native = native_final.to(torch.float32)
    native_seq = ["C" if row.get("is_cached") else "F" for row in native_rows]
    for fork_step, snap in sorted(snapshots.items()):
        for tail_policy in args.tail_policies_resolved:
            base = {
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "fork_step": int(fork_step),
                "tail_policy": tail_policy,
                "forecast_payload_mode": str(args.payload_mode),
                "native_source_action": snap.get("native_source_action"),
            }
            try:
                full_step = _run_forced_full_step(pipe, ctx, snap, args)
                reuse = _run_branch(pipe, ctx, snap, args, payload_mode="reuse", tail_policy=tail_policy)
                forecast = _run_branch(pipe, ctx, snap, args, payload_mode=args.payload_mode, tail_policy=tail_policy)
            except RuntimeError as exc:
                out.append({**base, "valid": False, "error": repr(exc)})
                continue

            full_noise = full_step["first_noise"].to(torch.float32)
            full_post = full_step["first_post"].to(torch.float32)
            reuse_noise = reuse["first_noise"].to(torch.float32)
            forecast_noise = forecast["first_noise"].to(torch.float32)
            reuse_post = reuse["first_post"].to(torch.float32)
            forecast_post = forecast["first_post"].to(torch.float32)
            z_reuse = reuse["final"].to(torch.float32)
            z_forecast = forecast["final"].to(torch.float32)
            reuse_rows = reuse["rows"]
            forecast_rows = forecast["rows"]
            reuse_first = reuse_rows[0] if reuse_rows else {}
            forecast_first = forecast_rows[0] if forecast_rows else {}
            reuse_seq = ["C" if row.get("is_cached") else "F" for row in reuse_rows]
            forecast_seq = ["C" if row.get("is_cached") else "F" for row in forecast_rows]
            native_suffix = native_seq[int(fork_step):]
            reuse_terminal = _norm(z_reuse - z_full)
            forecast_terminal = _norm(z_forecast - z_full)
            reuse_action = _norm(reuse_noise - full_noise)
            forecast_action = _norm(forecast_noise - full_noise)
            reuse_state = _norm(reuse_post - full_post)
            forecast_state = _norm(forecast_post - full_post)
            full_update = full_post - snap["latents"].to(torch.float32)
            out.append({
                **base,
                "valid": True,
                "error": "",
                "reuse_terminal_harm": reuse_terminal,
                "forecast_terminal_harm": forecast_terminal,
                "improvement_reuse_minus_forecast": reuse_terminal - forecast_terminal,
                "forecast_reuse_terminal_l2": _norm(z_forecast - z_reuse),
                "native_terminal_harm": _norm(z_native - z_full),
                "reuse_native_terminal_l2": _norm(z_reuse - z_native),
                "forecast_native_terminal_l2": _norm(z_forecast - z_native),
                "reuse_first_action_error": reuse_action,
                "forecast_first_action_error": forecast_action,
                "first_improvement_action_error": reuse_action - forecast_action,
                "reuse_first_action_error_rel_to_full": reuse_action / (_norm(full_noise) + EPS),
                "forecast_first_action_error_rel_to_full": forecast_action / (_norm(full_noise) + EPS),
                "first_improvement_action_error_rel_to_full": (
                    reuse_action - forecast_action
                ) / (_norm(full_noise) + EPS),
                "reuse_first_state_gap": reuse_state,
                "forecast_first_state_gap": forecast_state,
                "first_improvement_state_gap": reuse_state - forecast_state,
                "reuse_first_state_gap_rel": reuse_state / (_norm(full_post) + EPS),
                "forecast_first_state_gap_rel": forecast_state / (_norm(full_post) + EPS),
                "first_improvement_state_gap_rel": (reuse_state - forecast_state) / (_norm(full_post) + EPS),
                "forecast_minus_reuse_first_noise_l2": _norm(forecast_noise - reuse_noise),
                "forecast_minus_reuse_first_post_l2": _norm(forecast_post - reuse_post),
                "full_step_update_norm": _norm(full_update),
                "reuse_first_payload_used_cond": reuse_first.get("cond_payload_used"),
                "reuse_first_payload_used_uncond": reuse_first.get("uncond_payload_used"),
                "forecast_first_payload_used_cond": forecast_first.get("cond_payload_used"),
                "forecast_first_payload_used_uncond": forecast_first.get("uncond_payload_used"),
                "forecast_first_payload_fallback_cond": forecast_first.get("cond_payload_fallback"),
                "forecast_first_payload_fallback_uncond": forecast_first.get("uncond_payload_fallback"),
                "reuse_first_forced_action": reuse_first.get("reason"),
                "forecast_first_forced_action": forecast_first.get("reason"),
                "future_action_hamming_forecast_vs_reuse": _hamming(forecast_seq, reuse_seq),
                "reuse_action_hamming_vs_native_suffix": _hamming(reuse_seq, native_suffix),
                "forecast_action_hamming_vs_native_suffix": _hamming(forecast_seq, native_suffix),
                "reuse_cache_rate_suffix": float(sum(x == "C" for x in reuse_seq) / max(len(reuse_seq), 1)),
                "forecast_cache_rate_suffix": float(sum(x == "C" for x in forecast_seq) / max(len(forecast_seq), 1)),
            })
    return out


def _run_prompt(pipe: Any, prompt: str, prompt_id: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    seed = seed_for(int(args.seed), int(prompt_id))
    prompt_dir = args.output_dir / f"prompt_{int(prompt_id):05d}"
    rows_path = prompt_dir / "qwen_payload_causal_fork_rows.csv"
    if args.resume and rows_path.is_file() and (prompt_dir / "manifest.json").is_file():
        return []
    ctx = _prepare_context(pipe, prompt, seed, args)
    full_final = _run_full(pipe, ctx, args, prompt_id, seed)
    native_final, native_rows, snapshots = _native_with_snapshots(
        pipe,
        ctx,
        args,
        prompt_id,
        seed,
        args.fork_steps_resolved,
    )
    rows = _paired_rows(
        pipe,
        ctx,
        args,
        full_final=full_final,
        native_final=native_final,
        native_rows=native_rows,
        snapshots=snapshots,
        prompt_id=prompt_id,
        seed=seed,
    )
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows_path, rows)
    _write_json(prompt_dir / "native_reuse_rows.json", {"rows": native_rows})
    _write_json(
        prompt_dir / "manifest.json",
        {
            "schema": "qwen_payload_causal_fork_prompt.v1",
            "prompt_id": int(prompt_id),
            "seed": int(seed),
            "num_steps": int(args.num_steps),
            "payload_schedule_dir": str(args.payload_schedule_dir),
            "forecast_payload_mode": str(args.payload_mode),
            "fork_steps_requested": [int(x) for x in args.fork_steps_resolved],
            "fork_steps_materialized": sorted(int(x) for x in snapshots),
            "tail_policies": list(args.tail_policies_resolved),
            "complete": True,
        },
    )
    return rows


def main() -> int:
    args = parse_args()
    args.fork_steps_resolved = [step for step in _parse_ints(args.fork_steps) if 0 <= step < int(args.num_steps)]
    args.tail_policies_resolved = _parse_tail_policies(args.tail_policies)
    if not args.fork_steps_resolved:
        raise ValueError("no valid fork steps")
    if not args.run_name:
        args.run_name = f"qwen_payloadfork_{args.payload_mode}_n{args.limit}_s{args.seed}_{args.num_steps}"
    if not args.payload_schedule_dir.is_dir():
        raise FileNotFoundError(f"missing payload schedule dir: {args.payload_schedule_dir}")

    prompts_all = read_prompts(args.prompt_file, limit=args.limit if int(args.limit) > 0 else None)
    selected = _selected_prompt_ids(args, len(prompts_all))
    records = [(idx, prompts_all[idx]) for idx in selected]
    start, end = split_shard(len(records), int(args.shard_count), int(args.shard_idx))
    shard_records = records[start:end]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not shard_records:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.")
        return 0

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now():%H:%M:%S}] loading Qwen {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    pipe = _load_pipeline(args, dtype_map[args.dtype], device)
    install_qwen_coarse_forward(
        pipe,
        QwenCoarseConfig(
            mode="SeaCachePayload",
            num_steps=int(args.num_steps),
            first_enhance=int(args.first_enhance),
            seacache_thresh=float(args.seacache_thresh),
            payload_mode="reuse",
            payload_sigma=float(args.payload_sigma),
            payload_schedule_dir=str(args.payload_schedule_dir),
            true_cfg=True,
        ),
    )
    model_load_s = time.perf_counter() - t0
    manifest = {
        "schema": "qwen_payload_causal_fork.v1",
        "git_sha": git_sha(_PROJECT_ROOT),
        "run_name": args.run_name,
        "prompt_file": str(args.prompt_file),
        "prompt_count_after_limit": len(prompts_all),
        "selected_prompt_count": len(records),
        "payload_schedule_dir": str(args.payload_schedule_dir),
        "forecast_payload_mode": str(args.payload_mode),
        "tail_policies": list(args.tail_policies_resolved),
        "fork_steps": [int(x) for x in args.fork_steps_resolved],
        "seed": int(args.seed),
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "true_cfg_scale": float(args.true_cfg_scale),
        "dtype": args.dtype,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / f"manifest_shard{args.shard_idx:03d}of{args.shard_count:03d}.json", manifest)
    if int(args.shard_idx) == 0:
        _write_json(args.output_dir / "manifest.json", manifest)

    per_prompt = []
    wall_start = time.perf_counter()
    for prompt_id, prompt in shard_records:
        prompt_start = time.perf_counter()
        rows = _run_prompt(pipe, prompt, prompt_id, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - prompt_start
        valid_rows = sum(1 for row in rows if row.get("valid") is True)
        per_prompt.append({
            "idx": int(prompt_id),
            "prompt": prompt,
            "seed": seed_for(int(args.seed), int(prompt_id)),
            "denoise_s": float(elapsed),
            "decode_s": 0.0,
            "valid_rows": int(valid_rows),
        })
        print(
            f"[qwen_payload_fork] prompt={prompt_id} rows={len(rows)} valid={valid_rows} s={elapsed:.2f}",
            flush=True,
        )

    write_timing_json(
        args.output_dir / f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json",
        per_image=per_prompt,
        config={
            "mode": "qwen_payload_causal_fork",
            "backbone": "qwen_image",
            "payload_mode": str(args.payload_mode),
            "num_steps": int(args.num_steps),
            "seed": int(args.seed),
            "git_sha": git_sha(_PROJECT_ROOT),
        },
        model_load_s=model_load_s,
        wallclock_total_s=time.perf_counter() - wall_start + model_load_s,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    print(f"[qwen_payload_fork] done n={len(per_prompt)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
