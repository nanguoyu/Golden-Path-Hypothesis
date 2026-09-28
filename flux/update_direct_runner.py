#!/usr/bin/env python3
"""Direct update forecast runner for FLUX fixed-schedule probes.

This is a research-only intervention runner.  It does not construct a
transformer hidden residual payload.  Instead, it replays a fixed SeaCache
cached/full schedule and, on cached steps, advances the latent with a
history-only forecast of the scheduler update:

    z_{k+1} = z_k + \\hat U_k

where update history is updated only on committed full steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.oracle_runner import _decode_to_pil  # noqa: E402
from lib.io_utils import image_filename, read_prompts, seed_for, split_shard, write_timing_json  # noqa: E402
from lib.update_history_observer import (  # noqa: E402
    forecast_predictions as update_history_forecast_predictions,
    init_state as init_update_history_state,
    online_fields as update_history_online_fields,
    update_on_full as update_history_update_on_full,
)

EPS = 1e-12
UPDATE_MODES = ("reuse", "taylor_o1", "taylor_o2", "hicache_o2", "ensemble_mean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--payload_schedule_dir", type=Path, required=True)
    p.add_argument("--update_mode", choices=UPDATE_MODES, default="taylor_o1")
    p.add_argument("--update_sigma", type=float, default=0.5)
    p.add_argument("--forecast_space", choices=("update", "velocity"), default="update",
                   help=("History tensor to forecast on cached steps. `update` keeps the "
                         "original direct-update experiment. `velocity` forecasts the "
                         "transformer/model output v_k and applies the current solver "
                         "factor H_k at the intervention step."))
    p.add_argument("--shadow_full_update", action="store_true")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _norm(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    return float(tensor.detach().to(torch.float32).norm().item())


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


def _scaled(tensor: Optional[torch.Tensor], scale: float) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().to(torch.float32) * float(scale)


def _read_schedule_steps(schedule_dir: Path, prompt_idx: int, *, expected_num_steps: int) -> Set[int]:
    candidates = [
        schedule_dir / f"decisions_{prompt_idx:05d}.json",
        schedule_dir / f"prompt_{prompt_idx:05d}" / "decisions.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"missing schedule for prompt {prompt_idx}: {tried}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        schedule_prompt_idx = payload.get("prompt_idx")
        if schedule_prompt_idx is not None and int(schedule_prompt_idx) != int(prompt_idx):
            raise ValueError(
                f"schedule prompt_idx mismatch in {path}: "
                f"expected {prompt_idx}, got {schedule_prompt_idx}"
            )
        schedule_mode = payload.get("mode")
        if schedule_mode is not None and str(schedule_mode) != "SeaCachePayload":
            raise ValueError(
                f"schedule {path} must come from SeaCachePayload(reuse); "
                f"got mode={schedule_mode!r}"
            )
        schedule_payload_mode = payload.get("payload_mode")
        if schedule_payload_mode is not None and str(schedule_payload_mode) != "reuse":
            raise ValueError(
                f"schedule {path} must come from SeaCachePayload(reuse); "
                f"got payload_mode={schedule_payload_mode!r}"
            )
        rows = payload.get("per_step") or payload.get("decisions") or []
    else:
        raise ValueError(f"unsupported schedule JSON: {path}")
    if not isinstance(rows, list):
        raise ValueError(f"schedule rows are not a list: {path}")
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


def _ensure_step_index(scheduler, timestep) -> Optional[int]:
    if getattr(scheduler, "_step_index", None) is None and hasattr(scheduler, "_init_step_index"):
        scheduler._init_step_index(timestep)
    idx = getattr(scheduler, "_step_index", None)
    return None if idx is None else int(idx)


def _transformer_output(pipe, latents: torch.Tensor, timestep: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
    t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
    return pipe.transformer(
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


def _advance_scheduler_state(scheduler, timestep: torch.Tensor, latents: torch.Tensor) -> None:
    dummy_output = torch.zeros_like(latents)
    scheduler.step(dummy_output, timestep, latents, return_dict=False)


def _run_prompt(pipe, prompt: str, prompt_idx: int, args: argparse.Namespace) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    ctx = _encode_and_prepare(pipe, prompt, seed_for(args.seed, prompt_idx), args)
    action_steps = _read_schedule_steps(
        args.payload_schedule_dir,
        prompt_idx,
        expected_num_steps=int(args.num_steps),
    )
    sigmas = [float(x) for x in ctx["sigmas"]]
    latents = ctx["latents_init"].clone()
    history_state = init_update_history_state()
    mode_name = "SeaCacheDirectUpdate" if args.forecast_space == "update" else "SeaCacheDirectVelocity"
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for step, timestep in enumerate(ctx["timesteps"]):
            schedule_u = int(step in action_steps)
            do_full = not bool(schedule_u)
            step_size = sigmas[step + 1] - sigmas[step]
            row: Dict[str, Any] = {
                "step": int(step),
                "u": int(schedule_u),
                "schedule_locked": True,
                "schedule_u": int(schedule_u),
                "mode": mode_name,
                "update_mode": str(args.update_mode),
                "update_sigma": float(args.update_sigma),
                "forecast_space": str(args.forecast_space),
                "sigma_n": sigmas[step],
                "sigma_np1": sigmas[step + 1],
                "step_size_H": float(step_size),
                "abs_step_size_H": abs(float(step_size)),
            }
            row.update(update_history_online_fields(
                history_state,
                step=step,
                prefix=f"online_{args.forecast_space}_fd",
            ))

            if do_full:
                noise_pred = _transformer_output(pipe, latents, timestep, ctx)
                scheduler_step_index = _ensure_step_index(pipe.scheduler, timestep)
                latents_pre = latents
                latents_next = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                full_update = latents_next.detach().to(torch.float32) - latents_pre.detach().to(torch.float32)
                full_velocity = noise_pred.detach().to(torch.float32)
                history_tensor = full_update if args.forecast_space == "update" else full_velocity
                history_state = update_history_update_on_full(
                    history_state,
                    tensor=history_tensor,
                    step=step,
                    max_order=2,
                    sigma=float(args.update_sigma),
                    ema_beta=0.2,
                )
                row.update({
                    "scheduler_step_index": scheduler_step_index,
                    "direct_update_used": False,
                    "direct_update_available": None,
                    "direct_update_fallback": None,
                    "direct_update_fallback_reason": None,
                    "full_update_norm": _norm(full_update),
                    "full_velocity_norm": _norm(full_velocity),
                    "forecast_update_norm": None,
                    "forecast_velocity_norm": None,
                    "reuse_velocity_norm": None,
                    "shadow_full_update_norm": None,
                    "shadow_full_velocity_norm": None,
                    "forecast_to_shadow_update_err_abs": None,
                    "forecast_to_shadow_update_err_rel": None,
                    "forecast_to_shadow_update_cos": None,
                    "forecast_vs_reuse_update_err_ratio": None,
                    "forecast_to_shadow_velocity_err_abs": None,
                    "forecast_to_shadow_velocity_err_rel": None,
                    "forecast_to_shadow_velocity_cos": None,
                    "forecast_vs_reuse_velocity_err_ratio": None,
                    "history_updated": True,
                })
                latents = latents_next
            else:
                preds = update_history_forecast_predictions(
                    history_state,
                    step=step,
                    sigma=float(args.update_sigma),
                )
                forecast_tensor = preds.get(str(args.update_mode))
                reuse_tensor = preds.get("reuse")
                if args.forecast_space == "velocity":
                    forecast_velocity = forecast_tensor
                    reuse_velocity = reuse_tensor
                    forecast_update = _scaled(forecast_velocity, step_size)
                    reuse_update = _scaled(reuse_velocity, step_size)
                else:
                    forecast_update = forecast_tensor
                    reuse_update = reuse_tensor
                    if abs(float(step_size)) > 0.0:
                        forecast_velocity = _scaled(forecast_update, 1.0 / float(step_size))
                        reuse_velocity = _scaled(reuse_update, 1.0 / float(step_size))
                    else:
                        forecast_velocity = None
                        reuse_velocity = None
                fallback_reason = None
                if forecast_tensor is None:
                    fallback_reason = "forecast_unavailable"
                elif tuple(forecast_tensor.shape) != tuple(latents.shape):
                    fallback_reason = "forecast_shape_mismatch"

                scheduler_step_index = _ensure_step_index(pipe.scheduler, timestep)
                saved_step_index = getattr(pipe.scheduler, "_step_index", None)
                shadow_update = None
                shadow_velocity = None
                if bool(args.shadow_full_update):
                    noise_shadow = _transformer_output(pipe, latents, timestep, ctx)
                    shadow_next = pipe.scheduler.step(noise_shadow, timestep, latents, return_dict=False)[0]
                    shadow_update = shadow_next.detach().to(torch.float32) - latents.detach().to(torch.float32)
                    shadow_velocity = noise_shadow.detach().to(torch.float32)
                    pipe.scheduler._step_index = saved_step_index

                if fallback_reason is None:
                    assert forecast_update is not None
                    _advance_scheduler_state(pipe.scheduler, timestep, latents)
                    latents_next = (
                        latents.detach().to(torch.float32)
                        + forecast_update.detach().to(device=latents.device, dtype=torch.float32)
                    ).to(dtype=latents.dtype, device=latents.device)
                    direct_used = True
                    latents = latents_next
                else:
                    noise_pred = _transformer_output(pipe, latents, timestep, ctx)
                    pipe.scheduler._step_index = saved_step_index
                    latents_pre = latents
                    latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                    full_update = latents.detach().to(torch.float32) - latents_pre.detach().to(torch.float32)
                    full_velocity = noise_pred.detach().to(torch.float32)
                    history_tensor = full_update if args.forecast_space == "update" else full_velocity
                    history_state = update_history_update_on_full(
                        history_state,
                        tensor=history_tensor,
                        step=step,
                        max_order=2,
                        sigma=float(args.update_sigma),
                        ema_beta=0.2,
                    )
                    direct_used = False

                err_abs = None
                err_rel = None
                err_cos = None
                ratio = None
                vel_err_abs = None
                vel_err_rel = None
                vel_err_cos = None
                vel_ratio = None
                reuse_vel_err_abs = None
                reuse_vel_err_rel = None
                reuse_vel_err_cos = None
                if forecast_update is not None and shadow_update is not None and tuple(forecast_update.shape) == tuple(shadow_update.shape):
                    diff = forecast_update.detach().to(torch.float32) - shadow_update
                    err_abs = float(diff.norm().item())
                    shadow_norm = float(shadow_update.norm().item())
                    err_rel = float(err_abs / (shadow_norm + EPS))
                    err_cos = _cos(forecast_update, shadow_update)
                    if reuse_update is not None and tuple(reuse_update.shape) == tuple(shadow_update.shape):
                        denom = float((reuse_update.detach().to(torch.float32) - shadow_update).norm().item())
                        ratio = float(err_abs / (denom + EPS))
                if forecast_velocity is not None and shadow_velocity is not None and tuple(forecast_velocity.shape) == tuple(shadow_velocity.shape):
                    vel_diff = forecast_velocity.detach().to(torch.float32) - shadow_velocity
                    vel_err_abs = float(vel_diff.norm().item())
                    shadow_vel_norm = float(shadow_velocity.norm().item())
                    vel_err_rel = float(vel_err_abs / (shadow_vel_norm + EPS))
                    vel_err_cos = _cos(forecast_velocity, shadow_velocity)
                    if reuse_velocity is not None and tuple(reuse_velocity.shape) == tuple(shadow_velocity.shape):
                        reuse_vel_diff = reuse_velocity.detach().to(torch.float32) - shadow_velocity
                        reuse_vel_err_abs = float(reuse_vel_diff.norm().item())
                        reuse_vel_err_rel = float(reuse_vel_err_abs / (shadow_vel_norm + EPS))
                        reuse_vel_err_cos = _cos(reuse_velocity, shadow_velocity)
                        vel_ratio = float(vel_err_abs / (reuse_vel_err_abs + EPS))

                row.update({
                    "scheduler_step_index": scheduler_step_index,
                    "direct_update_used": bool(direct_used),
                    "direct_update_available": forecast_update is not None,
                    "direct_update_fallback": fallback_reason is not None,
                    "direct_update_fallback_reason": fallback_reason,
                    "full_update_norm": None,
                    "full_velocity_norm": None,
                    "forecast_update_norm": _norm(forecast_update),
                    "reuse_update_norm": _norm(reuse_update),
                    "forecast_velocity_norm": _norm(forecast_velocity),
                    "reuse_velocity_norm": _norm(reuse_velocity),
                    "shadow_full_update_norm": _norm(shadow_update),
                    "shadow_full_velocity_norm": _norm(shadow_velocity),
                    "forecast_to_shadow_update_err_abs": err_abs,
                    "forecast_to_shadow_update_err_rel": err_rel,
                    "forecast_to_shadow_update_cos": err_cos,
                    "forecast_vs_reuse_update_err_ratio": ratio,
                    "forecast_to_shadow_velocity_err_abs": vel_err_abs,
                    "forecast_to_shadow_velocity_err_rel": vel_err_rel,
                    "forecast_to_shadow_velocity_cos": vel_err_cos,
                    "reuse_to_shadow_velocity_err_abs": reuse_vel_err_abs,
                    "reuse_to_shadow_velocity_err_rel": reuse_vel_err_rel,
                    "reuse_to_shadow_velocity_cos": reuse_vel_err_cos,
                    "forecast_vs_reuse_velocity_err_ratio": vel_ratio,
                    "history_updated": bool(fallback_reason is not None),
                })
            rows.append(row)
    return latents.detach(), rows


def main() -> int:
    args = parse_args()
    mode_name = "SeaCacheDirectUpdate" if args.forecast_space == "update" else "SeaCacheDirectVelocity"
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} "
        f"dtype={args.dtype} mode=SeaCacheDirect{args.forecast_space.capitalize()}_{args.update_mode}",
        flush=True,
    )
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - process_start
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in {model_load_s:.1f}s; "
        f"shard {args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
        f"(global idx {start}..{end - 1})",
        flush=True,
    )

    per_image: List[Dict[str, Any]] = []
    skipped = 0
    H = (args.height // 16) * 16
    W = (args.width // 16) * 16
    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        out_path = args.output_dir / image_filename(global_idx)
        if args.resume and out_path.is_file():
            skipped += 1
            continue
        t0 = time.perf_counter()
        final_latent, rows = _run_prompt(pipe, prompt, global_idx, args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_s = time.perf_counter() - t0
        decode_t0 = time.perf_counter()
        _decode_to_pil(pipe, final_latent, H, W).save(out_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - decode_t0
        n_cached = sum(1 for row in rows if int(row.get("u", 0)) == 1)
        (args.output_dir / f"decisions_{global_idx:05d}.json").write_text(
            json.dumps({
                "prompt_idx": int(global_idx),
                "mode": mode_name,
                "update_mode": str(args.update_mode),
                "update_sigma": float(args.update_sigma),
                "forecast_space": str(args.forecast_space),
                "payload_schedule_dir": str(args.payload_schedule_dir),
                "shadow_full_update": bool(args.shadow_full_update),
                "n_cached": int(n_cached),
                "n_total": int(len(rows)),
                "cached_ratio": float(n_cached / max(1, len(rows))),
                "per_step": rows,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_image.append({
            "idx": int(global_idx),
            "denoise_s": float(denoise_s),
            "decode_s": float(decode_s),
            "n_cached_steps": int(n_cached),
            "n_full_steps": int(len(rows) - n_cached),
            "cached_ratio": float(n_cached / max(1, len(rows))),
        })
        if (local_idx + 1) % 10 == 0 or local_idx == len(shard_prompts) - 1:
            print(
                f"[shard {args.shard_idx}] {local_idx + 1}/{len(shard_prompts)} "
                f"idx={global_idx} {denoise_s + decode_s:.2f}s -> {out_path.name}",
                flush=True,
            )

    if skipped:
        print(f"[shard {args.shard_idx}] resumed: skipped {skipped} existing images", flush=True)
    if per_image:
        write_timing_json(
            args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json",
            per_image=per_image,
            config={
                "cache_mode": f"SeaCacheDirect{args.forecast_space.capitalize()}_{args.update_mode}",
                "mode_raw": "SeaCacheDirectUpdate" if args.forecast_space == "update" else "SeaCacheDirectVelocity",
                "update_mode": str(args.update_mode),
                "update_sigma": float(args.update_sigma),
                "forecast_space": str(args.forecast_space),
                "payload_schedule_dir": str(args.payload_schedule_dir),
                "shadow_full_update": bool(args.shadow_full_update),
                "num_steps": int(args.num_steps),
                "seed": int(args.seed),
                "shard_idx": int(args.shard_idx),
                "shard_count": int(args.shard_count),
            },
            model_load_s=float(model_load_s),
            wallclock_total_s=float(time.perf_counter() - process_start),
            device=str(device),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
