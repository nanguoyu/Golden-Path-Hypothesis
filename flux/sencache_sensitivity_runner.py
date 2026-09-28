#!/usr/bin/env python3
"""Collect FLUX SenCache sensitivity calibration rows.

This is compute-heavy and must run on compute nodes.  It estimates timestep-wise
directional sensitivity norms from full FLUX trajectories:

  J_x(t_i) ~= ||f(x_ref, t_i) - f(x_i, t_i)|| / ||x_ref - x_i||
  J_t(t_i) ~= ||f(x_i, t_ref) - f(x_i, t_i)|| / |t_ref - t_i|

where ``ref`` is the adjacent full-trajectory state/timestep.  A separate merge
script freezes robust per-step aggregates into a SenCache npz table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.a_k_probe import _encode_and_prepare  # noqa: E402
from flux.trajectory_deviation_runner import (  # noqa: E402
    _git_commit,
    _norm,
    _run_full_trace,
    install_trajectory_deviation,
    reset_td_state,
)
from lib.io_utils import read_prompts, seed_for, split_shard  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect FLUX SenCache sensitivity rows.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _model_output(pipe, ctx: Dict[str, Any], latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    tr = pipe.transformer
    tr._td_run_kind = "full"
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


def _rows_for_prompt(pipe, prompt: str, prompt_id: int, seed: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    reset_td_state(pipe)
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    full_trace = _run_full_trace(pipe, ctx, int(args.num_steps))
    timesteps = ctx["timesteps"]
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i, timestep in enumerate(timesteps):
            if len(timesteps) <= 1:
                ref = i
            elif i < len(timesteps) - 1:
                ref = i + 1
            else:
                ref = i - 1
            device = ctx["latents_init"].device
            x_i = full_trace["z_pre"][i].to(device, dtype=torch.float32)
            x_ref = full_trace["z_pre"][ref].to(device, dtype=torch.float32)
            o_i = full_trace["outputs"][i].to(device, dtype=torch.float32)
            t_i = timesteps[i]
            t_ref = timesteps[ref]
            o_x = _model_output(pipe, ctx, x_ref.to(dtype=ctx["latents_init"].dtype), t_i).detach().to(torch.float32)
            o_t = _model_output(pipe, ctx, x_i.to(dtype=ctx["latents_init"].dtype), t_ref).detach().to(torch.float32)
            dx = _norm(x_ref - x_i)
            dt = abs(float(t_ref.detach().cpu().item()) - float(t_i.detach().cpu().item()))
            j_x = (_norm(o_x - o_i) / dx) if dx > 0.0 else None
            j_t = (_norm(o_t - o_i) / dt) if dt > 0.0 else None
            rows.append({
                "prompt_id": int(prompt_id),
                "seed": int(seed),
                "step_index": int(i),
                "timestep": float(t_i.detach().cpu().item()),
                "reference_step_index": int(ref),
                "reference_timestep": float(t_ref.detach().cpu().item()),
                "latent_shape": json.dumps(list(x_i.shape)),
                "latent_numel": int(x_i.numel()),
                "delta_latent_norm": float(dx),
                "delta_t_abs": float(dt),
                "J_x_directional": j_x,
                "J_t_directional": j_t,
            })
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = args.output_dir / f"sencache_sensitivity_rows_shard{args.shard_idx}of{args.shard_count}.csv"
    if args.resume and shard_csv.is_file():
        print(f"[shard {args.shard_idx}] {shard_csv} exists, skip", flush=True)
        return 0

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice")
        return 0

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} shard={args.shard_idx}/{args.shard_count}", flush=True)
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    teardown = install_trajectory_deviation(
        pipe,
        mode="SeaCache",
        threshold=0.0,
        num_steps=int(args.num_steps),
        first_enhance=1,
    )
    rows: List[Dict[str, Any]] = []
    git_sha, git_dirty = _git_commit()
    try:
        for local_idx, prompt in enumerate(shard_prompts):
            prompt_id = start + local_idx
            t0 = time.perf_counter()
            rows.extend(_rows_for_prompt(pipe, prompt, prompt_id, seed_for(args.seed, prompt_id), args))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[shard {args.shard_idx}] prompt={prompt_id} rows={args.num_steps} dt={time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        teardown()
    for row in rows:
        row["git_commit"] = git_sha
        row["git_dirty"] = git_dirty
    _write_csv(shard_csv, rows)
    print(f"[shard {args.shard_idx}] wrote {shard_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
