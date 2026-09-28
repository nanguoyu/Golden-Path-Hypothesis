#!/usr/bin/env python3
"""Experiment 4: marginal cache-vs-full fork (Target A).

docs/research_plan_after_extension.md §3 Experiment 4 / §2 Level 3. The
deployable gate's real target is the marginal harm of choosing cache over
full at step n, given the current (drifted) trajectory state:

    Delta_N^{C-F}(n) = z_N^{cache at n, then full} - z_N^{full at n, then full}

Target A: after the fork both branches run full computation to N.

A base cache schedule (`--policy`) defines the trajectory up to the fork.
At fork step n:
    cache_F = { k in policy : k < n }            (step n full, full after)
    cache_C = cache_F + { n }                    (step n cache, full after)
Both branches are zero-order seacache; nothing is cached at steps > n.

Output per prompt:
  prompt_XXXXX/baseline.pt              z_N^full (reference)
  prompt_XXXXX/fork_nNN.pt              {n, z_N_F, z_N_C} (bf16, CPU)
  manifest.json
The marginal Delta_N is ||z_N_C - z_N_F||, computed by analysis/exp4_marginal.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from flux.oracle_runner import (  # noqa: E402
    CACHE_MODES, _run_one_pipe_call, _save_latent,
    install_oracle, reset_oracle_state,
)


def _int_list(spec: str) -> List[int]:
    return sorted({int(x) for x in str(spec).split(",") if x.strip()})


def _run_cache_set(pipe, prompt, seed, args, cache_steps) -> torch.Tensor:
    """One pipe call caching exactly `cache_steps`; returns z_N."""
    if cache_steps:
        teardown = install_oracle(
            pipe, cache_steps=list(cache_steps), num_steps=int(args.num_steps),
            cache_mode=str(args.cache_mode))
        try:
            reset_oracle_state(pipe)
            z = _run_one_pipe_call(pipe, prompt, seed, args)
        finally:
            teardown()
    else:
        z = _run_one_pipe_call(pipe, prompt, seed, args)
    return z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 4: marginal cache-vs-full fork.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--policy", required=True,
                   help="comma-separated base-policy cached steps.")
    p.add_argument("--forks", required=True,
                   help="comma-separated fork step indices n.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    N = int(args.num_steps)
    policy = _int_list(args.policy)
    forks = _int_list(args.forks)
    for n in forks:
        if not (1 <= n <= N - 1):
            raise ValueError(f"fork step {n} out of range 1..{N-1}")

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}] empty slice, exiting.", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts; "
          f"{len(forks)} fork points; policy |{len(policy)}|", flush=True)

    per_prompt = []
    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"
        if args.resume and manifest_path.is_file():
            try:
                if json.loads(manifest_path.read_text()).get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} done, skip", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        prompt_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + global_idx
        t_p = time.perf_counter()

        z_full = _run_cache_set(pipe, prompt, seed, args, [])
        _save_latent(z_full, prompt_dir / "baseline.pt")

        for n in forks:
            cache_F = [k for k in policy if k < n]
            cache_C = cache_F + [n]
            z_F = _run_cache_set(pipe, prompt, seed, args, cache_F)
            z_C = _run_cache_set(pipe, prompt, seed, args, cache_C)
            torch.save({"n": n, "num_steps": N,
                        "n_policy_before": len(cache_F),
                        "z_N_F": z_F.detach().to("cpu", torch.bfloat16),
                        "z_N_C": z_C.detach().to("cpu", torch.bfloat16)},
                       prompt_dir / f"fork_n{n:02d}.pt")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "marginal_fork", "prompt_idx": global_idx,
            "prompt": prompt, "seed": int(seed), "num_steps": N,
            "policy": policy, "forks": forks, "cache_mode": str(args.cache_mode),
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "marginal_fork", "num_steps": N, "policy": policy,
        "forks": forks, "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count), "base_seed": int(args.seed),
        "model_id": args.model_id,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
