#!/usr/bin/env python3
"""H4 additivity test: multi-step oracle on FLUX.

For each prompt, runs:
  - 1 baseline trajectory (full computation at every step)
  - M oracle trajectories per cache-ratio tier, each caching at a sampled
    SET S of timesteps (rest full). The actual residual-reuse rule at each
    cached step is governed by --cache_mode (same as flux/oracle_runner.py).

Per docs/my_research_plan.md §7 H4 + §10 Exp 3. The hypothesis is

    e_N^(S) ≈ Σ_{k∈S} e_N^(k)

i.e. multi-step cache error ≈ vector sum of single-step cache errors. We
test by measuring z_N^(cache-on-S) here, then comparing to
Σ_{k∈S} (z_N^(k) - z_N^(baseline)) (computed offline in analysis from
existing single-step oracle data).

The harness reuses flux/oracle_runner.install_oracle with the new
cache_steps frozenset interface added there.

Output per prompt:
    output_dir/prompt_XXXXX/
      baseline.pt              z_N latent (bf16, on CPU) — identical to
                               oracle_n100_s42_50/prompt_XXXXX/baseline.pt
                               when run with same (seed, num_steps, model).
      set_r{R}_idx{I:03d}.pt   z_N for cache set with ratio R% (set #I)
      manifest.json            prompt text, seed, per-set cache_steps list,
                               cache_mode, complete flag

Cost per prompt: 1 baseline + (n_ratios * n_sets_per_ratio) cache-set
trajectories × N forwards. With default ratios {10,20,30,40}% and 40 sets
each: 1 + 160 = 161 trajectories. On H100 bf16 ~500-800 s/prompt; n=20
prompts on 4-GPU sharded ≈ 1.5-2 h.

For H4 we typically only need n=20 prompts (n_sets_per_ratio gives the
statistical power within each prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux.oracle_runner import (  # noqa: E402
    CACHE_MODES,
    install_oracle,
    reset_oracle_state,
    _run_one_pipe_call,
    _decode_to_pil,
    _save_latent,
)
from lib.io_utils import read_prompts, split_shard  # noqa: E402


# ----------------------------------------------------------------------------
# Cache set sampling
# ----------------------------------------------------------------------------
def sample_cache_sets(
    num_steps: int,
    ratios_pct: List[int],
    n_sets_per_ratio: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample cache sets across multiple ratio tiers.

    Each cache set is a sorted list of step indices in [1, num_steps-1]
    (k=0 excluded — residual_history is empty at step 0 → forced full).
    Sampling is uniform random without replacement within each set.
    Sets are seeded deterministically by (seed, ratio, idx) so a re-run
    with the same seed produces the same sets.

    Returns a list of records:
        {"ratio_pct": int, "idx": int, "K": int, "cache_steps": List[int]}
    """
    eligible_steps = list(range(1, num_steps))  # 1..N-1
    out: List[Dict[str, Any]] = []
    for r in ratios_pct:
        K = int(round(len(eligible_steps) * r / 100.0))
        K = max(1, min(K, len(eligible_steps)))
        for idx in range(n_sets_per_ratio):
            sub_seed = (
                int(seed) * 1_000_003
                + int(r) * 1_009
                + int(idx)
            ) & 0x7FFFFFFF
            rng = np.random.default_rng(sub_seed)
            sel = rng.choice(eligible_steps, size=K, replace=False)
            steps = sorted(int(s) for s in sel)
            out.append({
                "ratio_pct": int(r),
                "idx": int(idx),
                "K": int(K),
                "cache_steps": steps,
            })
    return out


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-step oracle (H4 additivity test) on FLUX. "
                    "Samples random cache sets at multiple ratios per prompt; "
                    "runs each as a multi-step cache trajectory."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"],
                   default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose manifest.json marks complete.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache",
                   help="Per-step cache rule at cached steps. See "
                        "flux/oracle_runner.py.")
    p.add_argument("--hicache_max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--taylorseer_max_order", type=int, default=1)
    p.add_argument("--ratios_pct", default="10,20,30,40",
                   help="Comma-separated cache-ratio percentages "
                        "(e.g. 10,20,30,40 -> K = 5/10/15/20 cached steps of "
                        "49 eligible, for num_steps=50).")
    p.add_argument("--n_sets_per_ratio", type=int, default=40,
                   help="Number of random cache sets per ratio (default 40).")
    p.add_argument("--set_seed", type=int, default=12345,
                   help="Base seed for cache-set sampling (deterministic per "
                        "(prompt, ratio, idx) given this).")
    p.add_argument("--no_decode", action="store_true",
                   help="Skip per-set PNG decode (saves disk + decode time; "
                        "H4 analysis only needs the .pt latents).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    ratios_pct = [int(x.strip()) for x in str(args.ratios_pct).split(",")
                  if x.strip()]
    if not ratios_pct:
        raise SystemExit("--ratios_pct must contain at least one ratio")

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.",
              flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} "
          f"dtype={args.dtype}", flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in "
          f"{model_load_end - process_start:.1f}s; shard {args.shard_idx}/"
          f"{args.shard_count} has {len(shard_prompts)} prompts "
          f"(global idx {start}..{end - 1})", flush=True)

    N = int(args.num_steps)
    H = (args.height // 16) * 16
    W = (args.width // 16) * 16

    print(f"[shard {args.shard_idx}] N={N} ratios={ratios_pct}% "
          f"sets/ratio={args.n_sets_per_ratio} cache_mode={args.cache_mode}",
          flush=True)

    per_prompt_records = []

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"

        if args.resume and manifest_path.is_file():
            try:
                m = json.loads(manifest_path.read_text())
                if m.get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} "
                          f"already complete, skip", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        prompt_dir.mkdir(parents=True, exist_ok=True)
        per_image_seed = args.seed + global_idx
        t_prompt = time.perf_counter()

        # ---- Cache sets: deterministic per (set_seed, prompt_idx) ---------
        # Per-prompt seed mixed in so different prompts get different sets,
        # but a re-run of the same prompt with same set_seed is identical.
        set_seed_p = int(args.set_seed) ^ (int(per_image_seed) * 17)
        sets = sample_cache_sets(
            num_steps=N,
            ratios_pct=ratios_pct,
            n_sets_per_ratio=int(args.n_sets_per_ratio),
            seed=set_seed_p,
        )

        # ---- Baseline -----------------------------------------------------
        z_base = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
        _save_latent(z_base, prompt_dir / "baseline.pt")
        if not args.no_decode:
            _decode_to_pil(pipe, z_base, H, W).save(prompt_dir / "baseline.png")

        # ---- Cache-set sweep ---------------------------------------------
        per_set_records = []
        for set_rec in sets:
            R = set_rec["ratio_pct"]
            I = set_rec["idx"]
            cache_steps = set_rec["cache_steps"]

            teardown = install_oracle(
                pipe,
                cache_steps=cache_steps,
                num_steps=N,
                cache_mode=str(args.cache_mode),
                hicache_max_order=int(args.hicache_max_order),
                hicache_sigma=float(args.hicache_sigma),
                taylorseer_max_order=int(args.taylorseer_max_order),
            )
            try:
                reset_oracle_state(pipe)
                z_set = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
            finally:
                teardown()

            fname_stem = f"set_r{R:02d}_idx{I:03d}"
            _save_latent(z_set, prompt_dir / f"{fname_stem}.pt")
            if not args.no_decode:
                _decode_to_pil(pipe, z_set, H, W).save(prompt_dir / f"{fname_stem}.png")
            per_set_records.append({
                "ratio_pct": R,
                "idx": I,
                "K": set_rec["K"],
                "cache_steps": cache_steps,
                "file": f"{fname_stem}.pt",
            })

        t_done = time.perf_counter()
        prompt_seconds = float(t_done - t_prompt)

        manifest_data = {
            "prompt_idx": global_idx,
            "prompt": prompt,
            "seed": int(per_image_seed),
            "num_steps": N,
            "ratios_pct": ratios_pct,
            "n_sets_per_ratio": int(args.n_sets_per_ratio),
            "set_seed_p": int(set_seed_p),
            "cache_mode": str(args.cache_mode),
            "hicache_max_order": int(args.hicache_max_order),
            "hicache_sigma": float(args.hicache_sigma),
            "taylorseer_max_order": int(args.taylorseer_max_order),
            "model_id": args.model_id,
            "model_name": args.model_name,
            "dtype": args.dtype,
            "guidance": float(args.guidance),
            "width": W,
            "height": H,
            "sets": per_set_records,
            "files": {
                "baseline_latent": "baseline.pt",
                "baseline_image": "baseline.png" if not args.no_decode else None,
                "set_latents": [r["file"] for r in per_set_records],
            },
            "wall_seconds": prompt_seconds,
            "complete": True,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2,
                                            ensure_ascii=False))

        per_prompt_records.append({
            "global_idx": global_idx,
            "wall_seconds": prompt_seconds,
            "n_sets": len(per_set_records),
        })
        print(f"[shard {args.shard_idx}] prompt {global_idx} done "
              f"{prompt_seconds:.1f}s ({len(per_set_records)} sets) "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = (args.output_dir
                   / f"timing_shard{args.shard_idx}of{args.shard_count}.json")
    device_str = (torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else "cpu")
    timing_data = {
        "experiment": "multistep_oracle",
        "cache_mode": str(args.cache_mode),
        "ratios_pct": ratios_pct,
        "n_sets_per_ratio": int(args.n_sets_per_ratio),
        "num_steps": N,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "base_seed": int(args.seed),
        "set_seed": int(args.set_seed),
        "model_id": args.model_id,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "device": device_str,
        "model_load_s": float(model_load_end - process_start),
        "wallclock_total_s": float(process_end - process_start),
        "n_prompts_in_shard": len(shard_prompts),
        "per_prompt": per_prompt_records,
    }
    timing_path.write_text(json.dumps(timing_data, indent=2,
                                       ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}  "
          f"(total wall {process_end - process_start:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
