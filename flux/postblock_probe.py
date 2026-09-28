#!/usr/bin/env python3
"""Experiment 1: post-block propagation / full-step healing test.

docs/research_plan_after_extension.md §3 Experiment 1. Question: after a
cached block ends at step b, does the subsequent full computation correct,
keep, or amplify the accumulated latent error?

For an interval [a,b] (full at a, cache a+1..b-1, full from b) with
block-end error e_b = z~_b - z_b, the empirical post-block gain of a
perturbation eta injected at step b and propagated full to N is

    G_post(b, eta) = || T_{b:N}(z_b + eta) - z_N^full || / (||eta|| + eps)

where T_{b:N} is the full-computation map from step b. T_{b:N}(z_b) is the
reference final z_N^full, so only the perturbed propagation is re-run.

This version compares two directions at the SAME norm ||e_b||:
  - cache-induced direction  eta = e_b   (the real block-end error)
  - random direction         eta = randn, rescaled to ||e_b||
A direction-specific gap between the two means the post-block amplification
is not merely a function of b. (Clean-defect and leading-sensitive
directions from the plan are left to a later version.)

Output per prompt: prompt_XXXXX/postblock_aAA_bBB.json
  {a, b, norm_e_b, gain_cache, gain_random,
   per_step_err_cache: [||e|| for k=b..N], per_step_err_random: [...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from flux.oracle_runner import install_oracle, reset_oracle_state  # noqa: E402


def _pipe_kwargs(prompt: str, seed: int, args) -> dict:
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(seed))
    return dict(
        prompt=prompt,
        num_inference_steps=int(args.num_steps),
        guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent",
    )


def run_trace(pipe, prompt: str, seed: int, args,
              inject_at: Optional[int] = None,
              inject_latent: Optional[torch.Tensor] = None
              ) -> List[torch.Tensor]:
    """Run pipe; return [latent after step 0, ..., after step N-1].

    If inject_at is given, the callback overwrites the latent right after
    solver step `inject_at` with `inject_latent` (so step inject_at+1 sees
    it). Traces are kept on-device.
    """
    trace: List[torch.Tensor] = []

    def _cb(_pipe, step_index, _t, kw):
        if inject_at is not None and step_index == inject_at:
            kw["latents"] = inject_latent.to(kw["latents"].dtype)
        trace.append(kw["latents"].detach().clone())
        return kw

    kw = _pipe_kwargs(prompt, seed, args)
    kw["callback_on_step_end"] = _cb
    kw["callback_on_step_end_tensor_inputs"] = ["latents"]
    pipe(**kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return trace


def parse_intervals(spec: str, num_steps: int) -> List[Tuple[int, int]]:
    """Non-tail intervals only (b < N): a tail block has no post-block region."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        a_s, b_s = tok.split(":", 1)
        a, b = int(a_s), int(b_s)
        if not (0 <= a and a + 1 < b <= int(num_steps)):
            raise ValueError(f"interval [{a},{b}] invalid for N={num_steps}")
        if b < int(num_steps):
            out.append((a, b))
    if not out:
        raise ValueError("no non-tail intervals parsed")
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 1: post-block propagation probe.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--intervals", required=True)
    p.add_argument("--cache_mode", default="seacache")
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
    p.add_argument("--hicache_max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--taylorseer_max_order", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    N = int(args.num_steps)
    intervals = parse_intervals(args.intervals, N)

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
          f"{len(intervals)} non-tail intervals", flush=True)

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

        # reference trajectory: ref[i] = latent after solver step i
        ref = run_trace(pipe, prompt, seed, args)
        z_N_full = ref[-1]

        for (a, b) in intervals:
            # cached run [a,b] -> block-end error e_b = z~_b - z_b
            teardown = install_oracle(
                pipe, cache_steps=list(range(a + 1, b)), num_steps=N,
                cache_mode=str(args.cache_mode),
                hicache_max_order=int(args.hicache_max_order),
                hicache_sigma=float(args.hicache_sigma),
                taylorseer_max_order=int(args.taylorseer_max_order))
            try:
                reset_oracle_state(pipe)
                cached = run_trace(pipe, prompt, seed, args)
            finally:
                teardown()
            # latent at boundary b = input to step b = after step b-1
            z_b = ref[b - 1].float()
            e_b = cached[b - 1].float() - z_b
            norm_e_b = float(e_b.norm())

            # direction set: cache-induced + random (same norm)
            gen = torch.Generator(device=device).manual_seed(seed * 100003 + a * 317 + b)
            rnd = torch.randn(e_b.shape, generator=gen, device=device, dtype=torch.float32)
            rnd = rnd * (norm_e_b / (float(rnd.norm()) + 1e-12))
            dirs = {"cache": e_b, "random": rnd}

            rec = {"experiment": "postblock_probe", "prompt_idx": global_idx,
                   "a": a, "b": b, "num_steps": N, "norm_e_b": norm_e_b}
            for name, eta in dirs.items():
                inj = (z_b + eta).to(ref[0].dtype)
                prop = run_trace(pipe, prompt, seed, args,
                                 inject_at=b - 1, inject_latent=inj)
                # per-step error from boundary b..N vs reference
                per_err = [float((prop[i].float() - ref[i].float()).norm())
                           for i in range(b - 1, N)]
                gain = per_err[-1] / (norm_e_b + 1e-12)
                rec[f"gain_{name}"] = gain
                rec[f"per_step_err_{name}"] = per_err
            rec["complete"] = True
            (prompt_dir / f"postblock_a{a:02d}_b{b:02d}.json").write_text(
                json.dumps(rec, ensure_ascii=False))

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "postblock_probe", "prompt_idx": global_idx,
            "prompt": prompt, "seed": int(seed), "num_steps": N,
            "intervals": [[a, b] for (a, b) in intervals],
            "cache_mode": str(args.cache_mode),
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "postblock_probe", "num_steps": N,
        "shard_idx": int(args.shard_idx), "shard_count": int(args.shard_count),
        "base_seed": int(args.seed), "model_id": args.model_id,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
