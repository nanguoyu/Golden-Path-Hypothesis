#!/usr/bin/env python3
"""Step 3 of the gap-aware extension (docs/research_plan_extension.md §8).

Interval oracle (Oracle B): real contiguous-block cache cost.

For each interval [a, b]:
  - step a            : full computation (last refresh);
  - steps a+1..b-1    : cache / reuse with the chosen predictor;
  - steps b..N-1      : full computation again.

Measures the real interval error

    E_real(a, b) = z_N^{cache interval(a,b)} - z_N^{full}

and the per-step latent drift  ||e_k|| = ||z~_k - z_k||  inside and after
the interval. This is the data H7 needs to compare against the additive
proxies C_single = sum R_k (Phase 1 data) and C_gap = sum R_{a->k}
(Oracle A / Step 2 data), and to expose state-drift compounding.

It reuses `flux/oracle_runner.py::install_oracle` (which already accepts a
`cache_steps` set); a contiguous interval is just cache_steps = range(a+1, b).
The 7 locked cache modes are untouched.

Output per prompt:
    <output_dir>/prompt_XXXXX/
      baseline.pt / baseline.png            z_N^{full} (once per prompt)
      interval_aAA_bBB.pt / .png            z_N^{cache} for interval [a,b]
      interval_aAA_bBB.json                 per-step ||e_k||, final L2, etc.
      manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from flux.oracle_runner import (  # noqa: E402
    CACHE_MODES,
    _decode_to_pil,
    _save_latent,
    install_oracle,
    reset_oracle_state,
)


# ----------------------------------------------------------------------------
# Interval parsing
# ----------------------------------------------------------------------------
def parse_intervals(spec: str, num_steps: int) -> List[Tuple[int, int]]:
    """Parse '--intervals a:b,a:b,...' into a sorted, de-duplicated list.

    [a, b] means: full at a, cache a+1..b-1, full from b on. Requires
    0 <= a, a+1 < b <= num_steps (so the cache set range(a+1, b) is
    non-empty). b == num_steps is allowed (the interval runs to the end,
    caching the last step k = N-1 -- a legitimate diagnostic measurement).
    """
    out: List[Tuple[int, int]] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"bad interval token {tok!r}, expected 'a:b'")
        a_s, b_s = tok.split(":", 1)
        a, b = int(a_s), int(b_s)
        if not (0 <= a and a + 1 < b <= int(num_steps)):
            raise ValueError(
                f"interval [{a},{b}] invalid for num_steps={num_steps} "
                f"(need 0<=a, a+1<b<=N)"
            )
        out.append((a, b))
    if not out:
        raise ValueError("no intervals parsed from --intervals")
    return sorted(set(out))


# ----------------------------------------------------------------------------
# Pipe call with per-step latent trace
# ----------------------------------------------------------------------------
def _run_pipe_trace(pipe, prompt: str, seed: int, args
                    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """One pipe(...) call returning (z_N, [z_1, ..., z_N]).

    The trace is captured via `callback_on_step_end`: entry i is the packed
    latent after solver step i (= z_{i+1} in the plan's z_0..z_N indexing),
    detached + cloned to CPU bf16. trace[-1] == z_N.
    """
    trace: List[torch.Tensor] = []

    def _cb(_pipe, _step_index, _timestep, callback_kwargs):
        lat = callback_kwargs["latents"]
        trace.append(lat.detach().to("cpu", dtype=torch.bfloat16).clone())
        return callback_kwargs

    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        num_inference_steps=int(args.num_steps),
        guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent",
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result.images, trace


def _l2(x: torch.Tensor, y: torch.Tensor) -> float:
    """L2 norm of (x - y) in fp32."""
    return float((x.float() - y.float()).norm().item())


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interval oracle (Oracle B): real contiguous-block cache cost."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--intervals", required=True,
                   help="comma-separated 'a:b' intervals; full at a, cache "
                        "a+1..b-1, full from b. e.g. '3:18,8:30,26:50'.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache",
                   help="predictor used on cached steps inside the interval.")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="base seed; per-prompt seed = seed + global_idx.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose manifest.json marks complete.")
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
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded in {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts "
          f"(global {start}..{end - 1}); cache_mode={args.cache_mode}; "
          f"{len(intervals)} intervals", flush=True)

    H = (args.height // 16) * 16
    W = (args.width // 16) * 16
    per_prompt_records = []

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"

        if args.resume and manifest_path.is_file():
            try:
                if json.loads(manifest_path.read_text()).get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} complete, skip",
                          flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        prompt_dir.mkdir(parents=True, exist_ok=True)
        per_image_seed = args.seed + global_idx
        t_p = time.perf_counter()

        # ---- reference: full trajectory (no oracle installed) --------------
        z_full, ref_trace = _run_pipe_trace(pipe, prompt, per_image_seed, args)
        if len(ref_trace) != N:
            raise RuntimeError(
                f"prompt {global_idx}: ref trace len {len(ref_trace)}, expected {N}"
            )
        _save_latent(z_full, prompt_dir / "baseline.pt")
        _decode_to_pil(pipe, z_full, H, W).save(prompt_dir / "baseline.png")

        # ---- interval sweep ------------------------------------------------
        for (a, b) in intervals:
            cache_steps = list(range(a + 1, b))
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
                z_cache, cache_trace = _run_pipe_trace(pipe, prompt, per_image_seed, args)
            finally:
                teardown()
            if len(cache_trace) != N:
                raise RuntimeError(
                    f"prompt {global_idx} interval [{a},{b}]: cache trace len "
                    f"{len(cache_trace)}, expected {N}"
                )

            # per-step latent drift; per_step_e[i] = ||z~_{i+1} - z_{i+1}||
            per_step_e = [_l2(cache_trace[i], ref_trace[i]) for i in range(N)]
            tag = f"interval_a{a:02d}_b{b:02d}"
            _save_latent(z_cache, prompt_dir / f"{tag}.pt")
            _decode_to_pil(pipe, z_cache, H, W).save(prompt_dir / f"{tag}.png")
            (prompt_dir / f"{tag}.json").write_text(json.dumps({
                "experiment": "interval_oracle",
                "prompt_idx": global_idx,
                "interval": [a, b],
                "gap": b - a,
                "cache_mode": str(args.cache_mode),
                "cache_steps": cache_steps,
                "n_cached": len(cache_steps),
                "num_steps": N,
                "per_step_e": per_step_e,
                "final_latent_l2": per_step_e[-1],
                "complete": True,
            }, ensure_ascii=False))

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "interval_oracle",
            "prompt_idx": global_idx,
            "prompt": prompt,
            "seed": int(per_image_seed),
            "num_steps": N,
            "cache_mode": str(args.cache_mode),
            "intervals": [[a, b] for (a, b) in intervals],
            "hicache_max_order": int(args.hicache_max_order),
            "hicache_sigma": float(args.hicache_sigma),
            "taylorseer_max_order": int(args.taylorseer_max_order),
            "wall_seconds": wall,
            "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt_records.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing_path.write_text(json.dumps({
        "experiment": "interval_oracle",
        "cache_mode": str(args.cache_mode),
        "num_steps": N,
        "n_intervals": len(intervals),
        "intervals": [[a, b] for (a, b) in intervals],
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "base_seed": int(args.seed),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": device_str,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts),
        "per_prompt": per_prompt_records,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path} "
          f"(total wall {process_end - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
