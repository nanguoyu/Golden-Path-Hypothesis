"""Phase 4 — fixed-schedule image generator on FLUX.

Generates ONE image per prompt with an explicit, fixed cache-set (which
denoising steps to skip) and a chosen predictor (`cache_mode`). Used for the
tail-load and oracle schedules in the Phase 4 image-space confirmation
experiment.

The 7 locked cache modes in `flux/runner.py` only support their own gates
(IntervalGate / content-threshold); they cannot run an arbitrary explicit
schedule. This file reuses `flux/oracle_runner.py::install_oracle` (which
already accepts a `cache_steps` set) and is a research file — the locked
implementations are untouched.

Output: `img_<global_idx>.png` per prompt (same naming as `runner.py`), so
`evaluation/eval_metrics.py` can compare it against the no-cache baseline.

Native cache methods (SeaCache / HiCache / TaylorSeer with their own gates)
use `flux/runner.py` instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from lib.io_utils import image_filename, read_prompts, split_shard  # noqa: E402
from flux.oracle_runner import (  # noqa: E402
    CACHE_MODES,
    install_oracle,
    reset_oracle_state,
    _run_one_pipe_call,
    _decode_to_pil,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4 fixed-schedule image generator on FLUX.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--cache_steps", required=True,
                   help="comma-separated denoising step indices to cache, "
                        "e.g. '27,28,...,49'. Empty string = cache nothing.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache",
                   help="predictor used on cached steps (the cache mechanism).")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="base seed; per-prompt = seed + global_idx (matches "
                        "lib.io_utils.seed_for, so baseline/native runs pair up).")
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
                   help="cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="skip prompts whose img_<idx>.png already exists.")
    p.add_argument("--hicache_max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--taylorseer_max_order", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}

    cache_steps = sorted({int(x) for x in str(args.cache_steps).split(",")
                          if x.strip() != ""})

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.",
              flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}",
          flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[{datetime.now():%H:%M:%S}] Loaded in {time.perf_counter() - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} "
          f"prompts (global {start}..{end - 1}); cache_mode={args.cache_mode}, "
          f"{len(cache_steps)} cached steps", flush=True)

    H = (args.height // 16) * 16
    W = (args.width // 16) * 16

    # The cache-set is fixed across all prompts -> install the patched forward
    # once, reset only the per-trajectory state before each pipe() call.
    teardown = install_oracle(
        pipe,
        cache_steps=cache_steps,
        num_steps=int(args.num_steps),
        cache_mode=str(args.cache_mode),
        hicache_max_order=int(args.hicache_max_order),
        hicache_sigma=float(args.hicache_sigma),
        taylorseer_max_order=int(args.taylorseer_max_order),
    )

    per_image_records: list[dict[str, float | int]] = []
    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            out_path = args.output_dir / image_filename(global_idx)
            if args.resume and out_path.is_file():
                print(f"[shard {args.shard_idx}] idx {global_idx} exists, skip",
                      flush=True)
                continue
            t_p = time.perf_counter()
            reset_oracle_state(pipe)
            z = _run_one_pipe_call(pipe, prompt, args.seed + global_idx, args)
            _decode_to_pil(pipe, z, H, W).save(out_path)
            dt = time.perf_counter() - t_p
            per_image_records.append({
                "idx": int(global_idx),
                "denoise_s": float(dt),
                "decode_s": 0.0,
                "n_cached": int(len(cache_steps)),
                "cached_ratio": float(len(cache_steps) / max(int(args.num_steps), 1)),
            })
            print(f"[shard {args.shard_idx}] idx {global_idx} done {dt:.1f}s",
                  flush=True)
    finally:
        teardown()

    timing = {
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "cache_mode": args.cache_mode,
        "num_steps": args.num_steps,
        "n_cached_steps": len(cache_steps),
        "cache_steps": cache_steps,
        "n_images": len(per_image_records),
        "per_image": per_image_records,
        "mean_seconds_per_image": (
            sum(float(r["denoise_s"]) for r in per_image_records) / len(per_image_records)
            if per_image_records else None),
    }
    tpath = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    tpath.write_text(json.dumps(timing, indent=2))
    print(f"[shard {args.shard_idx}] wrote {tpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
