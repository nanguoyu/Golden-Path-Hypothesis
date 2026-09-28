#!/usr/bin/env python3
"""Collect Qwen-Image DPCache path costs from full true-CFG trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.dpcache import calibration_cost_tensor
from lib.io_utils import read_prompts, seed_for, split_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--sentinel_alpha", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _block_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            "Qwen DPCache calibration expected (encoder, hidden) block output"
        )
    encoder, hidden = output
    if not isinstance(encoder, torch.Tensor) or not isinstance(hidden, torch.Tensor):
        raise TypeError("Qwen DPCache block outputs must be tensors")
    return encoder, hidden


def main() -> int:
    args = parse_args()
    if (
        args.resume
        and args.out.is_file()
        and args.out.with_suffix(".json").is_file()
    ):
        print(f"[qwen-dpcache-calibrate] preserve {args.out}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen DPCache calibration requires CUDA")
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen DPCache calibration requires true_cfg_scale > 1")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to("cuda")
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        encoder, hidden = _block_output(output)
        captured.append((encoder.detach(), hidden.detach()))
        return output

    handle = pipe.transformer.transformer_blocks[-1].register_forward_hook(
        capture,
        with_kwargs=True,
    )
    sums = np.zeros(
        (args.num_steps, args.num_steps + 1, args.num_steps + 1),
        dtype=np.float64,
    )
    counts = np.zeros_like(sums, dtype=np.int64)
    prompt_rows: list[dict[str, int]] = []
    started = time.perf_counter()
    try:
        for local_idx, prompt in enumerate(selected):
            prompt_idx = start + local_idx
            prompt_seed = seed_for(args.seed, prompt_idx)
            captured.clear()
            generator = torch.Generator(device="cuda").manual_seed(prompt_seed)
            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    true_cfg_scale=args.true_cfg_scale,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    generator=generator,
                    output_type="latent",
                    return_dict=True,
                )
            expected = 2 * args.num_steps
            if len(captured) != expected:
                raise RuntimeError(
                    f"Qwen DPCache captured {len(captured)} block outputs; "
                    f"expected {expected}"
                )
            for branch_offset in (0, 1):
                trajectory = captured[branch_offset::2]
                for component in (0, 1):
                    cost = calibration_cost_tensor(
                        [row[component] for row in trajectory],
                        order=2,
                        sentinel_alpha=args.sentinel_alpha,
                    )
                    valid = np.isfinite(cost)
                    sums[valid] += cost[valid]
                    counts[valid] += 1
            prompt_rows.append({"prompt_idx": prompt_idx, "seed": prompt_seed})
            torch.cuda.synchronize()
            del trajectory
            captured.clear()
            torch.cuda.empty_cache()
            print(
                f"[qwen-dpcache-calibrate] shard={args.shard_idx} "
                f"{local_idx + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        handle.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=sums,
        cost_counts=counts,
        num_steps=np.asarray(args.num_steps, dtype=np.int64),
        order=np.asarray(2, dtype=np.int64),
        sentinel_alpha=np.asarray(args.sentinel_alpha, dtype=np.float64),
        branch_component_count=np.asarray(4, dtype=np.int64),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "qwen-image-dpcache-cost-shard-v1",
                "prompt_file": str(args.prompt_file),
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "prompts": prompt_rows,
                "cost_components": [
                    "cond_encoder",
                    "cond_hidden",
                    "uncond_encoder",
                    "uncond_hidden",
                ],
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
