#!/usr/bin/env python3
"""Collect DPCache endpoint costs from full FLUX calibration trajectories."""

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

from flux.oracle_runner import _run_one_pipe_call
from lib.dpcache import calibration_cost_tensor
from lib.io_utils import read_prompts, seed_for, split_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_name", choices=("flux-dev",), default="flux-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--sentinel_alpha", type=float, default=0.8)
    return parser.parse_args()


def _endpoint(output: Any) -> torch.Tensor:
    if isinstance(output, tuple) and len(output) == 2:
        return output[1]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"unsupported FLUX single-block output: {type(output).__name__}")


def main() -> int:
    args = parse_args()
    if args.num_steps != 50:
        raise SystemExit("DPCache screening calibration is frozen to 50 steps")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    last_single = pipe.transformer.single_transformer_blocks[-1]
    endpoints: list[torch.Tensor] = []

    def capture(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        endpoints.append(_endpoint(output).detach())
        return output

    handle = last_single.register_forward_hook(capture, with_kwargs=True)
    cost_sum = np.zeros(
        (args.num_steps, args.num_steps + 1, args.num_steps + 1),
        dtype=np.float64,
    )
    cost_count = np.zeros_like(cost_sum, dtype=np.int64)
    rows = []
    started = time.perf_counter()
    try:
        for local_index, prompt in enumerate(selected):
            global_index = start + local_index
            endpoints.clear()
            _run_one_pipe_call(
                pipe,
                prompt,
                seed_for(args.seed, global_index),
                args,
            )
            if len(endpoints) != args.num_steps:
                raise RuntimeError(
                    f"DPCache collected {len(endpoints)} endpoints; "
                    f"expected {args.num_steps}"
                )
            costs = calibration_cost_tensor(
                endpoints,
                order=2,
                sentinel_alpha=args.sentinel_alpha,
            )
            valid = np.isfinite(costs)
            cost_sum[valid] += costs[valid]
            cost_count[valid] += 1
            rows.append(
                {
                    "prompt_idx": global_index,
                    "seed": seed_for(args.seed, global_index),
                }
            )
            print(
                f"[dpcache-calibration] shard={args.shard_idx} "
                f"{local_index + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        handle.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=cost_sum,
        cost_counts=cost_count,
        num_steps=np.asarray(args.num_steps, dtype=np.int64),
        order=np.asarray(2, dtype=np.int64),
        sentinel_alpha=np.asarray(args.sentinel_alpha, dtype=np.float64),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "flux-dpcache-cost-shard-v1",
                "prompt_file": str(args.prompt_file),
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "prompts": rows,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
