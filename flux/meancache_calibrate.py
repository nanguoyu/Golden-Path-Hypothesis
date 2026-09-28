#!/usr/bin/env python3
"""Collect MeanCache edge costs from full FLUX trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.oracle_runner import _run_one_pipe_call
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
    parser.add_argument("--max_edge_gap", type=int, default=15)
    parser.add_argument("--jvp_spans", default="2,3,4,5")
    return parser.parse_args()


def _sample(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output if isinstance(output, torch.Tensor) else output.sample


def _trajectory_costs(
    latents: list[torch.Tensor],
    velocities: list[torch.Tensor],
    sigmas: torch.Tensor,
    spans: tuple[int, ...],
    *,
    max_edge_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    steps = len(velocities)
    sums = np.zeros((len(spans), steps + 1, steps + 1), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for source in range(steps):
        sums[:, source, source + 1] = 0.0
        counts[:, source, source + 1] = 1
        for span_index, span in enumerate(spans):
            reference = source - span
            if reference < 0:
                continue
            sigma_r = sigmas[reference].to(torch.float32)
            sigma_t = sigmas[source].to(torch.float32)
            rt = sigma_t - sigma_r
            if float(rt.abs().item()) < 1e-12:
                continue
            z_r = latents[reference].to(torch.float32)
            z_t = latents[source].to(torch.float32)
            v_r = velocities[reference].to(torch.float32)
            jvp = (z_t - z_r - rt * v_r) / (rt * rt)
            for destination in range(
                source + 1,
                min(steps, source + int(max_edge_gap)) + 1,
            ):
                ts = sigmas[destination].to(torch.float32) - sigma_t
                true_average = (
                    latents[destination].to(torch.float32) - z_t
                ) / ts
                predicted_average = velocities[source].to(torch.float32) + ts * jvp
                cost = (true_average - predicted_average).abs().mean().item()
                sums[span_index, source, destination] += float(cost)
                counts[span_index, source, destination] += 1
    return sums, counts


def main() -> int:
    args = parse_args()
    spans = tuple(int(value) for value in args.jvp_spans.split(",") if value)
    if not spans or min(spans) < 2:
        raise SystemExit("MeanCache calibration expects JVP spans >= 2")
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
    transformer = pipe.transformer
    original = transformer.forward
    trajectory_inputs: list[torch.Tensor] = []
    trajectory_velocities: list[torch.Tensor] = []

    def wrapped(module: Any, *forward_args: Any, **forward_kwargs: Any):
        latent = forward_kwargs.get(
            "hidden_states",
            forward_args[0] if forward_args else None,
        )
        output = original(*forward_args, **forward_kwargs)
        trajectory_inputs.append(latent.detach())
        trajectory_velocities.append(_sample(output).detach())
        return output

    had_instance_forward = "forward" in transformer.__dict__
    transformer.forward = types.MethodType(wrapped, transformer)
    total_sums = np.zeros((len(spans), args.num_steps + 1, args.num_steps + 1))
    total_counts = np.zeros_like(total_sums, dtype=np.int64)
    rows = []
    started = time.perf_counter()
    try:
        for local_index, prompt in enumerate(selected):
            global_index = start + local_index
            trajectory_inputs.clear()
            trajectory_velocities.clear()
            final = _run_one_pipe_call(
                pipe,
                prompt,
                seed_for(args.seed, global_index),
                args,
            ).detach()
            if len(trajectory_inputs) != args.num_steps:
                raise RuntimeError(
                    f"MeanCache collected {len(trajectory_inputs)} steps, "
                    f"expected {args.num_steps}"
                )
            trajectory = [*trajectory_inputs, final]
            sigmas = torch.as_tensor(
                pipe.scheduler.sigmas,
                device=trajectory[0].device,
                dtype=torch.float32,
            )
            sums, counts = _trajectory_costs(
                trajectory,
                trajectory_velocities,
                sigmas,
                spans,
                max_edge_gap=args.max_edge_gap,
            )
            total_sums += sums
            total_counts += counts
            rows.append(
                {
                    "prompt_idx": global_index,
                    "seed": seed_for(args.seed, global_index),
                }
            )
            print(f"[meancache-calibration] prompt={global_index}", flush=True)
    finally:
        if had_instance_forward:
            transformer.forward = original
        else:
            delattr(transformer, "forward")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=total_sums,
        cost_counts=total_counts,
        jvp_spans=np.asarray(spans, dtype=np.int64),
        num_steps=np.asarray(args.num_steps, dtype=np.int64),
        max_edge_gap=np.asarray(args.max_edge_gap, dtype=np.int64),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "flux-meancache-cost-shard-v1",
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
