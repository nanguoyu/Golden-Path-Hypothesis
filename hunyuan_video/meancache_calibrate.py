#!/usr/bin/env python3
"""Collect MeanCache edge costs from full HunyuanVideo trajectories."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hunyuan_video.backend import load_official_sampler, prediction_kwargs
from hunyuan_video.config import RunSpec, load_protocol
from lib.io_utils import read_prompts, seed_for, split_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_base", type=Path, required=True)
    parser.add_argument("--protocol_id", default="HY-CachePaper-480")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--max_edge_gap", type=int, default=15)
    parser.add_argument("--jvp_spans", default="2,3,4,5")
    return parser.parse_args()


def _velocity(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["x"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _trajectory_costs(
    latents: list[torch.Tensor],
    velocities: list[torch.Tensor],
    sigmas: torch.Tensor,
    spans: tuple[int, ...],
    *,
    max_edge_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute error of the MeanCache payload on every (source, destination) edge.

    Mirrors `flux/meancache_calibrate.py:50-89`. The cost is the mean absolute
    error, not an L2 norm, and the edge is only scored when the sliding lookback
    `source - span` exists on the trajectory.
    """

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
                true_average = (latents[destination].to(torch.float32) - z_t) / ts
                predicted_average = velocities[source].to(torch.float32) + ts * jvp
                cost = (true_average - predicted_average).abs().mean().item()
                sums[span_index, source, destination] += float(cost)
                counts[span_index, source, destination] += 1
    return sums, counts


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Hunyuan MeanCache calibration must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan MeanCache calibration requires CUDA")
    spans = tuple(int(value) for value in args.jvp_spans.split(",") if value)
    if not spans or min(spans) < 2:
        raise SystemExit("MeanCache calibration expects JVP spans >= 2")
    protocol = load_protocol(args.protocol_id)
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    sampler, _api, _load = load_official_sampler(args.model_base, protocol)
    transformer = sampler.pipeline.transformer
    trajectory_inputs: list[torch.Tensor] = []
    trajectory_velocities: list[torch.Tensor] = []

    def collect(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        x = inputs[0] if inputs else kwargs["x"]
        trajectory_inputs.append(x.detach())
        trajectory_velocities.append(_velocity(output).detach())

    handle = transformer.register_forward_hook(collect, with_kwargs=True)
    total_sums = np.zeros((len(spans), protocol.steps + 1, protocol.steps + 1))
    total_counts = np.zeros_like(total_sums, dtype=np.int64)
    rows = []
    started = time.perf_counter()
    try:
        for local_index, prompt in enumerate(selected):
            global_index = start + local_index
            seed = seed_for(args.seed, global_index)
            trajectory_inputs.clear()
            trajectory_velocities.clear()
            run = RunSpec(
                phase="meancache_calibration",
                task_id=f"meancache-cal-{args.shard_idx}-{global_index}",
                protocol_id=protocol.protocol_id,
                mode="original",
                prompt_id=f"meancache-cal-{global_index}",
                prompt=prompt,
                seed=seed,
                repeat=0,
            )
            sampler.predict(**prediction_kwargs(protocol, run))
            if len(trajectory_inputs) != protocol.steps:
                raise RuntimeError(
                    f"MeanCache collected {len(trajectory_inputs)} steps, "
                    f"expected {protocol.steps}"
                )
            sigmas = torch.as_tensor(
                sampler.pipeline.scheduler.sigmas,
                device=trajectory_inputs[0].device,
                dtype=torch.float32,
            )
            # The sampler decodes its final latent, so z_N is not returned. It is
            # exactly one Euler step past the last captured pair, and this
            # reproduces it bit for bit: the official scheduler upcasts the
            # sample to float32, multiplies the model output by the same dt and
            # never casts back (`scheduling_flow_match_discrete.py:237-242`).
            final = trajectory_inputs[-1].to(torch.float32) + trajectory_velocities[-1].to(
                torch.float32
            ) * (sigmas[protocol.steps] - sigmas[protocol.steps - 1])
            trajectory = [*trajectory_inputs, final]
            sums, counts = _trajectory_costs(
                trajectory,
                trajectory_velocities,
                sigmas,
                spans,
                max_edge_gap=args.max_edge_gap,
            )
            total_sums += sums
            total_counts += counts
            rows.append({"prompt_idx": global_index, "seed": seed})
            del trajectory, final
            trajectory_inputs.clear()
            trajectory_velocities.clear()
            torch.cuda.empty_cache()
            print(f"[hunyuan-meancache-calibration] prompt={global_index}", flush=True)
    finally:
        handle.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=total_sums,
        cost_counts=total_counts,
        jvp_spans=np.asarray(spans, dtype=np.int64),
        num_steps=np.asarray(protocol.steps, dtype=np.int64),
        max_edge_gap=np.asarray(args.max_edge_gap, dtype=np.int64),
        # FLUX and HunyuanVideo cost arrays have identical shapes, so the schedule
        # builder cannot tell them apart from the arrays alone; the tag is what
        # makes its `--model` more than free text.
        model=np.asarray("hunyuan_video"),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "hunyuan-video-meancache-cost-shard-v1",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.protocol_sha256,
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
