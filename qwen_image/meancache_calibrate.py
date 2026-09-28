#!/usr/bin/env python3
"""Collect Qwen-Image MeanCache edge costs from full true-CFG trajectories."""

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

from lib.io_utils import read_prompts, seed_for, split_shard
from qwen_image.meancache import guided_velocity


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
    parser.add_argument("--max_edge_gap", type=int, default=15)
    parser.add_argument("--jvp_spans", default="2,3,4,5")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _sample(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if not isinstance(sample, torch.Tensor):
        raise TypeError(f"unsupported Qwen output: {type(output).__name__}")
    return sample


def trajectory_costs(
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
            delta_sigma = sigma_t - sigma_r
            if float(delta_sigma.abs().item()) < 1e-12:
                continue
            z_r = latents[reference].to(torch.float32)
            z_t = latents[source].to(torch.float32)
            v_r = velocities[reference].to(torch.float32)
            jvp = (z_t - z_r - delta_sigma * v_r) / (delta_sigma * delta_sigma)
            stop = min(steps, source + int(max_edge_gap))
            for destination in range(source + 1, stop + 1):
                step_sigma = sigmas[destination].to(torch.float32) - sigma_t
                true_average = (
                    latents[destination].to(torch.float32) - z_t
                ) / step_sigma
                predicted_average = (
                    velocities[source].to(torch.float32) + step_sigma * jvp
                )
                cost = (true_average - predicted_average).abs().mean().item()
                sums[span_index, source, destination] += float(cost)
                counts[span_index, source, destination] += 1
    return sums, counts


class QwenMeanTrajectoryCollector:
    def __init__(self, transformer: Any, true_cfg_scale: float):
        self.transformer = transformer
        self.true_cfg_scale = float(true_cfg_scale)
        self._patch: tuple[bool, Any] | None = None
        self.reset()

    def reset(self) -> None:
        self.call_idx = 0
        self.latents: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.pending_cond: torch.Tensor | None = None

    def install(self) -> None:
        had_instance = "forward" in self.transformer.__dict__
        original = self.transformer.forward

        def wrapped(_module: Any, *args: Any, **kwargs: Any) -> Any:
            latent = kwargs.get(
                "hidden_states",
                args[0] if args else None,
            )
            if not isinstance(latent, torch.Tensor):
                raise RuntimeError("Qwen MeanCache calibration could not find latent")
            output = original(*args, **kwargs)
            prediction = _sample(output).detach()
            if self.call_idx % 2 == 0:
                self.latents.append(latent.detach().clone())
                self.pending_cond = prediction
            else:
                if self.pending_cond is None:
                    raise RuntimeError("Qwen MeanCache uncond call has no cond output")
                self.velocities.append(
                    guided_velocity(
                        self.pending_cond,
                        prediction,
                        true_cfg_scale=self.true_cfg_scale,
                    ).detach()
                )
                self.pending_cond = None
            self.call_idx += 1
            return output

        self.transformer.forward = types.MethodType(wrapped, self.transformer)
        self._patch = (had_instance, original)

    def restore(self) -> None:
        if self._patch is None:
            return
        had_instance, original = self._patch
        if had_instance:
            self.transformer.forward = original
        else:
            delattr(self.transformer, "forward")
        self._patch = None


def main() -> int:
    args = parse_args()
    if (
        args.resume
        and args.out.is_file()
        and args.out.with_suffix(".json").is_file()
    ):
        print(f"[qwen-meancache-calibrate] preserve {args.out}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen MeanCache calibration requires CUDA")
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen MeanCache calibration requires true_cfg_scale > 1")
    spans = tuple(int(value) for value in args.jvp_spans.split(",") if value)
    if not spans or min(spans) < 2:
        raise SystemExit("Qwen MeanCache calibration expects JVP spans >= 2")
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
    collector = QwenMeanTrajectoryCollector(
        pipe.transformer,
        args.true_cfg_scale,
    )
    collector.install()
    sums = np.zeros(
        (len(spans), args.num_steps + 1, args.num_steps + 1),
        dtype=np.float64,
    )
    counts = np.zeros_like(sums, dtype=np.int64)
    prompt_rows: list[dict[str, int]] = []
    started = time.perf_counter()
    try:
        for local_idx, prompt in enumerate(selected):
            prompt_idx = start + local_idx
            prompt_seed = seed_for(args.seed, prompt_idx)
            collector.reset()
            generator = torch.Generator(device="cuda").manual_seed(prompt_seed)
            with torch.no_grad():
                result = pipe(
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
            final_latent = getattr(result, "images", None)
            if not isinstance(final_latent, torch.Tensor):
                raise RuntimeError("Qwen pipeline did not return final latent tensor")
            if (
                len(collector.latents) != args.num_steps
                or len(collector.velocities) != args.num_steps
            ):
                raise RuntimeError(
                    "incomplete Qwen MeanCache trajectory: "
                    f"latents={len(collector.latents)} "
                    f"velocities={len(collector.velocities)}"
                )
            trajectory = [*collector.latents, final_latent.detach()]
            sigmas = torch.as_tensor(
                pipe.scheduler.sigmas,
                device=trajectory[0].device,
                dtype=torch.float32,
            )
            if len(sigmas) < args.num_steps + 1:
                raise RuntimeError("Qwen scheduler has an incomplete sigma trajectory")
            prompt_sums, prompt_counts = trajectory_costs(
                trajectory,
                collector.velocities,
                sigmas,
                spans,
                max_edge_gap=args.max_edge_gap,
            )
            sums += prompt_sums
            counts += prompt_counts
            prompt_rows.append({"prompt_idx": prompt_idx, "seed": prompt_seed})
            torch.cuda.synchronize()
            print(
                f"[qwen-meancache-calibrate] shard={args.shard_idx} "
                f"{local_idx + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        collector.restore()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=sums,
        cost_counts=counts,
        jvp_spans=np.asarray(spans, dtype=np.int64),
        num_steps=np.asarray(args.num_steps, dtype=np.int64),
        max_edge_gap=np.asarray(args.max_edge_gap, dtype=np.int64),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "qwen-image-meancache-cost-shard-v1",
                "prompt_file": str(args.prompt_file),
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "prompts": prompt_rows,
                "prediction_target": "post_true_cfg_guided_velocity",
                "true_cfg_scale": args.true_cfg_scale,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
