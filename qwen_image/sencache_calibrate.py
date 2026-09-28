#!/usr/bin/env python3
"""Collect Qwen-Image directional sensitivities for the native SenCache gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io_utils import read_prompts, seed_for, split_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
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


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float32).reshape(-1)[0].item())


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.to(torch.float32)).item())


class QwenSensitivityCollector:
    """Capture the conditional full trajectory and run adjacent probes."""

    def __init__(self, transformer: Any, num_steps: int):
        self.transformer = transformer
        self.num_steps = int(num_steps)
        self._patch: tuple[bool, Any] | None = None
        self.reset()

    def reset(self) -> None:
        self.call_idx = 0
        self.static_kwargs: dict[str, Any] | None = None
        self.latents: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def install(self) -> None:
        had_instance = "forward" in self.transformer.__dict__
        original = self.transformer.forward

        def wrapped(_module: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            if self.call_idx % 2 == 0:
                if args or "hidden_states" not in kwargs or "timestep" not in kwargs:
                    raise RuntimeError(
                        "Qwen SenCache calibration expects keyword transformer calls"
                    )
                if self.static_kwargs is None:
                    self.static_kwargs = {
                        key: value
                        for key, value in kwargs.items()
                        if key not in {"hidden_states", "timestep"}
                    }
                self.latents.append(kwargs["hidden_states"].detach().clone())
                self.timesteps.append(kwargs["timestep"].detach().clone())
                self.outputs.append(_sample(output).detach().clone())
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

    @torch.no_grad()
    def rows(self, *, prompt_id: int, seed: int) -> list[dict[str, Any]]:
        if (
            len(self.latents) != self.num_steps
            or len(self.timesteps) != self.num_steps
            or len(self.outputs) != self.num_steps
            or self.static_kwargs is None
            or self._patch is None
        ):
            raise RuntimeError(
                "incomplete Qwen SenCache trajectory: "
                f"latents={len(self.latents)} timesteps={len(self.timesteps)} "
                f"outputs={len(self.outputs)} expected={self.num_steps}"
            )
        original = self._patch[1]
        rows: list[dict[str, Any]] = []
        for step in range(self.num_steps):
            reference = step + 1 if step + 1 < self.num_steps else step - 1
            x_i = self.latents[step]
            x_ref = self.latents[reference]
            t_i = self.timesteps[step]
            t_ref = self.timesteps[reference]
            o_i = self.outputs[step].to(torch.float32)
            o_x = _sample(
                original(
                    hidden_states=x_ref,
                    timestep=t_i,
                    **self.static_kwargs,
                )
            ).detach().to(torch.float32)
            o_t = _sample(
                original(
                    hidden_states=x_i,
                    timestep=t_ref,
                    **self.static_kwargs,
                )
            ).detach().to(torch.float32)
            delta_x = _norm(x_ref - x_i)
            delta_t = abs(_scalar(t_ref) - _scalar(t_i))
            rows.append(
                {
                    "prompt_id": int(prompt_id),
                    "seed": int(seed),
                    "step_index": int(step),
                    "timestep": _scalar(t_i),
                    "reference_step_index": int(reference),
                    "reference_timestep": _scalar(t_ref),
                    "latent_shape": json.dumps(list(x_i.shape)),
                    "latent_numel": int(x_i.numel()),
                    "delta_latent_norm": delta_x,
                    "delta_t_abs": delta_t,
                    "J_x_directional": (
                        _norm(o_x - o_i) / delta_x if delta_x > 0.0 else None
                    ),
                    "J_t_directional": (
                        _norm(o_t - o_i) / delta_t if delta_t > 0.0 else None
                    ),
                }
            )
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty SenCache calibration shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen SenCache calibration requires CUDA")
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen SenCache calibration requires true_cfg_scale > 1")
    shard_path = args.output_dir / (
        f"sencache_sensitivity_rows_shard{args.shard_idx}of"
        f"{args.shard_count}.csv"
    )
    if args.resume and shard_path.is_file():
        print(f"[qwen-sencache-calibrate] preserve {shard_path}", flush=True)
        return 0
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
    collector = QwenSensitivityCollector(pipe.transformer, args.num_steps)
    collector.install()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for local_idx, prompt in enumerate(selected):
            prompt_idx = start + local_idx
            prompt_seed = seed_for(args.seed, prompt_idx)
            collector.reset()
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
                rows.extend(
                    collector.rows(prompt_id=prompt_idx, seed=prompt_seed)
                )
            torch.cuda.synchronize()
            print(
                f"[qwen-sencache-calibrate] shard={args.shard_idx} "
                f"{local_idx + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        collector.restore()
    _write_csv(shard_path, rows)
    print(
        f"[qwen-sencache-calibrate] wrote {shard_path} "
        f"rows={len(rows)} seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
