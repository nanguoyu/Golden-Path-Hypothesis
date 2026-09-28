#!/usr/bin/env python3
"""Collect HunyuanVideo SenCache sensitivity calibration rows.

This is compute-heavy and must run on compute nodes.  It estimates timestep-wise
directional sensitivity norms from full HunyuanVideo trajectories:

  J_x(t_i) ~= ||f(x_ref, t_i) - f(x_i, t_i)|| / ||x_ref - x_i||
  J_t(t_i) ~= ||f(x_i, t_ref) - f(x_i, t_i)|| / |t_ref - t_i|

where ``ref`` is the adjacent full-trajectory state/timestep, so every step costs
two extra transformer forwards on top of the trajectory itself.  ``x`` is the
latent the transformer receives before ``img_in`` and ``t`` its native
range(0, 1000) timestep — the same two quantities the online gate scores.

``analysis/merge_sencache_sensitivity.py --backbone hunyuan_video`` freezes the
per-step q90 aggregate of these rows into the npz table the gate reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hunyuan_video.backend import load_official_sampler, prediction_kwargs
from hunyuan_video.config import GenerationProtocol, RunSpec, load_protocol
from lib.io_utils import read_prompts, seed_for, split_shard


PRECISION_TO_TYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_base", type=Path, required=True)
    parser.add_argument("--protocol_id", default="HY-CachePaper-480")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_commit() -> tuple[str, bool]:
    sha = subprocess.check_output(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(_ROOT), "status", "--short"], text=True
        ).strip()
    )
    return sha, dirty


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).norm().item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TrajectoryRecorder:
    """Record every transformer input, output, and the shared conditioning."""

    def __init__(self, transformer: Any):
        self.transformer = transformer
        self.inputs: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.conditioning: dict[str, Any] | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "TrajectoryRecorder":
        self._handles.append(
            self.transformer.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.transformer.register_forward_hook(self._post, with_kwargs=True)
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _pre(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if len(args) > 2:
            raise RuntimeError("HunyuanVideo passes only x and t positionally")
        x = args[0] if args else kwargs["x"]
        t = args[1] if len(args) > 1 else kwargs["t"]
        if self.conditioning is None:
            self.conditioning = {
                key: value for key, value in kwargs.items() if key not in ("x", "t")
            }
            self.device = x.device
            self.dtype = x.dtype
        self.inputs.append(x.detach().to("cpu", torch.float32))
        self.timesteps.append(t.detach().clone())

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        value = output["x"] if isinstance(output, dict) else output
        self.outputs.append(value.detach().to("cpu", torch.float32))
        return output

    def forward(self, x: torch.Tensor, t: torch.Tensor, autocast: torch.dtype | None) -> torch.Tensor:
        """Re-run the recorded conditioning on a substituted (x, t) pair."""

        if self.conditioning is None or self.device is None or self.dtype is None:
            raise RuntimeError("no recorded HunyuanVideo transformer call to replay")
        model_x = x.to(device=self.device, dtype=self.dtype)
        with torch.autocast(
            device_type="cuda", dtype=autocast or torch.float32, enabled=autocast is not None
        ):
            with torch.no_grad():
                output = self.transformer(model_x, t, **self.conditioning)
        value = output["x"] if isinstance(output, dict) else output
        return value.detach().to("cpu", torch.float32)


def _rows_for_prompt(
    sampler: Any,
    protocol: GenerationProtocol,
    run: RunSpec,
    prompt_idx: int,
) -> list[dict[str, Any]]:
    transformer = sampler.pipeline.transformer
    target_dtype = PRECISION_TO_TYPE[protocol.precision]
    autocast = target_dtype if target_dtype != torch.float32 else None
    with TrajectoryRecorder(transformer) as recorder:
        sampler.predict(**prediction_kwargs(protocol, run))
    steps = len(recorder.inputs)
    if steps != protocol.steps or len(recorder.outputs) != steps:
        raise RuntimeError(
            f"recorded {steps} inputs / {len(recorder.outputs)} outputs "
            f"for {protocol.steps} steps"
        )

    rows: list[dict[str, Any]] = []
    for i in range(steps):
        if steps <= 1:
            ref = i
        elif i < steps - 1:
            ref = i + 1
        else:
            ref = i - 1
        x_i = recorder.inputs[i]
        x_ref = recorder.inputs[ref]
        o_i = recorder.outputs[i]
        t_i = recorder.timesteps[i]
        t_ref = recorder.timesteps[ref]
        o_x = recorder.forward(x_ref, t_i, autocast)
        o_t = recorder.forward(x_i, t_ref, autocast)
        dx = _norm(x_ref - x_i)
        dt = abs(
            float(t_ref.to(torch.float32).reshape(-1)[0].item())
            - float(t_i.to(torch.float32).reshape(-1)[0].item())
        )
        rows.append(
            {
                "prompt_id": int(prompt_idx),
                "seed": int(run.seed),
                "step_index": int(i),
                "timestep": float(t_i.to(torch.float32).reshape(-1)[0].item()),
                "reference_step_index": int(ref),
                "reference_timestep": float(t_ref.to(torch.float32).reshape(-1)[0].item()),
                "latent_shape": json.dumps(list(x_i.shape)),
                "latent_numel": int(x_i.numel()),
                "delta_latent_norm": float(dx),
                "delta_t_abs": float(dt),
                "J_x_directional": (_norm(o_x - o_i) / dx) if dx > 0.0 else None,
                "J_t_directional": (_norm(o_t - o_i) / dt) if dt > 0.0 else None,
                "protocol_id": protocol.protocol_id,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("HunyuanVideo sensitivity calibration must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo sensitivity calibration requires CUDA")

    protocol = load_protocol(args.protocol_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = args.output_dir / (
        f"sencache_sensitivity_rows_shard{args.shard_idx}of{args.shard_count}.csv"
    )
    if args.resume and shard_csv.is_file():
        print(f"[hunyuan-sencache-calib] {shard_csv} exists, skip", flush=True)
        return 0

    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    sampler, _api, _load = load_official_sampler(args.model_base, protocol)
    git_sha, git_dirty = _git_commit()
    rows: list[dict[str, Any]] = []
    for local_idx, prompt in enumerate(selected):
        idx = start + local_idx
        run = RunSpec(
            phase="sencache_sensitivity",
            task_id=f"sencache-calib-{args.shard_idx}-{idx}",
            protocol_id=protocol.protocol_id,
            mode="original",
            prompt_id=f"prompt-{idx}",
            prompt=prompt,
            seed=seed_for(args.seed, idx),
            repeat=0,
            method_config={},
        )
        started = time.perf_counter()
        rows.extend(_rows_for_prompt(sampler, protocol, run, idx))
        torch.cuda.empty_cache()
        print(
            f"[hunyuan-sencache-calib] shard={args.shard_idx} idx={idx} "
            f"steps={protocol.steps} dt={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    for row in rows:
        row["git_commit"] = git_sha
        row["git_dirty"] = git_dirty
    _write_csv(shard_csv, rows)
    print(f"[hunyuan-sencache-calib] wrote {shard_csv} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
