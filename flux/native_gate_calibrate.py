#!/usr/bin/env python3
"""Sweep FLUX native gate thresholds without budget closure or VAE decoding."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.coarse_native import FluxNativeGateAdapter, FluxNativeGateConfig
from flux.dicache_native import FluxDiCacheAdapter, FluxDiCacheConfig
from lib.io_utils import read_prompts, seed_for, split_shard


MODES = ("SeaCache", "TeaCache", "SenCache", "DiCache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--model_name",
        choices=("flux-dev", "flux-schnell"),
        default="flux-dev",
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--first_enhance", type=int, default=1)
    parser.add_argument("--sencache_sensitivity", type=Path)
    parser.add_argument("--sencache_threshold_start", type=float, default=0.005)
    parser.add_argument("--sencache_threshold_scale", default="auto")
    parser.add_argument("--sencache_switch_ratio", type=float, default=0.2)
    parser.add_argument("--sencache_max_skip", type=int, default=10)
    parser.add_argument("--sencache_ret_steps", type=int, default=0)
    parser.add_argument("--sencache_cutoff_steps", type=int, default=-1)
    parser.add_argument(
        "--dicache_error_choice",
        choices=("delta_y", "delta_minus"),
        default="delta_y",
    )
    parser.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    parser.add_argument("--dicache_probe_depth", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value.strip())
    if not values or any(
        not math.isfinite(value) or value <= 0.0
        for value in values
    ):
        raise ValueError("--thresholds must contain positive comma-separated values")
    if len(set(values)) != len(values):
        raise ValueError("--thresholds must not contain duplicates")
    return values


def threshold_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _sen_scale(value: str) -> str | float:
    return "auto" if value.strip().lower() == "auto" else float(value)


def build_adapter(
    pipe: Any,
    args: argparse.Namespace,
    threshold: float,
) -> FluxNativeGateAdapter | FluxDiCacheAdapter:
    if args.mode == "DiCache":
        return FluxDiCacheAdapter(
            pipe,
            FluxDiCacheConfig(
                num_steps=args.num_steps,
                threshold=threshold,
                error_choice=args.dicache_error_choice,
                ret_ratio=args.dicache_ret_ratio,
                probe_depth=args.dicache_probe_depth,
            ),
        )
    return FluxNativeGateAdapter(
        pipe,
        FluxNativeGateConfig(
            mode=args.mode.lower(),
            num_steps=args.num_steps,
            threshold=threshold,
            first_enhance=args.first_enhance,
            sencache_sensitivity_path=args.sencache_sensitivity,
            sencache_threshold_start=args.sencache_threshold_start,
            sencache_threshold_scale=_sen_scale(
                args.sencache_threshold_scale
            ),
            sencache_switch_ratio=args.sencache_switch_ratio,
            sencache_max_skip=args.sencache_max_skip,
            sencache_ret_steps=args.sencache_ret_steps,
            sencache_cutoff_steps=args.sencache_cutoff_steps,
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resume_count(
    path: Path,
    *,
    mode: str,
    threshold: float,
) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("native_method") != mode
        or float(payload.get("native_threshold")) != threshold
    ):
        raise RuntimeError(
            f"resume decision has incompatible method/threshold: {path}"
        )
    return int(payload["summary"]["n_cached"])


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX native gate calibration requires CUDA")
    if args.num_steps != 50:
        raise SystemExit("FLUX matched-baseline calibration is frozen to 50 steps")
    if args.mode == "SenCache" and (
        args.sencache_sensitivity is None
        or not args.sencache_sensitivity.is_file()
    ):
        raise SystemExit("SenCache sweep requires --sencache_sensitivity")

    values = parse_thresholds(args.thresholds)
    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    from diffusers import DiffusionPipeline
    from flux.oracle_runner import _run_one_pipe_call

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to("cuda")
    summaries: list[dict[str, Any]] = []
    for threshold in values:
        adapter = build_adapter(pipe, args, threshold)
        adapter.install()
        threshold_dir = args.output_dir / (
            f"{args.mode.lower()}_t{threshold_tag(threshold)}"
        )
        cache_counts: list[int] = []
        started = time.perf_counter()
        try:
            for local_idx, prompt in enumerate(selected):
                prompt_idx = start + local_idx
                decision_path = threshold_dir / f"decisions_{prompt_idx:05d}.json"
                if args.resume and decision_path.is_file():
                    cache_counts.append(
                        _resume_count(
                            decision_path,
                            mode=args.mode,
                            threshold=threshold,
                        )
                    )
                    continue
                prompt_seed = seed_for(args.seed, prompt_idx)
                adapter.reset(prompt_idx=prompt_idx, seed=prompt_seed)
                with torch.no_grad():
                    _run_one_pipe_call(pipe, prompt, prompt_seed, args)
                decisions = adapter.decisions()
                decisions.update(
                    {
                        "prompt": prompt,
                        "native_method": args.mode,
                        "native_threshold": threshold,
                        "calibration_only": True,
                    }
                )
                _write_json(decision_path, decisions)
                cache_counts.append(int(decisions["summary"]["n_cached"]))
                print(
                    f"[flux-native-gate] {args.mode} threshold={threshold:g} "
                    f"shard={args.shard_idx} "
                    f"{local_idx + 1}/{len(selected)}",
                    flush=True,
                )
        finally:
            adapter.restore()
        summaries.append(
            {
                "threshold": threshold,
                "directory": str(threshold_dir),
                "prompt_count": len(cache_counts),
                "cache_counts": cache_counts,
                "seconds": time.perf_counter() - started,
            }
        )

    _write_json(
        args.output_dir
        / f"summary_shard{args.shard_idx}of{args.shard_count}.json",
        {
            "format": "flux-native-gate-sweep-v1",
            "mode": args.mode,
            "prompt_file": str(args.prompt_file),
            "base_seed": args.seed,
            "shard_idx": args.shard_idx,
            "shard_count": args.shard_count,
            "thresholds": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
