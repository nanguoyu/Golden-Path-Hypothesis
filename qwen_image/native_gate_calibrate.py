#!/usr/bin/env python3
"""Sweep Qwen-Image native gate thresholds without budget closure or decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io_utils import read_prompts, seed_for, split_shard
from qwen_image._helpers import atomic_write_json, decisions_filename
from qwen_image.coarse_cache import (
    QwenCoarseConfig,
    install_qwen_coarse_forward,
    qwen_coarse_decisions,
    reset_qwen_coarse_state,
    restore_qwen_coarse_forward,
)
from qwen_image.dicache import (
    QwenDiCacheConfig,
    install_qwen_dicache,
    qwen_dicache_decisions,
    reset_qwen_dicache,
    restore_qwen_dicache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("SeaCache", "TeaCache", "SenCache", "DiCache"),
        required=True,
    )
    parser.add_argument("--thresholds", required=True)
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
    parser.add_argument("--first_enhance", type=int, default=1)
    parser.add_argument("--teacache_coefficients", type=Path, default=None)
    parser.add_argument("--sencache_sensitivity", type=Path, default=None)
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


def _thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--thresholds must contain positive comma-separated values")
    return values


def _tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _tea_coefficients(path: Path | None) -> tuple[float, ...]:
    if path is None or not path.is_file():
        raise ValueError("TeaCache sweep requires --teacache_coefficients")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("coefficients")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("TeaCache coefficients must contain five values")
    return tuple(float(value) for value in values)


def _sen_scale(value: str) -> str | float:
    return "auto" if value.strip().lower() == "auto" else float(value)


def _install(pipe: Any, args: argparse.Namespace, threshold: float) -> None:
    restore_qwen_coarse_forward(pipe)
    restore_qwen_dicache(pipe)
    if args.mode == "DiCache":
        install_qwen_dicache(
            pipe,
            QwenDiCacheConfig(
                num_steps=args.num_steps,
                threshold=threshold,
                error_choice=args.dicache_error_choice,
                ret_ratio=args.dicache_ret_ratio,
                probe_depth=args.dicache_probe_depth,
                true_cfg=True,
            ),
        )
        return
    config = QwenCoarseConfig(
        mode=args.mode,
        num_steps=args.num_steps,
        first_enhance=args.first_enhance,
        seacache_thresh=threshold,
        teacache_thresh=threshold,
        teacache_coeff_source=(
            "qwen_fitted" if args.mode == "TeaCache" else "identity"
        ),
        teacache_coefficients=(
            _tea_coefficients(args.teacache_coefficients)
            if args.mode == "TeaCache"
            else None
        ),
        sencache_sensitivity_path=(
            str(args.sencache_sensitivity)
            if args.sencache_sensitivity is not None
            else None
        ),
        sencache_threshold_start=args.sencache_threshold_start,
        sencache_threshold_main=threshold,
        sencache_threshold_scale=_sen_scale(args.sencache_threshold_scale),
        sencache_switch_ratio=args.sencache_switch_ratio,
        sencache_max_skip=args.sencache_max_skip,
        sencache_ret_steps=args.sencache_ret_steps,
        sencache_cutoff_steps=args.sencache_cutoff_steps,
        true_cfg=True,
    )
    install_qwen_coarse_forward(pipe, config)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen native gate calibration requires CUDA")
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen native gate calibration requires true_cfg_scale > 1")
    if args.mode == "SenCache" and (
        args.sencache_sensitivity is None
        or not args.sencache_sensitivity.is_file()
    ):
        raise SystemExit("SenCache sweep requires --sencache_sensitivity")
    values = _thresholds(args.thresholds)
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
    summary: list[dict[str, Any]] = []
    try:
        for threshold in values:
            _install(pipe, args, threshold)
            threshold_dir = args.output_dir / (
                f"{args.mode.lower()}_t{_tag(threshold)}"
            )
            threshold_dir.mkdir(parents=True, exist_ok=True)
            cache_counts: list[int] = []
            started = time.perf_counter()
            for local_idx, prompt in enumerate(selected):
                prompt_idx = start + local_idx
                decision_path = threshold_dir / decisions_filename(prompt_idx)
                if args.resume and decision_path.is_file():
                    existing = json.loads(decision_path.read_text(encoding="utf-8"))
                    if (
                        existing.get("mode") != args.mode
                        or float(existing.get("native_threshold")) != threshold
                    ):
                        raise RuntimeError(
                            f"resume decision has incompatible method/threshold: "
                            f"{decision_path}"
                        )
                    cache_counts.append(
                        int(existing.get("summary", {}).get("n_cached"))
                    )
                    continue
                prompt_seed = seed_for(args.seed, prompt_idx)
                if args.mode == "DiCache":
                    reset_qwen_dicache(
                        pipe,
                        prompt_idx=prompt_idx,
                        seed=prompt_seed,
                    )
                else:
                    reset_qwen_coarse_state(
                        pipe,
                        prompt_idx=prompt_idx,
                        seed=prompt_seed,
                    )
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
                decisions = (
                    qwen_dicache_decisions(pipe)
                    if args.mode == "DiCache"
                    else qwen_coarse_decisions(pipe)
                )
                decisions.update(
                    {
                        "prompt": prompt,
                        "native_threshold": threshold,
                        "calibration_only": True,
                    }
                )
                atomic_write_json(decision_path, decisions)
                cache_counts.append(int(decisions["summary"]["n_cached"]))
                print(
                    f"[qwen-native-gate] {args.mode} threshold={threshold:g} "
                    f"shard={args.shard_idx} {local_idx + 1}/{len(selected)}",
                    flush=True,
                )
            summary.append(
                {
                    "threshold": threshold,
                    "directory": str(threshold_dir),
                    "prompt_count": len(cache_counts),
                    "cache_counts": cache_counts,
                    "seconds": time.perf_counter() - started,
                }
            )
    finally:
        restore_qwen_coarse_forward(pipe)
        restore_qwen_dicache(pipe)
    atomic_write_json(
        args.output_dir
        / f"summary_shard{args.shard_idx}of{args.shard_count}.json",
        {
            "format": "qwen-image-native-gate-sweep-v1",
            "mode": args.mode,
            "prompt_file": str(args.prompt_file),
            "shard_idx": args.shard_idx,
            "shard_count": args.shard_count,
            "thresholds": summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
