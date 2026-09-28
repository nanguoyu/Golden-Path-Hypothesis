#!/usr/bin/env python3
"""Isolated FLUX runner for matched-budget baseline screening."""

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

from flux.coarse_native import FluxNativeGateAdapter, FluxNativeGateConfig
from flux.dicache_native import FluxDiCacheAdapter, FluxDiCacheConfig
from flux.l2p_output import FluxL2POutputAdapter
from lib.io_utils import (
    image_filename,
    read_prompts,
    seed_for,
    split_shard,
    write_timing_json,
)


MODES = (
    "original_control",
    "fixed_reuse_control",
    "seacache_native",
    "teacache_native",
    "sencache_native",
    "dicache_native",
    "taylorseer_fine_exact",
    "hicache_fine_exact",
    "l2p_output_exact",
    "foca_fine_exact",
    "toca_exact",
    "dpcache_exact",
    "budcache_exact",
    "meancache_exact",
)


class _OriginalControl:
    def __init__(self, *, num_steps: int) -> None:
        self.num_steps = int(num_steps)
        self.prompt_idx: int | None = None
        self.seed: int | None = None

    def reset(self, *, prompt_idx: int | None = None, seed: int | None = None) -> None:
        self.prompt_idx = prompt_idx
        self.seed = seed

    def decisions(self) -> dict[str, Any]:
        rows = [
            {"step": step, "action": "full", "u": 0}
            for step in range(self.num_steps)
        ]
        return {
            "schema": "flux_original_control_decisions.v1",
            "prompt_idx": self.prompt_idx,
            "seed": self.seed,
            "mode": "original_control",
            "num_steps": self.num_steps,
            "per_step": rows,
            "summary": {
                "n_total": self.num_steps,
                "n_full": self.num_steps,
                "n_cached": 0,
                "cache_ratio": 0.0,
            },
        }


class _FineExactAdapter:
    def __init__(
        self,
        pipe: Any,
        *,
        mode: str,
        cache_count: int,
        reset_fn: Any,
        action_steps: tuple[int, ...] = (),
    ) -> None:
        self.pipe = pipe
        self.mode = mode
        self.cache_count = int(cache_count)
        self.reset_fn = reset_fn
        self.action_steps = tuple(int(step) for step in action_steps)

    def reset(self, *, prompt_idx: int | None = None, **_: Any) -> None:
        if self.action_steps:
            from lib.flux_fine_scaffold import reset_per_image_state_fine

            reset_per_image_state_fine(
                self.pipe,
                action_steps=set(self.action_steps),
                prompt_idx=prompt_idx,
            )
        else:
            self.reset_fn(self.pipe)

    def decisions(self) -> dict[str, Any]:
        rows = list(getattr(self.pipe.transformer, "fine_cache_decisions", ()))
        if len(rows) != 50:
            raise RuntimeError(
                f"{self.mode} recorded {len(rows)} decisions; expected 50"
            )
        n_cached = sum(int(row.get("u", 0)) for row in rows)
        if n_cached != self.cache_count:
            raise RuntimeError(
                f"{self.mode} produced K={n_cached}; expected K={self.cache_count}"
            )
        return {
            "schema": "flux_fine_exact_decisions.v1",
            "mode": self.mode,
            "num_steps": 50,
            "target_cache_count": self.cache_count,
            "per_step": rows,
            "summary": {
                "n_total": 50,
                "n_full": 50 - n_cached,
                "n_cached": n_cached,
                "cache_ratio": n_cached / 50,
            },
        }


def _step_list(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item.strip())
    if tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError("cache steps must be sorted and unique")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--revision",
        default=None,
        help="optional model snapshot pin forwarded to from_pretrained",
    )
    parser.add_argument("--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int)
    parser.add_argument("--cache_steps", type=_step_list, default=())
    parser.add_argument("--schedule_json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--first_enhance", type=int)
    parser.add_argument("--interval", type=int, default=7)
    parser.add_argument("--max_order", type=int, default=1)
    parser.add_argument("--hicache_sigma", type=float, default=0.5)
    parser.add_argument("--sencache_sensitivity_path", type=Path)
    parser.add_argument("--sencache_threshold_start", type=float, default=0.005)
    parser.add_argument("--sencache_threshold_scale", default="auto")
    parser.add_argument("--sencache_switch_ratio", type=float, default=0.2)
    parser.add_argument("--sencache_max_skip", type=int, default=10)
    parser.add_argument("--sencache_ret_steps", type=int, default=0)
    parser.add_argument("--sencache_cutoff_steps", type=int, default=-1)
    parser.add_argument("--dicache_error_choice", choices=("delta_y", "delta_minus"), default="delta_y")
    parser.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    parser.add_argument("--dicache_probe_depth", type=int, default=1)
    parser.add_argument("--l2p_weights", type=Path)
    parser.add_argument("--l2p_min_abs_weight", type=float, default=0.0)
    parser.add_argument(
        "--foca_heun_variant",
        choices=("paper_literal", "anchored_prose", "none"),
        default="paper_literal",
    )
    parser.add_argument(
        "--foca_history_policy",
        choices=("recursive", "full_refresh_only"),
        default="recursive",
    )
    parser.add_argument("--foca_h", type=float, default=1.0)
    parser.add_argument("--toca_fresh_ratio", type=float, default=0.1)
    parser.add_argument("--toca_soft_fresh_weight", type=float, default=0.25)
    parser.add_argument("--toca_fresh_threshold", type=int, default=4)
    parser.add_argument("--dpcache_order", type=int, default=2)
    parser.add_argument("--meancache_jvp_span", type=int, default=4)
    return parser.parse_args()


def _schedule(args: argparse.Namespace) -> tuple[int, ...]:
    payload: dict[str, Any] = {}
    if args.schedule_json is not None:
        if not args.schedule_json.is_file():
            raise SystemExit(f"missing schedule JSON: {args.schedule_json}")
        raw = json.loads(args.schedule_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("cache_steps"), list):
            raise SystemExit("schedule JSON must contain a cache_steps list")
        payload = raw
        from_file = tuple(int(step) for step in raw["cache_steps"])
        if args.cache_steps and tuple(args.cache_steps) != from_file:
            raise SystemExit("--cache_steps differs from --schedule_json")
        args.cache_steps = from_file
    args._schedule_payload = payload
    return tuple(args.cache_steps)


def _validate(args: argparse.Namespace) -> tuple[int, ...]:
    if args.num_steps != 50:
        raise SystemExit("this screening lane is frozen to 50 denoising steps")
    fine_modes = {
        "taylorseer_fine_exact",
        "hicache_fine_exact",
        "foca_fine_exact",
    }
    if args.first_enhance is None:
        args.first_enhance = 3 if args.mode in fine_modes else 1
    explicit = _schedule(args)
    if explicit and explicit[0] <= 0:
        raise SystemExit("--cache_steps cannot include the first step")
    if args.mode == "sencache_native":
        if args.sencache_sensitivity_path is None or not args.sencache_sensitivity_path.is_file():
            raise SystemExit("sencache_native requires --sencache_sensitivity_path")
    fixed_modes = {
        "fixed_reuse_control",
        "taylorseer_fine_exact",
        "hicache_fine_exact",
        "l2p_output_exact",
        "foca_fine_exact",
        "toca_exact",
        "dpcache_exact",
        "budcache_exact",
        "meancache_exact",
    }
    native_modes = {
        "seacache_native",
        "teacache_native",
        "sencache_native",
        "dicache_native",
    }
    if args.mode in fixed_modes:
        if args.cache_count is None:
            raise SystemExit(f"{args.mode} requires --cache_count")
        if not 0 <= args.cache_count <= args.num_steps - 2:
            raise SystemExit(
                "--cache_count must preserve at least the first and last full steps"
            )
        if explicit and len(explicit) != args.cache_count:
            raise SystemExit("--cache_steps length differs from --cache_count")
    elif args.mode in native_modes:
        if args.cache_count is not None:
            raise SystemExit(
                f"{args.mode} is a native gate and must not receive --cache_count"
            )
    elif args.mode == "original_control":
        if args.cache_count not in (None, 0):
            raise SystemExit("original_control only accepts --cache_count 0")
    if args.mode == "l2p_output_exact":
        if args.l2p_weights is None or not args.l2p_weights.is_file():
            raise SystemExit("l2p_output_exact requires --l2p_weights")
    if args.mode in fixed_modes:
        if not explicit:
            raise SystemExit(f"{args.mode} requires an explicit exact-K schedule")
        if explicit != tuple(sorted(set(explicit))):
            raise SystemExit("--cache_steps must be sorted and unique")
        if explicit[-1] >= args.num_steps:
            raise SystemExit("--cache_steps contains an out-of-range step")
    if args.mode == "l2p_output_exact":
        return explicit
    if args.mode in fine_modes:
        if args.interval < 2:
            raise SystemExit("fine exact modes require interval >= 2")
        if args.first_enhance != 3:
            raise SystemExit("the FLUX fine comparisons freeze first_enhance=3")
        if args.mode == "taylorseer_fine_exact" and args.max_order != 1:
            raise SystemExit("the FLUX TaylorSeer comparison freezes max_order=1")
        if args.mode == "hicache_fine_exact":
            if args.max_order != 2:
                raise SystemExit("the FLUX HiCache comparison freezes max_order=2")
            if not 0.0 < args.hicache_sigma <= 1.0:
                raise SystemExit("hicache_sigma must be in (0, 1]")
    if args.mode == "dpcache_exact" and args.dpcache_order != 2:
        raise SystemExit("DPCache is frozen to its order-2 predictor")
    if args.mode == "meancache_exact" and args.meancache_jvp_span < 1:
        raise SystemExit("MeanCache JVP span must be positive")
    if args.mode not in fixed_modes and explicit:
        raise SystemExit(f"{args.mode} uses its own online gate, not --cache_steps")
    return explicit


def _install(args: argparse.Namespace, pipe: Any, cache_steps: tuple[int, ...]) -> tuple[Any, Any]:
    if args.mode == "original_control":
        return _OriginalControl(num_steps=args.num_steps), lambda: None
    if args.mode in {"seacache_native", "teacache_native", "sencache_native"}:
        adapter = FluxNativeGateAdapter(
            pipe,
            FluxNativeGateConfig(
                mode=args.mode.removesuffix("_native"),
                num_steps=args.num_steps,
                threshold=args.threshold,
                first_enhance=args.first_enhance,
                sencache_sensitivity_path=args.sencache_sensitivity_path,
                sencache_threshold_start=args.sencache_threshold_start,
                sencache_threshold_scale=args.sencache_threshold_scale,
                sencache_switch_ratio=args.sencache_switch_ratio,
                sencache_max_skip=args.sencache_max_skip,
                sencache_ret_steps=args.sencache_ret_steps,
                sencache_cutoff_steps=args.sencache_cutoff_steps,
            ),
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode == "dicache_native":
        adapter = FluxDiCacheAdapter(
            pipe,
            FluxDiCacheConfig(
                num_steps=args.num_steps,
                threshold=args.threshold,
                error_choice=args.dicache_error_choice,
                ret_ratio=args.dicache_ret_ratio,
                probe_depth=args.dicache_probe_depth,
            ),
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode in {"taylorseer_fine_exact", "hicache_fine_exact"}:
        if args.mode == "taylorseer_fine_exact":
            from flux import taylorseer_fine as fine_module

            teardown = fine_module.install(
                pipe,
                interval=args.interval,
                max_order=args.max_order,
                first_enhance=args.first_enhance,
                num_steps=args.num_steps,
            )
        else:
            from flux import hicache_fine as fine_module

            teardown = fine_module.install(
                pipe,
                interval=args.interval,
                max_order=args.max_order,
                sigma=args.hicache_sigma,
                first_enhance=args.first_enhance,
                num_steps=args.num_steps,
            )
        adapter = _FineExactAdapter(
            pipe,
            mode=args.mode,
            cache_count=args.cache_count,
            reset_fn=fine_module.reset_per_image_state,
            action_steps=cache_steps,
        )
        return adapter, teardown
    if args.mode == "l2p_output_exact":
        assert args.l2p_weights is not None
        adapter = FluxL2POutputAdapter(
            pipe,
            weights_path=args.l2p_weights,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            min_abs_weight=args.l2p_min_abs_weight,
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode == "foca_fine_exact":
        from flux import foca_fine as fine_module

        teardown = fine_module.install(
            pipe,
            interval=args.interval,
            first_enhance=args.first_enhance,
            num_steps=args.num_steps,
            heun_variant=args.foca_heun_variant,
            history_policy=args.foca_history_policy,
            h=args.foca_h,
        )
        adapter = _FineExactAdapter(
            pipe,
            mode=args.mode,
            cache_count=args.cache_count,
            reset_fn=fine_module.reset_per_image_state,
            action_steps=cache_steps,
        )
        return adapter, teardown
    if args.mode == "toca_exact":
        from flux.toca_exact import FluxToCaAdapter

        adapter = FluxToCaAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            fresh_ratio=args.toca_fresh_ratio,
            soft_fresh_weight=args.toca_soft_fresh_weight,
            fresh_threshold=args.toca_fresh_threshold,
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode == "dpcache_exact":
        from flux.dpcache_exact import FluxDPCacheAdapter

        adapter = FluxDPCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            order=args.dpcache_order,
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode in {"fixed_reuse_control", "budcache_exact"}:
        from flux.fixed_residual_exact import FluxFixedResidualAdapter

        adapter = FluxFixedResidualAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            mode=args.mode,
        )
        adapter.install()
        return adapter, adapter.restore
    if args.mode == "meancache_exact":
        from flux.meancache_exact import FluxMeanCacheAdapter

        raw_spans = getattr(args, "_schedule_payload", {}).get("jvp_spans", {})
        if isinstance(raw_spans, list):
            raw_spans = {
                step: span for step, span in enumerate(raw_spans) if span is not None
            }
        if not isinstance(raw_spans, dict):
            raise SystemExit("MeanCache schedule jvp_spans must be a mapping or list")
        adapter = FluxMeanCacheAdapter(
            pipe,
            cache_steps=cache_steps,
            num_steps=args.num_steps,
            jvp_span=args.meancache_jvp_span,
            jvp_spans={int(step): int(span) for step, span in raw_spans.items()},
        )
        adapter.install()
        return adapter, adapter.restore
    raise AssertionError(f"unhandled baseline mode: {args.mode}")


def _reset(adapter: Any, idx: int, seed: int) -> None:
    adapter.reset(prompt_idx=idx, seed=seed)


def _decisions(
    adapter: Any,
) -> dict[str, Any]:
    return adapter.decisions()


def main() -> int:
    args = parse_args()
    cache_steps = _validate(args)
    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    from flux.oracle_runner import _decode_to_pil, _run_one_pipe_call
    from diffusers import DiffusionPipeline

    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, revision=args.revision
    ).to("cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start
    adapter, teardown = _install(args, pipe, cache_steps)
    per_image: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    try:
        for local_idx, prompt in enumerate(selected):
            idx = start + local_idx
            out_path = args.output_dir / image_filename(idx)
            dec_path = args.output_dir / f"decisions_{idx:05d}.json"
            if args.resume and out_path.is_file() and dec_path.is_file():
                continue
            seed = seed_for(args.seed, idx)
            _reset(adapter, idx, seed)
            torch.cuda.synchronize()
            denoise_start = time.perf_counter()
            latent = _run_one_pipe_call(pipe, prompt, seed, args)
            denoise_s = time.perf_counter() - denoise_start
            decode_start = time.perf_counter()
            image = _decode_to_pil(
                pipe,
                latent,
                (args.height // 16) * 16,
                (args.width // 16) * 16,
            )
            torch.cuda.synchronize()
            decode_s = time.perf_counter() - decode_start
            image.save(out_path)
            decisions = _decisions(adapter)
            decisions.update({"prompt": prompt, "seed": seed, "image_file": out_path.name})
            dec_path.write_text(
                json.dumps(decisions, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            per_image.append(
                {
                    "idx": idx,
                    "seed": seed,
                    "denoise_s": denoise_s,
                    "decode_s": decode_s,
                    "n_cached": int(decisions["summary"]["n_cached"]),
                    "cache_ratio": float(decisions["summary"]["cache_ratio"]),
                    "image_file": out_path.name,
                    "decisions_file": dec_path.name,
                }
            )
            print(
                f"[flux-screen] mode={args.mode} idx={idx} "
                f"denoise={denoise_s:.2f}s decode={decode_s:.2f}s",
                flush=True,
            )
    finally:
        teardown()

    config = {
        "cache_mode": args.mode,
        "mode": args.mode,
        "backbone": "flux",
        "num_steps": args.num_steps,
        "cache_steps": list(cache_steps),
        "schedule_json": None if args.schedule_json is None else str(args.schedule_json),
        "threshold": args.threshold,
        "first_enhance": args.first_enhance,
        "interval": args.interval,
        "max_order": args.max_order,
        "hicache_sigma": args.hicache_sigma,
        "dicache_error_choice": args.dicache_error_choice,
        "dicache_ret_ratio": args.dicache_ret_ratio,
        "dicache_probe_depth": args.dicache_probe_depth,
        "l2p_weights": None if args.l2p_weights is None else str(args.l2p_weights),
        "foca_heun_variant": args.foca_heun_variant,
        "foca_history_policy": args.foca_history_policy,
        "foca_h": args.foca_h,
        "toca_fresh_ratio": args.toca_fresh_ratio,
        "toca_soft_fresh_weight": args.toca_soft_fresh_weight,
        "toca_fresh_threshold": args.toca_fresh_threshold,
        "dpcache_order": args.dpcache_order,
        "meancache_jvp_span": args.meancache_jvp_span,
        "width": args.width,
        "height": args.height,
        "guidance": args.guidance,
        "dtype": args.dtype,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "base_seed": args.seed,
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
    }
    if args.cache_count is not None:
        config["target_cache_count"] = args.cache_count
    write_timing_json(
        args.output_dir / f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json",
        per_image=per_image,
        config=config,
        model_load_s=load_s,
        wallclock_total_s=time.perf_counter() - wall_start + load_s,
        device=torch.cuda.get_device_name(0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
