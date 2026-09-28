#!/usr/bin/env python3
"""Single-GPU Qwen-Image runner for the local diffusion caching framework."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.io_utils import write_timing_json  # noqa: E402
from qwen_image._helpers import (  # noqa: E402
    DEFAULT_PROMPT_FILE,
    atomic_write_json,
    build_manifest,
    decisions_filename,
    image_filename,
    read_prompt_shard,
    seed_for,
    sha256_file,
)
from qwen_image.coarse_cache import (  # noqa: E402
    QwenCoarseConfig,
    install_qwen_coarse_forward,
    qwen_coarse_decisions,
    reset_qwen_coarse_state,
    restore_qwen_coarse_forward,
)
from qwen_image.dicache import (  # noqa: E402
    QwenDiCacheConfig,
    install_qwen_dicache,
    qwen_dicache_decisions,
    reset_qwen_dicache,
    restore_qwen_dicache,
)
from qwen_image.dpcache import (  # noqa: E402
    install_qwen_dpcache,
    qwen_dpcache_decisions,
    reset_qwen_dpcache,
    restore_qwen_dpcache,
)
from qwen_image.fine_scaffold import (  # noqa: E402
    QwenFineConfig,
    install_qwen_fine_forward,
    qwen_fine_decisions,
    reset_qwen_fine_state,
    restore_qwen_fine_forward,
)
from qwen_image.l2p import (  # noqa: E402
    QwenL2PConfig,
    install_qwen_l2p,
    qwen_l2p_decisions,
    reset_qwen_l2p,
    restore_qwen_l2p,
)
from qwen_image.meancache import (  # noqa: E402
    install_qwen_meancache,
    qwen_meancache_decisions,
    reset_qwen_meancache,
    restore_qwen_meancache,
)


def _step_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cache steps must be comma-separated integers"
        ) from error
    if steps != tuple(sorted(set(steps))):
        raise argparse.ArgumentTypeError("cache steps must be sorted and unique")
    return steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=[
            "original",
            "TaylorSeer_fine",
            "HiCache_fine",
            "SeaCache",
            "TeaCache",
            "SenCache",
            "DiCache",
            "SeaCachePayload",
            "L2P_output",
            "DPCache",
            "BudCache",
            "MeanCache",
        ],
        default="original",
    )
    p.add_argument(
        "--model_id",
        default="Qwen/Qwen-Image",
        help="HF repo id or local snapshot path.",
    )
    p.add_argument("--prompt_file", type=Path, default=DEFAULT_PROMPT_FILE)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--run_name", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--width", type=int, default=1328)
    p.add_argument("--height", type=int, default=1328)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--output_format", choices=["png", "jpg"], default="png")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--timing_only",
        action="store_true",
        help="Run generation and write timing shards without saving images or decision JSONs.",
    )
    p.add_argument(
        "--split_timing",
        action="store_true",
        help=(
            "Request latent output from the pipeline, then run the pipeline's "
            "native VAE decode path separately so denoise_s and decode_s are "
            "measured independently."
        ),
    )
    p.add_argument("--interval", type=int, default=7)
    p.add_argument("--max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--first_enhance", type=int, default=3)
    p.add_argument("--seacache_thresh", type=float, default=0.3)
    p.add_argument("--teacache_thresh", type=float, default=0.3)
    p.add_argument(
        "--exact_cache_count",
        type=int,
        default=None,
        help=(
            "Validate exactly K cache actions for an explicit fixed schedule. "
            "Dynamic gates do not accept this option."
        ),
    )
    p.add_argument(
        "--cache_steps",
        type=_step_list,
        default=(),
        help="Shared explicit cache-step schedule for fixed-schedule methods.",
    )
    p.add_argument(
        "--schedule_file",
        type=Path,
        default=None,
        help=(
            "JSON schedule containing cache_steps and optional cache_count/"
            "jvp_spans. Values must agree with any explicit CLI schedule options."
        ),
    )
    p.add_argument(
        "--teacache_coeff_source",
        choices=["identity", "flux_transfer", "qwen_fitted"],
        default="identity",
        help="Qwen has no registered TeaCache coeffs; identity is the auditable unfitted default.",
    )
    p.add_argument("--teacache_coefficients", type=Path, default=None)
    p.add_argument("--l2p_weights", type=Path, default=None)
    p.add_argument("--l2p_cache_steps", type=_step_list, default=())
    p.add_argument("--l2p_min_abs_weight", type=float, default=0.0)
    p.add_argument(
        "--payload_mode",
        choices=["reuse", "taylor_o1", "ensemble_mean"],
        default="reuse",
    )
    p.add_argument("--payload_sigma", type=float, default=0.5)
    p.add_argument("--payload_schedule_dir", type=Path, default=None)
    p.add_argument("--sencache_sensitivity", type=Path, default=None)
    p.add_argument("--sencache_threshold_start", type=float, default=0.005)
    p.add_argument("--sencache_threshold_main", type=float, default=0.08)
    p.add_argument("--sencache_threshold_scale", default="auto")
    p.add_argument("--sencache_switch_ratio", type=float, default=0.2)
    p.add_argument("--sencache_max_skip", type=int, default=10)
    p.add_argument("--sencache_ret_steps", type=int, default=0)
    p.add_argument("--sencache_cutoff_steps", type=int, default=-1)
    p.add_argument("--dicache_thresh", type=float, default=0.12)
    p.add_argument(
        "--dicache_error_choice",
        choices=["delta_y", "delta_minus"],
        default="delta_y",
    )
    p.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    p.add_argument("--dicache_probe_depth", type=int, default=1)
    p.add_argument("--dpcache_order", type=int, choices=[2], default=2)
    p.add_argument("--meancache_jvp_span", type=int, default=4)
    p.add_argument(
        "--meancache_jvp_spans",
        type=Path,
        default=None,
        help="Optional JSON mapping from cache step to MeanCache JVP span.",
    )
    p.add_argument("--appendix_compat", action="store_true")
    p.add_argument("--fair_v1", action="store_true")
    return p.parse_args()


def _normalize_schedule_file(args: argparse.Namespace) -> None:
    if getattr(args, "_schedule_file_normalized", False):
        return
    args._schedule_file_normalized = True
    args._schedule_jvp_spans = {}
    if args.schedule_file is None:
        return
    if not args.schedule_file.is_file():
        raise SystemExit(f"--schedule_file not found: {args.schedule_file}")
    try:
        payload = json.loads(args.schedule_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid --schedule_file: {args.schedule_file}") from error
    if not isinstance(payload, dict):
        raise SystemExit("--schedule_file must contain a JSON object")
    raw_steps = payload.get("cache_steps")
    if not isinstance(raw_steps, list):
        raise SystemExit("--schedule_file must contain a cache_steps list")
    try:
        steps = tuple(int(step) for step in raw_steps)
    except (TypeError, ValueError) as error:
        raise SystemExit("--schedule_file cache_steps must be integers") from error
    if steps != tuple(sorted(set(steps))):
        raise SystemExit("--schedule_file cache_steps must be sorted and unique")
    if args.cache_steps and tuple(args.cache_steps) != steps:
        raise SystemExit("--schedule_file and --cache_steps disagree")
    if args.l2p_cache_steps and tuple(args.l2p_cache_steps) != steps:
        raise SystemExit("--schedule_file and --l2p_cache_steps disagree")
    if payload.get("num_steps") is not None and int(payload["num_steps"]) != int(
        args.num_steps
    ):
        raise SystemExit("--schedule_file num_steps disagrees with --num_steps")
    cache_count = int(payload.get("cache_count", len(steps)))
    if cache_count != len(steps):
        raise SystemExit("--schedule_file cache_count disagrees with cache_steps")
    if args.exact_cache_count is not None and int(args.exact_cache_count) != cache_count:
        raise SystemExit(
            "--schedule_file cache_count disagrees with --exact_cache_count"
        )
    raw_spans = payload.get("jvp_spans", {})
    if raw_spans is not None and not isinstance(raw_spans, dict):
        raise SystemExit("--schedule_file jvp_spans must be a JSON object")
    try:
        spans = {
            int(step): int(span)
            for step, span in (raw_spans or {}).items()
        }
    except (TypeError, ValueError) as error:
        raise SystemExit("--schedule_file jvp_spans must map integers to integers") from error
    if args.meancache_jvp_spans is not None and spans:
        raise SystemExit(
            "MeanCache jvp_spans are present in both --schedule_file and "
            "--meancache_jvp_spans"
        )
    args.cache_steps = steps
    args.exact_cache_count = cache_count
    args._schedule_jvp_spans = spans


def _fixed_cache_steps(args: argparse.Namespace) -> tuple[int, ...]:
    _normalize_schedule_file(args)
    common = tuple(args.cache_steps)
    legacy_l2p = tuple(args.l2p_cache_steps)
    if common and legacy_l2p and common != legacy_l2p:
        raise SystemExit("--cache_steps and --l2p_cache_steps disagree")
    return common or legacy_l2p


def validate_args(args: argparse.Namespace) -> None:
    _normalize_schedule_file(args)
    if args.num_steps < 1:
        raise SystemExit("--num_steps must be positive")
    fine_mode = args.mode in {"TaylorSeer_fine", "HiCache_fine"}
    coarse_mode = args.mode in {
        "SeaCache",
        "TeaCache",
        "SenCache",
        "BudCache",
        "SeaCachePayload",
    }
    l2p_mode = args.mode == "L2P_output"
    required_fixed_mode = args.mode in {
        "L2P_output",
        "DPCache",
        "BudCache",
        "MeanCache",
    }
    optional_fixed_mode = args.mode in {"TaylorSeer_fine", "HiCache_fine"}
    if args.l2p_cache_steps and args.mode != "L2P_output":
        raise SystemExit(
            "--l2p_cache_steps is a legacy L2P-only alias; use --cache_steps"
        )
    fixed_steps = _fixed_cache_steps(args)
    schedule_spans = dict(getattr(args, "_schedule_jvp_spans", {}))
    if schedule_spans and args.mode != "MeanCache":
        raise SystemExit(
            "--schedule_file jvp_spans are only valid for MeanCache"
        )
    if schedule_spans and not set(schedule_spans).issubset(fixed_steps):
        raise SystemExit(
            "--schedule_file jvp_spans must refer only to cached steps"
        )
    locked_payload_mode = (
        args.mode == "SeaCachePayload" and args.payload_schedule_dir is not None
    )
    fixed_reuse_payload_mode = (
        args.mode == "SeaCachePayload"
        and args.payload_mode == "reuse"
        and bool(fixed_steps)
    )
    if args.exact_cache_count is not None:
        if not (
            required_fixed_mode
            or (optional_fixed_mode and fixed_steps)
            or locked_payload_mode
            or fixed_reuse_payload_mode
        ):
            raise SystemExit(
                "--exact_cache_count is only valid with an explicit fixed schedule"
            )
        if not 0 <= int(args.exact_cache_count) < int(args.num_steps):
            raise SystemExit(
                f"--exact_cache_count must be in [0, {int(args.num_steps) - 1}]"
            )
    if fixed_steps:
        if args.mode not in {
            "TaylorSeer_fine",
            "HiCache_fine",
            "L2P_output",
            "DPCache",
            "BudCache",
            "MeanCache",
            "SeaCachePayload",
        }:
            raise SystemExit("--cache_steps is not valid for a dynamic gate")
        if args.mode == "SeaCachePayload" and args.payload_mode != "reuse":
            raise SystemExit(
                "SeaCachePayload fixed cache_steps are only valid for reuse"
            )
        if any(step < 0 or step >= int(args.num_steps) for step in fixed_steps):
            raise SystemExit("--cache_steps contains a step outside the trajectory")
        if args.exact_cache_count is None:
            raise SystemExit("an explicit fixed schedule requires --exact_cache_count")
        if len(fixed_steps) != int(args.exact_cache_count):
            raise SystemExit("cache-step count does not match --exact_cache_count")
    elif required_fixed_mode:
        raise SystemExit(f"{args.mode} requires --cache_steps")
    if fine_mode:
        if int(args.interval) < 2:
            raise SystemExit("fine cache modes require --interval >= 2")
        if int(args.first_enhance) < 1:
            raise SystemExit("fine cache modes require --first_enhance >= 1")
        if int(args.max_order) < 0:
            raise SystemExit("fine cache modes require --max_order >= 0")
        if args.mode == "HiCache_fine" and not (0.0 < float(args.hicache_sigma) <= 1.0):
            raise SystemExit("HiCache_fine requires 0 < --hicache_sigma <= 1")
    if coarse_mode:
        if int(args.first_enhance) < 1:
            raise SystemExit("coarse cache modes require --first_enhance >= 1")
        if (
            args.mode == "SeaCachePayload"
            and args.payload_mode != "reuse"
            and args.payload_schedule_dir is None
        ):
            raise SystemExit(
                "Qwen SeaCachePayload forecast modes require --payload_schedule_dir"
            )
        if (
            args.payload_schedule_dir is not None
            and not args.payload_schedule_dir.is_dir()
        ):
            raise SystemExit(
                f"--payload_schedule_dir not found: {args.payload_schedule_dir}"
            )
        if args.teacache_coeff_source == "qwen_fitted":
            if (
                args.teacache_coefficients is None
                or not args.teacache_coefficients.is_file()
            ):
                raise SystemExit(
                    "qwen_fitted TeaCache requires --teacache_coefficients JSON"
                )
        if args.mode == "SenCache":
            if (
                args.sencache_sensitivity is None
                or not args.sencache_sensitivity.is_file()
            ):
                raise SystemExit("SenCache requires --sencache_sensitivity")
            if not 0.0 <= float(args.sencache_switch_ratio) <= 1.0:
                raise SystemExit("--sencache_switch_ratio must be in [0,1]")
            if int(args.sencache_max_skip) < 1:
                raise SystemExit("--sencache_max_skip must be positive")
            if str(args.sencache_threshold_scale).strip().lower() != "auto":
                try:
                    float(args.sencache_threshold_scale)
                except ValueError as error:
                    raise SystemExit(
                        "--sencache_threshold_scale must be 'auto' or a number"
                    ) from error
    if args.mode == "DiCache":
        if float(args.dicache_thresh) <= 0.0:
            raise SystemExit("--dicache_thresh must be positive")
        if not 0.0 <= float(args.dicache_ret_ratio) < 1.0:
            raise SystemExit("--dicache_ret_ratio must be in [0,1)")
        if int(args.dicache_probe_depth) < 1:
            raise SystemExit("--dicache_probe_depth must be positive")
    if args.mode == "MeanCache":
        if int(args.meancache_jvp_span) < 1:
            raise SystemExit("--meancache_jvp_span must be positive")
        if (
            args.meancache_jvp_spans is not None
            and not args.meancache_jvp_spans.is_file()
        ):
            raise SystemExit(
                f"--meancache_jvp_spans not found: {args.meancache_jvp_spans}"
            )
    if l2p_mode:
        if args.l2p_weights is None or not args.l2p_weights.is_file():
            raise SystemExit("L2P_output requires an existing --l2p_weights file")
        if 0 in fixed_steps:
            raise SystemExit("L2P_output cannot cache step 0")
    if args.true_cfg_scale <= 1.0 or args.negative_prompt is None:
        raise SystemExit(
            "Qwen-Image protocol requires true CFG: --true_cfg_scale > 1 and --negative_prompt"
        )
    if args.appendix_compat:
        if (int(args.width), int(args.height), int(args.num_steps)) != (1328, 1328, 50):
            raise SystemExit("Appendix-Compat requires 1328x1328 and 50 steps")
        if int(args.first_enhance) != 3:
            raise SystemExit("Appendix-Compat requires --first_enhance 3")


def _image_path(output_dir: Path, idx: int, output_format: str) -> Path:
    if output_format == "png":
        return output_dir / image_filename(idx)
    return output_dir / f"img_{int(idx)}.jpg"


def _image_complete(path: Path) -> bool:
    return Path(path).is_file() and Path(path).stat().st_size > 1024


def _save_image(img: Any, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.stem}.tmp.{Path(path).suffix.lstrip('.')}.{time.time_ns()}{path.suffix}"
    )
    if output_format == "png":
        img.save(tmp)
    else:
        img.save(tmp, quality=95, subsampling=0)
    tmp.replace(path)


def _validate_fixed_cache_count(
    args: argparse.Namespace,
    decisions: dict[str, Any],
    *,
    prompt_idx: int,
) -> None:
    if args.exact_cache_count is None:
        return
    actual = int(decisions.get("summary", {}).get("n_cached", -1))
    expected = int(args.exact_cache_count)
    if actual != expected:
        raise RuntimeError(
            f"fixed schedule cache count mismatch for prompt {prompt_idx}: "
            f"{actual} != {expected}"
        )


def _mode_label(args: argparse.Namespace) -> str:
    fixed_k = args.exact_cache_count
    if args.mode == "original":
        return f"original_s{args.seed}_{args.num_steps}"
    if args.mode in {"TaylorSeer_fine", "HiCache_fine"} and _fixed_cache_steps(args):
        name = "taylorseer_fine" if args.mode == "TaylorSeer_fine" else "hicache_fine"
        return f"{name}_exactK{fixed_k}_s{args.seed}_{args.num_steps}"
    if args.mode == "TaylorSeer_fine":
        return f"taylorseer_fine_i{args.interval}_o{args.max_order}_fe{args.first_enhance}_s{args.seed}_{args.num_steps}"
    if args.mode == "HiCache_fine":
        return (
            f"hicache_fine_i{args.interval}_o{args.max_order}_sig{args.hicache_sigma}"
            f"_fe{args.first_enhance}_s{args.seed}_{args.num_steps}"
        )
    if args.mode == "SeaCache":
        return (
            f"seacache_t{args.seacache_thresh}_fe{args.first_enhance}"
            f"_s{args.seed}_{args.num_steps}"
        )
    if args.mode == "TeaCache":
        return (
            f"teacache_{args.teacache_coeff_source}_t{args.teacache_thresh}"
            f"_fe{args.first_enhance}_s{args.seed}_{args.num_steps}"
        )
    if args.mode == "SenCache":
        return (
            f"sencache_ts{args.sencache_threshold_start}"
            f"_tm{args.sencache_threshold_main}_s{args.seed}_{args.num_steps}"
        )
    if args.mode == "DiCache":
        return (
            f"dicache_t{args.dicache_thresh}_{args.dicache_error_choice}"
            f"_s{args.seed}_{args.num_steps}"
        )
    if args.mode == "L2P_output":
        assert args.l2p_weights is not None
        return (
            f"l2p_output_exactK{args.exact_cache_count}"
            f"_w{args.l2p_weights.stem}_s{args.seed}_{args.num_steps}"
        )
    if args.mode in {"DPCache", "BudCache", "MeanCache"}:
        return f"{args.mode.lower()}_exactK{fixed_k}_s{args.seed}_{args.num_steps}"
    sched = "locked" if args.payload_schedule_dir is not None else "native"
    exact = "" if args.exact_cache_count is None else f"_exactK{args.exact_cache_count}"
    return (
        f"seacache_payload_{args.payload_mode}_{sched}_t{args.seacache_thresh}"
        f"_sig{args.payload_sigma}_fe{args.first_enhance}{exact}_s{args.seed}_{args.num_steps}"
    )


def _load_pipeline(args: argparse.Namespace, torch_dtype: Any, device: str) -> Any:
    try:
        from diffusers import QwenImagePipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "diffusers.QwenImagePipeline is not importable. "
            "Use the cluster cache env with a Qwen-Image capable diffusers version."
        ) from exc
    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    return pipe.to(device)


def _decode_latents_to_pil(pipe: Any, latents: Any, height: int, width: int) -> list[Any]:
    import torch

    latents = pipe._unpack_latents(
        latents,
        int(height),
        int(width),
        pipe.vae_scale_factor,
    )
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")


def _tea_coefficients(args: argparse.Namespace) -> tuple[float, ...] | None:
    if args.teacache_coeff_source != "qwen_fitted":
        return None
    assert args.teacache_coefficients is not None
    payload = json.loads(args.teacache_coefficients.read_text(encoding="utf-8"))
    values = payload.get("coefficients")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(
            "Qwen TeaCache coefficient file must contain five coefficients"
        )
    return tuple(float(value) for value in values)


def _sencache_threshold_scale(args: argparse.Namespace) -> str | float:
    value = str(args.sencache_threshold_scale).strip()
    return "auto" if value.lower() == "auto" else float(value)


def _meancache_jvp_spans(args: argparse.Namespace) -> dict[int, int]:
    spans = dict(getattr(args, "_schedule_jvp_spans", {}))
    if args.meancache_jvp_spans is not None:
        payload = json.loads(args.meancache_jvp_spans.read_text(encoding="utf-8"))
        values = payload.get("jvp_spans", payload)
        if not isinstance(values, dict):
            raise ValueError("MeanCache JVP span file must contain a JSON object")
        spans = {int(step): int(span) for step, span in values.items()}
    if any(step < 0 or step >= int(args.num_steps) for step in spans):
        raise ValueError("MeanCache JVP span file contains an out-of-range step")
    if any(span < 1 for span in spans.values()):
        raise ValueError("MeanCache JVP spans must be positive")
    return spans


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not args.run_name:
        args.run_name = _mode_label(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts, start_idx, prompt_count = read_prompt_shard(
        args.prompt_file,
        limit=args.limit,
        shard_idx=args.shard_idx,
        shard_count=args.shard_count,
    )
    if not prompts:
        print(
            f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.",
            flush=True,
        )
        return 0

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] loading {args.model_id} "
        f"dtype={args.dtype} mode={args.mode}",
        flush=True,
    )
    process_start = time.perf_counter()
    model_load_start = time.perf_counter()
    pipe = _load_pipeline(args, torch_dtype, device)
    model_load_s = time.perf_counter() - model_load_start

    restore_qwen_coarse_forward(pipe)
    restore_qwen_fine_forward(pipe)
    restore_qwen_l2p(pipe)
    restore_qwen_dicache(pipe)
    restore_qwen_dpcache(pipe)
    restore_qwen_meancache(pipe)

    fixed_steps = _fixed_cache_steps(args)
    if args.mode in {"TaylorSeer_fine", "HiCache_fine"}:
        install_qwen_fine_forward(
            pipe,
            QwenFineConfig(
                mode=args.mode,
                num_steps=args.num_steps,
                interval=args.interval,
                first_enhance=args.first_enhance,
                max_order=args.max_order,
                hicache_sigma=args.hicache_sigma,
                true_cfg=True,
                cache_steps=fixed_steps,
            ),
        )
    elif args.mode == "L2P_output":
        assert args.l2p_weights is not None
        install_qwen_l2p(
            pipe,
            QwenL2PConfig(
                weights_path=args.l2p_weights,
                cache_steps=fixed_steps,
                num_steps=args.num_steps,
                min_abs_weight=args.l2p_min_abs_weight,
                true_cfg=True,
            ),
        )
    elif args.mode == "DiCache":
        install_qwen_dicache(
            pipe,
            QwenDiCacheConfig(
                num_steps=args.num_steps,
                threshold=args.dicache_thresh,
                error_choice=args.dicache_error_choice,
                ret_ratio=args.dicache_ret_ratio,
                probe_depth=args.dicache_probe_depth,
                true_cfg=True,
            ),
        )
    elif args.mode == "DPCache":
        install_qwen_dpcache(
            pipe,
            cache_steps=fixed_steps,
            num_steps=args.num_steps,
            order=args.dpcache_order,
            true_cfg=True,
        )
    elif args.mode == "MeanCache":
        install_qwen_meancache(
            pipe,
            cache_steps=fixed_steps,
            num_steps=args.num_steps,
            jvp_span=args.meancache_jvp_span,
            jvp_spans=_meancache_jvp_spans(args),
            true_cfg=True,
            true_cfg_scale=args.true_cfg_scale,
        )
    elif args.mode in {
        "SeaCache",
        "TeaCache",
        "SenCache",
        "BudCache",
        "SeaCachePayload",
    }:
        install_qwen_coarse_forward(
            pipe,
            QwenCoarseConfig(
                mode=args.mode,
                num_steps=args.num_steps,
                first_enhance=args.first_enhance,
                seacache_thresh=args.seacache_thresh,
                teacache_thresh=args.teacache_thresh,
                teacache_coeff_source=args.teacache_coeff_source,
                teacache_coefficients=_tea_coefficients(args),
                payload_mode=args.payload_mode,
                payload_sigma=args.payload_sigma,
                payload_schedule_dir=(
                    str(args.payload_schedule_dir)
                    if args.payload_schedule_dir is not None
                    else None
                ),
                fixed_cache_steps=fixed_steps,
                sencache_sensitivity_path=(
                    None
                    if args.sencache_sensitivity is None
                    else str(args.sencache_sensitivity)
                ),
                sencache_threshold_start=args.sencache_threshold_start,
                sencache_threshold_main=args.sencache_threshold_main,
                sencache_threshold_scale=_sencache_threshold_scale(args),
                sencache_switch_ratio=args.sencache_switch_ratio,
                sencache_max_skip=args.sencache_max_skip,
                sencache_ret_steps=args.sencache_ret_steps,
                sencache_cutoff_steps=args.sencache_cutoff_steps,
                true_cfg=True,
            ),
        )

    manifest = build_manifest(
        args, repo_root=_PROJECT_ROOT, prompt_count=prompt_count, pipe=pipe
    )
    atomic_write_json(
        args.output_dir
        / f"manifest_shard{args.shard_idx:03d}of{args.shard_count:03d}.json",
        manifest,
    )
    if int(args.shard_idx) == 0:
        atomic_write_json(args.output_dir / "manifest.json", manifest)

    per_image = []
    wall_start = time.perf_counter()
    for local_i, prompt in enumerate(prompts):
        global_idx = start_idx + local_i
        seed = seed_for(args.seed, global_idx)
        img_path = _image_path(args.output_dir, global_idx, args.output_format)
        dec_path = args.output_dir / decisions_filename(global_idx)
        if (
            args.resume
            and _image_complete(img_path)
            and (args.mode == "original" or dec_path.is_file())
        ):
            print(f"[resume] skip idx={global_idx}", flush=True)
            continue

        if args.mode in {"TaylorSeer_fine", "HiCache_fine"}:
            reset_qwen_fine_state(pipe, prompt_idx=global_idx, seed=seed)
        elif args.mode == "L2P_output":
            reset_qwen_l2p(pipe, prompt_idx=global_idx, seed=seed)
        elif args.mode == "DiCache":
            reset_qwen_dicache(pipe, prompt_idx=global_idx, seed=seed)
        elif args.mode == "DPCache":
            reset_qwen_dpcache(pipe, prompt_idx=global_idx, seed=seed)
        elif args.mode == "MeanCache":
            reset_qwen_meancache(pipe, prompt_idx=global_idx, seed=seed)
        elif args.mode in {
            "SeaCache",
            "TeaCache",
            "SenCache",
            "BudCache",
            "SeaCachePayload",
        }:
            reset_qwen_coarse_state(pipe, prompt_idx=global_idx, seed=seed)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "true_cfg_scale": float(args.true_cfg_scale),
            "height": int(args.height),
            "width": int(args.width),
            "num_inference_steps": int(args.num_steps),
            "generator": generator,
            "return_dict": True,
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_start = time.perf_counter()
        result = pipe(
            **call_kwargs,
            output_type="latent" if args.split_timing else "pil",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_s = time.perf_counter() - denoise_start
        images = getattr(result, "images", None)
        decode_s = 0.0
        if args.split_timing:
            decode_start = time.perf_counter()
            images = _decode_latents_to_pil(
                pipe,
                images,
                int(args.height),
                int(args.width),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_s = time.perf_counter() - decode_start
        total_s = denoise_s + decode_s
        if not images:
            raise RuntimeError(
                f"Qwen-Image pipeline returned no images for idx={global_idx}"
            )
        if not args.timing_only:
            _save_image(images[0], img_path, args.output_format)

        if args.mode != "original":
            if args.mode in {"TaylorSeer_fine", "HiCache_fine"}:
                decisions = qwen_fine_decisions(pipe)
            elif args.mode == "L2P_output":
                decisions = qwen_l2p_decisions(pipe)
            elif args.mode == "DiCache":
                decisions = qwen_dicache_decisions(pipe)
            elif args.mode == "DPCache":
                decisions = qwen_dpcache_decisions(pipe)
            elif args.mode == "MeanCache":
                decisions = qwen_meancache_decisions(pipe)
            else:
                decisions = qwen_coarse_decisions(pipe)
            _validate_fixed_cache_count(args, decisions, prompt_idx=global_idx)
            if not args.timing_only:
                decisions.update(
                    {
                        "prompt": prompt,
                        "image_file": img_path.name,
                        "width": int(args.width),
                        "height": int(args.height),
                        "true_cfg_scale": float(args.true_cfg_scale),
                        "negative_prompt_sha256": manifest["negative_prompt_sha256"],
                    }
                )
                atomic_write_json(dec_path, decisions)

        per_image.append(
            {
                "idx": int(global_idx),
                "prompt": prompt,
                "seed": int(seed),
                "denoise_s": float(denoise_s),
                "decode_s": float(decode_s),
                "image_file": None if args.timing_only else img_path.name,
                "decisions_file": (
                    None
                    if args.timing_only or args.mode == "original"
                    else dec_path.name
                ),
                "timing_only": bool(args.timing_only),
            }
        )
        print(
            f"[qwen_image] idx={global_idx} seed={seed} total_s={total_s:.2f}",
            flush=True,
        )

    timing_config = {
        "cache_mode": args.mode,
        "mode": args.mode,
        "backbone": "qwen_image",
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "true_cfg_scale": float(args.true_cfg_scale),
        "dtype": args.dtype,
        "base_seed": int(args.seed),
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "model_id": args.model_id,
        "interval": int(args.interval),
        "max_order": int(args.max_order),
        "hicache_sigma": float(args.hicache_sigma),
        "first_enhance": int(args.first_enhance),
        "seacache_thresh": float(args.seacache_thresh),
        "teacache_thresh": float(args.teacache_thresh),
        "teacache_coeff_source": str(args.teacache_coeff_source),
        "teacache_coefficients": (
            None
            if args.teacache_coefficients is None
            else str(args.teacache_coefficients)
        ),
        "fixed_schedule_cache_count": args.exact_cache_count,
        "schedule_file": (
            None if args.schedule_file is None else str(args.schedule_file)
        ),
        "schedule_file_sha256": (
            None if args.schedule_file is None else sha256_file(args.schedule_file)
        ),
        "cache_steps": list(fixed_steps),
        "l2p_weights": None if args.l2p_weights is None else str(args.l2p_weights),
        "l2p_cache_steps": list(fixed_steps) if args.mode == "L2P_output" else [],
        "l2p_min_abs_weight": float(args.l2p_min_abs_weight),
        "payload_mode": str(args.payload_mode),
        "payload_sigma": float(args.payload_sigma),
        "payload_schedule_dir": (
            str(args.payload_schedule_dir)
            if args.payload_schedule_dir is not None
            else None
        ),
        "sencache_sensitivity": (
            None
            if args.sencache_sensitivity is None
            else str(args.sencache_sensitivity)
        ),
        "sencache_threshold_start": float(args.sencache_threshold_start),
        "sencache_threshold_main": float(args.sencache_threshold_main),
        "sencache_threshold_scale": str(args.sencache_threshold_scale),
        "sencache_switch_ratio": float(args.sencache_switch_ratio),
        "sencache_max_skip": int(args.sencache_max_skip),
        "sencache_ret_steps": int(args.sencache_ret_steps),
        "sencache_cutoff_steps": int(args.sencache_cutoff_steps),
        "dicache_thresh": float(args.dicache_thresh),
        "dicache_error_choice": str(args.dicache_error_choice),
        "dicache_ret_ratio": float(args.dicache_ret_ratio),
        "dicache_probe_depth": int(args.dicache_probe_depth),
        "dpcache_order": int(args.dpcache_order),
        "meancache_jvp_span": int(args.meancache_jvp_span),
        "meancache_jvp_spans": (
            None if args.meancache_jvp_spans is None else str(args.meancache_jvp_spans)
        ),
        "timing_only": bool(args.timing_only),
        "split_timing": bool(args.split_timing),
        "timing_scope": (
            "pipeline_latent_then_native_vae_decode"
            if args.split_timing
            else "monolithic_pipeline_call_stored_as_denoise_s"
        ),
        "output_format": args.output_format,
        "git_sha": manifest["git_sha"],
    }
    timing_path = (
        args.output_dir
        / f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json"
    )
    if args.resume and not per_image and timing_path.is_file():
        print(
            f"[qwen_image] resume preserved existing timing shard: {timing_path}",
            flush=True,
        )
        return 0

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    write_timing_json(
        timing_path,
        per_image=per_image,
        config=timing_config,
        model_load_s=model_load_s,
        wallclock_total_s=time.perf_counter() - wall_start + model_load_s,
        device=device_name,
    )
    print(
        f"[qwen_image] done mode={args.mode} n={len(per_image)} "
        f"wall_s={time.perf_counter() - process_start:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
