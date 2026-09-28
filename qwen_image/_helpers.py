"""Shared helpers for the Qwen-Image runner.

This module intentionally mirrors the external contracts used by the FLUX and
Wan2.1 paths: flat `img_<idx>.png` outputs, `seed = base + idx`, per-shard
manifests, and per-prompt decision JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lib.io_utils import image_filename, read_prompts, seed_for, split_shard


DEFAULT_PROMPT_FILE = Path(
    "reference/hicache/code/models/qwen_image/prompts/DrawBench200.txt"
)


def decisions_filename(global_idx: int) -> str:
    return f"decisions_{int(global_idx):05d}.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_status_short(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_prompt_shard(
    prompt_file: Path,
    *,
    limit: int,
    shard_idx: int,
    shard_count: int,
) -> Tuple[List[str], int, int]:
    prompts = read_prompts(prompt_file, limit=limit if limit > 0 else None)
    start, end = split_shard(len(prompts), shard_count, shard_idx)
    return prompts[start:end], start, len(prompts)


def _scheduler_snapshot(pipe: Any) -> Dict[str, Any]:
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        return {"class": None, "config": None, "config_sha256": None}
    config = getattr(scheduler, "config", None)
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = dict(config)
    else:
        try:
            config_dict = dict(config)
        except Exception:
            config_dict = {"repr": repr(config)}
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return {
        "class": f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}",
        "config": config_dict,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
    }


def pipeline_identity(pipe: Any) -> Dict[str, Any]:
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = None
    transformer = getattr(pipe, "transformer", None)
    block = None
    if transformer is not None and getattr(transformer, "transformer_blocks", None):
        block = transformer.transformer_blocks[0]
    return {
        "diffusers_version": diffusers_version,
        "pipeline_class": f"{pipe.__class__.__module__}.{pipe.__class__.__name__}",
        "transformer_class": (
            None
            if transformer is None
            else f"{transformer.__class__.__module__}.{transformer.__class__.__name__}"
        ),
        "block_class": (
            None
            if block is None
            else f"{block.__class__.__module__}.{block.__class__.__name__}"
        ),
        "scheduler": _scheduler_snapshot(pipe),
    }


def build_manifest(
    args: Any,
    *,
    repo_root: Path,
    prompt_count: int,
    pipe: Any | None = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema": "qwen_image_manifest.v1",
        "backbone": "qwen_image",
        "git_sha": git_sha(repo_root),
        "git_status_short": git_status_short(repo_root),
        "hostname": socket.gethostname(),
        "mode": args.mode,
        "granularity": (
            "none"
            if args.mode == "original"
            else (
                "fine_240"
                if str(args.mode).endswith("_fine")
                else (
                    "guided_velocity"
                    if args.mode == "MeanCache"
                    else (
                        "final_output"
                        if args.mode == "L2P_output"
                        else (
                            "coarse_feature_forecast"
                            if args.mode == "DPCache"
                            else "coarse_residual"
                        )
                    )
                )
            )
        ),
        "model_id": args.model_id,
        "prompt_file": str(args.prompt_file),
        "prompt_file_sha256": sha256_file(Path(args.prompt_file)),
        "prompt_count_after_limit": int(prompt_count),
        "run_name": args.run_name,
        "width": int(args.width),
        "height": int(args.height),
        "num_steps": int(args.num_steps),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "seed_rule": "base_plus_prompt_idx",
        "true_cfg_scale": float(args.true_cfg_scale),
        "negative_prompt_sha256": sha256_text(args.negative_prompt),
        "negative_prompt_repr": repr(args.negative_prompt),
        "output_format": args.output_format,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "limit": int(args.limit),
        "interval": int(args.interval),
        "max_order": int(args.max_order),
        "hicache_sigma": float(args.hicache_sigma),
        "first_enhance": int(args.first_enhance),
        "seacache_thresh": float(getattr(args, "seacache_thresh", 0.3)),
        "teacache_thresh": float(getattr(args, "teacache_thresh", 0.3)),
        "teacache_coeff_source": str(
            getattr(args, "teacache_coeff_source", "identity")
        ),
        "teacache_coefficients": (
            None
            if getattr(args, "teacache_coefficients", None) is None
            else str(getattr(args, "teacache_coefficients"))
        ),
        "teacache_coefficients_sha256": (
            None
            if getattr(args, "teacache_coefficients", None) is None
            else sha256_file(Path(getattr(args, "teacache_coefficients")))
        ),
        "fixed_schedule_cache_count": getattr(args, "exact_cache_count", None),
        "schedule_file": (
            None
            if getattr(args, "schedule_file", None) is None
            else str(getattr(args, "schedule_file"))
        ),
        "schedule_file_sha256": (
            None
            if getattr(args, "schedule_file", None) is None
            else sha256_file(Path(getattr(args, "schedule_file")))
        ),
        "cache_steps": list(
            getattr(args, "cache_steps", ()) or getattr(args, "l2p_cache_steps", ())
        ),
        "l2p_weights": (
            None
            if getattr(args, "l2p_weights", None) is None
            else str(getattr(args, "l2p_weights"))
        ),
        "l2p_weights_sha256": (
            None
            if getattr(args, "l2p_weights", None) is None
            else sha256_file(Path(getattr(args, "l2p_weights")))
        ),
        "l2p_cache_steps": list(
            (getattr(args, "cache_steps", ()) or getattr(args, "l2p_cache_steps", ()))
            if getattr(args, "mode", None) == "L2P_output"
            else getattr(args, "l2p_cache_steps", ())
        ),
        "l2p_min_abs_weight": float(getattr(args, "l2p_min_abs_weight", 0.0)),
        "payload_mode": str(getattr(args, "payload_mode", "reuse")),
        "payload_sigma": float(getattr(args, "payload_sigma", 0.5)),
        "payload_schedule_dir": (
            None
            if getattr(args, "payload_schedule_dir", None) is None
            else str(getattr(args, "payload_schedule_dir"))
        ),
        "sencache_sensitivity": (
            None
            if getattr(args, "sencache_sensitivity", None) is None
            else str(getattr(args, "sencache_sensitivity"))
        ),
        "sencache_sensitivity_sha256": (
            None
            if getattr(args, "sencache_sensitivity", None) is None
            else sha256_file(Path(getattr(args, "sencache_sensitivity")))
        ),
        "sencache_threshold_start": float(
            getattr(args, "sencache_threshold_start", 0.005)
        ),
        "sencache_threshold_main": float(
            getattr(args, "sencache_threshold_main", 0.08)
        ),
        "sencache_threshold_scale": str(
            getattr(args, "sencache_threshold_scale", "auto")
        ),
        "sencache_switch_ratio": float(getattr(args, "sencache_switch_ratio", 0.2)),
        "sencache_max_skip": int(getattr(args, "sencache_max_skip", 10)),
        "sencache_ret_steps": int(getattr(args, "sencache_ret_steps", 0)),
        "sencache_cutoff_steps": int(getattr(args, "sencache_cutoff_steps", -1)),
        "dicache_thresh": float(getattr(args, "dicache_thresh", 0.12)),
        "dicache_error_choice": str(getattr(args, "dicache_error_choice", "delta_y")),
        "dicache_ret_ratio": float(getattr(args, "dicache_ret_ratio", 0.2)),
        "dicache_probe_depth": int(getattr(args, "dicache_probe_depth", 1)),
        "dpcache_order": int(getattr(args, "dpcache_order", 2)),
        "meancache_jvp_span": int(getattr(args, "meancache_jvp_span", 4)),
        "meancache_jvp_spans": (
            None
            if getattr(args, "meancache_jvp_spans", None) is None
            else str(getattr(args, "meancache_jvp_spans"))
        ),
        "meancache_jvp_spans_sha256": (
            None
            if getattr(args, "meancache_jvp_spans", None) is None
            else sha256_file(Path(getattr(args, "meancache_jvp_spans")))
        ),
        "appendix_compat": bool(args.appendix_compat),
        "fair_v1": bool(args.fair_v1),
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
    }
    if pipe is not None:
        manifest["pipeline"] = pipeline_identity(pipe)
    return manifest


__all__ = [
    "DEFAULT_PROMPT_FILE",
    "atomic_write_json",
    "build_manifest",
    "decisions_filename",
    "image_filename",
    "pipeline_identity",
    "read_prompt_shard",
    "seed_for",
    "sha256_file",
    "sha256_text",
]
