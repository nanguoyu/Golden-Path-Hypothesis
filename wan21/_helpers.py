"""Shared Wan2.1 runner helpers.

The Wan2.1 path intentionally stays separate from the locked FLUX baseline
library.  These helpers cover only prompt sharding, manifests, and schedule
locking for the Wan2.1 research runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lib.io_utils import read_prompts, split_shard


def video_filename(global_idx: int) -> str:
    return f"video_{int(global_idx):05d}.mp4"


def decisions_filename(global_idx: int) -> str:
    return f"decisions_{int(global_idx):05d}.json"


def seed_for(base_seed: int, global_idx: int) -> int:
    return int(base_seed) + int(global_idx)


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


def build_manifest(args: Any, *, repo_root: Path, prompt_count: int) -> Dict[str, Any]:
    size = f"{int(args.width)}*{int(args.height)}"
    return {
        "git_sha": git_sha(repo_root),
        "hostname": socket.gethostname(),
        "mode": args.mode,
        "task": args.task,
        "model": "Wan2.1-T2V-1.3B" if args.task == "t2v-1.3B" else args.task,
        "ckpt_dir": str(args.ckpt_dir),
        "wan_repo": str(args.wan_repo),
        "prompt_file": str(args.prompt_file),
        "prompt_file_sha256": sha256_file(Path(args.prompt_file)),
        "prompt_count_after_limit": int(prompt_count),
        "run_name": args.run_name,
        "size": size,
        "width": int(args.width),
        "height": int(args.height),
        "num_frames": int(args.num_frames),
        "num_steps": int(args.num_steps),
        "sample_solver": args.sample_solver,
        "sample_shift": float(args.sample_shift),
        "guidance_scale": float(args.guidance_scale),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "limit": int(args.limit),
        "seacache_thresh": float(args.seacache_thresh),
        "teacache_thresh": float(args.teacache_thresh),
        "teacache_variant": args.teacache_variant,
        "fresh_threshold": int(getattr(args, "fresh_threshold", 5)),
        "max_order": int(getattr(args, "max_order", 1)),
        "hicache_sigma": float(getattr(args, "hicache_sigma", 0.5)),
        "payload_mode": args.payload_mode,
        "payload_schedule_dir": (
            None if args.payload_schedule_dir is None else str(args.payload_schedule_dir)
        ),
        "segment_layout": getattr(args, "segment_layout", None),
        "payload_sigma": float(args.payload_sigma),
        "payload_blend": float(args.payload_blend),
        "require_locked_schedule": bool(args.require_locked_schedule),
        "seacache_power_exp": float(args.seacache_power_exp),
        "seacache_norm_mode": args.seacache_norm_mode,
        "first_enhance": int(args.first_enhance),
        "offload_model": bool(args.offload_model),
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def extract_action_map(decisions: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in decisions.get("steps", []):
        step = int(row["step"])
        branches = row.get("branches", {})
        step_actions: Dict[str, str] = {}
        for name in ("cond", "uncond"):
            entry = branches.get(name)
            if entry is None:
                continue
            action = entry.get("action")
            if action not in {"full", "cache"}:
                raise ValueError(f"bad action at step {step} branch {name}: {action!r}")
            step_actions[name] = action
        out[step] = step_actions
    return out


def load_locked_schedule(schedule_dir: Path, prompt_idx: int) -> Dict[int, Dict[str, str]]:
    candidates = [
        Path(schedule_dir) / decisions_filename(prompt_idx),
        Path(schedule_dir) / f"prompt_{prompt_idx:05d}" / "decisions.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        joined = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"missing locked schedule for prompt {prompt_idx}: {joined}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return extract_action_map(data)


def flatten_rows(decisions: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for row in decisions.get("steps", []):
        step = int(row.get("step", -1))
        for branch, entry in (row.get("branches") or {}).items():
            payload = dict(entry)
            payload["step"] = step
            payload["branch"] = branch
            yield payload
