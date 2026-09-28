"""Output naming, prompt I/O, and `timing.json` writer used by all runners.

Conventions (kept compatible with `evaluation/eval_metrics.py` and the existing
`RUN/multi_gpu_launcher.py` merge step):

  - Filename per image: `img_<global_idx>.png` (no zero padding; preserved from
    the original BFL pipeline so existing eval scripts pair across runners).
  - Per-image seed: `base_seed + global_idx`.
  - `timing.json` schema:
      {
        "device":              str,
        "cache_mode":          str,
        "num_steps":           int,
        "n_images":            int,
        "shard_idx":           int,
        "shard_count":         int,
        "model_load_s":        float,
        "wallclock_total_s":   float,
        "denoise_per_image_s": {"mean", "std", "min", "max"},
        "decode_per_image_s":  {"mean", "std", "min", "max"},
        "latency_per_image_s": {"mean", "std", "min", "max"},
        "throughput_img_per_s": float,
        "per_image": [{"idx": int, "denoise_s": float, "decode_s": float}, ...],
        ...any extra config keys callers want to attach...
      }
    `eval_metrics.py:_latency_mean` reads `latency_per_image_s.mean` for the
    speedup field; the rest is informational.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ----- naming ----------------------------------------------------------------


def image_filename(global_idx: int) -> str:
    """Canonical per-image filename. Matches the existing BFL pipeline so the
    diffusers and BFL runners produce pair-able outputs."""
    return f"img_{int(global_idx)}.png"


def seed_for(base_seed: int, global_idx: int) -> int:
    return int(base_seed) + int(global_idx)


# ----- prompts ---------------------------------------------------------------


def read_prompts(path: Path, limit: Optional[int] = None) -> List[str]:
    """Read newline-separated prompts; strip blanks and `#` comments. `limit`
    is applied AFTER stripping, BEFORE sharding (callers shard themselves)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    if limit is not None and limit > 0:
        prompts = prompts[: int(limit)]
    return prompts


# ----- sharding --------------------------------------------------------------


def split_shard(total: int, shard_count: int, shard_idx: int) -> tuple[int, int]:
    """Match `RUN/multi_gpu_launcher.py:split_prompts` exactly.

    For 199 prompts across 4 shards: [0,50), [50,100), [100,150), [150,199).
    Crucially, `seed = base_seed + start + i` lines up across shards and across
    runners so paired evaluation is well-defined.
    """
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not (0 <= shard_idx < shard_count):
        raise ValueError(f"shard_idx {shard_idx} out of range [0, {shard_count})")
    base, remainder = divmod(total, shard_count)
    start = shard_idx * base + min(shard_idx, remainder)
    size = base + (1 if shard_idx < remainder else 0)
    return start, start + size


# ----- timing.json -----------------------------------------------------------


def _stats(xs: Iterable[float]) -> Dict[str, float]:
    xs = list(xs)
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"mean": mean, "std": math.sqrt(var), "min": min(xs), "max": max(xs)}


def carry_forward_timing_records(
    path: Path, per_image: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Union a resumed shard's fresh records with the ones already on disk.

    A `--resume` shard only regenerates the (prompt, seed) pairs missing from
    disk, so `per_image` holds just those. Overwriting the shard timing file
    with that list alone drops every record written by the previous attempt,
    which leaves the merged `timing.json` permanently short of the prompt count
    (and `RUN/merge_shard_timings.py --expected_records` failing forever) even
    though all images are on disk. Records already in `path` are kept for the
    indices this run skipped; fresh records win on collision. Missing or
    unreadable files are treated as "nothing to carry forward".

    Only the per-image records are recovered. `wallclock_total_s` of a resumed
    shard still covers the resumed portion only, so a resumed cell's throughput
    field is not comparable; the per-image latencies are exact.
    """
    fresh = {int(record["idx"]): record for record in per_image}
    kept: Dict[int, Dict[str, Any]] = {}
    try:
        previous = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}
    if isinstance(previous, dict):
        for record in previous.get("per_image") or []:
            try:
                idx = int(record["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx not in fresh:
                kept[idx] = record
    merged = {**kept, **fresh}
    return [merged[idx] for idx in sorted(merged)]


def write_plane_frame(path: Path, frame: Any) -> None:
    """Store one trajectory's orthonormal frame as float16 `.npy`, atomically.

    Callers write the frame *before* the JSON record whose skip-if-exists guard
    controls resume, so a kill between the two re-runs the generation rather
    than leaving a record that points at a missing file.

    float16 costs ~4e-4 relative error per component, which on a
    2.6e5-dimensional unit vector moves an inner product by ~1e-6 and halves a
    store that runs to gigabytes. That is 1.2e-2 degrees of principal angle in
    the worst case (measured, float64 frame vs its own round-trip), but only
    ~1e-4 degrees where the answers actually sit: near 90 degrees the arccos is
    flat, and over the 435 real trajectory pairs the round-trip moves no angle
    by more than 1.3e-4 degrees. Near-parallel subspaces are the regime to be
    careful in. Readers re-orthonormalize
    (`analysis.trajectory_math.principal_angles_deg` does).
    """
    import numpy as np

    array = np.asarray(frame, dtype=np.float16)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}.npy")
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(target)


def write_timing_json(
    path: Path,
    *,
    per_image: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_load_s: float,
    wallclock_total_s: float,
    device: str,
) -> None:
    """Serialize a per-shard `timing.json` matching `evaluation/eval_metrics.py`.

    `per_image[i]` must be `{"idx": int, "denoise_s": float, "decode_s": float}`.
    `config` is merged into the top-level dict (`cache_mode`, `num_steps`,
    `seacache_thresh`, etc.). `wallclock_total_s` includes model load.
    """
    denoise = [float(r["denoise_s"]) for r in per_image]
    decode = [float(r["decode_s"]) for r in per_image]
    latency = [d + e for d, e in zip(denoise, decode)]
    payload: Dict[str, Any] = {
        "device": device,
        "n_images": len(per_image),
        "model_load_s": float(model_load_s),
        "wallclock_total_s": float(wallclock_total_s),
        "denoise_per_image_s": _stats(denoise),
        "decode_per_image_s": _stats(decode),
        "latency_per_image_s": _stats(latency),
        "throughput_img_per_s": (
            len(per_image) / wallclock_total_s if wallclock_total_s > 0 else 0.0
        ),
        "per_image": per_image,
    }
    # Caller's config keys override defaults; we keep `latency_per_image_s` etc.
    # final by writing them AFTER the merge.
    merged = {**dict(config), **payload}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # tmp + replace, like the image/decision writers: a kill mid-write used to
    # leave truncated JSON that `carry_forward_timing_records` reads as "no
    # records to carry forward", silently losing a resumed shard's history.
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
