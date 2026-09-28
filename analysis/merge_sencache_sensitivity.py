#!/usr/bin/env python3
"""Merge backbone-specific SenCache sensitivity rows into a frozen npz table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SenCache sensitivity calibration rows.")
    p.add_argument("--acc", type=Path, required=True,
                   help="Directory containing sencache_sensitivity_rows_shard*of*.csv")
    p.add_argument("--out", type=Path, required=True,
                   help="Output npz path.")
    p.add_argument("--aggregation", choices=["mean", "median", "q75", "q90"], default="q90")
    p.add_argument(
        "--backbone",
        choices=["flux", "qwen_image", "hunyuan_video", "wan21"],
        # required rather than defaulted: a HunyuanVideo table written under the
        # flux schema loads without complaint (lib/sencache.load_sensitivity_table
        # does not check the tag) and is silently the wrong table, while the
        # matrix config freezes its digest, not its provenance
        required=True,
    )
    p.add_argument("--prompt_file", type=Path, default=None)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--model_id", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _agg(vals: List[float], name: str) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(arr))
    if name == "median":
        return float(np.median(arr))
    if name == "q75":
        return float(np.quantile(arr, 0.75))
    if name == "q90":
        return float(np.quantile(arr, 0.90))
    raise ValueError(name)


def _git_commit() -> tuple[Optional[str], Optional[bool]]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main() -> int:
    args = parse_args()
    paths = sorted(args.acc.glob("sencache_sensitivity_rows_shard*of*.csv"))
    if not paths:
        raise SystemExit(f"no sencache_sensitivity_rows_shard*of*.csv files under {args.acc}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_csv(path))
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(float(row["step_index"])), []).append(row)
    missing = sorted(set(range(int(args.num_steps))) - set(by_step))
    if missing:
        raise SystemExit(f"missing sensitivity rows for steps: {missing[:20]}")

    timesteps: List[float] = []
    jx: List[float] = []
    jt: List[float] = []
    jx_mean: List[float] = []
    jt_mean: List[float] = []
    jx_median: List[float] = []
    jt_median: List[float] = []
    jx_q90: List[float] = []
    jt_q90: List[float] = []
    counts: List[int] = []
    for step in range(int(args.num_steps)):
        group = by_step[step]
        t_vals = [x for x in (_float(r.get("timestep")) for r in group) if x is not None]
        x_vals = [x for x in (_float(r.get("J_x_directional")) for r in group) if x is not None]
        t_sens_vals = [x for x in (_float(r.get("J_t_directional")) for r in group) if x is not None]
        if not t_vals or not x_vals or not t_sens_vals:
            raise SystemExit(f"step {step} has incomplete finite sensitivity rows")
        timesteps.append(float(np.median(np.asarray(t_vals, dtype=float))))
        jx.append(_agg(x_vals, args.aggregation))
        jt.append(_agg(t_sens_vals, args.aggregation))
        jx_mean.append(_agg(x_vals, "mean"))
        jt_mean.append(_agg(t_sens_vals, "mean"))
        jx_median.append(_agg(x_vals, "median"))
        jt_median.append(_agg(t_sens_vals, "median"))
        jx_q90.append(_agg(x_vals, "q90"))
        jt_q90.append(_agg(t_sens_vals, "q90"))
        counts.append(min(len(x_vals), len(t_sens_vals)))

    git_sha, git_dirty = _git_commit()
    latent_numels = sorted({
        int(float(r["latent_numel"]))
        for r in rows
        if r.get("latent_numel") not in (None, "")
    })
    latent_shapes = sorted({
        str(r["latent_shape"])
        for r in rows
        if r.get("latent_shape") not in (None, "")
    })
    if args.backbone == "flux":
        notes = [
            "FLUX directional finite-difference sensitivity on packed latent tokens.",
            "Timesteps use FluxTransformer2DModel.forward units after its *1000 conversion.",
            "The frozen table is read-only during native online gating.",
        ]
    elif args.backbone == "hunyuan_video":
        notes = [
            "HunyuanVideo directional finite-difference sensitivity on the latent "
            "the transformer receives before img_in.",
            "Timesteps use the Tencent transformer's native range(0, 1000) units.",
            "The frozen table is read-only during native online gating.",
        ]
    elif args.backbone == "wan21":
        notes = [
            "Wan2.1 t2v-1.3B directional finite-difference sensitivity on the raw "
            "latent WanModel.forward receives before patch_embedding.",
            "Timesteps use WanModel.forward's native 0-1000 units, unscaled, which "
            "is also what the gate compares against at run time.",
            "Calibration observes the conditional branch; the two CFG branches share "
            "the same z_k, so the gate feature is branch-independent.",
            "The frozen table is read-only during native online gating.",
        ]
    else:
        notes = [
            "Qwen-Image directional finite-difference sensitivity on packed latent tokens.",
            "Timesteps use QwenImageTransformer2DModel.forward normalized units.",
            "Calibration uses the conditional branch that supplies the shared true-CFG action.",
            "The frozen table is read-only during native online gating.",
        ]
    metadata = {
        "schema": f"{args.backbone}_sencache_sensitivity_table.v1",
        "backbone": args.backbone,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_acc": str(args.acc),
        "source_files": [str(p) for p in paths],
        "row_count": int(len(rows)),
        "prompt_count": int(len({str(r.get("prompt_id")) for r in rows})),
        "num_steps": int(args.num_steps),
        "aggregation": str(args.aggregation),
        "prompt_file": None if args.prompt_file is None else str(args.prompt_file),
        "model_id": args.model_id,
        "model_name": args.model_name,
        "width": args.width,
        "height": args.height,
        "guidance": args.guidance,
        "dtype": args.dtype,
        "seed": args.seed,
        "limit": args.limit,
        "latent_numel_values": latent_numels,
        "latent_shape_values": latent_shapes,
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "notes": notes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        timesteps=np.asarray(timesteps, dtype=np.float64),
        J_x_norm=np.asarray(jx, dtype=np.float64),
        J_t_norm=np.asarray(jt, dtype=np.float64),
        J_x_mean=np.asarray(jx_mean, dtype=np.float64),
        J_t_mean=np.asarray(jt_mean, dtype=np.float64),
        J_x_median=np.asarray(jx_median, dtype=np.float64),
        J_t_median=np.asarray(jt_median, dtype=np.float64),
        J_x_q90=np.asarray(jx_q90, dtype=np.float64),
        J_t_q90=np.asarray(jt_q90, dtype=np.float64),
        row_count_per_step=np.asarray(counts, dtype=np.int64),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": str(summary_path), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
