#!/usr/bin/env python3
"""Merge L2P trajectory Gram shards and solve one causal weight matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.l2p import l2p_teacher_forced_errors, solve_l2p_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gram", type=Path, action="append", required=True)
    parser.add_argument("--holdout_gram", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.gram:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("gram"), torch.Tensor):
            raise ValueError(f"invalid L2P Gram shard: {path}")
        rows.append(payload)

    keys = ("model", "target", "num_steps", "prompt_file")
    expected = {key: rows[0].get(key) for key in keys}
    for path, row in zip(args.gram, rows):
        observed = {key: row.get(key) for key in keys}
        if observed != expected:
            raise ValueError(f"incompatible L2P Gram shard {path}: {observed} != {expected}")

    shard_count = int(rows[0]["shard_count"])
    indices = sorted(int(row["shard_idx"]) for row in rows)
    if indices != list(range(shard_count)):
        raise ValueError(f"incomplete L2P Gram shards: expected 0..{shard_count - 1}, got {indices}")
    gram = sum(
        (row["gram"].to(dtype=torch.float64) for row in rows),
        torch.zeros_like(rows[0]["gram"], dtype=torch.float64),
    )
    trajectory_count = sum(int(row["prompt_count"]) for row in rows)
    prompt_count = sum(
        int(row.get("prompt_pairs", row["prompt_count"]))
        for row in rows
    )
    weights = solve_l2p_weights(gram, ridge=float(args.ridge))
    train_errors = l2p_teacher_forced_errors(gram, weights)
    holdout_errors = None
    if args.holdout_gram:
        holdout_rows: list[dict[str, Any]] = []
        for path in args.holdout_gram:
            row = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(row, dict) or not isinstance(row.get("gram"), torch.Tensor):
                raise ValueError(f"invalid L2P holdout Gram shard: {path}")
            observed = {
                key: row.get(key)
                for key in ("model", "target", "num_steps")
            }
            required = {
                key: expected[key]
                for key in ("model", "target", "num_steps")
            }
            if observed != required:
                raise ValueError(
                    f"incompatible L2P holdout Gram shard {path}: "
                    f"{observed} != {required}"
                )
            holdout_rows.append(row)
        holdout_gram = sum(
            (row["gram"].to(dtype=torch.float64) for row in holdout_rows),
            torch.zeros_like(holdout_rows[0]["gram"], dtype=torch.float64),
        )
        holdout_errors = l2p_teacher_forced_errors(holdout_gram, weights)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "l2p-v1",
        "target": str(expected["target"]),
        "granularity": "final_output",
        "model": str(expected["model"]),
        "num_steps": int(expected["num_steps"]),
        "weights": weights,
        "ridge": float(args.ridge),
        "fit_method": "causal_gram_least_squares",
        "train_prompt_file": str(expected["prompt_file"]),
        "train_prompt_count": int(prompt_count),
        "train_trajectory_count": int(trajectory_count),
        "gram_shards": [str(path) for path in args.gram],
        "holdout_gram_shards": [str(path) for path in args.holdout_gram],
        "train_teacher_forced_errors": train_errors,
        "holdout_teacher_forced_errors": holdout_errors,
    }
    torch.save(checkpoint, args.out)
    manifest = {key: value for key, value in checkpoint.items() if key != "weights"}
    manifest["weights_shape"] = list(weights.shape)
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[l2p-solve] wrote {args.out} from {prompt_count} prompts "
        f"and {trajectory_count} feature trajectories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
