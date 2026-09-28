#!/usr/bin/env python3
"""Build the shared deterministic exact-K schedule for fixed predictors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.fixed_schedule import evenly_spaced_cache_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["flux", "qwen_image"],
        default="qwen_image",
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, required=True)
    parser.add_argument("--first_full_steps", type=int, default=3)
    parser.add_argument("--last_full_steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_steps = evenly_spaced_cache_steps(
        num_steps=args.num_steps,
        cache_count=args.cache_count,
        first_full_steps=args.first_full_steps,
        last_full_steps=args.last_full_steps,
    )
    cached = set(cache_steps)
    payload = {
        "format": f"{args.model.replace('_', '-')}-fixed-predictor-schedule-v1",
        "model": args.model,
        "role": "shared_fixed_predictor_schedule",
        "num_steps": args.num_steps,
        "cache_count": args.cache_count,
        "full_count": args.num_steps - args.cache_count,
        "cache_steps": list(cache_steps),
        "full_steps": [
            step for step in range(args.num_steps) if step not in cached
        ],
        "first_full_steps": args.first_full_steps,
        "last_full_steps": args.last_full_steps,
        "methods": ["TaylorSeer_fine", "HiCache_fine", "L2P_output"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
