#!/usr/bin/env python3
"""Materialize deterministic exact-K schedules used by baseline screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.dpcache import cache_steps_from_full, select_full_steps
from lib.fixed_schedule import evenly_spaced_cache_steps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_ROOT / "resources" / "baseline_exact" / "flux_k29",
    )
    parser.add_argument(
        "--dpcache_cost",
        type=Path,
        default=_ROOT / "reference" / "dpcache" / "code" / "final_3d_cost_matrix_flux.pkl",
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, default=29)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shared = evenly_spaced_cache_steps(
        num_steps=args.num_steps,
        cache_count=args.cache_count,
        first_full_steps=3,
        last_full_steps=1,
    )
    _write(
        args.output_dir / "shared_predictor_schedule.json",
        {
            "format": "flux-exact-cache-schedule-v1",
            "role": "shared_fixed_predictor_schedule",
            "num_steps": args.num_steps,
            "cache_count": args.cache_count,
            "cache_steps": list(shared),
            "full_steps": [
                step for step in range(args.num_steps) if step not in set(shared)
            ],
            "methods": [
                "taylorseer_fine_exact",
                "hicache_fine_exact",
                "l2p_output_exact",
                "foca_fine_exact",
                "toca_exact",
            ],
        },
    )
    _write(
        args.output_dir / "meancache_initial_schedule.json",
        {
            "format": "flux-meancache-schedule-v1",
            "role": "smoke_only_before_stability_calibration",
            "num_steps": args.num_steps,
            "cache_count": args.cache_count,
            "cache_steps": list(shared),
            "jvp_spans": {str(step): 4 for step in shared},
        },
    )

    with args.dpcache_cost.open("rb") as handle:
        cost = pickle.load(handle)
    full = select_full_steps(
        cost,
        total_steps=args.num_steps,
        full_count=args.num_steps - args.cache_count,
        first_full_steps=3,
        last_full_steps=1,
    )
    cache = cache_steps_from_full(full, total_steps=args.num_steps)
    _write(
        args.output_dir / "dpcache_official_cost_schedule.json",
        {
            "format": "flux-dpcache-schedule-v1",
            "role": "initial_screening_official_cost_tensor",
            "num_steps": args.num_steps,
            "cache_count": args.cache_count,
            "cache_steps": list(cache),
            "full_steps": list(full),
            "full_count": len(full),
            "first_full_steps": 3,
            "last_full_steps": 1,
            "order": 2,
            "cost_tensor": str(args.dpcache_cost.relative_to(_ROOT)),
            "cost_tensor_sha256": _sha256(args.dpcache_cost),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
