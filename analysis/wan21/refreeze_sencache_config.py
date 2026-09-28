#!/usr/bin/env python3
"""Write the Wan2.1 matrix config with SenCache's recalibrated cells.

Takes the frozen v1 payload, replaces only what SenCache owns -- its per-cell
threshold and its four frozen knobs -- re-computes the self hash and validates
the result through the loader the runner uses. Everything else, including the
eight other methods and every schedule table, is carried over unchanged, which
is what keeps the completed non-SenCache cells bound to the file they ran under.

A new version file is written rather than the old one edited: the 144 cells that
are not being re-run recorded v1's digest in their identity files, and rewriting
v1 in place would leave them pointing at a config that no longer describes them.

Plan: docs/sencache_recalibration_plan_zh.md S2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# Unconditional; see commit b35a0d3 -- `analysis/wan21/` shadows `wan21/` and a
# guarded insert is a no-op when PYTHONPATH already carries the root behind it.
sys.path.insert(0, str(_ROOT))

from hunyuan_video.config import hash_json  # noqa: E402
from wan21.matrix_config import HASH_FIELD, load_matrix_config  # noqa: E402

BUDGETS = ("K29", "K37", "K41")
DATASETS = ("penguin599", "vbench944")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=_ROOT / "resources/wan21/baseline_matrix_config.v1.json")
    parser.add_argument("--selection", type=Path, required=True,
                        help="frontier selection JSON; its groups are the datasets")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "resources/wan21/baseline_matrix_config.v2.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))

    before = {
        f"{dataset}/{budget}": {
            "threshold": payload["thresholds"]["sencache"][dataset][budget],
            **payload["method_params"]["sencache"][dataset][budget],
        }
        for dataset in DATASETS for budget in BUDGETS
    }

    for dataset in DATASETS:
        group = selection["groups"].get(dataset)
        if group is None:
            raise SystemExit(f"selection carries no group for {dataset}")
        for budget in BUDGETS:
            entry = group["budgets"][budget]
            frozen = entry.get("frozen")
            if frozen is None:
                raise SystemExit(
                    f"{dataset} {budget}: the frontier froze no pair; widen the "
                    f"threshold_main grid and re-run the sweep")
            payload["thresholds"]["sencache"][dataset][budget] = float(frozen["main"])
            payload["method_params"]["sencache"][dataset][budget] = {
                "sencache_first_enhance": 3.0,
                "sencache_threshold_start": float(frozen["start"]),
                "sencache_max_skip": float(entry["max_skip"]),
                "sencache_switch_ratio": float(entry["switch_ratio"]),
            }

    payload[HASH_FIELD] = hash_json(
        {key: value for key, value in payload.items() if key != HASH_FIELD})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    # the loader is the authority on whether this file is usable at all
    config = load_matrix_config(args.out)
    for dataset in DATASETS:
        for budget in BUDGETS:
            entry = config.entry("sencache", budget, dataset)
            after = {"threshold": entry.threshold, **entry.method_params}
            print(f"{dataset} {budget}: {before[f'{dataset}/{budget}']} -> {after}")
    print(f"[refreeze] {HASH_FIELD}={payload[HASH_FIELD]}")
    print(f"[refreeze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
