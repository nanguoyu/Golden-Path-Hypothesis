#!/usr/bin/env python3
"""Write the recalibrated SenCache pairs into the frozen image threshold sources.

The image lane keeps SenCache's operating point in two places, and both drive
matrix runs:

  * `resources/diffusiondb_clean10k_native_calibration_results/native_thresholds.json`
    -- the DiffusionDB wave, read by RUN/submit_{flux,qwen}_baseline_suite.sh
  * `resources/cross_model_multiseed_stage_d_tasks.tsv` -- the three formal
    datasets, whose `runtime_parameters` column each cell runs from

Both are rewritten here from one selection so they cannot drift apart. The Wan
config is handled by analysis/wan21/refreeze_sencache_config.py, which has to
re-hash a self-hashed file.

Superseded values are replaced, not annotated (repository rule); what changed is
recorded once, in the plan's section 9.

Plan: docs/sencache_recalibration_plan_zh.md S2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BUDGETS = (29, 37, 41)
MODELS = ("flux", "qwen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--native_thresholds", type=Path,
                        default=_ROOT / "resources/diffusiondb_clean10k_native_calibration_results/native_thresholds.json")
    parser.add_argument("--stage_d_tasks", type=Path,
                        default=_ROOT / "resources/cross_model_multiseed_stage_d_tasks.tsv")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _frozen(selection: dict, model: str, budget: int) -> dict:
    entry = selection["groups"][model]["budgets"][f"k{budget}"]
    if entry.get("frozen") is None:
        raise SystemExit(f"{model} K{budget}: selection froze no pair")
    return {
        "threshold_main": float(entry["frozen"]["main"]),
        "threshold_start": float(entry["frozen"]["start"]),
        "max_skip": int(entry["max_skip"]),
        "switch_ratio": float(entry["switch_ratio"]),
        "k_mean": float(entry["frozen"]["k_mean"]),
    }


def update_native_thresholds(path: Path, selection: dict, dry_run: bool) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for point in payload.get("operating_points", []):
        if point.get("method") != "SenCache":
            continue
        model, budget = point["model"], int(point["target_cache_count"])
        new = _frozen(selection, model, budget)
        old = (point.get("selected_threshold"), point.get("sencache_threshold_start"))
        point["selected_threshold"] = new["threshold_main"]
        point["sencache_threshold_start"] = new["threshold_start"]
        point["sencache_max_skip"] = new["max_skip"]
        point["sencache_switch_ratio"] = new["switch_ratio"]
        point["cache_count_mean"] = new["k_mean"]
        print(f"  native_thresholds {model} K{budget}: "
              f"main {old[0]} -> {new['threshold_main']:g}, "
              f"start {old[1]} -> {new['threshold_start']:g}, "
              f"n {new['max_skip']}, switch {new['switch_ratio']:g}")
        changed += 1
    if not dry_run:
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return changed


def update_stage_d(path: Path, selection: dict, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    fields = reader.fieldnames
    rows = list(reader)
    changed = 0
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("method") != "sencache":
            continue
        model, budget = row["model"], int(row["budget_k"])
        new = _frozen(selection, model, budget)
        row["runtime_parameters"] = json.dumps({
            "first_enhance": 1,
            "max_skip": new["max_skip"],
            "switch_ratio": new["switch_ratio"],
            "threshold_main": new["threshold_main"],
            "threshold_start": new["threshold_start"],
        }, separators=(",", ":"), sort_keys=True)
        changed += 1
        if (model, budget) not in seen:
            seen.add((model, budget))
            print(f"  stage_d {model} K{budget}: {row['runtime_parameters']}")
    if not dry_run:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return changed


def main() -> int:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    for model in MODELS:
        for budget in BUDGETS:
            _frozen(selection, model, budget)  # fail before writing anything
    n1 = update_native_thresholds(args.native_thresholds, selection, args.dry_run)
    n2 = update_stage_d(args.stage_d_tasks, selection, args.dry_run)
    print(f"[freeze] native operating points={n1} stage_d rows={n2}"
          f"{' (dry run)' if args.dry_run else ''}")
    if n1 != 6 or n2 != 54:
        raise SystemExit(f"expected 6 operating points and 54 stage-D rows, got {n1} and {n2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
