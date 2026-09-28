"""P3 of `docs/schedule_search_plan_zh.md`: the delivery list for P4.

Reads the six settings' `arbitration.json` (copied off the clusters into
`resources/schedule_search/arbitration/<model>_k<K>.json`), redoes the
cross-algorithm dedupe of each setting -- every algorithm delivers its own
candidate of highest arbitration mean, and algorithms that land on the same
bitstring deliver one schedule under one joined name -- and writes the P4 cell
list `resources/schedule_search/delivery.txt` in the `model K name bits` form
`RUN/schedule_search_eval_cells.py` reads, validating with that reader.

The pairwise Hamming distances among each setting's distinct delivered
schedules are printed with the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from RUN.schedule_search_eval_cells import read_schedules  # noqa: E402
from lib.schedule_search import bits_hamming, delivery_list  # noqa: E402

ARBITRATION_DIR = REPO / "resources" / "schedule_search" / "arbitration"
DELIVERY_FILE = REPO / "resources" / "schedule_search" / "delivery.txt"
SETTINGS = [(model, k) for model in ("flux", "qwen") for k in (29, 37, 41)]


def load_setting(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("mode") != "arbitrate":
        raise SystemExit(f"{path}: mode is {record.get('mode')!r}, expected arbitrate")
    arbitration = record["arbitration"]
    return {
        "model": str(record["model"]),
        "k": int(record["k"]),
        "path": path,
        "n_prompts": int(arbitration["n_prompts"]),
        "seed": int(arbitration["seed"]),
        "candidates": arbitration["candidates"],
    }


def report(setting: dict, delivery: list[dict]) -> None:
    print(f"== {setting['model']} K{setting['k']}  ({setting['path'].name})")
    print(
        f"   {len(setting['candidates'])} distinct candidates on "
        f"{setting['n_prompts']} captions at seed {setting['seed']}"
    )
    for row in sorted(
        setting["candidates"], key=lambda r: -r["arbitration_mean_psnr_db"]
    ):
        proposers = ", ".join(
            f"{p['algorithm']}#{p['rank'] + 1} cal={p['calibration_mean_psnr_db']:.3f}"
            for p in row["proposed_by"]
        )
        print(
            f"   {row['bits']}  arb mean {row['arbitration_mean_psnr_db']:7.3f} dB  "
            f"min {row['arbitration_min_psnr_db']:7.3f} dB  [{proposers}]"
        )
    print("   delivered:")
    for row in delivery:
        print(
            f"     {row['name']:<24} {row['bits']}  "
            f"mean {row['arbitration_mean_psnr_db']:.3f} dB  "
            f"min {row['arbitration_min_psnr_db']:.3f} dB"
        )
    if len(delivery) > 1:
        names = [row["name"] for row in delivery]
        width = max(len(name) for name in names)
        print("   pairwise Hamming among the delivered schedules:")
        print(" " * (width + 6) + "  ".join(f"{name:>{width}}" for name in names))
        for left in delivery:
            cells = "  ".join(
                f"{bits_hamming(left['bits'], right['bits']):>{width}}"
                for right in delivery
            )
            print(f"     {left['name']:<{width}} {cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arbitration_dir", type=Path, default=ARBITRATION_DIR)
    parser.add_argument("--out", type=Path, default=DELIVERY_FILE)
    parser.add_argument(
        "--objective", default="",
        help="read <model>_k<K>_<objective>.json and write delivery_<objective>.txt",
    )
    args = parser.parse_args()
    if args.objective:
        if args.out == DELIVERY_FILE:
            args.out = DELIVERY_FILE.with_name(f"delivery_{args.objective}.txt")

    lines = [
        "# P3 delivery list of docs/schedule_search_plan_zh.md, written by",
        "# analysis/schedule_search_delivery.py.  Columns: model K name bits.",
    ]
    found = 0
    for model, k in SETTINGS:
        suffix = f"_{args.objective}" if args.objective else ""
        path = args.arbitration_dir / f"{model}_k{k}{suffix}.json"
        if not path.is_file():
            print(f"[delivery] no arbitration record at {path} -- skipping")
            continue
        setting = load_setting(path)
        if setting["model"] != model or setting["k"] != k:
            raise SystemExit(
                f"{path}: record is {setting['model']} K{setting['k']}, "
                f"the filename says {model} K{k}"
            )
        found += 1
        delivery = delivery_list(setting["candidates"])
        report(setting, delivery)
        for row in delivery:
            lines.append(f"{model}\t{k}\t{row['name']}\t{row['bits']}")
    if not found:
        raise SystemExit(f"no arbitration records under {args.arbitration_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    schedules = read_schedules(args.out)
    print(
        f"[delivery] {found}/{len(SETTINGS)} settings, {len(schedules)} schedules "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
