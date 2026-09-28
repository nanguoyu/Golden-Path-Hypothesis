"""Place the FLUX-K41 searched schedules on the exhaustive truth table.

Reads the four search summaries copied under resources/schedule_search/search/
(flux_k41_<algorithm>.json), looks every selected schedule and arbitration
candidate up in the K41 table (four-pair mean PSNR), and writes
resources/schedule_search/k41_table_placement.json with the table value, the
gap to the table best and the rank position of each.  This is the plan's
"free panoramic check" (docs/schedule_search_plan_zh.md section 2).

    python analysis/schedule_search_table_placement.py \
        --table resources/exhaustive_k41/merged.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

SEARCH_DIR = Path("resources/schedule_search/search")
OUT = Path("resources/schedule_search/k41_table_placement.json")
FORCED = {0, 1, 2, 49}


def free_key(schedule: str) -> str:
    steps = [int(s) for s in schedule.split(",")]
    return ",".join(str(s) for s in steps if s not in FORCED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    wanted: dict[str, list[dict]] = {}
    for path in sorted(SEARCH_DIR.glob("flux_k41_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        algorithm = run["algorithm"]
        sel = run["selected"]
        wanted.setdefault(free_key(sel["schedule"]), []).append(
            {"algorithm": algorithm, "role": "selected", "schedule": sel["schedule"],
             "calibration_mean_psnr_db": sel["mean_psnr_db"]}
        )
        for rank, cand in enumerate(run["arbitration_candidates"]):
            wanted.setdefault(free_key(cand["schedule"]), []).append(
                {"algorithm": algorithm, "role": f"candidate_{rank}", "schedule": cand["schedule"],
                 "calibration_mean_psnr_db": cand["mean_psnr_db"]}
            )

    rows: dict[str, tuple[int, float]] = {}
    scores: list[float] = []
    with gzip.open(args.table, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        i_free, i_mean, i_rank = (header.index(c) for c in ("variable_full_steps", "mean_psnr_db", "rank"))
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            value = float(parts[i_mean])
            scores.append(value)
            if parts[i_free] in wanted:
                rows[parts[i_free]] = (int(parts[i_rank]), value)
    ordered = np.sort(np.asarray(scores))[::-1]
    best = float(ordered[0])

    out = {"table": str(args.table), "space_size": int(ordered.size), "table_best_mean_psnr_db": best, "schedules": []}
    for key, uses in wanted.items():
        rank, value = rows[key]
        position = int(np.searchsorted(-ordered, -value)) + 1
        out["schedules"].append(
            {"schedule": uses[0]["schedule"], "table_rank": rank, "table_mean_psnr_db": value,
             "gap_to_table_best_db": best - value, "position": position,
             "top_share": position / ordered.size, "uses": uses}
        )
    out["schedules"].sort(key=lambda r: r["position"])
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for row in out["schedules"]:
        who = ", ".join(f"{u['algorithm']}:{u['role']}" for u in row["uses"])
        print(f"{row['schedule']:26s} table {row['table_mean_psnr_db']:.3f}  gap {row['gap_to_table_best_db']:.3f}  "
              f"pos {row['position']:>7d} ({100*row['top_share']:.2f}%)  {who}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
