#!/usr/bin/env python3
"""Collapse per-generation full-trajectory JSONs into one table per (model, dataset).

Reads every `traj_*.json` written by `flux/full_trajectory_probe.py` /
`qwen_image/full_trajectory_probe.py` under the given result dirs and writes one
compact table per (model, dataset) for rsync back from the cluster. Parquet when
pandas + pyarrow are importable (array fields become list columns), otherwise
JSONL. Every field of every record is preserved either way; nothing is
aggregated here.

    python analysis/merge_full_traj.py \\
        --results_dir ~/full_traj_results/flux/drawbench_full_nfull_s41_50 \\
        --output_dir ~/full_traj_results/tables
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

KEY_COLUMNS = ("model", "dataset", "prompt_idx", "seed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, nargs="+", required=True,
                   help="Probe output dirs; searched recursively for traj_*.json.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--format", choices=["auto", "parquet", "jsonl"], default="auto",
                   help="`auto` = parquet when pandas+pyarrow import, else jsonl.")
    return p.parse_args()


def _resolve_format(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        return "jsonl"
    return "parquet"


def _load(paths: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple] = set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(record[c] for c in KEY_COLUMNS)
        if key in seen:
            raise SystemExit(f"duplicate record {key} at {path}")
        seen.add(key)
        groups[(record["model"], record["dataset"])].append(record)
    return groups


def main() -> int:
    args = parse_args()
    paths = sorted(
        {p for root in args.results_dir for p in Path(root).rglob("traj_*.json")}
    )
    if not paths:
        raise SystemExit(f"no traj_*.json found under {[str(r) for r in args.results_dir]}")

    groups = _load(paths)
    fmt = _resolve_format(args.format)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (model, dataset), records in sorted(groups.items()):
        records.sort(key=lambda r: (r["seed"], r["prompt_idx"]))
        stem = args.output_dir / f"full_traj_{model}_{dataset}"
        if fmt == "parquet":
            import pandas as pd

            out = stem.with_suffix(".parquet")
            pd.DataFrame(records).to_parquet(out, index=False)
        else:
            out = stem.with_suffix(".jsonl")
            with out.open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        seeds = sorted({r["seed"] for r in records})
        prompts = len({r["prompt_idx"] for r in records})
        print(f"[merge] {model}/{dataset}: {len(records)} rows "
              f"({prompts} prompts x seeds {seeds}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
