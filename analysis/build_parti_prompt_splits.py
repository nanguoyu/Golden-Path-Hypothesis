#!/usr/bin/env python3
"""Create deterministic PartiPrompts calibration/eval prompt files.

The source TSV is ordered by metadata, so taking the first N rows is not a
robust prompt-distribution test. This script stratifies by Category and
Challenge when those columns are present, then writes plain one-prompt-per-line
files that can be passed directly to the FLUX runners.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic Parti prompt splits.")
    p.add_argument("--tsv", type=Path, default=Path("resources/prompts/datasets/PartiPrompts.tsv"))
    p.add_argument("--out_dir", type=Path, default=Path("resources/prompts"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib_n", type=int, default=50)
    p.add_argument("--eval_n", type=int, default=200)
    p.add_argument("--prefix", default="partiprompts_stratified")
    return p.parse_args()


def _round_robin_stratified(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("Category", "")), str(row.get("Challenge", "")))
        buckets[key].append(row)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])
    rng.shuffle(keys)

    selected: list[dict[str, Any]] = []
    while len(selected) < n and keys:
        next_keys = []
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) == n:
                    break
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) < n:
        raise SystemExit(f"only selected {len(selected)} rows, need {n}")
    return selected


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(str(r["Prompt"]).strip() for r in rows) + "\n",
                    encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.tsv.is_file():
        raise SystemExit(f"TSV not found: {args.tsv}")
    with args.tsv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [dict(row, _row_id=i + 1) for i, row in enumerate(rows)
            if str(row.get("Prompt", "")).strip()]
    total = args.calib_n + args.eval_n
    if len(rows) < total:
        raise SystemExit(f"need {total} rows, found {len(rows)}")

    rng = random.Random(args.seed)
    selected = _round_robin_stratified(rows, total, rng)
    calib = selected[:args.calib_n]
    eval_rows = selected[args.calib_n:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = args.out_dir / f"{args.prefix}_calib{args.calib_n}_seed{args.seed}.txt"
    eval_path = args.out_dir / f"{args.prefix}_eval{args.eval_n}_seed{args.seed}.txt"
    meta_path = args.out_dir / f"{args.prefix}_calib{args.calib_n}_eval{args.eval_n}_seed{args.seed}.json"
    _write_split(calib_path, calib)
    _write_split(eval_path, eval_rows)
    meta = {
        "schema": "partiprompt_split_v1",
        "source_tsv": str(args.tsv),
        "seed": int(args.seed),
        "calib_file": str(calib_path),
        "eval_file": str(eval_path),
        "calib_n": len(calib),
        "eval_n": len(eval_rows),
        "calib_rows": calib,
        "eval_rows": eval_rows,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {calib_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
