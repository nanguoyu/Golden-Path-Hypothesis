#!/usr/bin/env python3
"""Pool native-gate path counts over the DISCOVERY prompts only.

`dataset_path_counts.tsv` was pooled over all 1,632 Parti prompts, which the
SPX plan (section 3, 冻结边界) demotes to screening-only: the formal candidates
must come from the pre-declared discovery split, or the schedule axis has seen
the prompts it will later be scored on. This tool is the missing producer. It
reads the raw `decisions_*.json` of the native-gate runs (on the cluster, where
they live), keeps only the decisions whose `prompt_idx` is in the split's
discovery role, pools across the given runs (the three seed streams of one
(model, K, gate)), and writes rows in the exact `dataset_path_counts.tsv`
format under a dataset label that cannot be mistaken for the pooled one --
`parti_discovery` -- so `build_sp_cross_schedules.py --path_counts <out>
--dataset parti_discovery` freezes the compliant candidates.

    python analysis/build_discovery_path_counts.py \\
        --split resources/sp_cross_schedules/parti_spx_splits.v1.json \\
        --model flux --method seacache --target_k 29 \\
        --run $DATA/.../sd_flux_parti_full_s0_k29_seacache \\
        --run $DATA/.../sd_flux_parti_full_s1_k29_seacache \\
        --run $DATA/.../sd_flux_parti_full_s2_k29_seacache \\
        --output discovery_path_counts.tsv --append
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_native_schedule_paths import (  # noqa: E402
    cache_steps,
    format_steps,
    full_steps,
    ranked_counter,
    schedule_from_payload,
)
from analysis.build_parti_spx_splits import load_split  # noqa: E402

DATASET_LABEL = "parti_discovery"
FIELDS = (
    "model", "dataset", "target_k", "method", "rank", "schedule", "cache_count",
    "count", "mass", "cache_steps", "full_steps", "prompt_count", "run_dirs",
)


def pool_runs(run_dirs: list[Path], allowed: set[int]) -> Counter[str]:
    """One Counter over every discovery decision of every given run.

    Integrity over convenience: every decision file must parse and carry a
    unique prompt_idx, and every discovery index must be present in every run
    -- a run missing discovery prompts is an incomplete extraction whose modal
    path may be an artifact of which prompts happen to exist.
    """
    pooled: Counter[str] = Counter()
    for run_dir in run_dirs:
        paths = sorted(Path(run_dir).glob("decisions_*.json"))
        if not paths:
            raise SystemExit(f"{run_dir} contains no decisions_*.json")
        seen: set[int] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            prompt_idx = int(payload.get("prompt_idx", -1))
            if prompt_idx < 0 or prompt_idx in seen:
                raise SystemExit(f"{path}: invalid or duplicate prompt_idx={prompt_idx}")
            seen.add(prompt_idx)
            if prompt_idx not in allowed:
                continue
            pooled[schedule_from_payload(payload, path)] += 1
        missing = allowed - seen
        if missing:
            raise SystemExit(
                f"{run_dir} is missing {len(missing)} discovery prompt(s), first "
                f"{min(missing)}; an incomplete run must not feed candidate "
                f"extraction")
    return pooled


def rows_for(pooled: Counter[str], *, model: str, method: str, target_k: int,
             run_dirs: list[Path]) -> list[dict[str, Any]]:
    total = sum(pooled.values())
    rows = []
    for rank, (schedule, count) in enumerate(ranked_counter(pooled), start=1):
        rows.append({
            "model": model,
            "dataset": DATASET_LABEL,
            "target_k": target_k,
            "method": method,
            "rank": rank,
            "schedule": schedule,
            "cache_count": schedule.count("1"),
            "count": count,
            "mass": count / total,
            "cache_steps": format_steps(cache_steps(schedule)),
            "full_steps": format_steps(full_steps(schedule)),
            "prompt_count": total,
            "run_dirs": ";".join(str(run) for run in run_dirs),
        })
    return rows


def write_rows(output: Path, rows: list[dict[str, Any]], append: bool) -> None:
    exists = output.is_file()
    if append and exists:
        # refuse to append the same (model, method, K) twice: last-wins pooling
        # across two invocations would double-count seeds
        head, *body = output.read_text(encoding="utf-8").splitlines()
        if head != "\t".join(FIELDS):
            raise SystemExit(f"{output} does not carry this tool's columns")
        keys = {tuple(line.split("\t")[i] for i in (0, 3, 2)) for line in body}
        for row in rows:
            key = (row["model"], row["method"], str(row["target_k"]))
            if key in keys:
                raise SystemExit(f"{output} already holds rows for {key}; pooling "
                                 f"twice would double-count seeds")
    with output.open("a" if (append and exists) else "w", encoding="utf-8") as sink:
        if not (append and exists):
            sink.write("\t".join(FIELDS) + "\n")
        for row in rows:
            sink.write("\t".join(str(row[field]) for field in FIELDS) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--role", default="discovery",
                        help="which role of the split feeds extraction "
                             "(default discovery; anything else is for diagnostics "
                             "and must never feed the builder)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target_k", type=int, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="one native-gate run directory; repeat per seed stream")
    parser.add_argument("--prompt_file_sha256",
                        help="digest of the prompt file the runs generated from; "
                             "checked against the split's binding when given")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    allowed = load_split(args.split, role=args.role,
                         prompt_file_sha256=args.prompt_file_sha256)
    pooled = pool_runs(args.run, allowed)
    rows = rows_for(pooled, model=args.model, method=args.method,
                    target_k=args.target_k, run_dirs=args.run)
    write_rows(args.output, rows, args.append)
    top = rows[0]
    print(f"{args.model}/{args.method}/K{args.target_k}: {len(rows)} paths over "
          f"{top['prompt_count']} discovery decisions; top mass {top['mass']:.3f} "
          f"(cache_count {top['cache_count']}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
