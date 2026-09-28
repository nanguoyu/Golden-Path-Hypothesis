#!/usr/bin/env python3
"""Extract the SPX schedule axis (8 schedules per model/K) into text files.

Sources (both frozen under `resources/cross_model_multiseed_stage_e_native_paths/`):

* `fixed_paths.tsv` — the formal fixed-schedule rows. Four of the eight SPX
  schedules come from here: `budcache`, `meancache`, `dpcache` (searched paths)
  and `taylorseer_o1` (the uniform schedule shared by the TaylorSeer O1 /
  HiCache O2 / L2P triplet), emitted as `uniform`.
* `dataset_path_counts.tsv` — pooled native-gate path frequencies. The other
  four schedules are the most frequent exact-K paths of `seacache`,
  `teacache`, `sencache`, `dicache` on `dataset=parti_full`. Ties in frequency
  are broken lexicographically on the bitstring.

Output: `resources/sp_cross_schedules/<model>_k<K>_<name>.txt`, one 50-character
`'1'=cache / '0'=full` bitstring per file, plus `manifest.tsv` recording every
source row, its mass, and the popcount check.

Checks (hard failures):

* bitstring is exactly `--num_steps` characters of `0`/`1`;
* `popcount == cache_count == target_k`;
* step 0 is a full step (every SPX runner forces step 0 full).

If a supplied discovery scope contains no exact-K path for one gate, that
gate-derived schedule is reported as unavailable and omitted.  The builder
never edits or projects a path to fill the missing source.

The `_top1` filename suffix is retained as a stable runner identifier.  For
native schedules it means top-frequency within the exact-K subset of the
supplied dataset scope, not necessarily rank 1 in the unconditioned source
table.  The committed `parti_full` outputs are exploratory pooled candidates;
a confirmatory SPX freeze must rebuild them from discovery-only path counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATHS_DIR = (
    REPO_ROOT / "resources" / "cross_model_multiseed_stage_e_native_paths"
)

# name -> method column value in fixed_paths.tsv
FIXED_SCHEDULES = {
    "budcache": "budcache",
    "meancache": "meancache",
    "dpcache": "dpcache",
    "uniform": "taylorseer_o1",
}
# name -> method column value in dataset_path_counts.tsv (exact-K modal path)
NATIVE_SCHEDULES = {
    "seacache_top1": "seacache",
    "teacache_top1": "teacache",
    "sencache_top1": "sencache",
    "dicache_top1": "dicache",
}
SCHEDULE_ORDER = (
    "budcache",
    "meancache",
    "dpcache",
    "uniform",
    "seacache_top1",
    "teacache_top1",
    "sencache_top1",
    "dicache_top1",
)
MANIFEST_FIELDS = (
    "model",
    "target_k",
    "name",
    "axis",
    "source_file",
    "source_method",
    "source_dataset",
    "source_rank",
    "family",
    "source_schema",
    "source_seed",
    "source_prompt_idx",
    "count",
    "mass",
    "cache_count",
    "popcount",
    "k_vs_target",
    "first_step",
    "last_step",
    "schedule",
    "path",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def manifest_path_text(path: Path) -> str:
    """Repo-relative when the output stays in the repo, absolute otherwise."""

    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def check_schedule(
    schedule: str,
    *,
    num_steps: int,
    cache_count: int,
    label: str,
) -> int:
    """Validate one bitstring and return its popcount."""

    bits = str(schedule).strip()
    if len(bits) != int(num_steps) or set(bits) - {"0", "1"}:
        raise SystemExit(
            f"{label}: schedule must be {num_steps} characters of 0/1, got {bits!r}"
        )
    popcount = bits.count("1")
    if popcount != int(cache_count):
        raise SystemExit(
            f"{label}: popcount {popcount} disagrees with source cache_count "
            f"{int(cache_count)}"
        )
    if bits[0] != "0":
        raise SystemExit(f"{label}: step 0 must be a full step")
    return popcount


def select_fixed(
    rows: Sequence[Mapping[str, str]],
    *,
    model: str,
    target_k: int,
    method: str,
) -> Mapping[str, str]:
    hits = [
        row
        for row in rows
        if row["model"] == model
        and int(row["target_k"]) == int(target_k)
        and row["method"] == method
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"fixed_paths: expected 1 row for {model}/K{target_k}/{method}, "
            f"got {len(hits)}"
        )
    return hits[0]


def select_native(
    rows: Sequence[Mapping[str, str]],
    *,
    model: str,
    target_k: int,
    method: str,
    dataset: str,
) -> Mapping[str, str] | None:
    hits = [
        row
        for row in rows
        if row["model"] == model
        and row["dataset"] == dataset
        and int(row["target_k"]) == int(target_k)
        and row["method"] == method
        and int(row["cache_count"]) == int(target_k)
    ]
    if not hits:
        return None
    return min(hits, key=lambda row: (-int(row["count"]), row["schedule"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed_paths",
        type=Path,
        default=NATIVE_PATHS_DIR / "fixed_paths.tsv",
    )
    parser.add_argument(
        "--path_counts",
        type=Path,
        default=NATIVE_PATHS_DIR / "dataset_path_counts.tsv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "resources" / "sp_cross_schedules",
    )
    parser.add_argument("--dataset", default="parti_full")
    parser.add_argument("--models", nargs="+", default=["flux", "qwen"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[29, 37, 41])
    parser.add_argument("--num_steps", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixed_rows = read_tsv(args.fixed_paths)
    native_rows = read_tsv(args.path_counts)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for model in args.models:
        for target_k in args.budgets:
            for name in SCHEDULE_ORDER:
                if name in FIXED_SCHEDULES:
                    method = FIXED_SCHEDULES[name]
                    row = select_fixed(
                        fixed_rows, model=model, target_k=target_k, method=method
                    )
                    axis = "fixed"
                    source_file = args.fixed_paths.name
                    dataset = ""
                    rank = ""
                    count = ""
                    mass = ""
                else:
                    method = NATIVE_SCHEDULES[name]
                    row = select_native(
                        native_rows,
                        model=model,
                        target_k=target_k,
                        method=method,
                        dataset=args.dataset,
                    )
                    if row is None:
                        print(
                            "[sp-cross-schedules] unavailable exact-K source: "
                            f"{model}/{args.dataset}/K{target_k}/{method}"
                        )
                        continue
                    axis = "native_exact_k_modal"
                    source_file = args.path_counts.name
                    dataset = args.dataset
                    rank = row["rank"]
                    count = row["count"]
                    mass = row["mass"]
                schedule = row["schedule"].strip()
                label = f"{model}_k{target_k}_{name}"
                popcount = check_schedule(
                    schedule,
                    num_steps=args.num_steps,
                    cache_count=int(row["cache_count"]),
                    label=label,
                )
                if popcount != int(target_k):
                    raise SystemExit(
                        f"{label}: exact-K protocol requires {target_k} cached "
                        f"steps, got {popcount}"
                    )
                path = args.out_dir / f"{label}.txt"
                path.write_text(schedule + "\n", encoding="utf-8")
                manifest.append(
                    {
                        "model": model,
                        "target_k": int(target_k),
                        "name": name,
                        "axis": axis,
                        "source_file": source_file,
                        "source_method": method,
                        "source_dataset": dataset,
                        "source_rank": rank,
                        "family": row.get("family", ""),
                        "source_schema": row.get("source_schema", ""),
                        "source_seed": row.get("source_seed", ""),
                        "source_prompt_idx": row.get("source_prompt_idx", ""),
                        "count": count,
                        "mass": mass,
                        "cache_count": int(row["cache_count"]),
                        "popcount": popcount,
                        "k_vs_target": popcount - int(target_k),
                        "first_step": "cache" if schedule[0] == "1" else "full",
                        "last_step": "cache" if schedule[-1] == "1" else "full",
                        "schedule": schedule,
                        "path": manifest_path_text(path),
                    }
                )

    manifest_path = args.out_dir / "manifest.tsv"
    write_tsv(manifest_path, manifest, MANIFEST_FIELDS)

    for entry in manifest:
        print(
            f"{entry['path']}\tK={entry['popcount']}"
            f"\tdK={entry['k_vs_target']:+d}"
            f"\tlast={entry['last_step']}"
            f"\tmass={entry['mass'] or '-'}"
        )
    print(f"[sp-cross-schedules] wrote {len(manifest)} schedules -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
