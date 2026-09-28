#!/usr/bin/env python3
"""Freeze the inputs of the image cache-bend layer.

`docs/image_cached_trajectory_plan_zh.md` section 4 (sampling design) and
section 7.1 (the all-zero schedule the reference wave runs on). Three artefacts,
all committed, all derived from data already in the repository so that a reader
can regenerate them and get the same bytes:

  ``resources/image_trajectory/prompt_sample.v1.json``
      the P prompt indices every cell and both models replay, drawn once from
      ``numpy.random.default_rng([20260824])`` without replacement, each tagged
      with its role in ``resources/sp_cross_schedules/parti_spx_splits.v1.json``
      so Q2 can be re-run on the held-out half.

  ``resources/image_trajectory/zero_schedule_50.txt``
      50 zeros. A schedule with no cache step is a no-cache generation, which is
      how the reference wave is produced by the cell runner instead of a second
      code path (plan section 7.1, last paragraph).

  ``resources/image_trajectory/cells.v1.tsv``
      the frozen cell list. Option A of section 10: the whole ``reuse`` column,
      the W1 full-rank schedule x payload sub-grid, the homologous
      ``meancache x mean_avg_vel`` cells and the FLUX ``gpf_o1 x taylor_o1``
      cells — 257 cells, every one of them already generated at seed 42 with
      1632 prompts, so the replay has something to be checked against.

Which cells exist is read from the per-prompt SPX quality tables
(``resources/spx/perprompt_spx_<model>.tsv.gz``), the same tables Q2 joins
against, so the cell list and the quality column cannot drift apart.

    python analysis/image_trajectory/build_inputs.py --check
    python analysis/image_trajectory/build_inputs.py --write
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODELS = ("flux", "qwen")
KS = (29, 37, 41)
NUM_STEPS = 50
BASE_SEED = 42
SAMPLE_SIZE = 50
SAMPLE_RNG_SEED = [20260824]

# Option A, plan section 4.2. `reuse` is the whole column; the rest is the W1
# full-rank sub-grid plus the two homologous strips.
W1_SCHEDULES = ("budcache", "dpcache", "uniform", "dicache_top1")
W1_PAYLOADS = ("taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")

OUT_DIR = _ROOT / "resources" / "image_trajectory"
SPLITS = _ROOT / "resources" / "sp_cross_schedules" / "parti_spx_splits.v1.json"
PROMPT_FILE = _ROOT / "resources" / "prompts" / "partiprompts_full_eval1632_seed42.txt"
SCHEDULE_DIRS = (
    _ROOT / "resources" / "sp_cross_schedules",
    _ROOT / "resources" / "spx_supplement_schedules",
)


def spx_cells(model: str) -> set[tuple[str, str, int]]:
    """`(schedule, payload, K)` of every seed-42 cell in the quality table."""
    path = _ROOT / "resources" / "spx" / f"perprompt_spx_{model}.tsv.gz"
    cells: set[tuple[str, str, int]] = set()
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:5] != ["schedule", "payload", "k", "seed", "prompt_idx"]:
            raise SystemExit(f"{path}: unexpected header {header[:5]}")
        for line in handle:
            schedule, payload, k, seed = line.split("\t", 4)[:4]
            if seed == str(BASE_SEED):
                cells.add((schedule, payload, int(k)))
    return cells


def in_option_a(schedule: str, payload: str) -> bool:
    if payload == "reuse":
        return True
    if schedule in W1_SCHEDULES and payload in W1_PAYLOADS:
        return True
    if schedule == "meancache" and payload == "mean_avg_vel":
        return True
    if schedule.startswith("gpf_o1") and payload == "taylor_o1":
        return True
    return False


def schedule_file(model: str, k: int, schedule: str) -> str:
    name = f"{model}_k{k}_{schedule}.txt"
    for directory in SCHEDULE_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate.relative_to(_ROOT))
    raise SystemExit(f"no schedule file for {model} k{k} {schedule}")


def build_cells() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for schedule, payload, k in sorted(spx_cells(model)):
            if not in_option_a(schedule, payload):
                continue
            sched_rel = schedule_file(model, k, schedule)
            bits = (_ROOT / sched_rel).read_text(encoding="utf-8").strip()
            rows.append({
                "cell_id": f"{model}_k{k}_{schedule}x{payload}",
                "model": model,
                "k": k,
                # the off-budget gate rows spend one step more than their
                # partition label; the label groups them, the realized count is
                # what the bitstring actually holds
                "k_realized": bits.count("1"),
                "schedule": schedule,
                "payload": payload,
                "cell_dir": f"{model}/k{k}/{schedule}x{payload}_s{BASE_SEED}",
                "schedule_file": sched_rel,
            })
    return rows


def build_reference_rows() -> list[dict[str, Any]]:
    """One reference stream per model: the same runner on an all-zero schedule."""
    return [{
        "cell_id": f"{model}_refs",
        "model": model,
        "k": 0,
        "k_realized": 0,
        "schedule": "refnone",
        "payload": "reuse",
        "cell_dir": f"{model}/refs_parti_s{BASE_SEED}",
        "schedule_file": "resources/image_trajectory/zero_schedule_50.txt",
    } for model in MODELS]


def build_sample() -> dict[str, Any]:
    prompts = [
        line.strip() for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    total = len(prompts)
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    if splits["prompt_count"] != total:
        raise SystemExit(
            f"split file counts {splits['prompt_count']} prompts, file holds {total}"
        )
    role_of: dict[int, str] = {}
    for role, members in splits["roles"].items():
        for idx in members:
            role_of[int(idx)] = role

    rng = np.random.default_rng(SAMPLE_RNG_SEED)
    chosen = sorted(int(v) for v in rng.choice(total, size=SAMPLE_SIZE, replace=False))
    roles = [role_of[i] for i in chosen]
    return {
        "schema": "image_trajectory.prompt_sample.v1",
        "prompt_file": str(PROMPT_FILE.relative_to(_ROOT)),
        "prompt_file_sha256": hashlib.sha256(
            PROMPT_FILE.read_bytes()).hexdigest(),
        "prompt_count": total,
        "sample_size": SAMPLE_SIZE,
        "rule": ("numpy.random.default_rng([20260824]).choice(1632, size=50, "
                 "replace=False), sorted ascending; shared by both models, "
                 "every cell and the reference wave"),
        "base_seed": BASE_SEED,
        "seed_rule": "base_plus_prompt_idx",
        "split_file": str(SPLITS.relative_to(_ROOT)),
        "prompt_indices": chosen,
        "roles": roles,
        "role_counts": {role: roles.count(role) for role in sorted(set(roles))},
        "held_out_indices": [i for i, r in zip(chosen, roles) if r != "discovery"],
        "discovery_indices": [i for i, r in zip(chosen, roles) if r == "discovery"],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["cell_id", "model", "k", "k_realized", "schedule", "payload",
               "cell_dir", "schedule_file"]
    lines = ["\t".join(columns)]
    lines += ["\t".join(str(row[c]) for c in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cells = build_cells()
    refs = build_reference_rows()
    sample = build_sample()
    zero = "0" * NUM_STEPS + "\n"

    counts: dict[str, int] = {}
    for row in cells:
        counts[f"{row['model']}_k{row['k']}"] = counts.get(
            f"{row['model']}_k{row['k']}", 0) + 1
    print(f"[image-traj] cells={len(cells)} references={len(refs)} "
          f"prompts/cell={SAMPLE_SIZE}")
    print(f"[image-traj] per (model, K): {counts}")
    print(f"[image-traj] sample roles: {sample['role_counts']}")

    # 256, not 257: the recalibrated SenCache gate makes the Qwen K29 modal
    # path exact-K, so qwen_k29_sencache_top1_off has no schedule any more
    if len(cells) != 256:
        raise SystemExit(f"option A is 256 cells, built {len(cells)}")

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "zero_schedule_50.txt").write_text(zero, encoding="utf-8")
        write_tsv(OUT_DIR / "cells.v1.tsv", cells + refs)
        (OUT_DIR / "prompt_sample.v1.json").write_text(
            json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        print(f"[image-traj] wrote {OUT_DIR}")
    if args.check:
        for name, produced in (
            ("zero_schedule_50.txt", zero),
            ("prompt_sample.v1.json", json.dumps(sample, indent=2) + "\n"),
        ):
            stored = (OUT_DIR / name).read_text(encoding="utf-8")
            if stored != produced:
                raise SystemExit(f"{name} on disk differs from a fresh build")
        print("[image-traj] committed inputs reproduce")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
