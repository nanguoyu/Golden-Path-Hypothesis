#!/usr/bin/env python3
"""Cell inventory for the image SPX supplement wave (plan `docs/sp_cross_supplement_plan_zh.md`).

Three groups of runs share one enumerator so that the submitter, the resume
logic and the staging step cannot disagree about what a cell is:

* **control / off_budget** — the 36 frozen supplement schedule rows
  (`resources/spx_supplement_schedules/`) x the `reuse` payload x 3 seeds.
  These answer the schedule-axis questions (P-S1..P-S4); the payload is held
  at the most neutral zero-order one on purpose.
* **robustness** — the three near-tied rank-2 gate paths, each with its own
  gate's native payload, because P4's paired reading needs the same payload on
  both sides of the pair.
* **dense** — the 60 densification rows (audit round 3): extra draws per rung
  of both Hamming ladders plus three extra random rows, on the four partitions
  whose basin the report quotes. Same `reuse` payload as the control rows,
  ONE seed instead of three.
* **correction** — `meancache x mean_avg_vel` re-run with the frozen per-edge
  `jvp_spans` of the matrix MeanCache solution. The original 540-cell wave ran
  these with the scalar global span 4 because `RUN/slurm_sp_cross.sh` had no
  passthrough for the spans file; the spans are half of the solved path, so
  the cell was not the homologous method's payload.

  The re-run writes under a **separate output root** (`--correction_root`,
  default `$DATA/sp_cross_spans`) rather than over the original directory: the
  superseded numbers leave the documents, not the disk.

Usage:

    python RUN/spx_supplement_cells.py plan                    # every cell, TSV
    python RUN/spx_supplement_cells.py plan --group control --model qwen
    python RUN/spx_supplement_cells.py plan --pending          # only unfinished
    python RUN/spx_supplement_cells.py eval_plan               # (acc, gt) pairs

`plan` prints one TSV row per run:

    group  model  k  schedule  payload  seed  outdir  state  images  decisions
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUPPLEMENT_DIR = REPO / "resources" / "spx_supplement_schedules"
MANIFEST = SUPPLEMENT_DIR / "manifest.tsv"
SPANS_DIR = SUPPLEMENT_DIR / "meancache_spans"
BASE_SCHEDULE_DIR = REPO / "resources" / "sp_cross_schedules"
PROMPT_FILE = REPO / "resources" / "prompts" / "partiprompts_full_eval1632_seed42.txt"
N_PROMPTS = 1632

SEEDS = {"flux": (41, 42, 43), "qwen": (42, 100042, 200042)}

#: The densification rows (`group == "dense"`, plan section 12) run ONE seed.
#: They buy within-rung spread on the schedule axis; the seed band is already
#: measured three ways over by the frozen rows, and 42 is the one seed both
#: models' frozen streams contain, so a dense row is directly comparable to the
#: seed-42 slice of the rung it densifies.
DENSE_SEED = 42

#: A robustness row is the rank-2 path of one gate, so it carries that gate's
#: own payload; a control row is a schedule-axis probe and carries `reuse`.
ROBUSTNESS_PAYLOAD = {
    "dicache_top1_r2": "di_two_anchor",
    "teacache_top1_r2": "reuse",
}

#: 50-step full-compute baselines, per (model, seed). Read off the 540-cell
#: wave's own `metrics.json` files, so the supplement evaluates against exactly
#: the references the main SPX cells did.
GT_DIRS = {
    ("flux", 41): "cache_results/flux/golden_path_seed_fixed_site_a_6f39bf5/parti_full_s41/original_gpseedfix_parti_full_n1632_s41_50_site_a_6f39bf5",
    ("flux", 42): "cache_results/flux/golden_path_seed_fixed_site_a_6f39bf5/parti_full_s42/original_gpseedfix_parti_full_n1632_s42_50_site_a_6f39bf5",
    ("flux", 43): "cache_results/flux/golden_path_seed_fixed_site_a_6f39bf5/parti_full_s43/original_gpseedfix_parti_full_n1632_s43_50_site_a_6f39bf5",
    ("qwen", 42): "cache_results/qwen_image/cross_dataset_qwen_xds_33fdaf0/originals/parti_full/qwen_parti_full_original50_s42_qwen_xds_33fdaf0",
    ("qwen", 100042): "cache_results/qwen_image/qwen_family_completion_v1/stage1/H1/parti_full/original_s100042",
    ("qwen", 200042): "cache_results/qwen_image/qwen_family_completion_v1/stage1/H2/parti_full/original_s200042",
}


def data_root() -> Path:
    return Path(os.environ.get("DATA", "outputs"))


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def enumerate_cells(
    *, spx_root: Path, correction_root: Path
) -> list[dict[str, object]]:
    """Every run of the supplement wave, in submission order (qwen first)."""

    cells: list[dict[str, object]] = []
    for row in read_manifest():
        group = row["group"]
        if group in ("control", "off_budget", "dense"):
            payload = "reuse"
        elif group == "robustness":
            payload = ROBUSTNESS_PAYLOAD[row["name"]]
        else:  # pragma: no cover - the frozen manifest has no other group
            raise SystemExit(f"unknown manifest group: {group}")
        model, budget_k = row["model"], int(row["target_k"])
        for seed in (DENSE_SEED,) if group == "dense" else SEEDS[model]:
            cells.append(
                {
                    "group": group,
                    "model": model,
                    "k": budget_k,
                    "schedule": row["name"],
                    "payload": payload,
                    "seed": seed,
                    "schedule_dir": SUPPLEMENT_DIR,
                    "spans": None,
                    "outdir": spx_root
                    / model
                    / f"k{budget_k}"
                    / f"{row['name']}x{payload}_s{seed}",
                }
            )

    for model in ("flux", "qwen"):
        for budget_k in (29, 37, 41):
            spans = SPANS_DIR / f"{model}_k{budget_k}.json"
            for seed in SEEDS[model]:
                cells.append(
                    {
                        "group": "correction",
                        "model": model,
                        "k": budget_k,
                        "schedule": "meancache",
                        "payload": "mean_avg_vel",
                        "seed": seed,
                        "schedule_dir": BASE_SCHEDULE_DIR,
                        "spans": spans,
                        "outdir": correction_root
                        / model
                        / f"k{budget_k}"
                        / f"meancachexmean_avg_vel_s{seed}",
                    }
                )

    # qwen is ~3x the FLUX cost per image, so it leads the wave.
    cells.sort(key=lambda c: (c["model"] != "qwen", c["k"], c["group"], c["schedule"], c["seed"]))
    return cells


def count_artifacts(outdir: Path) -> tuple[int, int]:
    if not outdir.is_dir():
        return 0, 0
    images = sum(
        1 for p in outdir.glob("img_*.png") if ".tmp." not in p.name
    )
    decisions = sum(
        1 for p in outdir.glob("decisions_*.json") if ".tmp." not in p.name
    )
    return images, decisions


def cell_state(outdir: Path, expected: int) -> tuple[str, int, int]:
    images, decisions = count_artifacts(outdir)
    complete = images >= expected and decisions >= expected
    return ("complete" if complete else "pending"), images, decisions


def job_name(cell: dict[str, object]) -> str:
    """Unique per run; `squeue` name matching is how the submitter deduplicates.

    The cell basename repeats across models and K levels, so both must be in
    the name (a zombie CG job keeps its name in the queue).
    """

    return (
        f"spxs_{cell['model'][:1]}{cell['k']}_"
        f"{cell['schedule']}x{cell['payload']}_s{cell['seed']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "eval_plan"):
        p = sub.add_parser(name)
        p.add_argument(
            "--group",
            choices=["control", "off_budget", "robustness", "correction", "dense"],
        )
        p.add_argument("--model", choices=["flux", "qwen"])
        p.add_argument("--k", type=int, choices=[29, 37, 41])
        p.add_argument("--pending", action="store_true", help="only rows that are not complete")
        p.add_argument("--expected", type=int, default=N_PROMPTS)
        p.add_argument("--spx_root", type=Path, default=None)
        p.add_argument("--correction_root", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = data_root()
    spx_root = args.spx_root or root / "sp_cross"
    correction_root = args.correction_root or root / "sp_cross_spans"

    cells = enumerate_cells(spx_root=spx_root, correction_root=correction_root)
    if args.group:
        cells = [c for c in cells if c["group"] == args.group]
    if args.model:
        cells = [c for c in cells if c["model"] == args.model]
    if args.k:
        cells = [c for c in cells if c["k"] == args.k]

    for cell in cells:
        state, images, decisions = cell_state(cell["outdir"], args.expected)
        if args.pending and state == "complete":
            continue
        if args.command == "eval_plan":
            if state != "complete":
                continue
            gt = root / GT_DIRS[(cell["model"], cell["seed"])]
            metrics = cell["outdir"] / "metrics.json"
            print(
                "\t".join(
                    [
                        cell["group"],
                        str(cell["outdir"]),
                        str(gt),
                        "done" if metrics.is_file() else "pending",
                        job_name(cell),
                    ]
                )
            )
            continue
        print(
            "\t".join(
                [
                    cell["group"],
                    cell["model"],
                    str(cell["k"]),
                    cell["schedule"],
                    cell["payload"],
                    str(cell["seed"]),
                    str(cell["outdir"]),
                    str(cell["schedule_dir"]),
                    str(cell["spans"]) if cell["spans"] else "-",
                    state,
                    str(images),
                    str(decisions),
                    job_name(cell),
                ]
            )
        )


if __name__ == "__main__":
    main()
