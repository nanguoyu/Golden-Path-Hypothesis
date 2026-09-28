#!/usr/bin/env python3
"""Cell inventory for P4 of the schedule-search experiment (`docs/schedule_search_plan_zh.md` S4).

P3 delivers a short list of distinct fixed schedules per (model, K). P4 runs
each of them through the image baseline-matrix protocol: 4 datasets x 3 seed
streams per model, `reuse` payload, evaluated against the matrix's own
full-compute references.

`--payload` selects one of the five payloads the SPX cross supports; it is part
of the cell directory name (`<name>x<payload>_s<seed>`, the layout
`RUN/slurm_sp_cross.sh` writes) and of the job name for every payload but
`reuse`, whose names are the ones the wave already queued.

Input is one plain text file, one schedule per line::

    # model  K   name        bitstring
    flux     41  ss_anneal   00010110111110111111011111111111111111101111111110
    qwen     29  ss_greedy   0001111000100110111001111001011100100111111101011

`name` is the schedule's identity everywhere downstream (cell directory, job
name, `schedule` column of the staged table). `ss_<algorithm>` is the plan's
convention; any name without whitespace works.

The cell grid is the schedules file crossed with::

    dataset x seed   -- DrawBench 200 / Parti 1632 / GenEval-style 553 /
                        DiffusionDB clean10k 10000, three seed streams each

The per-(model, dataset) prompt file and the per-(model, dataset, seed)
full-compute reference are NOT re-declared here: they are imported from
`analysis/build_sencache_recal_cells.py`, which is the in-repo table the
matrix's own re-run wave was built from. One copy, so a cell and the matrix
cannot disagree about which reference a run is paired with.

Usage::

    python RUN/schedule_search_eval_cells.py materialize --schedules FILE
    python RUN/schedule_search_eval_cells.py plan --schedules FILE
    python RUN/schedule_search_eval_cells.py plan --schedules FILE --pending --model qwen
    python RUN/schedule_search_eval_cells.py eval_plan --schedules FILE

`materialize` writes one `<model>_k<K>_<name>.txt` bitstring file per schedule
into `--schedule_dir` (default `resources/schedule_search/schedules`), the
layout `RUN/slurm_sp_cross.sh` resolves `--schedule` against.

`plan` prints one TSV row per generation cell::

    model k name payload dataset seed schedule_file prompt_file n_prompts
    base_output_dir outdir walltime state images decisions job_name

`eval_plan` prints one TSV row per complete cell::

    outdir gt_dir prompt_file n_prompts eval_shards eval_array_tasks
    metrics_state gt_state gt_cluster job_name
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.build_sencache_recal_cells import (  # noqa: E402
    DATASETS,
    REFERENCES,
    SEEDS,
)

#: The payload axis of the SPX cross (`RUN/multi_gpu_sp_cross.sh`); `reuse` is
#: the residual reuse the P4 wave was run with.
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_avg_vel", "di_two_anchor")
DEFAULT_PAYLOAD = "reuse"
NUM_STEPS = 50
#: {0, 1, 2, 49} are full for every searched schedule (plan S2, and
#: `resources/schedule_search/config.v1.json` `spaces.*.forced_full_steps`).
FORCED_FULL = (0, 1, 2, 49)
DATASET_ORDER = (
    "drawbench_full",
    "parti_full",
    "geneval_style",
    "diffusiondb_clean10k",
)
DATASET_TAG = {
    "drawbench_full": "db",
    "parti_full": "pt",
    "geneval_style": "ge",
    "diffusiondb_clean10k": "dd",
}
#: (metric shards, 4-GPU array tasks) per prompt count. DiffusionDB's 16/4 is
#: the shape the clean10k wave itself used
#: (`docs/cross_model_diffusiondb_clean10k_baseline_extension_plan_zh.md` S9.1);
#: the three small datasets fit one node's four GPUs.
EVAL_SHARDS = {200: (4, 1), 553: (4, 1), 1632: (4, 1), 10000: (16, 4)}

#: Measured seconds per image, `resources/cross_model_multiseed_stage_e_results/timing.tsv`
#: (`latency_s_mean`). `full` is the no-cache run the plan's cost table quotes
#: (S3); the per-K numbers are BudCache, the matrix method that is also a fixed
#: schedule with the `reuse` payload, so it is the homologous measurement for
#: these cells. A forecast payload is more expensive than reuse: the same table
#: measures hicache_o2 (= hermite_o2) at 3.881 s/img where BudCache is 2.374 on
#: flux k41, and 13.128 vs 6.694 on qwen k41; `PAYLOAD_COST` scales the
#: walltime by the larger of the two ratios per payload.
PAYLOAD_COST = {"reuse": 1.0, "mean_avg_vel": 1.1, "di_two_anchor": 1.1, "taylor_o1": 1.6, "hermite_o2": 2.0}
PER_IMAGE_SECONDS = {
    ("flux", 0): 10.189, ("flux", 29): 4.689, ("flux", 37): 3.155, ("flux", 41): 2.374,
    ("qwen", 0): 33.449, ("qwen", 29): 14.595, ("qwen", 37): 9.332, ("qwen", 41): 6.694,
}
#: Whole node, four shards; 1.25 is queue-side slack and 1800 s the model load.
GEN_GPUS = 4
GEN_SLACK = 1.25
GEN_LOAD_SECONDS = 1800
GEN_MAX_HOURS = 24


def per_image_seconds(model: str, budget_k: int) -> float:
    """Measured cost of one cached image; interpolated for an unmeasured K."""

    measured = PER_IMAGE_SECONDS.get((model, budget_k))
    if measured is not None:
        return measured
    full = PER_IMAGE_SECONDS[(model, 0)]
    return full * (NUM_STEPS - budget_k) / NUM_STEPS


def walltime(model: str, budget_k: int, n_prompts: int, payload: str = DEFAULT_PAYLOAD) -> str:
    """`--time` for one generation cell, rounded up to the hour."""

    gpu_seconds = n_prompts * per_image_seconds(model, budget_k) * PAYLOAD_COST[payload]
    seconds = GEN_LOAD_SECONDS + gpu_seconds / GEN_GPUS * GEN_SLACK
    hours = min(GEN_MAX_HOURS, max(2, -(-int(seconds) // 3600)))
    return f"{hours:02d}:00:00"


def data_root() -> Path:
    return Path(os.environ.get("DATA", "outputs"))


def expand(text: str) -> str:
    return text.replace("$DATA", str(data_root()))


def read_schedules(path: Path) -> list[dict[str, object]]:
    """Parse and validate the P3 delivery list."""

    out: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise SystemExit(f"{path}:{lineno}: expected 'model K name bits', got {raw!r}")
        model, budget_text, name, bits = fields
        if model not in SEEDS:
            raise SystemExit(f"{path}:{lineno}: unknown model {model!r}")
        if not budget_text.isdigit():
            raise SystemExit(f"{path}:{lineno}: K must be an integer, got {budget_text!r}")
        budget_k = int(budget_text)
        if len(bits) != NUM_STEPS or set(bits) - {"0", "1"}:
            raise SystemExit(
                f"{path}:{lineno}: schedule must be {NUM_STEPS} characters of 0/1"
            )
        if bits.count("1") != budget_k:
            raise SystemExit(
                f"{path}:{lineno}: {name} caches {bits.count('1')} steps, K says {budget_k}"
            )
        wrong = [step for step in FORCED_FULL if bits[step] != "0"]
        if wrong:
            raise SystemExit(f"{path}:{lineno}: {name} caches forced-full steps {wrong}")
        key = (model, budget_k, name)
        if key in seen:
            raise SystemExit(f"{path}:{lineno}: duplicate ({model}, k{budget_k}, {name})")
        seen.add(key)
        out.append({"model": model, "k": budget_k, "name": name, "bits": bits})
    if not out:
        raise SystemExit(f"{path}: no schedules")
    return out


def schedule_file(schedule_dir: Path, cell: dict) -> Path:
    return schedule_dir / f"{cell['model']}_k{cell['k']}_{cell['name']}.txt"


def materialize(schedules: list[dict], schedule_dir: Path) -> int:
    """Write the bitstrings out in the `<model>_k<K>_<name>.txt` layout."""

    schedule_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cell in schedules:
        target = schedule_file(schedule_dir, cell)
        text = f"{cell['bits']}\n"
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            continue
        target.write_text(text, encoding="utf-8")
        written += 1
        print(f"[cells] wrote {target}")
    print(f"[cells] {len(schedules)} schedules, {written} files written")
    return 0


def enumerate_cells(
    schedules: list[dict],
    root: Path,
    schedule_dir: Path,
    payload: str = DEFAULT_PAYLOAD,
) -> list[dict]:
    cells: list[dict] = []
    for entry in schedules:
        model = entry["model"]
        for dataset in DATASET_ORDER:
            prompt_file, n_prompts, _array_tasks = DATASETS[(model, dataset)]
            for seed in SEEDS[model]:
                gt_dir, gt_cluster = REFERENCES[(model, dataset, seed)]
                shards, array_tasks = EVAL_SHARDS[n_prompts]
                cells.append(
                    {
                        "name": entry["name"],
                        "payload": payload,
                        "model": model,
                        "k": entry["k"],
                        "dataset": dataset,
                        "seed": seed,
                        "prompt_file": prompt_file,
                        "n_prompts": n_prompts,
                        "schedule_file": schedule_file(schedule_dir, entry),
                        "base_output_dir": root / dataset,
                        "outdir": root
                        / dataset
                        / model
                        / f"k{entry['k']}"
                        / f"{entry['name']}x{payload}_s{seed}",
                        "gt_dir": Path(expand(gt_dir)),
                        "gt_cluster": gt_cluster,
                        "eval_shards": shards,
                        "eval_array_tasks": array_tasks,
                    }
                )
    # Qwen is ~3.3x the FLUX cost per image and DiffusionDB is 6x the next
    # dataset, so the long poles are submitted first.
    cells.sort(
        key=lambda c: (
            c["model"] != "qwen",
            -c["n_prompts"],
            c["k"],
            c["name"],
            c["dataset"],
            c["seed"],
        )
    )
    return cells


def count_artifacts(outdir: Path) -> tuple[int, int]:
    if not outdir.is_dir():
        return 0, 0
    images = sum(1 for p in outdir.glob("img_*.png") if ".tmp." not in p.name)
    decisions = sum(1 for p in outdir.glob("decisions_*.json") if ".tmp." not in p.name)
    return images, decisions


def job_name(cell: dict) -> str:
    """Unique per cell; `squeue` name matching is the submitter's deduplication.

    The `reuse` form carries no payload field, so the names of the cells the
    wave has already queued and finished are unchanged.
    """

    payload = cell.get("payload", DEFAULT_PAYLOAD)
    tail = "" if payload == DEFAULT_PAYLOAD else f"_{payload}"
    return (
        f"sse_{cell['model'][:1]}{cell['k']}_{DATASET_TAG[cell['dataset']]}_"
        f"{cell['name']}{tail}_s{cell['seed']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "plan", "eval_plan"):
        p = sub.add_parser(name)
        p.add_argument("--schedules", type=Path, required=True)
        p.add_argument(
            "--schedule_dir",
            type=Path,
            default=REPO / "resources" / "schedule_search" / "schedules",
        )
        p.add_argument("--payload", choices=PAYLOADS, default=DEFAULT_PAYLOAD)
        if name == "materialize":
            continue
        p.add_argument("--model", choices=sorted(SEEDS))
        p.add_argument("--dataset", choices=DATASET_ORDER)
        p.add_argument("--k", type=int)
        p.add_argument("--name")
        p.add_argument("--root", type=Path, default=None)
        p.add_argument(
            "--pending", action="store_true", help="only cells that are not complete"
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    schedules = read_schedules(args.schedules)
    if args.command == "materialize":
        return materialize(schedules, args.schedule_dir)

    root = args.root or data_root() / "schedule_search_eval"
    cells = enumerate_cells(schedules, root, args.schedule_dir, args.payload)
    if args.model:
        cells = [c for c in cells if c["model"] == args.model]
    if args.dataset:
        cells = [c for c in cells if c["dataset"] == args.dataset]
    if args.k:
        cells = [c for c in cells if c["k"] == args.k]
    if args.name:
        cells = [c for c in cells if c["name"] == args.name]

    for cell in cells:
        images, decisions = count_artifacts(cell["outdir"])
        state = (
            "complete"
            if images >= cell["n_prompts"] and decisions >= cell["n_prompts"]
            else "pending"
        )
        if args.pending and state == "complete":
            continue
        if args.command == "eval_plan":
            if state != "complete":
                continue
            metrics = cell["outdir"] / "metrics.json"
            print(
                "\t".join(
                    [
                        str(cell["outdir"]),
                        str(cell["gt_dir"]),
                        cell["prompt_file"],
                        str(cell["n_prompts"]),
                        str(cell["eval_shards"]),
                        str(cell["eval_array_tasks"]),
                        "done" if metrics.is_file() else "pending",
                        "gt_present" if cell["gt_dir"].is_dir() else "gt_missing",
                        cell["gt_cluster"],
                        job_name(cell),
                    ]
                )
            )
            continue
        print(
            "\t".join(
                [
                    cell["model"],
                    str(cell["k"]),
                    cell["name"],
                    cell["payload"],
                    cell["dataset"],
                    str(cell["seed"]),
                    str(cell["schedule_file"]),
                    cell["prompt_file"],
                    str(cell["n_prompts"]),
                    str(cell["base_output_dir"]),
                    str(cell["outdir"]),
                    walltime(cell["model"], cell["k"], cell["n_prompts"], cell["payload"]),
                    state,
                    str(images),
                    str(decisions),
                    job_name(cell),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
