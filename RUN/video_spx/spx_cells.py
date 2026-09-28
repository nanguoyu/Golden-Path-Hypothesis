#!/usr/bin/env python3
"""Enumerate the video SPX cells and emit each one's runner command line.

One source of truth for three things the wrappers would otherwise each restate:

  * which (schedule, payload, dataset) triples exist -- read off
    `resources/video_spx_schedules/manifest.tsv`, including the
    `feasible_<payload>` columns, so an infeasible cell is never submitted;
  * what a cell is called and where it writes -- the plan's section 7 name,
    `<row>x<payload>_<dataset>_K<k>_s<base seed>`;
  * the exact argv the backbone's `baseline_screen_runner.py` gets, including
    the three video-SPX flags (`--meancache_schedule` for `mean_vel`,
    `--max_order 2` for `hermite_o2`, `--spx_relax_warmup` where a payload's
    warmup would otherwise refuse the row).

The two 2026-08-23 supplements (plan section 10) live here as well: the sixth
payload column `mean_vel_global`, submitted for the MeanCache row only, and the
first-step-preserving ladder rows, whose schedule JSON asks for the `reuse`
column alone. `--grid supplement` selects exactly those cells.

It runs on the login node: it reads a TSV, stats output directories and prints.
No model, no GPU, no heavy IO.

    python RUN/video_spx/spx_cells.py plan --grid smoke
    python RUN/video_spx/spx_cells.py argv --cell ham4xhermite_o2_penguin599_K37_s54
    python RUN/video_spx/spx_cells.py plan --references

`--dataset` and `--budget` are deliberately NOT passed to the runners: both
refuse either flag unless `--matrix_config` comes with it, and an SPX cell is
not a matrix cell. Its identity is the schedule row, the bitstring and the
payload (plan section 8, "not doing" list); the dataset is in the directory
name and pinned by the prompt manifest's digest inside `cell_identity.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEDULE_ROOT = REPO / "resources/video_spx_schedules"
MANIFEST = SCHEDULE_ROOT / "manifest.tsv"
PROMPT_MANIFEST = {
    "penguin599": "resources/hunyuan_video/evaluation/penguin599.json",
    "vbench944": "resources/hunyuan_video/evaluation/vbench944.json",
}
BASE_SEED = {"penguin599": 54, "vbench944": 42}
DATASETS = ("penguin599", "vbench944")
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_vel", "di_two_anchor")
BACKBONES = ("hunyuan_video", "wan21")
PROMPT_COUNT = 150

#: Supplement S2 (plan section 10). `mean_vel` was not one payload: MeanCache's
#: own row ran the offline search's per-edge JVP spans, every other row the
#: global span 4. This sixth column re-runs MeanCache's table on the global
#: span, so the column is homogeneous; it is submitted for the MeanCache row
#: ONLY, because the other rows' `mean_vel` cells already are the global-span
#: payload and are not re-run. Its transport is the span-stripped copy the
#: schedule builder writes under `<backbone>/global_span/`.
SPAN_VARIANT_PAYLOAD = "mean_vel_global"
SPAN_VARIANT_BASE = "mean_vel"
SPAN_VARIANT_ROWS = ("meancache",)
SPAN_VARIANT_DIR = "global_span"

#: Supplement S1 (plan section 10). The first-step-preserving Hamming ladder;
#: its rows carry `payload_columns == ["reuse"]` in the schedule JSON, which is
#: what `iter_cells` reads, so this tuple is only used by the `supplement` grid.
SUPPLEMENT_ROWS = ("ham2f", "ham4f", "ham8f")

#: Checkpoint locations. Read from the environment so `spx_env.sh` stays the one
#: place they are configured; the fallbacks are site_c's.
DEFAULT_DATA = "outputs"


def model_base() -> str:
    return os.environ.get("MODEL_BASE", f"{DEFAULT_DATA}/hf/HunyuanVideo")


def wan_ckpt_dir() -> str:
    return os.environ.get("WAN21_CKPT_DIR", f"{DEFAULT_DATA}/hf/Wan2.1-T2V-1.3B")

#: Plan section 5.1. W1 is the full-rank subgrid every variance statistic needs;
#: W1b is the rest of the rows, which only answer P2/P3 and therefore only need
#: the reuse column. Owner decision 1 runs the full cross (grid `full`), and
#: these two stay available for a partial wave.
W1_ROWS = ("shared", "budcache", "meancache", "di_top1", "dp_rho2", "ham4", "uniform")

#: The V3 smoke of plan section 7, widened to cover every payload column once
#: per backbone rather than the plan's two cells: `di_two_anchor` and the
#: relaxed MeanCache warmup are new code and are what the smoke is for.
SMOKE_CELLS = (
    ("meancache", "mean_vel"),      # per-edge spans, the payload's own table
    ("ham4", "hermite_o2"),         # fine slots on a foreign schedule
    ("dp_rho2", "reuse"),           # the zero-order column, with T3 retained
    ("di_top1", "di_two_anchor"),   # the new fixed-schedule DiCache entry
    ("uniform", "mean_vel"),        # mean_vel on a row that caches step 3
    ("shared", "taylor_o1"),        # first-order slots
)
SMOKE_BUDGET = 37

#: Plan section 5.5: whole latent paths are kept for the reuse column and for
#: MeanCache x mean_vel only, 20 pairs per cell.
T3_PROMPT_COUNT = 10


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def schedule_json(backbone: str, schedule_id: str) -> dict[str, Any]:
    return json.loads((SCHEDULE_ROOT / backbone / f"{schedule_id}.json")
                      .read_text(encoding="utf-8"))


def schedule_path(backbone: str, schedule_id: str, payload: str) -> str:
    """Repo-relative `--meancache_schedule` transport for a cell."""
    if payload == SPAN_VARIANT_PAYLOAD:
        return f"resources/video_spx_schedules/{backbone}/{SPAN_VARIANT_DIR}/{schedule_id}.json"
    return f"resources/video_spx_schedules/{backbone}/{schedule_id}.json"


def keeps_full_path(row: str, payload: str) -> bool:
    return payload == "reuse" or (row == "meancache" and payload == "mean_vel")


def cell_name(row: str, payload: str, dataset: str, budget: str, seed: int) -> str:
    return f"{row}x{payload}_{dataset}_{budget}_s{seed}"


def reference_name(dataset: str, seed: int) -> str:
    return f"refs_{dataset}_s{seed}"


def iter_cells(rows: list[dict[str, str]], *, grid: str,
               backbones: Sequence[str], datasets: Sequence[str],
               budgets: Sequence[str] | None) -> Iterator[dict[str, Any]]:
    smoke = {(row, payload) for row, payload in SMOKE_CELLS}
    for record in rows:
        backbone, row, budget = record["backbone"], record["row"], record["budget"]
        if backbone not in backbones:
            continue
        if budgets and budget not in budgets:
            continue
        off_budget = bool(int(record["off_budget"]))
        # The schedule JSON, not the row's off-budget flag, decides the columns:
        # supplement S1's rows are on-budget and still run one column only.
        columns = tuple(schedule_json(backbone, record["schedule_id"])["payload_columns"])
        if row in SPAN_VARIANT_ROWS:
            columns += (SPAN_VARIANT_PAYLOAD,)
        for payload in columns:
            base = SPAN_VARIANT_BASE if payload == SPAN_VARIANT_PAYLOAD else payload
            if not int(record[f"feasible_{base}"]):
                continue
            if grid == "supplement" and not (
                    row in SUPPLEMENT_ROWS or payload == SPAN_VARIANT_PAYLOAD):
                continue
            if grid == "w1" and (row not in W1_ROWS):
                continue
            if grid == "w1b" and (row in W1_ROWS or payload != "reuse"):
                continue
            if grid == "smoke":
                if (row, payload) not in smoke:
                    continue
                if int(budget[1:]) != SMOKE_BUDGET:
                    continue
            for dataset in datasets:
                seed = BASE_SEED[dataset]
                yield {
                    "backbone": backbone,
                    "row": row,
                    "payload": payload,
                    "budget": budget,
                    "dataset": dataset,
                    "seed": seed,
                    "schedule_id": record["schedule_id"],
                    "cache_count": int(record["cache_count"]),
                    "off_budget": off_budget,
                    "keep_full_path": keeps_full_path(row, payload),
                    "cell": cell_name(row, payload, dataset, budget, seed),
                }


def runner_argv(cell: dict[str, Any], *, output_dir: Path, prompt_count: int,
                prompt_indices: Sequence[int] | None, retain_t3: bool) -> list[str]:
    """The backbone runner's argv for one cell."""
    backbone, payload = cell["backbone"], cell["payload"]
    schedule = schedule_json(backbone, cell["schedule_id"])
    mode = schedule["payload_modes"][
        SPAN_VARIANT_BASE if payload == SPAN_VARIANT_PAYLOAD else payload]
    if backbone == "hunyuan_video":
        argv = ["hunyuan_video/baseline_screen_runner.py", "--mode", mode,
                "--model_base", model_base()]
    else:
        argv = ["wan21/baseline_screen_runner.py", "--mode", mode,
                "--ckpt_dir", wan_ckpt_dir()]
    argv += ["--prompt_manifest", PROMPT_MANIFEST[cell["dataset"]],
             "--seed", str(cell["seed"]),
             "--output_dir", str(output_dir),
             "--retain_trajectory", "--resume",
             "--shard_idx", "0", "--shard_count", "1"]
    if prompt_indices is not None:
        argv += ["--prompt_indices", ",".join(str(index) for index in prompt_indices)]
    else:
        argv += ["--limit", str(prompt_count)]

    steps = ",".join(str(step) for step in schedule["cache_steps"])
    if payload in (SPAN_VARIANT_BASE, SPAN_VARIANT_PAYLOAD):
        # The schedule JSON is itself the --meancache_schedule transport: it
        # carries `cache_steps` and, for MeanCache's own row, the per-edge
        # `jvp_spans` the search solved. Every other row has none, and runs the
        # global span. `mean_vel_global` points at the span-stripped copy, so
        # MeanCache's own table runs the global span too.
        argv += ["--meancache_schedule",
                 schedule_path(backbone, cell["schedule_id"], payload)]
        if payload == SPAN_VARIANT_PAYLOAD:
            argv += ["--meancache_jvp_span", str(schedule["jvp_span_global"])]
        if backbone == "hunyuan_video":
            argv += ["--cache_count", str(cell["cache_count"])]
        argv += ["--spx_relax_warmup"]
    else:
        argv += ["--cache_count", str(cell["cache_count"]), "--cache_steps", steps]
        if payload == "hermite_o2" and backbone == "hunyuan_video":
            # Wan's hicache_o2 pins max_order in methods_glue; the Hunyuan
            # runner takes it from the operator when no config drives the cell
            # (baseline_screen_runner.py `_validate`).
            argv += ["--max_order", "2"]
        if payload == "reuse" and backbone == "wan21":
            # Wan's budcache forbids steps 1-2 by search convention; three gate
            # rows cache them. `ReuseMethod` needs only a previous residual,
            # which is what the Hunyuan twin enforces.
            argv += ["--spx_relax_warmup"]
    if retain_t3 and cell["keep_full_path"]:
        argv += ["--t3_cached", "--t3_seed", str(cell["seed"]),
                 "--t3_prompt_count", str(T3_PROMPT_COUNT)]
    return argv


def reference_argv(backbone: str, dataset: str, *, output_dir: Path,
                   prompt_count: int, t3_prompt_count: int) -> list[str]:
    seed = BASE_SEED[dataset]
    if backbone == "hunyuan_video":
        argv = ["hunyuan_video/baseline_screen_runner.py", "--mode", "original",
                "--model_base", model_base()]
    else:
        argv = ["wan21/baseline_screen_runner.py", "--mode", "original",
                "--ckpt_dir", wan_ckpt_dir()]
    argv += ["--prompt_manifest", PROMPT_MANIFEST[dataset],
             "--seed", str(seed), "--limit", str(prompt_count),
             "--output_dir", str(output_dir),
             "--retain_trajectory", "--t3_seed", str(seed),
             "--t3_prompt_count", str(t3_prompt_count),
             "--resume", "--shard_idx", "0", "--shard_count", "1"]
    return argv


def completed(directory: Path, expected: int) -> tuple[bool, int, int]:
    if not directory.is_dir():
        return False, 0, 0
    videos = len(list(directory.glob("video_*.mp4")))
    decisions = len(list(directory.glob("decisions_*.json")))
    return (videos >= expected and decisions >= expected), videos, decisions


def root_for(backbone: str, data: str) -> Path:
    return Path(data) / ("hunyuan_video" if backbone == "hunyuan_video" else "wan21") / "spx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "argv", "manifest"))
    parser.add_argument("--grid", choices=("full", "w1", "w1b", "smoke", "supplement"),
                        default="full",
                        help="supplement: only the two 2026-08-23 supplement families -- the "
                             "first-step-preserving Hamming ladder in the reuse column and "
                             "the MeanCache row's global-span mean_vel cell")
    parser.add_argument("--references", action="store_true",
                        help="enumerate the uncached reference runs instead of the cells")
    parser.add_argument("--backbone", action="append", choices=BACKBONES)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--budget", action="append", choices=("K29", "K37", "K41"))
    parser.add_argument("--cell", help="argv: the cell (or reference) name to emit")
    parser.add_argument("--data", default=os.environ.get("DATA", DEFAULT_DATA))
    parser.add_argument("--prompt_count", type=int, default=PROMPT_COUNT)
    parser.add_argument("--prompt_indices", type=str, default=None,
                        help="comma list; overrides --prompt_count (smoke runs)")
    parser.add_argument("--t3_prompt_count", type=int, default=T3_PROMPT_COUNT)
    parser.add_argument("--retain_t3", action="store_true",
                        help="cells: keep whole latent paths on the reuse column and on "
                             "meancache x mean_vel (plan section 5.5)")
    parser.add_argument("--include_complete", action="store_true",
                        help="plan: list cells whose output directory is already full")
    parser.add_argument("--out", type=Path,
                        help="manifest: where to write the evaluator's task manifest")
    return parser


def write_task_manifest(dataset: str, out: Path) -> Path:
    """The evaluator's `--task_manifest`: one JSON object per line, `task_idx`
    contiguous from 0, in the frozen evaluation manifest's own order.

    `--prompts` (one per line) cannot be used: two Penguin prompts contain a
    literal newline, and reading by line would shift 310 of them onto the wrong
    video. The order is the manifest's, which is the order the runners generate
    in, so `task_idx` is the `video_%05d.mp4` index.
    """
    payload = json.loads((REPO / PROMPT_MANIFEST[dataset]).read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(payload["items"]):
            handle.write(json.dumps({"task_idx": index, "prompt": str(item["prompt"])},
                                    ensure_ascii=False) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "manifest":
        datasets = tuple(args.dataset or DATASETS)
        if len(datasets) != 1 or args.out is None:
            raise SystemExit("manifest needs exactly one --dataset and an --out path")
        print(write_task_manifest(datasets[0], Path(args.out)))
        return 0
    backbones = tuple(args.backbone or BACKBONES)
    datasets = tuple(args.dataset or DATASETS)
    indices = ([int(item) for item in args.prompt_indices.split(",")]
               if args.prompt_indices else None)
    expected = len(indices) if indices is not None else int(args.prompt_count)

    if args.references:
        entries = []
        for backbone in backbones:
            for dataset in datasets:
                seed = BASE_SEED[dataset]
                name = reference_name(dataset, seed)
                directory = root_for(backbone, args.data) / name
                entries.append({"backbone": backbone, "cell": name,
                                "dir": directory, "dataset": dataset,
                                "argv": reference_argv(
                                    backbone, dataset, output_dir=directory,
                                    prompt_count=expected,
                                    t3_prompt_count=int(args.t3_prompt_count))})
    else:
        rows = read_manifest()
        entries = []
        for cell in iter_cells(rows, grid=args.grid, backbones=backbones,
                               datasets=datasets, budgets=tuple(args.budget or ())):
            directory = root_for(cell["backbone"], args.data) / cell["cell"]
            entries.append({**cell, "dir": directory,
                            "argv": runner_argv(cell, output_dir=directory,
                                                prompt_count=expected,
                                                prompt_indices=indices,
                                                retain_t3=bool(args.retain_t3))})

    if args.action == "argv":
        if not args.cell:
            raise SystemExit("argv needs --cell")
        chosen = [entry for entry in entries if entry["cell"] == args.cell]
        if len(chosen) != 1:
            raise SystemExit(f"{args.cell!r} matched {len(chosen)} cells; narrow it with "
                             f"--backbone / --references / --prompt_indices")
        print(" ".join(shlex.quote(token) for token in chosen[0]["argv"]))
        return 0

    for entry in entries:
        done, videos, decisions = completed(entry["dir"], expected)
        if done and not args.include_complete:
            continue
        print("\t".join([entry["backbone"], entry["cell"], str(entry["dir"]),
                         "complete" if done else "pending",
                         f"{videos}/{expected}", f"{decisions}/{expected}"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
