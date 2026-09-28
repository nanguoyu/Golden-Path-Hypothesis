#!/usr/bin/env python3
"""Stage the image SPX per-image metrics into two compact tables.

The 540-cell SPX wave plus the supplement wave live as one `metrics.json` per
cell on site_a; the matrix's native gate runs live on site_b. Every
statistic the supplement asks for is a per-prompt paired difference, so both
sides have to come down to one row per (cell, seed, prompt) before anything can
be computed. Two tables, both written gzipped:

`perprompt_spx_<model>.tsv.gz`
    schedule payload k seed prompt_idx psnr ssim lpips image_reward clip

`perprompt_native_<model>.tsv.gz`
    method k seed prompt_idx psnr ssim lpips image_reward clip path

    The runs come from `discovery_path_counts.tsv`'s own `run_dirs` column, not
    from a directory convention: the matrix's Qwen gate waves are spread over
    three campaign roots and two clusters, and those directories are by
    definition the population each gate's modal path was counted over. Run the
    mode on each cluster and concatenate; a run that is not on this cluster is
    named and skipped.

    `path` is the schedule the gate actually realised on that prompt, read off
    the run's own `decisions_<i>.json`. It is what splits P4's pairs into the
    modal subset (where the fixed-path run and the gate run are the same
    computation, so the difference is a deterministic zero) and the off-modal
    subset (where the comparison carries information).

Both modes are read-only over the run directories. Run them as CPU batch jobs;
a serial scan of ~350 x 1632 JSON files is not login-node work.

    python analysis/stage_spx_perprompt.py spx    --out resources/spx
    python analysis/stage_spx_perprompt.py native --out resources/spx
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.analyze_native_schedule_paths import schedule_from_payload  # noqa: E402
from analysis.sp_cross import parse_cell_name  # noqa: E402

METRICS = ("psnr", "ssim", "lpips", "image_reward", "clip")
MODELS = ("flux", "qwen")
GATES = ("seacache", "teacache", "sencache", "dicache")
BUDGETS = (29, 37, 41)
SEEDS = {"flux": (41, 42, 43), "qwen": (42, 100042, 200042)}
#: The matrix names the Qwen tree `qwen`; the SPX tree uses the same word, so
#: one model key covers both. `qwen_image` only appears in reference paths.


def data_root() -> Path:
    return Path(os.environ.get("DATA", "outputs"))


def read_per_image(path: Path) -> tuple[list[int], dict[str, list]] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("indices")
    per_image = payload.get("per_image") or {}
    if not indices or not per_image:
        return None
    columns = {}
    for metric in METRICS:
        values = per_image.get(metric)
        if values is None or len(values) != len(indices):
            return None
        columns[metric] = values
    return [int(i) for i in indices], columns


def fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def stage_spx(out_dir: Path, roots: list[Path]) -> dict[str, int]:
    """One row per (schedule, payload, K, seed, prompt) over every SPX root."""

    counts: dict[str, int] = {}
    for model in MODELS:
        rows: list[str] = []
        seen: set[tuple[str, str, int, int]] = set()
        for root in roots:
            model_dir = root / model
            if not model_dir.is_dir():
                continue
            for budget_dir in sorted(model_dir.iterdir()):
                if not budget_dir.name.startswith("k") or not budget_dir.name[1:].isdigit():
                    continue
                budget_k = int(budget_dir.name[1:])
                for cell_dir in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                    parsed = parse_cell_name(cell_dir.name)
                    if parsed is None:
                        continue
                    schedule, payload, seed = parsed
                    metrics_path = cell_dir / "metrics.json"
                    if not metrics_path.is_file():
                        continue
                    key = (schedule, payload, budget_k, seed)
                    if key in seen:
                        # A later root wins: the correction root holds the
                        # re-run of a cell the original root also has.
                        rows = [
                            line
                            for line in rows
                            if not line.startswith(
                                f"{schedule}\t{payload}\t{budget_k}\t{seed}\t"
                            )
                        ]
                    read = read_per_image(metrics_path)
                    if read is None:
                        print(f"[stage] SKIP (no per_image): {metrics_path}", flush=True)
                        continue
                    indices, columns = read
                    seen.add(key)
                    for position, prompt_idx in enumerate(indices):
                        rows.append(
                            "\t".join(
                                [
                                    schedule,
                                    payload,
                                    str(budget_k),
                                    str(seed),
                                    str(prompt_idx),
                                    *[fmt(columns[m][position]) for m in METRICS],
                                ]
                            )
                        )
                    print(f"[stage] {model} k{budget_k} {cell_dir.name}: {len(indices)}", flush=True)
        path = out_dir / f"perprompt_spx_{model}.tsv.gz"
        header = "\t".join(["schedule", "payload", "k", "seed", "prompt_idx", *METRICS])
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(header + "\n")
            if rows:
                handle.write("\n".join(rows) + "\n")
        counts[model] = len(rows)
        print(f"[stage] wrote {path} ({len(rows)} rows, {len(seen)} runs)", flush=True)
    return counts


def realised_path(decisions: Path) -> str | None:
    """The 50-bit path a run actually walked on one prompt.

    `schedule_from_payload` is the same reader `discovery_path_counts.tsv` was
    built with, including its `action` / `u` fallback -- the modal path and the
    per-prompt path have to be decided by one function or the modal subset is
    not the set the counts named.
    """

    try:
        return schedule_from_payload(
            json.loads(decisions.read_text(encoding="utf-8")), decisions
        )
    except (ValueError, KeyError):
        return None


def read_seed(directory: Path) -> int | None:
    """The seed a run was generated with, off its own decisions file."""

    for candidate in sorted(directory.glob("decisions_*.json"))[:1]:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        seed = payload.get("seed")
        if seed is not None:
            # Per-image seeds are base + global index; the run's base seed is
            # what the SPX cells key on.
            index = payload.get("prompt_idx")
            if index is not None:
                return int(seed) - int(index)
            return int(seed)
    return None


#: The matrix generated into stage_d and evaluated into a parallel stage_e tree
#: with an identical suffix, so a run directory names its decisions but not its
#: metrics. Campaign roots outside that pair keep both in one directory.
METRICS_TREE_SWAP = (
    ("cross_model_multiseed_stage_d_v1", "cross_model_multiseed_stage_e_v1"),
)


def metrics_for(gen_dir: Path, overlay: Path | None = None) -> Path | None:
    """The metrics.json belonging to a generation directory, wherever it lives.

    `overlay` covers the case where a run's decisions and its metrics ended up
    on different clusters: six Qwen gate runs were relayed to site_b for the path
    census (decisions only) while their evaluation stayed on site_a, so neither
    cluster can pair them alone. Relaying the metrics -- 1.2 MB each against
    1.6 GB of decisions -- and pointing the overlay at them keeps both original
    trees untouched.
    """

    local = gen_dir / "metrics.json"
    if local.is_file():
        return local
    if overlay is not None:
        candidate = overlay / gen_dir.name / "metrics.json"
        if candidate.is_file():
            return candidate
    text = str(gen_dir)
    for gen_tree, eval_tree in METRICS_TREE_SWAP:
        if gen_tree in text:
            candidate = Path(text.replace(gen_tree, eval_tree)) / "metrics.json"
            if candidate.is_file():
                return candidate
    return None


def discovery_run_dirs(path: Path) -> dict[tuple[str, int, str], list[str]]:
    """(model, K, gate) -> the run directories its discovery counts came from.

    This column, not a directory layout, is the authority on which runs define a
    gate's modal path: the matrix's qwen gate waves live under three different
    campaign roots and on two different clusters, and the modal path was counted
    over exactly these directories.
    """

    out: dict[tuple[str, int, str], list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["model"], int(row["target_k"]), row["method"])
            if key in out:
                continue
            out[key] = [text for text in row["run_dirs"].split(";") if text]
    return out


def stage_native(out_dir: Path, discovery: Path,
                 overlay: Path | None = None) -> dict[str, int]:
    """One row per (gate, K, seed, prompt) with the gate's realised path.

    Runs whose directory is not on this cluster are skipped and named; the two
    clusters' outputs are concatenated afterwards.
    """

    run_dirs = discovery_run_dirs(discovery)
    counts: dict[str, int] = {}
    for model in MODELS:
        rows: list[str] = []
        found = 0
        for budget_k in BUDGETS:
            for gate in GATES:
                for directory in run_dirs.get((model, budget_k, gate), []):
                    gen_dir = Path(directory)
                    if not gen_dir.is_dir():
                        print(f"[stage] not on this cluster: {directory}", flush=True)
                        continue
                    metrics_path = metrics_for(gen_dir, overlay)
                    if metrics_path is None:
                        print(f"[stage] no metrics for: {directory}", flush=True)
                        continue
                    read = read_per_image(metrics_path)
                    if read is None:
                        print(f"[stage] SKIP (no per_image): {directory}", flush=True)
                        continue
                    seed = read_seed(gen_dir)
                    if seed is None:
                        print(f"[stage] SKIP (no seed): {directory}", flush=True)
                        continue
                    indices, columns = read
                    found += 1
                    missing_paths = 0
                    for position, prompt_idx in enumerate(indices):
                        decisions = gen_dir / f"decisions_{prompt_idx:05d}.json"
                        path = realised_path(decisions) if decisions.is_file() else None
                        if path is None:
                            missing_paths += 1
                        rows.append(
                            "\t".join(
                                [
                                    gate,
                                    str(budget_k),
                                    str(seed),
                                    str(prompt_idx),
                                    *[fmt(columns[m][position]) for m in METRICS],
                                    path or "",
                                ]
                            )
                        )
                    print(
                        f"[stage] {model} k{budget_k} {gate} s{seed}: {len(indices)} rows, "
                        f"{missing_paths} without a path  <- {directory}",
                        flush=True,
                    )
        path = out_dir / f"perprompt_native_{model}.tsv.gz"
        header = "\t".join(["method", "k", "seed", "prompt_idx", *METRICS, "path"])
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write(header + "\n")
            if rows:
                handle.write("\n".join(rows) + "\n")
        counts[model] = len(rows)
        print(f"[stage] wrote {path} ({len(rows)} rows, {found} runs found here)", flush=True)
    return counts


def merge_spx(out_dir: Path, inputs: list[Path]) -> dict[str, int]:
    """Concatenate staged SPX tables; a later file's cell replaces an earlier one.

    The wave runs on two clusters -- Qwen on site_a, FLUX on site_c -- and the
    original 540 cells live on site_a either way, so one model's rows arrive in
    two pieces. Precedence is the same rule the multi-root scan uses: the last
    input wins, which is how the per-edge-span re-run replaces the global-span
    cell it supersedes.
    """

    counts: dict[str, int] = {}
    for model in MODELS:
        header = None
        rows: dict[tuple[str, str, str, str], list[str]] = {}
        for path in inputs:
            candidate = path / f"perprompt_spx_{model}.tsv.gz" if path.is_dir() else path
            if not candidate.is_file() or model not in candidate.name:
                continue
            with gzip.open(candidate, "rt", encoding="utf-8", newline="") as handle:
                lines = handle.read().splitlines()
            if not lines:
                continue
            header = header or lines[0]
            replaced = set()
            for line in lines[1:]:
                if not line:
                    continue
                schedule, payload, budget_k, seed, _rest = line.split("\t", 4)
                key = (schedule, payload, budget_k, seed)
                if key not in replaced:
                    replaced.add(key)
                    rows.pop(key, None)
                    rows[key] = []
                rows[key].append(line)
            print(f"[merge] {candidate}: {len(lines) - 1} rows, {len(replaced)} cells", flush=True)
        target = out_dir / f"perprompt_spx_{model}.tsv.gz"
        flat = [line for key in sorted(rows) for line in rows[key]]
        with gzip.open(target, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write((header or "") + "\n")
            if flat:
                handle.write("\n".join(flat) + "\n")
        counts[model] = len(flat)
        print(f"[merge] wrote {target} ({len(flat)} rows, {len(rows)} cells)", flush=True)
    return counts


def merge_native(out_dir: Path, inputs: list[Path]) -> dict[str, int]:
    """Same, for the native tables: the two clusters hold disjoint gate runs."""

    counts: dict[str, int] = {}
    for model in MODELS:
        header = None
        rows: dict[tuple[str, str, str], list[str]] = {}
        for path in inputs:
            candidate = path / f"perprompt_native_{model}.tsv.gz" if path.is_dir() else path
            if not candidate.is_file():
                continue
            with gzip.open(candidate, "rt", encoding="utf-8", newline="") as handle:
                lines = handle.read().splitlines()
            if not lines:
                continue
            header = header or lines[0]
            replaced = set()
            for line in lines[1:]:
                if not line:
                    continue
                method, budget_k, seed, _rest = line.split("\t", 3)
                key = (method, budget_k, seed)
                if key not in replaced:
                    replaced.add(key)
                    rows.pop(key, None)
                    rows[key] = []
                rows[key].append(line)
            print(f"[merge] {candidate}: {len(lines) - 1} rows, {len(replaced)} runs", flush=True)
        target = out_dir / f"perprompt_native_{model}.tsv.gz"
        flat = [line for key in sorted(rows) for line in rows[key]]
        with gzip.open(target, "wt", encoding="utf-8", newline="\n") as handle:
            handle.write((header or "") + "\n")
            if flat:
                handle.write("\n".join(flat) + "\n")
        counts[model] = len(flat)
        print(f"[merge] wrote {target} ({len(flat)} rows, {len(rows)} runs)", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["spx", "native", "merge_spx", "merge_native"])
    parser.add_argument("--out", type=Path, default=_ROOT / "resources" / "spx")
    parser.add_argument(
        "--spx_root",
        type=Path,
        action="append",
        default=None,
        help="repeatable; later roots win a duplicate cell (correction re-runs)",
    )
    parser.add_argument(
        "--metrics_overlay", type=Path, default=None,
        help="directory of <run_basename>/metrics.json, for runs whose metrics "
             "live on a different cluster than their decisions",
    )
    parser.add_argument(
        "--discovery",
        type=Path,
        default=_ROOT / "resources/sp_cross_schedules/discovery_path_counts.tsv",
    )
    parser.add_argument(
        "--inputs", type=Path, nargs="+", default=None,
        help="merge modes: staged directories or files, in precedence order "
             "(the last one carrying a cell wins)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = data_root()
    if args.mode == "merge_spx":
        merge_spx(args.out, [Path(p) for p in (args.inputs or [])])
    elif args.mode == "merge_native":
        merge_native(args.out, [Path(p) for p in (args.inputs or [])])
    elif args.mode == "spx":
        roots = args.spx_root or [root / "sp_cross", root / "sp_cross_spans"]
        stage_spx(args.out, [Path(p) for p in roots])
    else:
        stage_native(args.out, args.discovery, args.metrics_overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
