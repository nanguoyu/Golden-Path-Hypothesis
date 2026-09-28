#!/usr/bin/env python3
"""Collect the SenCache (threshold_start, threshold_main) admissible frontier.

Reads the count-only sweep output written by
RUN/submit_sencache_recal_frontier.sh (image lane) or
RUN/submit_wan21_sencache_frontier.sh (Wan2.1 lane) and emits, per
(model/dataset, start, main): mean realized K, its spread, and the first-ten-step
caching statistics that the plan's completion check needs.

`select` then reads that table and picks, per (model, budget), the strictest
start whose best main lands within tolerance of the target -- and, where no
admissible start reaches the target, both arms of the plan's section 9.1 trade.

Plan: docs/sencache_recalibration_plan_zh.md sections 3 and 9.1.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SWITCH_STEPS = 10  # round(50 * switch_ratio=0.2) -- the threshold_start window
BUDGETS = (29, 37, 41)
NUM_STEPS = 50
FIRST_ENHANCE = 3
UPSTREAM_MAX_SKIP = 10
#: The warmup differs by lane: the image calibration scripts default to 1 and
#: the matrix passes 1 explicitly, while the Wan lane freezes 3. It only moves
#: the unprotected bound -- every budget's strict window (10, or 6 at K41) is
#: larger than either value, so the protected ceilings, and therefore the knob
#: choice, are identical under both.
LANE_FIRST_ENHANCE = {"image": 1, "wan21": 3}
UPSTREAM_SWITCH_RATIO = 0.2
#: descending, so the search tries the largest strict region first
SWITCH_RATIO_LADDER = (0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06)


def strict_steps(switch_ratio: float, num_steps: int = NUM_STEPS,
                 first_enhance: int = FIRST_ENHANCE) -> int:
    """How many leading steps never cache: the warmup or the strict window."""
    return max(first_enhance, int(round(num_steps * switch_ratio)))


def structural_ceiling(switch_ratio: float, max_skip: int,
                       num_steps: int = NUM_STEPS,
                       first_enhance: int = FIRST_ENHANCE) -> int:
    """Largest realizable cache count, closed form.

    Steps 0..strict-1 and the terminal step are full by construction; the
    remaining window can hold runs of at most `max_skip` cached steps separated
    by full ones, so the ceiling is the window minus the fewest breaks that fit.
    """
    window = (num_steps - 1) - strict_steps(switch_ratio, num_steps, first_enhance)
    if window <= 0:
        return 0
    breaks = 0
    while window - breaks > max_skip * (breaks + 1):
        breaks += 1
    return window - breaks


def structural_ceiling_dp(switch_ratio: float, max_skip: int,
                          num_steps: int = NUM_STEPS,
                          first_enhance: int = FIRST_ENHANCE) -> int:
    """The same ceiling walked step by step, as an independent check."""
    strict = strict_steps(switch_ratio, num_steps, first_enhance)
    best = {0: 0}
    for step in range(num_steps):
        forced = step < strict or step >= num_steps - 1
        nxt: dict[int, int] = {}
        for run, cached in best.items():
            nxt[0] = max(nxt.get(0, -1), cached)
            if not forced and run < max_skip:
                nxt[run + 1] = max(nxt.get(run + 1, -1), cached + 1)
        best = nxt
    return max(best.values())


def run_limit_refusals(switch_ratio: float, max_skip: int,
                       num_steps: int = NUM_STEPS,
                       first_enhance: int = FIRST_ENHANCE) -> list[tuple[int, int]]:
    """Decision points where `max_skip` actually refuses a cache.

    Reachability is tracked over (step, consecutive_skips), which is the whole of
    the gate's state, so the answer is exact. An empty list means the run limit
    never fires on this lane: two values of `max_skip` that both return empty
    give bit-identical decisions, not merely an equal ceiling.
    """
    strict = strict_steps(switch_ratio, num_steps, first_enhance)
    states = {0}
    refusals: list[tuple[int, int]] = []
    for step in range(num_steps):
        forced = step < strict or step >= num_steps - 1
        if not forced:
            refusals += [(step, run) for run in sorted(states) if run >= max_skip]
        nxt = {0}
        for run in states:
            if not forced and run < max_skip:
                nxt.add(run + 1)
        states = nxt
    return refusals


def run_limit_is_inert(switch_ratio: float, max_skip: int) -> bool:
    """Whether `max_skip` is large enough to never fire on this lane."""
    return not run_limit_refusals(switch_ratio, max_skip)


def unprotected_ceiling(max_skip: int, num_steps: int = NUM_STEPS,
                        first_enhance: int = FIRST_ENHANCE) -> int:
    """Ceiling when the strict window is NOT protected -- only the warmup is.

    `structural_ceiling` assumes the strict window never caches, which is true
    of a strict `threshold_start` and false of a loose one. A row swept at a
    loose start is bounded by this instead, and reading it against the
    protected ceiling reports impossible-looking counts that are simply the
    wrong comparison.
    """
    return structural_ceiling(0.0, max_skip, num_steps, first_enhance)


def ceiling_knobs(target: int, margin: int = 2) -> dict[str, Any]:
    """The most conservative knobs whose ceiling clears `target + margin`.

    Largest strict region first, then the smallest max_skip that gets there,
    which is the order the plan's section 9.1 fixes. `max_skip` is never taken
    below upstream's 10 -- a budget that already fits leaves it untouched.
    """
    need = target + margin
    for switch_ratio in SWITCH_RATIO_LADDER:
        if structural_ceiling(switch_ratio, num_steps=NUM_STEPS, max_skip=NUM_STEPS) < need:
            continue  # this strict region caps out below the requirement
        max_skip = UPSTREAM_MAX_SKIP
        while structural_ceiling(switch_ratio, max_skip) < need:
            max_skip += 1
        return {
            "target": target,
            "switch_ratio": switch_ratio,
            "strict_steps": strict_steps(switch_ratio),
            "max_skip": max_skip,
            "ceiling": structural_ceiling(switch_ratio, max_skip),
            "margin": structural_ceiling(switch_ratio, max_skip) - target,
            "max_skip_is_upstream": max_skip == UPSTREAM_MAX_SKIP,
            "switch_ratio_is_upstream": switch_ratio == UPSTREAM_SWITCH_RATIO,
        }
    raise ValueError(f"no strict region reaches {need} cached steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="walk sweep output into a frontier table")
    collect.add_argument("--lane", choices=("image", "wan21"), required=True)
    collect.add_argument("--tasks", type=Path, required=True,
                         help="frontier_tasks.tsv written by the submitter")
    collect.add_argument("--out_tsv", type=Path, required=True)
    collect.add_argument("--out_json", type=Path, required=True)

    select = sub.add_parser("select", help="pick the frozen pairs off a frontier table")
    select.add_argument("--frontier", type=Path, required=True)
    select.add_argument("--out_json", type=Path, required=True)
    select.add_argument("--tolerance", type=float, default=0.3)

    prefix = sub.add_parser(
        "prefix",
        help="first-ten-step caching statistics of labelled cell directories",
    )
    prefix.add_argument("--cell", action="append", required=True,
                        metavar="LABEL=DIR",
                        help="repeatable; LABEL is free text, DIR holds decisions_*.json")
    prefix.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]


def _threshold_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


#: Every lane writes one row per solver step, but under its own key and with its
#: own spelling of the decision -- FLUX `per_step`/`u`, Qwen `steps`/`u`, Wan
#: `records`/`action`. Reading them by trying each key is what lets one code path
#: measure all three; a payload matching none of them is a schema change and has
#: to fail loudly rather than be counted as an empty run.
_ROW_KEYS = ("per_step", "steps", "records")


def _decision_bits(payload: dict[str, Any]) -> str:
    for key in _ROW_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            break
    else:
        raise KeyError(
            f"decision payload carries none of {_ROW_KEYS}; "
            f"schema={payload.get('schema')!r} keys={sorted(payload)[:12]}"
        )
    if all("step" in row for row in rows):
        rows = sorted(rows, key=lambda row: int(row["step"]))
    bits = []
    for row in rows:
        if "u" in row:
            bits.append("1" if int(row["u"]) == 1 else "0")
        elif "action" in row:
            bits.append("1" if str(row["action"]) == "cache" else "0")
        else:
            raise KeyError(
                f"decision row carries neither 'u' nor 'action': {sorted(row)[:12]}")
    return "".join(bits)


def _decision_files(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("decisions_*.json"))


def _longest_run(bits: str) -> int:
    best = run = 0
    for bit in bits:
        run = run + 1 if bit == "1" else 0
        best = max(best, run)
    return best


def _summarise(paths: Iterable[Path], strict: int = SWITCH_STEPS) -> dict[str, Any] | None:
    counts: list[int] = []
    prefixes: list[str] = []
    stricts: list[str] = []
    runs: list[int] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bits = _decision_bits(payload)
        if len(bits) != 50:
            continue
        counts.append(bits.count("1"))
        prefixes.append(bits[:SWITCH_STEPS])
        stricts.append(bits[:strict])
        runs.append(_longest_run(bits))
    if not counts:
        return None
    strict_counter = Counter(stricts)
    prefix_counter = Counter(prefixes)
    modal_prefix, modal_hits = prefix_counter.most_common(1)[0]
    return {
        "n_prompts": len(counts),
        "k_mean": statistics.fmean(counts),
        "k_std": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        "k_min": min(counts),
        "k_max": max(counts),
        "longest_cached_run": max(runs),
        "exact_fraction": {
            str(target): sum(1 for c in counts if c == target) / len(counts)
            for target in BUDGETS
        },
        # the plan's completion check: how much of the protected window is cached
        # the window the gate actually protects at this budget -- 10 steps at
        # switch_ratio 0.2, but 6 at K41's 0.12, where steps 6..9 are outside it
        # and cache legitimately
        "strict_window": strict,
        "strict_cache_mean": statistics.fmean(s.count("1") for s in stricts),
        "strict_modal_pattern": strict_counter.most_common(1)[0][0],
        # a fixed ten steps, so the three budgets stay comparable to each other
        "first10_cache_mean": statistics.fmean(p.count("1") for p in prefixes),
        "first10_unique_patterns": len(prefix_counter),
        "first10_modal_pattern": modal_prefix,
        "first10_modal_mass": modal_hits / len(prefixes),
    }


def collect(args: argparse.Namespace) -> int:
    tasks = _read_tsv(args.tasks)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    violations: list[str] = []
    lane_fe = LANE_FIRST_ENHANCE[args.lane]

    for task in tasks:
        start = float(task["start"])
        thresholds = [float(v) for v in task["thresholds"].split(",") if v.strip()]
        family = task["family"]
        budget = int(task["budget"])
        max_skip = int(task["max_skip"])
        switch_ratio = float(task["switch_ratio"])
        if args.lane == "image":
            group = task["model"]
            base = Path(task["output_dir"]) / "native_sen"

            def directory_for(value: float) -> Path:
                return base / f"sencache_t{_threshold_tag(value)}"
        else:
            group = task["dataset"]
            base = Path(task["out_root"])
            stem = Path(task["prompt_file"]).stem
            limit = task["limit"]

            def directory_for(value: float, stem=stem, limit=limit) -> Path:
                return base / f"t{value:.6f}_{stem}_n{limit}"

        for main in thresholds:
            directory = directory_for(main)
            summary = _summarise(_decision_files(directory),
                                 strict=strict_steps(switch_ratio,
                                                     first_enhance=lane_fe))
            if summary is None:
                missing.append(str(directory))
                continue
            # a row whose strict window never cached is bounded by the
            # protected ceiling; one that cached inside it is not
            protected = summary["strict_cache_mean"] == 0.0
            bound = (structural_ceiling(switch_ratio, max_skip,
                                        first_enhance=lane_fe) if protected
                     else unprotected_ceiling(max_skip, first_enhance=lane_fe))
            if summary["k_max"] > bound:
                violations.append(
                    f"{group} K{budget} start={start:g} main={main:g}: observed "
                    f"{summary['k_max']} cached steps above the "
                    f"{'protected' if protected else 'unprotected'} ceiling "
                    f"{bound} for max_skip={max_skip} switch_ratio={switch_ratio:g}"
                )
            rows.append({
                "group": group,
                "family": family,
                "budget": budget,
                "max_skip": max_skip,
                "switch_ratio": switch_ratio,
                "start": start,
                "main": main,
                "admissible": start <= main,
                "directory": str(directory),
                **summary,
            })

    rows.sort(key=lambda row: (row["group"], row["family"], row["start"], row["main"]))
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "group", "family", "budget", "max_skip", "switch_ratio",
        "start", "main", "admissible", "n_prompts", "k_mean", "k_std",
        "k_min", "k_max", "exact_29", "exact_37", "exact_41",
        "longest_cached_run", "strict_window", "strict_cache_mean",
        "first10_cache_mean", "first10_unique_patterns",
        "first10_modal_pattern", "first10_modal_mass",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join((
            row["group"], row["family"], str(row["budget"]), str(row["max_skip"]),
            f"{row['switch_ratio']:.12g}",
            f"{row['start']:.12g}", f"{row['main']:.12g}",
            "1" if row["admissible"] else "0", str(row["n_prompts"]),
            f"{row['k_mean']:.4f}", f"{row['k_std']:.4f}",
            str(row["k_min"]), str(row["k_max"]),
            f"{row['exact_fraction']['29']:.4f}",
            f"{row['exact_fraction']['37']:.4f}",
            f"{row['exact_fraction']['41']:.4f}",
            str(row["longest_cached_run"]),
            str(row["strict_window"]),
            f"{row['strict_cache_mean']:.4f}",
            f"{row['first10_cache_mean']:.4f}",
            str(row["first10_unique_patterns"]),
            row["first10_modal_pattern"],
            f"{row['first10_modal_mass']:.4f}",
        )))
    args.out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out_json.write_text(
        json.dumps({"lane": args.lane, "rows": rows, "missing": missing,
                    "ceiling_violations": violations},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[frontier] lane={args.lane} rows={len(rows)} missing={len(missing)} "
          f"ceiling_violations={len(violations)}")
    for line in violations[:10]:
        print(f"  VIOLATION: {line}")
    for path in missing[:10]:
        print(f"  missing: {path}")
    return 0


def select(args: argparse.Namespace) -> int:
    """Freeze one (start, main) per (group, sweep family).

    A family fixes the ceiling knobs, so the only choice left is the pair: the
    strictest `threshold_start` whose best `threshold_main` puts mean realized K
    within tolerance of the target, with `start <= main` enforced as the upstream
    relation. The n-ablation families are selected the same way as the canonical
    ones -- each was swept at its own run limit and needs its own `threshold_main`
    to hold the budget, so none of them may inherit the canonical family's pair.
    """
    rows = _read_tsv(args.frontier)
    chosen: dict[str, Any] = {"tolerance": args.tolerance, "groups": {}}

    for group in sorted({row["group"] for row in rows}):
        families = sorted({row["family"] for row in rows if row["group"] == group})
        per_family: dict[str, Any] = {}
        for family in families:
            pool = [
                row for row in rows
                if row["group"] == group
                and row["family"] == family
                and row["admissible"] == "1"
            ]
            if not pool:
                per_family[family] = {"family": family, "frozen": None,
                                      "reason": "no admissible pair swept"}
                continue
            target = int(pool[0]["budget"])
            # the knobs come off the swept rows, not from ceiling_knobs: an
            # ablation family deliberately runs a run limit that budget would
            # not otherwise use
            max_skip = int(pool[0]["max_skip"])
            switch_ratio = float(pool[0]["switch_ratio"])
            curve: dict[float, list[tuple]] = {}
            for row in pool:
                curve.setdefault(float(row["start"]), []).append((
                    float(row["main"]), float(row["k_mean"]), float(row["k_std"]),
                    float(row["first10_cache_mean"]),
                    int(row["first10_unique_patterns"]),
                    row["first10_modal_pattern"],
                    int(row["strict_window"]), float(row["strict_cache_mean"]),
                ))
            frozen = None
            rejected: list[dict[str, Any]] = []
            for start in sorted(curve):
                best = min(curve[start], key=lambda item: abs(item[1] - target))
                entry = {
                    "start": start, "main": best[0], "k_mean": best[1],
                    "k_std": best[2], "first10_cache_mean": best[3],
                    "first10_unique_patterns": best[4],
                    "first10_modal_pattern": best[5],
                    "strict_window": best[6], "strict_cache_mean": best[7],
                }
                if abs(best[1] - target) <= args.tolerance:
                    frozen = entry
                    break
                rejected.append({**entry, "miss": best[1] - target})
            per_family[family] = {
                "family": family,
                "target": target,
                "max_skip": max_skip,
                "switch_ratio": switch_ratio,
                "strict_steps": strict_steps(switch_ratio),
                "ceiling": structural_ceiling(switch_ratio, max_skip),
                "margin": structural_ceiling(switch_ratio, max_skip) - target,
                "run_limit_inert": run_limit_is_inert(switch_ratio, max_skip),
                "frozen": frozen,
                "starts_rejected_as_too_strict": rejected,
                "starts_swept": sorted(curve),
            }
        # the canonical families are also reachable by budget key, which is what
        # the Wan config refreeze reads
        for target in BUDGETS:
            canonical = per_family.get(f"k{target}")
            if canonical is not None:
                per_family[f"K{target}"] = canonical
        chosen["groups"][group] = {"budgets": per_family}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    unresolved = 0
    for group, payload in chosen["groups"].items():
        for key in sorted(payload["budgets"]):
            if key.startswith("K"):
                continue  # the alias, already printed under its family name
            entry = payload["budgets"][key]
            frozen = entry.get("frozen")
            if frozen is None:
                unresolved += 1
                print(f"{group:<11} {key:<8} NO PAIR within "
                      f"{args.tolerance} of {entry.get('target', '?')}")
                continue
            print(
                f"{group:<11} {key:<8} start={frozen['start']:<7g} "
                f"main={frozen['main']:<7g} n={entry['max_skip']:<3} "
                f"switch={entry['switch_ratio']:<5g} "
                f"K={frozen['k_mean']:6.3f}+-{frozen['k_std']:<5.2f} "
                f"(target {entry['target']}, ceiling {entry['ceiling']}) "
                f"strict[{frozen['strict_window']}]={frozen['strict_cache_mean']:.3f} "
                f"first10={frozen['first10_cache_mean']:.3f}"
            )
    if unresolved:
        print(f"[select] {unresolved} family/families unresolved -- widen the main grid")
    return 0


def prefix(args: argparse.Namespace) -> int:
    """The plan's completion check: is the protected window still budget-blind?

    Reads whichever decision schema the lane wrote -- the three differ -- so the
    old and new sides of a cell are measured by one code path.

    Reports, per labelled cell, how much of the first ten steps is cached and
    what the modal pattern over those ten bits is. Two budgets whose modal
    pattern and mean agree exactly are the failure this recalibration exists to
    remove.
    """
    out: dict[str, Any] = {"cells": {}}
    for item in args.cell:
        label, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--cell needs LABEL=DIR, got {item!r}")
        summary = _summarise(_decision_files(Path(directory)))
        if summary is None:
            print(f"{label}: no decision files under {directory}")
            continue
        out["cells"][label] = {"directory": directory, **summary}
        print(
            f"{label}: n={summary['n_prompts']} K={summary['k_mean']:.3f} "
            f"first10={summary['first10_cache_mean']:.3f} "
            f"patterns={summary['first10_unique_patterns']} "
            f"modal={summary['first10_modal_pattern']} "
            f"({summary['first10_modal_mass']:.3f})"
        )
    patterns = {
        label: cell["first10_modal_pattern"] for label, cell in out["cells"].items()
    }
    out["distinct_modal_patterns"] = len(set(patterns.values()))
    print(f"distinct first-ten modal patterns across {len(patterns)} cells: "
          f"{out['distinct_modal_patterns']}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "collect":
        return collect(args)
    if args.command == "prefix":
        return prefix(args)
    return select(args)


if __name__ == "__main__":
    raise SystemExit(main())
