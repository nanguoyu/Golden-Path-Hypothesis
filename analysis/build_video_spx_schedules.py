#!/usr/bin/env python3
"""Freeze the video SPX schedule set (`docs/video_sp_cross_plan_zh.md` section 2).

One row of the schedule axis is a 50-bit string, `'1'` = the step is cached,
which is the convention `analysis/video_trajectory/density_form.py` and
`merge_video_traj.py` already use.  Per (backbone, budget) the plan asks for
fourteen rows:

  F  `shared` `budcache` `meancache`      the three frozen matrix tables
  G  `sea_top1` `tea_top1` `sen_top1` `di_top1`
                                          each gate's modal realized path
  C  `uniform` `dp_rho2` `rand_1` `rand_2` `ham2` `ham4` `ham8`

plus, for the three (backbone, gate, K) combinations whose gate never realized
exactly K cached steps, one **off-budget** row that runs the gate's modal path
at whatever K it actually walks (owner decision 3, plan section 9).  Off-budget
rows are only ever paired with the `reuse` payload and never enter the variance
decomposition.

Two supplements were added afterwards (plan section 10, "supplement"), both
answers to the code audit of 2026-08-23:

  S1 `ham2f` `ham4f` `ham8f`, a second Hamming ladder whose swaps are confined
     to positions strictly after the MeanCache row's first cached step, so the
     rung keeps MeanCache's `first_cache_step` and the dose curve stops being
     confounded with "the first cached step moved earlier" (audit finding 3).
     Two rungs do not exist: at K41 only three steps above the anchor's first
     cached step are full on either backbone, and `ham8f` needs four of them to
     swap in.  Those are recorded, not repaired.
  S2 a span-stripped copy of each MeanCache row under `<backbone>/global_span/`,
     which is the `--meancache_schedule` transport for the `mean_vel_global`
     payload column: the same table run on the GLOBAL fallback JVP span every
     other row already uses, so the `mean_vel` column stops being two different
     payloads (audit finding 1).  The row itself is untouched; the variant is a
     second transport file, never a second row.

What this program does NOT do is edit bits after the fact.  Every row is either
read verbatim from a frozen artifact or constructed by one rule, and the rule
is recorded next to the row; a row that fails a self-check is refused rather
than repaired.

Self-checks, all fatal:

  * popcount equals the nominal K (off-budget rows excepted, whose realized
    count is recorded instead);
  * step 0 and step 49 are full on every row, so the sigma-span anchor rule of
    `density_form.check_bits` holds and no payload's terminal step is cached;
  * `ham2/4/8` sit at Hamming exactly 2 / 4 / 8 from the MeanCache row of the
    same (backbone, K), keep its popcount, and open no full-step gap wider than
    `MAX_GAP` = 15 steps;
  * the same command run twice writes byte-identical files.

Feasibility, not repair
-----------------------
Each payload column forbids some steps (its warmup and the terminal step), and
those sets differ per backbone.  Four gate modal paths cache a step some payload
must run full -- they are gate data, not a construction error -- so the manifest
carries one `feasible_<payload>` column per payload and the submitter skips the
infeasible cells.  Nothing is projected or shifted onto a legal neighbour.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.build_golden_path_family import (  # noqa: E402
    INF,
    j_best_schedules,
    schedule_bits,
    segment_cost_matrix,
)

SCHEMA = "video_spx.schedule.v1"
MANIFEST_SCHEMA = "video_spx.schedule_manifest.v1"

NUM_STEPS = 50
KS = (29, 37, 41)
BACKBONES = ("hunyuan_video", "wan21")
GATES = ("seacache", "teacache", "sencache", "dicache")
GATE_ROW = {"seacache": "sea_top1", "teacache": "tea_top1",
            "sencache": "sen_top1", "dicache": "di_top1"}
DATASETS = ("penguin599", "vbench944")

#: The frozen matrix table each F row reads. The triplet table is called
#: `triplet_*` on the Hunyuan lane and `shared_*` on the Wan lane.
SHARED_TABLE_KEY = {"hunyuan_video": "triplet", "wan21": "shared"}

CONFIG_PATH = {
    "hunyuan_video": _ROOT / "resources/hunyuan_video/baseline_matrix_config.v1.json",
    "wan21": _ROOT / "resources/wan21/baseline_matrix_config.v1.json",
}
GATE_PATH_COUNTS = {
    backbone: _ROOT / f"resources/video_native_gate_paths/{backbone}/dataset_path_counts.tsv"
    for backbone in BACKBONES
}
DENSITY_FORM = {
    backbone: _ROOT / f"resources/video_full_trajectory/{backbone}/density_form_{backbone}.json"
    for backbone in BACKBONES
}

OUTPUT_ROOT = _ROOT / "resources/video_spx_schedules"

#: Design controls keep these four steps full so that every payload column can
#: run them (plan section 2: the union of all payload warmups is {0,1,2} and
#: every payload forces the terminal step).
CONTROL_FORCED_FULL = (0, 1, 2, NUM_STEPS - 1)

#: `dp_rho2` and the Hamming ladder are capped at this distance between
#: consecutive full steps -- `segment_cost_matrix`'s `max_gap`, i.e. a run of at
#: most 14 consecutive cached steps. MeanCache's own multigraph carries no edge
#: longer than 15 either (`flux/meancache_calibrate.py --max_edge_gap`).
MAX_GAP = 15

#: The DP positive control's cost model: rho2 of the zero-order payload, on the
#: main rho2 dataset of `density_form_<T>.json`.
DP_EXPONENT = 1.0
DP_RHO2_DATASET = "vbench944"

#: The Hamming ladder swaps only inside this window, which keeps steps 0-2 full
#: on every constructed row and therefore keeps them legal for every payload
#: column, including `hermite_o2` whose warmup is the widest of the five. It
#: cannot start at 5: the K41 MeanCache table has only three full steps above
#: step 4, and `ham8` needs four of them.
SWAP_LOW = 3
SWAP_HIGH = NUM_STEPS - 2  # inclusive

#: The first-step-preserving ladder (`ham2f/4f/8f`) draws both halves of every
#: swap from `anchor_first_cache_step + 1 .. SWAP_HIGH` instead. The anchor's
#: first cached step is then neither swapped out nor undercut by a swapped-in
#: one, so the rung's `first_cache_step` is MeanCache's by construction -- which
#: is the whole point: on the original ladder the swap window starts at 3, the
#: MeanCache tables are full there, and a swapped-in cached step almost always
#: landed before the anchor's first, making Hamming and "how much earlier the
#: caching starts" inseparable (audit finding 3, 14 of 18 rungs).
LADDER_F_SUFFIX = "f"
LADDER_DRAWS = ((2, 11), (4, 12), (8, 13))
LADDER_F_DRAWS = ((2, 21), (4, 22), (8, 23))

#: Random exact-K rows draw from the interior the design controls are allowed
#: to use.
RAND_LOW = 3
RAND_HIGH = NUM_STEPS - 2  # inclusive

#: `numpy.random.default_rng([RNG_ROOT, backbone_code, K, draw])`. The root is
#: the plan's date; the codes are positional and are recorded in every row's
#: provenance so a re-run reproduces the draw without reading this file.
RNG_ROOT = 20260822
BACKBONE_CODE = {"hunyuan_video": 1, "wan21": 2}

#: Steps each payload column must run full, per backbone, as the runners
#: enforce them AFTER this experiment's two relaxations (plan section 8 item 2
#: plus the Wan reuse relaxation the off-budget SeaCache row needs):
#:   * MeanCache's runtime warmup {0..4} is relaxed to {0,1};
#:   * Wan's `budcache` warmup {0,1,2} is relaxed to {0}, matching the Hunyuan
#:     `reuse_exact` semantics (`ReuseMethod` needs only a previous residual).
#: Both relaxations are opt-in flags (`--spx_relax_warmup`); the frozen matrix
#: is unaffected.
PAYLOAD_FORBIDDEN = {
    "hunyuan_video": {
        "reuse": frozenset({0, NUM_STEPS - 1}),
        "taylor_o1": frozenset({0, NUM_STEPS - 1}),
        "hermite_o2": frozenset({0, 1, 2, NUM_STEPS - 1}),
        "mean_vel": frozenset({0, 1, NUM_STEPS - 1}),
        "di_two_anchor": frozenset({0, 1, NUM_STEPS - 1}),
    },
    "wan21": {
        "reuse": frozenset({0, NUM_STEPS - 1}),
        "taylor_o1": frozenset({0, NUM_STEPS - 1}),
        "hermite_o2": frozenset({0, 1, 2, NUM_STEPS - 1}),
        "mean_vel": frozenset({0, 1, NUM_STEPS - 1}),
        "di_two_anchor": frozenset({0, 1, NUM_STEPS - 1}),
    },
}
PAYLOADS = ("reuse", "taylor_o1", "hermite_o2", "mean_vel", "di_two_anchor")

#: The runner `--mode` each payload column maps to, per backbone (plan section 3).
PAYLOAD_MODE = {
    "hunyuan_video": {
        "reuse": "reuse_exact",
        "taylor_o1": "taylorseer_exact",
        "hermite_o2": "hicache_exact",
        "mean_vel": "meancache_exact",
        "di_two_anchor": "dicache",
    },
    "wan21": {
        "reuse": "budcache",
        "taylor_o1": "taylorseer_o1",
        "hermite_o2": "hicache_o2",
        "mean_vel": "meancache",
        "di_two_anchor": "dicache",
    },
}

#: MeanCache's global fallback JVP span for every row that is not MeanCache's
#: own table (plan section 3: the offline search only solved per-edge spans for
#: that one path).
GLOBAL_JVP_SPAN = 4

#: Supplement S2. The rows whose `--meancache_schedule` transport is also
#: written span-stripped, into `<backbone>/<SPAN_VARIANT_DIR>/`, and the payload
#: column that reads it. Only MeanCache's own table carries per-edge spans, so
#: only it needs the variant; every other row's `mean_vel` cell is already the
#: global-span payload and is NOT re-run.
SPAN_VARIANT_ROWS = ("meancache",)
SPAN_VARIANT_DIR = "global_span"
SPAN_VARIANT_PAYLOAD = "mean_vel_global"
SPAN_VARIANT_BASE_PAYLOAD = "mean_vel"


# ---------------------------------------------------------------------------
# bit helpers
# ---------------------------------------------------------------------------


def bits_of(cache_steps: Iterable[int]) -> str:
    cached = {int(step) for step in cache_steps}
    bad = sorted(step for step in cached if not 0 <= step < NUM_STEPS)
    if bad:
        raise SystemExit(f"cache steps outside 0..{NUM_STEPS - 1}: {bad}")
    return "".join("1" if step in cached else "0" for step in range(NUM_STEPS))


def steps_of(bits: str) -> list[int]:
    return [index for index, char in enumerate(bits) if char == "1"]


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("bitstrings must be the same length")
    return sum(a != b for a, b in zip(left, right))


def transpositions(left: str, right: str) -> int:
    """Half the Hamming distance, defined only at equal popcount.

    Plan section 5.3 counts "how many (cache step, full step) swaps apart" two
    equal-budget schedules are; at equal popcount that is exactly Hamming / 2.
    """
    if left.count("1") != right.count("1"):
        raise ValueError("transposition distance needs equal popcount")
    distance = hamming(left, right)
    if distance % 2:
        raise ValueError("equal-popcount bitstrings cannot be an odd distance apart")
    return distance // 2


def full_step_gaps(bits: str) -> list[int]:
    """Distances between consecutive full steps -- `segment_cost_matrix`'s gap.

    A schedule with no cached step between two adjacent full steps has gap 1,
    so `max(gaps) <= MAX_GAP` means at most `MAX_GAP - 1` consecutive cached
    steps.
    """
    full = [index for index, char in enumerate(bits) if char == "0"]
    return [b - a for a, b in zip(full, full[1:])]


def longest_cached_run(bits: str) -> int:
    best = current = 0
    for char in bits:
        current = current + 1 if char == "1" else 0
        best = max(best, current)
    return best


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


@dataclass
class Row:
    backbone: str
    row: str
    group: str
    budget: int              # the nominal K of the partition
    bits: str
    source: str
    off_budget: bool = False
    jvp_spans: dict[int, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    #: `None` = every payload column (the default). A row that is deliberately
    #: run in fewer columns says so here; the submitter reads it off the
    #: schedule JSON, so the manifest keeps its frozen column set.
    payload_columns: tuple[str, ...] | None = None

    @property
    def schedule_id(self) -> str:
        return f"{self.row}_K{self.budget}"

    @property
    def cache_steps(self) -> list[int]:
        return steps_of(self.bits)

    @property
    def popcount(self) -> int:
        return self.bits.count("1")


def check_row(row: Row) -> None:
    if len(row.bits) != NUM_STEPS or set(row.bits) - {"0", "1"}:
        raise SystemExit(f"{row.backbone} {row.schedule_id}: not a {NUM_STEPS}-bit 0/1 string")
    if row.bits[0] == "1":
        raise SystemExit(
            f"{row.backbone} {row.schedule_id}: step 0 is cached. No payload can run it and "
            f"the sigma-span anchor rule (density_form.check_bits) assumes it full.")
    if row.bits[-1] == "1":
        raise SystemExit(
            f"{row.backbone} {row.schedule_id}: the terminal step is cached; every payload "
            f"forces it full.")
    if not row.off_budget and row.popcount != row.budget:
        raise SystemExit(
            f"{row.backbone} {row.schedule_id}: popcount {row.popcount} != nominal K "
            f"{row.budget}")
    for step, span in row.jvp_spans.items():
        if step not in set(row.cache_steps):
            raise SystemExit(f"{row.backbone} {row.schedule_id}: jvp span for uncached step {step}")
        if int(span) < 1:
            raise SystemExit(f"{row.backbone} {row.schedule_id}: non-positive jvp span at {step}")


def feasibility(backbone: str, bits: str) -> dict[str, bool]:
    cached = set(steps_of(bits))
    return {payload: not (cached & forbidden)
            for payload, forbidden in PAYLOAD_FORBIDDEN[backbone].items()}


def columns_of(row: Row) -> tuple[str, ...]:
    """The payload columns a row is submitted in."""
    if row.payload_columns is not None:
        return tuple(row.payload_columns)
    return ("reuse",) if row.off_budget else tuple(PAYLOADS)


# -- F: the three frozen matrix tables --------------------------------------


def frozen_rows(backbone: str, config: dict[str, Any]) -> list[Row]:
    tables = config["schedule_tables"]
    shared_key = SHARED_TABLE_KEY[backbone]
    rows: list[Row] = []
    for k in KS:
        for row_name, table_id in (
            ("shared", f"{shared_key}_K{k}"),
            ("budcache", f"budcache_K{k}"),
            ("meancache", f"meancache_K{k}"),
        ):
            table = tables[table_id]
            bits = bits_of(table["cache_steps"])
            spans = {int(step): int(span)
                     for step, span in (table.get("jvp_spans") or {}).items()}
            rows.append(Row(
                backbone=backbone, row=row_name, group="frozen", budget=k, bits=bits,
                source=f"{CONFIG_PATH[backbone].name}::schedule_tables.{table_id}",
                jvp_spans=spans,
                provenance={
                    "table_id": table_id,
                    "table_source_file": table.get("source_file"),
                    "table_source_sha256": table.get("source_sha256"),
                    "matrix_config_sha256": config["config_sha256"],
                    "jvp_spans_are_per_edge": bool(spans),
                },
            ))
    return rows


# -- G: gate modal paths -----------------------------------------------------


def read_gate_counts(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """`(gate, K) -> rows`, both datasets pooled but each row keeping its dataset."""
    out: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            key = (record["method"], int(record["budget"][1:]))
            out[key].append({
                "dataset": record["dataset"],
                "count": int(record["count"]),
                "n_cached": int(record["n_cached"]),
                "bits": record["schedule"],
            })
    return dict(out)


def _pick_top1(records: Sequence[dict[str, Any]], *,
               exact_k: int | None) -> dict[str, Any] | None:
    """Highest pooled count, ties broken by the lexicographically smallest bits.

    Both datasets are generated in full, so the plan pools by raw count sum and
    does not weight them equally; `per_dataset_counts` records what each dataset
    contributed so the analysis can slice it later.
    """
    pool = [record for record in records
            if exact_k is None or record["n_cached"] == exact_k]
    if not pool:
        return None
    total: collections.Counter[str] = collections.Counter()
    per_dataset: dict[str, collections.Counter[str]] = {
        dataset: collections.Counter() for dataset in DATASETS}
    for record in pool:
        total[record["bits"]] += record["count"]
        per_dataset.setdefault(record["dataset"], collections.Counter())
        per_dataset[record["dataset"]][record["bits"]] += record["count"]
    best = max(total.values())
    tied = sorted(bits for bits, count in total.items() if count == best)
    bits = tied[0]
    grand_total = sum(record["count"] for record in records)
    return {
        "bits": bits,
        "count": int(best),
        "n_tied_at_top": len(tied),
        "tie_broken_lexicographically": len(tied) > 1,
        "share_of_exact_k_pool": best / sum(total.values()),
        "share_of_all_paths": best / grand_total if grand_total else None,
        "n_distinct_paths_in_pool": len(total),
        "per_dataset_counts": {dataset: int(counter.get(bits, 0))
                               for dataset, counter in per_dataset.items()},
        "per_dataset_top1_agrees": {
            dataset: (bool(counter) and
                      max(counter.items(), key=lambda item: (item[1], [-ord(c) for c in item[0]]))[0] == bits)
            for dataset, counter in per_dataset.items()},
    }


def gate_rows(backbone: str) -> tuple[list[Row], list[dict[str, Any]]]:
    counts = read_gate_counts(GATE_PATH_COUNTS[backbone])
    rows: list[Row] = []
    unavailable: list[dict[str, Any]] = []
    for k in KS:
        for gate in GATES:
            records = counts.get((gate, k), [])
            if not records:
                raise SystemExit(f"{backbone}: no census rows for {gate} K{k}")
            pick = _pick_top1(records, exact_k=k)
            off_budget = pick is None
            if off_budget:
                pick = _pick_top1(records, exact_k=None)
                assert pick is not None
            bits = pick["bits"]
            row_name = GATE_ROW[gate] + ("_off" if off_budget else "")
            provenance = {
                "gate": gate,
                "selection": ("pooled top-1 among exactly-K paths" if not off_budget else
                              "pooled top-1 among ALL realized paths; this gate never "
                              "realized exactly K, so the row runs at its own K "
                              "(owner decision 3: reuse column only, excluded from the "
                              "variance decomposition)"),
                "realized_cache_count": bits.count("1"),
                **{key: value for key, value in pick.items() if key != "bits"},
            }
            rows.append(Row(
                backbone=backbone, row=row_name, group="gate", budget=k, bits=bits,
                source=f"{GATE_PATH_COUNTS[backbone].relative_to(_ROOT)}",
                off_budget=off_budget, provenance=provenance,
            ))
            if off_budget:
                unavailable.append({"backbone": backbone, "gate": gate, "budget": k,
                                    "realized_cache_count": bits.count("1")})
    return rows, unavailable


# -- C: design controls ------------------------------------------------------


def uniform_bits(k: int) -> str:
    """Full steps evenly spaced over 3..48, with {0,1,2,49} forced full.

    Not `density_form.uniform_bits`: that control spaces the full steps over the
    whole trajectory and caches step 1 and step 2, which three of the five
    payload columns forbid. Same idea, rebuilt inside the legal interior.
    """
    forced = list(CONTROL_FORCED_FULL)
    n_full = NUM_STEPS - k
    remaining = n_full - len(forced)
    if remaining < 0:
        raise SystemExit(f"uniform K={k}: fewer full steps than the {len(forced)} forced ones")
    interior = np.unique(np.round(
        np.linspace(RAND_LOW, RAND_HIGH, remaining)).astype(int)) if remaining else np.array([], int)
    if interior.size != remaining:
        raise SystemExit(f"uniform K={k}: rounding collapsed {remaining} full steps onto "
                         f"{interior.size} distinct indices")
    full = sorted(set(forced) | {int(step) for step in interior})
    bits = schedule_bits(full, num_steps=NUM_STEPS)
    if bits.count("1") != k:
        raise SystemExit(f"uniform K={k}: built {bits.count('1')} cached steps")
    return bits


def force_full(cost: np.ndarray, steps: Iterable[int]) -> np.ndarray:
    """Make every DP schedule visit `steps` as full steps.

    A segment (a, b) that jumps over a forced step would cache it, so its cost
    becomes infinite and the DP cannot choose it. This is how the plan's
    "forced full {0,1,2,49}" reaches a solver whose only structural forced steps
    are the first and the last.
    """
    out = cost.copy()
    for step in steps:
        for a in range(int(step)):
            out[a, int(step) + 1:] = INF
    return out


def dp_rho2_bits(rho2: Sequence[float], sigmas: Sequence[float], k: int) -> tuple[str, float]:
    cost = segment_cost_matrix(np.asarray(rho2, dtype=float)[:NUM_STEPS],
                               np.asarray(sigmas, dtype=float)[:NUM_STEPS],
                               exponent=DP_EXPONENT, max_gap=MAX_GAP)
    cost = force_full(cost, CONTROL_FORCED_FULL[1:-1])
    best = j_best_schedules(cost, n_full=NUM_STEPS - k, j_best=1)
    if not best:
        raise SystemExit(f"dp_rho2 K={k}: no feasible schedule at max_gap={MAX_GAP}")
    value, full_steps = best[0]
    bits = schedule_bits(full_steps, num_steps=NUM_STEPS)
    if bits.count("1") != k:
        raise SystemExit(f"dp_rho2 K={k}: built {bits.count('1')} cached steps")
    return bits, float(value)


def rng_for(backbone: str, k: int, draw: int) -> np.random.Generator:
    return np.random.default_rng([RNG_ROOT, BACKBONE_CODE[backbone], int(k), int(draw)])


def random_bits(backbone: str, k: int, draw: int) -> str:
    rng = rng_for(backbone, k, draw)
    candidates = np.arange(RAND_LOW, RAND_HIGH + 1)
    if k > candidates.size:
        raise SystemExit(f"rand K={k}: only {candidates.size} legal interior steps")
    chosen = rng.choice(candidates, size=k, replace=False)
    return bits_of(int(step) for step in chosen)


def hamming_ladder_bits(anchor: str, distance: int, backbone: str, k: int,
                        draw: int, *, swap_low: int = SWAP_LOW,
                        strict: bool = True) -> tuple[str, dict[str, Any]] | tuple[None, dict[str, Any]]:
    """`distance / 2` random (cache step, full step) swaps away from `anchor`.

    Swaps are drawn inside `swap_low..SWAP_HIGH`. At the default `SWAP_LOW` that
    leaves steps 0-2 and the terminal step untouched and therefore legal for
    every payload; the first-step-preserving ladder raises `swap_low` past the
    anchor's first cached step instead, which keeps that step cached and admits
    no earlier one, so the rung's `first_cache_step` is the anchor's.

    A draw whose result opens a full-step gap wider than `MAX_GAP` is rejected
    and redrawn from the same stream, so the recipe is `(backbone, K, draw)`
    plus the rejection count, both recorded.

    `strict=False` reports an impossible rung instead of dying on it: with the
    window raised, a rung needs `distance / 2` full steps inside it, and the K41
    MeanCache tables have only three.
    """
    if distance % 2:
        raise SystemExit("the Hamming ladder moves in (cache, full) swaps, so it is even")
    swaps = distance // 2
    rng = rng_for(backbone, k, draw)
    cached = [step for step in steps_of(anchor) if swap_low <= step <= SWAP_HIGH]
    full = [step for step in range(swap_low, SWAP_HIGH + 1) if anchor[step] == "0"]
    # Key order is the frozen one: the original ladder's rows are already
    # committed and a re-run must reproduce their JSON byte for byte.
    def recipe(**extra: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rng": f"default_rng([{RNG_ROOT}, {BACKBONE_CODE[backbone]}, {k}, {draw}])",
            "swaps": swaps}
        out.update(extra)
        out["swap_window"] = [swap_low, SWAP_HIGH]
        if swap_low != SWAP_LOW:
            out["swappable_cached"] = len(cached)
            out["swappable_full"] = len(full)
        out["rejection_rules"] = ["popcount changed", "distance != target",
                                  f"full-step gap > {MAX_GAP}"]
        return out

    if len(cached) < swaps or len(full) < swaps:
        if strict:
            raise SystemExit(f"ham{distance} K={k}: too few swappable steps")
        return None, recipe(unavailable="too few swappable steps in the window")
    for attempt in range(1, 1001):
        out_steps = rng.choice(np.asarray(cached), size=swaps, replace=False)
        in_steps = rng.choice(np.asarray(full), size=swaps, replace=False)
        new = set(steps_of(anchor)) - {int(s) for s in out_steps} | {int(s) for s in in_steps}
        bits = bits_of(new)
        if bits.count("1") != anchor.count("1"):
            continue
        if hamming(bits, anchor) != distance:
            continue
        if max(full_step_gaps(bits), default=0) > MAX_GAP:
            continue
        return bits, recipe(draws_rejected=attempt - 1)
    if strict:
        raise SystemExit(f"ham{distance} K={k}: 1000 draws all violated the gap cap")
    return None, recipe(unavailable="1000 draws all violated the gap cap")


def control_rows(backbone: str, density: dict[str, Any],
                 mean_bits: dict[int, str]) -> tuple[list[Row], list[dict[str, Any]]]:
    rho2_dataset = DP_RHO2_DATASET
    if rho2_dataset not in density["rho2"]:
        raise SystemExit(f"{backbone}: density_form has no rho2 for {rho2_dataset}")
    rho2 = density["rho2"][rho2_dataset]
    sigmas = density["sigmas"]
    rows: list[Row] = []
    skipped: list[dict[str, Any]] = []
    for k in KS:
        rows.append(Row(
            backbone=backbone, row="uniform", group="control", budget=k,
            bits=uniform_bits(k),
            source="constructed: full steps evenly spaced over 3..48, {0,1,2,49} forced full",
            provenance={"forced_full": list(CONTROL_FORCED_FULL),
                        "interior": [RAND_LOW, RAND_HIGH]},
        ))
        bits, value = dp_rho2_bits(rho2, sigmas, k)
        rows.append(Row(
            backbone=backbone, row="dp_rho2", group="control", budget=k, bits=bits,
            source=f"{DENSITY_FORM[backbone].relative_to(_ROOT)}::rho2.{rho2_dataset}",
            provenance={"constructor": "segment_cost_matrix + force_full + j_best_schedules "
                                       "(analysis/build_golden_path_family.py)",
                        "exponent": DP_EXPONENT, "max_gap": MAX_GAP,
                        "rho2_dataset": rho2_dataset,
                        "forced_full": list(CONTROL_FORCED_FULL),
                        "dp_cost": value},
        ))
        for draw, name in ((1, "rand_1"), (2, "rand_2")):
            rows.append(Row(
                backbone=backbone, row=name, group="control", budget=k,
                bits=random_bits(backbone, k, draw),
                source="constructed: uniform draw of K steps from 3..48",
                provenance={"rng": f"default_rng([{RNG_ROOT}, {BACKBONE_CODE[backbone]}, "
                                   f"{k}, {draw}])",
                            "interior": [RAND_LOW, RAND_HIGH]},
            ))
        anchor = mean_bits[k]
        for distance, draw in LADDER_DRAWS:
            bits, recipe = hamming_ladder_bits(anchor, distance, backbone, k, draw)
            assert bits is not None
            rows.append(Row(
                backbone=backbone, row=f"ham{distance}", group="control", budget=k, bits=bits,
                source=f"constructed: {distance // 2} random swaps from the meancache_K{k} table",
                provenance={"anchor_row": "meancache", "target_hamming": distance,
                            "realized_hamming": hamming(bits, anchor), **recipe},
            ))
        # Supplement S1: the same ladder, swaps confined to positions strictly
        # after the anchor's first cached step.
        anchor_first = min(steps_of(anchor))
        for distance, draw in LADDER_F_DRAWS:
            bits, recipe = hamming_ladder_bits(
                anchor, distance, backbone, k, draw,
                swap_low=anchor_first + 1, strict=False)
            name = f"ham{distance}{LADDER_F_SUFFIX}"
            if bits is None:
                skipped.append({"backbone": backbone, "row": name, "budget": k,
                                "anchor_first_cache_step": anchor_first,
                                "reason": recipe["unavailable"],
                                "swappable_full": recipe["swappable_full"],
                                "swaps_needed": recipe["swaps"]})
                continue
            rows.append(Row(
                backbone=backbone, row=name, group="control", budget=k, bits=bits,
                source=(f"constructed: {distance // 2} random swaps from the meancache_K{k} "
                        f"table, drawn strictly after its first cached step"),
                payload_columns=("reuse",),
                provenance={"anchor_row": "meancache", "target_hamming": distance,
                            "realized_hamming": hamming(bits, anchor),
                            "first_step_preserving": True,
                            "anchor_first_cache_step": anchor_first,
                            "payload_columns_reason":
                                "supplement S1 answers the dose curve's confound, which is "
                                "read in the reuse column; the other four columns are not "
                                "re-run (plan section 10, supplement)",
                            **recipe},
            ))
    return rows, skipped


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


ROW_ORDER = ("shared", "budcache", "meancache",
             "sea_top1", "tea_top1", "sen_top1", "di_top1",
             "sea_top1_off", "tea_top1_off", "sen_top1_off", "di_top1_off",
             "uniform", "dp_rho2", "rand_1", "rand_2", "ham2", "ham4", "ham8",
             "ham2f", "ham4f", "ham8f")


def build_backbone(backbone: str) -> tuple[list[Row], list[dict[str, Any]],
                                           list[dict[str, Any]]]:
    config = json.loads(CONFIG_PATH[backbone].read_text(encoding="utf-8"))
    density = json.loads(DENSITY_FORM[backbone].read_text(encoding="utf-8"))
    rows = frozen_rows(backbone, config)
    mean_bits = {row.budget: row.bits for row in rows if row.row == "meancache"}
    gates, unavailable = gate_rows(backbone)
    rows += gates
    controls, skipped = control_rows(backbone, density, mean_bits)
    rows += controls
    bud_bits = {row.budget: row.bits for row in rows if row.row == "budcache"}
    for row in rows:
        check_row(row)
        anchor_mean = mean_bits[row.budget]
        anchor_bud = bud_bits[row.budget]
        row.provenance["hamming_to_meancache"] = hamming(row.bits, anchor_mean)
        row.provenance["hamming_to_budcache"] = hamming(row.bits, anchor_bud)
        row.provenance["transpositions_to_meancache"] = (
            None if row.popcount != anchor_mean.count("1")
            else transpositions(row.bits, anchor_mean))
        row.provenance["longest_cached_run"] = longest_cached_run(row.bits)
        row.provenance["max_full_step_gap"] = max(full_step_gaps(row.bits), default=0)
        row.provenance["first_cache_step"] = min(row.cache_steps, default=None)
    order = {name: index for index, name in enumerate(ROW_ORDER)}
    rows.sort(key=lambda item: (item.budget, order[item.row]))
    return rows, unavailable, skipped


def row_payload(row: Row) -> dict[str, Any]:
    feasible = feasibility(row.backbone, row.bits)
    return {
        "schema": SCHEMA,
        "schedule_id": row.schedule_id,
        "backbone": row.backbone,
        "row": row.row,
        "group": row.group,
        "budget": f"K{row.budget}",
        "nominal_k": row.budget,
        "off_budget": row.off_budget,
        "num_steps": NUM_STEPS,
        "bits": row.bits,
        "cache_steps": row.cache_steps,
        "cache_count": row.popcount,
        "jvp_spans": {str(step): int(span) for step, span in sorted(row.jvp_spans.items())},
        "jvp_span_global": GLOBAL_JVP_SPAN,
        "jvp_spans_are_per_edge": bool(row.jvp_spans),
        "payload_modes": PAYLOAD_MODE[row.backbone],
        "payload_feasible": feasible,
        "payload_forbidden_steps": {payload: sorted(steps) for payload, steps
                                    in PAYLOAD_FORBIDDEN[row.backbone].items()},
        "payload_columns": list(columns_of(row)),
        "source": row.source,
        "provenance": row.provenance,
    }


def span_variant_payload(row: Row) -> dict[str, Any]:
    """The span-stripped `--meancache_schedule` transport of a MeanCache row.

    Same table, no per-edge spans, so both runners fall back to the global
    `--meancache_jvp_span` on every edge (`hunyuan_video/methods/meancache.py`
    `span_for`) -- which is exactly what the other thirteen rows' `mean_vel`
    cells already ran. It is a transport file, not a schedule row: it never
    enters the manifest and the row it copies is untouched.
    """
    payload = row_payload(row)
    payload.update({
        "jvp_spans": {},
        "jvp_spans_are_per_edge": False,
        "payload_columns": [SPAN_VARIANT_PAYLOAD],
        "span_variant": "global",
        "variant_of": row.schedule_id,
        "source": f"{row.source} (per-edge jvp_spans stripped)",
    })
    payload["provenance"] = {
        **row.provenance,
        "span_variant_reason":
            "supplement S2: the mean_vel column was two different payloads -- MeanCache's "
            "own row ran the searched per-edge spans, every other row the global span "
            "(plan section 10, supplement). This cell re-runs MeanCache's table on the "
            "global span so the column is homogeneous; the searched-span cell stays as "
            "the same-source diagonal.",
        "jvp_spans_stripped": {str(step): int(span)
                               for step, span in sorted(row.jvp_spans.items())},
    }
    return payload


MANIFEST_COLUMNS = (
    "backbone", "budget", "row", "group", "schedule_id", "off_budget",
    "cache_count", "first_cache_step", "longest_cached_run", "max_full_step_gap",
    "hamming_to_meancache", "transpositions_to_meancache", "hamming_to_budcache",
    "modal_count", "modal_share", "n_tied_at_top",
    *(f"feasible_{payload}" for payload in PAYLOADS),
    "n_cells", "source", "bits",
)


def manifest_row(row: Row) -> dict[str, Any]:
    feasible = feasibility(row.backbone, row.bits)
    columns = list(columns_of(row))
    return {
        "backbone": row.backbone,
        "budget": f"K{row.budget}",
        "row": row.row,
        "group": row.group,
        "schedule_id": row.schedule_id,
        "off_budget": int(row.off_budget),
        "cache_count": row.popcount,
        "first_cache_step": row.provenance["first_cache_step"],
        "longest_cached_run": row.provenance["longest_cached_run"],
        "max_full_step_gap": row.provenance["max_full_step_gap"],
        "hamming_to_meancache": row.provenance["hamming_to_meancache"],
        "transpositions_to_meancache": row.provenance["transpositions_to_meancache"],
        "hamming_to_budcache": row.provenance["hamming_to_budcache"],
        "modal_count": row.provenance.get("count"),
        "modal_share": (None if row.provenance.get("share_of_exact_k_pool") is None
                        else round(float(row.provenance["share_of_exact_k_pool"]), 6)),
        "n_tied_at_top": row.provenance.get("n_tied_at_top"),
        **{f"feasible_{payload}": int(feasible[payload]) for payload in PAYLOADS},
        "n_cells": sum(1 for payload in columns if feasible[payload]),
        "source": row.source,
        "bits": row.bits,
    }


def write_outputs(rows: list[Row], unavailable: list[dict[str, Any]],
                  root: Path, skipped: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, str]] = []
    for row in rows:
        directory = root / row.backbone
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{row.schedule_id}.json"
        path.write_text(json.dumps(row_payload(row), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        if row.row in SPAN_VARIANT_ROWS:
            variant_dir = directory / SPAN_VARIANT_DIR
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / f"{row.schedule_id}.json").write_text(
                json.dumps(span_variant_payload(row), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            variants.append({"backbone": row.backbone, "schedule_id": row.schedule_id,
                             "payload": SPAN_VARIANT_PAYLOAD,
                             "transport": f"{row.backbone}/{SPAN_VARIANT_DIR}/"
                                          f"{row.schedule_id}.json"})
    manifest = root / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS),
                                delimiter="\t", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            payload = manifest_row(row)
            writer.writerow({key: ("" if payload[key] is None else payload[key])
                             for key in MANIFEST_COLUMNS})
    summary = {
        "schema": MANIFEST_SCHEMA,
        "produced_by": "analysis/build_video_spx_schedules.py",
        "plan": "docs/video_sp_cross_plan_zh.md section 2",
        "num_steps": NUM_STEPS,
        "cache_bit_convention": "'1' = cached step, '0' = real computation",
        "budgets": [f"K{k}" for k in KS],
        "payloads": list(PAYLOADS),
        "max_gap": MAX_GAP,
        "rng_recipe": f"numpy.random.default_rng([{RNG_ROOT}, backbone_code, K, draw]) "
                      f"with backbone_code {BACKBONE_CODE}",
        "schedule_count": len(rows),
        "schedules_per_backbone_budget": {
            f"{backbone}/K{k}": sum(1 for row in rows
                                    if row.backbone == backbone and row.budget == k)
            for backbone in BACKBONES for k in KS},
        "cell_count": sum(manifest_row(row)["n_cells"] for row in rows),
        "cells_per_backbone_budget": {
            f"{backbone}/K{k}": sum(manifest_row(row)["n_cells"] for row in rows
                                    if row.backbone == backbone and row.budget == k)
            for backbone in BACKBONES for k in KS},
        "off_budget_rows": unavailable,
        "infeasible_cells": [
            {"backbone": row.backbone, "schedule_id": row.schedule_id,
             "payloads": sorted(payload for payload, ok in feasibility(row.backbone, row.bits).items()
                                if not ok and payload in columns_of(row))}
            for row in rows
            if any(not ok for payload, ok in feasibility(row.backbone, row.bits).items()
                   if payload in columns_of(row))],
        "span_variant_transports": variants,
        "first_step_preserving_ladder_unavailable": list(skipped or []),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_all() -> tuple[list[Row], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[Row] = []
    unavailable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        backbone_rows, backbone_unavailable, backbone_skipped = build_backbone(backbone)
        rows += backbone_rows
        unavailable += backbone_unavailable
        skipped += backbone_skipped
    return rows, unavailable, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, unavailable, skipped = build_all()
    summary = write_outputs(rows, unavailable, args.output_root, skipped)
    print(f"[spx-schedules] {summary['schedule_count']} schedules, "
          f"{summary['cell_count']} (schedule, payload) cells")
    for key, value in sorted(summary["schedules_per_backbone_budget"].items()):
        print(f"  {key}: {value} rows, {summary['cells_per_backbone_budget'][key]} cells")
    for entry in unavailable:
        print(f"  [off-budget] {entry['backbone']} {entry['gate']} K{entry['budget']} "
              f"runs at {entry['realized_cache_count']} cached steps")
    for entry in summary["infeasible_cells"]:
        print(f"  [infeasible] {entry['backbone']} {entry['schedule_id']}: {entry['payloads']}")
    for entry in skipped:
        print(f"  [no-rung] {entry['backbone']} {entry['row']} K{entry['budget']}: "
              f"{entry['reason']} (needs {entry['swaps_needed']} full steps above "
              f"step {entry['anchor_first_cache_step']}, has {entry['swappable_full']})")
    for entry in summary["span_variant_transports"]:
        print(f"  [span-variant] {entry['backbone']} {entry['schedule_id']} -> "
              f"{entry['payload']} via {entry['transport']}")
    print(f"  manifest_sha256 {summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
