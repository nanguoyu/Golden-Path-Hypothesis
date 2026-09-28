#!/usr/bin/env python3
"""Freeze the image-SPX supplement schedules into resources/spx_supplement_schedules/.

    python analysis/build_spx_supplement_schedules.py            # verify against store
    python analysis/build_spx_supplement_schedules.py --write    # (re)write the store

Companion of `docs/sp_cross_supplement_plan_zh.md` section 3. The original 48
image-SPX schedules (`resources/sp_cross_schedules/`) stay untouched; this
store only adds the control rows the video SPX had and the image side lacked,
so the two sides answer the same H-a/H-b/H-c questions:

  rand_1 rand_2        uniform draw of K cached steps from 3..48, {0,1,2,49}
                       forced full -- the "no design at all" reference level;
  ham2f ham4f ham8f    1 / 2 / 4 random (cache step, full step) swaps from the
                       (model, K) MeanCache schedule, both swap ends strictly
                       AFTER the anchor's first cached step, so first_cache_step
                       is the anchor's (the video side's first-step-preserving
                       ladder; the plain ladder is skipped because its lesson --
                       dose confounded with an earlier first skip -- was already
                       learnt there). ham8f needs 4 swappable full steps above
                       the first cached step and both K41 MeanCache tables have
                       only 3, so that rung does not exist at K41 (recorded,
                       not patched);
  dp_rho2              the DP cost-equalisation optimum of the image reference
                       population's rho2 (same definition as the video side:
                       `build_golden_path_family.risk_profiles` component rho2,
                       exponent 1, max_gap 15, {0,1,2,49} forced full). NOT the
                       GPF rows: those used exponent 0.5 and were screened on
                       held-out prompts;
  sencache_top1_off    the gate's UNCONDITIONAL modal path where it is not
                       exact-K -- only SenCache K29 on both models (caches 30
                       steps). Enters the gate-convergence reading only, never
                       the variance decomposition;
  *_top1_r2            rank-2 robustness rows for the three near-tied modal
                       paths (flux dicache K29 292 vs 290, flux teacache K29
                       615 vs 588, qwen teacache K37 455 vs 450); frozen here,
                       run only if the plan's open decision 2 says yes.

  rand_3..rand_5,      the densification of 2026-08-24 (plan section 12): extra
  ham*f_d2 ham*f_d3    independent draws on the four partitions whose basin the
  ham*_d1 ham*_d2      report quotes -- flux K29/K37, qwen K29/K37 -- so a rung
                       is a distribution instead of one draw, and the free
                       ladder (`ham2/4/8`, swap window 3..48, first cached step
                       NOT preserved) exists on the image side for the first
                       time. These rows run one seed, not three.

Determinism: rand and ham draws come from
`numpy.random.default_rng([RNG_ROOT, MODEL_CODE[model], K, draw])` with the
draw indices below; a re-run of this script reproduces every byte of the store.
Rejection rules (popcount changed, Hamming != target, full-step gap > 15) skip
to the next draw of the same stream and the rejection count is recorded.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.build_golden_path_family import (  # noqa: E402
    j_best_schedules,
    read_population,
    risk_profiles,
    schedule_bits,
    segment_cost_matrix,
)
from analysis.build_video_spx_schedules import (  # noqa: E402
    bits_of,
    force_full,
    full_step_gaps,
    hamming,
    steps_of,
    transpositions,
)

NUM_STEPS = 50
KS = (29, 37, 41)
MODELS = ("flux", "qwen")

OUT_DIR = _ROOT / "resources/spx_supplement_schedules"
MANIFEST = OUT_DIR / "manifest.tsv"
SOURCE_SCHEDULES = _ROOT / "resources/sp_cross_schedules"
DISCOVERY_COUNTS = SOURCE_SCHEDULES / "discovery_path_counts.tsv"
TRAJ_TABLE = {
    "flux": _ROOT / "resources/full_trajectory/tables_jsonl/full_traj_flux_parti_full.jsonl",
    "qwen": _ROOT / "resources/full_trajectory/tables_jsonl/full_traj_qwen_parti_full.jsonl",
}

#: identical numerology to the video builder (`build_video_spx_schedules.py`),
#: new root so no image draw can collide with a video one.
RNG_ROOT = 20260823
MODEL_CODE = {"flux": 1, "qwen": 2}
RAND_DRAWS = ((1, "rand_1"), (2, "rand_2"))
LADDER_F_DRAWS = ((2, 21), (4, 22), (8, 23))  # the video side's f-ladder draws
RAND_LOW, RAND_HIGH = 3, NUM_STEPS - 2  # inclusive interior for rand rows
SWAP_HIGH = NUM_STEPS - 2
MAX_GAP = 15
DP_EXPONENT = 1.0
CONTROL_FORCED_FULL = (0, 1, 2, NUM_STEPS - 1)

#: (model, K, gate) whose exact-K modal path is nearly tied with rank 2 --
#: audit finding 7; frozen as robustness rows, run only on decision.
NEAR_TIES = (("flux", 29, "dicache"), ("flux", 29, "teacache"), ("qwen", 37, "teacache"))

#: ---- densification (audit round 3, final recommendation; plan section 12) ----
#: With one draw per rung and two random rows per partition the basin width and
#: the design-gain count are point readings: nothing in the data says how much
#: of a rung's drop is the dose and how much is which draw came out of the
#: stream. These rows add within-rung spread to the four partitions whose basin
#: the report actually quotes (K29 / K37 on both models; K41 is left alone --
#: its ladder is already selection-affected and ham8f does not exist there).
#: Every new row is a NEW draw index on the SAME rng family, so no existing
#: row's bytes move.
DENSE_PARTITIONS = (("flux", 29), ("flux", 37), ("qwen", 29), ("qwen", 37))
#: +3 uniform exact-K rows, same recipe as rand_1/rand_2. That most of them
#: start caching at step 3 is the mechanism being measured, not a bug to fix.
DENSE_RAND_DRAWS = ((3, "rand_3"), (4, "rand_4"), (5, "rand_5"))
#: +2 draws per rung of the first-step-preserving ladder, so each rung has
#: three independent draws counting the frozen one (`LADDER_F_DRAWS`).
DENSE_LADDER_F_DRAWS = ((2, (24, 25)), (4, (26, 27)), (8, (28, 29)))
#: The free ladder the image side never ran: swaps drawn from 3..48, so the
#: first cached step may move earlier. Two draws per rung, which is what makes
#: the two dose curves comparable rung by rung.
DENSE_LADDER_FREE_DRAWS = ((2, (6, 7)), (4, (8, 9)), (8, (10, 11)))
LADDER_FREE_SWAP_LOW = 3
#: The densification runs one seed, not three: it buys within-rung spread on
#: the schedule axis, and the seed band is already measured by the frozen rows.
#: 42 is the seed both models' frozen supplement seed streams share.
DENSE_SEED = 42

MANIFEST_COLUMNS = (
    "model", "target_k", "name", "group", "schedule", "popcount", "k_vs_target",
    "first_cache_step", "max_cached_run", "hamming_to_meancache",
    "transpositions_to_meancache", "count", "mass", "source", "rng", "path",
)


def rng_for(model: str, k: int, draw: int) -> np.random.Generator:
    return np.random.default_rng([RNG_ROOT, MODEL_CODE[model], int(k), int(draw)])


def rng_label(model: str, k: int, draw: int) -> str:
    return f"default_rng([{RNG_ROOT}, {MODEL_CODE[model]}, {k}, {draw}])"


def source_bits(model: str, k: int, stem: str) -> str:
    bits = (SOURCE_SCHEDULES / f"{model}_k{k}_{stem}.txt").read_text().strip()
    if len(bits) != NUM_STEPS or set(bits) - {"0", "1"}:
        raise SystemExit(f"{model}_k{k}_{stem}: not a {NUM_STEPS}-bit 0/1 string")
    return bits


def random_bits(model: str, k: int, draw: int) -> str:
    rng = rng_for(model, k, draw)
    chosen = rng.choice(np.arange(RAND_LOW, RAND_HIGH + 1), size=k, replace=False)
    return bits_of(int(step) for step in chosen)


def ladder_bits(anchor: str, distance: int, model: str, k: int, draw: int,
                *, swap_low: int | None = None,
                ) -> tuple[str | None, dict[str, Any]]:
    """`distance / 2` random (cache step, full step) swaps away from `anchor`.

    `swap_low=None` is the first-step-preserving window (`ham*f`): both swap
    ends are drawn strictly above the anchor's first cached step, so the rung
    inherits the anchor's `first_cache_step`. `swap_low=LADDER_FREE_SWAP_LOW`
    is the free ladder (`ham*`), which may move the first cached step earlier
    -- that confounding is the thing the two curves are compared on, not a
    defect to repair.
    """
    swaps = distance // 2
    if swap_low is None:
        swap_low = min(steps_of(anchor)) + 1
    cached = [s for s in steps_of(anchor) if swap_low <= s <= SWAP_HIGH]
    full = [s for s in range(swap_low, SWAP_HIGH + 1) if anchor[s] == "0"]
    recipe: dict[str, Any] = {
        "rng": rng_label(model, k, draw), "swaps": swaps,
        "swap_window": (swap_low, SWAP_HIGH),
        "swappable_cached": len(cached), "swappable_full": len(full),
    }
    if len(cached) < swaps or len(full) < swaps:
        recipe["unavailable"] = "too few swappable full steps in the window"
        return None, recipe
    rng = rng_for(model, k, draw)
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
        recipe["draws_rejected"] = attempt - 1
        return bits, recipe
    recipe["unavailable"] = "1000 draws all violated the gap cap"
    return None, recipe


def dp_rho2_bits(model: str, k: int) -> tuple[str, float, dict[str, Any]]:
    population = read_population(TRAJ_TABLE[model], num_steps=NUM_STEPS)
    profiles = risk_profiles(population)
    cost = segment_cost_matrix(profiles.rho2, population.sigmas[:NUM_STEPS],
                               exponent=DP_EXPONENT, max_gap=MAX_GAP)
    cost = force_full(cost, CONTROL_FORCED_FULL[1:-1])
    best = j_best_schedules(cost, n_full=NUM_STEPS - k, j_best=1)
    if not best:
        raise SystemExit(f"{model} dp_rho2 K={k}: no feasible schedule at max_gap={MAX_GAP}")
    value, full_steps = best[0]
    bits = schedule_bits(full_steps, num_steps=NUM_STEPS)
    if bits.count("1") != k:
        raise SystemExit(f"{model} dp_rho2 K={k}: built {bits.count('1')} cached steps")
    meta = {
        "rho2_rows": population.n_rows, "rho2_window": profiles.rho2_window,
        "dp_cost": float(value),
    }
    return bits, float(value), meta


def load_discovery() -> list[dict[str, str]]:
    with DISCOVERY_COUNTS.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def gate_rows(rows: list[dict[str, str]], model: str, k: int, gate: str,
              ) -> list[dict[str, str]]:
    group = [r for r in rows
             if r["model"] == model and r["method"] == gate and int(r["target_k"]) == k]
    return sorted(group, key=lambda r: (-int(r["count"]), r["schedule"]))


def build_rows() -> list[dict[str, Any]]:
    discovery = load_discovery()
    out: list[dict[str, Any]] = []

    def add(model: str, k: int, name: str, group: str, bits: str, source: str,
            rng: str = "", count: str = "", mass: str = "") -> None:
        mean = source_bits(model, k, "meancache")
        same_pop = bits.count("1") == mean.count("1")
        out.append({
            "model": model, "target_k": k, "name": name, "group": group,
            "schedule": bits, "popcount": bits.count("1"),
            "k_vs_target": bits.count("1") - k,
            "first_cache_step": bits.index("1"),
            "max_cached_run": max((len(r) for r in bits.replace("0", " ").split()), default=0),
            "hamming_to_meancache": hamming(bits, mean),
            "transpositions_to_meancache": transpositions(bits, mean) if same_pop else "",
            "count": count, "mass": mass, "source": source, "rng": rng,
            "path": f"resources/spx_supplement_schedules/{model}_k{k}_{name}.txt",
        })

    for model in MODELS:
        for k in KS:
            for draw, name in RAND_DRAWS:
                add(model, k, name, "control", random_bits(model, k, draw),
                    f"constructed: uniform draw of {k} cached steps from "
                    f"{RAND_LOW}..{RAND_HIGH}, {{0,1,2,49}} full",
                    rng=rng_label(model, k, draw))
            anchor = source_bits(model, k, "meancache")
            for distance, draw in LADDER_F_DRAWS:
                bits, recipe = ladder_bits(anchor, distance, model, k, draw)
                if bits is None:
                    print(f"[skip] {model} K{k} ham{distance}f: {recipe['unavailable']} "
                          f"(window {recipe['swap_window']}, "
                          f"{recipe['swappable_full']} swappable full steps)")
                    continue
                add(model, k, f"ham{distance}f", "control", bits,
                    f"constructed: {distance // 2} random (cache, full) swaps from "
                    f"{model}_k{k}_meancache, swap window {recipe['swap_window'][0]}.."
                    f"{recipe['swap_window'][1]} (first cached step preserved), "
                    f"gap cap {MAX_GAP}, {recipe['draws_rejected']} draws rejected",
                    rng=recipe["rng"])
            bits, _, meta = dp_rho2_bits(model, k)
            add(model, k, "dp_rho2", "control", bits,
                f"constructed: rho2 cost-equalisation DP on "
                f"{TRAJ_TABLE[model].relative_to(_ROOT)} "
                f"({meta['rho2_rows']} trajectories, window {meta['rho2_window']}), "
                f"exponent {DP_EXPONENT}, max_gap {MAX_GAP}, {{0,1,2,49}} full, "
                f"dp_cost {meta['dp_cost']:.6g}")
            # off-budget gate rows: unconditional modal path not exact-K
            for gate in ("seacache", "teacache", "sencache", "dicache"):
                ranked = gate_rows(discovery, model, k, gate)
                if not ranked:
                    raise SystemExit(f"{model} K{k} {gate}: no discovery rows")
                top = ranked[0]
                if int(top["cache_count"]) != k:
                    add(model, k, f"{gate}_top1_off", "off_budget", top["schedule"],
                        f"discovery_path_counts.tsv: {gate} unconditional rank-1, "
                        f"caches {top['cache_count']} of target {k}",
                        count=top["count"], mass=top["mass"])
    for model, k, gate in NEAR_TIES:
        ranked = [r for r in gate_rows(load_discovery(), model, k, gate)
                  if int(r["cache_count"]) == k]
        if len(ranked) < 2:
            raise SystemExit(f"{model} K{k} {gate}: no exact-K rank-2 path")
        second = ranked[1]
        add(model, k, f"{gate}_top1_r2", "robustness", second["schedule"],
            f"discovery_path_counts.tsv: {gate} exact-K rank-2 "
            f"(near-tie with rank-1: {ranked[0]['count']} vs {second['count']})",
            count=second["count"], mass=second["mass"])

    # ---- densification: extra draws on the four wide-basin partitions.
    # Appended after the frozen rows so every earlier manifest line keeps its
    # position as well as its bytes.
    for model, k in DENSE_PARTITIONS:
        for draw, name in DENSE_RAND_DRAWS:
            add(model, k, name, "dense", random_bits(model, k, draw),
                f"constructed: uniform draw of {k} cached steps from "
                f"{RAND_LOW}..{RAND_HIGH}, {{0,1,2,49}} full",
                rng=rng_label(model, k, draw))
        anchor = source_bits(model, k, "meancache")
        for distance, draws in DENSE_LADDER_F_DRAWS:
            for ordinal, draw in enumerate(draws, start=2):
                bits, recipe = ladder_bits(anchor, distance, model, k, draw)
                if bits is None:
                    raise SystemExit(
                        f"{model} K{k} ham{distance}f draw {draw}: {recipe['unavailable']}")
                add(model, k, f"ham{distance}f_d{ordinal}", "dense", bits,
                    f"constructed: {distance // 2} random (cache, full) swaps from "
                    f"{model}_k{k}_meancache, swap window {recipe['swap_window'][0]}.."
                    f"{recipe['swap_window'][1]} (first cached step preserved), "
                    f"gap cap {MAX_GAP}, {recipe['draws_rejected']} draws rejected",
                    rng=recipe["rng"])
        for distance, draws in DENSE_LADDER_FREE_DRAWS:
            for ordinal, draw in enumerate(draws, start=1):
                bits, recipe = ladder_bits(anchor, distance, model, k, draw,
                                           swap_low=LADDER_FREE_SWAP_LOW)
                if bits is None:
                    raise SystemExit(
                        f"{model} K{k} ham{distance} draw {draw}: {recipe['unavailable']}")
                add(model, k, f"ham{distance}_d{ordinal}", "dense", bits,
                    f"constructed: {distance // 2} random (cache, full) swaps from "
                    f"{model}_k{k}_meancache, swap window {recipe['swap_window'][0]}.."
                    f"{recipe['swap_window'][1]} (first cached step free), "
                    f"gap cap {MAX_GAP}, {recipe['draws_rejected']} draws rejected",
                    rng=recipe["rng"])
    return out


def render_manifest(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, delimiter="\t",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in MANIFEST_COLUMNS})
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="write the store (default: verify against it)")
    args = parser.parse_args()

    rows = build_rows()
    manifest = render_manifest(rows)
    files = {MANIFEST: manifest}
    for row in rows:
        files[_ROOT / row["path"]] = row["schedule"] + "\n"

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for path, content in sorted(files.items()):
            path.write_text(content, encoding="utf-8")
        print(f"wrote {len(files) - 1} schedules + manifest to {OUT_DIR.relative_to(_ROOT)}")
    else:
        stale = [path for path, content in files.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit("store differs from a fresh build (run with --write): "
                             + ", ".join(str(p.relative_to(_ROOT)) for p in sorted(stale)))
        print(f"store verified: {len(files) - 1} schedules + manifest byte-identical")

    by_group: dict[str, int] = {}
    for row in rows:
        by_group[row["group"]] = by_group.get(row["group"], 0) + 1
    print("rows:", len(rows), by_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
