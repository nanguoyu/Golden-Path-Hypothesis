#!/usr/bin/env python3
"""Density-form test of the geometric cost model rho2 against the video matrix's
schedule axis — plan section 3.7 (docs/video_full_trajectory_plan_zh.md, P5).

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
    python analysis/video_trajectory/density_form.py --backbone hunyuan_video \\
        --data_root outputs

What the test asks (identical to the image side, `analysis/density_form_test.py`):
the geometric cost model `sum rho2 * dsigma` does NOT predict that full steps
land on high-rho2 steps, it predicts *cost equalisation* — a gap covering a
high-rho2 stretch should be short in sigma, a gap over a low-rho2 stretch long.
So per schedule, for every maximal run of cached steps between two full steps:

    x = mean rho2 inside the gap
    y = sigma span, `sigma_anchor - sigma_{first full step after the gap}`
        (plan line 324: from the anchor to where the payload is replaced)
    c = sum_n rho2[n] * |sigma_anchor - sigma_n|

and report `spearman(x, y)` (cost equalisation predicts negative) plus the
coefficient of variation of `c`. `'1'` = cache, `'0'` = full throughout, the
`density_form_test.gaps_of` convention, which is also what
`merge_video_traj.py` writes into the merged table's `actions` field.

Objects under test, per backbone x per K in {29, 37, 41} (plan lines 329-338):

  * the three frozen fixed tables — `schedule_tables.budcache_K*`,
    `meancache_K*`, and HYV `triplet_K*` / Wan `shared_K*` (one table shared by
    taylorseer_o1 / hicache_o2 / l2p) — `cache_steps` turned into 50 bits;
  * the four dynamic gates' modal paths (seacache / teacache / sencache /
    dicache), one per dataset, read off the merged table's `actions` strings;
  * a positive control (a DP optimum of this very rho2) and a negative control
    (uniform placement).

Reuse, and the two places it had to be adapted at the call site
--------------------------------------------------------------
rho2 and the DP come from `analysis/build_golden_path_family.py`; the gap
statistics come from `analysis/density_form_test.py`. Neither maths is forked.
Two adaptations, both at the call site:

1. `read_population` cannot be pointed at `t1_merged.jsonl`: it assumes one
   dataset, one file, and every row clean, while the merged table holds 129,612
   rows of which 4,629 are references from two datasets and six streams. So the
   references are streamed with `step_profiles.load_references` and the frozen
   `Population` dataclass is constructed here, per dataset, from exactly the
   same accumulation (`d_perp/chord` and `spacing/|dsigma|/chord`, per-row
   normalisation BEFORE the population MEAN). `risk_profiles` then runs
   unchanged.
2. `measure` computes a Spearman as soon as there are 3 gaps, while only the
   image-side `main` applies `MIN_GAPS = 4`. The gate is applied here, as
   `main` does, so a 3-gap row cannot leak a rank statistic into the table.

The P4 arrays cannot supply rho2: `step_profiles_<T>.json` stores the MEDIAN
`d_perp/chord` and a velocity divided by sqrt(d), while rho2 needs the MEAN and
both terms divided by the chord. P5 therefore re-streams the merged table.

Two more differences from the image side, both required by the plan:

  * the image-side modal path conditions on `cache_count = K`; the video gates
    do not realise K (plan line 335), so the modal string is taken
    unconditionally and the realised count is reported next to the nominal one;
  * the plan says "that cell", but one row per (gate, dataset, K) needs the
    three seed-cells of that configuration pooled. They are pooled, and the
    per-seed modal strings are recorded too, so a seed-dependent mode is
    visible rather than hidden.

The permutation null (why a negative Spearman means nothing on its own)
----------------------------------------------------------------------
The plan's density correlation is SATURATED on both video backbones: it comes
out strongly negative for every object, the uniform negative control included,
so as published it cannot tell a designed schedule from an undesigned one. The
mechanism is a coupling between the rho2 shape and the non-uniform sigma grid —
rho2 is high early where dsigma is small and low late where dsigma is large — so
ANY schedule pairs its early gaps' high local rho2 with a short sigma span and
its late gaps' low local rho2 with a long one. The image side does not show this
(its uniform control sits at -0.015 / -0.14 / -0.40, p = 0.96 / 0.74 / 0.60 in
`resources/full_trajectory_analysis/a2_density_table.tsv`), so it is a property
of the video sigma grids, not of either implementation.

`--null_draws` random schedules per (rho2 dataset, K) therefore give the
statistic a reference distribution. A draw is a random K-subset of the steps a
real object here is allowed to cache, scored through the SAME code path as every
real row (`density_form_test.measure`, i.e. `gaps_of` + the plan's sigma-span
rule + `scipy.stats.spearmanr`) and dropped when it has fewer than `MIN_GAPS`
gaps, exactly as a real row would be. Each row then carries an empirical
percentile of its own statistic inside that null, and that percentile — not the
sign of the Spearman — is what says whether the schedule is unusual. The gap-cost
CV, which does discriminate in the predicted order, gets the same treatment
because a null strengthens a statistic that works as much as it deflates one
that does not.

The null conditions on the row's REALISED cache count, not on its nominal
budget: a gate modal path that caches 36 steps is compared with random 36-step
schedules, so nothing of the comparison rides on a K the schedule does not have.
The nominal budgets are always drawn as well, so the summary table has its
K29/K37/K41 rows whatever the gates realised.

Floors are cited, never hardcoded: the T3 curvature cross-check is a bf16
reading and cites the **bfloat16** P1 row, everything else is a T1 reading and
cites the **float32** row; both come out of
`docs/figures/video_full_trajectory/<T>/p1_floor_{float32,bfloat16}.json`
through `step_profiles.load_p1_floor`. The cross-check window is a knob
(`--xcheck_windows`) because the readable window is a per-backbone P1 result,
and a pass run below the bf16 floor says so in the JSON and on the page.

Four smaller rules that are easy to get wrong and are therefore explicit:

  * only `cells/` rows feed the modal counter — `cells_t3_rand50/` carries the same
    mode/dataset/budget/seed for ten re-run prompts and would double-count them
    (the mirror of `step_profiles.load_references`' `REFERENCE_PREFIX` guard);
  * a schedule may not cache step 0 (the anchor rule breaks), and a gate modal
    path that caches the LAST step is kept but flagged `tail_gap_open`, because
    its last gap has no closing full step and runs to sigma = 0;
  * the documented realised-K numbers describe a configuration, the modal path
    is one draw from a per-video K+-2 distribution, so a disagreement is a
    flagged row (`k_documented_mismatch`), not a dead run;
  * the section 3.1 kink verdict is read from the REPO's P4 JSON (`--p4_json`),
    not from `--out_tables`, which an operator is told to redirect.
"""

from __future__ import annotations

import os

# BLAS pools must be capped before numpy loads (plan section 5.2).
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.build_golden_path_family import (  # noqa: E402
    Population,
    cross_check_rho2,
    j_best_schedules,
    risk_profiles,
    schedule_bits,
    segment_cost_matrix,
)
from analysis.density_form_test import MIN_GAPS, flatten_span, measure  # noqa: E402
from analysis.video_trajectory.common import (  # noqa: E402
    atomic_savefig,
    atomic_write_json,
    atomic_write_text,
    output_complete,
    resolve_reuse,
)
from analysis.video_trajectory.step_profiles import (  # noqa: E402
    load_index,
    load_p1_floor,
    load_references,
)

BACKBONES = ("hunyuan_video", "wan21")
DEFAULT_DATA_ROOT = Path("outputs")
NUM_STEPS = 50
N_STATES = 51
KS = (29, 37, 41)
GATES = ("seacache", "teacache", "sencache", "dicache")
SHARED_METHODS = ("taylorseer_o1", "hicache_o2", "l2p")
MAIN_DATASET = "vbench944"          # plan line 343: main table, penguin replicates
RHO2_WINDOW = 5                     # >= 5 is enforced by risk_profiles itself
DEFAULT_XCHECK_WINDOWS = (5,)       # --xcheck_windows; each one re-reads every path
XCHECK_MARGIN = 3
DEFAULT_XCHECK_PATHS = 60           # plan line 344: T3 >= 60 per backbone
MIN_XCHECK_PATHS = 60               # the same floor, as a warning threshold
CELL_PREFIX = "cells/"              # NOT cells_t3_rand50/: the path-layer re-runs
SEED_CELLS_PER_CONFIG = 3           # three base seeds per (gate, dataset, budget)

# --- permutation null over schedules -------------------------------------
DEFAULT_NULL_DRAWS = 500            # --null_draws; 0 skips the null entirely
DEFAULT_NULL_SEED = 20260819        # --null_seed; recorded so a run is reproducible
NULL_BAND = (5.0, 95.0)             # the band `*_outside_null_90` is outside
NULL_FRAC_THRESHOLD = -0.5          # `null_frac_below_-0.5`: how saturated the sign is

# the third fixed table has a different key in the two frozen configs
SHARED_TABLE_KEY = {"hunyuan_video": "triplet", "wan21": "shared"}

# docs/video_full_results_report_zh.md section 5.1 (realised K) — the structural
# realised-K exceptions P0 already checked. Those numbers describe the
# CONFIGURATION's realised step count; the gate is per-video K+-2 (plan line
# 335), so a modal string that lands beside them is reported as a mismatch, not
# treated as a reason to throw the other 77 rows away. (HYV SenCache K41
# realises its tier since the second frozen knob; K29-side one-step cells
# HYV TeaCache K29=28 and Wan SeaCache K29=30 are in the footnote too.)
DOCUMENTED_K = {
    "hunyuan_video": {("sencache", 37): 36, ("teacache", 29): 28},
    "wan21": {("seacache", 29): 30, ("seacache", 41): 40, ("teacache", 37): 36},
}
# No two tiers currently share one configuration; the mechanism stays for the
# next gate that saturates. Each entry is (group id, is this the instance the
# summary counts).
DUPLICATE_CONFIG: dict[str, dict[tuple[str, int], tuple[str, bool]]] = {
    "hunyuan_video": {},
    "wan21": {},
}

FAMILY_COLOURS = {
    "searched": "#d62728", "shared-fixed": "#9467bd", "gate modal": "#1f77b4",
    "dp-positive": "#2ca02c", "uniform-null": "#7f7f7f",
}

TSV_COLUMNS = [
    "schedule", "family", "dataset", "rho2_dataset", "matched_dataset",
    "K_nominal", "k_realized", "k_realized_minus_nominal", "duplicate_config",
    "duplicate_group", "counted_instance", "k_deviation_documented",
    "k_documented", "k_documented_mismatch", "n_gaps", "tail_gap_open",
    "spearman", "p", "cost_cv", "null_k", "spearman_null_percentile",
    "spearman_outside_null_90", "cost_cv_null_percentile",
    "cost_cv_outside_null_90", "modal_count", "modal_share", "modal_tie",
    "n_distinct_paths", "n_rows", "n_seed_cells",
]

# Consequence of clamped (not truncated) smoothing windows, documented at
# build_golden_path_family.py:56-63 — restated, never re-derived.
END_CLAMP_BANDS = {"rho2_window": RHO2_WINDOW, "constant_steps": [[0, 2], [48, 49]]}

# quoted verbatim into the report's positive-control block, so it is a name and
# not `CAVEATS[i]`: inserting a caveat above it used to move the index silently
DP_CONTROL_CAVEAT = (
    "The DP positive control is a positive control FOR THE STATISTIC ONLY "
    "(density_form_test.py:26-31): it is a DP optimum of this very rho2, so it says "
    "the test can see cost equalisation when it is there, and says nothing about "
    "whether rho2 is the right geometry.")

SATURATION_CAVEAT = (
    "THE SIGN OF THE DENSITY SPEARMAN IS SATURATED ON THIS BACKBONE. Random "
    "K-subsets of the same steps come out strongly negative too, and so does the "
    "uniform negative control, because rho2 is high early where dsigma is small and "
    "low late where dsigma is large, so every schedule pairs short early spans with "
    "high local rho2. A negative Spearman is therefore NOT evidence that a schedule "
    "was designed; read spearman_null_percentile (and the schedule_null block) "
    "instead. The image side does not saturate (uniform control -0.015 / -0.14 / "
    "-0.40 at p = 0.96 / 0.74 / 0.60, resources/full_trajectory_analysis/"
    "a2_density_table.tsv), so this is a property of the video sigma grids.")

CAVEATS = [
    SATURATION_CAVEAT,
    "rho2 is a RELATIVE profile and must not be quoted as an absolute curvature "
    "magnitude (build_golden_path_family.py:73-79): the component route reproduces "
    "the direct measure's shape, not its scale, and the residual mismatch is not a "
    "single global factor.",
    f"Windows are clamped, not truncated, at the ends, so with rho2_window="
    f"{RHO2_WINDOW} the profile is CONSTANT over steps 0-2 and 48-49. Within those "
    f"bands the cost cannot rank one step against another; it costs the test nothing "
    f"(steps 0 and 49 are full under every object here) but the profile does not "
    f"resolve structure there and must not be read as if it did.",
    DP_CONTROL_CAVEAT,
    "The dynamic gates do not realise the nominal K (plan line 335). Rows are filed "
    "under the nominal budget and the realised cache count is a separate column. The "
    "three structurally documented deviations are CHECKED against the modal string and "
    "a disagreement is flagged loudly (k_documented / k_documented_mismatch), not "
    "silently accepted: the documented numbers describe the configuration, the modal "
    "string is one path, and the two can differ without either being wrong.",
    "Gate rows exist for both rho2 datasets. Only the rows with matched_dataset=true "
    "read a gate's schedule against the rho2 of the dataset it ran on; the "
    "off-diagonal rows are a cross-dataset sensitivity check and are tabulated "
    "separately in the markdown page. The figure plots the diagonal only.",
    "Modal paths pool the three seed-cells of one (gate, dataset, K) configuration; "
    "the per-seed modal strings are in the JSON so a seed-dependent mode is visible.",
    "The along-chord term of rho2 lives on the step-midpoint sigma grid and is "
    "carried to step index n; that half-step offset is an order of magnitude below "
    "the resolution the smoothing window imposes "
    "(build_golden_path_family.py:42-44).",
]


# ---------------------------------------------------------------------------
# bit strings
# ---------------------------------------------------------------------------


def bits_from_cache_steps(cache_steps: list[int], *, num_steps: int = NUM_STEPS) -> str:
    """`cache_steps` -> 50 bits with `'1'` = cache (plan line 322)."""
    cached = {int(s) for s in cache_steps}
    out_of_range = sorted(s for s in cached if not 0 <= s < num_steps)
    if out_of_range:
        raise SystemExit(f"cache_steps outside 0..{num_steps - 1}: {out_of_range}")
    return "".join("1" if n in cached else "0" for n in range(num_steps))


TAIL_GAP_NOTE = (
    "the last gap runs to the end of the schedule: step 49 is cached, so there is no "
    "'first full step after the gap' and `density_form_test.measure` closes that gap on "
    "sigmas[50] = the terminal sigma. The plan's span definition (line 324) has no such "
    "step, and the last step carries the largest single-step dsigma of the schedule, so "
    "this is the most leverage-bearing point in the scatter")


def check_bits(bits: str, label: str, *, num_steps: int = NUM_STEPS,
               allow_tail_gap: bool = False) -> str:
    """Every object under test has to be a `num_steps` 0/1 string whose step 0 is
    a real computation, and (unless the caller allows it) whose last step is too.

    The alphabet/length guard is `density_form_test.py:171-172`. The step-0 guard
    is new and load-bearing: the sigma-span anchor is `sigmas[max(first - 1, 0)]`
    (`density_form_test.py:114`), so a gap starting at step 0 would silently take
    its own first step as the anchor and report a span that is not the plan's
    definition. No `cache_steps` table contains 0, and no gate should ever cache
    the first step; if one does, the row is refused loudly instead.

    The last step is the mirror image of the same hole (`TAIL_GAP_NOTE`). For a
    frozen table or a control it is a construction error and is refused; for a
    gate modal path it is data, so the caller passes `allow_tail_gap=True`, the
    row is kept, and `tail_gap_open` records that its last span runs to sigma = 0
    rather than to a full step.
    """
    if len(bits) != num_steps or set(bits) - {"0", "1"}:
        raise SystemExit(f"{label}: not a {num_steps}-character 0/1 bitstring "
                         f"(len={len(bits)}, alphabet={sorted(set(bits))})")
    if bits[0] == "1":
        raise SystemExit(
            f"{label}: step 0 is cached. The sigma-span anchor rule (plan line 324, "
            f"density_form_test.py:114 `sigmas[max(first-1, 0)]`) assumes the step "
            f"before every gap is a full step; with step 0 cached the anchor would "
            f"silently become the gap's own first step. Refusing this row rather than "
            f"reporting a span that is not the plan's definition.")
    if bits[-1] == "1":
        if not allow_tail_gap:
            raise SystemExit(
                f"{label}: step {num_steps - 1} is cached, i.e. {TAIL_GAP_NOTE}. Every "
                f"frozen table and every control forces the last step full, so this is a "
                f"construction error rather than a reading; refusing it.")
        print(f"[WARN] {label}: {TAIL_GAP_NOTE}. The row is kept and flagged "
              f"tail_gap_open=true; read its last span accordingly.")
    return bits


def has_tail_gap(bits: str) -> bool:
    """Does the schedule's last gap run past the end (no closing full step)?"""
    return bits[-1] == "1"


def _finite(value: Any) -> float | None:
    """A float for the JSON, or None when it is not finite.

    `json.dumps` writes a bare `NaN`, which strict parsers reject; the P4 scripts
    map non-finite readings to null (`step_profiles._series`) and this table does
    the same. `spearmanr` returns NaN on a degenerate rank input and `cost_cv` is
    NaN when the gap costs sum to zero.
    """
    v = float(value)
    return v if math.isfinite(v) else None


def uniform_bits(k: int, *, num_steps: int = NUM_STEPS) -> str:
    """Negative control: `num_steps - k` full steps spread evenly, 0 and 49 full."""
    full = np.unique(np.round(np.linspace(0, num_steps - 1, num_steps - k)).astype(int))
    if full.size != num_steps - k:
        raise SystemExit(f"uniform K={k}: rounding collapsed {num_steps - k} full steps "
                         f"onto {full.size} distinct indices")
    bits = schedule_bits(full.tolist(), num_steps=num_steps)
    if bits.count("1") != k:
        raise SystemExit(f"uniform K={k}: built {bits.count('1')} cached steps")
    return bits


def dp_bits(rho2: np.ndarray, sigmas: np.ndarray, k: int, *,
            num_steps: int = NUM_STEPS) -> tuple[str, float]:
    """Positive control: the cheapest schedule under this very rho2.

    `exponent = 1.0` is the `reuse` / m = 0 payload rho2 belongs to
    (build_golden_path_family.py:15-17); forced full steps {0, num_steps - 1} are
    structural in the DP; `max_gap` is left off so the control is the unconstrained
    optimum of the cost the test measures.
    """
    cost = segment_cost_matrix(rho2[:num_steps], sigmas[:num_steps],
                               exponent=1.0, max_gap=None)
    best = j_best_schedules(cost, n_full=num_steps - k, j_best=1)
    if not best:
        raise SystemExit(f"DP found no schedule with {num_steps - k} full steps")
    value, full_steps = best[0]
    bits = schedule_bits(full_steps, num_steps=num_steps)
    if bits.count("1") != k:
        raise SystemExit(f"DP K={k}: built {bits.count('1')} cached steps")
    return bits, float(value)


# ---------------------------------------------------------------------------
# population profiles (adaptation 1: Population built at the call site)
# ---------------------------------------------------------------------------


def build_population(refs: Any, dataset: str, sigmas: np.ndarray, *,
                     merged: Path, backbone: str,
                     num_steps: int = NUM_STEPS) -> Population:
    """The `Population` of one dataset's clean references.

    Same accumulation as `build_golden_path_family.read_population:256-272` —
    each row normalised by its OWN chord before averaging, population MEAN (not
    median), `speed = spacing / |dsigma| / chord` — only vectorised over the
    already-streamed column arrays instead of re-reading a one-dataset jsonl.
    """
    mask = refs.mask(dataset=dataset)
    n_rows = int(mask.sum())
    if n_rows == 0:
        raise SystemExit(f"no reference rows for dataset {dataset!r}")
    chord = refs.cols["chord_len"][mask][:, None]
    if not np.all(chord > 0.0):
        raise SystemExit(f"{dataset}: non-positive chord_len in {int((chord <= 0).sum())} rows")
    d_sigma = np.abs(np.diff(sigmas))
    d_perp = refs.cols["d_perp"][mask]
    spacing = refs.cols["spacing"][mask]
    if d_perp.shape[1] != num_steps + 1 or spacing.shape[1] != num_steps:
        raise SystemExit(f"{dataset}: profile lengths {d_perp.shape[1]}/{spacing.shape[1]}, "
                         f"want {num_steps + 1}/{num_steps}")
    return Population(
        source=Path(merged),
        model=backbone,
        dataset=dataset,
        n_rows=n_rows,
        num_steps=int(num_steps),
        sigmas=np.asarray(sigmas, dtype=np.float64),
        sigmas_mid=0.5 * (sigmas[:-1] + sigmas[1:]),
        deviation=(d_perp / chord).mean(axis=0),
        speed=(spacing / chord / d_sigma).mean(axis=0),
    )


# ---------------------------------------------------------------------------
# gate modal paths (from the merged table's `actions`, never decisions_*.json)
# ---------------------------------------------------------------------------


def scan_cell_actions(merged: Path, *, gates: tuple[str, ...] = GATES,
                      limit: int | None = None, verbose: bool = True,
                      ) -> tuple[dict[tuple[str, str, int], dict[int, collections.Counter]],
                                 dict[str, Any]]:
    """`({(gate, dataset, K): {base_seed: Counter(bits)}}, meta)` from `t1_merged.jsonl`.

    The merged table already carries the 50-character `actions` string per cell
    row (`merge_video_traj.py:134-140`, `'1'` = cache), so the 129,612 small
    `decisions_*.json` files are never opened. The cheap substring prefilter is
    re-checked after parsing, exactly as `load_references` does.

    `cells_t3_rand50/` rows are dropped, and counted so the drop is visible. The merger
    tags them `kind="cell"` with the same mode/dataset/budget/base_seed as the
    matrix cell (`merge_video_traj.py:118-122`), so pooling them would double-count
    the sampled path-layer prompts and can move the modal string itself. This is the
    mirror of `step_profiles.load_references`' `REFERENCE_PREFIX` guard.
    """
    out: dict[tuple[str, str, int], dict[int, collections.Counter]] = {}
    per_cell: collections.Counter = collections.Counter()
    per_dir: collections.Counter = collections.Counter()
    n_lines = n_parsed = n_t3_dropped = 0
    with open(merged, encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if not any(f'"{g}"' in line for g in gates):
                continue
            rec = json.loads(line)
            mode = rec.get("mode")
            if mode not in gates or rec.get("kind") != "cell":
                continue
            source_dir = str(rec.get("source_dir", ""))
            if not source_dir.startswith(CELL_PREFIX):
                n_t3_dropped += 1   # path-layer re-runs of the sampled prompts
                continue
            n_parsed += 1
            per_dir[source_dir] += 1
            budget = rec.get("budget")
            if not isinstance(budget, str) or not budget.startswith("K"):
                raise SystemExit(f"{rec.get('source_dir')}: budget {budget!r} is not K<n>")
            k = int(budget[1:])
            dataset = str(rec["dataset"])
            base_seed = int(rec["base_seed"])
            key = (mode, dataset, k)
            cell = (mode, dataset, k, base_seed)
            if limit is not None and per_cell[cell] >= limit:
                continue
            per_cell[cell] += 1
            bits = rec.get("actions")
            if not isinstance(bits, str) or len(bits) != NUM_STEPS or set(bits) - {"0", "1"}:
                raise SystemExit(f"{rec.get('source_dir')} idx {rec.get('prompt_idx')}: "
                                 f"actions is not a {NUM_STEPS}-character 0/1 string "
                                 f"({bits!r})")
            n_cached = rec.get("n_cached")
            if n_cached is not None and int(n_cached) != bits.count("1"):
                raise SystemExit(f"{rec.get('source_dir')} idx {rec.get('prompt_idx')}: "
                                 f"n_cached={n_cached} but actions has {bits.count('1')} "
                                 f"cached steps")
            out.setdefault(key, {}).setdefault(base_seed, collections.Counter())[bits] += 1
    meta = {
        "n_lines": int(n_lines),
        "n_gate_cell_rows": int(n_parsed),
        "n_groups": len(out),
        "n_cell_dirs": len(per_dir),
        "n_dropped_not_cells_prefix": int(n_t3_dropped),
        "cell_prefix": CELL_PREFIX,
        "dropped_note": "rows whose source_dir is not under cells/ (i.e. the "
                        "cells_t3_rand50/ path-layer re-runs) are NOT pooled into the "
                        "modal counter; they would double-count those prompts",
    }
    if verbose:
        print(f"  scanned {n_lines:,} rows for gate decisions; parsed {n_parsed:,} gate "
              f"cell rows from {len(per_dir)} cells/ directories into {len(out)} "
              f"(gate, dataset, K) groups; dropped {n_t3_dropped:,} non-cells/ rows "
              f"(cells_t3_rand50/)", flush=True)
    return out, meta


def modal_path(seed_counters: dict[int, collections.Counter]) -> dict[str, Any]:
    """Pooled highest-frequency 0/1 string, lexical tie-break.

    The image-side rule (`resources/sp_cross_schedules/README.md:17`) conditions
    on `cache_count = K` before taking the top-1; the video gates do not realise
    K (plan line 335), so that condition is dropped and the realised count is
    reported instead. The three seed-cells of one configuration are pooled — they
    are the same gate under three noise draws — and each seed's own modal string
    is kept so a seed-dependent mode shows up in the JSON.
    """
    pooled: collections.Counter = collections.Counter()
    for counter in seed_counters.values():
        pooled.update(counter)
    if not pooled:
        raise SystemExit("no decisions rows for this (gate, dataset, K)")
    top = max(pooled.values())
    tied = sorted(b for b, c in pooled.items() if c == top)
    bits = tied[0]
    n_rows = int(sum(pooled.values()))
    per_seed = {}
    for seed, counter in sorted(seed_counters.items()):
        seed_top = max(counter.values())
        seed_tied = sorted(b for b, c in counter.items() if c == seed_top)
        seed_bits = seed_tied[0]
        per_seed[str(seed)] = {
            "bits": seed_bits, "count": int(seed_top), "rows": int(sum(counter.values())),
            "distinct_paths": len(counter), "k_realized": seed_bits.count("1"),
            "n_tied_at_top": len(seed_tied), "same_as_pooled": seed_bits == bits,
        }
    return {
        "bits": bits, "modal_count": int(top), "n_rows": n_rows,
        "modal_share": float(top) / n_rows, "n_distinct_paths": int(len(pooled)),
        "n_tied_at_top": len(tied),
        "modal_tie": len(tied) > 1,
        "tie_break": "lexicographically smallest of the tied strings",
        "tied_bits": tied if len(tied) > 1 else None,
        "n_seed_cells": len(seed_counters),
        "per_seed": per_seed,
        "seeds_agreeing_with_pooled": sum(1 for v in per_seed.values() if v["same_as_pooled"]),
    }


# ---------------------------------------------------------------------------
# objects under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Obj:
    label: str
    family: str
    dataset: str | None       # the schedule's own dataset (gate rows only)
    k_nominal: int
    bits: str
    duplicate_config: bool = False
    duplicate_group: str | None = None
    counted_instance: bool | None = None
    k_deviation_documented: bool = False
    k_documented: int | None = None
    k_documented_mismatch: bool = False
    modal: dict[str, Any] | None = None
    dp_cost: float | None = None


def fixed_table_objects(config: dict[str, Any], backbone: str) -> list[Obj]:
    """The three frozen tables per K (plan lines 330-332)."""
    tables = config["schedule_tables"]
    shared_key = SHARED_TABLE_KEY[backbone]
    shared_label = f"{shared_key} ({'/'.join(SHARED_METHODS)})"
    objs: list[Obj] = []
    for k in KS:
        for key, label, family in (
            (f"budcache_K{k}", "budcache", "searched"),
            (f"meancache_K{k}", "meancache", "searched"),
            (f"{shared_key}_K{k}", shared_label, "shared-fixed"),
        ):
            if key not in tables:
                raise SystemExit(f"{backbone}: schedule_tables has no {key!r} "
                                 f"(present: {sorted(tables)})")
            bits = check_bits(bits_from_cache_steps(tables[key]["cache_steps"]),
                              f"{backbone} {key}")
            if bits.count("1") != k:
                raise SystemExit(f"{key}: {bits.count('1')} cached steps, nominal K={k}")
            objs.append(Obj(label=label, family=family, dataset=None,
                            k_nominal=k, bits=bits))
    return objs


def gate_objects(cells: dict[tuple[str, str, int], dict[int, collections.Counter]],
                 backbone: str, datasets: list[str]) -> list[Obj]:
    """One modal-path object per (gate, dataset, K), with the P0 realised-K
    cross-check re-applied (plan lines 333-338)."""
    objs: list[Obj] = []
    documented = DOCUMENTED_K[backbone]
    duplicates = DUPLICATE_CONFIG[backbone]
    for k in KS:
        for gate in GATES:
            for dataset in datasets:
                key = (gate, dataset, k)
                if key not in cells:
                    raise SystemExit(
                        f"{backbone}: no cell rows for {gate} / {dataset} / K{k} in the "
                        f"merged table. Every gate is supposed to have three seed-cells "
                        f"per (dataset, budget); found {sorted(cells)}")
                modal = modal_path(cells[key])
                if modal["n_seed_cells"] != SEED_CELLS_PER_CONFIG:
                    print(f"[WARN] {backbone} {gate} / {dataset} / K{k}: pooled "
                          f"{modal['n_seed_cells']} seed-cells, not "
                          f"{SEED_CELLS_PER_CONFIG}. The modal path is still the pooled "
                          f"top-1, but it rests on fewer seeds than the matrix defines; "
                          f"n_seed_cells records it.")
                if modal["modal_tie"]:
                    print(f"[WARN] {backbone} {gate} / {dataset} / K{k}: "
                          f"{modal['n_tied_at_top']} paths tie at {modal['modal_count']} "
                          f"rows; the modal path is ambiguous and was broken "
                          f"lexicographically (modal_tie records it).")
                bits = check_bits(modal["bits"], f"{backbone} {gate} {dataset} K{k} modal",
                                  allow_tail_gap=True)
                realized = bits.count("1")
                expected = documented.get((gate, k))
                mismatch = expected is not None and realized != expected
                if mismatch:
                    print(
                        f"[WARN] {backbone} {gate} K{k} on {dataset}: modal path realises "
                        f"{realized} cached steps, docs/video_full_results_report_zh.md section 5.1 "
                        f"records this CONFIGURATION as a structural {expected}-step one. "
                        f"P0 checked that against the per-cell `n_cached` distribution; "
                        f"this is one path out of that distribution and the gate is "
                        f"per-video K+-2 (plan line 335), so the two can differ. The row "
                        f"is filed with k_documented={expected} and "
                        f"k_documented_mismatch=true — check it before quoting the row.")
                group, counted = duplicates.get((gate, k), (None, None))
                objs.append(Obj(
                    label=gate, family="gate modal", dataset=dataset, k_nominal=k,
                    bits=bits, duplicate_config=group is not None,
                    duplicate_group=group, counted_instance=counted,
                    k_deviation_documented=expected is not None, k_documented=expected,
                    k_documented_mismatch=mismatch, modal=modal))
    return objs


def control_objects(rho2: np.ndarray, sigmas: np.ndarray) -> list[Obj]:
    """Positive (DP on this rho2) and negative (uniform) controls, per K."""
    objs: list[Obj] = []
    for k in KS:
        bits, value = dp_bits(rho2, sigmas, k)
        objs.append(Obj(label="dp_optimum_e10", family="dp-positive", dataset=None,
                        k_nominal=k, bits=check_bits(bits, f"dp K{k}"), dp_cost=value))
        objs.append(Obj(label="uniform", family="uniform-null", dataset=None, k_nominal=k,
                        bits=check_bits(uniform_bits(k), f"uniform K{k}")))
    return objs


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure_object(obj: Obj, rho2: np.ndarray, sigmas: np.ndarray,
                   rho2_dataset: str) -> dict[str, Any]:
    """One table row. `MIN_GAPS` is applied HERE.

    `density_form_test.measure` computes a Spearman from 3 gaps up
    (`:119`) and only the image-side `main` applies `MIN_GAPS = 4`
    (`:175-177`); doing the same here is what keeps a 3-gap row from leaking a
    rank statistic into the archived table. Plan lines 339-341: below the
    threshold only the gap count is reported.
    """
    m = measure(obj.bits, rho2, sigmas)
    enough = int(m["n_gaps"]) >= MIN_GAPS
    realized = obj.bits.count("1")
    return {
        "schedule": obj.label,
        "family": obj.family,
        "dataset": obj.dataset,
        "rho2_dataset": rho2_dataset,
        "matched_dataset": None if obj.dataset is None else obj.dataset == rho2_dataset,
        "K_nominal": obj.k_nominal,
        "k_realized": realized,
        "k_realized_minus_nominal": realized - obj.k_nominal,
        "duplicate_config": obj.duplicate_config,
        "duplicate_group": obj.duplicate_group,
        "counted_instance": obj.counted_instance,
        "k_deviation_documented": obj.k_deviation_documented,
        "k_documented": obj.k_documented,
        "k_documented_mismatch": obj.k_documented_mismatch,
        "n_gaps": int(m["n_gaps"]),
        "tail_gap_open": has_tail_gap(obj.bits),
        "spearman": _finite(m["rho_density"]) if enough else None,
        "p": _finite(m["p"]) if enough else None,
        "cost_cv": _finite(m["cost_cv"]) if enough else None,
        "modal_count": None if obj.modal is None else obj.modal["modal_count"],
        "modal_share": None if obj.modal is None else obj.modal["modal_share"],
        "modal_tie": None if obj.modal is None else obj.modal["modal_tie"],
        "n_distinct_paths": None if obj.modal is None else obj.modal["n_distinct_paths"],
        "n_rows": None if obj.modal is None else obj.modal["n_rows"],
        "n_seed_cells": None if obj.modal is None else obj.modal["n_seed_cells"],
        "bits": obj.bits,
        "dp_cost": obj.dp_cost,
        "below_min_gaps": not enough,
        "tail_gap_note": TAIL_GAP_NOTE if has_tail_gap(obj.bits) else None,
        "local_rho2": [float(v) for v in m["local"]],
        "gap_sigma_span": [float(v) for v in m["span"]],
    }


# ---------------------------------------------------------------------------
# permutation null over random schedules
# ---------------------------------------------------------------------------

ALLOWED_STEPS_RULE = (
    "a draw may cache any step from 1 up to (but not including) the last step, "
    "i.e. range(1, NUM_STEPS - 1). The bound is DERIVED from the objects under test, "
    "not assumed: every frozen table and every gate modal path here leaves step 0 and "
    "step NUM_STEPS-1 full, so those two are structurally unavailable to a schedule and "
    "the interior is what a schedule chooses among. The interior is taken WHOLE even "
    "when some interior step happens to be cached by no object (wan21 never caches step "
    "1 — a gate warm-up property, not a constraint on schedules); restricting the null "
    "to the steps the real objects happened to use would put the objects' own design "
    "into the null and is exactly what the null is there to avoid. If a gate modal path "
    "ever does cache the last step (tail_gap_open), the last step joins the allowed set "
    "so the null covers the same schedule space the rows live in.")

NULL_SCORING_NOTE = (
    "each draw is scored by `density_form_test.measure`, i.e. the same `gaps_of` + "
    "plan sigma-span rule + `scipy.stats.spearmanr` call every real row goes through, "
    "and a draw with fewer than MIN_GAPS gaps is dropped exactly as a real row's rank "
    "statistics would be (n_draws_discarded).")

PERCENTILE_NOTE = (
    "midrank empirical percentile: 100 * (#draws below + 0.5 * #draws equal) / "
    "n_draws_scored, against the null for THIS row's rho2 dataset and THIS row's "
    "REALISED cache count. A low spearman percentile means the row is more negative "
    "than most random schedules of the same size; a percentile near 50 means the row's "
    "sign carries no information beyond what any schedule gets for free.")


def allowed_null_steps(objects: list[Obj], *, num_steps: int = NUM_STEPS,
                       ) -> tuple[list[int], dict[str, Any]]:
    """The steps a random schedule may cache, read off the objects under test.

    Returns `(allowed, provenance)`. See `ALLOWED_STEPS_RULE` for the choice; the
    provenance dict records what the data actually showed, so a future run whose
    objects break the assumption is visible rather than silently re-based.
    """
    if not objects:
        raise SystemExit("the permutation null needs the objects under test to derive "
                         "which steps a schedule may cache")
    used: set[int] = set()
    for obj in objects:
        used |= {n for n, c in enumerate(obj.bits) if c == "1"}
    first_step_cached = 0 in used
    last_step_cached = (num_steps - 1) in used
    if first_step_cached:                      # check_bits refuses this upstream
        raise SystemExit(
            "an object under test caches step 0, which the sigma-span anchor rule "
            "forbids (check_bits). The permutation null derives its step set from the "
            "objects, so it cannot be built on top of a schedule that broke that rule.")
    allowed = list(range(1, num_steps if last_step_cached else num_steps - 1))
    interior_unused = sorted(s for s in allowed if s not in used)
    provenance = {
        "allowed_steps": allowed,
        "n_allowed_steps": len(allowed),
        "rule": ALLOWED_STEPS_RULE,
        "num_steps": int(num_steps),
        "objects_cache_step_0": bool(first_step_cached),
        "objects_cache_last_step": bool(last_step_cached),
        "last_step_allowed": bool(last_step_cached),
        "interior_steps_cached_by_no_object": interior_unused,
        "interior_unused_note":
            "these interior steps are in the allowed set even though no object under "
            "test caches them; they are an observation about the objects, not a "
            "restriction imposed on the null",
    }
    return allowed, provenance


def null_rng(seed: int, dataset: str, k: int) -> np.random.Generator:
    """The generator for one (dataset, K) null.

    Seeded from `(seed, K, sha256(dataset))` rather than drawn off a single stream,
    so a null does not change when the SET of K values changes — that set comes from
    the gates' realised counts and can move between backbones and between runs.
    """
    tag = int.from_bytes(hashlib.sha256(dataset.encode("utf-8")).digest()[:4], "big")
    return np.random.default_rng([int(seed), int(k), tag])


def draw_null_bits(rng: np.random.Generator, allowed: list[int], k: int, *,
                   num_steps: int = NUM_STEPS) -> str:
    """One random schedule: `k` distinct steps drawn from `allowed`, as 50 bits."""
    if k > len(allowed):
        raise SystemExit(f"cannot draw a {k}-step schedule from {len(allowed)} allowed "
                         f"steps ({allowed[0]}..{allowed[-1]})")
    if k < 0:
        raise SystemExit(f"cannot draw a schedule with k={k}")
    picked = set(int(v) for v in rng.choice(np.asarray(allowed, dtype=np.int64),
                                            size=k, replace=False))
    return "".join("1" if n in picked else "0" for n in range(num_steps))


def _midrank_percentile(value: float, sample: np.ndarray) -> float | None:
    """Where `value` sits inside `sample`, in percent (`PERCENTILE_NOTE`)."""
    if sample.size == 0 or not math.isfinite(value):
        return None
    below = float(np.count_nonzero(sample < value))
    equal = float(np.count_nonzero(sample == value))
    return 100.0 * (below + 0.5 * equal) / float(sample.size)


def schedule_null(rho: np.ndarray, sigmas: np.ndarray, allowed: list[int], k: int, *,
                  dataset: str, n_draws: int, seed: int) -> dict[str, Any]:
    """`n_draws` random `k`-step schedules, scored like every real row."""
    rng = null_rng(seed, dataset, k)
    spearman: list[float] = []
    cost_cv: list[float] = []
    gaps: list[int] = []
    n_discarded = n_spearman_nonfinite = n_cost_cv_nonfinite = 0
    for _ in range(n_draws):
        bits = draw_null_bits(rng, allowed, k)
        m = measure(bits, rho, sigmas)
        if int(m["n_gaps"]) < MIN_GAPS:
            n_discarded += 1
            continue
        gaps.append(int(m["n_gaps"]))
        s, c = float(m["rho_density"]), float(m["cost_cv"])
        if math.isfinite(s):
            spearman.append(s)
        else:
            n_spearman_nonfinite += 1
        if math.isfinite(c):
            cost_cv.append(c)
        else:
            n_cost_cv_nonfinite += 1
    s_arr = np.asarray(spearman, dtype=np.float64)
    c_arr = np.asarray(cost_cv, dtype=np.float64)
    n_scored = n_draws - n_discarded

    def _q(arr: np.ndarray, q: float) -> float | None:
        return float(np.percentile(arr, q)) if arr.size else None

    return {
        "K": int(k),
        "dataset": dataset,
        "n_draws": int(n_draws),
        "n_draws_scored": int(n_scored),
        "n_draws_discarded": int(n_discarded),
        "discard_rule": f"fewer than MIN_GAPS = {MIN_GAPS} gaps",
        "n_spearman_nonfinite": int(n_spearman_nonfinite),
        "n_cost_cv_nonfinite": int(n_cost_cv_nonfinite),
        "n_gaps_median": float(np.median(gaps)) if gaps else None,
        "null_spearman_median": _q(s_arr, 50.0),
        "null_spearman_p05": _q(s_arr, NULL_BAND[0]),
        "null_spearman_p95": _q(s_arr, NULL_BAND[1]),
        f"null_frac_below_{NULL_FRAC_THRESHOLD}": (
            float(np.count_nonzero(s_arr < NULL_FRAC_THRESHOLD)) / s_arr.size
            if s_arr.size else None),
        "cost_cv_null_median": _q(c_arr, 50.0),
        "cost_cv_null_p05": _q(c_arr, NULL_BAND[0]),
        "cost_cv_null_p95": _q(c_arr, NULL_BAND[1]),
        "spearman_draws": sorted(float(v) for v in s_arr),
        "cost_cv_draws": sorted(float(v) for v in c_arr),
    }


def build_nulls(rows: list[dict[str, Any]], rho2: dict[str, np.ndarray],
                sigmas: np.ndarray, allowed: list[int], *, order: list[str],
                n_draws: int, seed: int, verbose: bool = True,
                ) -> dict[str, dict[int, dict[str, Any]]]:
    """One null per (rho2 dataset, K), for every K a row needs plus the three
    nominal budgets, so the summary table always has its K29/K37/K41 rows."""
    wanted: dict[str, set[int]] = {ds: set(KS) for ds in order}
    for row in rows:
        wanted.setdefault(row["rho2_dataset"], set(KS)).add(int(row["k_realized"]))
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for ds in order:
        out[ds] = {}
        for k in sorted(wanted[ds]):
            out[ds][k] = schedule_null(rho2[ds], sigmas, allowed, k, dataset=ds,
                                       n_draws=n_draws, seed=seed)
        if verbose:
            for k in sorted(out[ds]):
                blk = out[ds][k]
                med = blk["null_spearman_median"]
                lo, hi = blk["null_spearman_p05"], blk["null_spearman_p95"]
                frac = blk[f"null_frac_below_{NULL_FRAC_THRESHOLD}"]
                if med is None:
                    print(f"  null [{ds}] K={k:3d}: no draw cleared MIN_GAPS "
                          f"({blk['n_draws_discarded']}/{blk['n_draws']} discarded)")
                    continue
                print(f"  null [{ds}] K={k:3d}: spearman median {med:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}], frac < {NULL_FRAC_THRESHOLD} "
                      f"{frac:.2f}; cost cv median "
                      f"{blk['cost_cv_null_median']:.3f}; scored "
                      f"{blk['n_draws_scored']}/{blk['n_draws']}")
    return out


def null_report_block(nulls: dict[str, dict[int, dict[str, Any]]],
                      allowed_meta: dict[str, Any], *, order: list[str],
                      n_draws: int, seed: int) -> dict[str, Any]:
    """The `schedule_null` block of the report — the one thing the markdown page's
    null section reads, so it is built here rather than inline in `main`."""
    if n_draws <= 0:
        return {"ran": False, "skipped_because": "--null_draws 0", "n_draws": 0,
                "seed": None, "seed_default": DEFAULT_NULL_SEED,
                "min_gaps": int(MIN_GAPS), "allowed_steps": allowed_meta,
                "saturation_finding": SATURATION_CAVEAT, "by_dataset": {}}
    return {
        "ran": True,
        "n_draws": int(n_draws),
        "seed": int(seed),
        "seed_default": DEFAULT_NULL_SEED,
        "seed_recipe": "np.random.default_rng([seed, K, sha256(dataset)[:4]]) per "
                       "(dataset, K), so a null does not move when the SET of K values "
                       "changes — that set comes from the gates' realised counts",
        "min_gaps": int(MIN_GAPS),
        "scoring": NULL_SCORING_NOTE,
        "percentile_convention": PERCENTILE_NOTE,
        "band": {"low_percentile": NULL_BAND[0], "high_percentile": NULL_BAND[1],
                 "outside_flag": "*_outside_null_90 is true when the row's statistic "
                                 "falls outside [p05, p95] of its null"},
        "k_conditioning": "each row is compared with random schedules of its own "
                          "REALISED cache count (null_k = k_realized); the three nominal "
                          "budgets are always drawn as well",
        "allowed_steps": allowed_meta,
        "saturation_finding": SATURATION_CAVEAT,
        "by_dataset": {ds: {str(k): blk for k, blk in sorted(nulls[ds].items())}
                       for ds in order},
    }


def apply_null(row: dict[str, Any], nulls: dict[str, dict[int, dict[str, Any]]]) -> None:
    """Attach this row's percentiles, IN PLACE, from the null for its own rho2
    dataset and its own realised cache count."""
    blk = nulls.get(row["rho2_dataset"], {}).get(int(row["k_realized"]))
    row["null_k"] = None if blk is None else int(blk["K"])
    row["null_dataset"] = None if blk is None else blk["dataset"]
    row["null_draws_scored"] = None if blk is None else int(blk["n_draws_scored"])
    # (row statistic, draws in the null block, its p05, its p95)
    stats = (("spearman", "spearman_draws", "null_spearman_p05", "null_spearman_p95"),
             ("cost_cv", "cost_cv_draws", "cost_cv_null_p05", "cost_cv_null_p95"))
    for stat, draws_key, p05_key, p95_key in stats:
        value = row.get(stat)
        draws = None if blk is None else blk[draws_key]
        if blk is None or value is None or not draws:
            row[f"{stat}_null_percentile"] = None
            row[f"{stat}_outside_null_90"] = None
            continue
        sample = np.asarray(draws, dtype=np.float64)
        p05, p95 = blk[p05_key], blk[p95_key]
        row[f"{stat}_null_percentile"] = _midrank_percentile(float(value), sample)
        row[f"{stat}_outside_null_90"] = (
            None if p05 is None or p95 is None
            else bool(float(value) < p05 or float(value) > p95))


# ---------------------------------------------------------------------------
# T3 curvature cross-check
# ---------------------------------------------------------------------------


def t3_paths(t3_root: Path, n_paths: int) -> tuple[list[Path], dict[str, int]]:
    """Up to `n_paths` stored latent paths, balanced across the block-A streams."""
    base = Path(t3_root)
    if not base.is_dir():
        raise SystemExit(f"{base} does not exist; the curvature cross-check needs T3. "
                         f"Pass --xcheck_paths 0 to skip it.")
    per_stream = {p.name: sorted(p.glob("latents_*.pt"))
                  for p in sorted(base.iterdir()) if p.is_dir()}
    per_stream = {k: v for k, v in per_stream.items() if v}
    if not per_stream:
        raise SystemExit(f"no latents_*.pt under {base}")
    chosen: list[Path] = []
    taken = {k: 0 for k in per_stream}
    while len(chosen) < n_paths:
        progressed = False
        for stream, paths in per_stream.items():
            if len(chosen) >= n_paths:
                break
            if taken[stream] < len(paths):
                chosen.append(paths[taken[stream]])
                taken[stream] += 1
                progressed = True
        if not progressed:
            break
    if len(chosen) < n_paths:
        print(f"[WARN] the curvature cross-check asked for {n_paths} T3 paths and only "
              f"{len(chosen)} exist under {base} ({dict(taken)}). Plan line 344 sets a "
              f"floor of {MIN_XCHECK_PATHS} paths per backbone; the run continues and "
              f"records xcheck_paths_requested next to the realised n_trajectories, but "
              f"the reading is below the specification until the missing streams land.")
    elif len(chosen) < MIN_XCHECK_PATHS:
        print(f"[WARN] the curvature cross-check runs on {len(chosen)} T3 paths, below "
              f"the plan's floor of {MIN_XCHECK_PATHS} per backbone (plan line 344).")
    return chosen, taken


def run_cross_check(rho2: np.ndarray, sigmas: np.ndarray, paths: list[Path],
                    rho2_dataset: str, windows: tuple[int, ...],
                    n_paths_requested: int) -> list[dict[str, Any]]:
    """`build_golden_path_family.cross_check_rho2` on the stored bf16 paths.

    Run against the main-table rho2 only: the expensive part is
    `direct_curvature_profile`, which reads every 133-173 MB tensor in full, so a
    second run for the replication profile would double 8-10 GB of I/O for a pair
    of correlation coefficients. Which profile it used is recorded in the row.

    `cross_check_rho2` re-reads every path once PER WINDOW, so each extra window
    in `--xcheck_windows` costs another full pass over 8-10 GB. The knob exists
    because the readable window is a per-backbone P1 result (bf16: 9 on wan21,
    None on hunyuan_video) and the image-side builder runs `5 9`; the default
    stays at one window so nobody pays for two by accident.
    """
    rows = cross_check_rho2(rho2, paths, sigmas=sigmas, windows=windows,
                            num_steps=NUM_STEPS, margin=XCHECK_MARGIN)
    out: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        for key in ("direct", "floor", "corrected"):
            row[key] = [float(v) for v in np.asarray(row[key], dtype=float)]
        row["rho2_dataset"] = rho2_dataset
        row["margin"] = XCHECK_MARGIN
        row["interior_steps"] = [XCHECK_MARGIN, NUM_STEPS - XCHECK_MARGIN - 1]
        row["n_trajectories_requested"] = int(n_paths_requested)
        row["paths"] = [str(p) for p in paths]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# kink sensitivity (plan line 344-345)
# ---------------------------------------------------------------------------


def read_kink_verdict(path: Path) -> dict[str, Any]:
    """The P4 kink verdict, READ from `step_profiles_<T>.json`, never asserted.

    Plan line 344 makes the sensitivity re-run conditional on section 3.1 finding
    a kink. P4 found none on either backbone, but that has to come out of the P4
    artefact at run time so a future P4 re-run that flips it is visible here.

    The path defaults to the REPO copy (`resources/video_full_trajectory/<T>/`),
    not to `--out_tables`: redirecting the outputs is exactly what an operator is
    told to do when the reuse decision refuses, and it must not silently turn the
    plan's data-derived kink claim into "unknown". `--p4_json` overrides it.
    """
    path = Path(path)
    if not path.is_file():
        print(f"[warn] {path} not found: the section 3.1 kink verdict could not be read, "
              f"so this run cannot say whether the flatten-the-kink sensitivity re-run is "
              f"triggered. Run P4 (step_profiles.py) first if that claim is needed.")
        return {"source": str(path), "available": False, "kink_present": None,
                "sensitivity_rerun": "unknown: P4 step_profiles JSON not found"}
    report = json.loads(path.read_text(encoding="utf-8"))
    per_dataset = {}
    for ds, block in report.get("datasets", {}).items():
        dev = block.get("dev", {})
        per_dataset[ds] = {"kink_present": bool(dev.get("kink_present")),
                           "kinks": dev.get("kinks", [])}
    any_kink = any(v["kink_present"] for v in per_dataset.values())
    if any_kink:
        states = {ds: v["kinks"] for ds, v in per_dataset.items() if v["kink_present"]}
        print("[WARN] P4 now reports a kink in the deviation profile "
              f"({states}). Plan line 344-345 asks for the whole table to be re-run with "
              f"the kink flattened; rerun with --flatten_span LO:HI and compare.")
    return {
        "source": str(path), "available": True, "per_dataset": per_dataset,
        "kink_present": any_kink,
        "kink_rule": next((b.get("dev", {}).get("kink_rule")
                           for b in report.get("datasets", {}).values()), None),
        "sensitivity_rerun": ("TRIGGERED: rerun with --flatten_span" if any_kink
                              else "not triggered: P4 reports no kink on either dataset"),
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def _window_phrase(value: Any) -> str:
    """The P1 `min_readable_multistep_turn_window` entry, in words.

    `None` is P1's way of saying "no multi-step turn window clears the SNR rule
    in this dtype", which printed bare reads as a missing value.
    """
    if value is None:
        return "None — no multi-step turn window clears the P1 SNR rule in this dtype"
    return f"w = {value}"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_tsv(rows: list[dict[str, Any]], path: Path) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, delimiter="\t", fieldnames=TSV_COLUMNS,
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _fmt(row.get(k)) for k in TSV_COLUMNS})
    return atomic_write_text(path, buf.getvalue())


NULL_TABLE_COLUMNS = [
    "rho2_dataset", "K", "n_draws_scored", "n_draws_discarded",
    "null_spearman_median", "null_spearman_p05", "null_spearman_p95",
    f"null_frac_below_{NULL_FRAC_THRESHOLD}",
    "cost_cv_null_median", "cost_cv_null_p05", "cost_cv_null_p95",
]


def null_section(report: dict[str, Any], backbone: str) -> list[str]:
    """The permutation-null section of the markdown page."""
    block = report.get("schedule_null") or {}
    head = ["## The density Spearman is saturated: read it against a permutation null", ""]
    if not block.get("ran"):
        return head + [
            f"Not drawn in this pass ({block.get('skipped_because', 'skipped')}). "
            f"Without it the Spearman column below cannot be read as evidence of "
            f"anything: see the caveat list.", ""]
    steps = block["allowed_steps"]
    step_set_why = (
        "the last step is allowed because a gate modal path caches it"
        if steps["last_step_allowed"] else
        "step 0 and the last step are excluded because no object under test caches them")
    lines = head + [
        f"The density correlation the plan specifies (plan lines 320-327) comes out "
        f"strongly negative for EVERY object on {backbone}, the uniform negative control "
        f"included, so its sign cannot separate a designed schedule from an undesigned "
        f"one. The mechanism is a coupling between the rho2 shape and the non-uniform "
        f"sigma grid: rho2 is high early where dsigma is small and low late where dsigma "
        f"is large, so any schedule's early gaps pair a high local rho2 with a short "
        f"span and its late gaps do the opposite. The image side runs the same test "
        f"without saturating (its uniform control gives -0.015 / -0.14 / -0.40 at "
        f"p = 0.96 / 0.74 / 0.60, "
        f"`resources/full_trajectory_analysis/a2_density_table.tsv`), so this is a "
        f"property of the video sigma grids and not of either implementation.",
        "",
        f"{block['n_draws']} random schedules per (rho2 dataset, K), seed "
        f"{block['seed']}: each draw is a random K-subset of steps "
        f"{steps['allowed_steps'][0]}-{steps['allowed_steps'][-1]} "
        f"({steps['n_allowed_steps']} of {steps['num_steps']}; {step_set_why}), "
        f"scored through the same "
        f"`density_form_test.measure` call as every real row, and dropped when it has "
        f"fewer than {block['min_gaps']} gaps.",
        "",
        f"The statistic that DOES discriminate is the gap-cost CV, and it gets the same "
        f"null (`cost_cv_null_*`). The plan's definition of both statistics is unchanged; "
        f"this is a reference distribution for them, not a redefinition.",
        "",
        "| " + " | ".join(NULL_TABLE_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(NULL_TABLE_COLUMNS)) + "|",
    ]
    nominal = set(KS)
    for ds, per_k in block["by_dataset"].items():
        for k_str, blk in sorted(per_k.items(), key=lambda kv: int(kv[0])):
            cells = [ds] + [_fmt(blk.get(c)) or "-" for c in NULL_TABLE_COLUMNS[1:]]
            if int(k_str) not in nominal:
                cells[1] = f"{cells[1]} (realised)"
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Rows marked *(realised)* are the cache counts the dynamic gates actually "
        f"produce; each table row is compared with the null of its OWN realised count "
        f"(`null_k`), so no comparison rides on a budget the schedule does not have.",
        "",
        f"Percentile convention: {block['percentile_convention']}",
        "",
    ]
    return lines


def write_md(rows: list[dict[str, Any]], report: dict[str, Any], path: Path,
             backbone: str) -> Path:
    lines = [f"# rho2 vs the matrix schedule axis — {backbone} (plan section 3.7, P5)", ""]
    src = report["source"]
    lines += [
        f"rho2 from the clean references of `{src['merged']}`, "
        f"population MEAN of per-row-normalised profiles, smoothing window "
        f"{report['rho2_window']} in sigma; "
        + ", ".join(f"{ds}: {n:,} trajectories"
                    for ds, n in sorted(report["rho2_rows"].items())) + ".",
        "",
        f"`'1'` = cache step, `'0'` = real computation. Gap = a maximal run of `'1'`; "
        f"anchor = the full step immediately before it; sigma span = "
        f"`sigma_anchor - sigma_(first full step after the gap)`. Rank statistics are "
        f"reported from {MIN_GAPS} gaps up; below that only the gap count is.",
        "",
    ]
    if report.get("main_rho2_dataset_fallback"):
        lines += [f"**{MAIN_DATASET} is not in this table.** Plan line 343 fixes it as the "
                  f"main-table profile; this run fell back to "
                  f"`{report['main_rho2_dataset']}`, so the block below labelled "
                  f"\"main table\" is NOT the plan's main table.", ""]

    lines += null_section(report, backbone)

    warning = (
        "**Read `spearman_null_percentile`, not the sign of `spearman`.** On this "
        "backbone a negative density Spearman is not evidence that a schedule was "
        "designed: random schedules of the same size get one too (see the permutation "
        "null above). The percentile says where the row sits inside that null; "
        "`spearman_outside_null_90` is true only when it leaves the null's 5-95% band."
        if report.get("schedule_null", {}).get("ran") else
        "**The sign of `spearman` is not evidence of design on this backbone** and this "
        "run drew no permutation null (`--null_draws 0`), so there is no percentile "
        "column to read instead. Re-run with `--null_draws` before quoting these "
        "Spearman values.")

    def _table(subset: list[dict[str, Any]]) -> list[str]:
        block = ["| " + " | ".join(TSV_COLUMNS) + " |",
                 "|" + "|".join(["---"] * len(TSV_COLUMNS)) + "|"]
        for row in subset:
            block.append("| " + " | ".join(_fmt(row.get(c)) or "-" for c in TSV_COLUMNS) + " |")
        return block + [""]

    for rho2_ds in report["rho2_dataset_order"]:
        role = "main table" if rho2_ds == report["main_rho2_dataset"] else "replication"
        here = [r for r in rows if r["rho2_dataset"] == rho2_ds]
        # matched_dataset is None for the schedules that have no dataset of their
        # own (fixed tables, controls) and True for a gate read against the rho2
        # of the dataset it ran on; those are the plan's rows. The off-diagonal
        # gate rows are a cross-dataset check and must not be read as if they
        # were part of the same table.
        on = [r for r in here if r["matched_dataset"] is not False]
        off = [r for r in here if r["matched_dataset"] is False]
        lines += [f"## rho2 from {rho2_ds} ({role})", "", warning, ""] + _table(on)
        if off:
            lines += [f"### cross-dataset gate rows (rho2 from {rho2_ds}, schedule from the "
                      f"other dataset)", "",
                      "Sensitivity only: these gates never ran on this dataset. They are not "
                      "part of the row count the plan specifies and are excluded from F4.",
                      ""] + _table(off)
    xc = report.get("curvature_cross_check")
    floor_line = (
        f"Floor rows, read from the P1 dumps (`{Path(str(report['floor']['source'].get('float32'))).name}` "
        f"/ `{Path(str(report['floor']['source'].get('bfloat16'))).name}`), never transcribed: "
        f"every table row above is a T1 reading and cites **float32**, whose minimum readable "
        f"multi-step turn window is "
        f"{_window_phrase(report['floor']['min_readable_multistep_turn_window']['float32'])}; "
        f"the T3 curvature cross-check is a bf16 reading and cites **bfloat16**, "
        f"{_window_phrase(report['floor']['min_readable_multistep_turn_window']['bfloat16'])}.")
    if xc and xc.get("ran"):
        lines += ["## Curvature cross-check (T3, bf16)", ""]
        for row in xc["rows"]:
            requested = row.get("n_trajectories_requested")
            short = ("" if requested in (None, row["n_trajectories"])
                     else f" (of {requested} requested; that is all that is on disk)")
            lines += [
                f"- window {row['window']}: {row['n_trajectories']} stored paths{short}, "
                f"interior steps {row['interior_steps'][0]}-{row['interior_steps'][1]}, rho2 "
                f"from {row['rho2_dataset']}: Pearson {row['pearson_floor_sub']:.4f} / Spearman "
                f"{row['spearman_floor_sub']:.4f} against the floor-subtracted direct "
                f"measure (raw: {row['pearson_raw']:.4f} / {row['spearman_raw']:.4f}); "
                f"median rho2/direct ratio {row['ratio_median_floor_sub']:.4f}; SNR over the "
                f"interior {row['snr_min']:.2f}-{row['snr_max']:.2f} "
                f"(median {row['snr_median']:.2f})."]
        lines += [""]
        if xc.get("window_below_bf16_floor"):
            lines += [f"The bf16 P1 floor says no window below "
                      f"{xc['p1_floor_bfloat16_min_readable_multistep_turn_window']} is "
                      f"readable in bf16, and every window run here is smaller. Re-run with "
                      f"`--xcheck_windows "
                      f"{xc['p1_floor_bfloat16_min_readable_multistep_turn_window']}` before "
                      f"quoting these coefficients as a curvature reading.", ""]
        lines += [floor_line, ""]
    else:
        lines += ["## Curvature cross-check (T3, bf16)", "",
                  f"not run in this pass ({xc.get('skipped_because') if xc else 'skipped'}).",
                  "", floor_line, ""]
    kink = report["kink_sensitivity"]
    lines += ["## Sensitivity: flatten the section 3.1 kink", "",
              f"{kink['sensitivity_rerun']} (read from `{Path(kink['source']).name}`).", ""]
    lines += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return atomic_write_text(path, "\n".join(lines))


def write_figure(rows: list[dict[str, Any]], report: dict[str, Any], path: Path,
                 backbone: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_ds = report["main_rho2_dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    steps = np.arange(NUM_STEPS)
    for ds in report["rho2_dataset_order"]:
        rho = np.asarray(report["rho2"][ds], dtype=float)
        axes[0].plot(steps, rho, lw=1.6 if ds == main_ds else 1.1,
                     ls="-" if ds == main_ds else "--",
                     color="black" if ds == main_ds else "#888888",
                     label=f"{ds} ({report['rho2_rows'][ds]:,} refs)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("solver step n (0..49)")
    axes[0].set_ylabel(r"$\rho_2$  (derivatives taken in $\sigma$)")
    axes[0].set_title(r"$\rho_2$ risk profile (reuse order)" "\n"
                      f"{backbone}: population mean, window {report['rho2_window']}; "
                      f"steps 0-2 and 48-49 constant", fontsize=10)
    # the bands COVER steps 0-2 and 48-49, so they run half a step past the end
    # tick; shading 48..49 would leave step 49 half outside its own band
    for lo, hi in END_CLAMP_BANDS["constant_steps"]:
        axes[0].axvspan(lo - 0.5, hi + 0.5, color="#cccccc", alpha=0.35)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for ax, k in zip(axes[1:], (29, 41)):
        seen: set[str] = set()
        for row in rows:
            if row["rho2_dataset"] != main_ds or row["K_nominal"] != k:
                continue
            if row["matched_dataset"] is False:
                continue  # matched-dataset diagonal only, so each gate appears once
            fam = row["family"]
            ax.scatter(row["local_rho2"], row["gap_sigma_span"], s=26, alpha=0.85,
                       color=FAMILY_COLOURS[fam], label=fam if fam not in seen else None)
            seen.add(fam)
        ax.set_xscale("log")
        ax.set_xlabel(r"local $\rho_2$ inside the gap")
        ax.set_ylabel(r"gap length $\Delta\sigma$")
        ax.set_title(f"K={k} nominal ({NUM_STEPS - k} really computed):\n"
                     "cost equalisation predicts a downward trend", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{backbone}: density-form test of the geometric cost model - each point is "
                 f"one gap between two full steps of one schedule "
                 f"($\\rho_2$ from {main_ds}; gate rows on their own dataset)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    written = atomic_savefig(fig, path, dpi=140)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def xcheck_reuse(stored_params: dict[str, Any], json_path: Path,
                 n_paths: int, windows: tuple[int, ...]) -> str:
    """`"skip"` or `"recompute"` for the cross-check half of the reuse decision.

    `--xcheck_paths` cannot go through `common.reuse_decision`'s `caps` rule:
    `common._cap` maps 0 to infinity because in the P4 CLIs 0 means "no cap",
    while here 0 means "do not run the cross-check at all" — the opposite. Left
    in `caps`, a `--xcheck_paths 0` smoke run looks like the LARGER run and
    silently overwrites a completed 60-path cross-check, and the real run after
    it is refused. The knob is therefore compared here, where 0 is the smallest
    value it can take, together with the window set (a superset is more work).

    The ordering rule is `common.reuse_decision`'s, applied by hand: the same
    request skips, a request that is larger in BOTH dimensions (at least as many
    paths, at least the same windows) recomputes because the stored run was the
    smoke run, and anything else — a smaller request, or an incomparable window
    set — is refused so the operator has to type `--force` rather than lose the
    bigger reading. `--xcheck_paths 0` is simply the smallest request there is,
    which is the whole point: it can never overwrite a 60-path cross-check, and a
    60-path run after it recomputes instead of being refused.

    The stored realised count matters as much as the stored request: when the
    store asked for 60 and the disk only held 24, asking for 60 again cannot do
    better, so the magnitude compared is the larger of the two and that re-run is
    a skip rather than a permanent recompute.
    """
    stored_req = stored_params.get("xcheck_paths")
    stored_real = stored_params.get("xcheck_paths_realized")
    stored_paths = max((v for v in (stored_req, stored_real) if isinstance(v, int)),
                       default=None)
    if stored_paths is None:
        print(f"[recompute] {json_path.name} records no cross-check parameters")
        return "recompute"
    # a pass with no paths has no windows either, whichever side it is on
    stored_windows = set(stored_params.get("xcheck_windows") or ()) if stored_paths else set()
    want_windows = set(windows) if n_paths else set()
    if (n_paths, want_windows) == (stored_paths, stored_windows):
        return "skip"
    if n_paths >= stored_paths and want_windows >= stored_windows:
        print(f"[recompute] {json_path.name} stored a smaller cross-check "
              f"(paths: stored {stored_paths} vs requested {n_paths}; windows: stored "
              f"{sorted(stored_windows)} vs requested {sorted(want_windows)})")
        return "recompute"
    raise SystemExit(
        f"{json_path} stores a curvature cross-check this run would not simply extend:\n"
        f"  paths: stored {stored_paths} vs requested {n_paths}\n"
        f"  windows: stored {sorted(stored_windows)} vs requested {sorted(want_windows)}\n"
        f"Overwriting it would discard the larger reading. Pass --force to do that "
        f"anyway, or point --out_tables / --out_figs somewhere else.")


def null_reuse(stored_params: dict[str, Any], json_path: Path,
               n_draws: int, seed: int) -> str:
    """`"skip"` or `"recompute"` for the permutation-null half of the reuse decision.

    Same shape and same reason as `xcheck_reuse`, and for the same reason it cannot
    go through `common.reuse_decision`: `--null_draws` is not a "keep at most N" cap,
    so `common._cap`'s 0-means-infinity rule would read a `--null_draws 0` smoke run
    as the LARGER one and let it overwrite a finished 500-draw null. Here 0 is simply
    the smallest request there is.

    The ordering rule is `common.reuse_decision`'s, applied by hand: the same request
    skips, MORE draws at the same seed recompute (the stored run was the smaller one),
    and anything else — fewer draws, or a different seed — is refused so the operator
    types `--force` rather than losing the bigger or the differently-seeded null. A
    seed change is deliberately NOT an extension: it produces a different null from
    the same amount of work, and silently replacing one with the other would make the
    stored percentiles unreproducible.
    """
    if "null_draws" not in stored_params:
        print(f"[recompute] {json_path.name} records no permutation-null parameters")
        return "recompute"
    stored_draws = stored_params.get("null_draws")
    if not isinstance(stored_draws, int):
        print(f"[recompute] {json_path.name} records null_draws={stored_draws!r}")
        return "recompute"
    # a pass with no draws has no seed either, whichever side it is on
    stored_seed = stored_params.get("null_seed") if stored_draws else None
    want_seed = int(seed) if n_draws else None
    if (n_draws, want_seed) == (stored_draws, stored_seed):
        return "skip"
    if n_draws >= stored_draws and stored_seed in (None, want_seed):
        print(f"[recompute] {json_path.name} stored a smaller permutation null "
              f"(draws: stored {stored_draws} vs requested {n_draws})")
        return "recompute"
    raise SystemExit(
        f"{json_path} stores a permutation null this run would not simply extend:\n"
        f"  draws: stored {stored_draws} vs requested {n_draws}\n"
        f"  seed:  stored {stored_seed} vs requested {want_seed}\n"
        f"Overwriting it would discard the larger null, or replace it with a "
        f"differently-seeded one of the same size. Pass --force to do that anyway, or "
        f"point --out_tables / --out_figs somewhere else.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="$DATA (default %(default)s)")
    ap.add_argument("--merged", type=Path, default=None,
                    help="default <data_root>/<backbone>/matrix/trajectory/t1_merged.jsonl")
    ap.add_argument("--index", type=Path, default=None,
                    help="default <data_root>/<backbone>/matrix/trajectory/t1_index.json")
    ap.add_argument("--config", type=Path, default=None,
                    help="default resources/<backbone>/baseline_matrix_config.v1.json")
    ap.add_argument("--t3_root", type=Path, default=None,
                    help="default <data_root>/<backbone>/matrix/references_t3/")
    ap.add_argument("--out_tables", type=Path, default=None,
                    help="default resources/video_full_trajectory/<backbone>/")
    ap.add_argument("--out_figs", type=Path, default=None,
                    help="default docs/figures/video_full_trajectory/<backbone>/")
    ap.add_argument("--p1_floor_dir", type=Path, default=None,
                    help="directory holding p1_floor_{float32,bfloat16}.json "
                         "(default docs/figures/video_full_trajectory/<backbone>/)")
    ap.add_argument("--p4_json", type=Path, default=None,
                    help="the P4 step_profiles JSON the section 3.1 kink verdict is read "
                         "from (default resources/video_full_trajectory/<backbone>/"
                         "step_profiles_<backbone>.json — the REPO copy, so redirecting "
                         "--out_tables does not lose the verdict)")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep at most N references per stream AND N decisions per cell "
                         "(smoke run); unset means no cap. N must be >= 1")
    ap.add_argument("--xcheck_paths", type=int, default=DEFAULT_XCHECK_PATHS,
                    help="T3 paths for the curvature cross-check, balanced across the "
                         "block-A streams; 0 skips it. Each path is a full 133-173 MB "
                         "tensor read, so this is the dominant cost of the job "
                         "(default %(default)s)")
    ap.add_argument("--xcheck_windows", type=int, nargs="+", default=list(DEFAULT_XCHECK_WINDOWS),
                    metavar="W",
                    help="window(s) for the T3 curvature cross-check. Every window is "
                         "another full pass over the stored paths, so the default is one "
                         "window (%(default)s); pass e.g. `5 9` when the bf16 P1 floor says "
                         "the smaller window is not readable on this backbone")
    ap.add_argument("--null_draws", type=int, default=DEFAULT_NULL_DRAWS,
                    help="random schedules drawn per (rho2 dataset, K) for the "
                         "permutation null; 0 skips it. The null is what says whether a "
                         "row's Spearman means anything: the sign of that statistic is "
                         "saturated on both video backbones (see the module docstring), "
                         "so spearman_null_percentile is the column to read "
                         "(default %(default)s)")
    ap.add_argument("--null_seed", type=int, default=DEFAULT_NULL_SEED,
                    help="seed for the permutation null; recorded in the JSON so the "
                         "draws are reproducible, and a change of seed refuses reuse "
                         "instead of quietly replacing a stored null (default "
                         "%(default)s)")
    ap.add_argument("--flatten_span", metavar="LO:HI", default=None,
                    help="diagnostic (plan line 344-345): linearly interpolate rho2 across "
                         "these steps before measuring, to see how much of the table rests "
                         "on one feature. Writes nothing — the archived table is the "
                         "measurement, not the sensitivity check")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the outputs already exist")
    ap.add_argument("--no_figures", action="store_true", help="tables and JSON only")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    T = args.backbone
    traj = args.data_root / T / "matrix" / "trajectory"
    merged = args.merged or traj / "t1_merged.jsonl"
    index_path = args.index or traj / "t1_index.json"
    config_path = args.config or (_PROJECT_ROOT / "resources" / T /
                                  "baseline_matrix_config.v1.json")
    out_tables = args.out_tables or (_PROJECT_ROOT / "resources" / "video_full_trajectory" / T)
    out_figs = args.out_figs or (_PROJECT_ROOT / "docs" / "figures" /
                                 "video_full_trajectory" / T)
    t3_root = args.t3_root or (args.data_root / T / "matrix" / "references_t3")
    p4_json = args.p4_json or (_PROJECT_ROOT / "resources" / "video_full_trajectory" / T /
                               f"step_profiles_{T}.json")
    json_path = out_tables / f"density_form_{T}.json"
    md_path = out_tables / f"density_form_{T}.md"
    tsv_path = out_tables / f"a2_density_table_{T}.tsv"
    fig_path = out_figs / f"f4_density_form_{T}.png"

    # `--limit 0` would keep no rows at all (`load_references` and
    # `scan_cell_actions` both read it as "keep none") while `common._cap` reads
    # it as "no cap"; the two disagree, and the run would die on an empty table.
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 (leave it unset for no cap); 0 would keep "
                         "no references and no decisions at all")
    if args.xcheck_paths < 0:
        raise SystemExit("--xcheck_paths must be >= 0 (0 skips the curvature cross-check)")
    windows = tuple(sorted({int(w) for w in args.xcheck_windows}))
    if any(w < 2 for w in windows):
        raise SystemExit(f"--xcheck_windows must all be >= 2 (got {list(windows)}); the "
                         f"cross-check is a multi-step curvature reading")
    if args.null_draws < 0:
        raise SystemExit("--null_draws must be >= 0 (0 skips the permutation null)")

    if not config_path.is_file():
        raise SystemExit(f"frozen config not found: {config_path}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if int(config.get("num_steps", NUM_STEPS)) != NUM_STEPS:
        raise SystemExit(f"{config_path}: num_steps={config.get('num_steps')}, want {NUM_STEPS}")

    n_xcheck_paths = args.xcheck_paths
    flatten: tuple[int, int] | None = None
    if args.flatten_span:
        if args.flatten_span.count(":") != 1:
            raise SystemExit("--flatten_span takes LO:HI")
        lo, hi = (int(v) for v in args.flatten_span.split(":"))
        if not 0 < lo <= hi < NUM_STEPS - 1:
            raise SystemExit(f"--flatten_span {lo}:{hi} must leave a neighbour on each side "
                             f"of a {NUM_STEPS}-step profile (1 <= LO <= HI <= "
                             f"{NUM_STEPS - 2})")
        flatten = (lo, hi)
        if n_xcheck_paths:
            # the cross-check does not touch the flattened profile's question and
            # this run writes nothing, so it would be 8-10 GB of reads discarded
            print(f"[diagnostic] --flatten_span also switches the T3 curvature "
                  f"cross-check off (was --xcheck_paths {n_xcheck_paths}): it does not "
                  f"depend on the flattened profile and this run writes nothing.")
            n_xcheck_paths = 0
        print(f"[diagnostic] rho2 steps {lo}-{hi} will be flattened; this run is the "
              f"sensitivity check, not the archived table, and writes NOTHING "
              f"(same rule as density_form_test.py:151-156).")

    n_null_draws = args.null_draws
    run_params = {
        "limit": args.limit,
        "xcheck_paths": n_xcheck_paths,
        "xcheck_windows": list(windows) if n_xcheck_paths else [],
        "null_draws": n_null_draws,
        "null_seed": int(args.null_seed) if n_null_draws else None,
        "merged": str(merged),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    # `xcheck_*` is decided by `xcheck_reuse` and `null_*` by `null_reuse`, not by
    # `reuse_decision`: 0 means "do less" for both knobs and "no cap" there (see
    # `xcheck_reuse`'s docstring).
    reuse_params = {k: v for k, v in run_params.items()
                    if k not in ("xcheck_paths", "xcheck_windows",
                                 "null_draws", "null_seed")}

    if flatten is None:
        stored = resolve_reuse(json_path, reuse_params, caps=("limit",),
                               force=args.force, extra_outputs=[md_path, tsv_path])
        if stored is not None:
            stored_params = stored.get("run_params") or {}
            # both knobs are asked, and a "recompute" from one must not stop the
            # other from being asked: each refuses by raising, so a run that would
            # extend the cross-check while shrinking the null has to be refused,
            # not quietly recomputed. Only "skip" from both is a skip.
            verdicts = {xcheck_reuse(stored_params, json_path, n_xcheck_paths, windows),
                        null_reuse(stored_params, json_path, n_null_draws, args.null_seed)}
            verdict = "skip" if verdicts == {"skip"} else "recompute"
            missing = [] if args.no_figures else [
                p for p in (fig_path,) if not output_complete(p)]
            if verdict == "skip" and not missing:
                print(f"[skip] outputs already exist under {out_tables} and {out_figs} "
                      f"for limit={args.limit}, xcheck_paths={n_xcheck_paths}, "
                      f"xcheck_windows={list(windows)}, null_draws={n_null_draws}, "
                      f"null_seed={args.null_seed}; pass --force to recompute")
                return
            if missing:
                print(f"[recompute] {len(missing)} figure(s) missing, empty or truncated: "
                      f"{', '.join(p.name for p in missing)}")

    print(f"=== density_form {T} ===")
    floor = load_p1_floor(T, args.p1_floor_dir)
    index = load_index(index_path)
    sigmas = np.asarray(index["sigma_grids"][0], dtype=np.float64)
    if sigmas.shape[0] != N_STATES:
        raise SystemExit(f"sigma grid has {sigmas.shape[0]} entries, want {N_STATES}")
    refs = load_references(merged, index, limit=args.limit)
    datasets = refs.datasets()

    populations: dict[str, Population] = {}
    profiles: dict[str, Any] = {}
    rho2: dict[str, np.ndarray] = {}
    for ds in datasets:
        populations[ds] = build_population(refs, ds, sigmas, merged=merged, backbone=T)
        profiles[ds] = risk_profiles(populations[ds], rho2_window=RHO2_WINDOW)
        rho = profiles[ds].rho2
        if flatten is not None:
            rho = flatten_span(rho, *flatten)
        rho2[ds] = rho
        print(f"  rho2 [{ds}]: {populations[ds].n_rows:,} references, window "
              f"{profiles[ds].rho2_window}, range {rho.min():.4g}-{rho.max():.4g}")

    cells, cell_meta = scan_cell_actions(merged, limit=args.limit)
    objects = fixed_table_objects(config, T) + gate_objects(cells, T, datasets)

    main_fallback = MAIN_DATASET not in datasets
    if main_fallback:
        print(f"[WARN] plan line 343 fixes {MAIN_DATASET!r} as the main-table rho2 "
              f"(three streams, 2,832 references) with penguin as the replication, but "
              f"this run's references are {datasets}. Falling back to {datasets[0]!r}; "
              f"main_rho2_dataset_fallback records it, and the block the markdown page "
              f"labels 'main table' is NOT the plan's main table.")
    order = ([MAIN_DATASET] + [d for d in datasets if d != MAIN_DATASET]
             if not main_fallback else list(datasets))
    main_ds = order[0]

    rows: list[dict[str, Any]] = []
    all_objects: list[Obj] = list(objects)
    for rho2_ds in order:
        rho = rho2[rho2_ds]
        controls = control_objects(rho, sigmas)
        all_objects += controls
        for obj in objects + controls:
            rows.append(measure_object(obj, rho, sigmas, rho2_ds))

    # the null before the table is printed, so the percentile can be printed with
    # the statistic it qualifies
    allowed, allowed_meta = allowed_null_steps(all_objects)
    nulls: dict[str, dict[int, dict[str, Any]]] = {}
    if n_null_draws > 0:
        print(f"\npermutation null: {n_null_draws} random schedules per (rho2 dataset, K) "
              f"drawn from steps {allowed[0]}-{allowed[-1]} ({len(allowed)} of "
              f"{NUM_STEPS}), seed {args.null_seed}", flush=True)
        nulls = build_nulls(rows, rho2, sigmas, allowed, order=order,
                            n_draws=n_null_draws, seed=args.null_seed)
    null_block = null_report_block(nulls, allowed_meta, order=order,
                                   n_draws=n_null_draws, seed=args.null_seed)
    for row in rows:
        apply_null(row, nulls)

    print(f"\n{'schedule':34s} {'ds':10s} {'K':>3s} {'real':>4s} {'gaps':>5s} "
          f"{'spearman':>9s} {'p':>8s} {'cost cv':>8s} {'sp %ile':>8s} {'cv %ile':>8s}"
          f"  family")
    def _pct(row: dict[str, Any], key: str) -> str:
        v = row.get(key)
        return f"{v:8.1f}" if isinstance(v, float) else f"{'-':>8s}"

    for row in rows:
        if row["rho2_dataset"] != main_ds:
            continue
        if row["below_min_gaps"]:
            tail = "      -- too few gaps to rank --                  "
        elif row["spearman"] is None or row["cost_cv"] is None:
            tail = "      -- statistic not finite --                  "
        else:
            tail = (f"{row['spearman']:9.3f} {row['p']:8.4f} {row['cost_cv']:8.3f} "
                    f"{_pct(row, 'spearman_null_percentile')} "
                    f"{_pct(row, 'cost_cv_null_percentile')}")
        print(f"{row['schedule']:34s} {str(row['dataset'] or '-'):10s} "
              f"{row['K_nominal']:3d} {row['k_realized']:4d} {row['n_gaps']:5d} "
              f"{tail}  {row['family']}")

    bf16_floor_window = floor["min_readable_multistep_turn_window"]["bfloat16"]
    xcheck: dict[str, Any]
    if n_xcheck_paths > 0:
        paths, taken = t3_paths(t3_root, n_xcheck_paths)
        print(f"\ncurvature cross-check: {len(paths)} of {n_xcheck_paths} requested T3 "
              f"paths {taken}, window(s) {list(windows)} (each window re-reads every path "
              f"in full; this is the slow part)", flush=True)
        below_floor = (isinstance(bf16_floor_window, int)
                       and max(windows) < bf16_floor_window)
        if below_floor:
            print(f"[WARN] the P1 bfloat16 floor says no multi-step turn window below "
                  f"{bf16_floor_window} is readable on {T}, and the cross-check is a bf16 "
                  f"reading run at window(s) {list(windows)}. Re-run with "
                  f"`--xcheck_windows {bf16_floor_window}` before quoting these "
                  f"coefficients as a curvature reading; the SNR columns say how far "
                  f"above the quantisation floor this pass actually is.")
        xcheck = {
            "ran": True,
            "windows": list(windows),
            "rows": run_cross_check(rho2[main_ds], sigmas, paths, main_ds, windows,
                                    n_xcheck_paths),
            "paths_per_stream": taken,
            "n_trajectories": len(paths),
            "xcheck_paths_requested": int(n_xcheck_paths),
            "below_plan_floor_60_paths": len(paths) < MIN_XCHECK_PATHS,
            "window_below_bf16_floor": bool(below_floor),
            "dtype_of_T3": "bfloat16 (plan section 2.2: the T3 store is bf16 re-quantised)",
            "p1_floor_bfloat16_min_readable_multistep_turn_window": bf16_floor_window,
            "p1_floor_float32_min_readable_multistep_turn_window":
                floor["min_readable_multistep_turn_window"]["float32"],
            "floor_source": floor["source"],
        }
        for r in xcheck["rows"]:
            print(f"  window {r['window']}: pearson_floor_sub {r['pearson_floor_sub']:.4f}  "
                  f"spearman_floor_sub {r['spearman_floor_sub']:.4f}  "
                  f"(raw {r['pearson_raw']:.4f} / {r['spearman_raw']:.4f})  "
                  f"ratio_median {r['ratio_median_floor_sub']:.4f}  "
                  f"SNR {r['snr_min']:.2f}-{r['snr_max']:.2f}")
    else:
        xcheck = {"ran": False, "skipped_because": "--xcheck_paths 0",
                  "windows": [], "n_trajectories": 0,
                  "xcheck_paths_requested": 0,
                  "p1_floor_bfloat16_min_readable_multistep_turn_window": bf16_floor_window,
                  "p1_floor_float32_min_readable_multistep_turn_window":
                      floor["min_readable_multistep_turn_window"]["float32"],
                  "floor_source": floor["source"]}
    run_params["xcheck_paths_realized"] = int(xcheck["n_trajectories"])

    report: dict[str, Any] = {
        "backbone": T,
        "produced_by": "analysis/video_trajectory/density_form.py",
        "plan_sections": ["3.7"],
        "cache_bit_convention": "'1' = cache step, '0' = real computation "
                                "(density_form_test.gaps_of; merge_video_traj.py:134-140)",
        "sigma_span_definition": "sigma_anchor - sigma_(first full step after the gap), "
                                 "anchor = the full step immediately before the gap "
                                 "(plan lines 324-325)",
        "min_gaps": int(MIN_GAPS),
        "min_gaps_note": "rank statistics reported from MIN_GAPS gaps up; below that only "
                         "the gap count is (plan lines 339-341). density_form_test.measure "
                         "computes a Spearman from 3 gaps, so the gate is applied by this "
                         "caller, as the image-side main does.",
        "rho2_window": RHO2_WINDOW,
        "rho2_end_clamp": END_CLAMP_BANDS,
        "rho2_definition": "hypot(|d2(d_perp/chord)/dsigma2|, |d(spacing/|dsigma|/chord)/dsigma|), "
                           "per-row normalisation before the population MEAN, both terms "
                           "divided by the chord, local-polynomial derivatives in sigma with "
                           "clamped windows (build_golden_path_family.risk_profiles, unchanged)",
        "population_built_by": "streamed load_references, Population constructed at call site; "
                               "risk_profiles unchanged",
        "p4_profiles_not_reusable": "step_profiles_<T>.json stores the MEDIAN dev/chord and a "
                                    "velocity divided by sqrt(d); rho2 needs the MEAN and both "
                                    "terms divided by the chord, so the merged table is "
                                    "re-streamed here",
        "main_rho2_dataset": main_ds,
        "main_rho2_dataset_planned": MAIN_DATASET,
        "main_rho2_dataset_fallback": bool(main_fallback),
        "rho2_dataset_order": order,
        "rho2": {ds: [float(v) for v in rho2[ds]] for ds in order},
        "rho2_components": {ds: {"inplane": [float(v) for v in profiles[ds].inplane2],
                                 "alongchord": [float(v) for v in profiles[ds].alongchord2]}
                            for ds in order},
        "rho2_rows": {ds: populations[ds].n_rows for ds in order},
        "sigmas": [float(v) for v in sigmas],
        "sigmas_mid": [float(v) for v in populations[main_ds].sigmas_mid],
        "deviation_mean_over_chord": {ds: [float(v) for v in populations[ds].deviation]
                                      for ds in order},
        "speed_mean_over_chord": {ds: [float(v) for v in populations[ds].speed]
                                  for ds in order},
        "objects": [
            {"schedule": o.label, "family": o.family, "dataset": o.dataset,
             "K_nominal": o.k_nominal, "bits": o.bits, "k_realized": o.bits.count("1"),
             "duplicate_config": o.duplicate_config,
             "duplicate_group": o.duplicate_group,
             "counted_instance": o.counted_instance,
             "k_deviation_documented": o.k_deviation_documented,
             "k_documented": o.k_documented,
             "k_documented_mismatch": o.k_documented_mismatch,
             "tail_gap_open": has_tail_gap(o.bits),
             "modal": o.modal}
            for o in objects],
        "controls": {
            "positive": {"family": "dp-positive",
                         "constructor": "segment_cost_matrix + j_best_schedules + schedule_bits "
                                        "(build_golden_path_family)",
                         "exponent": 1.0, "max_gap": None, "j_best": 1,
                         "forced_full_steps": [0, NUM_STEPS - 1],
                         "bits": {ds: {str(k): next(r["bits"] for r in rows
                                                    if r["rho2_dataset"] == ds
                                                    and r["family"] == "dp-positive"
                                                    and r["K_nominal"] == k)
                                       for k in KS} for ds in order},
                         "caveat": DP_CONTROL_CAVEAT},
            "negative": {"family": "uniform-null",
                         "constructor": "np.unique(np.round(np.linspace(0, 49, 50-K)))"
                                        " full steps, 0 and 49 forced full",
                         "bits": {str(k): uniform_bits(k) for k in KS}},
        },
        "documented_k_exceptions": {f"{g}_K{k}": v
                                    for (g, k), v in DOCUMENTED_K[T].items()},
        "documented_k_note": "docs/video_full_results_report_zh.md section 5.1 records these as the "
                             "CONFIGURATION's realised step count; the modal string is one "
                             "path out of a per-video K+-2 distribution (plan line 335). A "
                             "disagreement is reported as k_documented_mismatch on the row "
                             "and warned about on stdout, not turned into a hard stop that "
                             "throws the other rows away.",
        "duplicate_config_rows": {f"{g}_K{k}": {"group": group, "counted_instance": counted}
                                  for (g, k), (group, counted)
                                  in sorted(DUPLICATE_CONFIG[T].items())},
        "duplicate_config_note": "HYV SenCache K37 and K41 are one configuration capped at 36 "
                                 "steps: BOTH rows are emitted and BOTH are marked "
                                 "duplicate_config with the same duplicate_group; "
                                 "counted_instance says which one a summary counts, so the "
                                 "test is counted once (plan lines 336-338)",
        "gate_decisions_scan": cell_meta,
        "schedule_null": null_block,
        "rows": rows,
        "curvature_cross_check": xcheck,
        "kink_sensitivity": read_kink_verdict(p4_json),
        "caveats": CAVEATS,
        "floor": floor,
        "floor_rows_cited": {
            "T1 readings (rho2, every table row)": "float32 (the T1 profiles were computed in "
                                                   "flight on float32 rows)",
            "T3 curvature cross-check": "bfloat16 (the T3 store is bf16 re-quantised)",
            "rho2 smoothing window": f">= {RHO2_WINDOW} points is itself the floor-motivated "
                                     f"choice (build_golden_path_family.py:45-56)",
        },
        "source": {**refs.meta, "index": str(index_path), "config": str(config_path),
                   "config_sha256_file": run_params["config_sha256"],
                   "config_sha256_declared": config.get("config_sha256"),
                   "t3_root": str(t3_root), "p4_json": str(p4_json)},
        "run_params": run_params,
    }
    if flatten is not None:
        report["flatten_span"] = list(flatten)

    if flatten is not None:
        print(f"\n[diagnostic] nothing written (--flatten_span). Compare the printed table "
              f"with {tsv_path.name} by hand.")
        return

    out_tables.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, report)
    print(f"\nwrote {json_path}")
    print(f"wrote {write_tsv(rows, tsv_path)}")
    print(f"wrote {write_md(rows, report, md_path, T)}")
    if not args.no_figures:
        print(f"wrote {write_figure(rows, report, fig_path, T)}")


if __name__ == "__main__":
    main()
