#!/usr/bin/env python3
"""Cache bending, T1 layer — plan section 3.9.1 (docs/video_full_trajectory_plan_zh.md, P6).

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
    python analysis/video_trajectory/cached_vs_reference.py --backbone hunyuan_video \\
        --data_root outputs

Object: every cell record `c` of the nine-method matrix against the reference
record `r` of the same (dataset, base_seed, prompt_idx) — the same z_T, the same
sigma grid, one run with cache and one without. 27 cells (9 canonical methods x
K in {29, 37, 41}) x 4,629 pairs = 124,983 cell rows per backbone.

T1 holds SCALARS computed per generation, each of them relative to THAT run's
own chord. Two paths' T1 rows cannot be subtracted into a difference vector, so
this layer compares profile against profile, never state against state. What
that costs is spelled out in `CANNOT_MEASURE` below, which is written into the
JSON, printed at the head of the markdown page and repeated in every figure
caption: the five quantities that need both paths' latents belong to the T3
layer (P7, `latent_paths.py cached`) and are neither computed nor stubbed here.

The eight quantities (plan section 3.9.1's table)
-------------------------------------------------
1. displacement ratio `spacing_c[n]/spacing_r[n]`, raw, plus the
   each-divided-by-own-chord difference;
2. velocity ratio `velocity_norm_c[n]/velocity_norm_r[n]` — see below, it is
   algebraically the same number as (1);
3. deviation difference `d_perp_c[n]/chord_c - d_perp_r[n]/chord_r`;
4. magnitude difference `(magnitude_c[n] - magnitude_r[n])/sqrt(d)`;
5. turn difference `turn_w5_c - turn_w5_r` (w=7 secondary), and the w=1 turn at
   the cache step and its neighbours WHEN the P1 float32 row says w=1 is
   readable — never assumed, read from the floor dump;
6. six per-trajectory scalar differences, aggregated median + IQR;
7. prefix identity before the first cache step k0 — expected bit-identical, and
   measured that way for all but a handful of single anomalous generations, so
   the maximum in that table is an outlier magnitude and NOT a reproducibility
   floor (the floor of a normal pair is exactly 0; `PrefixStat` says so too);
8. event alignment around each cache step, offsets -2..+5, stratified by whether
   step k+j is itself cached.

Everything is aggregated median + IQR (never mean-only) over the pairs of a
group, per index; the group is method x budget x dataset (54 per backbone), and
the plan's "27 rows" is the method x budget pooling, which is emitted as well.

Three readings that are easy to get wrong and are therefore explicit
-------------------------------------------------------------------
* The displacement ratio is NOT a same-state payload-vs-true-compute magnitude
  ratio. `CAV_DISPLACEMENT` says why, and travels with every table and figure;
  on Wan2.1 `CAV_WAN_RIDER` follows it, because the sentence about ||v_hat||*dsigma
  is a HunyuanVideo statement.
* The velocity ratio divides by the same |dsigma_n| on both sides, so it equals
  the displacement ratio exactly on both backbones. It is computed anyway and
  the maximum |disp - vel| over all pairs is reported as a consistency check —
  it is NOT independent evidence, and the JSON says so.
* The prefix-identity check covers ONLY the local quantities. `d_perp`,
  `max_dev_ratio`, `straightness`, `pca_evr` and the `update_*` shares each
  reference their own run's chord or whole path, and the cached run's endpoint
  differs, so they are NOT identical before k0 either. `NOT_PREFIX_IDENTICAL`
  carries that list with its one-line reason, so a non-zero `d_perp` prefix is
  not mistaken for a bug.

Rows that are not read
----------------------
`cells_t3_rand50/` rows carry the same mode/dataset/budget/base_seed as the matrix
cell for the sampled path-layer prompts (`merge_video_traj.py:118-122`), so they are
dropped on the `CELL_PREFIX` guard imported from `density_form` — the same guard
the P5 audit asked for. Here it matters twice: those rows would double-count ten
prompts in every median AND move the gates' modal path. Rows whose
`ref_z_T_match` is not True are excluded and counted per group; `ref_source_dir`
is cross-checked against the key lookup and a disagreement is a hard stop.

Floors are cited, never hardcoded. Every reading in this file is a T1 reading,
so the **float32** P1 row applies throughout (`step_profiles.load_p1_floor`);
the bfloat16 row is carried only to say what a T3 recomputation would cost and
is never used for a P6 verdict.

Cost: TWO streaming passes over `t1_merged.jsonl` — `step_profiles.load_references`
reads the 4,629 reference rows (~15 MB, held in memory), then `stream_cells` reads
the file again and matches each cell row against them. The reference reader is
shared with P4/P5 rather than duplicated here, which costs the second pass over
~2.5 GB of I/O; `source.passes` in the JSON says so. Peak accumulation is ~0.7 GB
per backbone (~165 MB of per-index profile rows plus ~430 MB of event-alignment
samples, all float32); the 1.27 GB table parses at a few ms per row, so budget
tens of minutes on one CPU node. `--limit N` caps the rows per cell for a smoke
run and shrinks both proportionally. This is an sbatch CPU job, never a login node.

Cell rows are reconciled against `t1_index.json`: every `cells/` directory's row
count must equal the index's `n_t1` for that directory, so a short read (a
truncated merge) or a duplicated line cannot be silently averaged.
"""

from __future__ import annotations

import os

# BLAS pools must be capped before numpy loads (plan section 5.2).
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable  # noqa: E402

import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.video_trajectory.common import (  # noqa: E402
    TURN_W1_INDEX_NOTE,
    atomic_savefig,
    atomic_write_json,
    atomic_write_text,
    output_complete,
    resolve_reuse,
)
from analysis.video_trajectory.density_form import (  # noqa: E402
    CELL_PREFIX,
    DOCUMENTED_K,
    DUPLICATE_CONFIG,
    GATES,
    SHARED_METHODS,
    modal_path,
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
POOLED = "__pooled__"          # the dataset slot of a method x budget row

# plan section 2.1: the offline-searched tables and the one table the three
# order-truncation methods share are the fixed-table family; the four gates
# decide per video.
FIXED_METHODS = ("budcache", "meancache") + SHARED_METHODS
METHODS = tuple(sorted(FIXED_METHODS + GATES))
FAMILY = {**{m: "fixed-table" for m in FIXED_METHODS},
          **{m: "dynamic-gate" for m in GATES}}

# plan section 2.4 index axes
W5, W7 = 5, 7
INDEX_CONVENTION = {
    "d_perp / magnitude / sigmas": "state n = 0..50 (0 = z_T, 50 = result)",
    "spacing / velocity_norm": "solver step n = 0..49 (spacing[n] = ||Z[n+1] - Z[n]||)",
    "turn_angle_deg / second_diff_norm": TURN_W1_INDEX_NOTE,
    "turn_angle_w5_deg / turn_angle_w7_deg":
        f"window centre c = {W5}..{N_STATES - 1 - W5} / {W7}..{N_STATES - 1 - W7}; "
        f"array index i maps to centre c = i + w, so the array index is NOT the centre",
    "actions": "50 characters, '1' = cache, '0' = real computation "
               "(merge_video_traj.py:134-140); k0 = index of the first '1'",
}

# every T1 array this script reads, with the length it must have
PROFILE_LEN = {
    "d_perp": N_STATES, "spacing": NUM_STEPS, "magnitude": N_STATES,
    "velocity_norm": NUM_STEPS, "turn_angle_deg": 49,
    "turn_angle_w5_deg": 41, "turn_angle_w7_deg": 37, "second_diff_norm": 49,
}

# one row of the per-pair profile buffer, in order (name, length, axis label)
ROW_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("disp_ratio", NUM_STEPS, "solver step n = 0..49"),
    ("disp_chordnorm_diff", NUM_STEPS, "solver step n = 0..49"),
    ("vel_ratio", NUM_STEPS, "solver step n = 0..49"),
    ("dev_diff", N_STATES, "state n = 0..50"),
    ("mag_diff", N_STATES, "state n = 0..50"),
    ("turn_w5_diff", 41, f"window centre c = {W5}..{N_STATES - 1 - W5}"),
    ("turn_w7_diff", 37, f"window centre c = {W7}..{N_STATES - 1 - W7}"),
)
ROW_WIDTH = sum(n for _, n, _ in ROW_LAYOUT)

SCALAR_COLUMNS: tuple[str, ...] = (
    "chord_ratio", "path_ratio", "straightness_diff", "max_dev_ratio_diff",
    "evr_first2_diff", "update_chord_share_diff",
    "dev_peak_state_cached", "dev_peak_value_cached",
    "dev_peak_state_reference", "dev_peak_value_reference", "dev_peak_state_shift",
    "disp_ratio_at_k0", "k_realized", "k0", "n_nan_disp_ratio",
)
SCALAR_WIDTH = len(SCALAR_COLUMNS)
# the six the plan asks for, in its order; the rest are readings of (3) and of
# the schedule that ride along in the same buffer
PLAN_SCALARS: tuple[str, ...] = SCALAR_COLUMNS[:6]
PEAK_SCALARS: tuple[str, ...] = SCALAR_COLUMNS[6:11]
# `disp_ratio_at_k0` is the ONE same-state reading of the ratio (CAV_DISPLACEMENT):
# it is read at EACH ROW's own first cache step, never at the group's median k0
EXTRA_SCALARS: tuple[str, ...] = PEAK_SCALARS + ("disp_ratio_at_k0",)

EVENT_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4, 5)
EVENT_STRATA: tuple[str, ...] = ("cache", "full")
EVENT_QUANTITIES: tuple[str, ...] = ("disp_ratio", "dev_diff", "dev_diff_state_shifted")
TURN_W1_OFFSETS: tuple[int, ...] = (-1, 0, 1)

# ---------------------------------------------------------------------------
# the two text blocks that must travel with every number this script emits
# ---------------------------------------------------------------------------

CAV_DISPLACEMENT = (
    "Displacement ratio `spacing_c[n]/spacing_r[n]` is NOT a same-state "
    "payload-vs-true-compute magnitude ratio. On HunyuanVideo the numerator is "
    "||v_hat(z_n^c)||*dsigma and the denominator ||v_theta(z_n^r)||*dsigma, and the two "
    "are on the SAME state only at the first cache step k0. After k0, z_n^c != z_n^r, so "
    "the ratio is 'the cached run's own step on its own state' over 'the reference's step "
    "on the reference state', and it INCLUDES the cached run's own genuinely-computed "
    "steps (plan section 3.9.1).")
CAV_WAN_RIDER = (
    "Wan2.1 rider (plan section 2.2): under the second-order UniPC multistep sampler "
    "z_{n+1} depends on the current AND the previous model output, so `velocity_norm` is a "
    "sigma-domain path speed, not one model evaluation. No 'model output' reading of "
    "either ratio is available on Wan, and the sentence above about ||v_hat||*dsigma is a "
    "HunyuanVideo statement that does NOT transfer.")
CAV_VELOCITY = (
    "The velocity ratio divides by the same |dsigma_n| on both sides of the pair, so it "
    "is algebraically the SAME number as the displacement ratio on both backbones. It is "
    "reported as a consistency check (`velocity_vs_displacement_ratio_max_abs_diff`), "
    "never as independent evidence.")
CAV_ENDPOINTS = (
    "States 0 and 50 of the deviation difference are zero on both sides by construction "
    "up to floating-point rounding (d_perp is measured from the chord through those two "
    "states, so the arithmetic leaves a ~1e-17 residue rather than an exact 0), and they "
    "are excluded from every peak search.")
CAV_FLOOR = (
    "The displacement ratio's own noise near step 0 is set by the float32 P1 row, not by "
    "the bfloat16 one: steps 0-8 are only about 5-7 long, so a small absolute rounding is "
    "a large relative one. `spacing_rel_bias` (float32) is printed next to the table. "
    "Divisions by an exactly zero denominator become NaN and are counted, never dropped "
    "silently.")
CAV_OVERLAP = (
    "Event-alignment windows overlap when cache runs are short: one solver step can enter "
    "several (k, j) samples, so the samples inside a (j, stratum) bucket are NOT "
    "independent. `n_samples` is reported per bucket and no interval is put on the median.")
CAV_POOLING = (
    "The pooled (method x budget) rows are PAIR-weighted across datasets, not a mean of "
    "the two dataset medians: a dataset with more prompts pulls the pooled median towards "
    "its own value. On the full matrix that is 3 x 599 penguin599 pairs against 3 x 944 "
    "vbench944 pairs, i.e. about 61 % vbench944; the actual split of THIS run is in each "
    "pooled row's `pooling.n_pairs_per_dataset`, next to the per-dataset min-max of every "
    "scalar median. The per-dataset rows are the unweighted readings.")

CAVEATS: tuple[str, ...] = (CAV_DISPLACEMENT, CAV_WAN_RIDER, CAV_VELOCITY,
                            CAV_ENDPOINTS, CAV_FLOOR, CAV_OVERLAP, CAV_POOLING)


def caveats_for(backbone: str) -> list[str]:
    """The caveats that are statements ABOUT this backbone. The Wan2.1 sampler
    rider is a Wan statement and is not printed on the HunyuanVideo page."""
    return [c for c in CAVEATS if c is not CAV_WAN_RIDER or backbone == "wan21"]


def ratio_caveats_for(backbone: str) -> list[str]:
    """The caveat(s) that must travel with any displacement/velocity-ratio
    figure. On Wan the HYV sentence is only correct once the rider follows it."""
    return [CAV_DISPLACEMENT] + ([CAV_WAN_RIDER] if backbone == "wan21" else [])

CANNOT_MEASURE: tuple[dict[str, str], ...] = (
    {"quantity": "per-state state difference ||z_n^c - z_n^r|| and its direction",
     "why": "T1 stores per-run scalars only; the two runs' latents are needed"},
    {"quantity": "the endpoint difference ||z_50^c - z_50^r||",
     "why": "the matrix's PSNR is a DECODED quantity, not a latent difference, and cannot "
            "stand in for it"},
    {"quantity": "the angle between the cached chord and the reference chord",
     "why": "T1 stores chord LENGTHS, not chord directions"},
    {"quantity": "the principal angles between the cached bend plane and the reference plane",
     "why": "T1 stores the PCA energy spectrum, not the plane's orientation"},
    {"quantity": "the share of the difference vector along the reference chord / inside the "
                 "reference bend plane / out of plane",
     "why": "there is no difference vector at this layer"},
)
CANNOT_MEASURE_OWNER = (
    "All five need BOTH paths' latents, i.e. the T3 layer = P7 "
    "(`analysis/video_trajectory/latent_paths.py cached`, plan section 3.9.3). This "
    "script does not compute, stub or prepare any of them.")

NOT_PREFIX_IDENTICAL: tuple[dict[str, str], ...] = tuple(
    {"field": f, "why": "references this run's own chord or whole path, and the cached "
                        "run's endpoint differs, so it is not equal before k0 either; its "
                        "prefix difference belongs in the profile tables, not in the "
                        "identity check"}
    for f in ("d_perp", "max_dev_ratio", "straightness", "pca_evr",
              "update_chord_share", "update_in_position_plane", "update_own_evr")
)

# plan section 3.9.1, the prefix-identity table: field -> how many leading
# entries must be bit-equal when the first cache step is k0
PREFIX_FIELDS: tuple[str, ...] = (
    "spacing", "velocity_norm", "magnitude", "turn_angle_deg", "second_diff_norm",
    "turn_angle_w5_deg", "turn_angle_w7_deg",
)
PREFIX_RULE = {
    "spacing": "steps n < k0 (step n moves state n to state n+1)",
    "velocity_norm": "steps n < k0",
    "magnitude": "states n <= k0 (the first state a cache at step k0 writes is k0+1)",
    "turn_angle_deg": "junctions n <= k0-2 (junction n uses states n..n+2)",
    "second_diff_norm": "junctions n <= k0-2",
    "turn_angle_w5_deg": f"centres c with c+{W5} <= k0 (and c >= {W5})",
    "turn_angle_w7_deg": f"centres c with c+{W7} <= k0 (and c >= {W7})",
}

NO_PREFIX_NOTE = ("no comparable prefix at this k0: the rule for `{field}` is "
                  "\"{rule}\", and no row of this group has a k0 large enough to leave "
                  "even one entry, so there is nothing measured here — the empty cells "
                  "are NOT a measured zero")

# a non-zero prefix reading is a per-generation anomaly, so the markdown lists
# every such row by name; the cap keeps a pathological pass from writing a page
# of them, and the TSV always carries the complete list
MAX_NONZERO_PREFIX_ROWS = 40


def prefix_length(field: str, k0: int) -> int:
    """How many leading entries of `field` must be bit-equal for a run whose
    first cache step is `k0`. Clipped to [0, the array's length]."""
    if field in ("spacing", "velocity_norm"):
        n = k0
    elif field == "magnitude":
        # states 0..k0. At k0 = 0 this is state 0 alone, the SHARED z_T: a
        # schedule that caches step 0 has an empty prefix everywhere else, but
        # ||z_T|| is still a comparable (trivially equal) reading.
        n = k0 + 1
    elif field in ("turn_angle_deg", "second_diff_norm"):
        n = k0 - 1
    elif field == "turn_angle_w5_deg":
        n = k0 - 2 * W5 + 1
    elif field == "turn_angle_w7_deg":
        n = k0 - 2 * W7 + 1
    else:
        raise KeyError(field)
    return int(max(0, min(n, PROFILE_LEN[field])))


# ---------------------------------------------------------------------------
# small buffers (object-per-sample overhead would dominate at 36M samples)
# ---------------------------------------------------------------------------


class RowBuffer:
    """Growable (n, width) float32 store, one row appended per pair."""

    __slots__ = ("_buf", "_n", "width")

    def __init__(self, width: int, capacity: int = 256) -> None:
        self.width = int(width)
        self._buf = np.empty((capacity, self.width), dtype=np.float32)
        self._n = 0

    def append(self, row: np.ndarray) -> None:
        if self._n == self._buf.shape[0]:
            grown = np.empty((self._buf.shape[0] * 2, self.width), dtype=np.float32)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n] = row
        self._n += 1

    def values(self) -> np.ndarray:
        return self._buf[:self._n]

    def __len__(self) -> int:
        return self._n


class FloatBuffer:
    """Growable 1-D float32 store for the event-alignment samples."""

    __slots__ = ("_buf", "_n")

    def __init__(self, capacity: int = 256) -> None:
        self._buf = np.empty(capacity, dtype=np.float32)
        self._n = 0

    def extend(self, values: np.ndarray) -> None:
        k = int(values.size)
        if k == 0:
            return
        if self._n + k > self._buf.size:
            grown = np.empty(max(self._buf.size * 2, self._n + k), dtype=np.float32)
            grown[:self._n] = self._buf[:self._n]
            self._buf = grown
        self._buf[self._n:self._n + k] = values
        self._n += k

    def values(self) -> np.ndarray:
        return self._buf[:self._n]

    def __len__(self) -> int:
        return self._n


class PrefixStat:
    """Running max |difference| over the identical-by-construction prefix of one
    field, in one group.

    The prefix is identical by construction, so the expected reading is an EXACT
    zero, and that is what almost every pair gives. `n_pairs_exactly_zero` is the
    headline of this statistic and `max_abs` is its tail: a group in which one
    generation ran on different numerics (different kernel, different node, a
    re-run) contributes one non-zero pair and sets `max_abs` on its own, so the
    maximum is an OUTLIER MAGNITUDE and never a typical run-to-run floor. The
    pair that set it is recorded (`max_abs_pair`) so a non-zero row can be named
    rather than generalised."""

    __slots__ = ("max_abs", "max_abs_index", "max_abs_pair", "max_rel", "max_rel_index",
                 "n_pairs", "n_pairs_with_prefix", "n_values", "n_pairs_exact_zero")

    def __init__(self) -> None:
        self.max_abs = 0.0
        self.max_abs_index: int | None = None
        self.max_abs_pair: str | None = None
        self.max_rel = 0.0
        self.max_rel_index: int | None = None
        self.n_pairs = 0
        self.n_pairs_with_prefix = 0
        self.n_values = 0
        self.n_pairs_exact_zero = 0

    def update(self, cached: np.ndarray, reference: np.ndarray,
               pair: str | None = None) -> None:
        self.n_pairs += 1
        if cached.size == 0:
            return
        self.n_pairs_with_prefix += 1
        self.n_values += int(cached.size)
        diff = np.abs(cached - reference)
        i = int(np.argmax(diff))
        # the index is set on the first comparison even when the difference is
        # exactly 0, so "index null" means ONLY "nothing was comparable" and
        # never "the prefix was exactly identical"
        if self.max_abs_index is None or float(diff[i]) > self.max_abs:
            self.max_abs = float(diff[i])
            self.max_abs_index = i
            # only a non-zero maximum names a pair: on an all-zero row every pair
            # is equally "the maximum" and naming one of them would invent an
            # anomaly where there is none
            self.max_abs_pair = pair if float(diff[i]) > 0.0 else None
        if not diff.any():
            self.n_pairs_exact_zero += 1
        nz = np.abs(reference) > 0
        if nz.any():
            rel = diff[nz] / np.abs(reference[nz])
            j = int(np.argmax(rel))
            if self.max_rel_index is None or float(rel[j]) > self.max_rel:
                self.max_rel = float(rel[j])
                self.max_rel_index = int(np.flatnonzero(nz)[j])

    def report(self, field: str) -> dict[str, Any]:
        """`n_values_compared == 0` reports NOTHING, not a zero: at small k0 this
        field has no comparable prefix at all (w=7 needs k0 >= 14, w=5 k0 >= 10),
        and a fabricated 0.0 in the prefix-identity table would be
        indistinguishable from a measured exact zero."""
        empty = self.n_values == 0
        n_nonzero = self.n_pairs_with_prefix - self.n_pairs_exact_zero
        return {
            "field": field,
            "prefix_rule": PREFIX_RULE[field],
            "max_abs_diff": None if empty else self.max_abs,
            "max_abs_diff_at_index": None if empty else self.max_abs_index,
            "max_abs_diff_pair": None if empty else self.max_abs_pair,
            "max_rel_diff": None if empty else self.max_rel,
            "max_rel_diff_at_index": None if empty else self.max_rel_index,
            "n_pairs": self.n_pairs,
            "n_pairs_with_a_prefix": self.n_pairs_with_prefix,
            "n_pairs_exactly_zero": None if empty else self.n_pairs_exact_zero,
            # the count that decides how `max_abs_diff` may be read: 0 = the row
            # is bit-identical throughout, 1 = one anomalous generation set the
            # maximum on its own
            "n_pairs_not_exactly_zero": None if empty else n_nonzero,
            "n_values_compared": self.n_values,
            "no_comparable_prefix": empty,
            "note": (NO_PREFIX_NOTE.format(field=field, rule=PREFIX_RULE[field])
                     if empty else None),
        }


class Group:
    """One method x budget x dataset cell's accumulation."""

    def __init__(self, method: str, budget: int, dataset: str) -> None:
        self.method = method
        self.budget = budget
        self.dataset = dataset
        self.profiles = RowBuffer(ROW_WIDTH)
        self.scalars = RowBuffer(SCALAR_WIDTH)
        self.prefix: dict[str, PrefixStat] = {f: PrefixStat() for f in PREFIX_FIELDS}
        self.events: dict[tuple[int, str, str], FloatBuffer] = {}
        self.turn_w1: dict[int, FloatBuffer] = {o: FloatBuffer() for o in TURN_W1_OFFSETS}
        self.actions: dict[int, collections.Counter] = {}   # base_seed -> Counter(bits)
        self.mode_raw: collections.Counter = collections.Counter()
        self.n_pairs = 0
        self.n_excluded_ref_mismatch = 0
        self.n_caches_step_0 = 0
        self.n_k_not_nominal = 0
        self.n_k_not_documented = 0
        self.n_no_cache_step = 0
        self.no_cache_step_keys: list[str] = []
        self.max_vel_minus_disp = 0.0

    def event_buffer(self, offset: int, stratum: str, quantity: str) -> FloatBuffer:
        key = (offset, stratum, quantity)
        buf = self.events.get(key)
        if buf is None:
            buf = self.events[key] = FloatBuffer()
        return buf


# ---------------------------------------------------------------------------
# per-pair maths
# ---------------------------------------------------------------------------


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num/den with a zero denominator mapped to NaN (counted by the caller)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), np.nan)


def pair_row(cell: dict[str, Any], ref: dict[str, np.ndarray | float]
             ) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """The 330-wide profile row of one pair, the named views into it, and the
    number of NaNs the displacement ratio picked up."""
    chord_c = float(cell["chord_len"])
    chord_r = float(ref["chord_len"])
    if not (chord_c > 0 and chord_r > 0):
        raise SystemExit(f"non-positive chord: cached {chord_c}, reference {chord_r}")
    sqrt_d = math.sqrt(float(cell["d"]))

    spacing_c = np.asarray(cell["spacing"], dtype=np.float64)
    spacing_r = np.asarray(ref["spacing"], dtype=np.float64)
    disp_ratio = _ratio(spacing_c, spacing_r)
    parts = {
        "disp_ratio": disp_ratio,
        "disp_chordnorm_diff": spacing_c / chord_c - spacing_r / chord_r,
        "vel_ratio": _ratio(np.asarray(cell["velocity_norm"], dtype=np.float64),
                            np.asarray(ref["velocity_norm"], dtype=np.float64)),
        "dev_diff": (np.asarray(cell["d_perp"], dtype=np.float64) / chord_c
                     - np.asarray(ref["d_perp"], dtype=np.float64) / chord_r),
        "mag_diff": (np.asarray(cell["magnitude"], dtype=np.float64)
                     - np.asarray(ref["magnitude"], dtype=np.float64)) / sqrt_d,
        "turn_w5_diff": (np.asarray(cell["turn_angle_w5_deg"], dtype=np.float64)
                         - np.asarray(ref["turn_angle_w5_deg"], dtype=np.float64)),
        "turn_w7_diff": (np.asarray(cell["turn_angle_w7_deg"], dtype=np.float64)
                         - np.asarray(ref["turn_angle_w7_deg"], dtype=np.float64)),
    }
    row = np.empty(ROW_WIDTH, dtype=np.float32)
    at = 0
    for name, length, _ in ROW_LAYOUT:
        row[at:at + length] = parts[name]
        at += length
    return row, parts, int(np.isnan(disp_ratio).sum())


def peak_readings(cell: dict[str, Any], ref: dict[str, np.ndarray | float]
                  ) -> tuple[float, float, float, float, float]:
    """Per-side peak of `d_perp / own chord` over states 1..49, and the shift.

    States 0 and 50 are constructed zero on both sides, so they are excluded
    (CAV_ENDPOINTS); the difference profile's own peak is read off the group median
    in `summarise_profile`."""
    dev_c = np.asarray(cell["d_perp"], dtype=np.float64)[1:N_STATES - 1] / float(cell["chord_len"])
    dev_r = np.asarray(ref["d_perp"], dtype=np.float64)[1:N_STATES - 1] / float(ref["chord_len"])
    ic, ir = int(np.argmax(dev_c)), int(np.argmax(dev_r))
    return float(ic + 1), float(dev_c[ic]), float(ir + 1), float(dev_r[ir]), float(ic - ir)


def event_samples(act: np.ndarray, parts: dict[str, np.ndarray], offset: int
                  ) -> dict[str, dict[str, np.ndarray]]:
    """`{stratum: {quantity: values}}` for one offset of one pair.

    Origin = every cache step k of this row's `actions`; the sample sits at
    index k+offset. `disp_ratio` is read at solver step k+offset, `dev_diff` at
    the SAME integer index on the state axis (the literal reading), and
    `dev_diff_state_shifted` at state k+1+offset, because the first state a
    cache at step k writes is state k+1. Both are kept and labelled.

    The stratum is whether step k+offset is itself a cache step, so an offset
    that leaves the 0..49 step axis is dropped for all three quantities.
    """
    ks = np.flatnonzero(act == 1)
    idx = ks + offset
    idx = idx[(idx >= 0) & (idx < NUM_STEPS)]
    out: dict[str, dict[str, np.ndarray]] = {}
    if idx.size == 0:
        return out
    is_cache = act[idx] == 1
    for stratum, mask in (("cache", is_cache), ("full", ~is_cache)):
        sel = idx[mask]
        if sel.size == 0:
            continue
        out[stratum] = {
            "disp_ratio": parts["disp_ratio"][sel],
            "dev_diff": parts["dev_diff"][sel],
            "dev_diff_state_shifted": parts["dev_diff"][sel + 1],
        }
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """A JSON-safe float: non-finite becomes null (`json.dumps` would otherwise
    write a bare NaN, which strict parsers reject)."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _series(values: Iterable[float]) -> list[float | None]:
    return [_finite(v) for v in values]


def summarise_profile(mat: np.ndarray) -> dict[str, Any]:
    """Median + IQR per index over the pairs of a group, with the per-index n.

    Non-finite entries are dropped per index (they are the guarded divisions),
    so the denominator is stated per index rather than per group."""
    if mat.shape[0] == 0:
        width = mat.shape[1] if mat.ndim == 2 else 0
        return {"median": [None] * width, "p25": [None] * width, "p75": [None] * width,
                "n": [0] * width, "n_pairs": 0}
    # +-inf would survive np.nanmedian but is excluded from the `n` below, i.e.
    # the stated denominator would not be the statistic's denominator; map every
    # non-finite entry to NaN first so the two always agree
    finite = np.isfinite(mat)
    clean = np.where(finite, mat, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN index columns
        med = np.nanmedian(clean, axis=0)
        p25 = np.nanpercentile(clean, 25, axis=0)
        p75 = np.nanpercentile(clean, 75, axis=0)
    return {"median": _series(med), "p25": _series(p25), "p75": _series(p75),
            "n": [int(v) for v in finite.sum(axis=0)],
            "n_pairs": int(mat.shape[0])}


def summarise_samples(values: np.ndarray) -> dict[str, Any]:
    """Median + IQR (primary) and the mean (alongside) of one sample bucket.

    The plan says "stratified average" for the event alignment and median + IQR
    everywhere else; both are emitted, median first, so the table is consistent
    with the rest of the file and the plan's word is still answered."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": None, "p25": None, "p75": None, "mean": None,
                "n_samples": int(values.size), "n_finite": 0}
    return {"median": _finite(np.median(finite)),
            "p25": _finite(np.percentile(finite, 25)),
            "p75": _finite(np.percentile(finite, 75)),
            "mean": _finite(np.mean(finite)),
            "n_samples": int(values.size), "n_finite": int(finite.size)}


def profile_views(mat: np.ndarray) -> dict[str, np.ndarray]:
    """Column slices of the packed profile buffer, by name."""
    out: dict[str, np.ndarray] = {}
    at = 0
    for name, length, _ in ROW_LAYOUT:
        out[name] = mat[:, at:at + length]
        at += length
    return out


def peak_of(median: list[float | None], lo: int, hi: int) -> dict[str, Any]:
    """Index and value of the largest |median| over the closed index range."""
    arr = np.asarray([np.nan if v is None else v for v in median], dtype=np.float64)
    window = arr[lo:hi + 1]
    if not np.isfinite(window).any():
        return {"index": None, "value": None, "abs_value": None}
    i = int(np.nanargmax(np.abs(window))) + lo
    return {"index": i, "value": _finite(arr[i]), "abs_value": _finite(abs(arr[i]))}


def merge_prefix(stats: list[PrefixStat], field: str) -> dict[str, Any]:
    """Pool the per-dataset prefix statistics of one field (max of maxima)."""
    merged = PrefixStat()
    for s in stats:
        merged.n_pairs += s.n_pairs
        merged.n_pairs_with_prefix += s.n_pairs_with_prefix
        merged.n_values += s.n_values
        merged.n_pairs_exact_zero += s.n_pairs_exact_zero
        # same rule as PrefixStat.update: a group that compared something keeps an
        # index even at 0, so a null index means "nothing comparable" only
        if s.max_abs_index is not None and (merged.max_abs_index is None
                                            or s.max_abs > merged.max_abs):
            merged.max_abs, merged.max_abs_index = s.max_abs, s.max_abs_index
            merged.max_abs_pair = s.max_abs_pair
        if s.max_rel_index is not None and (merged.max_rel_index is None
                                            or s.max_rel > merged.max_rel):
            merged.max_rel, merged.max_rel_index = s.max_rel, s.max_rel_index
    return merged.report(field)


def k_block(method: str, budget: int, backbone: str, k_realized: np.ndarray,
            n_k_not_nominal: int, n_k_not_documented: int) -> dict[str, Any]:
    """Realised K, not nominal (plan section 3.9.1's last paragraph).

    Fixed-table rows must realise exactly K per video, which P0 verified; a row
    that does not is FLAGGED (`n_rows_k_not_nominal`), never dropped. A gate
    decides per video, so "not nominal" has no meaning for it and the column is
    NULL for gate rows rather than 0 — a 0 there would read as "no video
    deviates from the budget" when in fact every one does.

    Where a number IS documented (`docs/video_full_results_report_zh.md section 5.1`), the
    verdict is per row: `k_documented_mismatch` is true when ANY video misses the
    documented count. The weaker bracket test (min <= documented <= max) is kept
    beside it as `k_documented_bracketed`, because a spread that merely straddles
    the documented number is not agreement. The HYV SenCache K37/K41
    duplicate-configuration marking rides along.
    """
    documented = DOCUMENTED_K.get(backbone, {}).get((method, budget))
    dup = DUPLICATE_CONFIG.get(backbone, {}).get((method, budget))
    fixed = FAMILY[method] == "fixed-table"
    have = k_realized.size > 0
    mean = float(np.mean(k_realized)) if have else None
    lo = int(np.min(k_realized)) if have else None
    hi = int(np.max(k_realized)) if have else None
    mismatch = bracketed = None
    if documented is not None:
        mismatch = (not have) or int(n_k_not_documented) > 0
        bracketed = bool(have and lo <= documented <= hi)
    return {
        "K_nominal": budget,
        "family": FAMILY[method],
        "k_realized_mean": _finite(mean),
        "k_realized_min": lo,
        "k_realized_max": hi,
        "n_rows_k_not_nominal": int(n_k_not_nominal) if fixed else None,
        "n_rows_k_not_nominal_scope":
            "fixed-table rows only; a gate decides per video, so it has no nominal "
            "per-video count and the column is null",
        "k_documented": documented,
        "n_rows_k_not_documented": int(n_k_not_documented) if documented is not None
                                   else None,
        "k_documented_mismatch": mismatch,
        "k_documented_bracketed": bracketed,
        "duplicate_config": dup is not None,
        "duplicate_group": dup[0] if dup else None,
        "counted_instance": dup[1] if dup else None,
    }


def summarise_group(groups: list[Group], method: str, budget: int, dataset: str,
                    backbone: str, floor: dict[str, Any]) -> dict[str, Any]:
    """One output row: a single dataset group, or the pooled method x budget."""
    mats = [g.profiles.values() for g in groups if len(g.profiles)]
    profiles_mat = (np.concatenate(mats, axis=0) if mats
                    else np.zeros((0, ROW_WIDTH), dtype=np.float32))
    scal = [g.scalars.values() for g in groups if len(g.scalars)]
    scalars_mat = (np.concatenate(scal, axis=0) if scal
                   else np.zeros((0, SCALAR_WIDTH), dtype=np.float32))
    views = profile_views(profiles_mat)

    profiles: dict[str, Any] = {}
    for name, _, axis in ROW_LAYOUT:
        profiles[name] = {"axis": axis, **summarise_profile(views[name])}
    # the deviation difference's peak, off the group median, endpoints excluded
    profiles["dev_diff"]["peak_excluding_endpoints"] = peak_of(
        profiles["dev_diff"]["median"], 1, N_STATES - 2)

    scalars = {name: summarise_samples(scalars_mat[:, i])
               for i, name in enumerate(SCALAR_COLUMNS)}

    # plan section 3: state the denominator of every number. A pooled row is
    # PAIR-weighted over datasets of very different size, so it also carries the
    # per-dataset pair counts and the per-dataset spread of each scalar median.
    pooling: dict[str, Any] = {
        "weighting": ("pair-weighted: every pair counts once, so a dataset with more "
                      "prompts pulls the pooled median towards its own value"
                      if dataset == POOLED else "single dataset, no pooling"),
        "n_pairs_per_dataset": {g.dataset: int(g.n_pairs) for g in groups},
    }
    if dataset == POOLED and len(groups) > 1:
        spread: dict[str, Any] = {}
        for name in PLAN_SCALARS + EXTRA_SCALARS:
            i = SCALAR_COLUMNS.index(name)
            per_ds = {g.dataset: summarise_samples(g.scalars.values()[:, i])["median"]
                      for g in groups if len(g.scalars)}
            vals = [v for v in per_ds.values() if v is not None]
            spread[name] = {"per_dataset": per_ds,
                            "min": min(vals) if vals else None,
                            "max": max(vals) if vals else None,
                            "mean_of_dataset_medians":
                                float(np.mean(vals)) if vals else None}
        pooling["scalar_median_across_datasets"] = spread
        pooling["note"] = CAV_POOLING

    events: dict[str, Any] = {}
    for offset in EVENT_OFFSETS:
        per_stratum: dict[str, Any] = {}
        for stratum in EVENT_STRATA:
            per_q: dict[str, Any] = {}
            for quantity in EVENT_QUANTITIES:
                chunks = [g.events[(offset, stratum, quantity)].values()
                          for g in groups if (offset, stratum, quantity) in g.events]
                vals = (np.concatenate(chunks) if chunks
                        else np.zeros(0, dtype=np.float32))
                per_q[quantity] = summarise_samples(vals)
            per_stratum[stratum] = per_q
        events[str(offset)] = per_stratum

    w1_readable = bool(floor.get("turn_w1_readable_float32"))
    turn_w1: dict[str, Any]
    if not w1_readable:
        turn_w1 = {"value": None, "reason": "w=1 not readable at the float32 floor"}
    else:
        turn_w1 = {}
        for offset in TURN_W1_OFFSETS:
            chunks = [g.turn_w1[offset].values() for g in groups if len(g.turn_w1[offset])]
            vals = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            turn_w1[str(offset)] = summarise_samples(vals)

    k_realized = scalars_mat[:, SCALAR_COLUMNS.index("k_realized")].astype(np.int64)
    n_pairs = int(profiles_mat.shape[0])
    counters: dict[int, collections.Counter] = {}
    for g in groups:
        for seed, counter in g.actions.items():
            counters.setdefault(seed, collections.Counter()).update(counter)
    modal = modal_path(counters) if counters else None
    if modal is not None:
        modal["k_realized"] = modal["bits"].count("1")
        # a gate that caches step 0 has an EMPTY prefix set, and one that caches
        # the last step has an open tail gap; both are kept and flagged, mirroring
        # density_form's treatment of the same two shapes
        modal["caches_step_0"] = modal["bits"][0] == "1"
        modal["tail_gap_open"] = modal["bits"][-1] == "1"
        modal["pooled_across_datasets"] = dataset == POOLED
        if dataset == POOLED and len(groups) > 1:
            modal["pooled_across_datasets_note"] = (
                "the gates' thresholds are set per dataset (plan section 2.1), so this "
                "pooled modal string may not be either dataset's own mode; the "
                "per-dataset rows carry those, and the figures mark those")

    return {
        "method": method,
        "family": FAMILY[method],
        "budget": budget,
        "dataset": dataset,
        "mode_raw": sorted({raw for g in groups for raw in g.mode_raw}),
        "n_pairs": n_pairs,
        "pooling": pooling,
        "n_excluded_ref_mismatch": int(sum(g.n_excluded_ref_mismatch for g in groups)),
        "n_rows_caching_step_0": int(sum(g.n_caches_step_0 for g in groups)),
        "n_rows_no_cache_step": int(sum(g.n_no_cache_step for g in groups)),
        "n_rows_no_cache_step_keys": [k for g in groups for k in g.no_cache_step_keys],
        "n_nan_disp_ratio_values": int(
            np.nansum(scalars_mat[:, SCALAR_COLUMNS.index("n_nan_disp_ratio")]))
        if scalars_mat.shape[0] else 0,
        "velocity_vs_displacement_ratio_max_abs_diff":
            _finite(max((g.max_vel_minus_disp for g in groups), default=0.0)),
        "k": k_block(method, budget, backbone, k_realized,
                     sum(g.n_k_not_nominal for g in groups),
                     sum(g.n_k_not_documented for g in groups)),
        "modal_path": modal,
        "scalars": scalars,
        "profiles": profiles,
        "events": events,
        "turn_w1_at_cache_step": turn_w1,
        "prefix_identity": {f: merge_prefix([g.prefix[f] for g in groups], f)
                            for f in PREFIX_FIELDS},
    }


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


def reference_lookup(refs: Any) -> dict[tuple[str, int, int], int]:
    keys = list(zip(refs.cols["dataset"], refs.cols["base_seed"], refs.cols["prompt_idx"]))
    table = {(str(d), int(s), int(p)): i for i, (d, s, p) in enumerate(keys)}
    if len(table) != len(keys):
        raise SystemExit(f"the reference table has {len(keys)} rows but only {len(table)} "
                         f"distinct (dataset, base_seed, prompt_idx) keys; the pairing key "
                         f"is not unique and every median would double-count")
    return table


def reference_row(refs: Any, i: int) -> dict[str, Any]:
    row: dict[str, Any] = {name: refs.cols[name][i] for name in PROFILE_LEN}
    row["chord_len"] = float(refs.cols["chord_len"][i])
    row["path_len"] = float(refs.cols["path_len"][i])
    row["straightness"] = float(refs.cols["straightness"][i])
    row["max_dev_ratio"] = float(refs.cols["max_dev_ratio"][i])
    row["update_chord_share"] = float(refs.cols["update_chord_share"][i])
    row["pca_evr"] = refs.cols["pca_evr"][i]
    row["d"] = float(refs.cols["d"][i])
    row["source_dir"] = str(refs.cols["source_dir"][i])
    return row


def reconcile_cell_dirs(index: dict[str, Any], seen: collections.Counter
                        ) -> dict[str, Any]:
    """Every `cells/` directory's row count against `t1_index.json`'s `n_t1`.

    `step_profiles.load_references` refuses a short reference read; the cell side
    needs the same guard, because a truncated merge would silently shrink a
    median's denominator and a duplicated line would silently double-count one
    prompt in every median. `seen` counts rows BEFORE `--limit` and before the
    `ref_z_T_match` exclusion, so the reconciliation is exact in a smoke run too.
    """
    expected = {k: int(v["n_t1"]) for k, v in (index.get("dirs") or {}).items()
                if v.get("kind") == "cell" and str(k).startswith(CELL_PREFIX)}
    if not expected:
        return {"reconciled": False,
                "reason": f"t1_index.json lists no {CELL_PREFIX} directories with an "
                          f"n_t1 count; nothing to reconcile the cell scan against"}
    bad = {k: {"index_n_t1": n, "rows_seen": int(seen.get(k, 0))}
           for k, n in sorted(expected.items()) if int(seen.get(k, 0)) != n}
    extra = sorted(set(seen) - set(expected))
    if bad or extra:
        raise SystemExit(
            f"the cell scan disagrees with t1_index.json: {len(bad)} of {len(expected)} "
            f"{CELL_PREFIX} directories have a row count the index does not list "
            f"({dict(list(bad.items())[:5])}{' ...' if len(bad) > 5 else ''})"
            + (f" and {len(extra)} directories were read that the index does not list "
               f"({extra[:5]})" if extra else "")
            + ". A short read means a truncated merge and a long one a duplicated line; "
              "either would move every median, so refusing to average.")
    return {"reconciled": True, "n_cell_dirs_in_index": len(expected),
            "n_cell_dirs_seen": len(seen),
            "n_rows_seen_under_cells_prefix": int(sum(seen.values())),
            "n_rows_expected_by_index": int(sum(expected.values()))}


def stream_cells(merged: Path, refs: Any, index: dict[str, Any], *, backbone: str,
                 floor: dict[str, Any], limit: int | None = None, verbose: bool = True
                 ) -> tuple[dict[tuple[str, int, str], Group], dict[str, Any]]:
    """One pass over the cell rows of `t1_merged.jsonl`, matched against the
    references that `load_references` read in the pass before this one.

    The gates' modal path is built from the very same rows (a `Counter` per
    (method, budget, dataset, base_seed), handed to `density_form.modal_path`)
    rather than by re-reading the file: same decision-loading conventions,
    same `cells/` guard.
    """
    lookup = reference_lookup(refs)
    groups: dict[tuple[str, int, str], Group] = {}
    per_cell: collections.Counter = collections.Counter()
    seen_per_dir: collections.Counter = collections.Counter()
    excluded: collections.Counter = collections.Counter()
    w1_readable = bool(floor.get("turn_w1_readable_float32"))
    n_lines = n_cell = n_t3_dropped = n_used = n_capped = n_no_cache = 0

    with open(merged, encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if '"cell"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("kind") != "cell":
                continue
            n_cell += 1
            source_dir = str(rec.get("source_dir", ""))
            if not source_dir.startswith(CELL_PREFIX):
                n_t3_dropped += 1   # path-layer re-runs of the sampled prompts
                continue
            method = rec.get("mode")
            if method not in FAMILY:
                raise SystemExit(f"{source_dir}: mode {method!r} is not one of the nine "
                                 f"canonical methods {METHODS}")
            budget_raw = rec.get("budget")
            if not isinstance(budget_raw, str) or not budget_raw.startswith("K"):
                raise SystemExit(f"{source_dir}: budget {budget_raw!r} is not K<n>")
            budget = int(budget_raw[1:])
            if budget not in KS:
                raise SystemExit(f"{source_dir}: budget K{budget} is not one of {KS}")
            dataset = str(rec["dataset"])
            base_seed = int(rec["base_seed"])
            prompt_idx = int(rec["prompt_idx"])
            key = (method, budget, dataset)
            group = groups.get(key)
            if group is None:
                group = groups[key] = Group(method, budget, dataset)

            # counted before the cap and before every exclusion: this is the
            # denominator `reconcile_cell_dirs` checks against t1_index.json
            seen_per_dir[source_dir] += 1

            cell_key = (method, budget, dataset, base_seed)
            # the cap counts PAIRS, not rows: a row that is excluded below would
            # otherwise spend a slot and a smoke run would silently yield fewer
            # than N pairs per cell while the JSON says limit=N
            if limit is not None and per_cell[cell_key] >= limit:
                n_capped += 1
                continue

            if rec.get("ref_z_T_match") is not True:
                group.n_excluded_ref_mismatch += 1
                excluded[f"{method}_K{budget}_{dataset}"] += 1
                continue

            i = lookup.get((dataset, base_seed, prompt_idx))
            if i is None:
                raise SystemExit(f"{source_dir} idx {prompt_idx}: no reference row for "
                                 f"({dataset}, s{base_seed}, {prompt_idx})")
            ref = reference_row(refs, i)
            # The merger may have joined the cell to the references_t3/ RERUN of
            # the same (dataset, seed) stream -- since block A/C those streams
            # cover the full index range and the merger prefers them. The row
            # differenced here always comes from this reader's own references/
            # lookup, so the only thing to check is that both labels name a
            # clean reference stream of the SAME (dataset, seed); anything else
            # is still an unknown pairing and is refused.
            def _stream(label: str) -> str | None:
                for root in ("references/", "references_t3/"):
                    if label.startswith(root):
                        return label[len(root):]
                return None
            merged_stream = _stream(str(rec.get("ref_source_dir") or ""))
            resolved_stream = _stream(str(ref["source_dir"]))
            if merged_stream is None or merged_stream != resolved_stream:
                raise SystemExit(
                    f"{source_dir} idx {prompt_idx}: the merger joined this cell to "
                    f"{rec.get('ref_source_dir')!r} but the (dataset, base_seed, "
                    f"prompt_idx) key resolves to {ref['source_dir']!r}. The two joins "
                    f"disagree; refusing to average over an unknown pairing.")

            for name, length in PROFILE_LEN.items():
                v = rec.get(name)
                if v is None or len(v) != length:
                    raise SystemExit(f"{source_dir} idx {prompt_idx}: {name} has "
                                     f"{None if v is None else len(v)} entries, want {length}")
            if abs(float(rec["d"]) - ref["d"]) > 0:
                raise SystemExit(f"{source_dir} idx {prompt_idx}: d={rec['d']} but the "
                                 f"reference has d={ref['d']}")

            bits = rec.get("actions")
            if not isinstance(bits, str) or len(bits) != NUM_STEPS or set(bits) - {"0", "1"}:
                raise SystemExit(f"{source_dir} idx {prompt_idx}: actions is not a "
                                 f"{NUM_STEPS}-character 0/1 string ({bits!r})")
            n_cached = rec.get("n_cached")
            if n_cached is not None and int(n_cached) != bits.count("1"):
                raise SystemExit(f"{source_dir} idx {prompt_idx}: n_cached={n_cached} but "
                                 f"actions has {bits.count('1')} cached steps")
            k_realized = bits.count("1")
            if "1" not in bits:
                # every other data anomaly here is counted and kept; a schedule
                # that caches nothing has no k0, so its prefix/event readings do
                # not exist — the row is skipped, named and counted, never
                # allowed to kill a pass that has already read most of the file
                n_no_cache += 1
                group.n_no_cache_step += 1
                if len(group.no_cache_step_keys) < 10:
                    group.no_cache_step_keys.append(
                        f"{source_dir} ({dataset}, s{base_seed}, idx {prompt_idx})")
                continue
            k0 = bits.index("1")
            if k0 == 0:
                group.n_caches_step_0 += 1
            if FAMILY[method] == "fixed-table" and k_realized != budget:
                group.n_k_not_nominal += 1
            documented_k = DOCUMENTED_K.get(backbone, {}).get((method, budget))
            if documented_k is not None and k_realized != documented_k:
                group.n_k_not_documented += 1
            group.actions.setdefault(base_seed, collections.Counter())[bits] += 1
            group.mode_raw[str(rec.get("mode_raw") or method)] += 1

            row, parts, n_nan = pair_row(rec, ref)
            group.profiles.append(row)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vd = np.nanmax(np.abs(parts["vel_ratio"] - parts["disp_ratio"]))
            if np.isfinite(vd):
                group.max_vel_minus_disp = max(group.max_vel_minus_disp, float(vd))

            peaks = peak_readings(rec, ref)
            evr_c = np.asarray(rec["pca_evr"], dtype=np.float64)
            evr_r = np.asarray(ref["pca_evr"], dtype=np.float64)
            scal = np.array([
                float(rec["chord_len"]) / ref["chord_len"],
                float(rec["path_len"]) / ref["path_len"],
                float(rec["straightness"]) - ref["straightness"],
                float(rec["max_dev_ratio"]) - ref["max_dev_ratio"],
                float(evr_c[0] + evr_c[1]) - float(evr_r[0] + evr_r[1]),
                float(rec["update_chord_share"]) - ref["update_chord_share"],
                *peaks,
                # the ONE same-state reading of the ratio, at THIS row's own k0
                float(parts["disp_ratio"][k0]),
                float(k_realized), float(k0), float(n_nan),
            ], dtype=np.float32)
            group.scalars.append(scal)

            # prefix identity: expected bit-identical, so a non-zero reading is
            # one nameable generation, not a floor (see PrefixStat)
            pair_id = (f"{source_dir} idx {prompt_idx} "
                       f"(seed {rec.get('seed')})")
            for field in PREFIX_FIELDS:
                n = prefix_length(field, k0)
                group.prefix[field].update(
                    np.asarray(rec[field][:n], dtype=np.float64),
                    np.asarray(ref[field], dtype=np.float64)[:n],
                    pair_id)

            act = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
            for offset in EVENT_OFFSETS:
                for stratum, per_q in event_samples(act, parts, offset).items():
                    for quantity, values in per_q.items():
                        group.event_buffer(offset, stratum, quantity).extend(
                            values.astype(np.float32))
            if w1_readable:
                turn_c = np.asarray(rec["turn_angle_deg"], dtype=np.float64)
                turn_r = ref["turn_angle_deg"]
                ks = np.flatnonzero(act == 1)
                for offset in TURN_W1_OFFSETS:
                    j = ks + offset
                    j = j[(j >= 0) & (j < PROFILE_LEN["turn_angle_deg"])]
                    if j.size:
                        group.turn_w1[offset].extend(
                            (turn_c[j] - turn_r[j]).astype(np.float32))
            group.n_pairs += 1
            per_cell[cell_key] += 1
            n_used += 1
            if verbose and n_used % 20000 == 0:
                print(f"  ... {n_used:,} pairs", flush=True)

    recon = reconcile_cell_dirs(index, seen_per_dir)
    meta = {
        "lines_scanned": n_lines,
        "cell_rows_seen": n_cell,
        "n_dropped_not_cells_prefix": n_t3_dropped,
        "cell_prefix": CELL_PREFIX,
        "dropped_note": "rows whose source_dir is not under cells/ (i.e. the "
                        "cells_t3_rand50/ path-layer re-runs) are excluded: they would "
                        "double-count those prompts in every median AND move the "
                        "gates' modal path (mirror of density_form.py:152,415)",
        "pairs_used": n_used,
        "n_cell_dirs": len(seen_per_dir),
        "n_groups": len(groups),
        "excluded_ref_z_T_mismatch": dict(sorted(excluded.items())),
        "excluded_ref_z_T_mismatch_total": int(sum(excluded.values())),
        "n_rows_skipped_over_limit": n_capped,
        "n_rows_no_cache_step": n_no_cache,
        "no_cache_step_note": "a schedule that caches nothing has no first cache step k0, "
                              "so its prefix and event readings do not exist; such rows "
                              "are skipped, named per group in n_rows_no_cache_step_keys "
                              "and counted here, never averaged and never fatal",
        "limit_per_cell": limit,
        "limit_counts": "pairs, not rows: rows excluded on ref_z_T_match or skipped for "
                        "having no cache step do not spend a slot of --limit",
        "turn_w1_accumulated": w1_readable,
        "index_reconciliation": recon,
    }
    if verbose:
        print(f"  scanned {n_lines:,} rows; {n_cell:,} cell rows, dropped "
              f"{n_t3_dropped:,} non-cells/ rows, excluded {sum(excluded.values()):,} on "
              f"ref_z_T_match, skipped {n_no_cache:,} with no cache step; paired "
              f"{n_used:,} rows into {len(groups)} groups", flush=True)
    return groups, meta


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

SCALAR_TSV_COLUMNS: tuple[str, ...] = (
    "method", "family", "mode_raw", "K_nominal", "dataset", "n_pairs",
    "n_pairs_per_dataset", "pooling_weighting",
    "n_excluded_ref_mismatch", "k_realized_mean", "k_realized_min", "k_realized_max",
    "n_rows_k_not_nominal", "k_documented", "n_rows_k_not_documented",
    "k_documented_mismatch", "k_documented_bracketed", "duplicate_config",
    "duplicate_group", "counted_instance", "n_rows_caching_step_0",
    "n_rows_no_cache_step", "modal_k_realized", "modal_share",
    "modal_tie", "tail_gap_open", "n_distinct_paths",
) + tuple(f"{name}_{stat}" for name in PLAN_SCALARS + EXTRA_SCALARS
          for stat in ("med", "p25", "p75"))

PREFIX_TSV_COLUMNS: tuple[str, ...] = (
    "method", "K_nominal", "dataset", "field", "prefix_rule", "n_pairs",
    "n_pairs_with_a_prefix", "n_pairs_exactly_zero", "n_pairs_not_exactly_zero",
    "n_values_compared", "no_comparable_prefix",
    "max_abs_diff", "max_abs_diff_at_index", "max_abs_diff_pair",
    "max_rel_diff", "max_rel_diff_at_index",
    "note",
)

EVENT_TSV_COLUMNS: tuple[str, ...] = (
    "method", "K_nominal", "dataset", "quantity", "offset", "stratum",
    "median", "p25", "p75", "mean", "n_samples", "n_finite",
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def _tsv(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> Path:
    columns = list(columns)
    lines = ["\t".join(columns)]
    lines += ["\t".join(_fmt(r.get(c)) for c in columns) for r in rows]
    return atomic_write_text(path, "\n".join(lines) + "\n")


def _md_table(columns: Iterable[str], rows: list[list[Any]]) -> list[str]:
    columns = list(columns)
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(v) or "-" for v in row) + " |")
    return out + [""]


def scalar_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in report["groups"]:
        modal = g.get("modal_path") or {}
        pool = g.get("pooling") or {}
        row: dict[str, Any] = {
            "method": g["method"], "family": g["family"],
            "mode_raw": ",".join(g.get("mode_raw") or []),
            "K_nominal": g["budget"],
            "dataset": "pooled" if g["dataset"] == POOLED else g["dataset"],
            "n_pairs": g["n_pairs"],
            "n_pairs_per_dataset": ",".join(
                f"{ds}={n}" for ds, n in sorted((pool.get("n_pairs_per_dataset")
                                                 or {}).items())),
            "pooling_weighting": ("pair-weighted" if g["dataset"] == POOLED
                                  else "single dataset"),
            "n_excluded_ref_mismatch": g["n_excluded_ref_mismatch"],
            "n_rows_caching_step_0": g["n_rows_caching_step_0"],
            "n_rows_no_cache_step": g["n_rows_no_cache_step"],
            "modal_k_realized": modal.get("k_realized"),
            "modal_share": modal.get("modal_share"),
            "modal_tie": modal.get("modal_tie"),
            "tail_gap_open": modal.get("tail_gap_open"),
            "n_distinct_paths": modal.get("n_distinct_paths"),
        }
        row.update({k: v for k, v in g["k"].items() if k != "family"})
        for name in PLAN_SCALARS + EXTRA_SCALARS:
            s = g["scalars"][name]
            row[f"{name}_med"] = s["median"]
            row[f"{name}_p25"] = s["p25"]
            row[f"{name}_p75"] = s["p75"]
        rows.append(row)
    return rows


def prefix_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in report["groups"]:
        for field in PREFIX_FIELDS:
            entry = g["prefix_identity"][field]
            rows.append({
                "method": g["method"], "K_nominal": g["budget"],
                "dataset": "pooled" if g["dataset"] == POOLED else g["dataset"],
                **entry})
    return rows


def event_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in report["groups"]:
        for offset in EVENT_OFFSETS:
            for stratum in EVENT_STRATA:
                for quantity in EVENT_QUANTITIES:
                    entry = g["events"][str(offset)][stratum][quantity]
                    rows.append({
                        "method": g["method"], "K_nominal": g["budget"],
                        "dataset": "pooled" if g["dataset"] == POOLED else g["dataset"],
                        "quantity": quantity, "offset": offset, "stratum": stratum,
                        **entry})
    return rows


def boundary_lines(report: dict[str, Any]) -> list[str]:
    """Plan section 3.9.2 — emitted, never implied."""
    lines = ["## 1. What the T1 layer CANNOT measure (plan section 3.9.2)", "",
             "These are boundaries of this layer, not gaps in this run. They belong in "
             "`docs/video_cached_trajectory_results.md` sections 1 and 6.", ""]
    for i, item in enumerate(report["cannot_measure"], start=1):
        lines.append(f"{i}. **{item['quantity']}** — {item['why']}.")
    lines += ["", report["cannot_measure_owner"], ""]
    fields = ", ".join(f"`{item['field']}`" for item in report["not_prefix_identical"])
    lines += ["Not the same thing, and easy to confuse with a bug: "
              f"{fields} are **not** prefix-identical before the first cache step "
              "either — each references its own run's chord or whole path, and the "
              "cached run's endpoint differs. Their prefix differences belong in the "
              "profile tables, not in the identity check of section 4, where they are "
              "deliberately absent.", ""]
    return lines


def figure_caption(report: dict[str, Any], kind: str, subject: str) -> str:
    """The caption block that travels with a figure: what it shows, the caveat,
    and the section 3.9.2 boundaries."""
    T = report["backbone"]
    if kind == "profiles":
        what = (f"{T}, method `{subject}`: displacement ratio "
                f"`spacing_c[n]/spacing_r[n]` (top) and deviation difference "
                f"`d_perp_c/chord_c - d_perp_r/chord_r` (bottom), median and IQR over the "
                f"pairs of each budget, one column per K; the pooled medians in the tables "
                f"are pair-weighted, the curves here are per dataset. The shaded verticals "
                f"are the cache steps of THIS dataset's own data-derived modal path (the "
                f"top-1 `actions` string over its rows, `density_form.modal_path`) — for a "
                f"fixed-table method every row carries the same string, so the marks are "
                f"that table read back off the data; they are not read from the frozen "
                f"config, and for a gate they are this dataset's mode, not the pooled one.")
    else:
        what = (f"{T}: event alignment around every cache step k, offsets -2..+5, "
                f"stratified by whether step k+j is itself cached, pooled over datasets "
                f"(pair-weighted). Median per bucket; samples inside a bucket are not "
                f"independent (overlapping windows). The bottom row draws `dev_diff` at "
                f"the LITERAL same index k+j; the labelled secondary "
                f"`dev_diff_state_shifted` (state k+1+j, the first state a cache at step k "
                f"writes) is in the JSON and the TSV, not on this figure.")
    bounds = "; ".join(item["quantity"] for item in report["cannot_measure"])
    ratio_caveats = " ".join(report.get("figure_caveats")
                             or ratio_caveats_for(T))
    return (f"{what} T1 layer only — this figure CANNOT show: {bounds} "
            f"({report['cannot_measure_owner']}) {ratio_caveats}")


def write_md(report: dict[str, Any], path: Path, figures: list[Path]) -> Path:
    T = report["backbone"]
    src = report["source"]
    floor = report["floor"]
    lines = [f"# Cache bending, T1 layer — {T} (plan section 3.9.1, P6)", ""]
    lines += [
        f"Object: cell record vs reference record, pairing key "
        f"`(dataset, base_seed, prompt_idx)`, same z_T. "
        f"{report['denominators']['n_pairs_total']:,} pairs over "
        f"{report['denominators']['n_groups']} method x budget x dataset groups, from "
        f"`{src['merged']}`; {report['denominators']['n_dropped_not_cells_prefix']:,} rows "
        f"dropped for not being under `{CELL_PREFIX}` and "
        f"{report['denominators']['n_excluded_ref_mismatch']:,} excluded because "
        f"`ref_z_T_match` was not true; "
        f"{report['denominators']['n_rows_no_cache_step']:,} rows were skipped for "
        f"caching nothing (no first cache step k0, so no prefix and no event reading). "
        f"Every `{CELL_PREFIX}` directory's row count was checked against "
        f"`t1_index.json` (`source.cell_scan.index_reconciliation`), so a truncated "
        f"merge or a duplicated line cannot pass as a median.",
        "",
        f"Every reading here is a T1 reading and cites the **float32** P1 row "
        f"(`{Path(str(floor['source'].get('float32'))).name}`); the bfloat16 row is "
        f"carried in the JSON only to say what a T3 recomputation would cost.",
        "",
    ]
    lines += boundary_lines(report)

    lines += ["## 2. Caveats that travel with every number below", ""]
    lines += [f"- {c}" for c in report["caveats"]] + [""]

    lines += ["## 3. Per-trajectory scalar differences (median [IQR])", "",
              f"{report['denominators']['n_pooled_rows']} pooled method x budget rows and "
              f"{report['denominators']['n_dataset_rows']} per-dataset rows "
              f"(the plan's \"27 rows\" is the method x budget count; the global rule is "
              f"one block per backbone, one row per dataset, so both are emitted). "
              f"`k_realized_*` is the REALISED step count, not the nominal budget.", ""]
    lines += [f"**Weighting.** {CAV_POOLING}", ""]
    ident_cols = ["method", "family", "K", "dataset", "n_pairs", "pairs per dataset",
                  "k_real mean", "k_real min", "k_real max", "rows k!=K (fixed only)",
                  "k documented", "rows k!=documented", "documented mismatch",
                  "documented bracketed", "duplicate config", "modal k", "modal share"]
    rows = scalar_rows(report)
    lines += _md_table(ident_cols, [
        [r["method"], r["family"], r["K_nominal"], r["dataset"], r["n_pairs"],
         r["n_pairs_per_dataset"],
         r["k_realized_mean"], r["k_realized_min"], r["k_realized_max"],
         r["n_rows_k_not_nominal"], r["k_documented"], r["n_rows_k_not_documented"],
         r["k_documented_mismatch"], r["k_documented_bracketed"],
         r["duplicate_config"], r["modal_k_realized"], r["modal_share"]]
        for r in rows])
    lines += ["`rows k!=K (fixed only)` is empty for the four gates on purpose: a gate "
              "decides per video, so it has no nominal per-video count and a 0 there "
              "would read as \"no video deviates from the budget\". `documented mismatch` "
              "is now a PER-ROW verdict (any video missing the documented count), with "
              "the weaker bracket test `min <= documented <= max` kept beside it.", ""]
    diff_cols = ["method", "K", "dataset"] + [n.replace("_", " ") for n in PLAN_SCALARS]
    lines += _md_table(diff_cols, [
        [r["method"], r["K_nominal"], r["dataset"]] +
        [f"{_fmt(r[f'{n}_med'])} [{_fmt(r[f'{n}_p25'])}, {_fmt(r[f'{n}_p75'])}]"
         for n in PLAN_SCALARS]
        for r in rows])

    lines += ["### 3.1 Deviation peak per side, its shift, and the one same-state "
              "ratio reading", "",
              "Quantity (3) of plan section 3.9.1 asks for the peak state and peak value "
              "of `d_perp / own chord` on EACH side and the shift between them; "
              "`disp_ratio_at_k0` is read at each row's OWN first cache step, the one "
              "index where the two runs are still on the same state.", ""]
    lines += _md_table(
        ["method", "K", "dataset"] + [n.replace("dev_peak_", "peak ").replace("_", " ")
                                      for n in EXTRA_SCALARS],
        [[r["method"], r["K_nominal"], r["dataset"]] +
         [f"{_fmt(r[f'{n}_med'])} [{_fmt(r[f'{n}_p25'])}, {_fmt(r[f'{n}_p75'])}]"
          for n in EXTRA_SCALARS]
         for r in rows])

    lines += ["## 4. Prefix identity before the first cache step", ""]
    pooled_prefix = [r for r in prefix_rows(report) if r["dataset"] == "pooled"]
    worst: dict[str, dict[str, Any]] = {}
    empty_fields: dict[str, int] = {}
    for r in pooled_prefix:
        if r["max_abs_diff"] is None:      # nothing comparable in this group
            empty_fields[r["field"]] = empty_fields.get(r["field"], 0) + 1
            continue
        cur = worst.get(r["field"])
        if cur is None or r["max_abs_diff"] > cur["max_abs_diff"]:
            worst[r["field"]] = r
    measured = [r for r in pooled_prefix if r["max_abs_diff"] is not None]
    nonzero_rows = [r for r in measured if r["n_pairs_not_exactly_zero"]]
    n_zero_rows = len(measured) - len(nonzero_rows)
    typical = max((r["n_pairs_with_a_prefix"] for r in measured), default=0)
    worst_n = max((r["n_pairs_not_exactly_zero"] for r in nonzero_rows), default=0)
    lines += [
        f"**The floor is an exact zero, not the maximum in the table below.** The "
        f"prefix is identical by construction, and that is what is measured: "
        f"{n_zero_rows} of the {len(measured)} pooled (method x budget x field) rows "
        f"are bit-identical in EVERY pair, and no row has more than {worst_n:,} pair(s) "
        f"that are not (out of up to {typical:,}). The run-to-run reproducibility floor "
        f"of a normal pair is therefore exactly 0.", "",
        "**A non-zero row is one nameable generation, not a floor.** Where "
        "`n_pairs_not_exactly_zero` is 1, a SINGLE cached generation set that "
        "row's maximum by itself and `max_abs_diff_pair` names it, so the "
        "reading is about that one video and says nothing about the rest of the "
        "group. The signature to check on any such pair is "
        "whether its divergence from the reference already exists BEFORE k0 (the "
        "run picked up different numerics — a different kernel or node, or a "
        "re-run) or begins exactly AT k0 (the cache event itself, which is not a "
        "prefix reading at all); `latent_paths.py` reads the two paths state by "
        "state and settles it. Read the `max abs diff` column as an outlier "
        "magnitude from one generation, never as the size of a typical "
        "run-to-run difference, and never as an instrument floor to compare "
        "other readings against — for that, use the P1 float32 row quoted at the "
        "end of this section.", "",
        "Only the local quantities are listed — see section 1 for the fields that "
        "are not prefix-identical by construction.", ""]
    lines += _md_table(
        ["field", "prefix rule", "worst group", "pairs exactly zero / pairs with a prefix",
         "max abs diff", "at index", "max rel diff", "the one pair that set the max"],
        [[f, worst[f]["prefix_rule"], f"{worst[f]['method']} K{worst[f]['K_nominal']}",
          f"{worst[f]['n_pairs_exactly_zero']} / {worst[f]['n_pairs_with_a_prefix']}",
          worst[f]["max_abs_diff"], worst[f]["max_abs_diff_at_index"],
          worst[f]["max_rel_diff"], worst[f]["max_abs_diff_pair"]]
         for f in PREFIX_FIELDS if f in worst])
    if nonzero_rows:
        shown = sorted(nonzero_rows,
                       key=lambda r: -r["max_abs_diff"])[:MAX_NONZERO_PREFIX_ROWS]
        lines += [f"**Every pooled row that is not bit-identical** "
                  f"({len(nonzero_rows)} of {len(measured)}"
                  + (f", worst {len(shown)} shown" if len(shown) < len(nonzero_rows)
                     else "")
                  + f"). `pairs not exactly zero` is the count that decides how the "
                  f"maximum may be read; where it is 1, one generation set the row's "
                  f"maximum by itself. The full list is in "
                  f"`{Path(report['artefacts']['prefix_tsv']).name}`.", ""]
        lines += _md_table(
            ["field", "group", "pairs not exactly zero / pairs with a prefix",
             "max abs diff", "max rel diff", "the one pair that set the max"],
            [[r["field"], f"{r['method']} K{r['K_nominal']}",
              f"{r['n_pairs_not_exactly_zero']} / {r['n_pairs_with_a_prefix']}",
              r["max_abs_diff"], r["max_rel_diff"], r["max_abs_diff_pair"]]
             for r in shown])
    else:
        lines += ["**Every pooled row is bit-identical in every pair**: no non-zero "
                  "prefix difference was measured anywhere in this pass.", ""]
    n_pooled_prefix = len({(r["method"], r["K_nominal"]) for r in pooled_prefix})
    if empty_fields:
        lines += ["**Fields with no comparable prefix at all.** An empty cell below and "
                  "in the TSV is NOT a measured zero: at a small first cache step k0 the "
                  "field's prefix rule leaves no entry to compare, so nothing was "
                  "measured and `max_abs_diff` / `max_rel_diff` / "
                  "`n_pairs_exactly_zero` are null rather than 0. `turn_angle_w5_deg` "
                  "needs k0 >= 10 and `turn_angle_w7_deg` k0 >= 14.", ""]
        lines += _md_table(
            ["field", "prefix rule", "pooled rows with no comparable prefix"],
            [[f, PREFIX_RULE[f], f"{n} / {n_pooled_prefix}"]
             for f, n in sorted(empty_fields.items())])
        gone = [f for f in PREFIX_FIELDS if f not in worst]
        if gone:
            lines += [f"Not measurable anywhere in this pass (every pooled row empty): "
                      f"{', '.join('`' + f + '`' for f in gone)}.", ""]
    lines += [f"Full table (every group x field): "
              f"`{Path(report['artefacts']['prefix_tsv']).name}`.",
              "",
              f"P1 float32 row for scale: median relative spacing bias "
              f"{_fmt(floor['spacing_rel_bias_median']['float32'])}, first readable "
              f"deviation state {floor['dperp_first_readable_state']['float32']}, minimum "
              f"readable multi-step turn window "
              f"{floor['min_readable_multistep_turn_window']['float32']}, w=1 readable: "
              f"{_fmt(bool(floor['turn_w1_readable_float32']))}.", ""]

    lines += ["## 5. Event alignment around a cache step", "",
              "Offsets j in -2..+5 from every cache step k, stratified by whether step "
              "k+j is itself cached. Median (primary) and mean are both in the JSON and "
              "the TSV; `dev_diff` is read at the same integer index k+j and, as a "
              "labelled secondary, at state k+1+j (the first state a cache at step k "
              "writes).", ""]
    lines += [f"Full table: `{Path(report['artefacts']['event_tsv']).name}`.", ""]

    lines += ["## 6. Figures", ""]
    if not figures:
        lines += ["This pass ran with `--no_figures`; the tables and the JSON above are "
                  "complete, the nine per-method figures and the event-alignment figure "
                  "are not on disk.", ""]
    for fig in figures:
        kind = "alignment" if "event_alignment" in fig.name else "profiles"
        subject = fig.stem.split("cached_profiles_")[-1].rsplit(f"_{T}", 1)[0] \
            if kind == "profiles" else T
        lines += [f"**`{fig.name}`** — {figure_caption(report, kind, subject)}", ""]

    lines += ["## 7. Reproduction", "",
              "```", report["reproduce"], "```", ""]
    return atomic_write_text(path, "\n".join(lines))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _band(ax, x, entry, colour, label) -> None:
    med = np.asarray([np.nan if v is None else v for v in entry["median"]], dtype=float)
    p25 = np.asarray([np.nan if v is None else v for v in entry["p25"]], dtype=float)
    p75 = np.asarray([np.nan if v is None else v for v in entry["p75"]], dtype=float)
    ax.plot(x, med, lw=1.5, color=colour, label=label)
    ax.fill_between(x, p25, p75, color=colour, alpha=0.20, lw=0)


def write_method_figure(report: dict[str, Any], method: str, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = report["backbone"]
    groups = [g for g in report["groups"]
              if g["method"] == method and g["dataset"] != POOLED]
    datasets = sorted({g["dataset"] for g in groups})
    fig, axes = plt.subplots(2, len(KS), figsize=(5.0 * len(KS), 7.2), squeeze=False)
    steps = np.arange(NUM_STEPS)
    states = np.arange(N_STATES)
    for col, k in enumerate(KS):
        here = [g for g in groups if g["budget"] == k]
        top, bottom = axes[0][col], axes[1][col]
        if not here:
            for ax in (top, bottom):
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"K={k}: no rows in this pass", ha="center",
                        va="center", fontsize=10, color="#777777")
            continue
        for g in here:
            colour = ["#1f77b4", "#d62728", "#2ca02c"][datasets.index(g["dataset"]) % 3]
            label = f"{g['dataset']} (n={g['n_pairs']:,})"
            _band(top, steps, g["profiles"]["disp_ratio"], colour, label)
            _band(bottom, states, g["profiles"]["dev_diff"], colour, label)
            # THIS dataset's own data-derived modal path, for both families: for a
            # fixed table every row carries the same string, so the mode IS that
            # frozen table read back off the data; for a gate it is this dataset's
            # mode (the gates' thresholds are set per dataset), never the pooled
            # string. The caption says exactly this.
            bits = (g.get("modal_path") or {}).get("bits")
            if bits:
                for n, b in enumerate(bits):
                    if b != "1":
                        continue
                    for ax in (top, bottom):
                        ax.axvline(n, color=colour, alpha=0.10, lw=1.6, zorder=0)
        top.axhline(1.0, color="black", lw=0.8, ls=":")
        bottom.axhline(0.0, color="black", lw=0.8, ls=":")
        lo = [g["k"]["k_realized_min"] for g in here if g["k"]["k_realized_min"] is not None]
        hi = [g["k"]["k_realized_max"] for g in here if g["k"]["k_realized_max"] is not None]
        realized = f"realised {min(lo)}-{max(hi)}" if lo and hi else "no rows"
        top.set_title(f"K={k} nominal ({realized})", fontsize=10)
        top.set_ylabel("displacement ratio\n" r"$spacing_c[n]/spacing_r[n]$")
        bottom.set_ylabel("deviation difference\n"
                          r"$d_\perp^c/chord_c - d_\perp^r/chord_r$")
        top.set_xlabel("solver step n = 0..49")
        bottom.set_xlabel("state n = 0..50")
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    fig.suptitle(f"{T} — {method}: cached run vs its own reference, median and IQR "
                 f"(shaded verticals = cache steps of that dataset's own modal path)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written = atomic_savefig(fig, path, dpi=140)
    plt.close(fig)
    return written


def write_event_figure(report: dict[str, Any], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = report["backbone"]
    methods = [m for m in METHODS if any(g["method"] == m for g in report["groups"])]
    cmap = plt.get_cmap("tab10")
    colours = {m: cmap(i % 10) for i, m in enumerate(methods)}
    fig, axes = plt.subplots(2, len(KS), figsize=(5.0 * len(KS), 7.4), squeeze=False)
    offsets = np.asarray(EVENT_OFFSETS, dtype=float)
    for col, k in enumerate(KS):
        for r, quantity in enumerate(("disp_ratio", "dev_diff")):
            ax = axes[r][col]
            for m in methods:
                g = next((x for x in report["groups"]
                          if x["method"] == m and x["budget"] == k
                          and x["dataset"] == POOLED), None)
                if g is None:
                    continue
                for stratum, ls in (("cache", "-"), ("full", "--")):
                    ys = [g["events"][str(j)][stratum][quantity]["median"]
                          for j in EVENT_OFFSETS]
                    ys = [np.nan if v is None else v for v in ys]
                    if not np.isfinite(ys).any():
                        continue
                    ax.plot(offsets, ys, ls, color=colours[m], lw=1.3,
                            label=m if stratum == "cache" else None)
            ax.axvline(0.0, color="black", lw=0.8, ls=":")
            ax.axhline(1.0 if quantity == "disp_ratio" else 0.0,
                       color="black", lw=0.8, ls=":")
            ax.set_xlabel("offset j from a cache step k")
            ax.set_ylabel("displacement ratio (median)" if quantity == "disp_ratio"
                          else "deviation difference (median)")
            ax.set_title(f"K={k} nominal", fontsize=10)
            ax.grid(alpha=0.3)
    # one legend for the whole figure: a per-axes legend would only name the
    # methods that happen to have rows at that budget
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colours[m], lw=1.6, label=m) for m in methods]
    handles += [Line2D([0], [0], color="black", lw=1.2, ls="-", label="step k+j cached"),
                Line2D([0], [0], color="black", lw=1.2, ls="--", label="step k+j full")]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{T} — event alignment: every cache step k as origin, pooled over "
                 f"datasets", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    written = atomic_savefig(fig, path, dpi=140)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMPLETION_CRITERIA = (
    "1. 27-row (method x budget) scalar-difference table WITH realised step counts, "
    "plus the per-dataset rows: cached_scalar_diffs_<T>.tsv",
    "2. per-method profile figures, 3 budgets each, displacement ratio and deviation "
    "difference median+IQR, cache steps marked: cached_profiles_<method>_<T>.png (9)",
    "3. event-alignment figure, offsets -2..+5, cache/full strata: "
    "cached_event_alignment_<T>.png",
    "4. prefix-identity check (spacing, velocity_norm, magnitude, turn_angle_deg, "
    "second_diff_norm, turn_angle_w{5,7}), reported as pairs-exactly-zero WITH the "
    "outlier maximum and the single pair that set it: cached_prefix_identity_<T>.tsv",
    "5. cached_profiles_<T>.json with every number, run_params, the caveats, the "
    "section 3.9.2 cannot_measure block, the floor citation and all denominators",
)


def artefact_paths(T: str, out_tables: Path, out_figs: Path) -> dict[str, Path]:
    paths = {
        "json": out_tables / f"cached_profiles_{T}.json",
        "md": out_tables / f"cached_profiles_{T}.md",
        "scalar_tsv": out_tables / f"cached_scalar_diffs_{T}.tsv",
        "prefix_tsv": out_tables / f"cached_prefix_identity_{T}.tsv",
        "event_tsv": out_tables / f"cached_event_alignment_{T}.tsv",
        "event_fig": out_figs / f"cached_event_alignment_{T}.png",
    }
    for m in METHODS:
        paths[f"fig_{m}"] = out_figs / f"cached_profiles_{m}_{T}.png"
    return paths


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
    ap.add_argument("--out_tables", type=Path, default=None,
                    help="default resources/video_full_trajectory/<backbone>/")
    ap.add_argument("--out_figs", type=Path, default=None,
                    help="default docs/figures/video_full_trajectory/<backbone>/")
    ap.add_argument("--p1_floor_dir", type=Path, default=None,
                    help="directory holding p1_floor_{float32,bfloat16}.json "
                         "(default docs/figures/video_full_trajectory/<backbone>/)")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep at most N cell rows per (method, budget, dataset, seed) "
                         "cell — a smoke run. The references are ALWAYS read in full "
                         "(4,629 rows, ~15 MB): capping them would break the pairing for "
                         "every prompt_idx above the cap. N must be >= 1")
    ap.add_argument("--check", action="store_true",
                    help="print the P6 completion criteria and which artefacts are on "
                         "disk, then exit; reads no experiment data")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the outputs already exist")
    ap.add_argument("--no_figures", action="store_true", help="tables and JSON only")
    return ap


def run_check(T: str, paths: dict[str, Path]) -> None:
    print(f"=== P6 completion criteria — {T} (plan section 6, P6 row; section 3.9 T1 half) ===")
    for line in COMPLETION_CRITERIA:
        print(f"  {line}")
    print("\nartefacts:")
    for name, path in paths.items():
        state = "ok " if output_complete(path) else "MISSING"
        size = f"{path.stat().st_size:,} B" if path.is_file() else "-"
        print(f"  [{state}] {name:24s} {path}  ({size})")
    print("\nsection 3.9.2 boundaries carried into every artefact (T3 layer = P7):")
    for item in CANNOT_MEASURE:
        print(f"  - {item['quantity']}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    T = args.backbone
    traj = args.data_root / T / "matrix" / "trajectory"
    merged = args.merged or traj / "t1_merged.jsonl"
    index_path = args.index or traj / "t1_index.json"
    out_tables = args.out_tables or (_PROJECT_ROOT / "resources" / "video_full_trajectory" / T)
    out_figs = args.out_figs or (_PROJECT_ROOT / "docs" / "figures" /
                                 "video_full_trajectory" / T)
    paths = artefact_paths(T, out_tables, out_figs)

    if args.check:
        run_check(T, paths)
        return

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1 (leave it unset for no cap); 0 would keep "
                         "no cell rows at all")
    if not merged.is_file():
        raise SystemExit(f"merged table not found: {merged}")

    floor_dir = args.p1_floor_dir or (_PROJECT_ROOT / "docs" / "figures" /
                                      "video_full_trajectory" / T)
    run_params = {
        "backbone": T,
        "merged": str(merged),
        "merged_bytes": merged.stat().st_size,
        "limit": args.limit,
        "no_figures": bool(args.no_figures),
        "p1_floor_dir": str(floor_dir),
    }
    table_outputs = [paths["md"], paths["scalar_tsv"], paths["prefix_tsv"],
                     paths["event_tsv"]]
    stored = resolve_reuse(paths["json"], run_params, caps=("limit",), force=args.force,
                           extra_outputs=table_outputs)
    if stored is not None:
        missing = [] if args.no_figures else [
            p for name, p in paths.items()
            if name.startswith("fig_") or name == "event_fig"
            if not output_complete(p)]
        if not missing:
            print(f"[skip] outputs already exist under {out_tables} and {out_figs} for "
                  f"limit={args.limit}; pass --force to recompute")
            return
        print(f"[recompute] {len(missing)} figure(s) missing, empty or truncated: "
              f"{', '.join(p.name for p in missing)}")

    print(f"=== cached_vs_reference {T} ===")
    # one resolved directory, used both for the read and for run_params
    floor = load_p1_floor(T, floor_dir)
    index = load_index(index_path)
    sigmas = np.asarray(index["sigma_grids"][0], dtype=np.float64)
    if sigmas.shape[0] != N_STATES:
        raise SystemExit(f"sigma grid has {sigmas.shape[0]} entries, want {N_STATES}")
    # the references are never capped: --limit is a per-cell cap, and a capped
    # reference table would leave every prompt_idx above the cap unpaired
    refs = load_references(merged, index, limit=None)
    groups, scan_meta = stream_cells(merged, refs, index, backbone=T, floor=floor,
                                     limit=args.limit)
    if not groups:
        raise SystemExit(f"no cell rows under {CELL_PREFIX} in {merged}")

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in KS:
            here = [g for (m, b, _), g in sorted(groups.items())
                    if m == method and b == budget]
            if not here:
                continue
            rows.append(summarise_group(here, method, budget, POOLED, T, floor))
            for g in sorted(here, key=lambda g: g.dataset):
                rows.append(summarise_group([g], method, budget, g.dataset, T, floor))

    pooled_rows = [r for r in rows if r["dataset"] == POOLED]
    n_pairs_total = sum(r["n_pairs"] for r in pooled_rows)
    n_pooled_expected = len(METHODS) * len(KS)
    if len(pooled_rows) != n_pooled_expected:
        print(f"[WARN] {len(pooled_rows)} pooled method x budget rows, not the "
              f"{n_pooled_expected} of the full matrix ({len(METHODS)} methods x "
              f"{len(KS)} budgets). Expected only for a --limit smoke run or a partial "
              f"store; every table below states its own denominator.")
    # `disp@k0` is the per-row reading at EACH row's own first cache step (a
    # median-k0 lookup into the profile would be a different quantity for gates)
    print(f"\n{'method':16s} {'K':>3s} {'realised':>9s} {'pairs':>7s} "
          f"{'disp@k0':>9s} {'dev peak':>10s}")
    for r in pooled_rows:
        peak = r["profiles"]["dev_diff"]["peak_excluding_endpoints"]
        at_k0 = r["scalars"]["disp_ratio_at_k0"]["median"]
        span = f"{r['k']['k_realized_min']}-{r['k']['k_realized_max']}"
        print(f"{r['method']:16s} {r['budget']:3d} {span:>9s} {r['n_pairs']:7,d} "
              f"{_fmt(at_k0):>9s} {_fmt(peak['value']):>10s} @ state {peak['index']}")

    report: dict[str, Any] = {
        "backbone": T,
        "produced_by": "analysis/video_trajectory/cached_vs_reference.py",
        "plan_sections": ["3.9.1"],
        "layer": "T1 (per-run scalars). The T3 layer is P7, latent_paths.py cached.",
        "pairing_key": "(dataset, base_seed, prompt_idx); same z_T, verified per row by "
                       "the merger's ref_z_T_match and cross-checked against ref_source_dir",
        "index_convention": INDEX_CONVENTION,
        # only the caveats that are statements about THIS backbone: the Wan2.1
        # sampler rider is not printed on the HunyuanVideo page
        "caveats": caveats_for(T),
        "figure_caveats": ratio_caveats_for(T),
        "pooling_note": CAV_POOLING,
        "cannot_measure": [dict(item) for item in CANNOT_MEASURE],
        "cannot_measure_owner": CANNOT_MEASURE_OWNER,
        "not_prefix_identical": [dict(item) for item in NOT_PREFIX_IDENTICAL],
        "prefix_identity_rule": dict(PREFIX_RULE),
        "prefix_identity_note":
            "expected 0 and measured as an EXACT 0 for all but a handful of pairs "
            "(plan section 3.9.1's last table row): read `n_pairs_exactly_zero` first. "
            "`max_abs_diff` is the tail of that statistic, not a floor — where "
            "`n_pairs_not_exactly_zero` is 1, a single cached generation that ran on "
            "different numerics (different kernel/node, or a re-run) set the maximum by "
            "itself, and `max_abs_diff_pair` names it. Do NOT quote it as the T1 "
            "run-to-run reproducibility floor; the floor of a normal pair is 0 and the "
            "instrument floor is the cited P1 float32 row",
        "event_alignment": {
            "offsets": list(EVENT_OFFSETS),
            "strata": list(EVENT_STRATA),
            "origin": "every cache step k of the row's own `actions`",
            "wan_branch_note":
                "on Wan2.1 the decisions are the cond-branch decisions "
                "(`branch_policy: cond_decides`, uncond follows cond) and the recorded "
                "path is the single CFG-composed path (plan section 2.2)",
            "index_mapping":
                "primary: `disp_ratio` at solver step k+j and `dev_diff` at state k+j "
                "(the literal same-index reading); secondary `dev_diff_state_shifted` at "
                "state k+1+j, because the first state a cache at step k writes is k+1. "
                "A sample is kept only when 0 <= k+j <= 49, which is also what defines "
                "its stratum.",
            "statistic": "median + IQR primary (consistent with every other table here), "
                         "mean stored alongside (the plan's word for this row is "
                         "'stratified average')",
            "independence": CAV_OVERLAP,
        },
        "turn_w1": {
            "readable_float32": bool(floor.get("turn_w1_readable_float32")),
            "offsets": list(TURN_W1_OFFSETS),
            "offset_mapping":
                "junction n uses states n..n+2, so a cache at step k touches junctions "
                "n in {k-1, k}; offset +1 adds n = k+1 ('the step after'). The plan does "
                "not name the junction indices, so all three offsets are emitted with "
                "this mapping spelled out and a later reader can re-slice.",
            "index_note": TURN_W1_INDEX_NOTE,
        },
        "groups": rows,
        "denominators": {
            "n_pairs_total": int(n_pairs_total),
            "n_groups": len({(g.method, g.budget, g.dataset) for g in groups.values()}),
            "n_pooled_rows": len(pooled_rows),
            "n_dataset_rows": len(rows) - len(pooled_rows),
            "n_pooled_rows_expected": n_pooled_expected,
            "n_dropped_not_cells_prefix": scan_meta["n_dropped_not_cells_prefix"],
            "n_excluded_ref_mismatch": scan_meta["excluded_ref_z_T_mismatch_total"],
            "n_rows_no_cache_step": scan_meta["n_rows_no_cache_step"],
            "n_rows_skipped_over_limit": scan_meta["n_rows_skipped_over_limit"],
            "n_references": len(refs),
            "limit_per_cell": args.limit,
            "pooled_rows_are_pair_weighted": True,
        },
        "realized_k_note":
            "budgets are summarised by REALISED step count, not nominal. Fixed-table rows "
            "must realise exactly K per video (P0); a row that does not is counted in "
            "n_rows_k_not_nominal and never dropped. The three documented deviations of "
            "docs/video_full_results_report_zh.md section 5.1 are imported from density_form "
            "(DOCUMENTED_K / DUPLICATE_CONFIG): HYV sencache K37 and K41 are ONE "
            "configuration capped at 36 — both rows are marked with the same "
            "duplicate_group and counted_instance says which one a summary counts.",
        "documented_k_exceptions": {f"{m}_K{k}": v
                                    for (m, k), v in DOCUMENTED_K.get(T, {}).items()},
        "duplicate_config_rows": {f"{m}_K{k}": {"group": grp, "counted_instance": counted}
                                  for (m, k), (grp, counted)
                                  in sorted(DUPLICATE_CONFIG.get(T, {}).items())},
        "modal_path_note":
            "gate rows also carry a pooled modal path built with density_form.modal_path "
            "(pooled over the three base seeds, lexical tie-break, per-seed strings kept). "
            "The counters are accumulated in this script's single pass rather than by "
            "re-reading the file, with the same cells/ guard.",
        "sigmas": [float(v) for v in sigmas],
        "floor": floor,
        "floor_rows_cited": {
            "every reading in this file": "float32 (T1 profiles were computed in flight on "
                                          "float32 rows)",
            "bfloat16": "carried only to say what a T3 recomputation would cost; never "
                        "used for a P6 verdict",
        },
        "completion_criteria": list(COMPLETION_CRITERIA),
        "artefacts": {name: str(p) for name, p in paths.items()},
        "source": {**refs.meta, "index": str(index_path), "cell_scan": scan_meta,
                   "passes": 2,
                   "passes_note":
                       "two streaming passes over t1_merged.jsonl: step_profiles."
                       "load_references reads the reference rows (shared with P4/P5 "
                       "rather than duplicated here), then stream_cells reads the file "
                       "again for the cell rows. Nothing is held between them except the "
                       "reference columns."},
        "reproduce": (f"OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 python "
                      f"analysis/video_trajectory/cached_vs_reference.py --backbone {T} "
                      f"--data_root {args.data_root}"
                      + (f" --limit {args.limit}" if args.limit else "")),
        "run_params": run_params,
    }

    out_tables.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths["json"], report)
    print(f"\nwrote {paths['json']}")
    print(f"wrote {_tsv(paths['scalar_tsv'], SCALAR_TSV_COLUMNS, scalar_rows(report))}")
    print(f"wrote {_tsv(paths['prefix_tsv'], PREFIX_TSV_COLUMNS, prefix_rows(report))}")
    print(f"wrote {_tsv(paths['event_tsv'], EVENT_TSV_COLUMNS, event_rows(report))}")

    figures: list[Path] = []
    if not args.no_figures:
        out_figs.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            if not any(r["method"] == method for r in rows):
                continue
            figures.append(write_method_figure(report, method, paths[f"fig_{method}"]))
            print(f"wrote {figures[-1]}")
        figures.append(write_event_figure(report, paths["event_fig"]))
        print(f"wrote {figures[-1]}")
    print(f"wrote {write_md(report, paths['md'], figures)}")


if __name__ == "__main__":
    main()
