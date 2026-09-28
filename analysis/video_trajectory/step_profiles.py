#!/usr/bin/env python3
"""Clean-reference per-step profiles and whole-trajectory shape scalars for one
video backbone — plan sections 3.1-3.6 (docs/video_full_trajectory_plan_zh.md, P4).

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
    python analysis/video_trajectory/step_profiles.py --backbone hunyuan_video \\
        --data_root outputs

Reads only the merged T1 table (`$DATA/<T>/matrix/trajectory/t1_merged.jsonl` +
`t1_index.json`) written by `merge_video_traj.py`, streaming it line by line and
keeping ONLY the 4,629 reference rows (`mode == "original"`, directory
`references/`); the 124,983 cell rows are dropped in the reader and the
`references_t3/` re-runs, if the merge was redone after P2, are dropped as well
(they duplicate two of the six streams). Retained payload is
51+50+51+50+49+41+37+49 = 378 floats plus ~12 scalars per row, ~15 MB as
float64; the startup line prints the measured figure.

Sections produced here:

  3.1 how far off the straight line   d_perp[n] / chord_len          states 1-49
  3.2 how big the state is            magnitude[n] / sqrt(d)         states 0-50
  3.3 how far one step moves          spacing[n] / chord_len         steps  0-49
  3.4 sampler vs model                spacing = velocity x |dsigma|  steps  0-49
  3.5 how much of it is common        per-step CV + two one-factor variance
                                      shares (between prompts / between noises)
  3.6 whole-trajectory shape scalars  chord / straightness / max_dev / PCA
                                      spectrum, per (dataset, base_seed)

Section 3.6's 3-D overlay needs full latent paths and lives in
`latent_paths.py`; the curvature profile, bend plane and update subspace of
section 3.8 live in `shape_scale.py`.

Index traps this file asserts rather than assumes (plan section 2.4):
`d_perp` / `magnitude` / `sigmas` are states n = 0..50, `spacing` /
`velocity_norm` are solver steps n = 0..49 with `spacing[n] = ||Z[n+1]-Z[n]||`,
and `d_perp[0] = d_perp[50] = 0` by construction, so every dispersion window
over the deviation profile runs 1..49. Sigma is non-uniform with the LAST step
the largest (HYV 0.0029 -> 0.1250, Wan 0.0041 -> 0.0925), so the per-step
figures carry a sigma axis on top.

Floors are cited, never re-measured, and they are READ — not transcribed —
from the P1 dumps `docs/figures/video_full_trajectory/<T>/p1_floor_{float32,
bfloat16}.json`, which carry the full per-step / per-state arrays. Every
reading here comes from the float32 in-flight rows, so the float32 row of that
dump applies throughout; the bf16 entries travel along only to say what a
recomputation of the same profile from the T3 store would cost. P2 re-issues
the floor dumps at >= 30 paths per dataset, and this file then follows.

`--combine` is the cross-backbone step: it reads both backbones'
`step_profiles_<T>.json` and writes the plan's F2 figure (4 profiles x 2
backbones per dataset) plus the per-profile backbone ordering statement.
"""

from __future__ import annotations

import os

# BLAS pools must be capped before numpy loads (plan section 5.2): the work is
# thousands of small reductions, and an uncapped pool pays a barrier on each.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.full_trajectory_analysis import (  # noqa: E402
    _cv_readings,
    _peak_readings,
    _spread,
)
from analysis.video_trajectory.common import (  # noqa: E402
    TURN_W1_INDEX_NOTE,
    atomic_savefig,
    atomic_write_json,
    atomic_write_text,
    output_complete,
    resolve_reuse,
)

BACKBONES = ("hunyuan_video", "wan21")
DEFAULT_DATA_ROOT = Path("outputs")
N_STATES = 51
REFERENCE_PREFIX = "references/"  # NOT references_t3/: those re-run two streams
FRAME_SEGMENT = "000_051"         # the whole-path T2 frame
SHARE_STATES = (1, 5, 10, 25, 40, 49)  # plan section 3.5 report states
LANDMARK_STATES = (0, 1, 2, 5, 10, 20, 25, 30, 40, 45, 47, 48, 49, 50)
SNR_MIN = 3.0                     # plan section 3.0 readability convention
FLOOR_SPACING_STEPS = (0, 1, 10, 20, 30, 40, 49)   # the p1_floor.md table columns
FLOOR_DEV_STATES = (1, 2, 3, 5, 10, 20, 30, 40, 49)

# ---------------------------------------------------------------------------
# P1 floors — READ from the P1 dumps, never transcribed and never recomputed
# here (plan section 3.0 is done; P4 only states which row applies to which
# reading, and P2 re-issues the dumps at >= 30 paths per dataset). Everything
# in this file is a T1 reading, so the float32 row applies throughout; the bf16
# entries are carried only so the JSON can say what a T3 recomputation of the
# same profile would cost.
# ---------------------------------------------------------------------------

DEFAULT_FLOOR_DIR = _PROJECT_ROOT / "docs" / "figures" / "video_full_trajectory"


def _snr(measured: Any, floor: Any) -> np.ndarray:
    """measured / floor with 0/0 -> nan (the two chord endpoints of d_perp)."""
    m = np.asarray(measured, dtype=np.float64)
    f = np.asarray(floor, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, m / np.where(f > 0, f, 1.0), np.nan)


def _window_verdicts(dump: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    """Per-window SNR summary of one dump, and the minimum readable MULTI-step
    window (the smallest w > 1 whose every centre clears SNR_MIN). Same rule as
    `analysis/video_trajectory/p1_floor_report.py`, which wrote p1_floor.md."""
    out: dict[str, Any] = {}
    min_w: int | None = None
    for w, entry in sorted(dump["windows"].items(), key=lambda kv: int(kv[0])):
        ratio = np.asarray(entry["ratio"], dtype=np.float64)
        centres = [int(c) for c in entry["centers"]]
        ok = bool((ratio >= SNR_MIN).all())
        out[str(w)] = {
            "centres": len(ratio),
            "centres_ge_snr_min": int((ratio >= SNR_MIN).sum()),
            "min_snr": float(np.nanmin(ratio)),
            "min_snr_at": centres[int(np.nanargmin(ratio))],
            "median_snr": float(np.nanmedian(ratio)),
            "readable": ok,
        }
        if ok and min_w is None and int(w) > 1:
            min_w = int(w)
    return out, min_w


def load_p1_floor(backbone: str, floor_dir: Path | None = None, *,
                  float32_json: Path | None = None,
                  bfloat16_json: Path | None = None) -> dict[str, Any]:
    """The P1 floor of one backbone, read off the two dumps P1 wrote.

    Everything below is either a stored array (`spacing_rel_med` 50 entries,
    `deviation.{measured_med,floor_med}` 51 entries, per-window `ratio`) or a
    verdict derived from one with the readability rule P1 itself used. Nothing
    is typed in, so re-issuing the dumps at P2 changes every floor row that
    this script and `shape_scale.py` print.
    """
    base = Path(floor_dir) if floor_dir is not None else (DEFAULT_FLOOR_DIR / backbone)
    paths = {"float32": Path(float32_json) if float32_json else base / "p1_floor_float32.json",
             "bfloat16": Path(bfloat16_json) if bfloat16_json else base / "p1_floor_bfloat16.json"}
    dumps: dict[str, dict[str, Any]] = {}
    for dtype, path in paths.items():
        if not path.is_file():
            raise SystemExit(
                f"P1 {dtype} floor dump not found at {path}. Every P4 reading has to state "
                f"its floor (plan section 2.2 point 3) and the floors are not recomputed "
                f"here; run P1 first or pass --p1_floor_dir.")
        dumps[dtype] = json.loads(path.read_text(encoding="utf-8"))

    windows: dict[str, Any] = {}
    min_multistep: dict[str, int | None] = {}
    for dtype, dump in dumps.items():
        windows[dtype], min_multistep[dtype] = _window_verdicts(dump)

    dev_snr = {dtype: _snr(d["deviation"]["measured_med"], d["deviation"]["floor_med"])
               for dtype, d in dumps.items()}
    first_readable = {}
    for dtype, snr in dev_snr.items():
        hits = [n for n in range(1, 50) if np.isfinite(snr[n]) and snr[n] >= SNR_MIN]
        first_readable[dtype] = hits[0] if hits else None

    spacing = {dtype: np.asarray(d["spacing_rel_med"], dtype=np.float64)
               for dtype, d in dumps.items()}
    plane = {dtype: (1.0 - d["plane_share"]["measured_med"],
                     1.0 - d["plane_share"]["planar_floor_med"])
             for dtype, d in dumps.items()}

    w1 = windows["float32"].get("1", {})
    floor: dict[str, Any] = {
        "backbone": backbone,
        "source": {dtype: str(p) for dtype, p in paths.items()},
        "n_trajectories": {dtype: d.get("n_trajectories") for dtype, d in dumps.items()},
        "snr_min": SNR_MIN,
        "applies_to_T1": "float32 row (the T1 profiles were computed in flight on "
                         "float32 rows)",
        "windows": windows,
        "min_readable_multistep_turn_window": min_multistep,
        "turn_w1_readable_float32": bool(w1.get("readable", False)),
        "turn_w1_min_snr_float32": w1.get("min_snr"),
        "turn_w1_min_snr_at_float32": w1.get("min_snr_at"),
        "dperp_snr": {dtype: [None if not np.isfinite(v) else float(v) for v in snr]
                      for dtype, snr in dev_snr.items()},
        "dperp_first_readable_state": first_readable,
        "spacing_rel_bias": {dtype: [float(v) for v in arr] for dtype, arr in spacing.items()},
        "spacing_rel_bias_median": {dtype: float(np.median(arr))
                                    for dtype, arr in spacing.items()},
        "magnitude_one_rounding_rel_shift":
            {dtype: float(np.median(np.abs(d["magnitude_rel_med"])))
             for dtype, d in dumps.items()},
        "chord_rel_floor": {dtype: float(d["chord_rel_med"]) for dtype, d in dumps.items()},
        "path_len_rel_bias": {dtype: float(d["path_len_rel_med"]) for dtype, d in dumps.items()},
        "straightness_rel_bias": {dtype: float(d["straightness_rel_med"])
                                  for dtype, d in dumps.items()},
        "offplane_energy_over_floor": {dtype: float(m / f) if f > 0 else None
                                       for dtype, (m, f) in plane.items()},
        "readability_rule": f"a reading is usable where measured / floor >= {SNR_MIN:g}; "
                            f"the minimum readable multi-step window is the smallest w > 1 "
                            f"whose every window centre clears it",
    }
    return floor

# plan section 2.2: what `velocity_norm` may be called on each backbone
SAMPLER_NOTE = {
    "hunyuan_video": ("FlowMatchDiscreteScheduler / Euler: velocity_norm[n] = "
                      "spacing[n]/|dsigma_n| is exactly the model output norm "
                      "||v_theta(z_n, sigma_n)||, so the sampler-vs-model split is exact"),
    "wan21": ("FlowUniPCMultistepScheduler / UniPC 2nd-order multistep: z_{n+1} depends "
              "on the current AND previous model outputs, so velocity_norm is the "
              "sigma-domain path speed only — the algebraic split spacing = |dsigma| x "
              "velocity is reported, no 'model output' reading"),
}

# plan section 3.5, the estimation boundaries that must travel with the two
# variance shares (1-3 copied from the image-side section 4.2, 4 is video-side
# specific, 5 states the mixed denominator convention of the ratio itself)
ESTIMATION_BOUNDARIES = [
    "1. The within-group variance uses the UNBIASED estimator (pooled sum of squares "
    "divided by sum of (group size - 1)); plugging in the raw within-group variance "
    "overstates the share.",
    "2. The initial noise is shared between prompts (~946 noise tensors for 4,629 "
    "references), so the prompts in the between-prompt bucket are not independent "
    "draws: both shares are point estimates, reported without an interval and without "
    "a claim about the direction of any bias.",
    "3. Around state 1 the deviation sits only a few multiples above the floor; the "
    "rounding is independent between generations and therefore lands in the "
    "same-prompt-different-seed bucket, which depresses the early-state share. This "
    "batch cannot separate that from real seed sensitivity.",
    "4. seed = base_seed + prompt_idx makes prompts i and i+1 share 2 of their 3 seeds, "
    "so adjacent prompts are not independent samples inside the between-prompt bucket "
    "either; both shares are read per prompt count / per noise count, never per record "
    "count.",
    "5. The ratio mixes two denominator conventions: share = 1 - MS_within / Var_pop, "
    "where the within term is the unbiased one of boundary 1 (pooled SS / (N - G)) but "
    "the TOTAL is the POPULATION variance of the used records (numpy ddof = 0, SS / N), "
    "not the unbiased total (SS / (N - 1)). Using the unbiased total instead moves the "
    "published shares by <= 4e-4 relative on the prompt shares and by about 1 % on the "
    "noise shares (whose groups are small), so no verdict here turns on the choice — "
    "but the shares of this file and of any file that uses the other convention are not "
    "interchangeable at that precision.",
]

INDEX_CONVENTION = {
    "d_perp / magnitude / sigmas": "state n = 0..50 (n = state after n solver steps; 0 = z_T)",
    "spacing / velocity_norm": "solver step n = 0..49 (spacing[n] = ||Z[n+1] - Z[n]||)",
    "turn_angle_deg / second_diff_norm": TURN_W1_INDEX_NOTE,
    "turn_angle_w5_deg / turn_angle_w7_deg": "window centre c = 5..45 / 7..43 "
                                             "(the array index is NOT the centre)",
    "d_perp[0], d_perp[50]": "zero by construction; excluded from every CV / share window",
}

# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

PROFILE_FIELDS: dict[str, int] = {
    "d_perp": 51, "spacing": 50, "magnitude": 51, "velocity_norm": 50,
    "turn_angle_deg": 49, "turn_angle_w5_deg": 41, "turn_angle_w7_deg": 37,
    "second_diff_norm": 49,
}
VEC5_FIELDS = ("pca_evr", "update_own_evr")
SCALAR_FIELDS = ("chord_len", "path_len", "straightness", "max_dev_ratio",
                 "perp_var_total", "recon_err_rel_2d", "recon_err_rel_3d",
                 "update_chord_share", "update_in_position_plane", "d")
ID_FIELDS = ("dataset", "base_seed", "prompt_idx", "seed", "z_T_sha256", "source_dir")


class References:
    """The 4,629 clean reference rows of one backbone, as column arrays."""

    def __init__(self, cols: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        self.cols = cols
        self.meta = meta

    def __len__(self) -> int:
        return len(self.cols["chord_len"])

    def mask(self, dataset: str | None = None, base_seed: int | None = None) -> np.ndarray:
        m = np.ones(len(self), dtype=bool)
        if dataset is not None:
            m &= self.cols["dataset"] == dataset
        if base_seed is not None:
            m &= self.cols["base_seed"] == base_seed
        return m

    def streams(self) -> list[tuple[str, int]]:
        pairs = {(str(d), int(s)) for d, s in zip(self.cols["dataset"], self.cols["base_seed"])}
        return sorted(pairs)

    def datasets(self) -> list[str]:
        return sorted({str(d) for d in self.cols["dataset"]})


def load_references(merged: Path, index: dict[str, Any], *, limit: int | None = None,
                    keep_frames: bool = False, verbose: bool = True) -> References:
    """Stream `t1_merged.jsonl` and keep only the clean references.

    The fast path is a substring test on the raw line; every surviving line is
    still parsed and re-checked, and the final count is verified against the
    per-directory counts in `t1_index.json`, so a formatting change in the
    merger turns into a loud stop rather than a silently short table.
    """
    prof: dict[str, list] = {k: [] for k in PROFILE_FIELDS}
    vec5: dict[str, list] = {k: [] for k in VEC5_FIELDS}
    scal: dict[str, list] = {k: [] for k in SCALAR_FIELDS}
    ident: dict[str, list] = {k: [] for k in ID_FIELDS}
    frames: list[str] = []
    per_stream: collections.Counter = collections.Counter()
    dtypes: collections.Counter = collections.Counter()
    n_lines = n_hit = n_t3_dropped = 0

    with open(merged, encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if '"original"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("mode") != "original":
                continue
            n_hit += 1
            source_dir = rec.get("source_dir", "")
            if not source_dir.startswith(REFERENCE_PREFIX):
                n_t3_dropped += 1  # references_t3/ re-runs of two streams (P2)
                continue
            key = (rec["dataset"], int(rec["base_seed"]))
            if limit is not None and per_stream[key] >= limit:
                continue
            per_stream[key] += 1
            for name, length in PROFILE_FIELDS.items():
                v = rec[name]
                if len(v) != length:
                    raise ValueError(f"{source_dir}: {name} has {len(v)} entries, want {length}")
                prof[name].append(v)
            for name in VEC5_FIELDS:
                vec5[name].append(rec[name])
            for name in SCALAR_FIELDS:
                scal[name].append(rec[name])
            for name in ID_FIELDS:
                ident[name].append(rec[name])
            dtypes[(rec.get("z_T_dtype"), rec.get("path_dtype"))] += 1
            if keep_frames:
                files = rec.get("frame_files") or {}
                frames.append(files.get(FRAME_SEGMENT)
                              or (list(files.values())[0] if files else ""))

    cols: dict[str, np.ndarray] = {}
    for name in PROFILE_FIELDS:
        cols[name] = np.asarray(prof[name], dtype=np.float64)
    for name in VEC5_FIELDS:
        cols[name] = np.asarray(vec5[name], dtype=np.float64)
    for name in SCALAR_FIELDS:
        cols[name] = np.asarray(scal[name], dtype=np.float64)
    cols["dataset"] = np.asarray(ident["dataset"], dtype=object)
    cols["base_seed"] = np.asarray(ident["base_seed"], dtype=np.int64)
    cols["prompt_idx"] = np.asarray(ident["prompt_idx"], dtype=np.int64)
    cols["seed"] = np.asarray(ident["seed"], dtype=np.int64)
    cols["z_T_sha256"] = np.asarray(ident["z_T_sha256"], dtype=object)
    cols["source_dir"] = np.asarray(ident["source_dir"], dtype=object)
    if keep_frames:
        cols["frame_file"] = np.asarray(frames, dtype=object)

    expected = {k: v["n_t1"] for k, v in index["dirs"].items()
                if k.startswith(REFERENCE_PREFIX)}
    want = sum(min(n, limit) if limit is not None else n for n in expected.values())
    got = len(cols["chord_len"])
    if got != want:
        raise SystemExit(
            f"read {got} reference rows but t1_index.json lists {want} in "
            f"{len(expected)} references/ directories (limit={limit}); the line "
            f"prefilter or the merge is out of step")

    footprint = sum(a.nbytes for a in cols.values() if a.dtype != object) / 1e6
    meta = {
        "merged": str(merged), "lines_scanned": n_lines, "original_rows_seen": n_hit,
        "references_kept": got, "references_t3_rows_dropped": n_t3_dropped,
        "cell_rows_discarded_in_reader": n_lines - n_hit,
        "per_stream": {f"{d}_s{s}": n for (d, s), n in sorted(per_stream.items())},
        "retained_float_payload_MB": round(footprint, 1),
        "limit": limit,
        # which dtypes the rows below were computed on — the floor row every
        # reading cites is chosen by this, not assumed (plan section 2.2)
        "row_dtypes": {f"z_T={z}/path={p}": n for (z, p), n in sorted(
            dtypes.items(), key=lambda kv: str(kv[0]))},
    }
    if verbose:
        print(f"  scanned {n_lines:,} rows; kept {got:,} clean references "
              f"({n_lines - n_hit:,} cell rows discarded in the reader, "
              f"{n_t3_dropped} references_t3 rows dropped); "
              f"retained payload {footprint:.1f} MB float64", flush=True)
    return References(cols, meta)


def load_index(path: Path) -> dict[str, Any]:
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("n_sigma_grids") != 1:
        raise SystemExit(f"{path}: {index.get('n_sigma_grids')} sigma grids, expected 1")
    return index


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _series(values: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def _cv(arr: np.ndarray) -> np.ndarray:
    """Population CV per column. The two chord endpoints of the deviation
    profile have mean 0, so 0/0 is expected there and comes back NaN; the
    windows that consume this always exclude them."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr.std(axis=0) / arr.mean(axis=0)


def _landmarks(profile: np.ndarray) -> dict[str, float]:
    return {str(n): float(profile[n]) for n in LANDMARK_STATES if n < len(profile)}


def group_variance_share(profiles: np.ndarray, group_ids: np.ndarray,
                         *, min_size: int = 2) -> dict[str, Any]:
    """One-factor variance share with the unbiased pooled within-group estimator.

    `share[k] = (var_total[k] - var_within[k]) / var_total[k]` at column k, over
    the rows that belong to a group of at least `min_size` members;
    `var_within = sum_g SS_g / sum_g (m_g - 1)`. For equal group sizes this is
    the same estimator as `full_trajectory_analysis.prompt_variance_share` (the
    image-side protocol this replicates); unequal sizes — the z_T groups run
    2..6 — need the pooled form, which is why that function is not called.

    The two terms use DIFFERENT denominators and the ratio is reported as such
    (ESTIMATION_BOUNDARIES entry 5): the within term is unbiased (N - G) while
    `total` is `ndarray.var(ddof=0)` over the used rows, i.e. the POPULATION
    variance (N). Switching the total to the unbiased (N - 1) form moves the
    published shares by <= 4e-4 relative (prompt) / about 1 % (noise, small
    groups); the ddof=0 pair is kept rather than silently changed, because the
    image-side protocol this replicates uses the same pair.
    """
    groups: dict[Any, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    used = [np.asarray(idx) for idx in groups.values() if len(idx) >= min_size]
    if not used:
        raise ValueError("no group reaches min_size")
    ss = np.zeros(profiles.shape[1], dtype=np.float64)
    dof = 0
    for idx in used:
        block = profiles[idx]
        ss += ((block - block.mean(axis=0)) ** 2).sum(axis=0)
        dof += len(idx) - 1
    rows = np.concatenate(used)
    total = profiles[rows].var(axis=0)
    within = ss / dof
    share = np.where(total > 0, (total - within) / np.where(total > 0, total, 1.0), np.nan)
    sizes = collections.Counter(len(idx) for idx in used)
    return {
        "share": share,
        "groups_used": len(used),
        "groups_total": len(groups),
        "rows_used": int(len(rows)),
        "group_size_histogram": {str(k): int(v) for k, v in sorted(sizes.items())},
        "excluded_groups_below_min_size": int(len(groups) - len(used)),
        "min_size": min_size,
    }


# ---------------------------------------------------------------------------
# 3.1 how far off the straight line
# ---------------------------------------------------------------------------


def deviation_readings(dev: np.ndarray, sigmas: np.ndarray, max_dev_ratio: np.ndarray,
                       floor: dict[str, Any]) -> dict[str, Any]:
    """`d_perp/chord` population profile, its peak, and the kink search.

    The two endpoints are zero by construction, so the peak is searched on
    states 1..49 and every index reported here is a STATE index.
    """
    if dev.shape[1] != N_STATES:
        raise ValueError(f"deviation profile has {dev.shape[1]} states, want {N_STATES}")
    if not np.allclose(dev[:, 0], 0.0, atol=0.0):
        raise ValueError("d_perp[0] is not exactly zero: index convention broken")
    end_residue = float(np.abs(dev[:, -1]).max())
    if end_residue > 1e-6:
        raise ValueError(f"d_perp[50]/chord max {end_residue:.2e}: not a float64 residue")

    med = np.median(dev, axis=0)
    q25, q75 = np.quantile(dev, 0.25, axis=0), np.quantile(dev, 0.75, axis=0)
    peaks = _peak_readings(med)              # searched over the full 51 (ends are 0)
    peak_state = int(peaks["peak_step"])
    per_traj_peak = dev[:, 1:50].argmax(axis=1) + 1
    mode_state = int(np.bincount(per_traj_peak).argmax())

    # kink: second difference of the median profile over the interior states.
    # A kink is an ISOLATED spike, so both tests have to pass: |d2| far above
    # the typical |d2| of the smooth hump (scale = the median over states
    # 2..48) AND far above its own two neighbours. The second test is what
    # keeps the steep but perfectly smooth tail near state 47 from being
    # reported as a kink, which the first test alone does.
    d2 = np.full(N_STATES, np.nan)
    d2[1:-1] = med[2:] - 2 * med[1:-1] + med[:-2]
    interior = np.abs(d2[2:49])
    background = float(np.median(interior))
    ratios = np.abs(d2) / background if background > 0 else np.full(N_STATES, np.nan)
    # the instrument SNR of the deviation reading at that state, straight out
    # of the P1 float32 dump — a different quantity from `over_background`,
    # which is a shape statistic of this same profile
    dev_snr = floor["dperp_snr"]["float32"]

    def snr_at(n: int) -> float | None:
        return dev_snr[n] if n < len(dev_snr) else None

    kinks = [{"state": int(n), "second_diff": float(d2[n]), "over_background": float(ratios[n]),
              "over_neighbours": float(abs(d2[n]) / max(abs(d2[n - 1]), abs(d2[n + 1]))),
              "dperp_snr_float32": snr_at(int(n))}
             for n in range(2, 49)
             if ratios[n] >= 5.0
             and abs(d2[n]) >= 3.0 * max(abs(d2[n - 1]), abs(d2[n + 1]))]
    worst = int(2 + np.nanargmax(np.abs(d2[2:49])))
    top = [int(2 + i) for i in np.argsort(np.abs(d2[2:49]))[::-1][:3]]

    return {
        "window_states": [1, 49],
        "profile_median": _series(med),
        "profile_q25": _series(q25),
        "profile_q75": _series(q75),
        "peak_state": peak_state,
        "peak": float(peaks["peak"]),
        "sigma_at_peak": float(sigmas[peak_state]),
        "half_first_state": int(peaks["half_first_step"]),
        "half_last_state": int(peaks["half_last_step"]),
        # an INCLUSIVE count of states at or above half the peak
        # (half_last - half_first + 1), not the span between the two endpoints:
        # states 24..49 count 26, and the span is 25
        "half_width_states_inclusive": int(peaks["half_width_steps"]),
        "plateau90_first_state": int(peaks["plateau90_first_step"]),
        "plateau90_last_state": int(peaks["plateau90_last_step"]),
        "per_trajectory_peak_state": _spread(per_traj_peak.astype(float)),
        "per_trajectory_peak_mode": mode_state,
        "per_trajectory_peak_mode_share": float((per_traj_peak == mode_state).mean()),
        "per_trajectory_peak_within_2_share":
            float((np.abs(per_traj_peak - mode_state) <= 2).mean()),
        # the population profile's peak and the per-trajectory max_dev_ratio are
        # two algorithms on ONE fact, not two pieces of evidence
        "max_dev_ratio_median": float(np.median(max_dev_ratio)),
        "peak_vs_max_dev_ratio_rel_diff":
            float((peaks["peak"] - np.median(max_dev_ratio)) / np.median(max_dev_ratio)),
        "second_diff": _series(d2),
        "second_diff_background": background,
        "second_diff_max_abs_state": worst,
        "second_diff_max_abs": float(d2[worst]),
        "second_diff_max_over_background": float(ratios[worst]),
        "second_diff_max_abs_dperp_snr_float32": snr_at(worst),
        "second_diff_top3_states": top,
        "dperp_snr_float32": dev_snr,
        "dperp_snr_float32_at_landmarks": {str(n): snr_at(n) for n in FLOOR_DEV_STATES},
        "kinks": kinks,
        "kink_present": bool(kinks),
        "kink_rule": "|second difference| >= 5x the median |second difference| over states "
                     "2..48 AND >= 3x both neighbours (an isolated spike, not a steep "
                     "smooth stretch). Blind spot, stated so P9 can re-judge from the full "
                     "`second_diff` series dumped above: a one-state spike of ANY height h "
                     "has second difference (h, -2h, h), so its ratio to its own neighbours "
                     "is exactly 2 and it never clears the 3x test — this rule fires on a "
                     "slope break, not on an isolated single-state bump",
        "kink_t3_crosscheck": "not run here: confirming a kink on latents is a T3 reading "
                              "and belongs to latent_paths.py; on the bf16 T3 store states "
                              f"below {floor['dperp_first_readable_state']['bfloat16']} are "
                              "under the floor, so an early kink could not be confirmed there",
        "floor_row": "float32 (T1 in-flight rows)",
        "floor_note": (f"d_perp readable from state "
                       f"{floor['dperp_first_readable_state']['float32']} on the float32 rows "
                       f"(P1 SNR {dev_snr[1]:.2e} at state 1); "
                       f"a kink checked on the bf16 T3 store instead would be "
                       f"unconfirmable at states < "
                       f"{floor['dperp_first_readable_state']['bfloat16']}"),
        "end_state_residue_max": end_residue,
    }


# ---------------------------------------------------------------------------
# 3.2 how big the state is
# ---------------------------------------------------------------------------


def magnitude_readings(mag_raw: np.ndarray, dim: float, sigmas: np.ndarray,
                       floor: dict[str, Any], z_T_dtype: str) -> dict[str, Any]:
    """`magnitude/sqrt(d)` profile plus the analytic sigma-interpolation check."""
    sqrt_d = math.sqrt(dim)
    mag = mag_raw / sqrt_d
    med = np.median(mag, axis=0)
    cv = _cv(mag)
    final = mag[:, -1]
    pred = np.sqrt((1.0 - sigmas[None, :]) ** 2 * final[:, None] ** 2 + sigmas[None, :] ** 2)
    dev = np.abs(mag - pred)
    flat = int(np.argmax(dev))
    return {
        "profile_median": _series(med),
        "profile_q25": _series(np.quantile(mag, 0.25, axis=0)),
        "profile_q75": _series(np.quantile(mag, 0.75, axis=0)),
        "start_over_sqrt_d": float(med[0]),
        "start_raw_median": float(np.median(mag_raw[:, 0])),
        "trough": float(med.min()),
        "trough_state": int(np.argmin(med)),
        "end_over_sqrt_d": float(med[-1]),
        "cv_state0": float(cv[0]),
        "cv_state50": float(cv[-1]),
        "cv_states_0_19_upper_bound": float(cv[:20].max()),
        "cv_states_0_19_argmax": int(np.argmax(cv[:20])),
        "cv_profile": _series(cv),
        "analytic_check": {
            "formula": "||Z[n]||/sqrt(d) ~ sqrt((1-sigma_n)^2 (||Z[50]||/sqrt(d))^2 + sigma_n^2)",
            # every reading below is over the (record, state) population, NOT over
            # the median profile: `max_abs_deviation` is the worst state of the
            # worst single trajectory, which is strictly larger than the median
            # profile's own distance from the analytic curve
            "population": "record x state (n_checked = n records x 51 states)",
            "max_abs_deviation": float(dev.max()),
            "max_abs_deviation_state": int(flat % dev.shape[1]),
            "max_rel_deviation": float((dev / np.maximum(pred, 1e-30)).max()),
            "median_abs_deviation": float(np.median(dev)),
            "n_checked": int(dev.size),
        },
        "landmarks": _landmarks(med),
        "sqrt_d": sqrt_d,
        "z_T_dtype": z_T_dtype,
        "floor_row": "float32 (T1 in-flight rows)",
        "floor_note": (f"one-rounding relative shift on the state norm is "
                       f"{floor['magnitude_one_rounding_rel_shift']['float32']:.1e} — "
                       f"irrelevant at 4 dp; z_T dtype is {z_T_dtype}"
                       + (", so state 0 is asserted to 4 significant digits only"
                          if z_T_dtype == "float16" else "")),
    }


# ---------------------------------------------------------------------------
# 3.3 how far one step moves
# ---------------------------------------------------------------------------


def spacing_readings(spacing: np.ndarray, chord: np.ndarray,
                     floor: dict[str, Any]) -> dict[str, Any]:
    """`spacing/chord` profile plus the floor-deducted upper bound.

    The deduction is PER STEP (plan section 3.3 / 3.4): the P1 dump tabulates
    the displacement bias b[n] for every one of the 50 steps, and dividing the
    whole profile by one constant instead would make the deducted row an
    algebraic copy of the raw one. On the float32 rows b runs 5.7e-8 -> 4.8e-10
    so the correction is invisible at every printed digit, but this is the code
    path a bf16/T3 recomputation would take, where b runs 0.31 -> 2.8e-4.
    """
    med = np.median(spacing, axis=0)
    raw = spacing * chord[:, None]  # back to absolute, for the path-share readings
    cumulative = np.cumsum(raw, axis=1) / raw.sum(axis=1, keepdims=True)
    bias = np.asarray(floor["spacing_rel_bias"]["float32"], dtype=np.float64)
    if bias.shape[0] != med.shape[0]:
        raise ValueError(f"P1 spacing bias has {bias.shape[0]} steps, profile has "
                         f"{med.shape[0]}")
    bias_med = float(floor["spacing_rel_bias_median"]["float32"])
    deducted = med / (1.0 + bias)                     # per step, the plan's row
    deducted_const = med / (1.0 + bias_med)           # the all-step-median footnote
    bias_bf16 = np.asarray(floor["spacing_rel_bias"]["bfloat16"], dtype=np.float64)
    return {
        "profile_median": _series(med),
        "profile_q25": _series(np.quantile(spacing, 0.25, axis=0)),
        "profile_q75": _series(np.quantile(spacing, 0.75, axis=0)),
        "first": float(med[0]),
        "last": float(med[-1]),
        "min": float(med.min()), "min_step": int(np.argmin(med)),
        "max": float(med.max()), "max_step": int(np.argmax(med)),
        "max_over_min": float(med.max() / med.min()),
        "path_share_first_10_steps": float(np.median(cumulative[:, 9])),
        "path_share_last_10_steps": float(np.median(1.0 - cumulative[:, -11])),
        "steps_to_half_path": float(np.median((cumulative >= 0.5).argmax(axis=1) + 1)),
        "landmarks": {str(n): float(med[n]) for n in (0, 1, 2, 5, 10, 20, 30, 40, 45, 48, 49)},
        "floor_row": "float32 (T1 in-flight rows)",
        "floor_deducted": {
            "bias_per_step": [float(v) for v in bias],
            "bias_all_step_median": bias_med,
            "bias_largest_step": float(bias.max()),
            "bias_largest_at_step": int(np.argmax(bias)),
            "upper_bound_profile": _series(deducted),
            "max_abs_change_vs_raw": float(np.abs(deducted - med).max()),
            "max_abs_change_vs_raw_constant_bias": float(np.abs(deducted_const - med).max()),
            "note": ("deducted per step with the P1 float32 displacement bias "
                     "(+%.1e at step 0 down to +%.1e at step 49, all-step median +%.1e), so "
                     "the floor-deducted upper bound differs from the raw profile by at most "
                     "%.1e — nothing at any printed digit. The same per-step deduction on the "
                     "bf16 column (+%.3f at step 0 down to +%.1e at step 49) is what a "
                     "recomputation of this profile from the T3 store would need; deducting "
                     "one constant instead would be a %.0f %% error there."
                     % (bias[0], bias[-1], bias_med, float(np.abs(deducted - med).max()),
                        bias_bf16[0], bias_bf16[-1],
                        100.0 * abs(bias_bf16[0] - float(np.median(bias_bf16)))
                        / (1.0 + bias_bf16[0]))),
        },
        "qualifier": "a large displacement is NOT a large cost of skipping that step",
    }


# ---------------------------------------------------------------------------
# 3.4 sampler vs model
# ---------------------------------------------------------------------------


def _segment_rows(spacing_raw: np.ndarray, velocity_raw: np.ndarray, dsigma: np.ndarray,
                  steps: tuple[int, ...], floor: dict[str, Any]) -> list[dict[str, Any]]:
    """`displacement ratio = dsigma ratio x velocity ratio` between step pairs.

    Each trajectory satisfies it exactly, and the median commutes with the
    constant dsigma factor, so the closure residual below is a check on the
    stored fields, not an approximation.
    """
    bias = np.asarray(floor["spacing_rel_bias"]["float32"], dtype=np.float64)
    rows = []
    pairs = [(steps[i], steps[j]) for i in range(len(steps)) for j in range(i + 1, len(steps))]
    for a, b in pairs:
        disp = float(np.median(spacing_raw[:, b] / spacing_raw[:, a]))
        vel = float(np.median(velocity_raw[:, b] / velocity_raw[:, a]))
        dsr = float(dsigma[b] / dsigma[a])
        floor_rel = float(bias[a] + bias[b])
        rows.append({
            "from_step": a, "to_step": b,
            "displacement_ratio": disp,
            "dsigma_ratio": dsr,
            "velocity_ratio": vel,
            "closure_residual": float(disp - dsr * vel),
            "floor_rel": floor_rel,
            "readable": bool(abs(disp - 1.0) >= 3.0 * floor_rel),
        })
    return rows


def velocity_readings(velocity_raw: np.ndarray, spacing_raw: np.ndarray, dim: float,
                      dsigma: np.ndarray, backbone: str, floor: dict[str, Any]) -> dict[str, Any]:
    """`velocity_norm/sqrt(d)` profile, its spread, and the segment closure.

    Same six numbers `full_trajectory_analysis.velocity_law` reports, computed
    on the array this function already holds (that function takes a DataFrame
    column, and P4's stated toolset is numpy/json/matplotlib).
    """
    sqrt_d = math.sqrt(dim)
    vel = velocity_raw / sqrt_d
    med = np.median(vel, axis=0)
    per = (vel.max(axis=1) - vel.min(axis=1)) / vel.mean(axis=1)
    per_tail = (vel[:, 1:].max(axis=1) - vel[:, 1:].min(axis=1)) / vel[:, 1:].mean(axis=1)
    # a per-step multiplicative bias b[n] on the displacement carries into the
    # velocity unchanged (dsigma is exact), so the deducted profile is
    # med[n]/(1+b[n]) — per step, not one constant, or the deducted spread
    # would be the raw spread by construction
    bias = np.asarray(floor["spacing_rel_bias"]["float32"], dtype=np.float64)
    if bias.shape[0] != med.shape[0]:
        raise ValueError(f"P1 spacing bias has {bias.shape[0]} steps, velocity profile has "
                         f"{med.shape[0]}")
    bias_med = float(floor["spacing_rel_bias_median"]["float32"])
    ded = med / (1.0 + bias)
    identity = np.abs(spacing_raw - velocity_raw * dsigma[None, :]) / np.maximum(spacing_raw, 1e-30)
    return {
        "profile_median": _series(med),
        "profile_q25": _series(np.quantile(vel, 0.25, axis=0)),
        "profile_q75": _series(np.quantile(vel, 0.75, axis=0)),
        "profile_min": float(med.min()), "profile_max": float(med.max()),
        "profile_min_step": int(np.argmin(med)), "profile_max_step": int(np.argmax(med)),
        "profile_spread": float((med.max() - med.min()) / med.mean()),
        "profile_spread_floor_deducted": float((ded.max() - ded.min()) / ded.mean()),
        "per_trajectory_spread_median": float(np.median(per)),
        "per_trajectory_spread_median_no_step0": float(np.median(per_tail)),
        "cross_prompt_cv_median": float(np.median(vel.std(axis=0) / vel.mean(axis=0))),
        "landmarks": {str(n): float(med[n]) for n in (0, 1, 2, 5, 10, 20, 30, 40, 45, 48, 49)},
        "identity_check": {
            "formula": "spacing[n] = velocity_norm[n] * |dsigma_n|",
            "max_rel_deviation": float(identity.max()),
            "median_rel_deviation": float(np.median(identity)),
        },
        "segments": {
            "first_0_to_2": _segment_rows(spacing_raw, velocity_raw, dsigma, (0, 1, 2), floor),
            "last_47_to_49": _segment_rows(spacing_raw, velocity_raw, dsigma, (47, 48, 49), floor),
        },
        "sampler_note": SAMPLER_NOTE[backbone],
        "floor_row": "float32 (T1 in-flight rows)",
        "floor_note": (f"deducted step by step with the P1 float32 displacement bias "
                       f"(+{bias[0]:.1e} at step 0 down to +{bias[-1]:.1e} at step 49, "
                       f"all-step median +{bias_med:.1e}); both backbones share the "
                       f"last-segment caliber because the largest |dsigma| is step 49"),
    }


# ---------------------------------------------------------------------------
# 3.5 how much of the geometry is common to every prompt
# ---------------------------------------------------------------------------


def dispersion_readings(profiles: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-step cross-(prompt, seed) CV of the four normalised profiles."""
    windows = {"dev_over_chord": (1, 49), "spacing_over_chord": (0, 49),
               "magnitude_over_sqrt_d": (0, 50), "velocity_over_sqrt_d": (0, 49)}
    out: dict[str, Any] = {}
    for name, (lo, hi) in windows.items():
        cv = _cv(profiles[name])
        full = np.full(profiles[name].shape[1], np.nan)
        full[lo:hi + 1] = cv[lo:hi + 1]
        out[name] = {"window": [lo, hi], "cv_profile": _series(full),
                     **_cv_readings(full, lo, hi)}
    return out


def _at_states(share: np.ndarray) -> dict[str, float | None]:
    return {str(n): (None if not np.isfinite(share[n]) else float(share[n]))
            for n in SHARE_STATES}


def prompt_share(dev: np.ndarray, prompt_idx: np.ndarray, *, n_seeds: int = 3
                 ) -> dict[str, Any]:
    """Between-prompt share of the `d_perp/chord` variance, one dataset.

    `group_variance_share` with `min_size = n_seeds`: the plan's design gives
    every prompt exactly 3 seeds, and a prompt that is short one generation is
    excluded and counted rather than aborting the run.
    """
    got = group_variance_share(dev, prompt_idx.astype(np.int64), min_size=n_seeds)
    share = got["share"].copy()
    share[[0, -1]] = np.nan   # both chord endpoints: a float64 residue, not a measurement
    return {
        "share_profile": _series(share),
        "at_states": _at_states(share),
        "groups": got["groups_used"], "groups_total": got["groups_total"],
        "rows": got["rows_used"],
        "group_size_histogram": got["group_size_histogram"],
        "excluded_groups_below_min_size": got["excluded_groups_below_min_size"],
        "min_size": n_seeds,
        "denominator": "prompts (not records)",
    }


def noise_share(dev: np.ndarray, z_T: np.ndarray) -> dict[str, Any]:
    """Between-noise share, grouped by `z_T_sha256` across BOTH datasets.

    Reported beside the prompt share as a second one-factor share, never as
    part of a three-way decomposition: the design is not fully crossed (each
    prompt has 3 noises, each noise <= 6 prompts, adjacent prompts share 2 of
    3 seeds).
    """
    got = group_variance_share(dev, np.asarray(z_T, dtype=object), min_size=2)
    share = got["share"].copy()
    share[[0, -1]] = np.nan
    return {
        "share_profile": _series(share),
        "at_states": _at_states(share),
        "groups": got["groups_used"], "groups_total": got["groups_total"],
        "rows": got["rows_used"],
        "group_size_histogram": got["group_size_histogram"],
        "excluded_groups_below_min_size": got["excluded_groups_below_min_size"],
        "grouping": "z_T_sha256 equality across both datasets (never seed arithmetic)",
        "denominator": "noise tensors (not records)",
        "not_a_three_way_decomposition":
            "two one-factor shares side by side; the design is not fully crossed",
    }


# ---------------------------------------------------------------------------
# 3.6 whole-trajectory shape scalars
# ---------------------------------------------------------------------------


def shape_rows(refs: References) -> list[dict[str, Any]]:
    """One row per (dataset, base_seed) — six per backbone."""
    c = refs.cols
    rows = []
    for dataset, seed in refs.streams():
        m = refs.mask(dataset, seed)
        evr = c["pca_evr"][m]
        final = c["magnitude"][m][:, -1]
        dev_abs = c["max_dev_ratio"][m] * c["chord_len"][m]
        top2 = evr[:, 0] + evr[:, 1]
        rows.append({
            "dataset": dataset, "base_seed": int(seed), "n": int(m.sum()),
            "chord_len_median": float(np.median(c["chord_len"][m])),
            "straightness_median": float(np.median(c["straightness"][m])),
            "max_dev_ratio_median": float(np.median(c["max_dev_ratio"][m])),
            "max_dev_ratio_q05": float(np.quantile(c["max_dev_ratio"][m], 0.05)),
            "max_dev_ratio_q95": float(np.quantile(c["max_dev_ratio"][m], 0.95)),
            "max_dev_over_final_norm_median": float(np.median(dev_abs / final)),
            "top2_evr_median": float(np.median(top2)),
            "top2_evr_q05": float(np.quantile(top2, 0.05)),
        })
    return rows


def evr_spectrum(refs: References, floor: dict[str, Any]) -> dict[str, Any]:
    c = refs.cols
    out: dict[str, Any] = {"by_dataset": {}}
    for dataset in refs.datasets():
        m = refs.mask(dataset)
        evr = c["pca_evr"][m]
        top2, top3 = evr[:, 0] + evr[:, 1], evr[:, :3].sum(axis=1)
        r2, r3 = c["recon_err_rel_2d"][m], c["recon_err_rel_3d"][m]
        out["by_dataset"][dataset] = {
            "n": int(m.sum()),
            "evr_median": [float(np.median(evr[:, i])) for i in range(evr.shape[1])],
            "top2_cumulative_median": float(np.median(top2)),
            "top3_cumulative_median": float(np.median(top3)),
            "recon_err_rel_2d_median": float(np.median(r2)),
            "recon_err_rel_3d_median": float(np.median(r3)),
            # residual^2 = 1 - cumulative share (k-D = chord + (k-1) PCs)
            "residual_check_2d_max_abs": float(np.abs(r2 ** 2 - (1.0 - evr[:, 0])).max()),
            "residual_check_3d_max_abs": float(np.abs(r3 ** 2 - (1.0 - top2)).max()),
        }
    out["residual_check_formula"] = "recon_err_rel_kd^2 = 1 - sum(pca_evr[:k-1])"
    out["floor_row"] = (
        "float32 (T1 in-flight rows); on the bf16 T3 store the same straightness "
        "carries a %+.1e inflation and off-plane energy sits only %.0fx above the "
        "floor" % (floor["straightness_rel_bias"]["bfloat16"],
                   floor["offplane_energy_over_floor"]["bfloat16"]))
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def analyse(refs: References, backbone: str, sigmas: np.ndarray,
            floor: dict[str, Any]) -> dict[str, Any]:
    c = refs.cols
    dsigma = np.abs(np.diff(sigmas))
    z_T_dtypes = {k.split("z_T=")[1].split("/")[0] for k in refs.meta["row_dtypes"]}
    z_T_dtype = sorted(z_T_dtypes)[0] if len(z_T_dtypes) == 1 else "mixed:" + \
        ",".join(sorted(z_T_dtypes))
    dim = float(np.median(c["d"]))
    if not np.all(c["d"] == dim):
        raise SystemExit("mixed latent dimensions in one backbone's references")

    out: dict[str, Any] = {"datasets": {}}
    for dataset in refs.datasets():
        m = refs.mask(dataset)
        chord = c["chord_len"][m]
        dev = c["d_perp"][m] / chord[:, None]
        spacing_raw = c["spacing"][m]
        spacing = spacing_raw / chord[:, None]
        magnitude = c["magnitude"][m] / math.sqrt(dim)
        velocity_raw = c["velocity_norm"][m]
        velocity = velocity_raw / math.sqrt(dim)
        entry: dict[str, Any] = {
            "n": int(m.sum()),
            "n_prompts": int(len(np.unique(c["prompt_idx"][m]))),
            "n_seeds": int(len(np.unique(c["base_seed"][m]))),
            "streams": [f"{d}_s{s}" for d, s in refs.streams() if d == dataset],
            "dev": deviation_readings(dev, sigmas, c["max_dev_ratio"][m], floor),
            "magnitude": magnitude_readings(c["magnitude"][m], dim, sigmas, floor, z_T_dtype),
            "spacing": spacing_readings(spacing, chord, floor),
            "velocity": velocity_readings(velocity_raw, spacing_raw, dim, dsigma,
                                          backbone, floor),
        }
        entry["cv"] = dispersion_readings({
            "dev_over_chord": dev, "spacing_over_chord": spacing,
            "magnitude_over_sqrt_d": magnitude, "velocity_over_sqrt_d": velocity})
        entry["variance_share_prompt"] = prompt_share(dev, c["prompt_idx"][m])
        entry["profiles"] = {
            "dev_over_chord": entry["dev"]["profile_median"],
            "dev_over_chord_q25": entry["dev"]["profile_q25"],
            "dev_over_chord_q75": entry["dev"]["profile_q75"],
            "spacing_over_chord": entry["spacing"]["profile_median"],
            "spacing_over_chord_q25": entry["spacing"]["profile_q25"],
            "spacing_over_chord_q75": entry["spacing"]["profile_q75"],
            "magnitude_over_sqrt_d": entry["magnitude"]["profile_median"],
            "magnitude_over_sqrt_d_q25": entry["magnitude"]["profile_q25"],
            "magnitude_over_sqrt_d_q75": entry["magnitude"]["profile_q75"],
            "velocity_over_sqrt_d": entry["velocity"]["profile_median"],
            "velocity_over_sqrt_d_q25": entry["velocity"]["profile_q25"],
            "velocity_over_sqrt_d_q75": entry["velocity"]["profile_q75"],
            "sigmas": _series(sigmas),
        }
        out["datasets"][dataset] = entry

    # the between-noise share is pooled over BOTH datasets by construction
    dev_all = c["d_perp"] / c["chord_len"][:, None]
    out["variance_share_noise_pooled"] = noise_share(dev_all, c["z_T_sha256"])
    out["estimation_boundaries"] = ESTIMATION_BOUNDARIES
    out["shape_by_stream"] = shape_rows(refs)
    out["evr_spectrum"] = evr_spectrum(refs, floor)
    out["z_T_dtype"] = z_T_dtype
    out["sigmas"] = _series(sigmas)
    out["dsigma"] = _series(dsigma)
    out["dim"] = dim
    out["sqrt_dim"] = math.sqrt(dim)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    lines = ["\t".join(header)]
    lines += ["\t".join("" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
                        for v in row) for row in rows]
    atomic_write_text(path, "\n".join(lines) + "\n")


def _md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def fmt(v: Any) -> str:
        if v is None:
            return "—"
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    return ["| " + " | ".join(header) + " |",
            "|" + "---|" * len(header)] + \
           ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]


def write_tables(report: dict[str, Any], backbone: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    md: list[str] = [f"# Per-step profiles and shape scalars — {backbone}", "",
                     f"Source `{report['source']['merged']}`; "
                     f"{report['source']['references_kept']:,} clean references "
                     f"({report['source']['cell_rows_discarded_in_reader']:,} cell rows "
                     f"discarded in the reader), row dtypes "
                     f"{report['source']['row_dtypes']}. Every reading below is a "
                     f"**float32-row** reading; the P1 floors are read from "
                     f"`{report['floor']['source']['float32']}` "
                     f"({report['floor']['n_trajectories']['float32']} paths).", ""]
    ds_names = list(report["datasets"])

    # 3.1 -------------------------------------------------------------------
    header = ["dataset", "n", "peak_state", "peak", "sigma_at_peak", "half_first",
              "half_last", "half_width_states_inclusive", "plateau90_first",
              "plateau90_last",
              "per_traj_peak_mode", "mode_share", "within_2_share",
              "peak_state_q05", "peak_state_q95",
              "max_dev_ratio_median", "peak_vs_max_dev_rel_diff"]
    rows = []
    for ds in ds_names:
        d = report["datasets"][ds]["dev"]
        rows.append([ds, report["datasets"][ds]["n"], d["peak_state"], d["peak"],
                     d["sigma_at_peak"], d["half_first_state"], d["half_last_state"],
                     d["half_width_states_inclusive"], d["plateau90_first_state"],
                     d["plateau90_last_state"], d["per_trajectory_peak_mode"],
                     d["per_trajectory_peak_mode_share"], d["per_trajectory_peak_within_2_share"],
                     d["per_trajectory_peak_state"]["q05"], d["per_trajectory_peak_state"]["q95"],
                     d["max_dev_ratio_median"], d["peak_vs_max_dev_ratio_rel_diff"]])
    p = out_dir / f"step_dev_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.1 distance from the straight line (`d_perp/chord`, states 1-49)", "",
           "`half_width_states_inclusive` COUNTS the states at or above half the peak, "
           "endpoints included — it is `half_last - half_first + 1`, not the span "
           "`half_last - half_first`. A run of states 24..49 is 26 states, width 25.", ""]
    md += _md_table(header, rows) + [""]
    krows = []
    for ds in ds_names:
        d = report["datasets"][ds]["dev"]
        krows.append([ds, d["kink_present"], d["second_diff_max_abs_state"],
                      d["second_diff_max_abs"], d["second_diff_max_over_background"],
                      d["second_diff_max_abs_dperp_snr_float32"], len(d["kinks"])])
        for k in d["kinks"]:
            krows.append([f"{ds}: kink", True, k["state"], k["second_diff"],
                          k["over_background"], k["dperp_snr_float32"], 1])
    kheader = ["dataset", "kink_present", "state", "second_diff", "over_background",
               "dperp_snr_float32", "n_kinks"]
    p = out_dir / f"step_dev_kink_{backbone}.tsv"
    _tsv(p, kheader, krows); written.append(p)
    md += ["Kink search on the median profile. `over_background` is |second difference| "
           "divided by its own median over states 2..48 (a shape statistic of this "
           "profile); `dperp_snr_float32` is the instrument SNR of the deviation reading "
           "at that state, from the P1 float32 dump. Rule: "
           f"{report['datasets'][ds_names[0]]['dev']['kink_rule']}", ""]
    md += _md_table(kheader, krows) + [""]

    # 3.2 -------------------------------------------------------------------
    header = ["dataset", "start_over_sqrt_d", "start_raw_median", "trough", "trough_state",
              "end_over_sqrt_d", "cv_state0", "cv_state50", "cv_states_0_19_upper_bound",
              "analytic_max_abs_dev_worst_record", "analytic_max_rel_dev_worst_record",
              "analytic_median_abs_dev", "analytic_n_checked"]
    rows = []
    for ds in ds_names:
        m = report["datasets"][ds]["magnitude"]
        rows.append([ds, m["start_over_sqrt_d"], m["start_raw_median"], m["trough"],
                     m["trough_state"], m["end_over_sqrt_d"], m["cv_state0"], m["cv_state50"],
                     m["cv_states_0_19_upper_bound"], m["analytic_check"]["max_abs_deviation"],
                     m["analytic_check"]["max_rel_deviation"],
                     m["analytic_check"]["median_abs_deviation"],
                     m["analytic_check"]["n_checked"]])
    p = out_dir / f"step_magnitude_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.2 size of the state (`magnitude/sqrt(d)`, states 0-50)", "",
           "The two `analytic_*_dev_worst_record` columns are maxima over EVERY "
           "(record, state) pair — `analytic_n_checked` = n records x 51 states — so "
           "each is the single worst state of the single worst trajectory, NOT the "
           "deviation of the median profile from the analytic curve. The median profile "
           "sits closer: `analytic_median_abs_dev` beside them is the median over the "
           "same (record, state) population, and re-reading the check on the median "
           "profile alone gives a smaller number again. Quote the worst-record columns "
           "as a per-trajectory upper bound, never as \"the profile deviates by this "
           "much\".", ""]
    md += _md_table(header, rows) + [""]

    # 3.3 -------------------------------------------------------------------
    header = ["dataset", "first", "last", "min", "min_step", "max", "max_step", "max_over_min",
              "path_share_first_10", "path_share_last_10", "steps_to_half_path",
              "floor_deducted_max_change"]
    rows = []
    for ds in ds_names:
        s = report["datasets"][ds]["spacing"]
        rows.append([ds, s["first"], s["last"], s["min"], s["min_step"], s["max"], s["max_step"],
                     s["max_over_min"], s["path_share_first_10_steps"],
                     s["path_share_last_10_steps"], s["steps_to_half_path"],
                     s["floor_deducted"]["max_abs_change_vs_raw"]])
    p = out_dir / f"step_spacing_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.3 distance moved by one step (`spacing/chord`, steps 0-49)", "",
           "**Qualifier (verbatim): a large displacement is NOT a large cost of skipping "
           "that step.**", ""]
    md += _md_table(header, rows) + [""]
    md += [report["datasets"][ds_names[0]]["spacing"]["floor_deducted"]["note"], ""]

    # 3.4 -------------------------------------------------------------------
    header = ["dataset", "profile_min", "min_step", "profile_max", "max_step", "spread",
              "spread_floor_deducted", "per_traj_spread_med", "per_traj_spread_med_no_step0",
              "cross_prompt_cv_med", "identity_max_rel_dev"]
    rows = []
    for ds in ds_names:
        v = report["datasets"][ds]["velocity"]
        rows.append([ds, v["profile_min"], v["profile_min_step"], v["profile_max"],
                     v["profile_max_step"], v["profile_spread"], v["profile_spread_floor_deducted"],
                     v["per_trajectory_spread_median"], v["per_trajectory_spread_median_no_step0"],
                     v["cross_prompt_cv_median"], v["identity_check"]["max_rel_deviation"]])
    p = out_dir / f"step_velocity_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.4 sampler vs model (`spacing = velocity_norm x |dsigma|`)", "",
           report["datasets"][ds_names[0]]["velocity"]["sampler_note"], ""]
    md += _md_table(header, rows) + [""]

    sheader = ["dataset", "segment", "from_step", "to_step", "displacement_ratio",
               "dsigma_ratio", "velocity_ratio", "closure_residual", "floor_rel", "readable"]
    srows = []
    for ds in ds_names:
        for seg, entries in report["datasets"][ds]["velocity"]["segments"].items():
            for e in entries:
                srows.append([ds, seg, e["from_step"], e["to_step"], e["displacement_ratio"],
                              e["dsigma_ratio"], e["velocity_ratio"], e["closure_residual"],
                              e["floor_rel"], e["readable"]])
    p = out_dir / f"step_velocity_segments_{backbone}.tsv"
    _tsv(p, sheader, srows); written.append(p)
    md += ["First / last segment decomposition (displacement ratio = dsigma ratio x "
           "velocity ratio):", ""]
    md += _md_table(sheader, srows) + [""]

    # 3.5 -------------------------------------------------------------------
    header = ["dataset", "profile", "window", "cv_median", "cv_min", "cv_min_at",
              "cv_max", "cv_max_at"]
    rows = []
    for ds in ds_names:
        for name, cv in report["datasets"][ds]["cv"].items():
            rows.append([ds, name, f"{cv['window'][0]}-{cv['window'][1]}", cv["median"],
                         cv["min"], cv["min_step"], cv["max"], cv["max_step"]])
    p = out_dir / f"cross_prompt_cv_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.5 how much of the geometry is common across prompts", "",
           "Per-step CV across (prompt, seed), then the median over steps. One run of "
           "this script sees one backbone; the per-profile backbone ordering the plan "
           "asks for, and the 8-line F2 figure (4 profiles x 2 backbones, per dataset), "
           "come from `--combine` once both backbones' JSONs exist.", ""]
    md += _md_table(header, rows) + [""]

    header = ["factor", "scope", "groups", "rows", "denominator"] + \
             [f"state_{n}" for n in SHARE_STATES]
    rows = []
    for ds in ds_names:
        s = report["datasets"][ds]["variance_share_prompt"]
        rows.append(["between prompts", ds, s["groups"], s["rows"], s["denominator"]] +
                    [s["at_states"][str(n)] for n in SHARE_STATES])
    ns = report["variance_share_noise_pooled"]
    rows.append(["between noises", "both datasets pooled", ns["groups"], ns["rows"],
                 ns["denominator"]] + [ns["at_states"][str(n)] for n in SHARE_STATES])
    p = out_dir / f"variance_share_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["Two one-factor shares of the `d_perp/chord` variance (NOT a three-way "
           "decomposition):", ""]
    md += _md_table(header, rows) + [""]
    md += [f"Noise groups: {ns['groups']} of {ns['groups_total']} used "
           f"(size histogram {ns['group_size_histogram']}, "
           f"{ns['excluded_groups_below_min_size']} singleton groups excluded); "
           f"grouping = {ns['grouping']}."]
    for ds in ds_names:
        s = report["datasets"][ds]["variance_share_prompt"]
        md += [f"Prompt groups, {ds}: {s['groups']} of {s['groups_total']} used "
               f"(size histogram {s['group_size_histogram']}, "
               f"{s['excluded_groups_below_min_size']} below the {s['min_size']}-seed "
               f"minimum)."]
    md += [""]
    md += ["Estimation boundaries:", ""] + [f"- {b}" for b in ESTIMATION_BOUNDARIES] + [""]

    # 3.6 -------------------------------------------------------------------
    header = ["dataset", "base_seed", "n", "chord_len_median", "straightness_median",
              "max_dev_ratio_median", "max_dev_ratio_q05", "max_dev_ratio_q95",
              "max_dev_over_final_norm_median", "top2_evr_median", "top2_evr_q05"]
    rows = [[r["dataset"], r["base_seed"], r["n"], r["chord_len_median"],
             r["straightness_median"], r["max_dev_ratio_median"], r["max_dev_ratio_q05"],
             r["max_dev_ratio_q95"], r["max_dev_over_final_norm_median"],
             r["top2_evr_median"], r["top2_evr_q05"]] for r in report["shape_by_stream"]]
    p = out_dir / f"shape_summary_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.6 whole-trajectory shape scalars (one row per stream)", ""]
    md += _md_table(header, rows) + [""]

    header = ["dataset", "n", "evr1", "evr2", "evr3", "evr4", "evr5", "top2_cum", "top3_cum",
              "recon_err_rel_2d", "recon_err_rel_3d", "residual_check_2d_max",
              "residual_check_3d_max"]
    rows = []
    for ds, e in report["evr_spectrum"]["by_dataset"].items():
        rows.append([ds, e["n"], *e["evr_median"], e["top2_cumulative_median"],
                     e["top3_cumulative_median"], e["recon_err_rel_2d_median"],
                     e["recon_err_rel_3d_median"], e["residual_check_2d_max_abs"],
                     e["residual_check_3d_max_abs"]])
    p = out_dir / f"evr_spectrum_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["Chord-orthogonal PCA spectrum "
           f"(check: {report['evr_spectrum']['residual_check_formula']}):", ""]
    md += _md_table(header, rows) + [""]

    p = out_dir / f"step_profiles_{backbone}.md"
    atomic_write_text(p, "\n".join(md) + "\n")
    written.append(p)
    print("\n".join(md))
    return written


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _sigma_axis(ax, sigmas: np.ndarray, n: int) -> None:
    """Second x axis carrying sigma, because sigma is NOT uniform in n
    (plan section 2.4: the last step spans 43x / 23x the first)."""
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = [t for t in (0, 10, 20, 30, 40, 45, 49) if t < n]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{sigmas[t]:.2f}" for t in ticks], fontsize=7.5)
    top.set_xlabel("sigma at that index (non-uniform grid)", fontsize=8)


def write_f1(report: dict[str, Any], backbone: str, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = np.array([v for v in report["sigmas"]], dtype=float)
    panels = [
        ("dev_over_chord", "distance from the straight line\n(fraction of the chord)",
         "state n (0 = the initial noise)"),
        ("spacing_over_chord", "distance moved by one step\n(fraction of the chord)",
         "solver step n (state n -> n+1)"),
        ("velocity_over_sqrt_d", "speed of the path\n(velocity norm / sqrt(d))",
         "solver step n (state n -> n+1)"),
        ("magnitude_over_sqrt_d", "size of the state\n(norm / sqrt(d))",
         "state n (0 = the initial noise)"),
    ]
    written = []
    for dataset, cell in report["datasets"].items():
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
        for ax, (key, ylabel, xlabel) in zip(axes, panels):
            med = np.array([np.nan if v is None else v for v in cell["profiles"][key]], float)
            q25 = np.array([np.nan if v is None else v
                            for v in cell["profiles"][f"{key}_q25"]], float)
            q75 = np.array([np.nan if v is None else v
                            for v in cell["profiles"][f"{key}_q75"]], float)
            x = np.arange(len(med))
            ax.plot(x, med, color="#1f77b4", lw=1.8)
            ax.fill_between(x, q25, q75, color="#1f77b4", alpha=0.15)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
            low, high = ax.get_ylim()          # clear a strip for the annotations
            ax.set_ylim(low - 0.16 * (high - low), high)
            cv = cell["cv"][key]["median"]
            ax.text(0.03, 0.04, f"cross-prompt CV, median over the window: {cv * 100:.1f}%",
                    transform=ax.transAxes, fontsize=8.5, color="#1f77b4")
            if key == "dev_over_chord":
                peak = cell["dev"]["peak_state"]
                ax.plot([peak], [med[peak]], "o", color="#1f77b4", ms=6)
                # in the corner, not beside the point: the twin sigma axis sits
                # directly above the peak and the two labels would overlap
                ax.text(0.03, 0.90, f"peak at state {peak} "
                        f"(sigma {cell['dev']['sigma_at_peak']:.3f})",
                        transform=ax.transAxes, fontsize=8.5, color="#1f77b4")
            _sigma_axis(ax, sigmas, len(med))
        fig.suptitle(
            f"{backbone} / {dataset}: what the clean denoising path does at each of the "
            f"50 steps\nline = median over {cell['n']:,} references "
            f"({cell['n_prompts']:,} prompts x {cell['n_seeds']} seeds), band = the middle "
            f"50 % at that index; float32-row reading", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        p = atomic_savefig(fig, out_dir / f"f1_step_geometry_{backbone}_{dataset}.png", dpi=140)
        plt.close(fig)
        written.append(p)
    return written


def f1_paths(report: dict[str, Any], backbone: str, out_dir: Path) -> list[Path]:
    """The F1 panels this report would produce — one per dataset, so the reuse
    check can name them (their file names depend on the data)."""
    return [out_dir / f"f1_step_geometry_{backbone}_{ds}.png" for ds in report["datasets"]]


def write_f2(report: dict[str, Any], backbone: str, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    share_min = 0.0
    keys = [("dev_over_chord", "distance from the line", "-"),
            ("spacing_over_chord", "distance moved by one step", "--"),
            ("magnitude_over_sqrt_d", "size of the state", ":"),
            ("velocity_over_sqrt_d", "speed of the path", "-.")]
    colors = {"penguin599": "#d62728", "vbench944": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for dataset, cell in report["datasets"].items():
        color = colors.get(dataset, "0.4")
        for key, name, dash in keys:
            y = np.array([np.nan if v is None else v
                          for v in cell["cv"][key]["cv_profile"]], float)
            axes[0].plot(np.arange(len(y)), y, color=color, ls=dash, lw=1.5,
                         label=f"{dataset}: {name}")
        share = np.array([np.nan if v is None else v
                          for v in cell["variance_share_prompt"]["share_profile"]], float)
        share_min = min(share_min, float(np.nanmin(share)))
        axes[1].plot(np.arange(len(share)), share, color=color, lw=1.8,
                     label=f"{dataset}: between prompts "
                           f"({cell['variance_share_prompt']['groups']:,} prompts x 3 seeds)")
    noise = np.array([np.nan if v is None else v
                      for v in report["variance_share_noise_pooled"]["share_profile"]], float)
    share_min = min(share_min, float(np.nanmin(noise)))
    axes[1].plot(np.arange(len(noise)), noise, color="0.25", ls="--", lw=1.8,
                 label=f"both datasets: between noises "
                       f"({report['variance_share_noise_pooled']['groups']:,} z_T groups)")
    axes[0].set_ylabel("spread across prompts and seeds at that index\n"
                       "(coefficient of variation)")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.7)
    axes[0].legend(fontsize=7, ncol=2, loc="upper center")
    axes[1].set_ylabel("share of the deviation variance that is\nbetween prompts / "
                       "between noises")
    # an unbiased share can come out slightly negative where the factor explains
    # nothing; clipping the axis at 0 would hide that, so the axis follows the data
    axes[1].set_ylim(min(0.0, share_min) - 0.03, 1.0)
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].legend(fontsize=7.5, loc="lower right")
    for ax in axes:
        ax.set_xlabel("state n (deviation / size) or solver step n (displacement / speed)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{backbone}: how much of the per-step geometry is common to every prompt\n"
                 "states 0 and 50 lie on the chord by construction, so the deviation curves "
                 "have nothing to report there; two one-factor shares, not a decomposition",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = atomic_savefig(fig, out_dir / f"f2_cross_prompt_{backbone}.png", dpi=140)
    plt.close(fig)
    return p


def write_f3a(refs: References, report: dict[str, Any], backbone: str, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = refs.cols
    panels = [("straightness", "path length / chord length\n(1.000 = perfectly straight)"),
              ("max_dev_ratio", "furthest distance from the line\n(fraction of the chord)"),
              ("top2_evr", "share of the deviation in its\nleading two directions"),
              ("recon_err_rel_3d", "residual after chord + 2 directions\n(relative to chord alone)")]
    values = {"straightness": c["straightness"], "max_dev_ratio": c["max_dev_ratio"],
              "top2_evr": c["pca_evr"][:, 0] + c["pca_evr"][:, 1],
              "recon_err_rel_3d": c["recon_err_rel_3d"]}
    datasets = refs.datasets()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for ax, (key, ylabel) in zip(axes, panels):
        for i, dataset in enumerate(datasets):
            m = refs.mask(dataset)
            ax.boxplot([values[key][m]], positions=[i], widths=0.4, showfliers=False,
                       medianprops=dict(color="#1f77b4", lw=1.8))
            for seed in sorted({int(s) for s in c["base_seed"][m]}):
                ms = refs.mask(dataset, seed)
                ax.plot([i], [np.median(values[key][ms])], "o", color="#d62728", ms=4,
                        mfc="white", mew=1.2, zorder=3)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_xlim(-0.6, len(datasets) - 0.4)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], color="#1f77b4", lw=2, label="all references of that dataset"),
               plt.Line2D([], [], color="#d62728", marker="o", ls="", ms=4, mfc="white",
                          mew=1.2, label="one base seed's median (3 per dataset)")]
    axes[0].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle(f"{backbone}: whole-trajectory shape over all {len(refs):,} clean references\n"
                 "box = spread over the individual trajectories, dots = the three base seeds' "
                 "medians; float32-row reading", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = atomic_savefig(fig, out_dir / f"f3a_shape_scalars_{backbone}.png", dpi=140)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# --combine: the cross-backbone half of plan section 3.5
# ---------------------------------------------------------------------------

CV_PROFILES = (("dev_over_chord", "distance from the line"),
               ("spacing_over_chord", "distance moved by one step"),
               ("magnitude_over_sqrt_d", "size of the state"),
               ("velocity_over_sqrt_d", "speed of the path"))


def combine_ordering(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per profile, per dataset: which backbone is the more dispersed one.

    Plan section 3.5 asks for the dispersion ordering to be stated PROFILE BY
    PROFILE ("which backbone is more spread on which profile, and where it is
    the other way round"), never as one blanket sentence, so this returns one
    row per (dataset, profile) with both medians and the named winner.
    """
    rows: list[dict[str, Any]] = []
    datasets = sorted({ds for rep in reports.values() for ds in rep["datasets"]})
    for dataset in datasets:
        for key, label in CV_PROFILES:
            got = {T: rep["datasets"][dataset]["cv"][key]["median"]
                   for T, rep in reports.items() if dataset in rep["datasets"]}
            if len(got) < 2:
                rows.append({"dataset": dataset, "profile": key, "label": label,
                             "more_dispersed": None, "note": "only one backbone has "
                             f"this dataset ({sorted(got)})", **got})
                continue
            order = sorted(got.items(), key=lambda kv: kv[1], reverse=True)
            (hi, hv), (lo, lv) = order[0], order[-1]
            rows.append({
                "dataset": dataset, "profile": key, "label": label, **got,
                "more_dispersed": hi, "ratio_hi_over_lo": float(hv / lv) if lv else None,
                "statement": f"on {dataset}, the {label} profile is more spread across "
                             f"prompts on {hi} (CV median {hv:.4g}) than on {lo} "
                             f"({lv:.4g}), a factor {hv / lv:.2f}" if lv else None,
            })
    return rows


def write_combined(reports: dict[str, dict[str, Any]], out_tables: Path, out_figs: Path,
                   *, no_figures: bool = False) -> list[Path]:
    """F2 as the plan defines it (8 lines = 4 profiles x 2 backbones, per
    dataset) plus the per-profile ordering table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    rows = combine_ordering(reports)
    header = ["dataset", "profile"] + sorted(reports) + ["more_dispersed", "ratio_hi_over_lo"]
    table = [[r["dataset"], r["profile"]] + [r.get(T) for T in sorted(reports)]
             + [r.get("more_dispersed"), r.get("ratio_hi_over_lo")] for r in rows]
    p = out_tables / "cross_prompt_cv_ordering.tsv"
    _tsv(p, header, table); written.append(p)

    md = ["# Cross-prompt dispersion, both backbones (plan section 3.5)", "",
          "Per-step CV across (prompt, seed), median over the profile's window; one row "
          "per (dataset, profile). The ordering is stated profile by profile — the plan "
          "does not accept a single blanket sentence about which backbone is more "
          "spread.", "",
          "Sources: " + ", ".join(
              f"`{T}` ({rep['source']['references_kept']:,} references, limit "
              f"{rep['source']['limit']})" for T, rep in sorted(reports.items())), ""]
    md += _md_table(header, table) + [""]
    md += [f"- {r['statement']}" for r in rows if r.get("statement")] + [""]
    p = out_tables / "cross_prompt_cv_ordering.md"
    atomic_write_text(p, "\n".join(md) + "\n"); written.append(p)
    print("\n".join(md))

    if no_figures:
        return written
    datasets = sorted({ds for rep in reports.values() for ds in rep["datasets"]})
    dashes = {"dev_over_chord": "-", "spacing_over_chord": "--",
              "magnitude_over_sqrt_d": ":", "velocity_over_sqrt_d": "-."}
    colors = {"hunyuan_video": "#d62728", "wan21": "#1f77b4"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 4.8),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        n_lines = 0
        for T, rep in sorted(reports.items()):
            cell = rep["datasets"].get(dataset)
            if cell is None:
                continue
            for key, label in CV_PROFILES:
                y = np.array([np.nan if v is None else v
                              for v in cell["cv"][key]["cv_profile"]], float)
                ax.plot(np.arange(len(y)), y, color=colors.get(T, "0.4"), ls=dashes[key],
                        lw=1.5, label=f"{T}: {label}")
                n_lines += 1
        ax.set_title(f"{dataset} ({n_lines} lines)", fontsize=10)
        ax.set_xlabel("state n (deviation / size) or solver step n (displacement / speed)")
        ax.set_ylabel("spread across prompts and seeds at that index\n"
                      "(coefficient of variation)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper center")
    fig.suptitle("How much of the per-step geometry is common to every prompt — both video "
                 "backbones\n4 profiles x 2 backbones per dataset; states 0 and 50 lie on "
                 "the chord by construction, so the deviation curves have nothing to "
                 "report there", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p = atomic_savefig(fig, out_figs / "f2_cross_prompt_both_backbones.png", dpi=140)
    plt.close(fig)
    written.append(p)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, default=None,
                    help="required unless --combine is given")
    ap.add_argument("--combine", action="store_true",
                    help="cross-backbone step: read both backbones' step_profiles_<T>.json "
                         "and write the plan's 8-line F2 plus the per-profile dispersion "
                         "ordering. Run it after both single-backbone runs; with --combine, "
                         "--out_tables/--out_figs mean the PARENT directory of the two "
                         "per-backbone ones")
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
                    help="keep at most N references per stream (smoke run)")
    ap.add_argument("--sample", type=int, default=None,
                    help="accepted so the three P4 scripts share one CLI; sections 3.1-3.6 "
                         "have no sampled quantity, so it is unused here")
    ap.add_argument("--workers", type=int, default=None,
                    help="accepted for CLI symmetry; this script is single-process "
                         "(BLAS threads come from OMP_NUM_THREADS/OPENBLAS_NUM_THREADS)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when the outputs already exist")
    ap.add_argument("--no_figures", action="store_true", help="tables and JSON only")
    return ap


def run_combine(args: argparse.Namespace) -> None:
    base_tables = args.out_tables or (_PROJECT_ROOT / "resources" / "video_full_trajectory")
    base_figs = args.out_figs or (_PROJECT_ROOT / "docs" / "figures" / "video_full_trajectory")
    reports: dict[str, dict[str, Any]] = {}
    for T in BACKBONES:
        p = base_tables / T / f"step_profiles_{T}.json"
        if not p.is_file():
            raise SystemExit(f"--combine needs both backbones; {p} is missing. Run "
                             f"`--backbone {T}` first.")
        reports[T] = json.loads(p.read_text(encoding="utf-8"))
    limits = {T: rep["source"]["limit"] for T, rep in reports.items()}
    if len(set(map(str, limits.values()))) != 1:
        raise SystemExit(f"the two backbones' JSONs come from runs with different --limit "
                         f"({limits}); rerun the smaller one before combining")
    out_md = base_tables / "cross_prompt_cv_ordering.md"
    out_png = base_figs / "f2_cross_prompt_both_backbones.png"
    if not args.force and out_md.is_file() and (args.no_figures or out_png.is_file()):
        print(f"[skip] {out_md} and {out_png} already exist; pass --force to recompute")
        return
    base_tables.mkdir(parents=True, exist_ok=True)
    for p in write_combined(reports, base_tables, base_figs, no_figures=args.no_figures):
        print(f"wrote {p}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.combine:
        run_combine(args)
        return
    if args.backbone is None:
        raise SystemExit("--backbone is required (or pass --combine)")
    T = args.backbone
    matrix = args.data_root / T / "matrix"
    merged = args.merged or (matrix / "trajectory" / "t1_merged.jsonl")
    index_path = args.index or (matrix / "trajectory" / "t1_index.json")
    out_tables = args.out_tables or (_PROJECT_ROOT / "resources" / "video_full_trajectory" / T)
    out_figs = args.out_figs or (_PROJECT_ROOT / "docs" / "figures" / "video_full_trajectory" / T)
    json_path = out_tables / f"step_profiles_{T}.json"

    # An existing output only stands for this run when it was produced with the
    # same parameters: `--limit 3` writes a complete-looking JSON, and without
    # this the production run would print [skip] and leave the smoke numbers in
    # place (the P9 doc would then quote 18 references).
    run_params = {"limit": args.limit, "merged": str(merged)}
    stored = resolve_reuse(json_path, run_params, caps=("limit",), force=args.force,
                           extra_outputs=[out_tables / f"step_profiles_{T}.md"])
    if stored is not None:
        expected = [] if args.no_figures else (
            f1_paths(stored, T, out_figs) + [out_figs / f"f2_cross_prompt_{T}.png",
                                             out_figs / f"f3a_shape_scalars_{T}.png"])
        missing = [p for p in expected if not output_complete(p)]
        if not missing:
            print(f"[skip] outputs already exist under {out_tables} and {out_figs} "
                  f"for limit={args.limit}; pass --force to recompute")
            return
        print(f"[recompute] {len(missing)} figure(s) missing, empty or truncated: "
              f"{', '.join(p.name for p in missing)}")

    print(f"=== step_profiles {T} ===")
    floor = load_p1_floor(T, args.p1_floor_dir)
    index = load_index(index_path)
    sigmas = np.asarray(index["sigma_grids"][0], dtype=np.float64)
    if sigmas.shape[0] != N_STATES:
        raise SystemExit(f"sigma grid has {sigmas.shape[0]} entries, want {N_STATES}")
    refs = load_references(merged, index, limit=args.limit)

    report = analyse(refs, T, sigmas, floor)
    report["backbone"] = T
    report["produced_by"] = "analysis/video_trajectory/step_profiles.py"
    report["plan_sections"] = ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6 (scalars)"]
    report["index_convention"] = INDEX_CONVENTION
    report["floor"] = floor
    report["source"] = {**refs.meta, "index": str(index_path)}
    report["streams"] = refs.meta["per_stream"]
    report["run_params"] = run_params

    out_tables.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, report)
    write_tables(report, T, out_tables)
    print(f"\nwrote {json_path}")
    if not args.no_figures:
        for p in write_f1(report, T, out_figs):
            print(f"wrote {p}")
        print(f"wrote {write_f2(report, T, out_figs)}")
        print(f"wrote {write_f3a(refs, report, T, out_figs)}")


if __name__ == "__main__":
    main()
