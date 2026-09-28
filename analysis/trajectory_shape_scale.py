#!/usr/bin/env python3
"""Trajectory shape at scale: the three quantities that used to rest on 30
stored trajectories, recomputed over the whole 2-model x 4-dataset x 3-seed
grid produced by `RUN/multi_gpu_full_traj.sh`.

  1. coarse-window curvature profile  — where along the path the bend sits,
     now a per-generation record field instead of a 30-sample side computation
  2. bend-plane geometry              — which plane each trajectory bends in,
     from the stored [chord, PC1, PC2] frames; pairwise, so it cannot be
     reduced to per-trajectory scalars
  3. update subspace                  — already a record field, reported here
     across datasets and models rather than on one cell

Section 2 also asks whether the bend plane is set by the initial noise or by
the prompt. `seed_for = base + prompt_idx` collides across datasets and seed
streams, so the store already holds pairs that share the noise but not the
prompt, pairs that share the prompt but not the noise, and (the great
majority) pairs that share neither. Membership comes from the recorded
`z_T_sha256`, so it is read off the data rather than inferred from the seed
arithmetic.

    python analysis/trajectory_shape_scale.py --root ~/full_traj_shape
    python analysis/trajectory_shape_scale.py --root ~/full_traj_shape \\
           --out shape_scale.json --figures figs/

Cap the BLAS thread pool before running this on a many-core host
(`OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16`). The work is a few large
matrix products among thousands of small factorizations, and an uncapped pool
pays a synchronization barrier across every core for each small one: measured
on a 256-core host, two thirds of the run went to system time and the job was
still unfinished after 45 minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import CURVATURE_WINDOW, window_centers  # noqa: E402

PROFILE_KEY = f"turn_angle_w{CURVATURE_WINDOW}_deg"
NULL_PAIRS = 500  # random-plane pairs drawn for the "no alignment" reference;
# the null concentrates hard in high dimension, so a few hundred pins the median


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


WHOLE_PATH = "000_051"  # segment tag of the whole 51-row path


def load_cell(run_dir: Path, segment: str = WHOLE_PATH
              ) -> tuple[list[dict], np.ndarray | None]:
    """All records of one run dir plus the frames of ONE segment, `[n, 3, d]`.

    Records without a frame for that segment are kept (the profile sections
    still use them) and the frame array comes back None if none has one.
    Schema v5 stores a map of segment tag to filename; v3/v4 stored a single
    `frame_file`, which is the whole path.
    """
    records = []
    for path in sorted(run_dir.glob("traj_*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    frames = []
    for record in records:
        files = record.get("frame_files")
        name = files.get(segment) if isinstance(files, dict) else (
            record.get("frame_file") if segment == WHOLE_PATH else None)
        record.pop("_has_frame", None)
        if not name:
            continue
        frames.append(np.load(run_dir / name).astype(np.float32))
        record["_has_frame"] = True
    return records, (np.stack(frames) if frames else None)


def available_segments(root: Path) -> list[str]:
    """Segment tags the store actually holds, in path order."""
    for path in sorted(root.glob("*/*/traj_*.json"))[:1]:
        record = json.loads(path.read_text(encoding="utf-8"))
        files = record.get("frame_files")
        if isinstance(files, dict) and files:
            return sorted(files)
        if record.get("frame_file"):
            return [WHOLE_PATH]
    return []


def load_root(root: Path, segment: str = WHOLE_PATH
              ) -> dict[tuple[str, str, int], tuple[list[dict], np.ndarray | None]]:
    """Every cell under `<root>/<model>/<run_name>/`, keyed (model, dataset, seed).

    Holds every frame in memory at once: at the planned grid that is ~11 GiB
    here, and section 2 peaks near 30 GiB while it stacks and orthonormalizes.
    Sized for the run host, not for a laptop.
    """
    cells: dict[tuple[str, str, int], tuple[list[dict], np.ndarray | None]] = {}
    for run_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        records, frames = load_cell(run_dir, segment)
        if not records:
            continue
        head = records[0]
        key = (head["model"], head["dataset"], int(head["seed"]))
        if key in cells:
            raise ValueError(f"two run dirs claim cell {key}; second is {run_dir}")
        cells[key] = (records, frames)
    return cells


def orthonormalize(frames: np.ndarray) -> np.ndarray:
    """Undo the float16 store's loss of exact orthonormality, per frame."""
    return np.stack([np.linalg.qr(f.T)[0].T for f in frames]).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. coarse-window curvature profile
# ---------------------------------------------------------------------------


def profiles_of(records: list[dict]) -> np.ndarray:
    missing = {r.get("schema") for r in records if PROFILE_KEY not in r}
    if missing:
        raise SystemExit(
            f"records without '{PROFILE_KEY}' (schemas {sorted(missing)}); the coarse-window "
            f"profile arrived in full_trajectory.v3, so a run dir holding older records has to "
            f"be regenerated rather than mixed in"
        )
    return np.array([r[PROFILE_KEY] for r in records], dtype=np.float64)


def _centers_for(profile_len: int) -> list[int]:
    return window_centers(profile_len + 2 * CURVATURE_WINDOW)


def profile_summary(records: list[dict]) -> dict[str, Any]:
    """Where along the path the bend sits, and how tightly the individual
    trajectories agree about it.

    Location is read directly — per trajectory, which center carries the
    trough and which the peak — because a correlation cannot answer it: the
    profile's shared U shape dominates any correlation, so two cells whose
    troughs sit 12 steps apart still correlate at ~0.8. `norm_dev` is the
    location-sensitive companion: after scaling every profile to unit mean
    (removing the level, which `peak_over_min` reports separately), the
    largest absolute gap to the cell's median profile, as a fraction of the
    mean. `cv` is the cross-trajectory spread per center and still contains
    the level variation.
    """
    prof = profiles_of(records)
    med = np.median(prof, axis=0)
    centers = np.array(_centers_for(prof.shape[1]))
    scaled = prof / prof.mean(axis=1, keepdims=True)
    dev = np.abs(scaled - med / med.mean()).max(axis=1)
    troughs = centers[prof.argmin(axis=1)]
    peaks = centers[prof.argmax(axis=1)]
    cv = prof.std(axis=0) / prof.mean(axis=0)
    return {
        "n": int(prof.shape[0]),
        "median_profile": [float(v) for v in med],
        "centers": [int(c) for c in centers],
        "peak_center": int(centers[int(med.argmax())]),
        "min_center": int(centers[int(med.argmin())]),
        "peak_over_min": float(med.max() / med.min()),
        "trough_center_med": float(np.median(troughs)),
        "trough_center_iqr": float(np.quantile(troughs, 0.75) - np.quantile(troughs, 0.25)),
        "trough_center_q05_q95": [float(np.quantile(troughs, 0.05)),
                                  float(np.quantile(troughs, 0.95))],
        "peak_center_med": float(np.median(peaks)),
        "norm_dev_med": float(np.median(dev)),
        "norm_dev_q95": float(np.quantile(dev, 0.95)),
        "cv_med": float(np.median(cv)),
        "cv_max": float(cv.max()),
    }


def profile_agreement(medians: dict[Any, np.ndarray]) -> dict[str, float]:
    """How close a set of cell-median profiles are to each other: where their
    troughs sit, the largest gap between their unit-mean shapes, and the level
    ratio. No correlation here either, for the reason in `profile_summary`."""
    keys = sorted(medians)
    scaled = {k: medians[k] / medians[k].mean() for k in keys}
    centers = np.array(_centers_for(len(medians[keys[0]])))
    trough = {k: int(centers[int(medians[k].argmin())]) for k in keys}
    dev, ratio, shift = [], [], []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            dev.append(float(np.abs(scaled[a] - scaled[b]).max()))
            ratio.append(float(medians[a].mean() / medians[b].mean()))
            shift.append(abs(trough[a] - trough[b]))
    ratio = np.array(ratio)
    ratio = np.maximum(ratio, 1.0 / ratio)  # direction-free level gap
    return {
        "pairs": len(dev),
        "norm_dev_max": float(np.max(dev)) if dev else float("nan"),
        "norm_dev_med": float(np.median(dev)) if dev else float("nan"),
        "trough_shift_max": int(np.max(shift)) if shift else 0,
        "level_ratio_max": float(ratio.max()) if len(ratio) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 2. bend-plane geometry
# ---------------------------------------------------------------------------


def pair_angles_deg(gram: np.ndarray, pairs) -> tuple[np.ndarray, np.ndarray]:
    """Principal angles (degrees) of many plane pairs at once, read off the 2x2
    blocks of the Gram matrix of all stacked plane rows. Returns
    `(theta1, theta2)` with theta1 <= theta2 elementwise.

    Closed form rather than `np.linalg.svd` per pair: at this grid's size that
    is ~1e5 calls into LAPACK on 2x2 inputs, and each one pays a
    synchronization barrier across the whole BLAS thread pool, which costs
    more than the arithmetic by orders of magnitude.

    The singular values come from the half-sum / half-difference form
    s = (q +/- p)/2 with p = |(a-d, b+c)| and q = |(a+d, b-c)|. Going through
    ||M||_F and |det M| instead looks simpler but destroys the answer for
    nearly identical planes: there F^2 and 4 det^2 agree to ~1e-23 against
    values of 4, so the discriminant underflows and the two singular values
    collapse onto each other. p and q stay well separated in the same regime.

    Resolution near 0 degrees is set by the Gram's precision, not by this
    formula: arccos is vertical there, so a float32 Gram (what the caller
    builds, to keep a multi-gigabyte stack in memory) resolves ~0.03 degrees.
    That is far below any angle this analysis distinguishes.
    """
    idx = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    i, j = 2 * idx[:, 0], 2 * idx[:, 1]
    a = gram[i, j].astype(np.float64)
    b = gram[i, j + 1].astype(np.float64)
    c = gram[i + 1, j].astype(np.float64)
    d = gram[i + 1, j + 1].astype(np.float64)
    p = np.hypot(a - d, b + c)
    q = np.hypot(a + d, b - c)
    s1 = 0.5 * (q + p)
    s2 = 0.5 * np.abs(q - p)
    theta1 = np.degrees(np.arccos(np.clip(s1, 0.0, 1.0)))   # larger cosine
    theta2 = np.degrees(np.arccos(np.clip(s2, 0.0, 1.0)))
    return theta1, theta2


def random_plane_null(dim: int, n_pairs: int = NULL_PAIRS, seed: int = 0) -> dict[str, float]:
    """Median principal angles between independent random 2-planes in `dim`
    dimensions — the reference a "planes are not shared" reading needs.

    Uniform on the Grassmannian: QR of a Gaussian dim x 2 matrix gives a
    rotation-invariant orthonormal basis, so the pair is genuinely random.
    """
    rng = np.random.default_rng(seed)
    t1, t2 = [], []
    for _ in range(n_pairs):
        a = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
        b = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
        singular = np.clip(np.linalg.svd(a @ b.T, compute_uv=False), 0.0, 1.0)
        theta = np.degrees(np.arccos(singular))
        t1.append(theta[0]); t2.append(theta[1])
    return {
        "n_pairs": n_pairs,
        "theta1_med": float(np.median(t1)),
        "theta1_q05": float(np.quantile(t1, 0.05)),
        "theta2_med": float(np.median(t2)),
    }


def plane_population(gram: np.ndarray) -> dict[str, Any]:
    """Singular spectrum of every plane row stacked together (via their Gram,
    which the caller already has). Trajectories that all bent in one shared
    plane would make this rank 2; the fraction of the total captured by the
    leading directions says how far from that it is."""
    eig = np.clip(np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1], 0.0, None)
    total = float(eig.sum())
    cumulative = np.cumsum(eig) / total
    return {
        "rows": int(gram.shape[0]),
        "top2_share": float(cumulative[1]),
        "top10_share": float(cumulative[min(9, len(eig) - 1)]),
        "dims_for_50pct": int(np.searchsorted(cumulative, 0.50) + 1),
        "dims_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
    }


def plane_population_reference(n_planes: int, dim: int, seed: int = 0) -> dict[str, Any]:
    """`plane_population` on the same number of *random* 2-planes in the same
    dimension — the "no alignment at all" reading the real one is compared to.

    Drawn rather than assumed: with 2n rows in d dimensions the Gram of random
    orthonormal pairs is not flat but Marchenko-Pastur spread, so the naive
    0.5 * rows reference overstates the no-alignment baseline by ~9% at this
    grid's aspect ratio and would read as alignment where there is none.
    """
    rng = np.random.default_rng(seed)
    rows = np.empty((2 * n_planes, dim), dtype=np.float32)
    for i in range(n_planes):
        rows[2 * i:2 * i + 2] = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
    return plane_population(rows @ rows.T)


def split_pairs(records: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """The index pairs that share the initial noise, and those that share the
    prompt. Pairs sharing neither are the great majority and are NOT returned —
    the caller samples them, since enumerating ~1e6 of them buys nothing.

    `seed_for = base_seed + prompt_idx` collides across datasets and seed
    streams, so the store already holds both groups; membership is decided by
    the recorded `z_T_sha256` and by (dataset, prompt_idx), never inferred from
    the seed arithmetic. Prompt identity is positional because the four
    datasets' first 120 prompts do not overlap in text.
    """
    by_noise: dict[str, list[int]] = defaultdict(list)
    by_prompt: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_noise[record["z_T_sha256"]].append(i)
        by_prompt[(record["dataset"], int(record["prompt_idx"]))].append(i)

    same_noise, same_prompt = set(), set()
    for group in by_noise.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                same_noise.add((group[a], group[b]) if group[a] < group[b] else (group[b], group[a]))
    for group in by_prompt.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                same_prompt.add((group[a], group[b]) if group[a] < group[b] else (group[b], group[a]))
    # a pair sharing both would be the same generation twice
    both = same_noise & same_prompt
    return {
        "same_noise_diff_prompt": sorted(same_noise - both),
        "same_prompt_diff_noise": sorted(same_prompt - both),
        "same_both": sorted(both),
    }


def sample_pairs(n: int, count: int, seed: int, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """`count` distinct index pairs from `n` items, skipping `exclude`.

    Capped at what exists: asking for more than `C(n,2) - |exclude|` would
    otherwise spin against the retry guard and silently return a short set.
    """
    count = min(count, n * (n - 1) // 2 - len(exclude))
    rng = np.random.default_rng(seed)
    out: set[tuple[int, int]] = set()
    guard = 0
    while len(out) < count and guard < 50 * count:
        guard += 1
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair in exclude or pair in out:
            continue
        out.add(pair)
    return sorted(out)


def angle_stats(gram: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    if len(pairs) == 0:
        return {"pairs": 0}
    t1, t2 = pair_angles_deg(gram, pairs)
    return {
        "pairs": len(pairs),
        "theta1_med": float(np.median(t1)),
        "theta1_q05": float(np.quantile(t1, 0.05)),
        "theta1_min": float(np.min(t1)),
        "theta2_med": float(np.median(t2)),
    }


def model_plane_report(model_cells, *, max_random_pairs: int = 20000) -> dict[str, Any]:
    """Every plane statistic for one model's cells, as one dict.

    `model_cells` is a list of `(key, (records, frames))` in a fixed order. All
    pair indices address the concatenated FRAME stack, so the cell boundaries
    are counted in frames: a cell that was once run without `--save_frame`
    leaves frameless records behind, and counting records instead would shift
    every later cell's slice and label cross-cell pairs `within_cell`.
    """
    records = [r for _, (recs, _) in model_cells for r in recs if r.get("_has_frame")]
    frames = orthonormalize(np.concatenate([f for _, (_, f) in model_cells]))
    n, dim = frames.shape[0], frames.shape[2]
    if n != len(records):
        raise ValueError(f"{n} frames but {len(records)} framed records")
    chords = frames[:, 0, :]
    plane_rows = frames[:, 1:, :].reshape(2 * n, dim)
    gram = plane_rows @ plane_rows.T
    entry: dict[str, Any] = {"n": n, "dim": dim}

    offsets = np.cumsum([0] + [f.shape[0] for _, (_, f) in model_cells])
    within = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        within.extend((i, j) for i in range(start, end) for j in range(i + 1, end))
    entry["within_cell"] = angle_stats(gram, within)
    entry["within_cell"]["cells"] = len(model_cells)

    split = split_pairs(records)
    for name in ("same_noise_diff_prompt", "same_prompt_diff_noise"):
        entry[name] = angle_stats(gram, split[name])
        entry[name]["chord"] = chord_stats(chords, split[name])
    related = (set(split["same_noise_diff_prompt"]) | set(split["same_prompt_diff_noise"])
               | set(split["same_both"]))
    unrelated = sample_pairs(n, max_random_pairs, seed=1, exclude=related)
    entry["unrelated"] = angle_stats(gram, unrelated)
    entry["unrelated"]["chord"] = chord_stats(chords, unrelated)
    entry["random_plane_null"] = random_plane_null(dim)
    entry["population_reference"] = plane_population_reference(n, dim)
    entry["population"] = plane_population(gram)
    return entry


def chord_stats(chords: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Acute angle between two trajectories' chord directions, same pair sets.
    The chord is dominated by -z_T, so this is mostly a check that the noise
    grouping means what it claims to.

    Reported acute (|cos|, so 0..90) because the rows come out of a QR, whose
    sign convention flips whichever vectors have a positive leading component;
    the signed angle would be 180-theta for about half of them.
    """
    if not pairs:
        return {"pairs": 0}
    idx_a = np.array([p[0] for p in pairs])
    idx_b = np.array([p[1] for p in pairs])
    cos = np.einsum("ij,ij->i", chords[idx_a], chords[idx_b]).astype(np.float64)
    ang = np.degrees(np.arccos(np.clip(np.abs(cos), 0.0, 1.0)))
    return {"pairs": len(pairs), "chord_angle_med": float(np.median(ang))}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None,
                        help="run root written by RUN/multi_gpu_full_traj.sh")
    parser.add_argument("--replot", type=Path, default=None,
                        help="skip the analysis and draw --figures from a saved --out JSON "
                             "(the compute host need not be the one with matplotlib)")
    parser.add_argument("--out", type=Path, default=None, help="dump every number as JSON")
    parser.add_argument("--figures", type=Path, default=None, help="directory for the figures")
    parser.add_argument("--max_random_pairs", type=int, default=20000,
                        help="cap on the sampled unrelated-pair set per model")
    args = parser.parse_args()

    if args.replot is not None:
        if args.figures is None:
            raise SystemExit("--replot needs --figures")
        write_figures(args.figures, json.loads(args.replot.read_text(encoding="utf-8")))
        return
    if args.root is None:
        raise SystemExit("--root is required (or --replot JSON --figures DIR)")

    segments = available_segments(args.root) or [WHOLE_PATH]
    cells = load_root(args.root, WHOLE_PATH if WHOLE_PATH in segments else segments[0])
    if not cells:
        raise SystemExit(f"no cells under {args.root}")
    report: dict[str, Any] = {"root": str(args.root), "cells": {}}

    # ---- inventory ------------------------------------------------------
    print("=== inventory ===")
    devices, schemas = set(), set()
    for key in sorted(cells):
        records, frames = cells[key]
        devices.update(r.get("device_name", "?") for r in records)
        schemas.update(r.get("schema", "?") for r in records)
        n_frames = 0 if frames is None else frames.shape[0]
        print(f"  {key[0]:5s} {key[1]:22s} s{key[2]:<7d} n={len(records):4d} frames={n_frames:4d}")
        report["cells"][f"{key[0]}|{key[1]}|{key[2]}"] = {
            "n": len(records), "frames": n_frames}
    print(f"  schemas={sorted(schemas)}  devices={sorted(devices)}")
    report["schemas"] = sorted(schemas)
    report["devices"] = sorted(devices)
    if len(devices) > 1:
        print("  [WARN] more than one device: same seed gives different z_T across "
              "architectures, so the noise-sharing split below is only valid within one")

    # ---- 1. curvature profile -------------------------------------------
    print(f"\n=== 1. coarse-window curvature profile (w={CURVATURE_WINDOW}) ===")
    print("  cell                                  n   peak@  min@  peak/min  "
          "trough[q05,q95] iqr   shape_gap  cv")
    medians: dict[tuple[str, str, int], np.ndarray] = {}
    for key in sorted(cells):
        records, _ = cells[key]
        summary = profile_summary(records)
        medians[key] = np.array(summary["median_profile"])
        report["cells"][f"{key[0]}|{key[1]}|{key[2]}"]["profile"] = summary
        lo, hi = summary["trough_center_q05_q95"]
        print(f"  {key[0]:5s} {key[1]:22s} s{key[2]:<7d} {summary['n']:4d}  "
              f"{summary['peak_center']:4d}  {summary['min_center']:4d}  "
              f"{summary['peak_over_min']:7.2f}   "
              f"{summary['trough_center_med']:5.1f} [{lo:.0f},{hi:.0f}] "
              f"{summary['trough_center_iqr']:5.1f}   "
              f"{summary['norm_dev_med']:.3f}  {summary['cv_med']:.3f}")

    report["profile_agreement"] = {}
    for label, subset in (
        ("across seeds, within (model, dataset)", "seed"),
        ("across datasets, within (model, seed)", "dataset"),
        ("across everything, within model", "model"),
    ):
        groups: dict[Any, dict[Any, np.ndarray]] = defaultdict(dict)
        for key, med in medians.items():
            model, dataset, seed = key
            bucket = {"seed": (model, dataset), "dataset": (model, seed), "model": model}[subset]
            groups[bucket][key] = med
        merged = [profile_agreement(g) for g in groups.values() if len(g) > 1]
        if not merged:
            continue
        agreement = {
            "groups": len(merged),
            "pairs": sum(m["pairs"] for m in merged),
            "norm_dev_max": max(m["norm_dev_max"] for m in merged),
            "norm_dev_med": float(np.median([m["norm_dev_med"] for m in merged])),
            "trough_shift_max": max(m["trough_shift_max"] for m in merged),
            "level_ratio_max": max(m["level_ratio_max"] for m in merged),
        }
        report["profile_agreement"][subset] = agreement
        print(f"  {label}: {agreement['groups']} groups / {agreement['pairs']} pairs, "
              f"trough shift max={agreement['trough_shift_max']} steps, "
              f"shape gap med={agreement['norm_dev_med']:.4f} "
              f"max={agreement['norm_dev_max']:.4f}, "
              f"level ratio max={agreement['level_ratio_max']:.3f}")

    # ---- 2. bend-plane geometry -----------------------------------------
    print("\n=== 2. bend-plane geometry ===")
    print(f"  segments stored: {', '.join(segments)}")
    report["segments"] = segments
    report["planes"] = {}
    for segment in segments:
        seg_cells = cells if segment == (WHOLE_PATH if WHOLE_PATH in segments
                                         else segments[0]) else load_root(args.root, segment)
        for model in sorted({k[0] for k in seg_cells}):
            model_cells = [(k, seg_cells[k]) for k in sorted(seg_cells) if k[0] == model]
            if any(frames is None for _, (_, frames) in model_cells):
                print(f"  {model} [{segment}]: no frames stored, skipped")
                continue
            entry = model_plane_report(model_cells, max_random_pairs=args.max_random_pairs)
            entry["segment"] = segment
            n, dim = entry["n"], entry["dim"]
            report["planes"].setdefault(model, {})[segment] = entry

            print(f"  {model} [{segment}]  n={n} trajectories, d={dim}")
            for name in ("within_cell", "same_noise_diff_prompt", "same_prompt_diff_noise",
                         "unrelated"):
                stat = entry[name]
                if not stat["pairs"]:
                    continue
                chord_note = (f"  chord {stat['chord']['chord_angle_med']:6.2f}"
                              if "chord" in stat else "")
                print(f"    {name:24s} {stat['pairs']:7d} pairs  "
                      f"theta1 med={stat['theta1_med']:6.2f} q05={stat['theta1_q05']:6.2f} "
                      f"min={stat['theta1_min']:6.2f}  theta2 med={stat['theta2_med']:6.2f}{chord_note}")
            null = entry["random_plane_null"]
            print(f"    {'random-plane null':24s} {null['n_pairs']:7d} pairs  "
                  f"theta1 med={null['theta1_med']:6.2f} q05={null['theta1_q05']:6.2f}"
                  f"{'':17s}theta2 med={null['theta2_med']:6.2f}")
            pop, ref = entry["population"], entry["population_reference"]
            print(f"    population of {pop['rows']} plane directions: top-2 hold "
                  f"{pop['top2_share']:.4f}, top-10 {pop['top10_share']:.4f}; "
                  f"{pop['dims_for_50pct']} dims for 50%, {pop['dims_for_90pct']} for 90%")
            print(f"    {'':18s}same count of random planes: top-2 {ref['top2_share']:.4f}, "
                  f"top-10 {ref['top10_share']:.4f}; {ref['dims_for_50pct']} / "
                  f"{ref['dims_for_90pct']} dims")

    # ---- 3. update subspace ---------------------------------------------
    print("\n=== 3. update subspace ===")
    print("  cell                                  n   chord_share  in_plane  own_evr2")
    report["update"] = {}
    for key in sorted(cells):
        records, _ = cells[key]
        share = np.array([r["update_chord_share"] for r in records], dtype=np.float64)
        plane = np.array([r["update_in_position_plane"] for r in records], dtype=np.float64)
        own2 = np.array([r["update_own_evr"][0] + r["update_own_evr"][1] for r in records])
        entry = {
            "n": len(records),
            "chord_share_med": float(np.nanmedian(share)),
            "in_position_plane_med": float(np.nanmedian(plane)),
            "own_evr2_med": float(np.nanmedian(own2)),
        }
        report["update"][f"{key[0]}|{key[1]}|{key[2]}"] = entry
        print(f"  {key[0]:5s} {key[1]:22s} s{key[2]:<7d} {entry['n']:4d}      "
              f"{entry['chord_share_med']:.4f}    {entry['in_position_plane_med']:.4f}   "
              f"{entry['own_evr2_med']:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.figures is not None:
        write_figures(args.figures, report)


def write_figures(out_dir: Path, report: dict[str, Any]) -> None:
    """Two panels per model: every cell's median curvature profile, so the
    "same place, different level" reading is visible rather than asserted, and
    the pairwise plane angles by what the two generations share.

    Driven by the report dict rather than by the loaded frames, so it can be
    replayed from a saved JSON on a host that has matplotlib — the machine
    with 5.8 GB of frames on it need not be the same machine.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = report["cells"]
    models = sorted({k.split("|")[0] for k in cells})
    colors = {"drawbench_full": "C0", "geneval_style": "C1",
              "parti_full": "C2", "diffusiondb_clean10k": "C3"}

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        keys = sorted(k for k in cells if k.startswith(f"{model}|"))
        labelled: set[str] = set()
        n_per_cell = cells[keys[0]]["profile"]["n"]
        for key in keys:
            dataset = key.split("|")[1]
            profile = cells[key]["profile"]
            label = dataset if dataset not in labelled else None
            labelled.add(dataset)
            ax.plot(profile["centers"], profile["median_profile"],
                    color=colors.get(dataset, "k"), alpha=0.85, lw=1.2, label=label)
        ax.set_xlabel("denoising step (centre of the measurement window)")
        ax.set_ylabel(f"turn angle over the {CURVATURE_WINDOW} steps\n"
                      f"before and after (degrees)")
        ax.set_title(f"{model.upper()}  —  {len(keys)} curves = 4 datasets x 3 random seeds\n"
                     f"each curve is the median over {n_per_cell} prompts",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="dataset", title_fontsize=8)
    fig.suptitle("How sharply the trajectory turns at each denoising step",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "curvature_profile.png", dpi=140)

    # Paper conventions for this panel: 5.5 in print width, 7.6 pt labels,
    # 7.2 pt ticks, no in-figure title and no explanatory text block (the
    # caption carries both).  The three measured groups share one hue at three
    # lightnesses, an olive that avoids every cache-method colour; the
    # random-plane reference is drawn as an outlined grey bar, the colour the
    # paper reserves for a random control.
    groups = [("same_noise_diff_prompt", "same\nnoise", "#3B4A1C"),
              ("same_prompt_diff_noise", "same\nprompt", "#7E9445"),
              ("unrelated", "neither", "#C6D49B"),
              ("random_plane_null", "random\nplanes", "0.86")]
    display = {"flux": "FLUX.1-dev", "qwen": "Qwen-Image"}
    fs_lab, fs_tick, fig_w, fig_h = 7.6, 7.2, 5.5, 2.35
    style = {"font.family": "DejaVu Sans", "font.size": fs_lab,
             "axes.titlesize": fs_lab, "axes.labelsize": fs_lab,
             "axes.linewidth": 0.7, "xtick.major.width": 0.7,
             "ytick.major.width": 0.7, "xtick.major.size": 2.4,
             "ytick.major.size": 2.4}
    restore = {key: plt.rcParams[key] for key in style}
    plt.rcParams.update(style)
    fig, axes = plt.subplots(1, len(models), figsize=(fig_w, fig_h),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.092, right=0.986, bottom=0.155, top=0.885,
                        wspace=0.10)
    for ax, model in zip(axes[0], models):
        entry = report["planes"].get(model)
        if entry is None:
            continue
        values = [entry[key]["theta1_med"] for key, _, _ in groups]
        counts = [entry[key].get("pairs", entry[key].get("n_pairs", 0))
                  for key, _, _ in groups]
        bars = ax.bar(range(len(groups)), values, width=0.66,
                      color=[c for _, _, c in groups], linewidth=0.0)
        bars[-1].set_edgecolor("0.45")      # the reference reads as an outline
        bars[-1].set_linewidth(0.7)
        # only the angle sits over the bar; the pair counts go in the caption,
        # where they do not have to fit in a 0.6 in slot
        for bar, value in zip(bars, values):
            # a white rim so the labels of the taller bars stay readable where
            # they cross the 90 degree line
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.6,
                    f"{value:.1f}°", ha="center", va="bottom",
                    fontsize=fs_tick - 0.6, color="0.25", zorder=5,
                    path_effects=[matplotlib.patheffects.withStroke(
                        linewidth=1.6, foreground="white")])
        print(f"  {model} plane angles: " + ", ".join(
            f"{label.replace(chr(10), ' ')} {value:.1f} deg over {count:,} pairs"
            for (_, label, _), value, count in zip(groups, values, counts)))
        ax.axhline(90.0, color="0.35", ls=(0, (1.6, 1.6)), lw=0.7, zorder=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([label for _, label, _ in groups], linespacing=1.05)
        ax.set_xlim(-0.62, len(groups) - 0.38)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 30, 60, 90])
        ax.tick_params(labelsize=fs_tick, pad=1.6)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(display.get(model, model.upper()), pad=3.0)
    axes[0][0].set_ylabel("median principal angle (degrees)", labelpad=2.0)
    meta = {"CreationDate": None, "Creator": "trajectory_shape_scale.py",
            "Producer": "matplotlib"}
    fig.savefig(out_dir / "bend_plane_angles.png", dpi=400)
    fig.savefig(out_dir / "bend_plane_angles.pdf", metadata=meta)
    plt.close(fig)
    plt.rcParams.update(restore)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
