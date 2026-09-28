#!/usr/bin/env python3
"""Reproduce every number and figure in `docs/full_trajectory_results.md`.

The doc's tables were first produced by one-off session scripts, which is not
good enough for an archive: a number nobody can recompute cannot be checked.
This module regenerates them from the stored tables and the 30 stored full
trajectories, prints them in the doc's own layout, and rebuilds F0/F1/F2/F3.

    python analysis/full_trajectory_analysis.py                  # numbers only
    python analysis/full_trajectory_analysis.py --figures OUT    # + figures
    python analysis/full_trajectory_analysis.py --step-profiles  # + profiles JSON

Inputs (both untracked, produced by the Slurm wave):
  resources/full_trajectory/tables_jsonl/full_traj_<model>_<dataset>.jsonl
  resources/full_trajectory/latents_flux/latents_*.pt      (30 trajectories)
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import (  # noqa: E402
    CURVATURE_WINDOW,
    plane_frame,
    principal_angles_deg,
    turn_angles_window_deg,
)

TABLES = Path("resources/full_trajectory/tables_jsonl")
LATENTS = "resources/full_trajectory/latents_{model}/latents_*.pt"
STEP_PROFILES_JSON = Path("resources/full_trajectory_analysis/step_profiles.json")
FLUX_D, QWEN_D = 262144, 440896
DIM = {"flux": FLUX_D, "qwen": QWEN_D}
PROMPTS = {"drawbench_full": 200, "geneval_style": 553,
           "parti_full": 1632, "diffusiondb_clean10k": 10000}
DATASETS = ("drawbench_full", "geneval_style", "parti_full", "diffusiondb_clean10k")
STYLES = {"flux": ("#1f77b4", "-"), "qwen": ("#d62728", "--")}  # figure colours
DATASET_NAMES = {"drawbench_full": "DrawBench", "geneval_style": "GenEval",
                 "parti_full": "Parti", "diffusiondb_clean10k": "DiffusionDB"}


def load(model: str, dataset: str) -> pd.DataFrame | None:
    path = TABLES / f"full_traj_{model}_{dataset}.jsonl"
    return pd.read_json(path, lines=True) if path.is_file() else None


def _stack(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].to_numpy())


# --- section 5.1 / 5.2 : per-dataset shape summary ----------------------


def shape_summary(frame: pd.DataFrame) -> dict[str, float]:
    chord = frame["chord_len"].to_numpy()
    final = np.array([m[-1] for m in frame["magnitude"]])
    evr = _stack(frame, "pca_evr")
    evr2 = evr[:, 0] + evr[:, 1]
    evr3 = evr2 + evr[:, 2]
    sigma_grids = {tuple(s) for s in frame["sigmas"]}
    return {
        "n": len(frame),
        "chord_med": float(np.median(chord)),
        "dev_med": float(frame.max_dev_ratio.median()),
        "dev_q05": float(frame.max_dev_ratio.quantile(0.05)),
        "dev_q95": float(frame.max_dev_ratio.quantile(0.95)),
        "dev_over_data": float((frame.max_dev_ratio * chord / final).median()),
        "straightness": float(frame.straightness.median()),
        "evr_components": [float(np.median(evr[:, i])) for i in range(evr.shape[1])],
        "evr2_med": float(np.median(evr2)),
        "evr2_q05": float(np.quantile(evr2, 0.05)),
        "evr3_med": float(np.median(evr3)),
        "recon_rel_2d": float(frame.recon_err_rel_2d.median()),
        "recon_rel_3d": float(frame.recon_err_rel_3d.median()),
        # section 1: how many mutually distinct initial noise draws the cell
        # holds, and whether its sigma grid is unique across rows.
        "z_T_clusters": int(frame.z_T_sha256.nunique()),
        "sigma_grids": len(sigma_grids),
    }


# --- section 4.1 / 4.2 : per-step dispersion + variance decomposition -------------


def prompt_variance_share(profiles: np.ndarray, prompt_idx: np.ndarray,
                          *, n_seeds: int = 3) -> np.ndarray:
    """Per-step share of the cross-(prompt, seed) variance that is between
    prompts, using the UNBIASED within-group estimator (divide by m-1); the
    plug-in estimator overstates the share."""
    order = np.argsort(prompt_idx, kind="stable")
    ordered = profiles[order]
    counts = np.unique(prompt_idx, return_counts=True)[1]
    if counts.min() != n_seeds or counts.max() != n_seeds:
        raise ValueError(f"expected {n_seeds} seeds per prompt, "
                         f"got {counts.min()}-{counts.max()}")
    grouped = ordered.reshape(-1, n_seeds, profiles.shape[1])
    within = ((grouped - grouped.mean(axis=1, keepdims=True)) ** 2).sum(axis=1).mean(axis=0)
    total = ordered.var(axis=0)
    # a step every trajectory shares exactly (the two chord endpoints of the
    # deviation profile) has zero total variance and no share to report
    return np.where(total > 0, (total - within / (n_seeds - 1)) / np.where(total > 0, total, 1),
                    np.nan)


def dispersion(frame: pd.DataFrame, *, n_seeds: int = 3) -> dict[str, object]:
    """Cross-(prompt, seed) CV of the normalised deviation profile, and the
    between-prompt share of its variance."""
    chord = frame["chord_len"].to_numpy()
    dperp = _stack(frame, "d_perp") / chord[:, None]
    # states 1..49: the two endpoints lie on the chord by construction, so their
    # mean is a float64 residue and the CV there is 0/0 (see the block comment
    # below). Everything in between is reported.
    inner = slice(1, 50)
    cv = dperp[:, inner].std(axis=0) / dperp[:, inner].mean(axis=0)
    share = prompt_variance_share(dperp, frame["prompt_idx"].to_numpy(), n_seeds=n_seeds)
    return {
        "cv_med": float(np.median(cv)),
        "cv_min": float(cv.min()),
        "cv_max": float(cv.max()),
        "share_early": [float(v) for v in share[1:7]],
        "share_late_min": float(share[7:49].min()),
        "share_late_max": float(share[7:49].max()),
    }


# --- section 3.4 : velocity law --------------------------------------------


def velocity_law(frame: pd.DataFrame, *, dim: int) -> dict[str, float]:
    vel = _stack(frame, "velocity_norm") / math.sqrt(dim)
    med = np.median(vel, axis=0)
    per = (vel.max(axis=1) - vel.min(axis=1)) / vel.mean(axis=1)
    per_tail = (vel[:, 1:].max(axis=1) - vel[:, 1:].min(axis=1)) / vel[:, 1:].mean(axis=1)
    return {
        "profile_min": float(med.min()),
        "profile_max": float(med.max()),
        "profile_spread": float((med.max() - med.min()) / med.mean()),
        "per_traj_spread_med": float(np.median(per)),
        "per_traj_spread_med_no_step0": float(np.median(per_tail)),
        "cross_prompt_cv": float(np.median(vel.std(axis=0) / vel.mean(axis=0))),
    }


# --- per-step population profiles -------------------------------------------
#
# Index conventions, straight out of analysis/trajectory_math.py. Getting these
# wrong is the one way to produce a plausible-looking but meaningless profile,
# so they are stated once here and asserted in `step_profiles`:
#
#   state index n = 0..50 (51 values)   d_perp[n], magnitude[n], sigmas[n]
#       n is the state AFTER n solver steps; n = 0 is z_T, n = 50 is the result.
#   step index  n = 0..49 (50 values)   spacing[n], velocity_norm[n]
#       spacing[n] = ||Z[n+1] - Z[n]||, the displacement solver step n produces.
#
# d_perp[0] and d_perp[50] are zero by construction (both endpoints lie on the
# chord), so every dispersion reading over the deviation profile is reported on
# states 1..49 only: at the two endpoints the mean is a float64 residue and the
# CV is 0/0.

LANDMARK_STATES = (0, 1, 2, 5, 10, 20, 30, 35, 37, 40, 42, 45, 47, 48, 49, 50)


def _landmarks(profile: np.ndarray) -> dict[str, float]:
    return {str(n): float(profile[n]) for n in LANDMARK_STATES if n < len(profile)}


def _peak_readings(profile: np.ndarray) -> dict[str, float]:
    """Where a single-humped profile peaks and how wide the hump is."""
    peak_at = int(np.argmax(profile))
    peak = float(profile[peak_at])
    half = np.flatnonzero(profile >= peak / 2)
    plateau = np.flatnonzero(profile >= 0.9 * peak)
    return {
        "peak_step": peak_at,
        "peak": peak,
        "half_first_step": int(half[0]),
        "half_last_step": int(half[-1]),
        "half_width_steps": int(half[-1] - half[0] + 1),
        "plateau90_first_step": int(plateau[0]),
        "plateau90_last_step": int(plateau[-1]),
    }


def _spread(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _series(values: np.ndarray) -> list:
    """Profile as strict JSON: the steps with nothing to report become null."""
    return [None if not np.isfinite(v) else float(v) for v in values]


def _cv(arr: np.ndarray) -> np.ndarray:
    """Population CV down the trajectory axis, per step."""
    return arr.std(axis=0) / arr.mean(axis=0)


def _cv_readings(cv: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    """CV summary over states/steps `lo..hi` inclusive, plus where it extremes."""
    window = cv[lo:hi + 1]
    return {
        "range": [lo, hi],
        "median": float(np.median(window)),
        "min": float(window.min()),
        "min_step": int(lo + np.argmin(window)),
        "max": float(window.max()),
        "max_step": int(lo + np.argmax(window)),
    }


def step_profiles(frame: pd.DataFrame, *, dim: int, n_seeds: int = 3) -> dict[str, object]:
    """Population per-step profiles for one (model, dataset) cell.

    Four profiles, each a median over every trajectory in the cell:
      deviation from the chord / chord length  (states 0..50)
      per-step displacement    / chord length  (steps  0..49)
      state norm               / sqrt(d)       (states 0..50)
      velocity norm            / sqrt(d)       (steps  0..49)
    plus the per-step cross-(prompt, seed) CV of each, i.e. how much of the
    profile is common to every prompt in the cell.
    """
    chord = frame["chord_len"].to_numpy()
    dev = _stack(frame, "d_perp") / chord[:, None]
    spacing = _stack(frame, "spacing") / chord[:, None]
    magnitude = _stack(frame, "magnitude") / math.sqrt(dim)
    velocity = _stack(frame, "velocity_norm") / math.sqrt(dim)
    if (dev.shape[1], spacing.shape[1]) != (51, 50) or magnitude.shape[1] != 51:
        raise ValueError(f"unexpected profile lengths: dev {dev.shape[1]}, "
                         f"spacing {spacing.shape[1]}, magnitude {magnitude.shape[1]}")
    if not np.allclose(dev[:, [0, -1]], 0.0, atol=1e-9):
        raise ValueError("d_perp endpoints are not zero: index convention broken")

    sigmas = np.median(_stack(frame, "sigmas"), axis=0)
    dev_med, spacing_med = np.median(dev, axis=0), np.median(spacing, axis=0)
    mag_med, vel_med = np.median(magnitude, axis=0), np.median(velocity, axis=0)
    dev_cv, spacing_cv = _cv(dev[:, 1:50]), _cv(spacing)
    mag_cv, vel_cv = _cv(magnitude), _cv(velocity)
    # dev_cv is stored on states 1..49; pad the two construction-zero endpoints
    # with NaN so the stored array stays 51 long like the profile it describes.
    dev_cv_full = np.concatenate([[np.nan], dev_cv, [np.nan]])
    per_traj_peak = dev.argmax(axis=1)
    peak_mode = int(np.bincount(per_traj_peak).argmax())
    cumulative = np.cumsum(spacing, axis=1) / spacing.sum(axis=1, keepdims=True)
    share = prompt_variance_share(dev, frame["prompt_idx"].to_numpy(), n_seeds=n_seeds)
    # state 50's deviation is a float64 residue around zero, not a measurement,
    # so its variance decomposition is decomposing rounding error; state 0 is
    # exactly zero and already comes back NaN
    share[[0, -1]] = np.nan

    return {
        "n": int(len(frame)),
        "n_prompts": int(frame["prompt_idx"].nunique()),
        "n_seeds": n_seeds,
        "dim": dim,
        "profiles": {
            "dev_over_chord": _series(dev_med),
            "dev_over_chord_q25": _series(np.quantile(dev, 0.25, axis=0)),
            "dev_over_chord_q75": _series(np.quantile(dev, 0.75, axis=0)),
            "dev_over_chord_cv": _series(dev_cv_full),
            "dev_prompt_variance_share": _series(share),
            "spacing_over_chord": _series(spacing_med),
            "spacing_over_chord_cv": _series(spacing_cv),
            "magnitude_over_sqrt_d": _series(mag_med),
            "magnitude_over_sqrt_d_cv": _series(mag_cv),
            "velocity_over_sqrt_d": _series(vel_med),
            "velocity_over_sqrt_d_cv": _series(vel_cv),
            "sigmas": _series(sigmas),
        },
        "dev": {
            **_peak_readings(dev_med),
            "sigma_at_peak": float(sigmas[int(np.argmax(dev_med))]),
            "landmarks": _landmarks(dev_med),
            "per_trajectory_peak_step": _spread(per_traj_peak.astype(float)),
            "per_trajectory_peak_mode": peak_mode,
            "per_trajectory_peak_mode_share": float((per_traj_peak == peak_mode).mean()),
            "per_trajectory_peak_within_2_share":
                float((np.abs(per_traj_peak - peak_mode) <= 2).mean()),
            "max_dev_ratio_median": float(frame["max_dev_ratio"].median()),
            "cv": _cv_readings(dev_cv_full, 1, 49),
            "prompt_share_early": [float(v) for v in share[1:7]],
            "prompt_share_late_min": float(share[7:49].min()),
            "prompt_share_late_max": float(share[7:49].max()),
        },
        "magnitude": {
            "start": float(mag_med[0]),
            "end": float(mag_med[-1]),
            "trough_step": int(np.argmin(mag_med)),
            "trough": float(mag_med.min()),
            "trough_over_start": float(mag_med.min() / mag_med[0]),
            "end_over_start": float(mag_med[-1] / mag_med[0]),
            "start_times_sqrt_d": float(mag_med[0] * math.sqrt(dim)),
            "landmarks": _landmarks(mag_med),
            "cv": _cv_readings(mag_cv, 0, 50),
        },
        "spacing": {
            "first": float(spacing_med[0]),
            "last": float(spacing_med[-1]),
            "max": float(spacing_med.max()),
            "max_step": int(np.argmax(spacing_med)),
            "min": float(spacing_med.min()),
            "min_step": int(np.argmin(spacing_med)),
            "max_over_min": float(spacing_med.max() / spacing_med.min()),
            "path_share_last_10_steps": float(np.median(1.0 - cumulative[:, -11])),
            "path_share_first_10_steps": float(np.median(cumulative[:, 9])),
            "steps_to_half_path": float(np.median(
                (cumulative >= 0.5).argmax(axis=1) + 1)),
            "landmarks": _landmarks(spacing_med),
            "cv": _cv_readings(spacing_cv, 0, 49),
        },
        "velocity": {
            "min": float(vel_med.min()),
            "max": float(vel_med.max()),
            "landmarks": _landmarks(vel_med),
            "cv": _cv_readings(vel_cv, 0, 49),
        },
    }


def write_step_profiles(path: Path) -> dict[str, object]:
    """Every (model, dataset) cell's per-step profiles, as one JSON file."""
    cells: dict[str, object] = {}
    for model, dim in (("flux", FLUX_D), ("qwen", QWEN_D)):
        for dataset in DATASETS:
            frame = load(model, dataset)
            if frame is None:
                continue
            cells[f"{model}/{dataset}"] = step_profiles(frame, dim=dim)
            del frame
    payload = {
        "source": "resources/full_trajectory/tables_jsonl",
        "produced_by": "analysis/full_trajectory_analysis.py --step-profiles",
        "index_convention": {
            "dev_over_chord/magnitude_over_sqrt_d/sigmas": "state 0..50 (state after n steps; 0 = z_T)",
            "spacing_over_chord/velocity_over_sqrt_d": "solver step 0..49 (step n moves Z[n] to Z[n+1])",
            "dev_over_chord_cv/dev_prompt_variance_share":
                "null at states 0 and 50, which lie on the chord by construction",
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    return payload


# --- 30-trajectory readings, superseded by the shape doc ------------------


def load_trajectory(path: str) -> np.ndarray:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=True)
    tensor = obj if torch.is_tensor(obj) else next(
        v for v in obj.values() if torch.is_tensor(v) and v.ndim >= 2
    )
    return tensor.to(torch.float32).numpy().reshape(tensor.shape[0], -1).astype(np.float64)


def curvature_profile(paths: list[str], window: int = CURVATURE_WINDOW):
    rows = [turn_angles_window_deg(load_trajectory(path), window) for path in paths]
    profiles = np.array(rows)
    centers = np.arange(window, window + profiles.shape[1])
    return centers, np.median(profiles, axis=0), profiles


# --- cross-prompt plane alignment (30 trajectories) ------------------------------


def _frame_or_die(path: str) -> np.ndarray:
    """`plane_frame` returns None on a degenerate chord or a rank-deficient
    residual. Real trajectories clear both by ~1e11, so this is a loud stop
    rather than a filter: a None here means the stored file is not a
    trajectory."""
    frame = plane_frame(load_trajectory(path))
    if frame is None:
        raise ValueError(f"{path}: no chord-orthogonal plane, cannot align")
    return frame


def plane_alignment(paths: list[str], *, n_null: int = 200, seed: int = 0):
    frames = [_frame_or_die(p)[1:] for p in paths]
    dim = frames[0].shape[1]
    angles = np.array([
        principal_angles_deg(a, b) for a, b in itertools.combinations(frames, 2)
    ])
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        a = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
        b = np.linalg.qr(rng.standard_normal((dim, 2)))[0].T
        null.append(principal_angles_deg(a, b))
    null = np.array(null)
    return {
        "pairs": len(angles),
        "theta1_med": float(np.median(angles[:, 0])),
        "theta2_med": float(np.median(angles[:, 1])),
        "null_theta1_med": float(np.median(null[:, 0])),
        "null_theta2_med": float(np.median(null[:, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=None,
                        help="directory to write F1/F2 into (skipped if omitted)")
    parser.add_argument("--step-profiles", type=Path, nargs="?", default=None,
                        const=STEP_PROFILES_JSON,
                        help=f"write per-step profiles JSON (default path: "
                             f"{STEP_PROFILES_JSON}); skipped if omitted")
    args = parser.parse_args()

    if args.step_profiles is not None:
        payload = write_step_profiles(args.step_profiles)
        print(f"=== per-step profiles ({len(payload['cells'])} cells) ===")
        for key, cell in payload["cells"].items():
            d, m, s = cell["dev"], cell["magnitude"], cell["spacing"]
            print(f"  {key:34s} n={cell['n']:6d}  "
                  f"dev peak {d['peak']:.4f} @ step {d['peak_step']:2d} "
                  f"(half-width {d['half_first_step']}-{d['half_last_step']}), "
                  f"CV med {d['cv']['median']:.3f}; "
                  f"|z| {m['start']:.3f}->{m['trough']:.3f} @ {m['trough_step']:2d} "
                  f"->{m['end']:.3f}; "
                  f"step size {s['first']:.4f}->{s['max']:.4f} @ {s['max_step']:2d} "
                  f"(x{s['max_over_min']:.2f})")
        print(f"  written to {args.step_profiles}")

    print("=== 5.1 / 5.2  shape summary per (model, dataset) ===")
    for model, dim in (("flux", FLUX_D), ("qwen", QWEN_D)):
        for dataset in DATASETS:
            frame = load(model, dataset)
            if frame is None:
                continue
            s = shape_summary(frame)
            print(f"  {model:5s} {dataset:22s} n={s['n']:6d} chord={s['chord_med']:6.1f} "
                  f"dev={s['dev_med']:.3f} [{s['dev_q05']:.3f},{s['dev_q95']:.3f}] "
                  f"dev/|z_final|={s['dev_over_data']:.3f} straight={s['straightness']:.4f} "
                  f"evr2={s['evr2_med']:.3f} (q05 {s['evr2_q05']:.3f}) "
                  f"evr3={s['evr3_med']:.3f}")
            print(f"        evr components "
                  + "/".join(f"{v:.3f}" for v in s["evr_components"])
                  + f"   recon rel 2d={s['recon_rel_2d']:.3f} 3d={s['recon_rel_3d']:.3f}"
                  + f"   distinct z_T={s['z_T_clusters']:6d}"
                  + f"   sigma grids={s['sigma_grids']}")

    # DDB is the per-step section's dataset on both backbones: 10,000 prompts x
    # 3 seeds is the only cell big enough to split the per-step variance into a
    # prompt part and a seed part without the estimate being mostly noise.
    for model, dim in (("flux", FLUX_D), ("qwen", QWEN_D)):
        frame = load(model, "diffusiondb_clean10k")
        if frame is None:
            continue
        d = dispersion(frame)
        print(f"\n=== 4.1 / 4.2 per-step dispersion ({model} DDB, n={len(frame)}) ===")
        print(f"  d_perp/chord CV: med={d['cv_med']:.3f} "
              f"range {d['cv_min']:.3f}-{d['cv_max']:.3f}")
        print("  between-prompt variance share, steps 1-6: "
              + " ".join(f"{v:.3f}" for v in d["share_early"]))
        print(f"  between-prompt variance share, steps 7-48: "
              f"{d['share_late_min']:.3f}-{d['share_late_max']:.3f}")
        v = velocity_law(frame, dim=dim)
        print(f"=== 3.4 velocity law ({model} DDB) ===")
        print(f"  median profile {v['profile_min']:.3f}-{v['profile_max']:.3f} "
              f"(spread {v['profile_spread']:.3f}); per-trajectory spread "
              f"med={v['per_traj_spread_med']:.3f} "
              f"(excl. step 0 {v['per_traj_spread_med_no_step0']:.3f}); "
              f"cross-prompt CV={v['cross_prompt_cv']:.3f}")

    # These two are the 30-stored-trajectory readings. Both are superseded at
    # scale by docs/full_trajectory_shape_results.md (2,880 generations); they
    # stay here because F1/F2 are drawn from the same stored trajectories and
    # the two tiers have to keep agreeing.
    for model in ("flux", "qwen"):
        paths = sorted(glob.glob(LATENTS.format(model=model)))
        if not paths:
            continue
        centers, med, _ = curvature_profile(paths)
        print(f"\n=== curvature profile ({model}, w={CURVATURE_WINDOW}, "
              f"{len(paths)} trajectories; superseded by the shape doc) ===")
        print("  " + "  ".join(f"c{c}={v:.1f}" for c, v in zip(centers, med) if c % 5 == 0))
        a = plane_alignment(paths)
        print(f"=== plane alignment ({model}, {a['pairs']} pairs; superseded) ===")
        print(f"  principal angles med: {a['theta1_med']:.2f} / {a['theta2_med']:.2f} deg"
              f"   random-plane null: {a['null_theta1_med']:.2f} / {a['null_theta2_med']:.2f} deg")

    if args.figures is not None:
        write_figures(args.figures)


def write_figures(out_dir: Path) -> None:
    """F0 (resolution limit), F1 (per-step geometry), F2 (how much of it is
    common to every prompt), F3 (3-D overlay)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_turn_angle_floor(out_dir, plt)

    for model in ("flux", "qwen"):
        latent_paths = sorted(glob.glob(LATENTS.format(model=model)))
        if not latent_paths:
            continue
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        for path in latent_paths:
            Z = load_trajectory(path)
            frame = _frame_or_die(path)
            ax.plot((Z - Z[0]) @ frame[0], (Z - Z.mean(axis=0)) @ frame[1],
                    (Z - Z.mean(axis=0)) @ frame[2], lw=1.0, alpha=0.75)
        ax.set_xlabel("along chord"); ax.set_ylabel("PC1"); ax.set_zlabel("PC2")
        ax.set_title(f"{model.upper()}: {len(latent_paths)} denoising trajectories "
                     "(DrawBench, one seed)\neach rotated into its own frame: "
                     "start-to-end line + the 2 directions it curves in", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_traj_3d_overlay.png", dpi=140)
        plt.close(fig)

    _write_step_geometry(out_dir, plt)
    _write_shape_by_dataset(out_dir, plt)
    _write_step_dispersion(out_dir, plt)
    print(f"figures written to {out_dir}")


def _write_turn_angle_floor(out_dir: Path, plt) -> None:
    """F0: where this data stops being able to resolve geometry.

    The single-step turn angle plotted against the angle bf16 rounding alone
    produces. The two curves lie on top of each other for most of the path, and
    that is the statement: it fixes the resolution limit that makes every other
    per-step reading in the doc checkable.
    """
    from analysis.trajectory_bf16_floor import floor_profile

    colors = dict(zip(DATASETS, ["#1f77b4", "#2ca02c", "#9467bd", "#d62728"]))
    names = {"drawbench_full": "DrawBench", "geneval_style": "GenEval",
             "parti_full": "Parti", "diffusiondb_clean10k": "DiffusionDB"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    drawn = False
    for ax, model in zip(axes, ("flux", "qwen")):
        for dataset in DATASETS:
            frame = load(model, dataset)
            if frame is None:
                continue
            arr = _stack(frame, "turn_angle_deg")
            ax.plot(np.arange(arr.shape[1]), np.median(arr, axis=0),
                    color=colors[dataset], lw=1.6,
                    label=f"{names[dataset]} ({PROMPTS[dataset]:,} prompts x 3 seeds)")
            drawn = True
            del frame, arr
        # each backbone gets its OWN floor: the rounding error scales with the
        # latent's magnitude and dimension, so FLUX's dashed line would
        # misstate where Qwen's readings stop being geometry
        paths = sorted(glob.glob(LATENTS.format(model=model)))
        if paths:
            floors = np.median(
                np.array([floor_profile(load_trajectory(p), 1) for p in paths]), axis=0)
            ax.plot(np.arange(len(floors)), floors, "k--", lw=1.4,
                    label=f"bf16 rounding alone ({len(paths)} trajectories)")
        ax.set_yscale("log")
        # the legend sits over the curve's own high shoulder at junction 0, so
        # the axis is opened above the data rather than autoscaled onto it
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 3.2)
        ax.set_xlabel("junction n (between step n and step n+1)")
        ax.set_ylabel("turn angle over one step (degrees)")
        ax.set_title(model.upper(), fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    if not drawn:
        plt.close(fig)
        return
    fig.suptitle("The resolution limit of this data: over one step, the measured turn angle "
                 "is the rounding error\n"
                 "solid = median measured angle, dashed = angle bf16 rounding produces on a "
                 "path that is exactly straight", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / "turn_angle_floor.png", dpi=140)
    plt.close(fig)


# each profile gets its own file: the three are read one at a time, and packing
# them into one sheet makes every panel small enough that the thing being shown
# -- four curves lying on one another -- stops being legible
def _write_step_geometry(out_dir: Path, plt) -> None:
    """F1: what the path does at each of the 50 steps.

    Four profiles the tables have always carried and nobody has ever plotted:
    how far off the straight line, how far one step moves, how fast the model
    is driving, how big the state is. The two backbones share axes so the
    shapes can be compared rather than two spreads quoted against each other.
    """
    # the x axis is NOT the same object on all four: deviation and magnitude are
    # properties of a state (0..50), spacing and speed of a solver step (0..49).
    # One shared label would quietly invite reading a state index off a step
    # panel, so each panel carries its own.
    panels = [
        ("dev_over_chord", "distance from the straight line\n(fraction of its length)", "dev",
         "state n (0 = the initial noise)"),
        ("spacing_over_chord", "distance moved by one step\n(fraction of the chord)", "spacing",
         "solver step n (state n -> n+1)"),
        ("velocity_over_sqrt_d", "speed of the path\n(velocity norm / sqrt(d))", "velocity",
         "solver step n (state n -> n+1)"),
        ("magnitude_over_sqrt_d", "size of the state\n(norm / sqrt(d))", "magnitude",
         "state n (0 = the initial noise)"),
    ]
    for dataset in DATASETS:
        _step_geometry_one(out_dir, plt, panels, dataset)


def _step_geometry_one(out_dir: Path, plt, panels, dataset: str) -> None:
    """One dataset's copy of F1. Same code, same panels, same styling for all
    four, so the only thing that differs between the four sheets is the data."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    drawn = False
    for rank, model in enumerate(("qwen", "flux")):
        frame = load(model, dataset)
        if frame is None:
            continue
        cell = step_profiles(frame, dim=DIM[model])
        chord = frame["chord_len"].to_numpy()
        raw = {"dev_over_chord": _stack(frame, "d_perp") / chord[:, None],
               "spacing_over_chord": _stack(frame, "spacing") / chord[:, None],
               "velocity_over_sqrt_d": _stack(frame, "velocity_norm") / math.sqrt(DIM[model]),
               "magnitude_over_sqrt_d": _stack(frame, "magnitude") / math.sqrt(DIM[model])}
        color, dash = STYLES[model]
        label = f"{model.upper()} (n={cell['n']:,})"
        for ax, (key, ylabel, readings, _xlabel) in zip(axes, panels):
            med = np.array(cell["profiles"][key], dtype=float)
            x = np.arange(len(med))
            ax.plot(x, med, color=color, ls=dash, lw=1.8, label=label)
            ax.fill_between(x, np.quantile(raw[key], .25, axis=0),
                            np.quantile(raw[key], .75, axis=0), color=color, alpha=0.13)
            ax.set_ylabel(ylabel)
            # the band is narrower than the line on two of the three panels, so
            # the number that says how narrow goes on the panel itself
            ax.text(0.03, 0.06 + 0.07 * rank, f"{model.upper()}: spread across prompts, "
                    f"median {cell[readings]['cv']['median'] * 100:.1f}%",
                    transform=ax.transAxes, color=color, fontsize=8.5)
            if key == "dev_over_chord":
                peak = cell["dev"]["peak_step"]
                ax.plot([peak], [med[peak]], "o", color=color, ms=6)
                ax.annotate(f"step {peak}", (peak, med[peak]), color=color,
                            textcoords="offset points", xytext=(-4, 8), ha="right", fontsize=9)
        drawn = True
        del frame, raw
    if not drawn:
        plt.close(fig)
        return
    for ax, panel in zip(axes, panels):  # clear the strip the spread numbers sit in
        low, high = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if t >= 0])  # all four are norms
        ax.set_ylim(low - 0.22 * (high - low), high)
        ax.set_xlabel(panel[3])
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=9)
    n_traj = PROMPTS[dataset] * 3
    fig.suptitle(
        f"What the denoising path does at each of the 50 steps ({DATASET_NAMES[dataset]}, "
        f"{PROMPTS[dataset]:,} prompts x 3 seeds per model)\n"
        f"line = median over all {n_traj:,} trajectories of that backbone, "
        "band = the middle 50% of them at that step", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"step_geometry_{dataset}.png", dpi=140)
    plt.close(fig)


def _write_shape_by_dataset(out_dir: Path, plt) -> None:
    """F3a: the whole-trajectory shape scalars, on every dataset and seed.

    Section 5's numbers are computed on all 74,310 trajectories but appear only
    as min-max columns, and the only picture in that section is a
    30-trajectory illustration from one dataset and one seed. These are the four
    scalars that section reports, drawn on the grid they were computed on.

    Box = the spread over that cell's individual trajectories, which is what
    says whether a difference between datasets is large next to the variation
    between trajectories inside one. The three dots on each box are the three
    seeds' medians: if one of these scalars answered to the seed rather than to
    the content, those three would spread as far as the four datasets do.
    """
    panels = [
        ("straightness", "path length / chord length\n(1.000 = perfectly straight)"),
        ("max_dev_ratio", "furthest distance from the line\n(fraction of the chord)"),
        ("evr_top2", "share of the deviation in its\nleading two directions"),
        ("recon_err_rel_3d", "residual after chord + 2 directions\n(relative to chord alone)"),
    ]
    values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for model in ("flux", "qwen"):
        for dataset in DATASETS:
            frame = load(model, dataset)
            if frame is None:
                continue
            values[(model, dataset)] = {
                "straightness": frame["straightness"].to_numpy(dtype=float),
                "max_dev_ratio": frame["max_dev_ratio"].to_numpy(dtype=float),
                "evr_top2": np.array([e[0] + e[1] for e in frame["pca_evr"]], dtype=float),
                "recon_err_rel_3d": frame["recon_err_rel_3d"].to_numpy(dtype=float),
                "seed": frame["seed"].to_numpy(),
            }
            del frame
    if not values:
        return

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    offset = {"flux": -0.19, "qwen": 0.19}
    for ax, (key, ylabel) in zip(axes, panels):
        for model in ("flux", "qwen"):
            color = STYLES[model][0]
            for i, dataset in enumerate(DATASETS):
                cell = values.get((model, dataset))
                if cell is None:
                    continue
                pos = i + offset[model]
                ax.boxplot([cell[key]], positions=[pos], widths=0.30, showfliers=False,
                           medianprops=dict(color=color, lw=1.8),
                           boxprops=dict(color=color), whiskerprops=dict(color=color),
                           capprops=dict(color=color))
                for seed in np.unique(cell["seed"]):
                    ax.plot([pos], [np.median(cell[key][cell["seed"] == seed])], "o",
                            color=color, ms=3.5, mfc="white", mew=1.1, zorder=3)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DATASET_NAMES[d] for d in DATASETS], fontsize=9)
        ax.set_xlim(-0.6, len(DATASETS) - 0.4)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], color=STYLES[m][0], lw=2, label=m.upper())
               for m in ("flux", "qwen")]
    handles.append(plt.Line2D([], [], color="0.35", marker="o", ls="", ms=4, mfc="white",
                              mew=1.1, label="one seed's median"))
    axes[0].legend(handles=handles, fontsize=8.5, loc="best")
    fig.suptitle(
        "The whole-trajectory shape, on all four datasets and all three seeds "
        "(74,310 trajectories)\n"
        "box = spread over the individual trajectories of that cell, "
        "dots = the three seeds' medians", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / "shape_by_dataset.png", dpi=140)
    plt.close(fig)


def _write_step_dispersion(out_dir: Path, plt) -> None:
    """F2: how much of the per-step geometry is the same for every prompt.

    Left: the per-step spread of each profile across 10,000 prompts x 3 seeds.
    Right: how much of that spread is a prompt's own reproducible signature
    rather than seed noise -- a per-step quantity with real structure, which
    the doc used to carry as a pair of endpoint numbers in a subordinate clause.
    """
    profiles = [("dev_over_chord_cv", "distance from the line", "-"),
                ("spacing_over_chord_cv", "distance moved by one step", "--"),
                ("magnitude_over_sqrt_d_cv", "size of the state", ":"),
                ("velocity_over_sqrt_d_cv", "speed of the path", "-.")]
    payload = json.loads(STEP_PROFILES_JSON.read_text()) if STEP_PROFILES_JSON.is_file() else None
    if payload is None:
        print(f"  (skipping F2: {STEP_PROFILES_JSON} not found; run --step-profiles first)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for model in ("flux", "qwen"):
        cell = payload["cells"].get(f"{model}/diffusiondb_clean10k")
        if cell is None:
            continue
        color = STYLES[model][0]
        for key, name, dash in profiles:
            y = np.array([np.nan if v is None else v for v in cell["profiles"][key]],
                         dtype=float)
            axes[0].plot(np.arange(len(y)), y, color=color, ls=dash, lw=1.5,
                         label=f"{model.upper()}: {name}")
        share = np.array([np.nan if v is None else v for v in
                          cell["profiles"]["dev_prompt_variance_share"]], dtype=float)
        axes[1].plot(np.arange(len(share)), share, color=color, ls=STYLES[model][1], lw=1.8,
                     label=f"{model.upper()} (n={cell['n']:,}, "
                           f"{cell['n_prompts']:,} prompts x {cell['n_seeds']} seeds)")
    axes[0].set_ylabel("spread across prompts and seeds at that step\n"
                       "(coefficient of variation)")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.85)  # room for the 8-entry legend
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper center")
    axes[1].set_ylabel("share of that spread which is a prompt's own\n"
                       "reproducible signature (rest is seed noise)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        # both panels read the deviation profile, which is a per-state quantity
        ax.set_xlabel("state n (0 = the initial noise)")
        ax.grid(alpha=0.3)
    fig.suptitle("How much of the per-step geometry is the same for every prompt "
                 "(DiffusionDB, 10,000 prompts x 3 seeds per model)\n"
                 "states 0 and 50 lie on the chord by construction, so the deviation "
                 "curves have nothing to report there", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / "step_dispersion.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
