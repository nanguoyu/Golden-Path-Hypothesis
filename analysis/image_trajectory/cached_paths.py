#!/usr/bin/env python3
"""The image cache-bend layer: how a cached run's path leaves the reference's.

`docs/image_cached_trajectory_plan_zh.md` sections 7.2 and 7.3. One pair is a
cached path `Z^c` and the no-cache path `Z^r` of the same (model, prompt_idx)
at seed 42, both `[51, d]`. Everything below is defined exactly as on the video
side (`docs/video_full_trajectory_plan_zh.md` section 3.9.3), so the two
modalities can sit in one table:

  D[n]            ``||Z^c[n] - Z^r[n]||``, divided by the reference chord
  direction       the difference vector's energy along the reference chord,
                  inside the reference bend plane, and off it
  scalar diffs    chord ratio, straightness difference, max-deviation
                  difference, difference of the top-2 chord-orthogonal EVR
  event alignment ``Delta D[k+j] = D[k+j+1] - D[k+j]`` with the origin at each
                  cache step, stratified by what step ``k+j`` itself is
  early offset    ``D[10]`` and the area of ``D`` over ``n <= 10`` — the image
                  side's own reading, the left-hand side of Q2

The pair-level arithmetic is imported from the video pipeline rather than
copied: `cached_pair_readings`, `landmark_payload`, `first_cache_step`,
`plane_rows`, `angle_between_deg`, `first_principal_angle` and
`CachedEventAccumulator` are plain ndarray functions and are the definitions
themselves. What is written here is the image side's own: the SPX cell naming,
the three image decisions schemas, and the endpoint block (the video's endpoint
carries a float32 T1 chord beside the bf16 one, and the image layer has one
store, so its field names would claim a distinction that does not exist here).

Subcommands
-----------
``inventory``  what is on disk, per cell, against the frozen manifest
``cell``       one cell -> ``cells/<cell_id>.json``
``floors``     the four floor readings of section 7.3
``merge``      every cell json -> ``cached_bend_<model>.json`` + the per-pair
               early/quality join table

CPU only; ``torch`` is imported lazily to open the ``.pt`` files. Runs as an
sbatch job, never on a login node.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.trajectory_math import (  # noqa: E402
    chord, deviation_profile, orthogonal_pca, path_length, plane_frame,
)
from analysis.video_trajectory.cached_vs_reference import (  # noqa: E402
    NUM_STEPS, summarise_profile,
)
from analysis.video_trajectory.latent_paths import (  # noqa: E402
    BF16_REL_RMS, CachedEventAccumulator, angle_between_deg,
    cached_pair_readings, first_cache_step, first_principal_angle,
    landmark_payload, plane_rows,
)

N_STATES = NUM_STEPS + 1
LANDMARKS = (25, 40, 50)  # plus each pair's own k0 + 1
EARLY_N = 10  # the "early" window of Q2: states 1..10
DEFAULT_DATA_ROOT = Path("outputs")
MANIFEST = _ROOT / "resources" / "image_trajectory" / "cells.v1.tsv"
SAMPLE = _ROOT / "resources" / "image_trajectory" / "prompt_sample.v1.json"
EXCLUSIONS = _ROOT / "resources" / "image_trajectory" / "prompt_exclusions.v1.json"

CAV_EARLY = (
    "a row whose first cache step is later than state 10 has D[n] = 0 for every "
    "n <= 10 by construction, not by measurement; those rows are the control "
    "group of the first-jump hypothesis and are reported as a labelled stratum, "
    "never averaged in as if they were small readings"
)
CAV_FRAME = (
    "the direction frame is recomputed from the bf16 reference path, not stored "
    "in flight; the fp32 dual-store subset bounds what that costs the frame's "
    "orientation"
)
CAV_EXCLUDED = (
    "prompt indices the replay could not reproduce are dropped before any "
    "reading is taken, including the event-alignment buckets, so every number "
    "here is over one population; which indices, and why, is recorded in "
    "resources/image_trajectory/prompt_exclusions.v1.json"
)
CAV_NORM = (
    "D is divided by the same pair's reference chord length, measured on the "
    "same bf16 store as D itself, so the ratio has one store and not two"
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["k"] = int(row["k"])
        row["k_realized"] = int(row["k_realized"])
    return rows


def load_sample(path: Path = SAMPLE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def excluded_prompts(path: Path = EXCLUSIONS) -> set[int]:
    """Prompt indices the replay could not reproduce, with the reason recorded.

    The frozen draw is never edited — it is what was drawn. An index that the
    byte check found the replay does not reproduce is filtered here instead, so
    the draw, the failure and the disposition each stay in one place.
    """
    path = Path(path)
    if not path.is_file():
        return set()
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {int(entry["prompt_idx"]) for entry in blob.get("excluded", [])}


def load_path(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """One retained path as float32 `[51, d]`, plus its manifest fields."""
    import torch

    blob = torch.load(path, weights_only=True, map_location="cpu")
    Z = blob["path"].to(torch.float32).numpy()
    meta = {k: v for k, v in blob.items() if k != "path"}
    if Z.shape[0] != N_STATES:
        raise SystemExit(f"{path}: {Z.shape[0]} states, expected {N_STATES}")
    return Z, meta


def actions_from_decisions(path: Path) -> str:
    """The 50-character action string of ONE generation, from its own decisions.

    Three schemas write image decisions: the FLUX SPX residual/DiCache adapters
    (`per_step`), the Qwen SPX adapters (`steps`), and the locked MeanCache
    adapters, which keep their own schema. All three carry `cache_steps`, and
    the per-step list is what the run actually did — both are read and required
    to agree, because a silent disagreement between them is exactly the failure
    the byte check exists to catch.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = record.get("per_step") or record.get("steps") or record.get("records")
    declared = record.get("cache_steps")
    bits: str | None = None
    if rows:
        if len(rows) != NUM_STEPS:
            raise SystemExit(f"{path}: {len(rows)} step rows, expected {NUM_STEPS}")
        bits = "".join("1" if r.get("action") == "cache" else "0" for r in rows)
    if declared is not None:
        from_declared = "".join(
            "1" if step in set(int(v) for v in declared) else "0"
            for step in range(NUM_STEPS)
        )
        if bits is not None and bits != from_declared:
            raise SystemExit(
                f"{path}: per-step actions and cache_steps disagree")
        bits = bits or from_declared
    if bits is None:
        raise SystemExit(f"{path}: no per-step actions and no cache_steps")
    return bits


def schedule_bits(schedule_file: str) -> str:
    return (_ROOT / schedule_file).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# per-pair readings
# ---------------------------------------------------------------------------


def _series(values: Iterable[float]) -> list[float | None]:
    return [None if not math.isfinite(float(v)) else float(v) for v in values]


def path_scalars(Z: np.ndarray) -> dict[str, float]:
    """The four shape scalars the plan differences between the two paths."""
    _, chord_len = chord(Z)
    _, _, max_dev = deviation_profile(Z)
    _, straightness = path_length(Z)
    pca = orthogonal_pca(Z)
    return {
        "chord_len": float(chord_len),
        "straightness": float(straightness),
        "max_dev_ratio": float(max_dev),
        "pca_evr_top2": float(pca["pca_evr"][0] + pca["pca_evr"][1]),
    }


def endpoint_block(Zc: np.ndarray, Zr: np.ndarray, Fr: np.ndarray | None,
                   D: np.ndarray, norm_r: np.ndarray, chord_r: float
                   ) -> dict[str, Any]:
    """`D[50]` in two normalisations, the chord angle, and the plane angle.

    One store on both sides, so there is one reference chord and one chord
    direction; the video's `_t1` / `_t3` pair of names has nothing to name here.
    """
    cached_chord = Zc[-1].astype(np.float64) - Zc[0].astype(np.float64)
    ref_chord = Zr[-1].astype(np.float64) - Zr[0].astype(np.float64)
    theta1, theta2 = first_principal_angle(
        plane_rows(Zc), None if Fr is None else Fr[1:3])
    ang = angle_between_deg(cached_chord, ref_chord)
    return {
        "D50": float(D[-1]),
        "D50_over_chord_ref": float(D[-1] / chord_r) if chord_r > 0 else None,
        "D50_over_norm_ref": float(D[-1] / norm_r[-1]) if norm_r[-1] > 0 else None,
        "chord_angle_deg": ang if math.isfinite(ang) else None,
        "plane_angle1_deg": theta1 if math.isfinite(theta1) else None,
        "plane_angle2_deg": theta2 if math.isfinite(theta2) else None,
        "cached_chord_len": float(np.linalg.norm(cached_chord)),
        "ref_chord_len": float(chord_r),
    }


def early_block(D: np.ndarray, chord_r: float, k0: int | None) -> dict[str, Any]:
    """Q2's left-hand side: the early offset of one pair.

    `D10` is the offset at state 10 and `early_auc` the sum over states 1..10,
    both divided by the reference chord. A row whose first cache step is at or
    after state 10 has both exactly 0 — by definition, and that is the point
    (`CAV_EARLY`); the flag says so rather than leaving a reader to infer it.
    """
    scale = chord_r if chord_r > 0 else float("nan")
    return {
        "D10_over_chord_ref": float(D[EARLY_N] / scale),
        "early_auc_over_chord_ref": float(D[1:EARLY_N + 1].sum() / scale),
        "structurally_zero_early": bool(k0 is not None and k0 >= EARLY_N),
        "k0": None if k0 is None else int(k0),
    }


def first_jump_block(D: np.ndarray, chord_r: float, k0: int | None,
                     sigmas: list[float]) -> dict[str, Any]:
    """Q4's main reading: the step the first cache event puts into D, and the
    growth over the five steps after it, beside the sigma geometry at k0."""
    if k0 is None:
        return {"delta_D_k0": None, "slope_k0_plus_5": None,
                "sigma_k0": None, "sigma_step_k0": None}
    scale = chord_r if chord_r > 0 else float("nan")
    delta = float((D[k0 + 1] - D[k0]) / scale)
    hi = min(k0 + 5, NUM_STEPS)
    slope = float((D[hi] - D[k0 + 1]) / scale / max(hi - (k0 + 1), 1))
    return {
        "delta_D_k0": delta,
        "slope_k0_plus_5": slope,
        "sigma_k0": float(sigmas[k0]) if k0 < len(sigmas) else None,
        "sigma_step_k0": (float(sigmas[k0] - sigmas[k0 + 1])
                          if k0 + 1 < len(sigmas) else None),
    }


def pair_record(*, prompt_idx: int, role: str, Zc: np.ndarray, Zr: np.ndarray,
                Fr: np.ndarray | None, actions: str, sigmas: list[float],
                ref_scalars: dict[str, float]) -> dict[str, Any]:
    D, norm_r, shares = cached_pair_readings(Zc, Zr, Fr)
    chord_r = float(ref_scalars["chord_len"])
    k0 = first_cache_step(actions)
    n_prefix = N_STATES if k0 is None else min(k0 + 1, N_STATES)
    prefix_max = float(np.max(D[:n_prefix])) if n_prefix > 0 else float("nan")
    cached_scalars = path_scalars(Zc)
    landmarks = {"k0_plus_1": landmark_payload(
        None if k0 is None else k0 + 1, D, norm_r, shares, chord_r)}
    for n in LANDMARKS:
        landmarks[str(n)] = landmark_payload(n, D, norm_r, shares, chord_r)
    return {
        "prompt_idx": int(prompt_idx),
        "role": role,
        "k0": k0,
        "n_cached": actions.count("1"),
        "D_over_chord_ref": _series(D / chord_r) if chord_r > 0
        else [None] * N_STATES,
        "prefix": {
            "n_states_compared": int(n_prefix),
            "has_cache_step": k0 is not None,
            "max_abs_D": prefix_max if math.isfinite(prefix_max) else None,
            "exactly_zero": bool(n_prefix > 0 and prefix_max == 0.0),
        },
        "endpoint": endpoint_block(Zc, Zr, Fr, D, norm_r, chord_r),
        "early": early_block(D, chord_r, k0),
        "first_jump": first_jump_block(D, chord_r, k0, sigmas),
        "landmarks": landmarks,
        "scalar_diff": {
            "chord_ratio": (cached_scalars["chord_len"] / chord_r
                            if chord_r > 0 else None),
            "straightness_diff": (cached_scalars["straightness"]
                                  - ref_scalars["straightness"]),
            "max_dev_ratio_diff": (cached_scalars["max_dev_ratio"]
                                   - ref_scalars["max_dev_ratio"]),
            "pca_evr_top2_diff": (cached_scalars["pca_evr_top2"]
                                  - ref_scalars["pca_evr_top2"]),
        },
        "_D": D,  # kept in memory for the event accumulator, dropped on write
    }


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------


class ReferenceCache:
    """The reference paths, their frames and their shape scalars, kept once.

    A cell walks 50 prompts and every one of them needs the same reference path,
    its `[chord, PC1, PC2]` frame and its four shape scalars. Recomputing the
    frame per pair would triple the cost of the cell; holding one model's whole
    sample is 1.3 GB (FLUX) / 2.3 GB (Qwen) as float32.
    """

    def __init__(self, ref_dir: Path) -> None:
        self.ref_dir = Path(ref_dir)
        self._paths: dict[int, np.ndarray] = {}
        self._frames: dict[int, np.ndarray | None] = {}
        self._scalars: dict[int, dict[str, float]] = {}
        self._meta: dict[int, dict[str, Any]] = {}

    def get(self, idx: int) -> tuple[np.ndarray, np.ndarray | None,
                                     dict[str, float], dict[str, Any]]:
        if idx not in self._paths:
            from lib.retained_trajectory import trajectory_filename

            Z, meta = load_path(self.ref_dir / trajectory_filename(idx))
            self._paths[idx] = Z
            frame = plane_frame(Z.astype(np.float64))
            self._frames[idx] = None if frame is None else np.asarray(frame)
            self._scalars[idx] = path_scalars(Z)
            self._meta[idx] = meta
        return (self._paths[idx], self._frames[idx], self._scalars[idx],
                self._meta[idx])


def run_cell(row: dict[str, Any], *, data_root: Path, indices: list[int],
             roles: dict[int, str], refs: ReferenceCache,
             drop: set[int] | None = None) -> dict[str, Any]:
    from lib.retained_trajectory import trajectory_filename

    cell_dir = data_root / "image_cached_traj" / row["cell_dir"]
    want_bits = schedule_bits(row["schedule_file"])
    drop = drop or set()
    kept_indices = [i for i in indices if i not in drop]
    events = CachedEventAccumulator()
    pairs: list[dict[str, Any]] = []
    missing: list[int] = []
    action_mismatch: list[int] = []
    for pair_id, idx in enumerate(kept_indices):
        traj = cell_dir / trajectory_filename(idx)
        dec = cell_dir / f"decisions_{idx:05d}.json"
        if not traj.is_file() or not dec.is_file():
            missing.append(idx)
            continue
        actions = actions_from_decisions(dec)
        if actions != want_bits:
            action_mismatch.append(idx)
            continue
        Zc, cmeta = load_path(traj)
        Zr, Fr, ref_scalars, rmeta = refs.get(idx)
        if Zc.shape != Zr.shape:
            raise SystemExit(
                f"{row['cell_id']} idx {idx}: cached {Zc.shape} vs reference {Zr.shape}")
        if cmeta.get("z_T_sha256") != rmeta.get("z_T_sha256"):
            raise SystemExit(
                f"{row['cell_id']} idx {idx}: cached and reference started from "
                f"different noise; the pair is not a pair")
        rec = pair_record(prompt_idx=idx, role=roles.get(idx, "unknown"),
                          Zc=Zc, Zr=Zr, Fr=Fr, actions=actions,
                          sigmas=list(cmeta.get("sigmas") or []),
                          ref_scalars=ref_scalars)
        D = rec.pop("_D")
        events.add(actions, np.diff(D), rec["k0"], ref_scalars["chord_len"],
                   pair_id)
        pairs.append(rec)
        del Zc, D

    profile = np.asarray(
        [[np.nan if v is None else v for v in p["D_over_chord_ref"]] for p in pairs],
        dtype=np.float64).reshape(-1, N_STATES)
    return {
        "schema": "image_trajectory.cached_cell.v1",
        "cell_id": row["cell_id"],
        "model": row["model"],
        "k": row["k"],
        "k_realized": row["k_realized"],
        "schedule": row["schedule"],
        "payload": row["payload"],
        "schedule_bits": want_bits,
        "k0_row": first_cache_step(want_bits),
        "n_pairs": len(pairs),
        "n_requested": len(kept_indices),
        "n_excluded": len(indices) - len(kept_indices),
        "excluded_prompt_indices": sorted(drop),
        "missing_prompt_idx": missing,
        "action_mismatch_prompt_idx": action_mismatch,
        "D_over_chord_ref_profile": summarise_profile(profile),
        "events": events.summary(),
        "pairs": pairs,
        "caveats": [CAV_EARLY, CAV_FRAME, CAV_NORM, CAV_EXCLUDED],
    }


# ---------------------------------------------------------------------------
# floors (plan section 7.3)
# ---------------------------------------------------------------------------


def _round_trip_bf16(Z32: np.ndarray) -> np.ndarray:
    import torch

    return torch.from_numpy(Z32).to(torch.bfloat16).to(torch.float32).numpy()


def bf16_floor(Z32: np.ndarray) -> dict[str, float]:
    """Two routes to the bf16 store's own noise level, in chord units.

    displacement route: round the path to bf16 and back, and read the per-state
    displacement it introduces. deviation route: the same round trip's effect
    on the perpendicular deviation profile. The larger of the two is the bound
    a D reading has to clear to be a measurement.

    The input has to be a path that was STORED in float32. Rounding the bf16
    store to bf16 again is the identity and would report a floor of exactly
    zero, which is why this is measured on the fp32 dual-store subset rather
    than on the production paths.

    D is a distance between two independently rounded paths, so its store noise
    is up to sqrt(2) times a single path's rounding displacement when the two
    roundings are independent — and exactly 0 before the first cache step,
    where the two runs round the same float32 state. The single-path routes are
    reported, matching the video side's definition, with that factor stated
    rather than folded in.
    """
    Zb = _round_trip_bf16(Z32)
    _, chord_len = chord(Z32)
    disp = np.linalg.norm(Zb.astype(np.float64) - Z32.astype(np.float64), axis=1)
    dev32, _, _ = deviation_profile(Z32)
    dev16, _, _ = deviation_profile(Zb)
    dev_gap = np.abs(np.asarray(dev16) - np.asarray(dev32))
    return {
        "chord_len": float(chord_len),
        "displacement_route": float(disp.max() / chord_len),
        "deviation_route": float(dev_gap.max() / chord_len),
        "floor": float(max(disp.max(), dev_gap.max()) / chord_len),
    }


def run_floors(args: argparse.Namespace) -> None:
    from lib.retained_trajectory import trajectory_filename

    sample = load_sample(args.sample)
    out: dict[str, Any] = {"schema": "image_trajectory.floors.v1", "models": {}}
    for model in ("flux", "qwen"):
        base = args.data_root / "image_cached_traj" / model
        ref_dir = base / f"refs_parti_s{sample['base_seed']}"
        rep_dir = base / f"refs_parti_s{sample['base_seed']}_repeat"
        f32_dir = base / f"refs_parti_s{sample['base_seed']}_fp32"
        entry: dict[str, Any] = {}

        rows = []
        lossless_self = lossless_pair = compared = 0
        for idx in sample["prompt_indices"][: args.floor_paths]:
            path = f32_dir / trajectory_filename(idx)
            if not path.is_file():
                continue
            Z32 = load_path(path)[0]
            rows.append(bf16_floor(Z32))
            # the store is only lossy if the states carry bits bf16 cannot
            # hold. Both samplers run in bf16, so their states may already be
            # bf16-exact, in which case the floor is not "small" but zero, and
            # saying so needs this check rather than a zero that reads like a bug
            compared += 1
            Zb = _round_trip_bf16(Z32)
            lossless_self += int(np.array_equal(Zb, Z32))
            stored = ref_dir / trajectory_filename(idx)
            if stored.is_file():
                lossless_pair += int(np.array_equal(load_path(stored)[0], Z32))
        entry["bf16_store"] = {
            "n_paths": len(rows),
            "measured_on": "the fp32 dual-store reference subset",
            "route": "max(displacement, deviation) per path, in chord units",
            "analytic_rel_rms": BF16_REL_RMS,
            "pair_factor_note": (
                "D differences two independently rounded paths: up to sqrt(2) "
                "times a single path's displacement after k0, exactly 0 before it"),
            "median": (float(np.median([r["floor"] for r in rows])) if rows else None),
            "max": (float(np.max([r["floor"] for r in rows])) if rows else None),
            "displacement_median": (float(np.median([r["displacement_route"] for r in rows]))
                                    if rows else None),
            "deviation_median": (float(np.median([r["deviation_route"] for r in rows]))
                                 if rows else None),
            "n_paths_bf16_exact": lossless_self,
            "n_paths_equal_to_bf16_store": lossless_pair,
            "n_paths_compared": compared,
            "store_is_lossless": bool(compared and lossless_self == compared
                                      and lossless_pair == compared),
            "why": ("both samplers step in bfloat16 and the callback reads the "
                    "state the sampler itself holds, so a float32 store of that "
                    "state carries no bit a bfloat16 store would lose"),
        }

        repeats = []
        for idx in sample["prompt_indices"]:
            a, b = ref_dir / trajectory_filename(idx), rep_dir / trajectory_filename(idx)
            if a.is_file() and b.is_file():
                Za, Zb = load_path(a)[0], load_path(b)[0]
                repeats.append(float(np.max(np.abs(Za.astype(np.float64)
                                                   - Zb.astype(np.float64)))))
        entry["reproduction"] = {
            "n_paths": len(repeats),
            "max_abs_state_diff": (max(repeats) if repeats else None),
            "all_exactly_zero": bool(repeats) and max(repeats) == 0.0,
        }

        orient = []
        for idx in sample["prompt_indices"]:
            a, b = ref_dir / trajectory_filename(idx), f32_dir / trajectory_filename(idx)
            if not (a.is_file() and b.is_file()):
                continue
            Zb16, Z32 = load_path(a)[0], load_path(b)[0]
            F16 = plane_frame(Zb16.astype(np.float64))
            F32 = plane_frame(Z32.astype(np.float64))
            if F16 is None or F32 is None:
                continue
            t1, _ = first_principal_angle(np.asarray(F16)[1:3], np.asarray(F32)[1:3])
            orient.append({
                "chord_angle_deg": angle_between_deg(np.asarray(F16)[0],
                                                     np.asarray(F32)[0]),
                "plane_angle1_deg": t1,
            })
        entry["store_orientation"] = {
            "n_paths": len(orient),
            "chord_angle_deg_median": (float(np.median([o["chord_angle_deg"] for o in orient]))
                                       if orient else None),
            "chord_angle_deg_max": (float(np.max([o["chord_angle_deg"] for o in orient]))
                                    if orient else None),
            "plane_angle1_deg_median": (float(np.median([o["plane_angle1_deg"] for o in orient]))
                                        if orient else None),
            "plane_angle1_deg_max": (float(np.max([o["plane_angle1_deg"] for o in orient]))
                                     if orient else None),
        }
        out["models"][model] = entry

    _write_json(args.out_dir / "floors.json", out)
    print(json.dumps(out["models"], indent=2)[:2000])


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

PERPROMPT_COLUMNS = (
    "model", "k", "schedule", "payload", "prompt_idx", "role", "k0",
    "n_cached", "D10_over_chord_ref", "early_auc_over_chord_ref",
    "structurally_zero_early", "D50_over_chord_ref", "delta_D_k0",
    "slope_k0_plus_5", "chord_ratio", "straightness_diff",
    "max_dev_ratio_diff", "pca_evr_top2_diff", "chord_angle_deg",
    "plane_angle1_deg", "share_chord_50", "share_in_plane_50",
    "share_off_plane_50", "share_chord_k0p1", "share_in_plane_k0p1",
    "share_off_plane_k0p1",
)


def _cell_rows(cell: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair in cell["pairs"]:
        lm50 = pair["landmarks"]["50"]
        lmk0 = pair["landmarks"]["k0_plus_1"]
        rows.append({
            "model": cell["model"], "k": cell["k"],
            "schedule": cell["schedule"], "payload": cell["payload"],
            "prompt_idx": pair["prompt_idx"], "role": pair["role"],
            "k0": pair["k0"], "n_cached": pair["n_cached"],
            "D10_over_chord_ref": pair["early"]["D10_over_chord_ref"],
            "early_auc_over_chord_ref": pair["early"]["early_auc_over_chord_ref"],
            "structurally_zero_early": int(pair["early"]["structurally_zero_early"]),
            "D50_over_chord_ref": pair["endpoint"]["D50_over_chord_ref"],
            "delta_D_k0": pair["first_jump"]["delta_D_k0"],
            "slope_k0_plus_5": pair["first_jump"]["slope_k0_plus_5"],
            "chord_ratio": pair["scalar_diff"]["chord_ratio"],
            "straightness_diff": pair["scalar_diff"]["straightness_diff"],
            "max_dev_ratio_diff": pair["scalar_diff"]["max_dev_ratio_diff"],
            "pca_evr_top2_diff": pair["scalar_diff"]["pca_evr_top2_diff"],
            "chord_angle_deg": pair["endpoint"]["chord_angle_deg"],
            "plane_angle1_deg": pair["endpoint"]["plane_angle1_deg"],
            "share_chord_50": lm50["share_chord"],
            "share_in_plane_50": lm50["share_in_plane"],
            "share_off_plane_50": lm50["share_off_plane"],
            "share_chord_k0p1": lmk0["share_chord"],
            "share_in_plane_k0p1": lmk0["share_in_plane"],
            "share_off_plane_k0p1": lmk0["share_off_plane"],
        })
    return rows


def _median(values: Iterable[Any]) -> float | None:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None


def _cell_summary(cell: dict[str, Any], profile: np.ndarray) -> dict[str, Any]:
    pairs = cell["pairs"]
    prefix_ok = [p for p in pairs if p["prefix"]["has_cache_step"]]
    return {
        "cell_id": cell["cell_id"], "model": cell["model"], "k": cell["k"],
        # read off the bitstring every pair in this cell was verified against,
        # so a summary can never carry a realized count from an older schedule
        "k_realized": cell["schedule_bits"].count("1"),
        "schedule": cell["schedule"],
        "payload": cell["payload"], "k0_row": cell["k0_row"],
        "n_pairs": len(pairs),
        "prefix_identity": {
            "n_rows_with_cache_step": len(prefix_ok),
            "n_exactly_zero": sum(1 for p in prefix_ok if p["prefix"]["exactly_zero"]),
            "max_abs_D_before_k0": max(
                (p["prefix"]["max_abs_D"] for p in prefix_ok
                 if p["prefix"]["max_abs_D"] is not None), default=None),
        },
        "median": {
            "D10_over_chord_ref": _median(p["early"]["D10_over_chord_ref"] for p in pairs),
            "early_auc_over_chord_ref": _median(
                p["early"]["early_auc_over_chord_ref"] for p in pairs),
            "D50_over_chord_ref": _median(
                p["endpoint"]["D50_over_chord_ref"] for p in pairs),
            "delta_D_k0": _median(p["first_jump"]["delta_D_k0"] for p in pairs),
            "slope_k0_plus_5": _median(p["first_jump"]["slope_k0_plus_5"] for p in pairs),
            "chord_ratio": _median(p["scalar_diff"]["chord_ratio"] for p in pairs),
            "straightness_diff": _median(
                p["scalar_diff"]["straightness_diff"] for p in pairs),
            "max_dev_ratio_diff": _median(
                p["scalar_diff"]["max_dev_ratio_diff"] for p in pairs),
            "pca_evr_top2_diff": _median(
                p["scalar_diff"]["pca_evr_top2_diff"] for p in pairs),
            "chord_angle_deg": _median(p["endpoint"]["chord_angle_deg"] for p in pairs),
            "plane_angle1_deg": _median(p["endpoint"]["plane_angle1_deg"] for p in pairs),
            "share_off_plane_50": _median(
                p["landmarks"]["50"]["share_off_plane"] for p in pairs),
            "share_in_plane_50": _median(
                p["landmarks"]["50"]["share_in_plane"] for p in pairs),
            "share_chord_50": _median(
                p["landmarks"]["50"]["share_chord"] for p in pairs),
            "share_off_plane_k0p1": _median(
                p["landmarks"]["k0_plus_1"]["share_off_plane"] for p in pairs),
        },
        "D_over_chord_ref_profile": summarise_profile(profile),
        "events": cell["events"],
    }


def run_merge(args: argparse.Namespace) -> None:
    cells_dir = args.cells_dir
    files = sorted(cells_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no cell json under {cells_dir}")
    drop = excluded_prompts()
    per_model: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    n_dropped = 0
    for path in files:
        cell = json.loads(path.read_text(encoding="utf-8"))
        kept = [p for p in cell["pairs"] if p["prompt_idx"] not in drop]
        n_dropped += len(cell["pairs"]) - len(kept)
        cell["pairs"] = kept
        profile = np.asarray(
            [[np.nan if v is None else v for v in p["D_over_chord_ref"]]
             for p in kept], dtype=np.float64).reshape(-1, N_STATES)
        per_model.setdefault(cell["model"], []).append(_cell_summary(cell, profile))
        rows.extend(_cell_rows(cell))
    if drop:
        print(f"[image-traj] excluded prompt indices {sorted(drop)}: "
              f"{n_dropped} pairs dropped ({EXCLUSIONS.name})")

    args.out_tables.mkdir(parents=True, exist_ok=True)
    for model, summaries in sorted(per_model.items()):
        payload = {
            "schema": "image_trajectory.cached_bend.v1",
            "model": model,
            "n_cells": len(summaries),
            "n_pairs": sum(s["n_pairs"] for s in summaries),
            "landmarks": list(LANDMARKS),
            "early_n": EARLY_N,
            "excluded_prompt_indices": sorted(drop),
            "n_pairs_dropped_by_exclusion": n_dropped,
            "caveats": [CAV_EARLY, CAV_FRAME, CAV_NORM, CAV_EXCLUDED],
            "cells": sorted(summaries, key=lambda s: s["cell_id"]),
        }
        _write_json(args.out_tables / f"cached_bend_{model}.json", payload)
        print(f"[image-traj] {model}: {len(summaries)} cells, "
              f"{payload['n_pairs']} pairs")

    out = args.out_tables / "perprompt_bend.tsv.gz"
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    with gzip.open(tmp, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PERPROMPT_COLUMNS),
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k])
                             for k in PERPROMPT_COLUMNS})
    tmp.replace(out)
    print(f"[image-traj] wrote {out} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# inventory / CLI
# ---------------------------------------------------------------------------


def run_inventory(args: argparse.Namespace) -> None:
    from lib.retained_trajectory import trajectory_filename

    sample = load_sample(args.sample)
    indices = sample["prompt_indices"]
    rows = load_manifest(args.manifest)
    report: list[dict[str, Any]] = []
    for row in rows:
        cell_dir = args.data_root / "image_cached_traj" / row["cell_dir"]
        have = sum(1 for i in indices
                   if (cell_dir / trajectory_filename(i)).is_file())
        imgs = sum(1 for i in indices if (cell_dir / f"img_{i}.png").is_file())
        report.append({"cell_id": row["cell_id"], "model": row["model"],
                       "k": row["k"], "latents": have, "images": imgs,
                       "want": len(indices)})
    done = [r for r in report if r["latents"] == r["want"]]
    print(f"[image-traj] complete cells: {len(done)}/{len(report)}")
    for r in report:
        if r["latents"] != r["want"]:
            print(f"  incomplete {r['cell_id']}: {r['latents']}/{r['want']} latents, "
                  f"{r['images']}/{r['want']} images")
    if args.out_dir is not None:
        _write_json(args.out_dir / "inventory.json",
                    {"schema": "image_trajectory.inventory.v1", "cells": report})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_cells(args: argparse.Namespace) -> None:
    sample = load_sample(args.sample)
    indices = sample["prompt_indices"]
    roles = dict(zip(indices, sample["roles"]))
    rows = load_manifest(args.manifest)
    wanted = [r for r in rows if r["schedule"] != "refnone"]
    if args.model:
        wanted = [r for r in wanted if r["model"] == args.model]
    if args.only:
        keep = set(args.only.split(","))
        wanted = [r for r in wanted if r["cell_id"] in keep]
    wanted = [r for i, r in enumerate(wanted) if i % args.groups == args.task]
    if not wanted:
        print("[image-traj] no cells selected")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drop = excluded_prompts()
    if drop:
        print(f"[image-traj] excluding prompt indices {sorted(drop)}")
    caches: dict[str, ReferenceCache] = {}
    for row in wanted:
        out = args.out_dir / f"{row['cell_id']}.json"
        if out.is_file() and not args.force:
            print(f"[image-traj] skip {row['cell_id']} (already written)")
            continue
        if row["model"] not in caches:
            caches[row["model"]] = ReferenceCache(
                args.data_root / "image_cached_traj" / row["model"]
                / f"refs_parti_s{sample['base_seed']}")
        cell = run_cell(row, data_root=args.data_root, indices=indices,
                        roles=roles, refs=caches[row["model"]], drop=drop)
        _write_json(out, cell)
        print(f"[image-traj] {row['cell_id']}: {cell['n_pairs']} pairs, "
              f"missing={len(cell['missing_prompt_idx'])}, "
              f"action_mismatch={len(cell['action_mismatch_prompt_idx'])}",
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory")
    inv.add_argument("--out_dir", type=Path, default=None)
    inv.set_defaults(func=run_inventory)

    cells = sub.add_parser("cell")
    cells.add_argument("--out_dir", type=Path, required=True)
    cells.add_argument("--model", default=None)
    cells.add_argument("--only", default=None)
    cells.add_argument("--groups", type=int, default=1)
    cells.add_argument("--task", type=int, default=0)
    cells.add_argument("--force", action="store_true")
    cells.set_defaults(func=run_cells)

    floors = sub.add_parser("floors")
    floors.add_argument("--out_dir", type=Path, required=True)
    floors.add_argument("--floor_paths", type=int, default=10)
    floors.set_defaults(func=run_floors)

    merge = sub.add_parser("merge")
    merge.add_argument("--cells_dir", type=Path, required=True)
    merge.add_argument("--out_tables", type=Path, required=True)
    merge.set_defaults(func=run_merge)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
