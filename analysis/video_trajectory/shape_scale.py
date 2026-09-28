#!/usr/bin/env python3
"""Trajectory shape of one video backbone's clean references — plan section 3.8
(docs/video_full_trajectory_plan_zh.md, P4): where the path bends, which plane
it bends in, and where each step's displacement lands.

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
    python analysis/video_trajectory/shape_scale.py --backbone hunyuan_video \\
        --data_root outputs

Three blocks, two data tiers:

  3.8 section 2  curvature profile      T1  `turn_angle_w{5,7}_deg` + `turn_angle_deg`
  3.8 section 3  bend plane, pairwise   T2  4,629 stored `[chord, PC1, PC2]` frames
  3.8 section 4  update subspace        T1  `update_chord_share` / `_in_position_plane`
                                            / `update_own_evr`

Window choice is decided by P1, not here, and it is READ from the P1 dumps
(`docs/figures/video_full_trajectory/<T>/p1_floor_{float32,bfloat16}.json`) via
`step_profiles.load_p1_floor`, never transcribed — so a P2 re-issue of those
dumps moves every verdict below. w=5 is the primary profile and w=7 the
robustness row while the float32 minimum readable multi-step window stays below
the plan's contingency threshold of 9; the per-junction w=1 profile, which the
image-side data could not resolve, is produced whenever the dump says the
single-step turn clears the floor. Whether the plan's contingency ("if a
backbone needs w >= 9, read the mid-path flattest position from T3") triggers is
computed from that same number and written into the JSON — it is the input to
the P8 block-C gate.

Memory (plan section 7). The frame stack is held as float16 —
4,629 x 3 x d x 2 B = 36 GB (HYV) / 47 GB (Wan) — and the pairwise work goes
through two blocked Gram matrices built in ONE streaming pass over the frames
(plane rows 9,258 x 9,258, chord rows 4,629 x 4,629), never by reading frames
pair by pair. A float32 copy of the stack would be 72 / 94 GB and is never
made; the float16 store's loss of exact orthonormality is undone on the Gram
instead, by whitening each trajectory's own 2x2 diagonal block (an exact
re-orthonormalisation of its two plane rows, which is what
`trajectory_shape_scale.orthonormalize` does on materialised rows).
`--frames_per_stream 240` gives the plan's cheap first version (1,440 planes);
`--skip_planes` runs the two T1 blocks alone.

Floor row: the T2 frames were computed in flight on the float32 path, so the
plane readings are float32-row readings and the bf16 window / off-plane floors
do NOT apply to them. float16 is the frame's *storage* dtype — a direction
quantisation, removed by the Gram whitening; a float32 Gram resolves ~0.03
degrees near 0, far below any angle distinguished here
(`trajectory_shape_scale.pair_angles_deg` docstring), and this file accumulates
the Gram in float64.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import argparse  # noqa: E402
import collections  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import window_centers  # noqa: E402
from analysis.trajectory_shape_scale import (  # noqa: E402
    angle_stats,
    plane_population,
    random_plane_null,
    sample_pairs,
)
from analysis.video_trajectory.common import (  # noqa: E402
    atomic_savefig,
    atomic_write_json,
    atomic_write_text,
    read_json_if_readable,
    resolve_reuse,
    turn_index,
)
from analysis.video_trajectory.step_profiles import (  # noqa: E402
    BACKBONES,
    DEFAULT_DATA_ROOT,
    N_STATES,
    References,
    _md_table,
    _tsv,
    load_index,
    load_p1_floor,
    load_references,
)

WINDOWS = (5, 7)          # both stored per generation; 5 = primary (P1), 7 = robustness
SINGLE_KEY = "turn_angle_deg"   # w = 1, indexed by junction n = 0..48
ENDPOINT_CENTERS = (5, 45)      # plan section 3.8-2.1, only exist in the w=5 profile
NULL_PAIRS = 500
GRAM_CHUNK = 4096               # latent columns converted to float32 per pass
T3_CONTINGENCY_WINDOW = 9       # plan section 3.8-2: w >= 9 sends the mid-path
#                                 flattest position to T3


def profile_key(w: int) -> str:
    return f"turn_angle_w{w}_deg"


# ---------------------------------------------------------------------------
# 3.8 section 2 — curvature profile (T1)
# ---------------------------------------------------------------------------


def window_profile_summary(prof: np.ndarray, centers: np.ndarray) -> dict[str, Any]:
    """`trajectory_shape_scale.profile_summary` generalised to any window.

    That function hard-codes the w=5 centre map (`window_centers(len+10, 5)`),
    which is right for w=5 and silently wrong for w=7 and w=1 — the array index
    is not the centre (plan section 2.4). Same statistics, explicit centres.
    """
    med = np.median(prof, axis=0)
    scaled = prof / prof.mean(axis=1, keepdims=True)
    dev = np.abs(scaled - med / med.mean()).max(axis=1)
    troughs = centers[prof.argmin(axis=1)]
    peaks = centers[prof.argmax(axis=1)]
    # CV ACROSS RECORDS at each window centre (axis 0 = records), one value per
    # centre; `cv_med` below is the median of that profile over centres. It is
    # NOT a per-trajectory CV (the spread of one trajectory's own curvature
    # along the path), which is a different and much larger number.
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
        "cv_definition": "CV across records at one window centre; `_med` / `_max` are "
                         "the median / max of that per-centre profile. Not a "
                         "per-trajectory CV.",
        "cv_across_records_med": float(np.median(cv)),
        "cv_across_records_max": float(cv.max()),
    }


def _pair_agreement(medians: dict[str, np.ndarray], pairs: list[tuple[str, str]],
                    centers: np.ndarray) -> dict[str, Any]:
    """Shape gap and level ratio over a chosen set of stream pairs.

    Shape gap = max |a/mean(a) - b/mean(b)| (level removed), level ratio =
    mean(a)/mean(b) made direction-free — the same two quantities
    `trajectory_shape_scale.profile_agreement` computes, but over a chosen pair
    list rather than every pair of the dict, because the plan asks for three
    separate comparison rows (same dataset / cross dataset / all).
    """
    scaled = {k: v / v.mean() for k, v in medians.items()}
    trough = {k: int(centers[int(v.argmin())]) for k, v in medians.items()}
    dev, ratio, shift = [], [], []
    for a, b in pairs:
        dev.append(float(np.abs(scaled[a] - scaled[b]).max()))
        r = medians[a].mean() / medians[b].mean()
        ratio.append(float(max(r, 1.0 / r)))
        shift.append(abs(trough[a] - trough[b]))
    return {
        "pairs": len(pairs),
        "shape_gap_max": float(np.max(dev)), "shape_gap_med": float(np.median(dev)),
        "level_ratio_max": float(np.max(ratio)), "level_ratio_med": float(np.median(ratio)),
        "trough_shift_max": int(np.max(shift)), "trough_shift_med": float(np.median(shift)),
    }


def curvature_block(refs: References, backbone: str, floor: dict[str, Any]) -> dict[str, Any]:
    streams = refs.streams()
    out: dict[str, Any] = {"streams": [f"{d}_s{s}" for d, s in streams], "windows": {}}

    for w in WINDOWS:
        key = profile_key(w)
        centers = np.asarray(window_centers(N_STATES, w))
        per_stream, medians = {}, {}
        for dataset, seed in streams:
            name = f"{dataset}_s{seed}"
            prof = refs.cols[key][refs.mask(dataset, seed)]
            if prof.shape[1] != len(centers):
                raise ValueError(f"{key}: {prof.shape[1]} entries, {len(centers)} centres")
            summary = window_profile_summary(prof, centers)
            per_stream[name] = summary
            medians[name] = np.asarray(summary["median_profile"])
        entry: dict[str, Any] = {
            "window": w, "centers": [int(c) for c in centers], "by_stream": per_stream,
            "role": "primary (P1 float32 minimum readable window)" if w == 5 else "robustness",
        }
        if w == 5:
            # section 2.1: the two ends of the window-5 profile, per stream
            entry["endpoints"] = {}
            for c in ENDPOINT_CENTERS:
                idx = int(np.flatnonzero(centers == c)[0])
                vals = {k: float(v[idx]) for k, v in medians.items()}
                entry["endpoints"][f"center_{c}"] = {
                    "by_stream": vals, "min": min(vals.values()), "max": max(vals.values()),
                    "spread_min_max": [min(vals.values()), max(vals.values())]}
            # section 2.3: three comparison rows
            names = list(medians)
            same_ds = [(a, b) for a, b in itertools.combinations(names, 2)
                       if a.split("_s")[0] == b.split("_s")[0]]
            cross_ds = [(a, b) for a, b in itertools.combinations(names, 2)
                        if a.split("_s")[0] != b.split("_s")[0]]
            all_pairs = list(itertools.combinations(names, 2))
            entry["stability"] = {
                "same_dataset_other_seed": _pair_agreement(medians, same_ds, centers),
                "cross_dataset_any_pairing": _pair_agreement(medians, cross_ds, centers),
                "all_streams": _pair_agreement(medians, all_pairs, centers),
            }
        out["windows"][f"w{w}"] = entry

    # w = 1: the per-junction profile the image side could not resolve. Labelled
    # by the plan section 2.4 junction index (0..48) — the same convention
    # `latent_paths.py` stores, via the shared `common.turn_index`.
    junctions, w1_note = turn_index(N_STATES, 1)
    single = {}
    for dataset, seed in streams:
        prof = refs.cols[SINGLE_KEY][refs.mask(dataset, seed)]
        if prof.shape[1] != len(junctions):
            raise ValueError(f"{SINGLE_KEY}: {prof.shape[1]} entries, {len(junctions)} junctions")
        single[f"{dataset}_s{seed}"] = window_profile_summary(prof, np.asarray(junctions))
    out["windows"]["w1"] = {
        "window": 1, "index_object": w1_note,
        "centers": junctions, "by_stream": single,
        "role": "new reading: readable only because the T1 rows are float32",
        "readable_float32": floor["turn_w1_readable_float32"],
        "min_snr_float32": floor["turn_w1_min_snr_float32"],
    }

    min_f32 = floor["min_readable_multistep_turn_window"]["float32"]
    min_bf16 = floor["min_readable_multistep_turn_window"]["bfloat16"]
    triggered = bool(min_f32 is None or min_f32 >= T3_CONTINGENCY_WINDOW)
    out["window_verdict"] = {
        # "multistep": on float32 rows w = 1 is readable at every junction too,
        # so this is the minimum readable MULTI-step window, which is what P1
        # measured and what the P8 block-C gate reads
        "min_readable_multistep_window_float32": min_f32,
        "min_readable_multistep_window_bfloat16": min_bf16,
        "turn_w1_readable_float32": floor["turn_w1_readable_float32"],
        "primary_window": 5,
        "t3_contingency_window": T3_CONTINGENCY_WINDOW,
        "t3_contingency_triggered": triggered,
        "note": ("P1 gives w = %s as the minimum readable multi-step window on the float32 "
                 "in-flight rows these T1 profiles were computed on, and w = 1 %s at all 49 "
                 "junctions, so the plan's 'if w >= %d is needed, read the mid-path flattest "
                 "position from T3' contingency %s here. It would apply only to a bf16/T3 "
                 "recomputation (bf16 needs w = %s). This is the P1 verdict that feeds the "
                 "P8 block-C gate."
                 % (min_f32 if min_f32 else "> 11",
                    "readable" if floor["turn_w1_readable_float32"] else "NOT readable",
                    T3_CONTINGENCY_WINDOW,
                    "triggers" if triggered else "does not trigger",
                    min_bf16 if min_bf16 else "> 11")),
        "floor_source": floor["source"],
        "floor_row": "float32 (T1 in-flight rows)",
    }
    return out


# ---------------------------------------------------------------------------
# 3.8 section 4 — update subspace (T1)
# ---------------------------------------------------------------------------


def update_block(refs: References) -> dict[str, Any]:
    c = refs.cols
    rows = {}
    for dataset, seed in refs.streams():
        m = refs.mask(dataset, seed)
        chord_share = c["update_chord_share"][m]
        in_plane = c["update_in_position_plane"][m]
        own2 = c["update_own_evr"][m][:, 0] + c["update_own_evr"][m][:, 1]
        rows[f"{dataset}_s{seed}"] = {
            "n": int(m.sum()),
            "chord_share_med": float(np.nanmedian(chord_share)),
            "in_position_plane_med": float(np.nanmedian(in_plane)),
            "own_evr2_med": float(np.nanmedian(own2)),
            # `update_in_position_plane` is nan for a degenerate straight path
            # (analysis/trajectory_math.update_subspace); counted, never dropped silently
            "nan_chord_share": int(np.isnan(chord_share).sum()),
            "nan_in_position_plane": int(np.isnan(in_plane).sum()),
            "nan_own_evr2": int(np.isnan(own2).sum()),
        }
    ranges = {}
    for key in ("chord_share_med", "in_position_plane_med", "own_evr2_med"):
        vals = [r[key] for r in rows.values()]
        ranges[key] = {"min": float(min(vals)), "max": float(max(vals))}
    return {"by_stream": rows, "cross_stream_range": ranges,
            "floor_row": "float32 (T1 in-flight rows)",
            "note": "whole-path scalars, not per-step profiles; never indexed by n"}


# ---------------------------------------------------------------------------
# 3.8 section 3 — bend plane (T2), one streaming pass into two Gram matrices
# ---------------------------------------------------------------------------


def select_frames(refs: References, frames_per_stream: int | None) -> np.ndarray:
    if not frames_per_stream:
        return np.arange(len(refs))
    keep = []
    for dataset, seed in refs.streams():
        idx = np.flatnonzero(refs.mask(dataset, seed))
        keep.append(idx[:frames_per_stream])
    return np.sort(np.concatenate(keep))


def load_frame_stack(refs: References, matrix_root: Path, sel: np.ndarray, dim: int,
                     workers: int) -> tuple[np.ndarray, np.ndarray]:
    """`(plane_rows [2n, d], chord_rows [n, d])`, both float16.

    One file per trajectory, read once. The two PC rows are written into the
    `pair_angles_deg` layout straight away (rows 2i and 2i+1 belong to
    trajectory i), so the plane stack never has to be reshaped or copied.
    """
    n = len(sel)
    plane = np.empty((2 * n, dim), dtype=np.float16)
    chord = np.empty((n, dim), dtype=np.float16)
    files = [matrix_root / str(refs.cols["source_dir"][i]) / str(refs.cols["frame_file"][i])
             for i in sel]

    def read(k: int) -> None:
        arr = np.load(files[k], mmap_mode=None)
        if arr.shape != (3, dim):
            raise ValueError(f"{files[k]}: shape {arr.shape}, want (3, {dim})")
        chord[k] = arr[0]
        plane[2 * k] = arr[1]
        plane[2 * k + 1] = arr[2]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(read, range(n)):
            pass
    return plane, chord


def gram_float64(rows: np.ndarray, chunk: int = GRAM_CHUNK) -> np.ndarray:
    """`rows @ rows.T` for a float16 stack, accumulated in float64.

    Chunked along the latent dimension rather than over row tiles: each latent
    column block is converted to float32 once (peak temporary = rows x chunk x
    4 B) and every entry of the Gram is updated by one large BLAS call, so the
    36 GB stack is read once instead of once per row tile.
    """
    m, d = rows.shape
    gram = np.zeros((m, m), dtype=np.float64)
    for k0 in range(0, d, chunk):
        block = np.asarray(rows[:, k0:k0 + chunk], dtype=np.float32)
        gram += (block @ block.T).astype(np.float64)
    return 0.5 * (gram + gram.T)


def whiten_plane_gram(gram: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Re-orthonormalise every trajectory's own two plane rows, on the Gram.

    Row pair i has 2x2 Gram block B_i; with B_i = L_i L_i^T (Cholesky), the
    rows M_i = L_i^{-1} R_i are orthonormal and span the same plane, which is
    all a principal angle depends on. Applying the block-diagonal M to the Gram
    on both sides therefore gives exactly the Gram of the re-orthonormalised
    stack — without ever materialising a float32 copy of the 36 GB of rows.
    """
    n = gram.shape[0] // 2
    i0, i1 = 2 * np.arange(n), 2 * np.arange(n) + 1
    blocks = np.empty((n, 2, 2), dtype=np.float64)
    blocks[:, 0, 0] = gram[i0, i0]
    blocks[:, 0, 1] = gram[i0, i1]
    blocks[:, 1, 0] = gram[i1, i0]
    blocks[:, 1, 1] = gram[i1, i1]
    drift = {
        "max_abs_norm_minus_1": float(np.abs(np.sqrt(np.diagonal(blocks, axis1=1,
                                                                 axis2=2)) - 1.0).max()),
        "max_abs_cross_inner_product": float(np.abs(blocks[:, 0, 1]).max()),
    }
    inv_l = np.linalg.inv(np.linalg.cholesky(blocks))

    def apply_left(mat: np.ndarray) -> np.ndarray:
        return (inv_l @ mat.reshape(n, 2, -1)).reshape(2 * n, -1)

    whitened = apply_left(np.ascontiguousarray(apply_left(gram).T))
    return 0.5 * (whitened + whitened.T), drift


def normalise_chord_gram(gram: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(gram))
    return gram / np.outer(scale, scale)


def chord_angle_stats(gram_unit: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Acute angle between two chord directions, read off the normalised Gram.

    Acute (|cos|) for the same reason `trajectory_shape_scale.chord_stats` uses
    it: the stored frame's sign convention comes out of a QR and flips freely.
    """
    if not pairs:
        return {"pairs": 0}
    idx = np.asarray(pairs, dtype=np.int64)
    cos = np.abs(gram_unit[idx[:, 0], idx[:, 1]])
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return {"pairs": len(pairs), "chord_angle_med": float(np.median(ang)),
            "chord_angle_q05": float(np.quantile(ang, 0.05)),
            "chord_angle_min": float(np.min(ang))}


def pair_classes(refs: References, sel: np.ndarray, *, sample: int, seed: int
                 ) -> dict[str, Any]:
    """The four pair classes of plan section 3.8-3, as index pairs into `sel`.

    Membership is decided by `z_T_sha256` equality and by (dataset,
    prompt_idx), never by seed arithmetic (plan section 2.5).
    """
    z_T = [str(refs.cols["z_T_sha256"][i]) for i in sel]
    dataset = [str(refs.cols["dataset"][i]) for i in sel]
    prompt = [int(refs.cols["prompt_idx"][i]) for i in sel]
    base = [int(refs.cols["base_seed"][i]) for i in sel]
    n = len(sel)

    by_noise: dict[str, list[int]] = collections.defaultdict(list)
    by_prompt: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    by_stream: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for i in range(n):
        by_noise[z_T[i]].append(i)
        by_prompt[(dataset[i], prompt[i])].append(i)
        by_stream[(dataset[i], base[i])].append(i)

    same_noise = [(a, b) for g in by_noise.values() for a, b in itertools.combinations(g, 2)]
    same_prompt = [(a, b) for g in by_prompt.values() for a, b in itertools.combinations(g, 2)]
    overlap = set(same_noise) & set(same_prompt)  # would be one generation twice
    same_noise = [p for p in same_noise if p not in overlap]
    same_prompt = [p for p in same_prompt if p not in overlap]

    related = set(same_noise) | set(same_prompt) | overlap
    unrelated = sample_pairs(n, sample, seed=seed, exclude=related)

    within: dict[str, list[tuple[int, int]]] = {}
    rng = np.random.default_rng(seed + 1)
    for (ds, bs), members in sorted(by_stream.items()):
        # inside one stream no two rows share a prompt or a noise, so every pair
        # is a "both different" pair; sample rather than enumerate
        members = np.asarray(members)
        total = len(members) * (len(members) - 1) // 2
        take = min(sample, total)
        picked: set[tuple[int, int]] = set()
        guard = 0
        while len(picked) < take and guard < 50 * max(take, 1):
            guard += 1
            a, b = int(rng.integers(len(members))), int(rng.integers(len(members)))
            if a == b:
                continue
            pair = (int(members[min(a, b)]), int(members[max(a, b)]))
            picked.add(pair)
        within[f"{ds}_s{bs}"] = sorted(picked)

    noise_sizes = collections.Counter(len(g) for g in by_noise.values())
    return {
        "same_noise_diff_prompt": same_noise,
        "same_prompt_diff_noise": same_prompt,
        "both_different": unrelated,
        "within_stream_both_different": within,
        "counts": {
            "n_frames": n,
            "noise_groups_total": len(by_noise),
            "noise_groups_used": int(sum(1 for g in by_noise.values() if len(g) >= 2)),
            "noise_group_size_histogram": {str(k): int(v) for k, v in sorted(noise_sizes.items())},
            "prompt_groups": len(by_prompt),
            "same_noise_pairs": len(same_noise),
            "same_prompt_pairs": len(same_prompt),
            "both_different_pairs": len(unrelated),
            "same_generation_pairs_excluded": len(overlap),
        },
    }


def plane_block(refs: References, matrix_root: Path, *, frames_per_stream: int | None,
                sample: int, workers: int, null_pairs: int, reference_planes: int,
                gram_chunk: int, seed: int = 1) -> dict[str, Any]:
    sel = select_frames(refs, frames_per_stream)
    dim = int(refs.cols["d"][0])
    n = len(sel)
    gb = 3 * n * dim * 2 / 1e9
    print(f"  loading {n:,} T2 frames as float16 ({gb:.1f} GB) from {matrix_root}", flush=True)
    plane_rows, chord_rows = load_frame_stack(refs, matrix_root, sel, dim, workers)

    print("  building the plane Gram (2n x 2n) in one pass ...", flush=True)
    gram_plane_raw = gram_float64(plane_rows, gram_chunk)
    print("  building the chord Gram (n x n) in one pass ...", flush=True)
    gram_chord = normalise_chord_gram(gram_float64(chord_rows, gram_chunk))
    del plane_rows, chord_rows  # the 36 GB stack is not needed past this point

    gram_plane, drift = whiten_plane_gram(gram_plane_raw)
    del gram_plane_raw

    classes = pair_classes(refs, sel, sample=sample, seed=seed)
    entry: dict[str, Any] = {
        "n_frames": n, "dim": dim, "frames_per_stream": frames_per_stream,
        "frame_stack_GB_float16": round(gb, 2),
        # mmap: the dtype is in the header, the 7.8-10.2 MB body is not read
        "stored_frame_dtype": str(np.load(
            matrix_root / str(refs.cols["source_dir"][sel[0]])
            / str(refs.cols["frame_file"][sel[0]]), mmap_mode="r").dtype),
        "float16_store_drift": drift,
        "counts": classes["counts"],
        "floor_row": "float32 (T2 frames were computed on the in-flight float32 path); "
                     "float16 is the storage dtype only and is undone by the Gram whitening",
        "statistical_caliber": "same-noise pairs are read per NOISE "
                               f"({classes['counts']['noise_groups_used']} groups), not per pair",
        "classes": {},
    }
    for name in ("same_noise_diff_prompt", "same_prompt_diff_noise", "both_different"):
        pairs = classes[name]
        stat = angle_stats(gram_plane, pairs)
        stat["chord"] = chord_angle_stats(gram_chord, pairs)
        entry["classes"][name] = stat

    entry["classes"]["within_stream_both_different"] = {}
    for stream, pairs in classes["within_stream_both_different"].items():
        stat = angle_stats(gram_plane, pairs)
        stat["chord"] = chord_angle_stats(gram_chord, pairs)
        entry["classes"]["within_stream_both_different"][stream] = stat

    print(f"  random-plane null ({null_pairs} pairs, d = {dim:,}) ...", flush=True)
    entry["random_plane_null"] = random_plane_null(dim, n_pairs=null_pairs, seed=seed)
    # plan section 3.8-3 asks for the chord angle in all four classes AND the
    # reference, so the null needs its own chord row rather than a blank cell
    entry["random_chord_null"] = random_chord_null(dim, n_pairs=null_pairs, seed=seed)

    print("  concentration of the plane directions ...", flush=True)
    entry["population"] = plane_population(gram_plane)
    n_ref = n if reference_planes < 0 else reference_planes
    if n_ref > 0:
        print(f"  drawn reference: {n_ref:,} random 2-planes in d = {dim:,} ...", flush=True)
        entry["population_reference"] = random_plane_population(n_ref, dim, seed=seed,
                                                               chunk=gram_chunk)
        entry["population_reference"]["planes_drawn"] = n_ref
        entry["population_reference"]["same_count_as_data"] = bool(n_ref == n)
    return entry


def random_chord_null(dim: int, *, n_pairs: int = NULL_PAIRS, seed: int = 0,
                      chunk: int = GRAM_CHUNK) -> dict[str, Any]:
    """Acute angle between two independent random directions in `dim` dims —
    the chord-side counterpart of `random_plane_null`.

    Drawn the same way the plane null draws its bases (Gaussian, normalised),
    in latent-column chunks so nothing of size `n_pairs x dim` is materialised
    at once, and reduced with the same |cos| convention as
    `chord_angle_stats`, whose sign is free.
    """
    rng = np.random.default_rng(seed)
    dots = np.zeros(n_pairs, dtype=np.float64)
    na = np.zeros(n_pairs, dtype=np.float64)
    nb = np.zeros(n_pairs, dtype=np.float64)
    for k0 in range(0, dim, chunk):
        width = min(chunk, dim - k0)
        a = rng.standard_normal((n_pairs, width))
        b = rng.standard_normal((n_pairs, width))
        dots += (a * b).sum(axis=1)
        na += (a * a).sum(axis=1)
        nb += (b * b).sum(axis=1)
    cos = np.abs(dots / np.sqrt(na * nb))
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return {"pairs": int(n_pairs), "chord_angle_med": float(np.median(ang)),
            "chord_angle_q05": float(np.quantile(ang, 0.05)),
            "chord_angle_min": float(np.min(ang))}


def random_plane_population(n_planes: int, dim: int, *, seed: int = 0,
                            chunk: int = GRAM_CHUNK) -> dict[str, Any]:
    """`plane_population` on `n_planes` DRAWN random 2-planes of the same
    dimension — the "no alignment at all" reference.

    Same object as `trajectory_shape_scale.plane_population_reference`, which
    materialises a `[2n, dim]` float32 array; at this grid that is 48 GB (HYV)
    / 63 GB (Wan) on top of the frame stack, so the Gram is accumulated over
    latent-column chunks instead and each plane's own 2x2 block is whitened,
    which is exactly the QR that function does per plane. Drawn rather than
    assumed because the random Gram is Marchenko-Pastur spread, not flat.
    """
    rng = np.random.default_rng(seed)
    gram = np.zeros((2 * n_planes, 2 * n_planes), dtype=np.float64)
    for k0 in range(0, dim, chunk):
        block = rng.standard_normal((2 * n_planes, min(chunk, dim - k0)), dtype=np.float32)
        gram += (block @ block.T).astype(np.float64)
    whitened, _ = whiten_plane_gram(0.5 * (gram + gram.T))
    return plane_population(whitened)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def write_tables(report: dict[str, Any], backbone: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    md = [f"# Trajectory shape — {backbone} (plan section 3.8)", "",
          f"{report['source']['references_kept']:,} clean references; curvature and update "
          f"subspace from T1 (float32-row readings), bend planes from the "
          f"{report.get('planes', {}).get('n_frames', 0):,} stored T2 frames. "
          f"P1 floors read from `{report['floor']['source']['float32']}` / "
          f"`{report['floor']['source']['bfloat16']}`.", ""]

    curv = report["curvature"]
    w5 = curv["windows"]["w5"]
    md += ["## 3.8-2 curvature profile", "",
           f"Window verdict: {curv['window_verdict']['note']}", ""]

    header = ["stream", "n", "peak_center", "min_center", "peak_over_min",
              "center_5_deg", "center_45_deg", "trough_center_med", "trough_iqr",
              "trough_q05", "trough_q95", "cv_across_records_med"]
    rows = []
    for stream, s in w5["by_stream"].items():
        rows.append([stream, s["n"], s["peak_center"], s["min_center"], s["peak_over_min"],
                     w5["endpoints"]["center_5"]["by_stream"][stream],
                     w5["endpoints"]["center_45"]["by_stream"][stream],
                     s["trough_center_med"], s["trough_center_iqr"],
                     s["trough_center_q05_q95"][0], s["trough_center_q05_q95"][1],
                     s["cv_across_records_med"]])
    p = out_dir / f"curvature_w5_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["Window 5 (primary), per stream — section 2.1 endpoint readings and section 2.2 "
           "flattest position (per-trajectory trough centre):", "",
           "`cv_across_records_med`: at each window centre, the CV ACROSS the stream's "
           "records; the column is the median of that per-centre profile. It measures "
           "how much prompts disagree about the curvature at a given position, and is "
           "NOT the CV of one trajectory's curvature along its own path — that "
           "per-trajectory quantity is a different and substantially larger number, and "
           "is not computed here.", ""]
    md += _md_table(header, rows) + [""]
    for c in ENDPOINT_CENTERS:
        e = w5["endpoints"][f"center_{c}"]
        md += [f"- centre {c}: stream range {e['min']:.4g}–{e['max']:.4g} degrees"]
    md += [""]

    header = ["window", "stream", "n", "peak_center", "min_center", "peak_over_min",
              "trough_center_med", "trough_iqr", "trough_q05", "trough_q95",
              "cv_across_records_med"]
    rows = []
    for wkey, entry in curv["windows"].items():
        for stream, s in entry["by_stream"].items():
            rows.append([wkey, stream, s["n"], s["peak_center"], s["min_center"],
                         s["peak_over_min"], s["trough_center_med"], s["trough_center_iqr"],
                         s["trough_center_q05_q95"][0], s["trough_center_q05_q95"][1],
                         s["cv_across_records_med"]])
    p = out_dir / f"curvature_flattest_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["All three windows (w=7 robustness row, w=1 the new per-junction profile; for "
           "w=1 the index object is the junction n = 0..48, not a window centre). "
           "`cv_across_records_med` is the cross-record CV per centre, median over "
           "centres — see the note above the previous table:", ""]
    md += _md_table(header, rows) + [""]

    header = ["comparison", "pairs", "shape_gap_max", "shape_gap_med", "level_ratio_max",
              "level_ratio_med", "trough_shift_max"]
    rows = [[name, s["pairs"], s["shape_gap_max"], s["shape_gap_med"], s["level_ratio_max"],
             s["level_ratio_med"], s["trough_shift_max"]]
            for name, s in w5["stability"].items()]
    p = out_dir / f"curvature_stability_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["Section 2.3 stability (shape gap = max |a/mean(a) - b/mean(b)|, level ratio = "
           "mean ratio):", ""]
    md += _md_table(header, rows) + [""]

    if "planes" in report:
        pl = report["planes"]
        header = ["class", "pairs", "theta1_med", "theta1_q05", "theta1_min", "theta2_med",
                  "chord_angle_med", "chord_angle_min"]
        rows = []
        for name in ("same_noise_diff_prompt", "same_prompt_diff_noise", "both_different"):
            s = pl["classes"][name]
            rows.append([name, s["pairs"], s.get("theta1_med"), s.get("theta1_q05"),
                         s.get("theta1_min"), s.get("theta2_med"),
                         s["chord"].get("chord_angle_med"), s["chord"].get("chord_angle_min")])
        for stream, s in pl["classes"]["within_stream_both_different"].items():
            rows.append([f"within_stream:{stream}", s["pairs"], s.get("theta1_med"),
                         s.get("theta1_q05"), s.get("theta1_min"), s.get("theta2_med"),
                         s["chord"].get("chord_angle_med"), s["chord"].get("chord_angle_min")])
        null = pl["random_plane_null"]
        cnull = pl.get("random_chord_null", {})
        rows.append(["random_plane_null", null["n_pairs"], null["theta1_med"],
                     null["theta1_q05"], None, null["theta2_med"],
                     cnull.get("chord_angle_med"), cnull.get("chord_angle_min")])
        p = out_dir / f"bend_plane_angles_{backbone}.tsv"
        _tsv(p, header, rows); written.append(p)
        md += ["## 3.8-3 bend plane (first principal angle, degrees)", "",
               f"**{pl['statistical_caliber']}.** Floor row: {pl['floor_row']}. The last "
               "row is the drawn reference: independent random 2-planes for the plane "
               "columns and independent random directions for the chord columns, both in "
               f"d = {pl['dim']:,}.", ""]
        md += _md_table(header, rows) + [""]

        header = ["population", "rows", "top2_share", "top10_share", "dims_for_50pct",
                  "dims_for_90pct"]
        rows = [["stored frames", pl["population"]["rows"], pl["population"]["top2_share"],
                 pl["population"]["top10_share"], pl["population"]["dims_for_50pct"],
                 pl["population"]["dims_for_90pct"]]]
        if "population_reference" in pl:
            r = pl["population_reference"]
            rows.append([f"drawn random planes (n={r['planes_drawn']}, same dimension)",
                         r["rows"], r["top2_share"], r["top10_share"], r["dims_for_50pct"],
                         r["dims_for_90pct"]])
        p = out_dir / f"plane_concentration_{backbone}.tsv"
        _tsv(p, header, rows); written.append(p)
        md += ["Section 3.4 concentration of all plane directions, against a drawn "
               "same-count same-dimension random reference:", ""]
        md += _md_table(header, rows) + [""]
        md += [f"float16 store drift before whitening: max |‖row‖ - 1| = "
               f"{pl['float16_store_drift']['max_abs_norm_minus_1']:.2e}, max |PC1·PC2| = "
               f"{pl['float16_store_drift']['max_abs_cross_inner_product']:.2e}.", ""]

    upd = report["update"]
    header = ["stream", "n", "chord_share_med", "in_position_plane_med", "own_evr2_med",
              "nan_in_position_plane"]
    rows = [[stream, s["n"], s["chord_share_med"], s["in_position_plane_med"],
             s["own_evr2_med"], s["nan_in_position_plane"]]
            for stream, s in upd["by_stream"].items()]
    p = out_dir / f"update_subspace_{backbone}.tsv"
    _tsv(p, header, rows); written.append(p)
    md += ["## 3.8-4 update subspace (per-step displacement)", ""]
    md += _md_table(header, rows) + [""]
    md += ["Cross-stream range: " + ", ".join(
        f"{k} {v['min']:.4g}–{v['max']:.4g}" for k, v in upd["cross_stream_range"].items()), ""]

    p = out_dir / f"shape_scale_{backbone}.md"
    atomic_write_text(p, "\n".join(md) + "\n")
    written.append(p)
    print("\n".join(md))
    return written


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def write_curvature_figure(report: dict[str, Any], backbone: str, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    curv = report["curvature"]
    colors = {"penguin599": "#d62728", "vbench944": "#1f77b4"}
    styles = ["-", "--", ":"]
    panels = [("w5", "window w = 5 (primary; P1 minimum readable window)",
               "turn angle over the 5 steps before vs after (degrees)",
               "window centre c (state index)"),
              ("w7", "window w = 7 (robustness)",
               "turn angle over the 7 steps before vs after (degrees)",
               "window centre c (state index)"),
              ("w1", "single step w = 1 (readable only on float32 rows)",
               "turn angle between consecutive steps (degrees)",
               "junction n (between step n and step n+1)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, (key, title, ylabel, xlabel) in zip(axes, panels):
        entry = curv["windows"][key]
        seen: dict[str, int] = {}
        for stream, s in entry["by_stream"].items():
            dataset = stream.split("_s")[0]
            i = seen.get(dataset, 0)
            seen[dataset] = i + 1
            ax.plot(s["centers"], s["median_profile"], color=colors.get(dataset, "0.4"),
                    ls=styles[i % len(styles)], lw=1.5, label=stream)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)
    counts = [s["n"] for s in curv["windows"]["w5"]["by_stream"].values()]
    per = (f"{min(counts):,}" if min(counts) == max(counts)
           else f"{min(counts):,}-{max(counts):,}")
    fig.suptitle(f"{backbone}: how sharply the clean trajectory turns along the path\n"
                 f"{len(counts)} lines = 2 datasets x 3 base seeds; each line is that "
                 f"stream's median over its {per} references (float32-row reading)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p = atomic_savefig(fig, out_dir / f"curvature_profile_{backbone}.png", dpi=140)
    plt.close(fig)
    return p


def write_plane_figure(report: dict[str, Any], backbone: str, out_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pl = report.get("planes")
    if pl is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    def med_or_none(values: list) -> float | None:
        kept = [v for v in values if v is not None]
        return float(np.median(kept)) if kept else None

    within = [s for s in pl["classes"]["within_stream_both_different"].values() if s["pairs"]]
    labels = [("SAME noise\ndiff. prompt", "same_noise_diff_prompt", "C3"),
              ("same prompt\nDIFF. noise", "same_prompt_diff_noise", "C0"),
              ("both\ndifferent", "both_different", "0.6")]
    groups = [(label, pl["classes"][key].get("theta1_med"), pl["classes"][key]["pairs"],
               pl["classes"][key]["chord"].get("chord_angle_med"), color)
              for label, key, color in labels]
    groups.append(("both different,\nwithin one stream",
                   med_or_none([s.get("theta1_med") for s in within]),
                   int(sum(s["pairs"] for s in within)),
                   med_or_none([s["chord"].get("chord_angle_med") for s in within]), "0.75"))
    groups.append(("random planes\n(reference)", pl["random_plane_null"]["theta1_med"],
                   pl["random_plane_null"]["n_pairs"],
                   pl.get("random_chord_null", {}).get("chord_angle_med"), "0.88"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, which in zip(axes, ("plane", "chord")):
        values = [g[1] if which == "plane" else g[3] for g in groups]
        drawn = [(i, v) for i, v in enumerate(values) if v is not None]
        bars = ax.bar([i for i, _ in drawn], [v for _, v in drawn],
                      color=[groups[i][4] for i, _ in drawn], width=0.65)
        for (i, v), bar in zip(drawn, bars):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                    f"{v:.1f} deg\n{groups[i][2]:,} pairs", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_ylim(0, 118)
        ax.axhline(90.0, color="k", ls=":", lw=0.9)
        ax.set_ylabel("first principal angle between the two bending planes\n(median, degrees)"
                      if which == "plane" else
                      "acute angle between the two chord directions\n(median, degrees)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].text(-0.45, 111.0, "dotted line at 90 deg = nothing in common",
                 fontsize=7.5, ha="left", va="center", color="0.35")
    fig.suptitle(f"{backbone}: do two clean trajectories bend in the same plane?\n"
                 f"every pair drawn from {pl['n_frames']:,} stored frames; "
                 f"{pl['statistical_caliber']}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = atomic_savefig(fig, out_dir / f"bend_plane_angles_{backbone}.png", dpi=140)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
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
    ap.add_argument("--sample", type=int, default=20000,
                    help="sampled both-different pairs (also the per-stream cap)")
    ap.add_argument("--frames_per_stream", type=int, default=0,
                    help="0 = every frame; 240 gives the plan's cheap first version "
                         "(1,440 planes)")
    ap.add_argument("--null_pairs", type=int, default=NULL_PAIRS,
                    help="random-plane null pairs (default %(default)s)")
    ap.add_argument("--reference_planes", type=int, default=-1,
                    help="drawn random planes for the concentration reference; -1 = the same "
                         "count as the data (the plan's caliber), 0 = skip. Drawing it costs "
                         "about as much as the data Gram itself")
    ap.add_argument("--gram_chunk", type=int, default=GRAM_CHUNK,
                    help="latent columns converted to float32 per Gram pass")
    ap.add_argument("--workers", type=int, default=8, help="frame-reading threads")
    ap.add_argument("--skip_planes", action="store_true",
                    help="T1 blocks only; do not touch the 36-47 GB of T2 frames")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    T = args.backbone
    matrix = args.data_root / T / "matrix"
    merged = args.merged or (matrix / "trajectory" / "t1_merged.jsonl")
    index_path = args.index or (matrix / "trajectory" / "t1_index.json")
    out_tables = args.out_tables or (_PROJECT_ROOT / "resources" / "video_full_trajectory" / T)
    out_figs = args.out_figs or (_PROJECT_ROOT / "docs" / "figures" / "video_full_trajectory" / T)
    json_path = out_tables / f"video_shape_scale_{T}.json"

    # the parameters that decide what the numbers are; an existing output only
    # stands for this run when they agree (a `--limit 3 --frames_per_stream 8`
    # smoke run must not satisfy the production run)
    run_params = {"limit": args.limit, "frames_per_stream": args.frames_per_stream,
                  "sample": args.sample, "null_pairs": args.null_pairs,
                  "reference_planes": args.reference_planes,
                  "skip_planes": bool(args.skip_planes), "merged": str(merged)}
    expected = [out_tables / f"shape_scale_{T}.md"]
    if not args.no_figures:
        expected.append(out_figs / f"curvature_profile_{T}.png")
        if not args.skip_planes:
            expected.append(out_figs / f"bend_plane_angles_{T}.png")
    stored = resolve_reuse(json_path, run_params, force=args.force,
                           caps=("limit", "frames_per_stream", "sample", "null_pairs",
                                 "reference_planes"),
                           extra_outputs=expected)
    if stored is not None:
        print(f"[skip] outputs already exist under {out_tables} and {out_figs} for "
              f"these parameters; pass --force to recompute")
        return

    # --skip_planes --force would otherwise drop the `planes` block from the
    # JSON while leaving the bend-plane figure on disk, i.e. a table and a
    # figure that no longer describe the same run
    carried = None
    if args.skip_planes:
        previous = read_json_if_readable(json_path)
        if previous is not None and "planes" in previous:
            prev = previous.get("run_params", {})
            same_selection = (prev.get("limit") == args.limit
                              and prev.get("frames_per_stream") == args.frames_per_stream
                              and prev.get("sample") == args.sample)
            if not same_selection:
                raise SystemExit(
                    f"{json_path} holds a `planes` block from a different reference "
                    f"selection ({prev.get('limit')=}, {prev.get('frames_per_stream')=}) "
                    f"and --skip_planes would leave it, and "
                    f"{out_figs / f'bend_plane_angles_{T}.png'}, inconsistent with the T1 "
                    f"blocks about to be written. Drop --skip_planes, or delete both first.")
            carried = previous["planes"]
            print("  --skip_planes: carrying the existing planes block forward "
                  "(same reference selection)")

    print(f"=== shape_scale {T} ===")
    floor = load_p1_floor(T, args.p1_floor_dir)
    index = load_index(index_path)
    refs = load_references(merged, index, limit=args.limit, keep_frames=True)

    report: dict[str, Any] = {
        "backbone": T,
        "produced_by": "analysis/video_trajectory/shape_scale.py",
        "plan_sections": ["3.8-2 curvature", "3.8-3 bend plane", "3.8-4 update subspace"],
        "source": {**refs.meta, "index": str(index_path), "matrix_root": str(matrix)},
        "streams": refs.meta["per_stream"],
        "floor": floor,
        "run_params": run_params,
        "curvature": curvature_block(refs, T, floor),
        "update": update_block(refs),
    }
    if not args.skip_planes:
        report["planes"] = plane_block(
            refs, matrix, frames_per_stream=args.frames_per_stream or None,
            sample=args.sample, workers=args.workers, null_pairs=args.null_pairs,
            reference_planes=args.reference_planes, gram_chunk=args.gram_chunk)
    elif carried is not None:
        report["planes"] = {**carried, "carried_forward_from_previous_run": True}

    out_tables.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, report)
    write_tables(report, T, out_tables)
    print(f"\nwrote {json_path}")
    if not args.no_figures:
        print(f"wrote {write_curvature_figure(report, T, out_figs)}")
        p = write_plane_figure(report, T, out_figs)
        if p is not None:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
