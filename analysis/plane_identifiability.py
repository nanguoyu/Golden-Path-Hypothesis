#!/usr/bin/env python3
"""What a bend-plane angle of 61 degrees means: the resolution of the
measurement, and what is shared when two trajectories share their noise.

`analysis/trajectory_shape_scale.py` compares the two-dimensional bend planes
of pairs of trajectories and reports, on the whole path, same-noise pairs at
theta1 = 61 deg against unrelated pairs at 87.4 and a random-2-plane null at
89.8. Three things it cannot settle, in the order they have to be settled:

  1. RESOLUTION. 61 deg is only a large number if two planes that ought to
     agree come back closer than that. The sampler ran in bf16, so a run whose
     arithmetic had rounded differently is an equally valid realisation of the
     same trajectory; the angle between the planes fitted to those two runs is
     the floor below which nothing is distinguishable. This is measured, on
     archived full latent paths, one floor per segment -- not bounded from the
     eigenvalue spectrum. A relative eigengap cannot answer it: the governing
     bound (Davis-Kahan) is gap over PERTURBATION, and normalising the gap by
     an eigenvalue instead leaves a quantity that is flat across a 15x range
     in real determinacy.

  2. WHAT the noise fixes. The 2-plane angle folds together a strongly
     determined first direction and a weaker second one, so section B repeats
     the pairings on the first direction alone, where no span ambiguity
     exists, and against two controls: the other trajectory's CHORD (a shared
     plane would be reproduced exactly by PC1 being the partner's chord, and
     the chord is dominated by -z_T, so same-noise pairs would align for a
     reason that has nothing to do with bending), and a null of planes sharing
     PC1 and nothing else, which is the reference theta2 has to be read
     against -- NOT the random-2-plane null.

  3. Whether one trajectory even HAS one plane. Section C fits the same
     trajectory's plane on its first 16 steps and on its last 13, which share
     no step. This is NOT a reference scale for section B: on real data it
     comes back at 84 deg, worse than two different generations that share
     their noise, because a short window resolves its own plane far less well
     (see the per-segment floors) and because a path that is not exactly
     planar turns its local plane as it goes. Read it as what it is -- how far
     the early plane is from the late one -- and read it next to the 0:16
     floor, not next to section B.

Sections B and C read the frame store; section A reads archived latents,
which exist for 30 trajectories per model and are the only place a per-segment
spectrum can come from at all (`pca_evr` in the records is computed once, on
the whole path).

  python analysis/plane_identifiability.py --root ~/full_traj_seg
  python analysis/plane_identifiability.py --root ~/full_traj_seg \
         --pair_segment 038_051 --turn_segments 000_016,038_051
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import plane_frame  # noqa: E402
from analysis.trajectory_shape_scale import (  # noqa: E402
    available_segments,
    load_cell,
    orthonormalize,
    pair_angles_deg,
    sample_pairs,
    split_pairs,
)

WHOLE_PATH = "000_051"
NULL_PAIRS = 500
MAX_UNRELATED = 20000
UNRELATED_SEED = 1
DEFAULT_LATENTS = _PROJECT_ROOT / "resources/full_trajectory"

# Half a bfloat16 ulp, relative. The sampler ran in bf16, so this is the scale
# at which a differently-rounded but equally faithful run would differ, and
# therefore the scale a "these two planes are the same plane" claim lives at.
BF16_HALF_ULP = 2.0 ** -9
FLOOR_REPEATS = 3


# ---------------------------------------------------------------------------
# store walking
# ---------------------------------------------------------------------------


def run_dirs_by_model(root: Path) -> dict[str, list[Path]]:
    """`<root>/<model>/<run_name>/` grouped by the model the records name.

    Every record is checked, not just the first: a run dir holding two models'
    records would otherwise be filed under whichever sorted first and analysed
    as one population.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for run_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        models = set()
        for path in sorted(run_dir.glob("traj_*.json")):
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            if "model" not in record:
                raise SystemExit(f"record without a 'model' field: {path}")
            models.add(record["model"])
        if not models:
            continue
        if len(models) > 1:
            raise SystemExit(f"{run_dir} holds records for {sorted(models)}; one model per run dir")
        groups[models.pop()].append(run_dir)
    return dict(groups)


def parse_tag(tag: str) -> tuple[int, int]:
    """`038_051` -> `(38, 51)`, the half-open row range the frame was built on."""
    start, _, stop = tag.partition("_")
    return int(start), int(stop)


def frame_positions(records: list[dict]) -> dict[int, int]:
    """`prompt_idx` -> that record's row in the run dir's frame stack.

    The stack holds only the framed records, in record order, so the row is a
    running count over those and not the index into `records`.
    """
    positions: dict[int, int] = {}
    row = 0
    for record in records:
        if not record.get("_has_frame"):
            continue
        key = int(record["prompt_idx"])
        if key in positions:
            raise ValueError(f"prompt_idx {key} appears twice in one run dir")
        positions[key] = row
        row += 1
    return positions


def _quant(values: np.ndarray) -> dict[str, float]:
    return {
        "med": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# A. the resolution of a plane angle, measured
# ---------------------------------------------------------------------------


def plane_angles(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Both principal angles between two `[2, d]` orthonormal frames."""
    singular = np.clip(np.linalg.svd(a @ b.T, compute_uv=False), 0.0, 1.0)
    theta = np.degrees(np.arccos(singular))
    return float(theta[0]), float(theta[1])


def floor_for_trajectory(Z: np.ndarray, segments: list[str], rng: np.random.Generator,
                         repeats: int) -> dict[str, list[tuple[float, float, float]]]:
    """Angle between the planes fitted to one trajectory and to differently
    rounded realisations of it, per segment.

    The perturbation is relative because a bfloat16 ulp is relative: an
    absolute jitter would be enormous where the latent is small and invisible
    where it is large, and would measure neither the arithmetic's uncertainty
    nor anything else.
    """
    out: dict[str, list[tuple[float, float, float]]] = {}
    for tag in segments:
        lo, hi = parse_tag(tag)
        window = Z[lo:hi]
        base = plane_frame(window)
        if base is None:
            continue
        rows = []
        for _ in range(repeats):
            jitter = 1.0 + BF16_HALF_ULP * rng.uniform(-1.0, 1.0, size=window.shape)
            moved = plane_frame(window * jitter)
            if moved is None:
                continue
            t1, t2 = plane_angles(base[1:], moved[1:])
            pc1 = np.degrees(np.arccos(np.clip(abs(float(base[1] @ moved[1])), 0.0, 1.0)))
            rows.append((t1, t2, pc1))
        if rows:
            out[tag] = rows
    return out


def floor_report(latent_root: Path, segments: list[str], *, repeats: int = FLOOR_REPEATS,
                 seed: int = 0) -> dict[str, Any]:
    """The measured floor per (model, segment), over every archived latent path.

    Reported as a distribution rather than a single number: the floor is a
    property of the trajectory, and a segment whose worst trajectory resolves
    to 12 degrees cannot support a claim about a 6-degree difference even if
    its median resolves to 1.
    """
    report: dict[str, Any] = {}
    for model_dir in sorted(p for p in latent_root.glob("latents_*") if p.is_dir()):
        model = model_dir.name.removeprefix("latents_")
        paths = sorted(model_dir.glob("latents_*.pt"))
        if not paths:
            continue
        import torch  # local: the store sections do not need it

        rng = np.random.default_rng(seed)
        collected: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for path in paths:
            Z = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
            for tag, rows in floor_for_trajectory(Z, segments, rng, repeats).items():
                collected[tag].extend(rows)
        entry: dict[str, Any] = {"trajectories": len(paths), "repeats": repeats,
                                 "relative_jitter": BF16_HALF_ULP}
        for tag, rows in collected.items():
            arr = np.array(rows, dtype=np.float64)
            entry[tag] = {"fits": len(rows), "theta1": _quant(arr[:, 0]),
                          "theta2": _quant(arr[:, 1]), "first_direction": _quant(arr[:, 2])}
        report[model] = entry
    return report


# ---------------------------------------------------------------------------
# B. one direction at a time, with the controls that make it readable
# ---------------------------------------------------------------------------


def direction_stats(cos: np.ndarray, pairs: int) -> dict[str, Any]:
    """Acute angle summary from the |cos| of a set of direction pairs.

    Acute because the sign of a principal direction is not defined: the probe
    stores whatever sign its QR produced, so `d` and `-d` are one direction.
    No minimum is reported -- it is an order statistic and these groups differ
    in size by more than an order of magnitude, so their minima are not
    comparable with each other or with the null.
    """
    if pairs == 0:
        return {"pairs": 0}
    theta = np.degrees(np.arccos(np.clip(np.abs(cos), 0.0, 1.0)))
    return {"pairs": pairs, "theta_med": float(np.median(theta)),
            "theta_q05": float(np.quantile(theta, 0.05)),
            "theta_q95": float(np.quantile(theta, 0.95)),
            "cos_med": float(np.median(np.abs(cos)))}


def pair_cos(gram: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    idx = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return gram[idx[:, 0], idx[:, 1]].astype(np.float64)


def random_direction_null(dim: int, n_pairs: int = NULL_PAIRS, seed: int = 0) -> dict[str, Any]:
    """Acute angle between independent random unit vectors in `dim` dimensions.

    Drawn rather than taken from the large-`dim` formula, for the same reason
    `random_plane_null` is drawn: the point of a null is that it was not
    assumed. This is also the right null for the SECOND principal angle of two
    planes that share their first direction exactly and nothing else, which is
    the reference a large theta2 has to be read against.
    """
    rng = np.random.default_rng(seed)
    cos = np.empty(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        a = rng.standard_normal(dim)
        b = rng.standard_normal(dim)
        cos[i] = float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))
    stats = direction_stats(cos, n_pairs)
    stats["also_the_null_for"] = "theta2 of two planes sharing PC1 exactly"
    return stats


def index_distances(records: list[dict], pairs: list[tuple[int, int]]) -> dict[str, int]:
    """How far apart the two prompts of each pair sit in their prompt files.

    Printed because the dataset split below is not the criterion it looks
    like: which pairs land in which half is decided by the seed arithmetic,
    and the resulting halves differ per model.
    """
    counts = Counter(abs(int(records[i]["prompt_idx"]) - int(records[j]["prompt_idx"]))
                     for i, j in pairs)
    return {str(k): int(counts[k]) for k in sorted(counts)}


def direction_report(records: list[dict], frames: np.ndarray, row: int,
                     *, splits: dict[str, list[tuple[int, int]]],
                     unrelated: list[tuple[int, int]]) -> dict[str, Any]:
    """Every pair group's acute angle for ONE stored direction.

    `row` indexes the stored frame: 0 is the chord, 1 the first bend direction,
    2 the second. It is passed rather than fixed so the chord can be run
    through the identical code path as a control -- reading the chord by
    mistake would otherwise look like a spectacular confirmation, since the
    chord is dominated by -z_T and same-noise pairs share z_T exactly.
    """
    if not 0 <= row < frames.shape[1]:
        raise ValueError(f"row {row} outside the stored frame's {frames.shape[1]} rows")
    rows = np.ascontiguousarray(frames[:, row, :])
    gram = rows @ rows.T
    entry: dict[str, Any] = {}
    for name in ("same_noise_diff_prompt", "same_prompt_diff_noise"):
        entry[name] = direction_stats(pair_cos(gram, splits[name]), len(splits[name]))
    entry["unrelated"] = direction_stats(pair_cos(gram, unrelated), len(unrelated))

    ########################
    # Same-noise pairs arise from seed collisions, and which collisions exist
    # depends on how far apart the model's base seeds are. Splitting by dataset
    # separates "two prompts a couple of indices apart in one authored file"
    # from "the same index in two different files" for one model and is empty
    # for the other, so both halves carry the index distances that produced
    # them and an empty half is reported rather than hidden.
    ########################
    same_ds, cross_ds = [], []
    for i, j in splits["same_noise_diff_prompt"]:
        (same_ds if records[i]["dataset"] == records[j]["dataset"] else cross_ds).append((i, j))
    for name, group in (("same_noise_same_dataset", same_ds),
                        ("same_noise_cross_dataset", cross_ds)):
        entry[name] = direction_stats(pair_cos(gram, group), len(group))
        entry[name]["prompt_index_distances"] = index_distances(records, group)
    return entry


def chord_control(frames: np.ndarray, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """One trajectory's first bend direction against the OTHER's chord.

    If the stored PC1 were really the partner's chord with the own chord taken
    out -- the shape a plane-sharing artefact would have -- this comparison
    would return the chord-chord angle exactly. Both orderings are pooled
    because the pair is unordered.
    """
    if not pairs:
        return {"pairs": 0}
    idx = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    pc1, chord = frames[:, 1, :], frames[:, 0, :]
    cos = np.concatenate([
        np.einsum("ij,ij->i", pc1[idx[:, 0]], chord[idx[:, 1]]),
        np.einsum("ij,ij->i", pc1[idx[:, 1]], chord[idx[:, 0]]),
    ]).astype(np.float64)
    return direction_stats(cos, 2 * len(pairs))


# ---------------------------------------------------------------------------
# C. how far one trajectory's own plane turns along the path
# ---------------------------------------------------------------------------


def plane_turn_report(run_dirs: list[Path], seg_a: str, seg_b: str) -> dict[str, Any]:
    """Principal angles between each trajectory's own plane on two windows.

    This is a measurement of the trajectory, not of the instrument: a path
    that is not exactly planar carries its local plane around as it goes, and
    a short window resolves that plane far less sharply than the whole path
    does. Read it against the matching per-segment floor from section A.

    Processed one run dir at a time: this needs two segments in memory where
    every other section needs one. Trajectories are matched by `prompt_idx`
    rather than by position in the frame stack, because a record missing a
    frame for one segment but not the other would otherwise shift the
    alignment and silently compare different trajectories.
    """
    t1_all, t2_all, pc1_all = [], [], []
    matched = 0
    for run_dir in run_dirs:
        recs_a, frames_a = load_cell(run_dir, seg_a)
        recs_b, frames_b = load_cell(run_dir, seg_b)
        if frames_a is None or frames_b is None:
            continue
        idx_a, idx_b = frame_positions(recs_a), frame_positions(recs_b)
        shared = sorted(set(idx_a) & set(idx_b))
        if not shared:
            continue
        fa = orthonormalize(frames_a[[idx_a[p] for p in shared]])
        fb = orthonormalize(frames_b[[idx_b[p] for p in shared]])
        frames_a = frames_b = None

        # interleave as [A_plane_0, B_plane_0, A_plane_1, ...] so plane 2t is
        # trajectory t's window A and 2t+1 its window B, the row layout
        # pair_angles_deg indexes
        n = len(shared)
        rows = np.empty((4 * n, fa.shape[2]), dtype=np.float32)
        rows[0::4], rows[1::4] = fa[:, 1, :], fa[:, 2, :]
        rows[2::4], rows[3::4] = fb[:, 1, :], fb[:, 2, :]
        gram = rows @ rows.T
        t1, t2 = pair_angles_deg(gram, [(2 * t, 2 * t + 1) for t in range(n)])
        t1_all.append(t1)
        t2_all.append(t2)
        cos1 = np.abs(np.einsum("ij,ij->i", fa[:, 1, :], fb[:, 1, :]).astype(np.float64))
        pc1_all.append(np.degrees(np.arccos(np.clip(cos1, 0.0, 1.0))))
        matched += n

    if not t1_all:
        return {"trajectories": 0}
    t1, t2, pc1 = np.concatenate(t1_all), np.concatenate(t2_all), np.concatenate(pc1_all)
    if not (len(t1) == len(t2) == len(pc1) == matched):
        raise ValueError(f"counted {matched} trajectories but produced {len(t1)} angles")
    return {"trajectories": matched, "segments": [seg_a, seg_b],
            "theta1": _quant(t1), "theta2": _quant(t2), "first_direction": _quant(pc1)}


# ---------------------------------------------------------------------------


def model_report(run_dirs: list[Path], pair_segment: str, turn_segments: tuple[str, str]
                 ) -> dict[str, Any]:
    records: list[dict] = []
    frame_blocks: list[np.ndarray] = []
    seen_cells: set[tuple[str, str, int]] = set()
    for run_dir in run_dirs:
        recs, frames = load_cell(run_dir, pair_segment)
        if not recs:
            continue
        head = recs[0]
        cell = (head["model"], head["dataset"], int(head["seed"]))
        if cell in seen_cells:
            raise SystemExit(f"two run dirs claim cell {cell}; second is {run_dir}")
        seen_cells.add(cell)
        records.extend(recs)
        if frames is not None:
            frame_blocks.append(frames)

    devices = sorted({r["device_name"] for r in records if r.get("device_name")})
    report: dict[str, Any] = {"segment": pair_segment, "cells": len(seen_cells),
                              "records": len(records), "devices": devices}
    if not frame_blocks:
        raise SystemExit(f"no frames stored for segment {pair_segment} under these run dirs")

    framed = [r for r in records if r.get("_has_frame")]
    frames = orthonormalize(np.concatenate(frame_blocks))
    frame_blocks.clear()
    if frames.shape[0] != len(framed):
        raise ValueError(f"{frames.shape[0]} frames but {len(framed)} framed records")

    splits = split_pairs(framed)
    related = (set(splits["same_noise_diff_prompt"]) | set(splits["same_prompt_diff_noise"])
               | set(splits["same_both"]))
    unrelated = sample_pairs(frames.shape[0], MAX_UNRELATED, seed=UNRELATED_SEED, exclude=related)

    report["framed"] = frames.shape[0]
    report["dim"] = int(frames.shape[2])
    # The store is float16, so the three stored rows come back slightly
    # non-orthonormal and the first bend direction carries a little of the
    # chord. Every angle below is biased by whatever is left, so what is left
    # is reported: ~1e-8 after the fix against ~1e-5 without it.
    report["max_chord_leak_into_pc1"] = float(
        np.abs(np.einsum("ij,ij->i", frames[:, 0, :], frames[:, 1, :])).max())
    report["first_direction"] = direction_report(framed, frames, 1, splits=splits,
                                                 unrelated=unrelated)
    report["second_direction"] = direction_report(framed, frames, 2, splits=splits,
                                                  unrelated=unrelated)
    report["chord_control"] = direction_report(framed, frames, 0, splits=splits,
                                               unrelated=unrelated)
    report["pc1_vs_other_chord"] = {
        name: chord_control(frames, splits[name] if name != "unrelated" else unrelated)
        for name in ("same_noise_diff_prompt", "unrelated")}
    report["random_direction_null"] = random_direction_null(int(frames.shape[2]))
    del frames
    report["plane_turn"] = plane_turn_report(run_dirs, *turn_segments)
    return report


def validate_segments(pair_segment: str, turn_segments: tuple[str, ...],
                      stored: list[str]) -> None:
    """Everything decidable from the arguments alone, before a gigabyte is read.

    On the real grid each model's sections take tens of minutes, so a mistyped
    tag that only surfaces at the end discards all of it — and `available_segments`
    reporting nothing (a store with no frames at all) must not be read as
    "every tag is fine".
    """
    if len(turn_segments) != 2:
        raise SystemExit("--turn_segments takes exactly two comma-separated tags")
    (a0, a1), (b0, b1) = (parse_tag(t) for t in turn_segments)
    if a0 < b1 and b0 < a1:
        raise SystemExit(
            f"--turn_segments {turn_segments[0]},{turn_segments[1]} overlap; the two windows "
            f"must share no step, or they would agree by construction")
    if not stored:
        raise SystemExit("the store holds no frames for any segment")
    for tag in (pair_segment, *turn_segments):
        if tag not in stored:
            raise SystemExit(f"segment {tag!r} is not in the store; it holds {stored}")


def _fmt(entry: dict[str, Any]) -> str:
    if not entry.get("pairs"):
        return "     (no pairs at all — see prompt_index_distances / the seed arithmetic)"
    return (f"{entry['pairs']:>7d} pairs  theta med={entry['theta_med']:6.2f}"
            f"  [q05 {entry['theta_q05']:6.2f}, q95 {entry['theta_q95']:6.2f}]"
            f"  cos med={entry['cos_med']:.4f}")


def _fmt_q(entry: dict[str, float]) -> str:
    return f"med={entry['med']:6.2f}  q95={entry['q95']:6.2f}  max={entry['max']:6.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="store root holding <model>/<run_name>/ dirs")
    parser.add_argument("--pair_segment", default=WHOLE_PATH,
                        help="segment the pair comparisons run on (default %(default)s)")
    parser.add_argument("--turn_segments", default="000_016,038_051",
                        help="two DISJOINT segment tags for the self-comparison "
                             "(default %(default)s)")
    parser.add_argument("--latents", type=Path, default=DEFAULT_LATENTS,
                        help="archived full latent paths for the floor (default %(default)s)")
    parser.add_argument("--floor_repeats", type=int, default=FLOOR_REPEATS)
    parser.add_argument("--out", type=Path, default=None,
                        help="report json (default <root>/plane_identifiability.json)")
    args = parser.parse_args()

    turn_segments = tuple(s.strip() for s in args.turn_segments.split(","))
    validate_segments(args.pair_segment, turn_segments, available_segments(args.root))
    out = args.out or (args.root / "plane_identifiability.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    groups = run_dirs_by_model(args.root)
    if not groups:
        raise SystemExit(f"no <model>/<run_name>/ dirs with records under {args.root}")

    report: dict[str, Any] = {"root": str(args.root), "pair_segment": args.pair_segment,
                              "turn_segments": list(turn_segments), "models": {}}

    print("=== A. resolution: how far a plane moves under bf16-scale rounding ===")
    print(f"    (archived latents under {args.latents}; a differently rounded run of the "
          f"SAME trajectory)")
    floors = floor_report(args.latents, sorted({args.pair_segment, *turn_segments}),
                          repeats=args.floor_repeats)
    report["floor"] = floors
    if not floors:
        print(f"    no latents under {args.latents} — sections B and C have no resolution "
              f"and their angles cannot be called large or small")
    for model, entry in sorted(floors.items()):
        print(f"  {model}: {entry['trajectories']} trajectories x {entry['repeats']} rounds")
        for tag in sorted(k for k in entry if k not in {"trajectories", "repeats",
                                                        "relative_jitter"}):
            seg = entry[tag]
            print(f"     {tag}  theta1 {_fmt_q(seg['theta1'])}   theta2 {_fmt_q(seg['theta2'])}"
                  f"   PC1 {_fmt_q(seg['first_direction'])}")

    for model in sorted(groups):
        print(f"\n=== {model} ===", flush=True)
        entry = model_report(groups[model], args.pair_segment, turn_segments)
        report["models"][model] = entry
        if len(entry["devices"]) > 1:
            print(f"  WARNING: records come from {entry['devices']}; the same seed gives a "
                  f"different initial noise across GPU architectures, which is the split "
                  f"section B rests on")
        print(f"  {entry['cells']} cells, {entry['records']} records, {entry['framed']} framed, "
              f"d={entry['dim']}, segment {entry['segment']}")

        for section, label in (("first_direction", "B. FIRST bend direction"),
                               ("second_direction", "B'. second bend direction"),
                               ("chord_control", "B''. CHORD (control: shares -z_T by "
                                                 "construction, must NOT be read as bending)")):
            print(f"  {label}")
            for name in ("same_noise_diff_prompt", "same_noise_same_dataset",
                         "same_noise_cross_dataset", "same_prompt_diff_noise", "unrelated"):
                print(f"     {name:<26}{_fmt(entry[section][name])}")
                dists = entry[section][name].get("prompt_index_distances")
                if dists:
                    print(f"     {'':<26}   prompt index distances: {dists}")
        print("  B'''. PC1 against the OTHER trajectory's chord (control: an artefact would "
              "return the chord-chord angle)")
        for name, stats in entry["pc1_vs_other_chord"].items():
            print(f"     {name:<26}{_fmt(stats)}")
        null = entry["random_direction_null"]
        print(f"     {'random-direction null':<26}{_fmt(null)}")
        print(f"     {'':<26}   {null['also_the_null_for']}")

        turn = entry["plane_turn"]
        print(f"  C. one trajectory's own plane, {turn.get('segments')} — NOT a reference for B")
        if not turn.get("trajectories"):
            print("     (no trajectory carries both windows)")
        else:
            print(f"     {turn['trajectories']} trajectories   theta1 {_fmt_q(turn['theta1'])}"
                  f"   theta2 {_fmt_q(turn['theta2'])}")
            print(f"     first direction against itself: {_fmt_q(turn['first_direction'])}")

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
