#!/usr/bin/env python3
"""bf16 quantization floor for the per-step trajectory readings.

The archived trajectory records store latents at the pipeline's bf16
precision, so a reading taken over a short stretch of the path can be pure
quantization noise rather than geometry. This module measures that floor
directly on the stored trajectories and reports, per reading, how far the
measurement sits above it — the numbers that decide which per-step readings
in `docs/full_trajectory_results.md` are usable.

All four floors share one construction: build an fp64 point whose geometry is
zero (or exactly known) by construction, round it once through bf16, and read
out what the rounding alone produced.

  turn angle  for a window `w`, replace the middle point of the triple
              (n, n+w, n+2w) by the exact collinear midpoint of its two
              endpoints. A straight path must read 0 deg.
  deviation   replace every state by its orthogonal projection on the chord.
              The perpendicular distance is 0 by construction.
  spacing /   the stored points are already on the bf16 grid, so re-rounding
  magnitude   them is a no-op and would report a floor of exactly zero. The
              fp64 points a rounding can still act on are the exact midpoints
              of consecutive states: their norms and separations are known in
              fp64, and rounding them once gives the relative amount by which
              the instrument moves each reading.

`--quant_dtype float32` measures the same floors for a float32 store (the
video backbones' T1 profiles are computed on float32 rows in flight, see
docs/video_full_trajectory_plan_zh.md section 3.0). The stored paths are still
the bf16 T3 files, so the geometry is approximate, and one construction
detail changes: the exact midpoint of two bf16-grid points is float32-
representable, so under float32 the straightened middle point sits at the
arc-length fraction of its window and the scalar base point at 1/3 of the
step instead of 1/2 -- generic fp64 points a float32 rounding acts on.

Usage:
    python analysis/trajectory_bf16_floor.py                 # w=1 and w=5, + scalars
    python analysis/trajectory_bf16_floor.py --windows 1 3 5 9
    python analysis/trajectory_bf16_floor.py --quant_dtype float32 --json floor.json
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

DEFAULT_LATENTS = "resources/full_trajectory/latents_flux/latents_*.pt"
QUANT_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
_QUANT = torch.bfloat16  # set by --quant_dtype; bf16 keeps the archived behaviour


def load_trajectory(path: str | Path) -> np.ndarray:
    """[N+1, d] fp64 view of one stored bf16 trajectory."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    tensor = obj if torch.is_tensor(obj) else next(
        v for v in obj.values() if torch.is_tensor(v) and v.ndim >= 2
    )
    return tensor.to(torch.float32).numpy().reshape(tensor.shape[0], -1).astype(np.float64)


def _bf16(x: np.ndarray) -> np.ndarray:
    """One rounding through the quantization dtype under test (bf16 = the
    precision the latents are stored at)."""
    return torch.from_numpy(x).to(_QUANT).to(torch.float32).numpy().astype(np.float64)


def _base_fraction() -> float:
    """Where along a step the scalar-floor base point sits: the exact midpoint
    for bf16 (unchanged), 1/3 for float32 where the midpoint of two bf16-grid
    points is float32-exact and a rounding would be a no-op."""
    return 0.5 if _QUANT == torch.bfloat16 else 1.0 / 3.0


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees, via the half-angle identity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    ua, ub = a / na, b / nb
    return math.degrees(
        2.0 * math.atan2(float(np.linalg.norm(ua - ub)), float(np.linalg.norm(ua + ub)))
    )


def turn_profile(Z: np.ndarray, window: int) -> np.ndarray:
    """Measured turn angle over `window`-step chords, one per junction."""
    return np.array([
        _angle(Z[n + window] - Z[n], Z[n + 2 * window] - Z[n + window])
        for n in range(Z.shape[0] - 2 * window)
    ])


def floor_profile(Z: np.ndarray, window: int) -> np.ndarray:
    """Angle produced by bf16 rounding alone on an exactly straightened triple."""
    out = []
    for n in range(Z.shape[0] - 2 * window):
        a, c = Z[n], Z[n + 2 * window]
        if _QUANT == torch.bfloat16:
            mid = 0.5 * (a + c)
        else:  # generic point on the segment: the arc-length fraction of the window
            b = Z[n + window]
            t = float(np.linalg.norm(b - a) / (np.linalg.norm(b - a) + np.linalg.norm(c - b)))
            mid = a + t * (c - a)
        q = _bf16(mid)
        out.append(_angle(q - a, c - q))
    return np.array(out)


def deviation_profile(Z: np.ndarray) -> np.ndarray:
    """Measured distance from the chord, per state, as a fraction of the chord."""
    u = Z[-1] - Z[0]
    chord = float(np.linalg.norm(u))
    u = u / chord
    r = Z - Z[0]
    perp = r - np.outer(r @ u, u)
    return np.linalg.norm(perp, axis=1) / chord


def deviation_floor(Z: np.ndarray) -> np.ndarray:
    """Distance from the chord read out from a path that lies exactly on it.

    Every state is replaced by its orthogonal projection on the chord, so the
    perpendicular component is zero in fp64; rounding those points once through
    bf16 gives what the instrument reports for a path with no deviation at all.
    """
    u = Z[-1] - Z[0]
    chord = float(np.linalg.norm(u))
    u = u / chord
    t = (Z - Z[0]) @ u
    projected = Z[0] + np.outer(t, u)
    return deviation_profile(np.vstack([Z[:1], _bf16(projected)[1:-1], Z[-1:]]))


def scalar_floor(Z: np.ndarray) -> dict[str, np.ndarray]:
    """Relative amount by which one bf16 rounding moves `magnitude`/`spacing`.

    Both readings are norms of stored points, which are already on the bf16
    grid; the points a fresh rounding can act on are the exact fp64 midpoints
    `a` of consecutive states. `a` carries the trajectory's own magnitude, and
    the pair `(a, a + step)` is separated by exactly one step's displacement —
    so rounding them once reports the floor at each reading's own scale.

    Returned SIGNED, because the two behave differently: rounding perturbs the
    two endpoints of a step independently, which lengthens their separation on
    average, so the spacing floor is a positive bias that does not average away
    over many trajectories. The magnitude floor is a norm of one rounded point
    and stays symmetric.
    """
    step = np.diff(Z, axis=0)
    a = Z[:-1] + _base_fraction() * step
    a_q, b_q = _bf16(a), _bf16(a + step)
    mag = np.linalg.norm(a, axis=1)
    sep = np.linalg.norm(step, axis=1)
    spacing_rel = (np.linalg.norm(b_q - a_q, axis=1) - sep) / sep

    # the chord is the same construction at the whole trajectory's separation.
    # the base point has to be off the bf16 grid or the rounding is a no-op:
    # Z[0] + (Z[-1] - Z[0]) reconstructs two stored points exactly.
    chord_vec = Z[-1] - Z[0]
    c0 = a[0]
    c0_q, c1_q = _bf16(c0), _bf16(c0 + chord_vec)
    chord = float(np.linalg.norm(chord_vec))
    chord_rel = float((np.linalg.norm(c1_q - c0_q) - chord) / chord)
    # path length is the sum of the steps, so it inherits their bias by weight
    path_rel = float((spacing_rel * sep).sum() / sep.sum())
    return {
        "magnitude_rel": (np.linalg.norm(a_q, axis=1) - mag) / mag,
        "spacing_rel": spacing_rel,
        "chord_rel": chord_rel,
        "path_len_rel": path_rel,
        "straightness_rel": path_rel - chord_rel,
    }


def plane_share_floor(Z: np.ndarray) -> tuple[float, float]:
    """Measured and floor value of the top-2 principal share of the deviation.

    Floor construction: keep only the deviation's own top-2 plane, so the share
    is exactly 1 by construction, then round the resulting points once through
    bf16 and re-measure. Whatever falls below 1 is energy the rounding alone
    scattered out of the plane.
    """
    u = Z[-1] - Z[0]
    u = u / np.linalg.norm(u)
    r = Z - Z[0]
    resid = r - np.outer(r @ u, u)

    def _share(x: np.ndarray) -> float:
        s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
        return float(s[:2].sum() / s.sum())

    basis = np.linalg.svd(resid - resid.mean(axis=0), full_matrices=False)[2][:2]
    flat = Z[0] + np.outer(r @ u, u) + (resid @ basis.T) @ basis  # exactly planar
    return _share(resid), _share(deviation_residual(_bf16(flat), u, Z[0]))


def deviation_residual(Z: np.ndarray, u: np.ndarray, origin: np.ndarray) -> np.ndarray:
    r = Z - origin
    return r - np.outer(r @ u, u)


def profiles(paths: Iterable[str], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-trajectory measured and floor profiles, stacked."""
    measured, floors = [], []
    for path in paths:
        Z = load_trajectory(path)
        measured.append(turn_profile(Z, window))
        floors.append(floor_profile(Z, window))
    return np.stack(measured), np.stack(floors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", default=DEFAULT_LATENTS,
                        help="glob of stored full trajectories")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--quant_dtype", choices=sorted(QUANT_DTYPES), default="bfloat16",
                        help="dtype the floors are measured for (default bf16, the store)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write every median profile printed here to this file")
    parser.add_argument("--tail", type=int, default=6,
                        help="how many trailing junctions to print individually")
    parser.add_argument("--per-state", action="store_true",
                        help="dump the whole measured/floor ratio profile rather than a "
                             "handful of states -- the results doc quotes states this "
                             "summary does not name, and a reading that cannot be "
                             "reprinted is a reading that cannot be checked")
    args = parser.parse_args()
    global _QUANT
    _QUANT = QUANT_DTYPES[args.quant_dtype]

    paths = sorted(glob.glob(args.latents))
    if not paths:
        raise SystemExit(f"no trajectories matched {args.latents}")
    dump: dict = {"quant_dtype": args.quant_dtype, "n_trajectories": len(paths),
                  "latents": args.latents, "windows": {}}
    print(f"quantization dtype: {args.quant_dtype}  ({len(paths)} trajectories)")

    for window in args.windows:
        measured, floors = profiles(paths, window)
        med_m = np.median(measured, axis=0)
        med_f = np.median(floors, axis=0)
        ratio = med_m / med_f
        n = len(ratio)
        dump["windows"][str(window)] = {
            "centers": [window + i for i in range(n)],
            "measured_med_deg": med_m.tolist(), "floor_med_deg": med_f.tolist(),
            "ratio": ratio.tolist()}
        print(f"\n=== window w={window}  ({len(paths)} trajectories, {n} junctions) ===")
        print(f"  ratio measured/floor: median={np.median(ratio):.2f} "
              f"min={ratio.min():.2f} max={ratio.max():.2f}")
        bulk = ratio[: max(n - args.tail, 0)]
        if bulk.size:
            print(f"  junctions 0..{n - args.tail - 1}: median={np.median(bulk):.2f} "
                  f"max={bulk.max():.2f}  ({int((bulk >= 3).sum())} of {bulk.size} at >=3x)")
        print(f"  trailing {args.tail} junctions:")
        for i in range(max(n - args.tail, 0), n):
            print(f"    junction {i:2d}: measured={med_m[i]:6.2f} deg  "
                  f"floor={med_f[i]:5.2f} deg  ratio={ratio[i]:5.2f}x")

    trajectories = [load_trajectory(path) for path in paths]

    dev_m = np.median(np.array([deviation_profile(Z) for Z in trajectories]), axis=0)
    dev_f = np.median(np.array([deviation_floor(Z) for Z in trajectories]), axis=0)
    inner = slice(1, 50)  # states 0 and 50 lie on the chord by construction
    dev_ratio = dev_m[inner] / dev_f[inner]
    print(f"\n=== deviation from the chord  ({len(paths)} trajectories, states 1..49) ===")
    print(f"  ratio measured/floor: median={np.median(dev_ratio):.1f}x "
          f"min={dev_ratio.min():.1f}x (state {1 + int(dev_ratio.argmin())}) "
          f"max={dev_ratio.max():.1f}x (state {1 + int(dev_ratio.argmax())})")
    print("  " + "  ".join(f"s{n}={dev_m[n] / dev_f[n]:.0f}x" for n in (1, 5, 10, 20, 30, 40, 49)))
    dump["deviation"] = {"measured_med": dev_m.tolist(), "floor_med": dev_f.tolist()}
    if args.per_state:
        print("  every state:")
        for n in range(1, 50):
            print(f"    state {n:2d}: measured={dev_m[n]:.6f}  floor={dev_f[n]:.6f}  "
                  f"ratio={dev_m[n] / dev_f[n]:7.2f}x")

    scalars = [scalar_floor(Z) for Z in trajectories]
    print(f"\n=== state norm / step displacement  ({len(paths)} trajectories, steps 0..49) ===")
    for key, label in (("magnitude_rel", "magnitude"), ("spacing_rel", "spacing ")):
        rel = np.median(np.array([s[key] for s in scalars]), axis=0)
        print(f"  {label}: one rounding shifts the reading by {np.median(rel):+.2e} relative "
              f"-> resolved to 1 part in {1 / abs(np.median(rel)):,.0f}")
        print(f"             per step: {rel[0]:+.2e} (step 0) ... {rel[-1]:+.2e} (step 49), "
              f"largest {rel[np.abs(rel).argmax()]:+.2e} at step {int(np.abs(rel).argmax())}")
    dump["magnitude_rel_med"] = np.median(np.array([s["magnitude_rel"] for s in scalars]), axis=0).tolist()
    dump["spacing_rel_med"] = np.median(np.array([s["spacing_rel"] for s in scalars]), axis=0).tolist()
    spacing_bias = np.median(np.array([s["spacing_rel"] for s in scalars]), axis=0)
    if spacing_bias.min() > 0:
        print("  NOTE: the spacing shift is one-signed (a stretch at every step), so it is a "
              "bias, not noise that averages out over many trajectories.")

    print(f"\n=== whole-trajectory scalars  ({len(paths)} trajectories) ===")
    for key, label in (("chord_rel", "chord length"), ("path_len_rel", "path length "),
                       ("straightness_rel", "straightness")):
        rel = float(np.median([s[key] for s in scalars]))
        dump[f"{key}_med"] = rel
        print(f"  {label}: {rel:+.2e} relative -> resolved to 1 part in {1 / abs(rel):,.0f}")

    shares = np.array([plane_share_floor(Z) for Z in trajectories])
    measured_gap = 1.0 - np.median(shares[:, 0])
    floor_gap = 1.0 - np.median(shares[:, 1])
    print(f"  top-2 plane share: measured {np.median(shares[:, 0]):.4f} "
          f"(off-plane {measured_gap:.4f}), exactly-planar path reads "
          f"{np.median(shares[:, 1]):.4f} (off-plane {floor_gap:.2e}) "
          f"-> off-plane energy is {measured_gap / floor_gap:,.0f}x the floor")
    dump["plane_share"] = {"measured_med": float(np.median(shares[:, 0])),
                           "planar_floor_med": float(np.median(shares[:, 1]))}
    if args.json is not None:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dump), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
