#!/usr/bin/env python3
"""Trajectory regularity metrics over a sampled latent path.

Pure functions, no model / IO / torch dependency: every entry point takes a
latent sequence `Z` of shape `[N+1, d]` (row `n` = latent after solver step
`n`, row 0 = z_T) plus the scheduler `sigmas` of length `N+1`, and returns
plain Python floats / lists of floats.

`Z` is expected fp32 (the probes cast the captured bf16 latents before calling
in); the reductions here promote to float64 so that ~5e5-dimensional sums do
not lose the small perpendicular components the deviation profile is about.

Quantity definitions (docs/research_plan_full_trajectory.md section 3):

  chord         c = Z[N] - Z[0],  u = c / ||c||
  d_perp[n]     || (Z[n] - Z[0]) - ((Z[n] - Z[0]) . u) u ||        (N+1 values)
  max_dev_ratio max_n d_perp[n] / ||c||
  path_len      sum_n ||Z[n+1] - Z[n]||
  straightness  path_len / ||c||        (1.0 = perfectly straight, > 1 winding)
  spacing[n]    ||Z[n+1] - Z[n]||                                  (N values)
  magnitude[n]  ||Z[n]||                                           (N+1 values)
  turn[n]       angle(dZ[n], dZ[n+1]) in degrees                   (N-1 values)
  turn_w[n]     angle(Z[n+w]-Z[n], Z[n+2w]-Z[n+w]) in degrees      (N+1-2w values)
  curvature[n]  ||dZ[n+1] - dZ[n]||                                (N-1 values)
  velocity[n]   ||dZ[n]|| / |sigma[n+1] - sigma[n]|                (N values)

The chord-orthogonal PCA uses the (N+1)-point Gram trick: after centering and
removing the chord direction, the residual matrix P is [N+1, d] with rank <=
N, so the eigenvalues of the tiny (N+1) x (N+1) Gram matrix `P P^T` are exactly
the eigenvalues of the d x d covariance, and no d x d matrix is ever formed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

TOP_K = 5  # number of explained-variance ratios reported
_ROUNDING = 1e-20  # energy ratio below which a residual is float64 noise
_RANK_FLOOR = 1e-12  # eigenvalue ratio below which a direction is not resolved
RECON_DIMS = (1, 2, 3)  # subspace dimensions reported as reconstruction errors
CURVATURE_WINDOW = 5  # single-step turns sit at the bf16 quantization floor
# Both windows are stored per generation. w=5 clears the bf16 floor everywhere
# on FLUX (3.4-25x) but only at 29 of 41 window centers on Qwen (min 2.6x),
# where it biased the measured trough four steps late; w=7 clears it on both
# (Qwen min 3.8x). Recomputing needs the latents, which are not stored, so
# the second window has to be written at generation time or not at all.
CURVATURE_WINDOWS = (5, 7)
PLANE_PCS = 2  # chord-orthogonal principal components kept in the stored frame


def _as_f64(x: Any) -> np.ndarray:
    """float64 view of a numpy array or CPU torch tensor."""
    return np.asarray(x, dtype=np.float64)


def _prepare(Z: Any) -> np.ndarray:
    Z = _as_f64(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be [N+1, d], got shape {Z.shape}")
    if Z.shape[0] < 2:
        raise ValueError(f"Z needs at least 2 points, got {Z.shape[0]}")
    return Z


# ---------------------------------------------------------------------------
# chord frame
# ---------------------------------------------------------------------------


def chord(Z: Any) -> tuple[np.ndarray, float]:
    """Init-to-final chord direction `u` and its length. `u` is all-zero when
    the chord degenerates (Z[N] == Z[0])."""
    Z = _prepare(Z)
    c = Z[-1] - Z[0]
    length = float(np.linalg.norm(c))
    u = c / length if length > 0.0 else np.zeros_like(c)
    return u, length


def deviation_profile(Z: Any) -> tuple[list[float], float, float]:
    """Perpendicular deviation from the chord: `(d_perp[0..N], chord_len,
    max_dev_ratio)`. Endpoints are 0 by construction."""
    Z = _prepare(Z)
    u, chord_len = chord(Z)
    if chord_len == 0.0:
        nan = [float("nan")] * Z.shape[0]
        return nan, 0.0, float("nan")
    rel = Z - Z[0]
    along = rel @ u
    perp = rel - np.outer(along, u)
    d_perp = np.linalg.norm(perp, axis=1)
    return [float(v) for v in d_perp], chord_len, float(d_perp.max() / chord_len)


# ---------------------------------------------------------------------------
# per-step profiles
# ---------------------------------------------------------------------------


def step_spacing(Z: Any) -> list[float]:
    """||Z[n+1] - Z[n]|| for n = 0..N-1."""
    Z = _prepare(Z)
    return [float(v) for v in np.linalg.norm(np.diff(Z, axis=0), axis=1)]


def magnitudes(Z: Any) -> list[float]:
    """||Z[n]|| for n = 0..N."""
    Z = _prepare(Z)
    return [float(v) for v in np.linalg.norm(Z, axis=1)]


def path_length(Z: Any) -> tuple[float, float]:
    """`(path_len, straightness_ratio)` with straightness = path / chord."""
    Z = _prepare(Z)
    path = float(np.linalg.norm(np.diff(Z, axis=0), axis=1).sum())
    _, chord_len = chord(Z)
    ratio = path / chord_len if chord_len > 0.0 else float("nan")
    return path, ratio


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees, nan if either is zero-length.

    Uses the half-angle atan2 identity rather than arccos(cos) so small angles
    keep full float64 precision.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    ua, ub = a / na, b / nb
    return math.degrees(2.0 * math.atan2(
        float(np.linalg.norm(ua - ub)), float(np.linalg.norm(ua + ub))
    ))


def turn_angles_deg(Z: Any) -> list[float]:
    """Angle between consecutive step vectors, degrees, for n = 0..N-2.

    On bf16 latents these single-step turns sit at the quantization floor for
    all but the last couple of junctions (docs/full_trajectory_results.md
    section 2), so read `turn_angles_window_deg` instead for curvature.
    """
    Z = _prepare(Z)
    dZ = np.diff(Z, axis=0)
    return [_angle_deg(a, b) for a, b in zip(dZ[:-1], dZ[1:])]


def turn_angles_window_deg(Z: Any, window: int = CURVATURE_WINDOW) -> list[float]:
    """Coarse-window turn angle: angle between the displacement over the
    `window` steps before junction `n` and the one over the `window` steps
    after it, in degrees, for n = window .. N - window.

    Summing `window` consecutive steps raises the signal above the bf16
    quantization floor that swamps the single-step turn, at the price of
    resolving only features wider than the window. Returns an empty list when
    the trajectory is too short to hold one full window on each side.
    """
    Z = _prepare(Z)
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return [
        _angle_deg(Z[n + w] - Z[n], Z[n + 2 * w] - Z[n + w])
        for n in range(Z.shape[0] - 2 * w)
    ]


def window_centers(n_rows: int, window: int = CURVATURE_WINDOW) -> list[int]:
    """Row indices the `turn_angles_window_deg` entries are centred on."""
    w = int(window)
    return list(range(w, max(w, n_rows - w)))


def second_difference_norms(Z: Any) -> list[float]:
    """Discrete curvature ||dZ[n+1] - dZ[n]|| for n = 0..N-2."""
    Z = _prepare(Z)
    dZ = np.diff(Z, axis=0)
    return [float(v) for v in np.linalg.norm(np.diff(dZ, axis=0), axis=1)]


def velocity_norms(Z: Any, sigmas: Sequence[float]) -> list[float]:
    """||dZ[n]|| / |sigma[n+1] - sigma[n]| for n = 0..N-1 (nan on a zero gap)."""
    Z = _prepare(Z)
    s = _as_f64(sigmas).ravel()
    if s.shape[0] != Z.shape[0]:
        raise ValueError(f"sigmas has {s.shape[0]} entries, Z has {Z.shape[0]} rows")
    spacing = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    dsigma = np.abs(np.diff(s))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dsigma > 0.0, spacing / dsigma, np.nan)
    return [float(x) for x in v]


# ---------------------------------------------------------------------------
# PCA on the chord-orthogonal complement
# ---------------------------------------------------------------------------


def orthogonal_pca(Z: Any, top_k: int = TOP_K) -> Dict[str, Any]:
    """PCA of the centred, chord-orthogonal residuals via the Gram trick.

    Returns explained-variance ratios of the leading `top_k` components plus
    the Frobenius reconstruction error of the 1/2/3-D approximation (absolute
    and relative to the total residual norm). Dimension counting follows the
    root paper (reference/trajectory_regularity/paper/secs/visualization.tex):
    the k-D approximation keeps the chord plus the top k-1 principal
    components, so `recon_err_1d` is the full perpendicular residual
    sqrt(perp_var_total) (chord only, `recon_err_rel_1d` == 1), and the
    paper's "~85% variance explained at 3-D" corresponds to
    `pca_evr[0] + pca_evr[1]`. A perfectly straight trajectory has no
    residual at all; that degenerate case reports zeros and
    `perp_var_total == 0`.
    """
    Z = _prepare(Z)
    u, _ = chord(Z)
    Y = Z - Z.mean(axis=0, keepdims=True)
    P = Y - np.outer(Y @ u, u)
    gram = P @ P.T
    gram = 0.5 * (gram + gram.T)  # symmetrize away the float64 asymmetry
    eig = np.linalg.eigvalsh(gram)[::-1]
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())

    evr = np.zeros(top_k, dtype=np.float64)
    if total > 0.0:
        take = min(top_k, eig.shape[0])
        evr[:take] = eig[:take] / total

    out: Dict[str, Any] = {
        "pca_evr": [float(v) for v in evr],
        "perp_var_total": total,
    }
    for k in RECON_DIMS:
        # paper convention: k-D approximation = chord + (k - 1) top PCs
        pcs = k - 1
        tail = float(eig[pcs:].sum()) if eig.shape[0] > pcs else 0.0
        err = math.sqrt(max(tail, 0.0))
        out[f"recon_err_{k}d"] = err
        out[f"recon_err_rel_{k}d"] = err / math.sqrt(total) if total > 0.0 else 0.0
    return out


# ---------------------------------------------------------------------------
# the plane the trajectory bends in
# ---------------------------------------------------------------------------


def _position_plane(
    Z: np.ndarray, u: np.ndarray, n_pcs: int = PLANE_PCS, rank_floor: float = 0.0
) -> np.ndarray | None:
    """Orthonormal `[n_pcs, d]` basis of the leading chord-orthogonal directions
    of the *positions*, or None when the trajectory carries no bend above
    float64 rounding. `Z` must already be float64 and `u` a unit chord.

    The Gram trick applies as in `orthogonal_pca`: P is [N+1, d] of rank <= N,
    so its right singular vectors come from the tiny (N+1)x(N+1) Gram matrix.

    `rank_floor > 0` additionally rejects a residual whose `n_pcs`-th
    eigenvalue is below that fraction of the first: the directions are then
    numerically rank-deficient and the trailing rows of the QR are arbitrary.
    A caller that only projects onto the span can leave it at 0 (an arbitrary
    direction carrying no energy contributes nothing to a projection); a caller
    that stores or compares the directions themselves must set it.
    """
    centred = Z - Z.mean(axis=0, keepdims=True)
    P = centred - np.outer(centred @ u, u)
    gram = P @ P.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1][:n_pcs]
    if order.shape[0] < n_pcs:
        return None
    # a straight trajectory's off-chord content is float64 rounding; fitting a
    # plane to it would report an arbitrary fraction, so bail out on a relative
    # threshold rather than on exact zero
    if float(values[order].sum()) <= _ROUNDING * float((centred ** 2).sum()):
        return None
    if rank_floor > 0.0 and float(values[order[-1]]) <= rank_floor * float(values[order[0]]):
        return None
    return np.linalg.qr((vectors[:, order].T @ P).T)[0].T


def parse_segments(spec: str, n_rows: int) -> list[tuple[int, int]]:
    """Parse "a:b,a:b,..." into validated half-open row ranges.

    Every segment must hold enough rows for a `[chord, PC1, PC2]` frame to
    mean anything; the check is here rather than at the call site so a
    malformed spec fails at argument-parse time, not after a GPU hour.
    """
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo, _, hi = chunk.partition(":")
        if not _:
            raise ValueError(f"segment {chunk!r} must look like 'a:b'")
        a, b = int(lo), int(hi)
        if not (0 <= a < b <= n_rows):
            raise ValueError(f"segment {a}:{b} outside 0..{n_rows}")
        if b - a < 2 + PLANE_PCS:
            raise ValueError(
                f"segment {a}:{b} has {b - a} rows; a chord plus {PLANE_PCS} "
                f"principal directions needs at least {2 + PLANE_PCS}"
            )
        if (a, b) in out:
            raise ValueError(f"segment {a}:{b} given twice")
        out.append((a, b))
    if not out:
        raise ValueError("no segments parsed")
    return out


def segment_tag(a: int, b: int) -> str:
    """Filename/record key for one segment. Stable across runs."""
    return f"{a:03d}_{b:03d}"


def plane_frame(Z: Any, n_pcs: int = PLANE_PCS) -> np.ndarray | None:
    """`[1 + n_pcs, d]` orthonormal frame of one trajectory: the chord
    direction followed by the leading chord-orthogonal principal directions of
    the positions. None when the chord degenerates or the residual is
    numerically rank-deficient, since the frame's whole content is the
    directions themselves.

    This is the object that has to be stored to compare *where* two
    trajectories bend: that comparison is pairwise and no per-trajectory scalar
    can stand in for it. Every other quantity in this module is invariant to
    the frame's orientation; this one is the orientation.
    """
    Z = _prepare(Z)
    u, chord_len = chord(Z)
    if chord_len == 0.0:
        return None
    basis = _position_plane(Z, u, n_pcs, rank_floor=_RANK_FLOOR)
    if basis is None:
        return None
    return np.vstack([u[None, :], basis])


def principal_angles_deg(A: Any, B: Any) -> list[float]:
    """Principal angles (degrees, ascending) between the row spaces of two
    orthonormal bases of equal width.

    0 = the two subspaces share that direction, 90 = orthogonal in it. Inputs
    are re-orthonormalized first, so a basis that has been round-tripped
    through a low-precision store is still handled exactly.
    """
    A, B = _as_f64(A), _as_f64(B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"bases must be 2-D, got {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"bases live in different spaces: {A.shape} vs {B.shape}")
    Qa = np.linalg.qr(A.T)[0].T
    Qb = np.linalg.qr(B.T)[0].T
    singular = np.clip(np.linalg.svd(Qa @ Qb.T, compute_uv=False), 0.0, 1.0)
    return [float(v) for v in np.degrees(np.arccos(singular))]


# ---------------------------------------------------------------------------
# dimensionality of the step-to-step update
# ---------------------------------------------------------------------------


def update_subspace(Z: Any, top_k: int = TOP_K) -> Dict[str, Any]:
    """How low-dimensional the *updates* are, as opposed to the positions.

    Positions sit ~95% inside a chord + 2-PC frame, but an update is a
    difference and differencing amplifies whatever high-frequency content the
    smooth arc does not contain, so the two are separate questions. Three
    quantities, all computed on the same trajectory:

      `chord_share`
          fraction of the update energy that points along the chord. Mostly a
          restatement of "each step moves forward"; reported because it is the
          part the next number must be measured *without*, otherwise it
          dominates and inflates the answer.
      `in_position_plane`
          of the off-chord update energy, the fraction captured by the top-2
          PCs of the positions' off-chord residuals — i.e. do the updates stay
          in the plane the trajectory bends in?
      `own_evr`
          explained-variance ratios of the off-chord updates' *own* leading
          directions. Not mean-centred: the question is how many directions the
          update vectors span, not how they vary about their mean.

    A perfectly straight trajectory has no off-chord content at all; that case
    reports `nan` for the two fractions rather than 0/0.
    """
    Z = _prepare(Z)
    u, chord_len = chord(Z)
    out: Dict[str, Any] = {
        "chord_share": float("nan"),
        "in_position_plane": float("nan"),
        "own_evr": [0.0] * top_k,
    }
    if chord_len == 0.0:
        return out

    # updates: split into chord and off-chord parts (independent of the plane,
    # so this is reported even when the plane below degenerates)
    dZ = np.diff(Z, axis=0)
    total = float((dZ ** 2).sum())
    if total <= 0.0:
        return out
    along = np.outer(dZ @ u, u)
    off = dZ - along
    out["chord_share"] = float((along ** 2).sum() / total)
    off_energy = float((off ** 2).sum())
    if off_energy <= _ROUNDING * total:
        return out

    # positions: the plane they bend in — same object `plane_frame` stores
    basis = _position_plane(Z, u)
    if basis is None:
        return out
    out["in_position_plane"] = float(((off @ basis.T) ** 2).sum() / off_energy)

    # updates' own leading directions (uncentred Gram of the off-chord parts)
    gram_u = off @ off.T
    eig = np.linalg.eigvalsh(0.5 * (gram_u + gram_u.T))[::-1]
    eig = np.clip(eig, 0.0, None)
    total_u = float(eig.sum())
    if total_u > 0.0:
        take = min(top_k, eig.shape[0])
        evr = np.zeros(top_k, dtype=np.float64)
        evr[:take] = eig[:take] / total_u
        out["own_evr"] = [float(v) for v in evr]
    return out


# ---------------------------------------------------------------------------
# one-call bundle
# ---------------------------------------------------------------------------


def trajectory_metrics(Z: Any, sigmas: Sequence[float]) -> Dict[str, Any]:
    """Every metric of section 3 for one trajectory, as JSON-ready plain types."""
    Z = _prepare(Z)
    d_perp, chord_len, max_dev_ratio = deviation_profile(Z)
    path_len, straightness = path_length(Z)
    metrics: Dict[str, Any] = {
        "chord_len": chord_len,
        "path_len": path_len,
        "max_dev_ratio": max_dev_ratio,
        "straightness": straightness,
        "d_perp": d_perp,
        "spacing": step_spacing(Z),
        "magnitude": magnitudes(Z),
        "turn_angle_deg": turn_angles_deg(Z),
        **{f"turn_angle_w{w}_deg": turn_angles_window_deg(Z, w)
           for w in CURVATURE_WINDOWS},
        "second_diff_norm": second_difference_norms(Z),
        "velocity_norm": velocity_norms(Z, sigmas),
    }
    metrics.update(orthogonal_pca(Z))
    update = update_subspace(Z)
    metrics["update_chord_share"] = update["chord_share"]
    metrics["update_in_position_plane"] = update["in_position_plane"]
    metrics["update_own_evr"] = update["own_evr"]
    return metrics
