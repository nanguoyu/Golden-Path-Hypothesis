#!/usr/bin/env python3
"""Video full-trajectory T3 layer: the readings that need whole latent paths.

Two subcommands, two plan stages:

``refs`` (P4)
    docs/video_full_trajectory_plan_zh.md sections 3.10 (same noise / different
    prompt, per-state difference) and 3.6 (the 3-D overlay), plus the direct
    multi-window turn arrays of sections 3.8-2 / 3.0, all read off block-A
    reference paths.

``cached`` (P7)
    section 3.9.3, the T3 layer of the cache bend: a cached run's whole path
    ``Z^c`` against the same-prompt reference path ``Z^r``. See the second
    docstring block further down (``run_cached``); it runs in three stages
    (``cells`` / ``refside`` / ``merge``) over the 162 directories of
    ``cells_t3_rand50/``, all of which sit on site_a.

Input of ``refs`` is
block A of T3 -- ``$DATA/<T>/matrix/references_t3/<ds>_s<base>/latents_%05d.pt``,
bf16 ``[51, d]``, 120 per stream -- the matching T2 frame ``[chord, PC1, PC2]``
of the same generation, and the T1 reference rows for ``z_T_sha256``,
``chord_len``, ``spacing``, ``magnitude`` and ``d`` -- preferring, when both
exist, the ``references_t3/`` row written by the same run as the stored path,
and reporting which row won and at which ``path_dtype`` (the ratios below put a
bf16 numerator over a float32 denominator, and the plan requires that to be
stated).

    OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
    python analysis/video_trajectory/latent_paths.py refs \\
        --backbone hunyuan_video --data_root outputs \\
        --out_tables resources/video_full_trajectory/hunyuan_video \\
        --out_figs  docs/figures/video_full_trajectory/hunyuan_video

Written atomically, and reused only when the recorded run parameters
(``--limit`` / ``--sample`` / ``--overlay_paths`` / ``--streams`` /
``--windows`` / ``--seed`` / ``--g_floor_source``) match the ones asked for, so
a cheap smoke run cannot stand in for the production one:

    <out_tables>/latent_paths_<T>.json   every number, plus the per-path
                                         overlay coordinates and the direct
                                         multi-window turn arrays
    <out_tables>/latent_paths_<T>.md     the section 3.10 tables
    <out_figs>/g_profile_<T>.png         G[n] median profiles, three classes
    <out_figs>/<T>_traj_3d_overlay.png   30 + 30 paths in their own frames

Which floor applies, stated per reading (plan section 2.2 point 3):

  * everything computed from a T3 path is a **bf16** reading. The separation
    state of section 3.10 is defined against 3x the bf16 floor of P1
    (``p1_floor_bfloat16.json``); ``straightness`` / ``path_len`` recomputed
    here carry the +2.0e-2 (HYV) / +1.7e-2 (Wan) inflation and ``chord``
    the -2.9e-4 / -1.9e-4 shift, so the overlay is a shape picture, not a
    length measurement, and the direct turn arrays are below the bf16
    readable window (P1: bf16 needs w > 11 on HYV, w = 9 on Wan) -- they are
    stored with that verdict attached, not read here.
  * the direction frame is the stored **T2** frame, computed in flight on the
    float32 path, so the three shares are a float32-row reading of direction
    even though the magnitudes they weight are bf16.

Deliberately not here:

  * no rho_2 profile and no correlation coefficient -- section 3.7 is P5. The
    turn arrays are stored so P5 can correlate against them; nothing is
    correlated in P4.
  * the ``refs`` subcommand never opens the cached path layer and never compares a T2
    frame with a T3-recomputed one: both belong to ``cached`` (P7) below.
  * no results document and no generation. P7 writes only under
    ``resources/video_full_trajectory/<T>/`` and
    ``docs/figures/video_full_trajectory/<T>/``; the two results pages are P9
    and the block-C gate is P8.

CPU only, numpy/json/matplotlib; ``torch`` is imported lazily to open the
``.pt`` files. Runs as an sbatch CPU job, never on a login node.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import math
import os
import re
import socket
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import (  # noqa: E402
    plane_frame, principal_angles_deg, turn_angles_window_deg,
)
from analysis.trajectory_shape_scale import orthonormalize  # noqa: E402
from analysis.video_trajectory.common import (  # noqa: E402
    atomic_savefig, atomic_write_json, atomic_write_text, output_complete,
    read_json_if_readable, resolve_reuse, turn_index,
)
# P7 reuses P6's method/budget vocabulary and its two sampling caveats verbatim,
# so the T1 and T3 layers of section 3.9 cannot drift apart.
from analysis.video_trajectory.cached_vs_reference import (  # noqa: E402
    CAV_OVERLAP, CAV_POOLING, EVENT_OFFSETS, KS, MAX_NONZERO_PREFIX_ROWS,
    METHODS, NUM_STEPS, summarise_profile, summarise_samples,
)
from analysis.video_trajectory.merge_video_traj import BUDGET_K, parse_dir  # noqa: E402

_DEFAULT_FLOOR_DIR = _PROJECT_ROOT / "docs" / "figures" / "video_full_trajectory"
DEFAULT_DATA_ROOT = Path("outputs")
DEFAULT_STREAMS = ("penguin599_s54", "vbench944_s42")  # plan section 4.2 block A
N_STATES = 51
LANDMARKS = (1, 10, 25, 40, 50)  # plan section 3.10: states the shares are read at
SNR_MIN = 3.0  # plan section 3.0 readability convention, reused for "separated"
TURN_WINDOWS = (1, 5, 7, 9, 11)  # same windows P1 measured floors for
# round-to-nearest bf16 (7 explicit mantissa bits), relative RMS of the error
# vector over a UNIFORM mantissa on [1, 2): (2^-7/sqrt(12)) / sqrt(E[m^2]) with
# E[m^2] = 7/3. (A log-uniform mantissa would give 3/(2 ln 2) ~ 2.164 instead,
# i.e. ~4 % larger; the uniform value is the one this constant carries, and it
# is reported beside two data-driven routes, never alone.)
BF16_REL_RMS = (2.0 ** -7 / math.sqrt(12.0)) / math.sqrt(7.0 / 3.0)


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------


def _quantiles(values: Iterable[float]) -> dict[str, Any]:
    """n / median / IQR / 5-95% of a 1-D sample, nan-safe."""
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "p5": float("nan"), "p95": float("nan")}
    p5, p25, p50, p75, p95 = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(a.size), "median": float(p50), "p25": float(p25),
            "p75": float(p75), "p5": float(p5), "p95": float(p95)}


def _profile_stats(mat: np.ndarray) -> dict[str, list[float]]:
    """Per-column median / p25 / p75 of a [n_pairs, n_states] block."""
    if mat.size == 0:
        empty = [float("nan")] * mat.shape[1]
        return {"n": 0, "median": list(empty), "p25": list(empty), "p75": list(empty)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p25, p50, p75 = np.nanpercentile(mat, [25, 50, 75], axis=0)
    return {"n": int(mat.shape[0]), "median": [float(v) for v in p50],
            "p25": [float(v) for v in p25], "p75": [float(v) for v in p75]}


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}g}"


# ---------------------------------------------------------------------------
# P1 floors (cited, never re-measured)
# ---------------------------------------------------------------------------


def load_p1_floors(floor_dir: Path, override: Path | None) -> dict[str, Any]:
    """The bf16 floor dump of P1 for this backbone."""
    path = override or (floor_dir / "p1_floor_bfloat16.json")
    if not path.is_file():
        raise SystemExit(
            f"P1 bf16 floor dump not found at {path}. Every T3 reading has to state "
            f"its floor (plan section 2.2); pass --p1_floor_json.")
    return json.loads(path.read_text(encoding="utf-8"))


def readable_windows(floors: dict[str, Any]) -> dict[str, Any]:
    """Which turn windows clear the bf16 floor, straight out of the P1 dump."""
    out: dict[str, Any] = {}
    for w, entry in floors.get("windows", {}).items():
        ratio = np.asarray(entry["ratio"], dtype=np.float64)
        out[str(w)] = {
            "centres": len(ratio),
            "centres_ge_snr_min": int((ratio >= SNR_MIN).sum()),
            "min_snr": float(np.nanmin(ratio)),
            "readable": bool((ratio >= SNR_MIN).all()),
        }
    readable = [int(w) for w, v in out.items() if v["readable"] and int(w) > 1]
    # "multistep": w = 1 is excluded on purpose, so this is the same quantity
    # p1_floor.md calls the minimum readable MULTI-step window
    out["min_readable_multistep_window_bf16"] = min(readable) if readable else None
    return out


def g_floor_estimates(floors: dict[str, Any], chord_med: float,
                      spacing_med: np.ndarray, magnitude_med: np.ndarray
                      ) -> dict[str, Any]:
    """Floor on ``G[n] = ||Z^a[n] - Z^b[n]||`` from the bf16 store.

    G is the norm of the difference of two independently rounded states, which
    is the same object as the spacing floor (a difference of two rounded
    states one step apart) and as the deviation floor (the rounding error of
    one state, whose off-chord part is all but 1 of ~1e6 directions). Two
    independent routes out of the P1 dump, plus the analytic value; they are
    reported side by side because a disagreement between them is a statement
    about the model ``measured^2 = true^2 + floor^2``, not something to hide:

      deviation route  sqrt(2) * median_n floor_dev[n] * chord_len
                       (``deviation.floor_med`` is in chord units)
      spacing route    median_n spacing_true[n] * sqrt((1+b_n)^2 - 1)
                       (``spacing_rel_med`` is the relative inflation b_n;
                       spacing_true comes from the float32 T1 rows)
      analytic         sqrt(2) * BF16_REL_RMS * median_n ||Z[n]||
    """
    dev = np.asarray(floors["deviation"]["floor_med"], dtype=np.float64)[1:50]
    dev_route = math.sqrt(2.0) * float(np.median(dev)) * chord_med

    b = np.asarray(floors["spacing_rel_med"], dtype=np.float64)
    per_step = spacing_med * np.sqrt(np.clip((1.0 + b) ** 2 - 1.0, 0.0, None))
    sp_route = float(np.median(per_step))

    analytic = math.sqrt(2.0) * BF16_REL_RMS * float(np.median(magnitude_med))
    return {
        "deviation_route": dev_route,
        "spacing_route": sp_route,
        "spacing_route_per_step": [float(v) for v in per_step],
        "spacing_route_min": float(np.min(per_step)),
        "spacing_route_max": float(np.max(per_step)),
        "analytic": analytic,
        "max_route": max(dev_route, sp_route),
        "chord_len_median": chord_med,
        "note": ("all three are the bf16 instrument floor on a difference of two "
                 "stored states; the separation state is reported under each"),
    }


# ---------------------------------------------------------------------------
# T1 reference rows (streamed; only the block-A streams are kept)
# ---------------------------------------------------------------------------


# exactly the fields read downstream: the three denominators (`chord_len`,
# `spacing`, `magnitude`), the pairing key, the dimension, and the two fields
# that say WHICH generation and which dtype the denominators came from
T1_KEEP = ("chord_len", "z_T_sha256", "d", "spacing", "magnitude",
           "source_dir", "path_dtype")


def stream_t1_rows(t1_path: Path, streams: dict[str, tuple[str, int]],
                   wanted_idx: dict[str, set[int]]) -> dict[tuple[str, int], dict[str, Any]]:
    """One pass over ``t1_merged.jsonl``; keep the reference rows of the block-A
    streams at the prompt indices that actually have a stored path.

    Retained payload per row = 50 + 51 floats + a few scalars; at most
    2 streams x 120 indices x 2 candidate source dirs, i.e. a few MB. The
    124,983 cell rows and the other 4,389 reference rows are discarded in the
    reader and never materialised.

    When both a ``references_t3/`` row (the run that also stored the path) and
    the older ``references/`` row exist for the same (stream, idx), the
    ``references_t3/`` one wins — it is the same generation as the bf16 path
    whose norms it normalises. Which row won is reported per key, because the
    denominators are float32 T1 quantities dividing a bf16 T3 quantity and the
    plan (section 2.2 point 3) requires the mix to be stated, not assumed.
    """
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    want = {(ds, seed): name for name, (ds, seed) in streams.items()}
    n_seen = 0
    with open(t1_path, encoding="utf-8") as fh:
        for line in fh:
            n_seen += 1
            # cheap pre-filter: cell rows never carry mode "original"
            if '"mode": "original"' not in line:
                continue
            row = json.loads(line)
            if row.get("mode") != "original" or row.get("kind", "reference") != "reference":
                continue
            key = (row.get("dataset"), row.get("base_seed"))
            name = want.get(key)
            if name is None:
                continue
            idx = row.get("prompt_idx")
            if idx not in wanted_idx.get(name, ()):  # type: ignore[arg-type]
                continue
            slim = {k: row.get(k) for k in T1_KEEP}
            prev = by_key.get((name, idx))
            # prefer the row produced by the same run as the stored path
            if prev is None or ((slim.get("source_dir") or "").startswith("references_t3/")
                                and not (prev.get("source_dir") or "").startswith("references_t3/")):
                by_key[(name, idx)] = slim
    print(f"  t1: scanned {n_seen} rows, retained {len(by_key)} reference rows "
          f"(~{len(by_key) * 110 * 8 / 1e6:.1f} MB of float payload)", flush=True)
    return by_key


# ---------------------------------------------------------------------------
# T3 / T2 loading
# ---------------------------------------------------------------------------


_LAT_RE = re.compile(r"latents_(\d+)\.pt$")


def stream_dirs(data_root: Path, backbone: str, names: Iterable[str]
                ) -> dict[str, Path]:
    root = data_root / backbone / "matrix" / "references_t3"
    out: dict[str, Path] = {}
    for name in names:
        d = root / name
        if not d.is_dir():
            raise SystemExit(f"block-A stream directory missing: {d}")
        out[name] = d
    return out


def parse_stream(name: str) -> tuple[str, int]:
    """``penguin599_s54`` -> ``("penguin599", 54)``."""
    m = re.fullmatch(r"(.+)_s(\d+)", name)
    if m is None:
        raise SystemExit(f"stream {name!r} is not <dataset>_s<base_seed>")
    return m.group(1), int(m.group(2))


def available_paths(dirs: dict[str, Path], limit: int | None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, d in dirs.items():
        idx = sorted(int(_LAT_RE.search(p.name).group(1))  # type: ignore[union-attr]
                     for p in d.glob("latents_*.pt"))
        out[name] = idx[:limit] if limit is not None else idx
    return out


def load_latents(path: Path) -> np.ndarray:
    """Stored bf16 ``[51, ...]`` -> float32 ``[51, d]``. 266 MB (HYV) / 346 MB (Wan)."""
    import torch  # lazy: the only reason this script needs torch
    Z = torch.load(path, map_location="cpu", weights_only=True)
    arr = Z.to(torch.float32).numpy()
    return arr.reshape(arr.shape[0], -1)


def find_frame(stream_dir: Path, idx: int, dataset: str, base_seed: int,
               data_root: Path, backbone: str) -> tuple[Path | None, str]:
    """The T2 frame of this generation: the one written by the same run if it
    is there, else the matrix reference of the same (dataset, base seed).

    The fallback is built from (dataset, base seed), not from the chosen T1
    row's ``source_dir``: that row is preferentially the ``references_t3/`` one,
    i.e. the directory just globbed, so keying the fallback off it would make it
    unreachable and a block-A directory missing its frames would silently fall
    to "missing" instead of the ``references/`` frame, which exists for every
    prompt.
    """
    local = sorted(stream_dir.glob(f"frame_{idx:05d}_*.npy"))
    if local:
        return local[0], "same_run"
    rec = data_root / backbone / "matrix" / "references" / f"{dataset}_s{base_seed}"
    cand = sorted(rec.glob(f"frame_{idx:05d}_*.npy"))
    if cand:
        return cand[0], "matrix_reference"
    return None, "missing"


def load_frame(path: Path) -> np.ndarray:
    """float16 ``[3, d]`` -> re-orthonormalised, SIGN-ALIGNED float64 ``[3, d]``.

    The store is a direction quantisation, so the frame is re-orthonormalised
    on load (plan section 3.8-3); a float64 Gram resolves ~0.03 deg near 0,
    below anything read here.

    ``orthonormalize`` re-orthonormalises with ``np.linalg.qr``, and LAPACK's
    Householder QR fixes the sign of each Q column from the sign of that
    column's leading entry: it returns ``-row`` whenever the stored row's
    leading component is positive, i.e. for about half of all generations.
    Shares and principal angles are sign-invariant and never noticed, but the
    endpoint's chord ANGLE is not — a flipped chord row reads ``180 - theta``.
    Every row is therefore re-aligned to the row it replaced, so row 0 is the
    stored chord direction and not its negative.
    """
    frame = np.load(path)
    if frame.ndim != 2 or frame.shape[0] < 3:
        raise SystemExit(f"{path}: expected [3, d] frame, got {frame.shape}")
    raw = frame[:3].astype(np.float64)
    F = orthonormalize(raw.astype(np.float32)[None])[0].astype(np.float64)
    signs = np.where(np.einsum("ij,ij->i", raw, F) < 0.0, -1.0, 1.0)
    return F * signs[:, None]


class FrameCache:
    """T2 frames, loaded on demand and kept a few at a time.

    All 240 block-A frames at once would be 7.5 GB (HYV) / 9.8 GB (Wan) as
    float64, so they are read per pair instead: the file is 7.8 / 10.2 MB and
    the pair lists are sorted, so the same frame is reused across consecutive
    pairs.
    """

    def __init__(self, paths: dict[tuple[str, int], Path | None], capacity: int = 8) -> None:
        self.paths = paths
        self.capacity = max(2, capacity)
        self._store: dict[tuple[str, int], np.ndarray | None] = {}

    def get(self, key: tuple[str, int]) -> np.ndarray | None:
        if key in self._store:
            hit = self._store.pop(key)
        else:
            p = self.paths.get(key)
            hit = load_frame(p) if p is not None else None
        self._store[key] = hit
        while len(self._store) > self.capacity:
            self._store.pop(next(iter(self._store)))
        return hit


class PathCache:
    """A handful of loaded paths, keyed by (stream, idx), LRU by insertion.

    A path is 266 MB (HYV) / 346 MB (Wan) as float32; the default capacity of
    2 is what a pair needs and nothing more. The pair list is sorted, so a
    capacity of 2 already reuses the ``a`` side across consecutive pairs.
    """

    def __init__(self, dirs: dict[str, Path], capacity: int = 2) -> None:
        self.dirs = dirs
        self.capacity = max(1, capacity)
        self._store: dict[tuple[str, int], np.ndarray] = {}
        self.loads = 0

    def get(self, key: tuple[str, int]) -> np.ndarray:
        hit = self._store.pop(key, None)
        if hit is None:
            name, idx = key
            hit = load_latents(self.dirs[name] / f"latents_{idx:05d}.pt")
            self.loads += 1
        self._store[key] = hit
        while len(self._store) > self.capacity:
            self._store.pop(next(iter(self._store)))
        return hit


# ---------------------------------------------------------------------------
# section 3.10: per-state difference of a pair
# ---------------------------------------------------------------------------


def pair_readings(Za: np.ndarray, Zb: np.ndarray,
                  Fa: np.ndarray | None, Fb: np.ndarray | None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``G[n]`` and the direction shares of ``Z^a[n] - Z^b[n]``.

    Returns ``(G[51], shares_a[51, 3], shares_b[51, 3])`` with the columns
    (along the frame's chord, inside the frame's bend plane, off plane) as
    fractions of the difference's total energy; the three sum to 1 because
    ``[chord, PC1, PC2]`` is orthonormal and the two PCs are chord-orthogonal
    by construction. Both frames are used so the caller can report the
    two-direction average the plan asks for.

    One state at a time in float64: the difference of two bf16-derived float32
    values is exact, so the only rounding here is the reduction, and the
    working set is ~10 MB (HYV) rather than another whole path.
    """
    n_states = Za.shape[0]
    G = np.zeros(n_states, dtype=np.float64)
    shares = [np.full((n_states, 3), np.nan), np.full((n_states, 3), np.nan)]
    for n in range(n_states):
        v = Za[n].astype(np.float64)
        v -= Zb[n]
        g2 = float(v @ v)
        G[n] = math.sqrt(max(g2, 0.0))
        if g2 <= 0.0:
            continue
        for slot, F in enumerate((Fa, Fb)):
            if F is None:
                continue
            c = F @ v
            chord_share = float(c[0] ** 2) / g2
            plane_share = float(c[1] ** 2 + c[2] ** 2) / g2
            off = 1.0 - chord_share - plane_share
            shares[slot][n] = (chord_share, plane_share, max(off, 0.0))
    return G, shares[0], shares[1]


def build_same_noise_pairs(keys: list[tuple[str, int]], sha: dict[tuple[str, int], str]
                           ) -> tuple[list[tuple[tuple[str, int], tuple[str, int]]], dict[str, Any]]:
    """Pair by ``z_T_sha256`` equality, never by seed arithmetic (plan 2.5)."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for k in keys:
        s = sha.get(k)
        if s:
            groups.setdefault(s, []).append(k)
    pairs = []
    sizes: dict[int, int] = {}
    cross = 0
    for members in groups.values():
        sizes[len(members)] = sizes.get(len(members), 0) + 1
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
                cross += int(members[i][0] != members[j][0])
    meta = {
        "n_noise_groups_total": len(groups),
        "n_noise_groups_used": sum(1 for m in groups.values() if len(m) >= 2),
        "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "n_pairs": len(pairs),
        "n_cross_dataset_pairs": cross,
    }
    return sorted(pairs), meta


def build_same_prompt_pairs(keys: list[tuple[str, int]], sha: dict[tuple[str, int], str]
                            ) -> tuple[list[tuple[tuple[str, int], tuple[str, int]]], dict[str, Any]]:
    """Same (dataset, prompt_idx) across seed streams, different noise.

    Two stored paths share a prompt exactly when they sit in two streams of
    the SAME dataset at the SAME index; their z_T must differ (different
    stream base seed), and any accidental sha collision is refused rather
    than paired. With only block A on disk every dataset has one stream and
    this returns no pairs -- the caller reports the class unavailable.
    """
    by_prompt: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for k in keys:
        dataset = k[0].rsplit("_s", 1)[0]
        by_prompt.setdefault((dataset, k[1]), []).append(k)
    pairs = []
    refused_same_sha = 0
    for members in by_prompt.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if sha.get(a) and sha.get(a) == sha.get(b):
                    refused_same_sha += 1
                    continue
                pairs.append((a, b))
    meta = {
        "n_pairs": len(pairs),
        "n_prompts_used": sum(1 for m in by_prompt.values() if len(m) >= 2),
        "n_pairs_refused_same_sha": refused_same_sha,
        "pairing": "same (dataset, prompt_idx) across seed streams",
    }
    return sorted(pairs), meta


def build_both_different_pairs(keys: list[tuple[str, int]], sha: dict[tuple[str, int], str],
                               count: int, seed: int
                               ) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """``count`` pairs that share neither the noise nor the prompt, drawn from
    the block-A paths as random matchings so no path is read more often than
    the others."""
    rng = np.random.default_rng(seed)
    n = len(keys)
    if n < 2:
        return []
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for _ in range(4 * max(1, count) // max(1, n // 2) + 8):
        perm = rng.permutation(n)
        added = 0
        for i in range(0, n - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            if sha.get(keys[a]) == sha.get(keys[b]):
                continue  # same noise: that is the other class
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            added += 1
            if len(out) >= count:
                break
        if len(out) >= count or added == 0:
            break  # a whole matching added nothing: the pool is exhausted
    return sorted((keys[a], keys[b]) for a, b in out)


def run_pairs(pairs, cache: PathCache, frames: FrameCache,
              chord: dict[tuple[str, int], float], dim: int) -> dict[str, Any]:
    """G profiles and shares for one pair class."""
    n_pairs = len(pairs)
    G = np.full((n_pairs, N_STATES), np.nan)
    G_chord = np.full((n_pairs, N_STATES), np.nan)
    shares = np.full((n_pairs, N_STATES, 3), np.nan)
    n_shares = 0
    for i, (ka, kb) in enumerate(pairs):
        Za, Zb = cache.get(ka), cache.get(kb)
        Fa, Fb = frames.get(ka), frames.get(kb)
        g, sa, sb = pair_readings(Za, Zb, Fa, Fb)
        G[i] = g
        mean_chord = 0.5 * (chord[ka] + chord[kb])
        G_chord[i] = g / mean_chord if mean_chord > 0 else np.nan
        if Fa is not None or Fb is not None:
            stack = [s for s, F in ((sa, Fa), (sb, Fb)) if F is not None]
            with warnings.catch_warnings():  # state 0 of a same-noise pair is 0/0
                warnings.simplefilter("ignore", RuntimeWarning)
                shares[i] = np.nanmean(np.stack(stack), axis=0)
            n_shares += 1
        if (i + 1) % 25 == 0 or i + 1 == n_pairs:
            print(f"    pair {i + 1}/{n_pairs} (path loads so far: {cache.loads})",
                  flush=True)
    sqrt_d = math.sqrt(dim)
    out: dict[str, Any] = {
        "n_pairs": n_pairs,
        "n_pairs_with_frame": n_shares,
        "G_abs": _profile_stats(G),
        "G_over_sqrt_d": _profile_stats(G / sqrt_d),
        "G_over_mean_chord": _profile_stats(G_chord),
        "G0": _quantiles(G[:, 0]),
        "G50_over_mean_chord": _quantiles(G_chord[:, 50]),
        "landmarks": {},
    }
    for n in LANDMARKS:
        out["landmarks"][str(n)] = {
            "G_over_sqrt_d": _quantiles(G[:, n] / sqrt_d),
            "G_over_mean_chord": _quantiles(G_chord[:, n]),
            "share_chord": _quantiles(shares[:, n, 0]),
            "share_in_plane": _quantiles(shares[:, n, 1]),
            "share_off_plane": _quantiles(shares[:, n, 2]),
            # the plan's parallel wording in section 3.9.3 is "of the energy
            # left after removing the chord", kept alongside the total shares
            "share_in_plane_of_offchord": _quantiles(
                shares[:, n, 1] / np.clip(1.0 - shares[:, n, 0], 1e-12, None)),
        }
    return out


def separation_state(median_profile: list[float], floor: float) -> int | None:
    """First state n >= 1 whose median G exceeds ``SNR_MIN`` x the bf16 floor."""
    for n in range(1, len(median_profile)):
        v = median_profile[n]
        if math.isfinite(v) and v > SNR_MIN * floor:
            return n
    return None


# ---------------------------------------------------------------------------
# section 3.6 overlay + the direct multi-window turn arrays
# ---------------------------------------------------------------------------


def path_shape(Z32: np.ndarray, windows: Iterable[int]) -> dict[str, Any]:
    """Own-frame coordinates and direct turn arrays of one stored path.

    The frame is recomputed per path with ``trajectory_math.plane_frame``
    (plan section 3.6 (f)); it is *not* compared with the T2 frame, that row
    is section 3.9.3 / P7. Peak memory is ~4x the path as float64 inside
    ``plane_frame`` (2.1 GB HYV / 2.8 GB Wan), so paths are handled one at a
    time here and the pair cache is not used.
    """
    Zf = np.asarray(Z32, dtype=np.float64)
    frame = plane_frame(Zf)
    chord_len = float(np.linalg.norm(Zf[-1] - Zf[0]))
    steps = np.linalg.norm(np.diff(Zf, axis=0), axis=1)
    out: dict[str, Any] = {
        "chord_len_bf16": chord_len,
        "path_len_bf16": float(steps.sum()),
        "straightness_bf16": float(steps.sum() / chord_len) if chord_len > 0 else float("nan"),
        "coords": None,
        "turn": {},
    }
    if frame is not None and chord_len > 0:
        rel = Zf - Zf[0]
        coords = (rel @ frame.T) / chord_len  # [51, 3], chord-normalised
        out["coords"] = [[float(v) for v in row] for row in coords]
    for w in windows:
        labels, note = turn_index(Zf.shape[0], w)
        out["turn"][str(w)] = {
            "centers": labels,
            "index_object": note,
            "deg": [float(v) for v in turn_angles_window_deg(Zf, w)],
        }
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def write_markdown(path: Path, rep: dict[str, Any]) -> None:
    T = rep["backbone"]
    fl = rep["g_floor"]
    L: list[str] = [
        f"# P4 T3 layer — {T} (plan sections 3.10, 3.6 overlay, 3.8-2 turn arrays)", "",
        f"Block A of T3: {rep['n_paths_total']} stored paths over "
        f"{len(rep['streams'])} streams "
        + ", ".join(f"`{k}` ({v['n_paths']})" for k, v in rep["streams"].items())
        + f"; d = {rep['d']}, sqrt(d) = {math.sqrt(rep['d']):.1f}.", "",
        "**Floor.** Every magnitude below is a bf16 reading (the T3 store). "
        f"P1 bf16 rows: chord {_fmt(rep['p1']['chord_rel_med'], 3)}, "
        f"path_len {_fmt(rep['p1']['path_len_rel_med'], 3)}, "
        f"straightness {_fmt(rep['p1']['straightness_rel_med'], 3)}, "
        f"spacing bias {_fmt(rep['p1']['spacing_rel_med'][0], 3)} at step 0 decaying to "
        f"{_fmt(rep['p1']['spacing_rel_med'][49], 3)} at step 49. "
        "The direction frame is the stored T2 frame, computed in flight on the "
        "float32 path, so the three shares are a float32-row reading of direction.", "",
        "**Denominators.** `G ÷ mean chord` and the G floor divide a bf16-derived "
        "quantity by float32 T1 quantities (`chord_len`, `spacing`, `magnitude`) of the "
        f"same generations: {rep['t1_denominator_provenance']['rows_by_source_dir']}, "
        f"path dtype {rep['t1_denominator_provenance']['path_dtypes']}, read from "
        f"`{rep['t1_merged']}`. "
        f"{rep['t1_denominator_provenance']['note']}", "",
        "## 1. bf16 floor on G (a difference of two stored states)", "",
        "| route | floor (abs) | ÷ sqrt(d) | ÷ median chord |",
        "|---|---:|---:|---:|",
    ]
    sqrt_d = math.sqrt(rep["d"])
    for name in ("deviation_route", "spacing_route", "analytic"):
        v = fl[name]
        L.append(f"| {name} | {_fmt(v)} | {_fmt(v / sqrt_d)} | "
                 f"{_fmt(v / fl['chord_len_median'])} |")
    L += ["",
          f"Spacing route per step spans {_fmt(fl['spacing_route_min'])}–"
          f"{_fmt(fl['spacing_route_max'])} (median {_fmt(fl['spacing_route'])}); the model "
          "behind it is measured² = true² + floor², so a flat span is the model holding. "
          f"Separation state below uses **{rep['g_floor_source']}** "
          f"({_fmt(rep['g_floor_used'])}) and the rule median G[n] > {SNR_MIN:g} x floor; "
          "the state under each of the three routes is in the next table.", ""]

    L += ["## 2. G[n] = ||Z^a[n] − Z^b[n]|| by pair class", "",
          "| class | pairs | shared-noise groups | distinct z_T among the paths used | "
          "G[0] median | separation state (dev / spacing / max route) | "
          "G[1] ÷√d | G[10] ÷√d | G[25] ÷√d | G[40] ÷√d | G[50] ÷√d | G[50] ÷ mean chord |",
          "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, cls in rep["classes"].items():
        if cls.get("unavailable"):
            L.append(f"| {name} | — | — | — | — | not available — {cls['unavailable']} | "
                     + " | ".join("—" for _ in range(6)) + " |")
            continue
        sep = cls["separation_state"]
        lm = cls["landmarks"]
        groups = cls.get("n_noise_groups_used")
        L.append(
            f"| {name} | {cls['n_pairs']} | {groups if groups is not None else '—'} | "
            f"{cls.get('n_distinct_z_T', '—')} | "
            f"{_fmt(cls['G0']['median'])} | "
            f"{sep['deviation']} / {sep['spacing']} / {sep['max']} | "
            + " | ".join(_fmt(lm[str(n)]["G_over_sqrt_d"]["median"]) for n in LANDMARKS)
            + f" | {_fmt(cls['G50_over_mean_chord']['median'])} |")
    L += ["",
          "The two count columns are different things: **shared-noise groups** is the "
          "plan's caliber (read per noise, not per pair) and exists only where a class is "
          "built out of z_T groups; **distinct z_T** merely counts how many noise tensors "
          "the paths in that class span. IQR of every entry is in the json; `G[0]` is "
          "reported as measured, not asserted to be 0 (the two paths share z_T by "
          "`z_T_sha256`, but the store is bf16).", ""]

    L += ["## 3. Direction of the difference (shares of the total energy, "
          "two-direction average)", "",
          "| class | state | along chord | in bend plane | off plane | in-plane share of the off-chord energy | pairs with ≥1 frame |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, cls in rep["classes"].items():
        if cls.get("unavailable"):
            continue
        for n in LANDMARKS:
            e = cls["landmarks"][str(n)]
            L.append(f"| {name} | {n} | {_fmt(e['share_chord']['median'])} | "
                     f"{_fmt(e['share_in_plane']['median'])} | "
                     f"{_fmt(e['share_off_plane']['median'])} | "
                     f"{_fmt(e['share_in_plane_of_offchord']['median'])} | "
                     f"{cls['n_pairs_with_frame']} |")
    L += ["", f"Frames used: {rep['frame_sources']}. The three shares sum to 1 **per pair** "
          "by construction ([chord, PC1, PC2] is orthonormal and the two PCs are "
          "chord-orthogonal); the columns above are per-share medians over pairs, which "
          "need not sum to 1. The last column is the same in-plane energy as a fraction "
          "of what is left after the chord is removed — the wording section 3.9.3 uses.", ""]

    ov = rep["overlay"]
    L += ["## 4. Overlay paths and direct turn arrays (bf16)", "",
          f"{ov['n_paths']} paths ({', '.join(f'{k}: {v}' for k, v in ov['per_stream'].items())}), "
          "each rotated into its own [chord, PC1, PC2] frame and divided by its own chord. "
          "Lengths recomputed here carry the P1 bf16 inflation quoted above, so the overlay "
          "is a shape picture, not a length measurement.", "",
          "| stream | paths | chord (bf16) median | straightness (bf16) median |",
          "|---|---:|---:|---:|"]
    for name, s in ov["streams"].items():
        L.append(f"| {name} | {s['n']} | {_fmt(s['chord_len_bf16']['median'])} | "
                 f"{_fmt(s['straightness_bf16']['median'], 6)} |")
    L += ["", "| window w | centres | centres with bf16 SNR ≥ 3 (P1) | readable on bf16 | "
          "median turn at first centre | median turn at the middle centre |",
          "|---:|---:|---:|---|---:|---:|"]
    for w in rep["windows"]:
        p1w = rep["p1_windows"].get(str(w), {})
        t = ov["turn_summary"].get(str(w), {})
        L.append(f"| {w} | {p1w.get('centres', '—')} | "
                 f"{p1w.get('centres_ge_snr_min', '—')}/{p1w.get('centres', '—')} | "
                 f"{'yes' if p1w.get('readable') else 'no'} | "
                 f"{_fmt(t.get('first_centre_median'))} | {_fmt(t.get('mid_centre_median'))} |")
    mw = rep["p1_windows"].get("min_readable_multistep_window_bf16")
    L += ["", f"P1 minimum readable multi-step window on bf16: **w = {mw if mw else '> 11'}**; "
          "on float32 "
          "rows (which is what the T1 profiles are) P1 read w = 5 and w = 1 everywhere, so "
          "the plan's contingency 'if a backbone needs w ≥ 9, read the flattest centre off T3' "
          "does not trigger for the T1 profiles. The arrays here are stored for P5 to "
          "correlate against; no correlation is computed in P4.", ""]
    atomic_write_text(path, "\n".join(L) + "\n")


def write_figures(fig_dir: Path, rep: dict[str, Any], force: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

    fig_dir.mkdir(parents=True, exist_ok=True)
    T = rep["backbone"]
    sqrt_d = math.sqrt(rep["d"])

    # --- G[n] profiles ----------------------------------------------------
    out = fig_dir / f"g_profile_{T}.png"
    if force or not output_complete(out):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        colors = {"same_noise_diff_prompt": "C0", "both_different": "C3",
                  "same_prompt_diff_noise": "C2"}
        states = np.arange(N_STATES)
        for name, cls in rep["classes"].items():
            if cls.get("unavailable") or not cls.get("n_pairs"):
                continue
            c = colors.get(name, "k")
            for ax, key, in ((axes[0], "G_over_sqrt_d"), (axes[1], "G_over_mean_chord")):
                med = np.asarray(cls[key]["median"], dtype=float)
                lo = np.asarray(cls[key]["p25"], dtype=float)
                hi = np.asarray(cls[key]["p75"], dtype=float)
                ax.plot(states, med, color=c, lw=1.6,
                        label=f"{name} (n={cls['n_pairs']})")
                ax.fill_between(states, lo, hi, color=c, alpha=0.18, lw=0)
        floor = float(rep["g_floor_used"])
        axes[0].axhline(floor / sqrt_d, color="0.4", ls=":", lw=1.0,
                        label="bf16 floor")
        axes[0].axhline(SNR_MIN * floor / sqrt_d, color="0.4", ls="--", lw=1.0,
                        label=f"{SNR_MIN:g}x floor (separation rule)")
        axes[0].set_yscale("log")
        axes[0].set_ylabel(r"$\|Z^a[n]-Z^b[n]\|\ /\ \sqrt{d}$  (log)")
        axes[1].set_ylabel(r"$\|Z^a[n]-Z^b[n]\|\ /\ \mathrm{mean(chord)}$")
        for ax in axes:
            ax.set_xlabel("state n (0 = z_T, 50 = result)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{T} — per-state difference between two whole paths, by what the "
                     f"pair shares (bf16 T3 store; band = IQR over pairs)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        atomic_savefig(fig, out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")

    # --- 3-D overlay ------------------------------------------------------
    out = fig_dir / f"{T}_traj_3d_overlay.png"
    if force or not output_complete(out):
        fig = plt.figure(figsize=(7.5, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        colors = ["C0", "C1", "C2", "C3"]
        for ci, (name, entries) in enumerate(rep["overlay"]["paths"].items()):
            first = True
            for coords in entries:
                if coords is None:
                    continue
                a = np.asarray(coords, dtype=float)
                ax.plot(a[:, 0], a[:, 1], a[:, 2], color=colors[ci % len(colors)],
                        lw=0.7, alpha=0.65,
                        label=f"{name} (n={len(entries)})" if first else None)
                first = False
        ax.set_xlabel("chord"); ax.set_ylabel("PC1"); ax.set_zlabel("PC2")
        ax.set_title(f"{T} — every path in its own [chord, PC1, PC2] frame,\n"
                     "divided by its own chord (bf16 store: shape, not length)",
                     fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        atomic_savefig(fig, out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_refs(args: argparse.Namespace) -> None:
    tables = Path(args.out_tables)
    figs = Path(args.out_figs)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    json_path = tables / f"latent_paths_{args.backbone}.json"
    md_path = tables / f"latent_paths_{args.backbone}.md"

    # An existing JSON only stands for this run when the parameters that decide
    # the numbers agree; a `--limit 5 --sample 20` smoke run must not satisfy
    # the production run, and a truncated file must fall through to a recompute
    # rather than abort.
    run_params = {"limit": args.limit, "sample": args.sample,
                  "overlay_paths": args.overlay_paths, "streams": list(args.streams),
                  "windows": list(args.windows), "seed": args.seed,
                  "g_floor_source": args.g_floor_source}
    rep = resolve_reuse(json_path, run_params, force=args.force,
                        caps=("limit", "sample", "overlay_paths"))
    if rep is not None:
        print(f"{json_path} exists and matches these parameters — reusing it "
              f"(pass --force to recompute)")
        if not output_complete(md_path):
            write_markdown(md_path, rep)
            print(f"  wrote {md_path}")
        write_figures(figs, rep, False)
        return

    if args.workers:
        try:
            import torch
            torch.set_num_threads(args.workers)
        except Exception:  # torch is only needed to open .pt files
            pass

    data_root = Path(args.data_root)
    dirs = stream_dirs(data_root, args.backbone, args.streams)
    idx_by_stream = available_paths(dirs, args.limit)
    streams_meta = {name: parse_stream(name) for name in dirs}
    keys = [(name, i) for name in dirs for i in idx_by_stream[name]]
    if not keys:
        raise SystemExit(f"no latents_*.pt under {list(dirs.values())}")

    t1_path = Path(args.t1_merged) if args.t1_merged else (
        data_root / args.backbone / "matrix" / "trajectory" / "t1_merged.jsonl")
    rows = stream_t1_rows(t1_path, streams_meta,
                          {k: set(v) for k, v in idx_by_stream.items()})
    missing = [k for k in keys if k not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} stored paths have no T1 reference row "
                         f"(first: {missing[:3]})")

    dim = int(rows[keys[0]]["d"])
    sha = {k: rows[k]["z_T_sha256"] for k in keys}
    chord = {k: float(rows[k]["chord_len"]) for k in keys}
    # which generation and which dtype every denominator came from (plan
    # section 2.2 point 3: a reading that mixes tiers has to say so)
    src_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    for k in keys:
        src = str(rows[k].get("source_dir") or "?").split("/")[0]
        src_counts[src] = src_counts.get(src, 0) + 1
        dt = str(rows[k].get("path_dtype") or "?")
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    provenance = {
        "rows_by_source_dir": src_counts,
        "path_dtypes": dtype_counts,
        "preference": "the references_t3/ row (same run as the stored path) when both "
                      "exist, else the references/ row of the same (dataset, base seed)",
        "note": ("These are float32 in-flight quantities; G itself is bf16, so every "
                 "ratio below mixes a bf16 numerator with a float32 denominator — the "
                 "bf16 floor is the binding one."
                 if set(dtype_counts) <= {"float32"} else
                 f"Mixed path dtypes among the T1 rows: {dtype_counts}."),
    }
    resident = args.cache_paths * N_STATES * dim * 4 / 1e9
    frame_budget = args.cache_frames * 3 * dim * 8 / 1e9
    print(f"backbone {args.backbone}: {len(keys)} block-A paths, d = {dim}; "
          f"resident path budget {args.cache_paths} x {N_STATES} x {dim} x 4 B = "
          f"{resident:.2f} GB, frame budget {args.cache_frames} x 3 x {dim} x 8 B = "
          f"{frame_budget:.2f} GB", flush=True)

    # T2 frames: located now, loaded on demand (see FrameCache)
    frame_paths: dict[tuple[str, int], Path | None] = {}
    frame_sources: dict[str, int] = {}
    for k in keys:
        dataset, base_seed = streams_meta[k[0]]
        p, src = find_frame(dirs[k[0]], k[1], dataset, base_seed, data_root, args.backbone)
        frame_sources[src] = frame_sources.get(src, 0) + 1
        frame_paths[k] = p
    frames = FrameCache(frame_paths, args.cache_frames)
    print(f"  T2 frames: {frame_sources}", flush=True)

    floor_dir = Path(args.p1_floor_dir) if args.p1_floor_dir else (
        _DEFAULT_FLOOR_DIR / args.backbone)
    floors = load_p1_floors(floor_dir,
                            Path(args.p1_floor_json) if args.p1_floor_json else None)
    spacing_med = np.median(np.asarray([rows[k]["spacing"] for k in keys],
                                       dtype=np.float64), axis=0)
    magnitude_med = np.median(np.asarray([rows[k]["magnitude"] for k in keys],
                                         dtype=np.float64), axis=0)
    chord_med = float(np.median([chord[k] for k in keys]))
    gfloor = g_floor_estimates(floors, chord_med, spacing_med, magnitude_med)
    floor_used = (gfloor["max_route"] if args.g_floor_source == "max"
                  else gfloor[f"{args.g_floor_source}_route"])

    cache = PathCache(dirs, args.cache_paths)
    classes: dict[str, Any] = {}

    same_pairs, same_meta = build_same_noise_pairs(keys, sha)
    print(f"same-noise pairs: {same_meta['n_pairs']} over "
          f"{same_meta['n_noise_groups_used']} noises "
          f"({same_meta['n_cross_dataset_pairs']} cross-dataset)", flush=True)
    # `n_noise_groups_used` (the plan's "read per noise, not per pair" caliber)
    # and `n_distinct_z_T` (how many different noise tensors the sampled
    # participants happen to span) are different quantities and are never
    # printed in one column: only the same-noise class has a noise caliber.
    classes["same_noise_diff_prompt"] = {
        **run_pairs(same_pairs, cache, frames, chord, dim), **same_meta,
        "n_distinct_z_T": len({sha[k] for p in same_pairs for k in p}),
        "caliber": "read per noise, not per pair",
    }

    both_pairs = build_both_different_pairs(keys, sha, args.sample, args.seed)
    print(f"both-different pairs: {len(both_pairs)} sampled from {len(keys)} paths "
          f"(seed {args.seed})", flush=True)
    classes["both_different"] = {
        **run_pairs(both_pairs, cache, frames, chord, dim),
        "n_noise_groups_used": None,   # no noise is shared inside this class
        "n_distinct_z_T": len({sha[k] for p in both_pairs for k in p}),
        "caliber": "read per pair; every pair has two different z_T, so there is no "
                   "noise-count caliber here",
        "sampling": f"random matchings over the {len(keys)} block-A paths, "
                    f"seed {args.seed}, same-z_T pairs excluded",
    }
    prompt_pairs, prompt_meta = build_same_prompt_pairs(keys, sha)
    if prompt_pairs:
        print(f"same-prompt pairs: {prompt_meta['n_pairs']} over "
              f"{prompt_meta['n_prompts_used']} prompts", flush=True)
        classes["same_prompt_diff_noise"] = {
            **run_pairs(prompt_pairs, cache, frames, chord, dim), **prompt_meta,
            "n_noise_groups_used": None,   # no noise is shared inside this class
            "n_distinct_z_T": len({sha[k] for p in prompt_pairs for k in p}),
            "caliber": "read per prompt, not per pair",
        }
    else:
        classes["same_prompt_diff_noise"] = {
            "unavailable": ("every dataset has a single stream on disk, so no two "
                            "stored paths share a prompt; the control needs the "
                            "block-C streams (P8)"),
        }

    for name, cls in classes.items():
        if cls.get("unavailable"):
            continue
        med = cls["G_abs"]["median"]
        cls["separation_state"] = {
            "deviation": separation_state(med, gfloor["deviation_route"]),
            "spacing": separation_state(med, gfloor["spacing_route"]),
            "max": separation_state(med, gfloor["max_route"]),
            "rule": f"first state n>=1 with median G[n] > {SNR_MIN:g} x the bf16 floor",
        }

    # --- overlay + direct turn arrays -------------------------------------
    p1_windows = readable_windows(floors)
    overlay: dict[str, Any] = {"paths": {}, "streams": {}, "per_stream": {},
                               "turn_by_path": {}, "turn_summary": {}, "n_paths": 0}
    turn_stack: dict[str, list[list[float]]] = {}
    for name in dirs:
        take = idx_by_stream[name][:args.overlay_paths]
        coords_list: list[Any] = []
        chords, straight = [], []
        per_path: dict[str, Any] = {}
        for i in take:
            Z = load_latents(dirs[name] / f"latents_{i:05d}.pt")
            shape = path_shape(Z, args.windows)
            del Z
            coords_list.append(shape["coords"])
            chords.append(shape["chord_len_bf16"])
            straight.append(shape["straightness_bf16"])
            # kept per path as well as pooled: section 3.7 (P5) may want to
            # correlate rho_2 against the direct curvature path by path
            per_path[str(i)] = {w: e["deg"] for w, e in shape["turn"].items()}
            for w, entry in shape["turn"].items():
                turn_stack.setdefault(w, []).append(entry["deg"])
            print(f"    overlay {name} idx {i}", flush=True)
        overlay["paths"][name] = coords_list
        overlay["turn_by_path"][name] = per_path
        overlay["per_stream"][name] = len(take)
        overlay["streams"][name] = {
            "n": len(take), "idx": take,
            "chord_len_bf16": _quantiles(chords),
            "straightness_bf16": _quantiles(straight),
        }
        overlay["n_paths"] += len(take)
    for w, stack in turn_stack.items():
        arr = np.asarray(stack, dtype=np.float64)
        centres, index_note = turn_index(N_STATES, int(w))
        mid = len(centres) // 2
        entry = p1_windows.get(w)
        overlay["turn_summary"][w] = {
            "centers": centres, "n_paths": int(arr.shape[0]),
            "median_profile": [float(v) for v in np.nanmedian(arr, axis=0)],
            "first_centre_median": float(np.nanmedian(arr[:, 0])),
            "mid_centre_median": float(np.nanmedian(arr[:, mid])),
            "floor_note": ("bf16 reading; P1 says this window is "
                           + ("readable" if entry and entry.get("readable")
                              else "below the bf16 floor" if entry
                              else "not measured by P1") + " on the T3 store"),
            # the same convention shape_scale.py stores, via common.turn_index
            "index_note": index_note,
        }

    rep: dict[str, Any] = {
        "backbone": args.backbone,
        "plan_sections": ["3.10", "3.6 overlay", "3.8-2 turn arrays (stored, not read)"],
        "data_root": str(data_root),
        "streams": {name: {"dataset": streams_meta[name][0],
                           "base_seed": streams_meta[name][1],
                           "dir": str(dirs[name]),
                           "n_paths": len(idx_by_stream[name])} for name in dirs},
        "n_paths_total": len(keys),
        "d": dim,
        "t1_merged": str(t1_path),
        "t1_denominator_provenance": provenance,
        "frame_sources": frame_sources,
        "p1": {k: floors[k] for k in ("chord_rel_med", "path_len_rel_med",
                                      "straightness_rel_med", "spacing_rel_med")},
        "p1_windows": p1_windows,
        "p1_source": str(Path(args.p1_floor_json) if args.p1_floor_json
                         else floor_dir / "p1_floor_bfloat16.json"),
        "g_floor": gfloor,
        "g_floor_source": args.g_floor_source,
        "g_floor_used": floor_used,
        "windows": list(args.windows),
        "sample": args.sample,
        "seed": args.seed,
        "limit": args.limit,
        "run_params": run_params,
        "classes": classes,
        "overlay": overlay,
        "path_loads": cache.loads,
        "floor_statement": (
            "T3 magnitudes are bf16 readings; the T2 direction frames were computed "
            "in flight on the float32 path, so the shares are float32-row readings of "
            "direction. No reproduction floor and no T2-vs-T3 plane comparison here: "
            "both belong to section 3.9.3 (P7)."),
    }
    atomic_write_json(json_path, rep)
    print(f"wrote {json_path} ({json_path.stat().st_size / 1e6:.1f} MB); "
          f"{cache.loads} path loads")
    write_markdown(md_path, rep)
    print(f"wrote {md_path}")
    write_figures(figs, rep, True)


# ===========================================================================
# section 3.9.3 (P7): the T3 layer of the cache bend
# ===========================================================================
#
# One pair = (cells_t3_rand50 directory, prompt idx): the cached run's whole
# path Z^c against the same-prompt reference path Z^r of `references_t3/`.
# 27 cells (9 methods x 3 budgets) x 2 datasets x 3 seed streams = 162
# directories per backbone, holding between them the 50 (dataset, seed stream,
# prompt idx) the frozen sample table drew — 25 per dataset out of a pool of
# 360 (3 streams x idx 0-119) — so a (method, budget) row pools 25 + 25 = 50
# pairs and a backbone carries 27 x 50 = 1,350.
#
# Three stages: `--stage cells` writes ONE json per directory, `--stage merge`
# concatenates them, joins the three streams of a (method, budget, dataset)
# into that dataset's 25 pairs, and takes the medians over the pooled 50.
# `--stage refside` is the reference-side block (reproducibility floor,
# T2-vs-T3 plane check, bf16 lower bound); it needs `references/` AND
# `references_t3/`, and is a property of the references rather than of the
# cached sample.

CACHED_SCHEMA = "video_full_trajectory.cached_bend.v1"
CACHED_LANDMARKS: tuple[int, ...] = (25, 40, 50)   # plus each pair's OWN k0+1
CELLS_T3_DIRNAME = "cells_t3_rand50"   # the sampled path layer
DEFAULT_SAMPLE_TABLE = (_PROJECT_ROOT / "resources" / "video_full_trajectory"
                        / "t3_extension_samples.v1.json")
REFSIDE_IDX_LIMIT = 10         # the reproduction floor reads idx 0-9 per stream
PAIRS_PER_GROUP = 50           # sample size: 25 prompts x 2 datasets
POOLED = "__pooled__"          # the dataset slot of a (method, budget) row
EXPECTED_CELLS = len(METHODS) * len(KS) * 2        # 27 x 2 datasets = 54 groups
IN_PLANE_IQR_TRIGGER = 0.3     # plan 4.2 escalation trigger — REPORTED, never acted on

_LAT_GLOB = "latents_*.pt"
# the per-generation T1 fields P7 reads; the file is the same record
# `merge_video_traj.py:145-162` turns into a t1_merged.jsonl row, read here
# directly because it sits next to the path it describes and is therefore
# available on both filesystems
CACHED_T1_KEEP = ("chord_len", "magnitude", "spacing", "d", "z_T_sha256",
                  "prompt_idx", "seed", "path_dtype", "num_steps", "mode", "budget")

CAV_FRAME = (
    "The direction frame is the reference's stored **T2** frame: computed in flight on the "
    "float32 path and stored as float16 `[3, d]` (plan section 2.1, 3.9.3). The three "
    "shares are therefore a DIRECTION read from a float32 computation kept at float16, "
    "while the magnitude they weight is bf16. Recomputing the frame from the bf16 T3 store "
    "would put a bf16 direction on both sides and add nothing; how much the store moves a "
    "plane's orientation is the T2-vs-T3 principal-angle row, which is the caveat attached "
    "to the share table — and because the T2 file is itself float16, that row is an UPPER "
    "bound on the bf16 effect, not a pure one (bf16 is ~8x coarser than float16, so bf16 "
    "still dominates it).")
CAV_PREFIX = (
    "`D[n]` for n <= k0 has expected value EXACTLY 0, not a rounding-sized number: the two "
    "runs hold bit-identical float32 states there and identical float32 values round to "
    "identical bf16. A non-zero prefix is a per-generation run-to-run anomaly — the same "
    "object P6 found in the T1 prefixes — and is listed by name, never used as a floor. A "
    "cache at step k0 first alters state k0+1, so the identity window is states 0..k0 "
    "inclusive (P6's `prefix_length('magnitude', k0)` convention).")
CAV_ENDPOINT_ASYMMETRY = (
    "The endpoint plane angle is asymmetric: the cached plane is recomputed from the bf16 "
    "T3 store, the reference plane is the float32 T2 frame. The T2-vs-T3 row measures that "
    "asymmetry on the reference side and is quoted next to this column. The cached side "
    "has no float32 alternative — no cached T2 frame was generated (plan section 4.2) — and "
    "that is stated rather than worked around.")
CAV_DELTA_NORM = (
    "Delta-D is reported raw and divided by `chord_r`. `Delta D / D[k]` is NOT reported: "
    "`D[k]` is exactly 0 at and before k0 and tiny just after it, so the relative form is "
    "unbounded. The plan does not specify a normalisation for the increments; chord_r is "
    "the safe reading because it is the normalisation section 3.9.3 already uses for D.")
CAV_NEG_OFFSET = (
    "Samples whose solver step lies strictly before k0 are structurally exactly 0 — the "
    "two runs have not diverged yet, so `D[k] = D[k+1] = 0` — and are EXCLUDED from every "
    "median / IQR / mean in these tables; only their count is kept, in "
    "`n_samples_before_k0`. Averaging them in would read as 'the step regressed to 0'. A "
    "bucket whose samples are all before k0 still appears, with `n_samples = 0` and its "
    "count, so the reader sees that it was measured and found structurally empty.")
CAV_FLOOR = (
    "Two floors, in this order. (1) The run-to-run reproduction floor: the same reference "
    "prompt generated twice on the same machine — the `references/` matrix wave, which "
    "holds every prompt of the dataset, against the `references_t3/` re-run, which holds "
    "idx 0-119; the floor is read on the first 10 indices of each stream, and it is a "
    "property of the references, not of the sampled cached pairs. If the two runs come out "
    "bit-identical the floor is "
    "identically 0 at every state and the statement is 'reproduction is bit-exact on this "
    "machine, so the only floor left is the bf16 store' — NOT 'the floor is unmeasurable'. "
    "(2) The bf16 store's rounding bound from the P1 dump, which applies only AFTER k0; "
    "before k0 the two runs hold identical values, which round identically, so the bound "
    "there is 0. Where (1) is non-zero it is the binding floor and (2) is subordinate.")
CAV_CHORD_DIRECTION = (
    "Two reference chord directions are reported for the endpoint angle: row 0 of the T2 "
    "frame (the same frame the decomposition uses — primary) and the T3 recompute "
    "`Z^r[50] - Z^r[0]` (bf16 — secondary). What the store does to a chord DIRECTION is "
    "the angle BETWEEN those two directions (`ref_chord_t2_vs_t3_deg`), not the difference "
    "of their two angles to the cached chord — that difference is reported too "
    "(`chord_angle_t2_minus_t3_deg`) but it vanishes whenever the two reference directions "
    "happen to sit at equal angles from the cached chord, so it can read 0 for a real "
    "direction shift. The cached chord has no such choice: it is bf16 on both readings.")
CAV_TURN_WINDOW = (
    "No turn or curvature reading enters P7 — every quantity here is a difference of two "
    "states or a subspace angle — so P1's minimum readable multi-step turn window in bf16 "
    "binds nothing here. Its value for this backbone is read out of the P1 dump into the "
    "refside block (`p1_turn_window`) and printed there, never hardcoded.")

CACHED_CAVEATS: tuple[str, ...] = (
    CAV_FRAME, CAV_PREFIX, CAV_CHORD_DIRECTION, CAV_ENDPOINT_ASYMMETRY,
    CAV_DELTA_NORM, CAV_NEG_OFFSET, CAV_FLOOR, CAV_TURN_WINDOW,
    CAV_OVERLAP, CAV_POOLING,
)

CACHED_COMPLETION = (
    ("D[n] figure", "cached_D_profile_<T>.png"),
    ("decomposition share table, frame taken from T2", "cached_shares_<T>.tsv"),
    ("endpoint principal-angle table", "cached_endpoint_<T>.tsv"),
    ("reproducibility-floor row", "cached_floor_<T>.tsv"),
    ("bf16 lower bound", "cached_bend_<T>.json -> bf16_bound"),
    ("T2-vs-T3 plane check row", "cached_plane_check_<T>.tsv"),
    (f"{PAIRS_PER_GROUP} pairs per (method, budget), denominator stated per row",
     "every table's n_pairs / n_pairs_dropped columns"),
)


# --------------------------------------------------------------------------
# small helpers shared by the three stages
# --------------------------------------------------------------------------


def _to_series(values: Iterable[float]) -> list[float | None]:
    """A JSON-safe array: non-finite becomes null (`json.dumps` would otherwise
    write a bare NaN, which strict parsers reject)."""
    out: list[float | None] = []
    for v in np.asarray(list(values), dtype=np.float64):
        out.append(float(v) if math.isfinite(v) else None)
    return out


def _from_series(values: Iterable[Any]) -> np.ndarray:
    """The inverse of `_to_series`, used when the merge reads a per-cell json."""
    return np.asarray([np.nan if v is None else float(v) for v in values],
                      dtype=np.float64)


def _tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return str(value)


def _write_tsv(path: Path, columns: Iterable[str], rows: list[dict[str, Any]]) -> Path:
    columns = list(columns)
    lines = ["\t".join(columns)]
    lines += ["\t".join(_tsv_value(r.get(c)) for c in columns) for r in rows]
    return atomic_write_text(path, "\n".join(lines) + "\n")


def angle_between_deg(u: Any, v: Any) -> float:
    """Angle between two vectors in degrees; NaN when either degenerates."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu <= 0.0 or nv <= 0.0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1.0, 1.0))))


def plane_rows(Z: Any) -> np.ndarray | None:
    """PC1/PC2 of a path's own `[chord, PC1, PC2]` frame — rows 1 and 2, never
    the 3-row frame (the chord row is not part of the bend plane).

    Peak memory is ~3x the path as float64 inside `plane_frame`, so the caller
    must not hold more than the pair it is working on.
    """
    F = plane_frame(np.asarray(Z, dtype=np.float64))
    return None if F is None else np.asarray(F[1:3], dtype=np.float64)


def first_principal_angle(A: Any, B: Any) -> tuple[float, float]:
    """`(theta_1, theta_2)` in degrees between two 2-D subspaces, or NaNs."""
    if A is None or B is None:
        return float("nan"), float("nan")
    angles = principal_angles_deg(np.asarray(A, dtype=np.float64),
                                  np.asarray(B, dtype=np.float64))
    while len(angles) < 2:
        angles.append(float("nan"))
    return float(angles[0]), float(angles[1])


def load_traj_row(path: Path) -> dict[str, Any]:
    """The per-generation T1 record sitting next to a stored path."""
    if not path.is_file():
        raise SystemExit(f"per-generation T1 record missing: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {k: rec.get(k) for k in CACHED_T1_KEEP}


def load_actions(path: Path) -> tuple[str, int]:
    """The 50-character action string of ONE generation, from that run's own
    `decisions_%05d.json` — the same rule the merger uses
    (`merge_video_traj.py:134-140`), applied to the cached run's own file and
    never to the matrix cell's (plan section 4.3)."""
    if not path.is_file():
        raise SystemExit(f"decisions file missing: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    bits = "".join("1" if r.get("action") == "cache" else "0"
                   for r in rec.get("records", []))
    if len(bits) != NUM_STEPS:
        raise SystemExit(f"{path}: {len(bits)} decisions, expected {NUM_STEPS}")
    return bits, bits.count("1")


def first_cache_step(actions: str) -> int | None:
    """k0 = the first cached solver step, or None when the row caches nothing."""
    i = actions.find("1")
    return None if i < 0 else i


def cached_cell_meta(name: str) -> dict[str, Any]:
    """`<mode>_<dataset>_K<K>_s<base>` -> canonical method + budget + stream key."""
    try:
        meta = parse_dir(name, "cell")
    except ValueError as exc:
        raise SystemExit(f"{CELLS_T3_DIRNAME} directory {name!r}: {exc}") from exc
    return {"cell_dir": name, "mode_raw": meta["mode_raw"], "method": meta["mode"],
            "dataset": meta["dataset"], "budget": meta["budget"],
            "K_nominal": BUDGET_K[meta["budget"]], "base_seed": meta["base_seed"]}


def load_sample_table(path: Path, backbone: str) -> dict[str, Any]:
    """The frozen draw: which manifest indices each `cells_t3_rand50` directory
    holds, plus the design constants every stage quotes as its denominator.

    The table is the authority on the expected pair count — it is 5-12 per
    directory, not one constant — so a directory it does not name is refused
    rather than measured with an unknown denominator.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_dir = payload.get("by_directory", {}).get(backbone)
    if not by_dir:
        raise SystemExit(f"{path}: no by_directory entry for backbone {backbone!r}")
    design = payload.get("design", {})
    return {
        "path": str(path),
        "schema": payload.get("schema"),
        "date": payload.get("date"),
        "by_directory": {k: [int(i) for i in v] for k, v in by_dir.items()},
        "pairs_per_cell": int(design.get("pairs_per_cell", PAIRS_PER_GROUP)),
        "pairs_per_dataset": int(design.get("pairs_per_dataset",
                                            PAIRS_PER_GROUP // 2)),
        "per_stream_counts": payload.get("per_stream_counts"),
    }


def cached_cell_dirs(data_root: Path, backbone: str, *, only: str | None = None,
                     cells_file: str | None = None
                     ) -> tuple[list[Path], list[str], list[str]]:
    """`(cells with stored paths, empty leftovers, names asked for but absent)`.

    A directory with zero `latents_*.pt` is reported as empty and never written
    as a 0-pair cell, because a 0-pair cell would satisfy the merge's
    completeness check while carrying no data.
    """
    root = Path(data_root) / backbone / "matrix" / CELLS_T3_DIRNAME
    if not root.is_dir():
        raise SystemExit(f"path-layer directory missing: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    absent: list[str] = []
    if cells_file:
        wanted = [ln.strip() for ln in Path(cells_file).read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
        absent = [n for n in wanted if n not in set(names)]
        names = [n for n in names if n in set(wanted)]
    if only:
        names = [n for n in names if fnmatch.fnmatch(n, only)]
    selected, empty = [], []
    for n in names:
        d = root / n
        if next(d.glob(_LAT_GLOB), None) is None:
            empty.append(n)
            continue
        selected.append(d)
    return selected, empty, absent


def stored_indices(directory: Path, limit: int | None) -> list[int]:
    idx = sorted(int(_LAT_RE.search(p.name).group(1))  # type: ignore[union-attr]
                 for p in directory.glob(_LAT_GLOB))
    return idx[:limit] if limit is not None else idx


# --------------------------------------------------------------------------
# the per-pair readings (plan section 3.9.3)
# --------------------------------------------------------------------------


def cached_pair_readings(Zc: np.ndarray, Zr: np.ndarray, Fr: np.ndarray | None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`D[n] = ||Z^c[n] - Z^r[n]||`, `||Z^r[n]||`, and the three direction shares.

    Same arithmetic as `pair_readings` (section 3.10), with ONE frame instead of
    two — here the frame is the reference's and there is no two-direction
    average to take. Columns of `shares` are (along the reference chord, inside
    the reference bend plane after the chord is removed, off plane); they sum to
    1 per state because `[chord, PC1, PC2]` is orthonormal and the two PCs are
    chord-orthogonal by construction.

    One state at a time in float64: both inputs are float32 widened from the
    same bf16 store, so the subtraction is exact and only the reduction rounds;
    the working set is one state (~10 MB HYV), not another whole path.
    """
    n_states = Zc.shape[0]
    D = np.zeros(n_states, dtype=np.float64)
    norm_r = np.zeros(n_states, dtype=np.float64)
    shares = np.full((n_states, 3), np.nan, dtype=np.float64)
    for n in range(n_states):
        r = Zr[n].astype(np.float64)
        norm_r[n] = math.sqrt(max(float(r @ r), 0.0))
        v = Zc[n].astype(np.float64) - r
        g2 = float(v @ v)
        D[n] = math.sqrt(max(g2, 0.0))
        if Fr is None or g2 <= 0.0:
            continue
        c = Fr @ v
        chord_share = float(c[0] * c[0]) / g2
        plane_share = float(c[1] * c[1] + c[2] * c[2]) / g2
        shares[n] = (chord_share, plane_share,
                     max(1.0 - chord_share - plane_share, 0.0))
    return D, norm_r, shares


def cached_endpoint(Zc: np.ndarray, Zr: np.ndarray, Fr: np.ndarray | None,
                    D: np.ndarray, norm_r: np.ndarray, chord_r_t1: float,
                    chord_r_bf16: float) -> dict[str, Any]:
    """`D[50]` in three normalisations, the chord angle (both reference chord
    directions) and the first principal angle between the two bend planes."""
    cached_chord = Zc[-1].astype(np.float64) - Zc[0].astype(np.float64)
    ref_chord_t3 = Zr[-1].astype(np.float64) - Zr[0].astype(np.float64)
    ang_t2 = angle_between_deg(cached_chord, Fr[0]) if Fr is not None else float("nan")
    ang_t3 = angle_between_deg(cached_chord, ref_chord_t3)
    cached_plane = plane_rows(Zc)
    theta1, theta2 = first_principal_angle(cached_plane,
                                           None if Fr is None else Fr[1:3])
    # what the store does to a chord DIRECTION is the angle between the two
    # reference chord directions themselves; the difference of their two angles
    # to the cached chord is a different (and weaker) quantity, kept alongside
    ref_gap = angle_between_deg(Fr[0], ref_chord_t3) if Fr is not None else float("nan")
    delta = abs(ang_t2 - ang_t3) if math.isfinite(ang_t2) and math.isfinite(ang_t3) \
        else float("nan")
    return {
        "D50": float(D[-1]),
        "D50_over_chord_ref": float(D[-1] / chord_r_t1) if chord_r_t1 > 0 else None,
        "D50_over_norm_ref": float(D[-1] / norm_r[-1]) if norm_r[-1] > 0 else None,
        "chord_angle_vs_t2_deg": ang_t2 if math.isfinite(ang_t2) else None,
        "chord_angle_vs_t3_deg": ang_t3 if math.isfinite(ang_t3) else None,
        "ref_chord_t2_vs_t3_deg": ref_gap if math.isfinite(ref_gap) else None,
        "chord_angle_t2_minus_t3_deg": delta if math.isfinite(delta) else None,
        "plane_angle1_deg": theta1 if math.isfinite(theta1) else None,
        "plane_angle2_deg": theta2 if math.isfinite(theta2) else None,
        "cached_chord_len_bf16": float(np.linalg.norm(cached_chord)),
        "ref_chord_len_bf16": chord_r_bf16,
        "ref_chord_len_t1_float32": chord_r_t1,
        "cached_plane_recovered": cached_plane is not None,
    }


def landmark_payload(state: int | None, D: np.ndarray, norm_r: np.ndarray,
                     shares: np.ndarray, chord_r_t1: float) -> dict[str, Any]:
    """One landmark row of one pair. `state=None` (a row that never caches has
    no k0+1) gives an all-null row, which the aggregation drops by count."""
    if state is None or not (0 <= state < D.shape[0]):
        return {"state": None, "D": None, "D_over_chord_ref": None,
                "D_over_norm_ref": None, "share_chord": None,
                "share_in_plane": None, "share_off_plane": None}
    s = shares[state]
    return {
        "state": int(state),
        "D": float(D[state]),
        "D_over_chord_ref": float(D[state] / chord_r_t1) if chord_r_t1 > 0 else None,
        "D_over_norm_ref": float(D[state] / norm_r[state]) if norm_r[state] > 0 else None,
        "share_chord": None if not math.isfinite(s[0]) else float(s[0]),
        "share_in_plane": None if not math.isfinite(s[1]) else float(s[1]),
        "share_off_plane": None if not math.isfinite(s[2]) else float(s[2]),
    }


def build_pair_record(*, prompt_idx: int, Zc: np.ndarray, Zr: np.ndarray,
                      Fr: np.ndarray | None, frame_source: str,
                      frame_file: str | None, actions: str, n_cached: int,
                      ref_row: dict[str, Any], cell_row: dict[str, Any]
                      ) -> dict[str, Any]:
    """Everything P7 reads off one (cached path, reference path) pair.

    Per-pair values are stored, not per-cell medians: the pooling that produces
    the plan's 20-pair rows crosses the two datasets, which can sit on different
    filesystems, so the merge has to be able to re-take the medians.
    """
    chord_r_t1 = float(ref_row["chord_len"])
    magnitude_t1 = np.asarray(ref_row["magnitude"], dtype=np.float64)
    D, norm_r, shares = cached_pair_readings(Zc, Zr, Fr)
    chord_r_bf16 = float(np.linalg.norm(Zr[-1].astype(np.float64)
                                        - Zr[0].astype(np.float64)))
    k0 = first_cache_step(actions)
    n_prefix = N_STATES if k0 is None else min(k0 + 1, N_STATES)
    prefix_max = float(np.max(D[:n_prefix])) if n_prefix > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        d_over_norm = np.where(norm_r > 0, D / norm_r, np.nan)
        mag_gap = np.where(magnitude_t1 > 0, (norm_r - magnitude_t1) / magnitude_t1,
                           np.nan)
    dD = np.diff(D)
    landmarks = {"k0_plus_1": landmark_payload(None if k0 is None else k0 + 1,
                                               D, norm_r, shares, chord_r_t1)}
    for n in CACHED_LANDMARKS:
        landmarks[str(n)] = landmark_payload(n, D, norm_r, shares, chord_r_t1)
    return {
        "prompt_idx": int(prompt_idx),
        "seed_reference": ref_row.get("seed"),
        "seed_cached": cell_row.get("seed"),
        "z_T_sha256": ref_row.get("z_T_sha256"),
        "z_T_sha256_cached": cell_row.get("z_T_sha256"),
        # comparable only when BOTH sides recorded it; "one side did not record
        # it" is a different statement from "the two runs started from
        # different noise" and is counted separately rather than folded in
        "z_T_sha256_comparable": bool(ref_row.get("z_T_sha256")
                                      and cell_row.get("z_T_sha256")),
        "z_T_sha256_match": bool(ref_row.get("z_T_sha256")
                                 == cell_row.get("z_T_sha256")),
        "k0": k0, "n_cached": int(n_cached), "actions": actions,
        "frame_source": frame_source, "frame_file": frame_file,
        "chord_r_t1_float32": chord_r_t1,
        "chord_r_bf16": chord_r_bf16,
        "chord_rel_gap": (chord_r_bf16 - chord_r_t1) / chord_r_t1 if chord_r_t1 > 0 else None,
        "D": _to_series(D),
        "D_over_chord_ref": _to_series(D / chord_r_t1) if chord_r_t1 > 0
        else [None] * N_STATES,
        "D_over_norm_ref": _to_series(d_over_norm),
        "norm_ref_bf16": _to_series(norm_r),
        "magnitude_t1_float32": _to_series(magnitude_t1),
        "magnitude_rel_gap_med": float(np.nanmedian(mag_gap)),
        "magnitude_rel_gap_absmax": float(np.nanmax(np.abs(mag_gap))),
        "shares": [[None if not math.isfinite(v) else float(v) for v in row]
                   for row in shares],
        "dD": _to_series(dD),
        "prefix": {
            "n_states_compared": int(n_prefix),
            # a row that caches NOTHING has no cache-induced prefix at all: the
            # whole path is then a run-to-run difference, which is a different
            # object from "the identity window before the first cache" and is
            # labelled as such instead of being listed as a prefix anomaly
            "has_cache_step": k0 is not None,
            "rule": ("states 0..k0 inclusive (a cache at step k0 first writes state k0+1)"
                     if k0 is not None else
                     "no cache step in this row: all 51 states compared, and the reading "
                     "is a run-to-run difference, NOT a prefix-identity check"),
            "max_abs_D": prefix_max if math.isfinite(prefix_max) else None,
            "exactly_zero": bool(n_prefix > 0 and prefix_max == 0.0),
            "expected": "exactly 0" if k0 is not None else "not a prefix-identity claim",
        },
        "D0": float(D[0]),
        "endpoint": cached_endpoint(Zc, Zr, Fr, D, norm_r, chord_r_t1, chord_r_bf16),
        "landmarks": landmarks,
    }


# --------------------------------------------------------------------------
# event alignment on Delta D (plan section 3.9.3, last bullet)
# --------------------------------------------------------------------------


class CachedEventAccumulator:
    """Two views of `Delta D[k] = D[k+1] - D[k]`, solver-step axis k = 0..49.

    1. stratified per-step: the median increment at each k, split by whether
       step k is itself a cache step. The sign and median of the `full` stratum
       is the plan's "does it regress or keep diverging".
    2. event-aligned: origin at every cache step, offsets from P6's
       `EVENT_OFFSETS` so the two layers' event figures line up, stratified by
       whether step k+j is itself cached.

    `CAV_OVERLAP` applies verbatim: windows overlap when cache runs are short,
    one step can enter several (k, j) buckets, so the samples inside a bucket
    are not independent — `n_samples` is reported and no interval is put on the
    median. Samples whose step lies strictly before k0 are structurally 0 (both
    `D[k]` and `D[k+1]` are 0 there) and are NOT pushed into the bucket at all;
    only their count is kept, in `n_samples_before_k0`. A bucket that ends up
    with nothing but such samples is still emitted, with `n_samples = 0`, so
    the reader can tell "measured and structurally empty" from "never reached".
    """

    def __init__(self) -> None:
        self.per_step: dict[tuple[int, str], list[tuple[float, float]]] = {}
        self.event: dict[tuple[int, str], list[tuple[float, float]]] = {}
        self.before_k0: dict[tuple[str, int, str], int] = {}
        self.n_pairs = 0
        self.pairs_per_bucket: dict[tuple[str, int, str], set[int]] = {}

    def add(self, actions: str, dD: np.ndarray, k0: int | None, chord_r: float,
            pair_id: int) -> None:
        self.n_pairs += 1
        scale = chord_r if chord_r > 0 else float("nan")
        act = np.frombuffer(actions.encode("ascii"), dtype=np.uint8) - ord("0")
        for k in range(NUM_STEPS):
            stratum = "cache" if act[k] else "full"
            key = ("per_step", k, stratum)
            if k0 is not None and k < k0:
                # D[k] = D[k+1] = 0 by construction: counted, never averaged in
                self.before_k0[key] = self.before_k0.get(key, 0) + 1
                self.pairs_per_bucket.setdefault(key, set())
                continue
            self._push(self.per_step, (k, stratum), dD[k], scale)
            self.pairs_per_bucket.setdefault(key, set()).add(pair_id)
        if k0 is None:
            return
        for k in np.flatnonzero(act == 1):
            for j in EVENT_OFFSETS:
                kj = int(k) + j
                if not 0 <= kj < NUM_STEPS:
                    continue
                stratum = "cache" if act[kj] else "full"
                key = ("event", j, stratum)
                if kj < k0:
                    self.before_k0[key] = self.before_k0.get(key, 0) + 1
                    self.pairs_per_bucket.setdefault(key, set())
                    continue
                self._push(self.event, (j, stratum), dD[kj], scale)
                self.pairs_per_bucket.setdefault(key, set()).add(pair_id)

    @staticmethod
    def _push(store: dict[tuple[int, str], list[tuple[float, float]]],
              key: tuple[int, str], value: float, scale: float) -> None:
        store.setdefault(key, []).append((float(value), float(value) / scale))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_pairs": self.n_pairs, "per_step": [], "event": []}
        for view, store in (("per_step", self.per_step), ("event", self.event)):
            # a bucket whose samples were ALL structurally before k0 has no
            # entry in `store`; it is still emitted (n_samples = 0) so the
            # reader can tell it apart from a bucket the schedule never reached
            keys = set(store) | {(i, s) for (v, i, s) in self.before_k0 if v == view}
            rows = []
            for (index, stratum) in sorted(keys):
                arr = np.asarray(store.get((index, stratum), []),
                                 dtype=np.float64).reshape(-1, 2)
                rows.append({
                    "view": view, "index": int(index), "stratum": stratum,
                    "dD": summarise_samples(arr[:, 0]),
                    "dD_over_chord_ref": summarise_samples(arr[:, 1]),
                    "n_pairs": len(self.pairs_per_bucket.get((view, index, stratum), ())),
                    "n_samples_before_k0": self.before_k0.get((view, index, stratum), 0),
                })
            out[view] = rows
        out["index_axis"] = {
            "per_step": "solver step k = 0..49 (step k moves state k to state k+1); "
                        "Delta D[k] = D[k+1] - D[k]",
            "event": f"offset j from a cache step k, j in {list(EVENT_OFFSETS)}; the "
                     f"sample is Delta D[k+j] and the stratum is what step k+j itself is",
        }
        out["caveats"] = [CAV_OVERLAP, CAV_NEG_OFFSET, CAV_DELTA_NORM]
        return out


# --------------------------------------------------------------------------
# stage `cells`: one json per path-layer directory
# --------------------------------------------------------------------------


def cached_shared_params(args: argparse.Namespace) -> dict[str, Any]:
    """The run parameters every shard must agree on. The cell SELECTOR is not
    among them: a shard necessarily passes its own `--only` / `--cells`, which
    is the whole point of sharding, so the selector is recorded per shard and
    reported instead of being compared."""
    return {"backbone": args.backbone,
            "streams": list(args.streams), "n_states": N_STATES,
            "schema": CACHED_SCHEMA}


def _provenance(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "cluster": args.cluster or os.environ.get("SLURM_CLUSTER_NAME") or "unrecorded",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mtimes": {name: (p.stat().st_mtime if p.exists() else None)
                   for name, p in paths.items()},
        "data_root": str(args.data_root),
    }


def run_cached_cells(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    cells_out = Path(args.out_tables) / "cells"
    cells_out.mkdir(parents=True, exist_ok=True)
    streams_meta = {name: parse_stream(name) for name in args.streams}
    stream_of = {(ds, seed): name for name, (ds, seed) in streams_meta.items()}

    dirs, empty, absent = cached_cell_dirs(data_root, args.backbone,
                                           only=args.only, cells_file=args.cells)
    if absent:
        raise SystemExit(f"--cells names {len(absent)} directories that do not exist under "
                         f"{data_root / args.backbone / 'matrix' / CELLS_T3_DIRNAME}: "
                         f"{absent}")
    if empty:
        print(f"skipping {len(empty)} {CELLS_T3_DIRNAME} directories with no stored "
              f"latents — not present here, NOT a 0-pair cell: {empty}", flush=True)
    if not dirs:
        raise SystemExit(f"no {CELLS_T3_DIRNAME} directory with stored latents matched the "
                         f"selection; nothing to do on this filesystem")
    sample = load_sample_table(Path(args.sample_list), args.backbone)
    params = {**cached_shared_params(args), "stage": "cells",
              "sample_table": sample["schema"], "sample_table_date": sample["date"],
              "pairs_per_cell": sample["pairs_per_cell"],
              "pairs_per_dataset": sample["pairs_per_dataset"]}
    print(f"{len(dirs)} cells selected on {socket.gethostname()}; sample table "
          f"{sample['path']} ({sample['pairs_per_cell']} pairs per (method, budget))",
          flush=True)

    for cell in dirs:
        meta = cached_cell_meta(cell.name)
        out_path = cells_out / f"cached_bend_{args.backbone}__{cell.name}.json"
        wanted = sample["by_directory"].get(cell.name)
        if wanted is None:
            raise SystemExit(
                f"{cell.name} is not one of the {len(sample['by_directory'])} directories "
                f"the sample table {sample['path']} draws for {args.backbone}. Its "
                f"expected pair count is therefore unknown; do not measure it.")
        on_disk = [i for i in stored_indices(cell, None) if i in set(wanted)]
        stored = resolve_reuse(out_path, params, caps=("pairs_per_cell",),
                               force=args.force)
        if stored is not None:
            # the run parameters alone do not say how much of the cell was on
            # disk when it was written: a shard started while the reference
            # subset was still being copied in produces a SHORT cell, and its
            # parameters look identical afterwards. Compare against what is
            # readable now, so a cell that has since grown is recomputed.
            n_stored = int(stored.get("n_pairs", -1))
            if n_stored >= len(on_disk):
                print(f"[skip] {out_path.name} already covers these parameters "
                      f"({n_stored} pairs, {len(on_disk)} readable now)", flush=True)
                continue
            print(f"[recompute] {out_path.name} holds {n_stored} pairs but {len(on_disk)} "
                  f"latents are readable in {cell.name} now", flush=True)
        stream = stream_of.get((meta["dataset"], meta["base_seed"]))
        if stream is None:
            raise SystemExit(
                f"{cell.name}: (dataset, base seed) = ({meta['dataset']}, "
                f"{meta['base_seed']}) is not one of --streams {list(args.streams)}; "
                f"the sampled path layer was generated on the three penguin599 streams "
                f"(54, 55, 56) and the three vbench944 streams (42, 43, 44)")
        ref_dir = data_root / args.backbone / "matrix" / "references_t3" / stream
        if not ref_dir.is_dir() or next(ref_dir.glob(_LAT_GLOB), None) is None:
            raise SystemExit(
                f"reference paths for {cell.name} are not on this filesystem: {ref_dir} "
                f"is missing or holds no {_LAT_GLOB}. Copy the block-A reference subset "
                f"(latents + frame_%05d_000_051.npy + traj_%05d.json for the idx this "
                f"shard needs) before running this shard — a cell is never emitted with "
                f"0 pairs.")

        idxs = on_disk
        if not idxs:
            raise SystemExit(
                f"{cell.name} holds latents, but none of the {len(wanted)} indices the "
                f"sample table drew for it ({wanted}); this directory was generated "
                f"against a different draw")
        cache = PathCache({"cached": cell, "reference": ref_dir}, args.cache_paths)
        frame_counts: dict[str, int] = {}
        pairs: list[dict[str, Any]] = []
        for i in idxs:
            ref_lat = ref_dir / f"latents_{i:05d}.pt"
            if not ref_lat.is_file():
                raise SystemExit(f"{cell.name} idx {i}: reference path missing at {ref_lat}")
            frame_path, frame_src = find_frame(ref_dir, i, meta["dataset"],
                                               meta["base_seed"], data_root, args.backbone)
            frame_counts[frame_src] = frame_counts.get(frame_src, 0) + 1
            Fr = load_frame(frame_path) if frame_path is not None else None
            ref_row = load_traj_row(ref_dir / f"traj_{i:05d}.json")
            cell_row = load_traj_row(cell / f"traj_{i:05d}.json")
            actions, n_cached = load_actions(cell / f"decisions_{i:05d}.json")
            Zc = cache.get(("cached", i))
            Zr = cache.get(("reference", i))
            if Zc.shape != Zr.shape:
                raise SystemExit(f"{cell.name} idx {i}: cached path {Zc.shape} and "
                                 f"reference path {Zr.shape} disagree")
            rec = build_pair_record(
                prompt_idx=i, Zc=Zc, Zr=Zr, Fr=Fr, frame_source=frame_src,
                frame_file=None if frame_path is None else str(frame_path),
                actions=actions, n_cached=n_cached, ref_row=ref_row, cell_row=cell_row)
            # the whole pairing rests on filename index equality between two
            # directories written by different runs (and, for 23 HunyuanVideo
            # cells, different clusters). Plan section 2.5 pairs by z_T equality,
            # so a mismatch is not "a noisy pair", it is NOT THIS PROMPT — refuse
            # rather than average it into a median.
            if rec["z_T_sha256_comparable"] and not rec["z_T_sha256_match"]:
                raise SystemExit(
                    f"{cell.name} idx {i}: the cached run and the reference run do not "
                    f"share z_T (cached {rec['z_T_sha256_cached']!r} vs reference "
                    f"{rec['z_T_sha256']!r}). Plan section 2.5 pairs by z_T equality, so "
                    f"this is a re-indexed or mismatched stream, not a measurement — fix "
                    f"the pairing before merging.")
            pairs.append(rec)
            del Fr
            print(f"    {cell.name} idx {i}: D[50]/chord = "
                  f"{_fmt(pairs[-1]['endpoint']['D50_over_chord_ref'])}, k0 = "
                  f"{pairs[-1]['k0']}, frame {frame_src}", flush=True)
        del cache

        report = {
            "schema": CACHED_SCHEMA, "stage": "cells",
            "plan_section": "3.9.3",
            "backbone": args.backbone,
            **meta,
            "cell_path": str(cell),
            "reference_dir": str(ref_dir),
            "reference_stream": stream,
            "d": int(load_traj_row(ref_dir / f"traj_{idxs[0]:05d}.json")["d"]),
            "n_pairs": len(pairs),
            "n_pairs_expected": len(wanted),
            "prompt_indices_expected": list(wanted),
            "n_pairs_dropped": max(0, len(wanted) - len(pairs)),
            "n_pairs_dropped_reason": (
                None if len(pairs) >= len(wanted) else
                f"only {len(pairs)} of the {len(wanted)} sampled prompt indices have a "
                f"stored latent in {cell.name}: missing "
                f"{sorted(set(wanted) - set(idxs))}"),
            "n_pairs_z_T_not_comparable": sum(
                1 for p in pairs if not p["z_T_sha256_comparable"]),
            "frame_sources": frame_counts,
            "run_params": params,
            "shard": {"only": args.only, "cells_file": args.cells,
                      "sample_list": sample["path"],
                      **_provenance({"cell": cell, "reference": ref_dir}, args)},
            "pairs": pairs,
        }
        atomic_write_json(out_path, report)
        print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, "
              f"{len(pairs)} pairs)", flush=True)


# --------------------------------------------------------------------------
# stage `refside`: reproducibility floor + T2-vs-T3 plane check + bf16 bound
# --------------------------------------------------------------------------


def refside_path(out_tables: Path, backbone: str) -> Path:
    return Path(out_tables) / "cells" / f"cached_refside_{backbone}.json"


def run_cached_refside(args: argparse.Namespace) -> None:
    """The reference-side block, emitted ONCE per backbone.

    Both halves are properties of the references, not of any cell, so they are
    computed here and merged as a single block:

      * the reproducibility floor `||Z^r'[n] - Z^r[n]||`, where `Z^r` is the
        re-run (`references_t3/`, idx 0-119 on each of the six streams) and
        `Z^r'` the matrix wave (`references/`, every prompt of the dataset), so
        the floor IS computable on the first `--refside_idx_limit` indices of
        every stream — this stage needs a filesystem carrying both.
      * the T2-vs-T3 first principal angle: the stored T2 plane of a generation
        against the plane recomputed from that same generation's bf16 T3.
      * the bf16 lower bound, straight out of the P1 dump via
        `g_floor_estimates` — `D` is the same object as `G` (the norm of a
        difference of two independently rounded states), so the deviation /
        spacing / analytic triple applies unchanged.
    """
    data_root = Path(args.data_root)
    out_path = refside_path(Path(args.out_tables), args.backbone)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # the floor dump is loaded BEFORE the reuse decision, because which dump it
    # is (and which dtype it describes) is part of what this stage measured: a
    # re-issued P1 dump has to invalidate the stored bound, not be skipped past
    floor_dir = (Path(args.p1_floor_dir) if args.p1_floor_dir
                 else _DEFAULT_FLOOR_DIR / args.backbone)
    floor_json = Path(args.p1_floor_json) if args.p1_floor_json else None
    floors = load_p1_floors(floor_dir, floor_json)
    floor_source = str(floor_json or (floor_dir / "p1_floor_bfloat16.json"))
    streams_used = list(args.refside_streams or args.streams)
    params = {**cached_shared_params(args), "stage": "refside",
              "streams": streams_used,
              "refside_idx_limit": args.refside_idx_limit,
              "p1_floor_source": floor_source,
              "p1_quant_dtype": floors.get("quant_dtype")}
    stored = resolve_reuse(out_path, params, caps=("refside_idx_limit",),
                           force=args.force)
    if stored is not None:
        print(f"[skip] {out_path.name} already covers these parameters")
        return

    streams: dict[str, Any] = {}
    chords: list[float] = []
    spacing_rows: list[list[float]] = []
    magnitude_rows: list[list[float]] = []

    for name in streams_used:
        dataset, base_seed = parse_stream(name)
        t3_dir = data_root / args.backbone / "matrix" / "references_t3" / name
        wave_dir = data_root / args.backbone / "matrix" / "references" / name
        if not t3_dir.is_dir():
            raise SystemExit(f"block-A stream directory missing: {t3_dir}")
        if not wave_dir.is_dir() or next(wave_dir.glob(_LAT_GLOB), None) is None:
            raise SystemExit(
                f"the matrix-wave copy of this stream is not on this filesystem "
                f"({wave_dir} missing or holding no {_LAT_GLOB}), so the reproducibility "
                f"floor cannot be computed here. Run `--stage refside` on the cluster that "
                f"holds both references/ and references_t3/ (site_a); the sharded cell "
                f"stage does not need it.")
        idxs = stored_indices(t3_dir, args.refside_idx_limit)
        entries: list[dict[str, Any]] = []
        angles: list[tuple[float, float]] = []
        missing_wave: list[int] = []
        frame_counts: dict[str, int] = {}
        cache = PathCache({"t3": t3_dir, "wave": wave_dir}, args.cache_paths)
        for i in idxs:
            row = load_traj_row(t3_dir / f"traj_{i:05d}.json")
            chords.append(float(row["chord_len"]))
            spacing_rows.append(list(row["spacing"]))
            magnitude_rows.append(list(row["magnitude"]))
            frame_path, frame_src = find_frame(t3_dir, i, dataset, base_seed,
                                               data_root, args.backbone)
            frame_counts[frame_src] = frame_counts.get(frame_src, 0) + 1
            Fr = load_frame(frame_path) if frame_path is not None else None
            Zr = cache.get(("t3", i))
            # (a) T2 plane vs the plane recomputed from the SAME generation's T3.
            # `ang1`/`ang2` are the FIRST and SECOND principal angle (nothing to
            # do with the T1/T2/T3 data layers named on either side of them).
            ang1, ang2 = first_principal_angle(plane_rows(Zr),
                                               None if Fr is None else Fr[1:3])
            angles.append((ang1, ang2))
            # (b) the reproduction floor, same three normalisations as D
            if not (wave_dir / f"latents_{i:05d}.pt").is_file():
                missing_wave.append(i)
                del Fr
                continue
            Zw = cache.get(("wave", i))
            entry = build_pair_record(
                prompt_idx=i, Zc=Zw, Zr=Zr, Fr=Fr, frame_source=frame_src,
                frame_file=None if frame_path is None else str(frame_path),
                actions="0" * NUM_STEPS, n_cached=0, ref_row=row,
                cell_row=load_traj_row(wave_dir / f"traj_{i:05d}.json"))
            entry["plane_angle_t2_vs_t3_deg"] = None if not math.isfinite(ang1) else ang1
            entries.append(entry)
            del Fr
            print(f"    floor {name} idx {i}: max_n ||Z^r'[n]-Z^r[n]|| = "
                  f"{_fmt(max(v for v in entry['D'] if v is not None))}, "
                  f"T2-vs-T3 angle {_fmt(ang1)} deg", flush=True)
        del cache
        if not entries:
            raise SystemExit(
                f"stream {name}: none of the {len(idxs)} block-A indices this stage read "
                f"has a matrix-wave copy under {wave_dir} (missing: {missing_wave}), so "
                f"the run-to-run reproduction floor has NO pairs for this stream. The "
                f"plan's P7 acceptance list needs that row, and an empty floor table "
                f"would pass it vacuously. Copy the matrix-wave latents for idx "
                f"{idxs[:3]}... in, or run this stage where they live.")
        first = np.asarray([a for a, _ in angles], dtype=np.float64)
        streams[name] = {
            "dataset": dataset, "base_seed": base_seed,
            "t3_dir": str(t3_dir), "wave_dir": str(wave_dir),
            "n_paths_read": len(idxs),
            "n_floor_pairs": len(entries),
            "idx_without_a_matrix_wave_copy": missing_wave,
            "plane_check": {
                "n": int(np.isfinite(first).sum()),
                "first_angle_deg": {**_quantiles(first),
                                    "max": float(np.nanmax(first)) if first.size else None},
                "second_angle_deg": _quantiles([b for _, b in angles]),
                # which frame each angle used: `same_run` is the clean reading
                # (that generation's own T2 vs its own T3). `matrix_reference`
                # is the matrix WAVE's frame — a different generation, the very
                # one used as Z^r' for the floor — so those rows mix run-to-run
                # difference into a number meant to isolate store rounding.
                "frame_sources": dict(frame_counts),
                "n_paths_same_run_frame": frame_counts.get("same_run", 0),
                "n_paths_matrix_reference_frame": frame_counts.get("matrix_reference", 0),
                "what_it_is": "T2 frame rows 1,2 (computed in flight on the float32 path, "
                              "stored float16) vs plane_frame() rows 1,2 recomputed from "
                              "the same generation's bf16 T3; the deviation from 0 deg is "
                              "what the store does to a plane's orientation — an UPPER "
                              "bound on the bf16 part, because the T2 file is itself "
                              "float16, and not a clean reading at all on the paths "
                              "counted under `matrix_reference` (there the frame comes "
                              "from a DIFFERENT generation of the same prompt)",
            },
            "pairs": entries,
        }

    chord_med = float(np.median(chords)) if chords else float("nan")
    spacing_med = np.median(np.asarray(spacing_rows, dtype=np.float64), axis=0)
    magnitude_med = np.median(np.asarray(magnitude_rows, dtype=np.float64), axis=0)
    bound = g_floor_estimates(floors, chord_med, spacing_med, magnitude_med)
    bound["applies"] = ("only AFTER k0: before k0 the two runs hold identical float32 "
                        "states, which round to identical bf16, so the bound there is 0")
    bound["source"] = floor_source
    bound["quant_dtype"] = floors.get("quant_dtype")

    # the two P1 scalars the cell stage's store cross-checks are compared
    # against, carried here so the reader does not have to open a second file
    mag_rel = np.asarray(floors.get("magnitude_rel_med") or [], dtype=np.float64)
    expectations = {
        "chord_rel_med": floors.get("chord_rel_med"),
        "magnitude_rel_med_median": (float(np.median(mag_rel)) if mag_rel.size
                                     else None),
        "magnitude_rel_med_absmax": (float(np.max(np.abs(mag_rel))) if mag_rel.size
                                     else None),
        "source": floor_source,
        "what": "P1's own bf16-vs-float32 relative gaps for the chord length and the "
                "per-state magnitude; the cell stage's `store_cross_checks` measure the "
                "same two gaps on the P7 references and should land at this size",
    }
    windows = readable_windows(floors)
    report = {
        "schema": CACHED_SCHEMA, "stage": "refside", "plan_section": "3.9.3",
        "backbone": args.backbone,
        "run_params": params,
        "shard": _provenance({}, args),
        "streams": streams,
        "bf16_bound": bound,
        "p1_store_expectations": expectations,
        "p1_turn_window": {
            "min_readable_multistep_window_bf16":
                windows.get("min_readable_multistep_window_bf16"),
            "per_window": {w: v for w, v in windows.items() if w.isdigit()},
            "source": floor_source,
            "note": "read out of the P1 dump, never hardcoded. No P7 quantity uses a "
                    "turn or curvature window; this row is here so the caveat that "
                    "mentions the window can quote a measured number.",
        },
        "floor_sources": {
            "Z_r": "matrix/references_t3/<ds>_s<base>/ — the block-A re-run, the same "
                   "generation as the T2 frame and the T1 chord used everywhere else",
            "Z_r_prime": "matrix/references/<ds>_s<base>/ — the matrix wave, used ONLY "
                         "as the second draw of the same prompt",
        },
        "caveats": [CAV_FLOOR, CAV_FRAME, CAV_PREFIX],
    }
    atomic_write_json(out_path, report)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------
# stage `merge`: concatenate the per-cell json, pool to 50 pairs, write tables
# --------------------------------------------------------------------------


def _pair_profile(pairs: list[dict[str, Any]], key: str) -> np.ndarray:
    if not pairs:
        return np.zeros((0, N_STATES))
    return np.asarray([_from_series(p[key]) for p in pairs], dtype=np.float64)


def _landmark_stats(pairs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    rows = [p["landmarks"][name] for p in pairs]
    states = [r["state"] for r in rows if r["state"] is not None]
    out: dict[str, Any] = {
        "state": (int(states[0]) if len(set(states)) == 1 else None),
        "state_range": [min(states), max(states)] if states else None,
        "n_pairs": len(rows),
        "n_pairs_with_a_state": len(states),
    }
    for field in ("D", "D_over_chord_ref", "D_over_norm_ref",
                  "share_chord", "share_in_plane", "share_off_plane"):
        out[field] = summarise_samples(
            np.asarray([np.nan if r[field] is None else r[field] for r in rows],
                       dtype=np.float64))
    chord = np.asarray([np.nan if r["share_chord"] is None else r["share_chord"]
                        for r in rows], dtype=np.float64)
    plane = np.asarray([np.nan if r["share_in_plane"] is None else r["share_in_plane"]
                        for r in rows], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["share_in_plane_of_offchord"] = summarise_samples(
            plane / np.clip(1.0 - chord, 1e-12, None))
    return out


def _schedule_spread(actions: list[str]) -> dict[str, Any]:
    """The distinct 0/1 schedules inside one group, and the modal one's share."""
    counts = collections.Counter(actions)
    if not counts:
        return {"n_distinct": 0, "n_pairs_on_modal": 0, "modal_share": None}
    top = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {"n_distinct": len(counts), "n_pairs_on_modal": int(top[1]),
            "modal_share": float(top[1]) / len(actions)}


def summarise_cached_group(pairs: list[dict[str, Any]], *, method: str, budget: str,
                           dataset: str, mode_raw: str, cells: list[str],
                           expected: int = PAIRS_PER_GROUP) -> dict[str, Any]:
    """One row of the plan's table: (method, budget) pooled over the datasets,
    or one (method, budget, dataset) half of it."""
    events = CachedEventAccumulator()
    for j, p in enumerate(pairs):
        chord = p["chord_r_t1_float32"] or float("nan")
        events.add(p["actions"], _from_series(p["dD"]), p["k0"], chord, j)
    k0s = [p["k0"] for p in pairs if p["k0"] is not None]
    # a pair with NO cache step makes no prefix-identity claim (there is no k0
    # to be before), so it is named separately instead of being listed next to
    # genuine prefix anomalies
    with_prefix = [p for p in pairs if p["prefix"].get("has_cache_step", True)]
    without_prefix = [{"cell": p["_cell"], "prompt_idx": p["prompt_idx"],
                       "max_abs_D_over_all_51_states": p["prefix"]["max_abs_D"]}
                      for p in pairs if not p["prefix"].get("has_cache_step", True)]
    nonzero_prefix = [{"cell": p["_cell"], "prompt_idx": p["prompt_idx"],
                       "max_abs_D": p["prefix"]["max_abs_D"], "k0": p["k0"]}
                      for p in with_prefix if not p["prefix"]["exactly_zero"]]
    frame_counts: dict[str, int] = {}
    for p in pairs:
        frame_counts[p["frame_source"]] = frame_counts.get(p["frame_source"], 0) + 1
    endpoint = {field: summarise_samples(
        np.asarray([np.nan if p["endpoint"][field] is None else p["endpoint"][field]
                    for p in pairs], dtype=np.float64))
        for field in ("D50", "D50_over_chord_ref", "D50_over_norm_ref",
                      "chord_angle_vs_t2_deg", "chord_angle_vs_t3_deg",
                      "ref_chord_t2_vs_t3_deg", "chord_angle_t2_minus_t3_deg",
                      "plane_angle1_deg",
                      "plane_angle2_deg")}
    landmarks = {name: _landmark_stats(pairs, name)
                 for name in ("k0_plus_1",) + tuple(str(n) for n in CACHED_LANDMARKS)}
    return {
        "method": method, "mode_raw": mode_raw, "budget": budget,
        "K_nominal": BUDGET_K[budget], "dataset": dataset,
        "cells": sorted(cells),
        "n_pairs": len(pairs),
        "n_pairs_expected": expected,
        "n_pairs_dropped": expected - len(pairs),
        "n_pairs_dropped_reason": (
            None if len(pairs) >= expected else
            "one of the seed-stream directories of this (method, budget, dataset) stored "
            "fewer latents than the sample table drew for it, or is missing from this "
            "merge entirely (see cells_missing / the per-cell n_pairs_dropped_reason)"),
        "n_pairs_with_a_t2_frame": sum(1 for p in pairs if p["frame_source"] != "missing"),
        "frame_sources": frame_counts,
        # a z_T mismatch is refused at the cell stage; the counts survive here so
        # a merge of cell json written before that check still shows them
        "n_pairs_z_T_mismatch": sum(
            1 for p in pairs
            if p.get("z_T_sha256_comparable", True) and not p["z_T_sha256_match"]),
        "n_pairs_z_T_not_comparable": sum(
            1 for p in pairs if not p.get("z_T_sha256_comparable", True)),
        # the sampled prompts do not all take the modal path, so a dynamic gate's
        # k0 is a DISTRIBUTION over the group's pairs, not one number: the value
        # counts are carried next to the min / max
        "k0": {"min": min(k0s) if k0s else None, "max": max(k0s) if k0s else None,
               "counts": {str(v): int(c) for v, c in
                          sorted(collections.Counter(k0s).items())},
               "mode": (max(collections.Counter(k0s).items(),
                            key=lambda kv: (kv[1], -kv[0]))[0] if k0s else None),
               "n_pairs_without_a_cache_step": len(pairs) - len(k0s)},
        "n_cached": summarise_samples(np.asarray([p["n_cached"] for p in pairs],
                                                 dtype=np.float64)),
        # how many DIFFERENT schedules the sampled prompts took inside this
        # group: a dynamic gate is only "one path per cell" if this is 1
        "schedules": _schedule_spread([p["actions"] for p in pairs]),
        "profiles": {
            "D_abs": summarise_profile(_pair_profile(pairs, "D")),
            "D_over_chord_ref": summarise_profile(_pair_profile(pairs, "D_over_chord_ref")),
            "D_over_norm_ref": summarise_profile(_pair_profile(pairs, "D_over_norm_ref")),
        },
        "landmarks": landmarks,
        "endpoint": endpoint,
        "events": events.summary(),
        "prefix": {
            "rule": "states 0..k0 inclusive",
            "expected": "exactly 0",
            "n_pairs_with_a_prefix": len(with_prefix),
            "n_pairs_exactly_zero": len(with_prefix) - len(nonzero_prefix),
            "n_pairs_not_exactly_zero": len(nonzero_prefix),
            "pairs_not_exactly_zero": nonzero_prefix,
            "n_pairs_without_a_cache_step": len(without_prefix),
            "pairs_without_a_cache_step": without_prefix,
            "max_abs_D_over_pairs": max(
                [p["prefix"]["max_abs_D"] or 0.0 for p in with_prefix], default=None),
            "note": CAV_PREFIX,
        },
        "store_cross_checks": {
            "chord_rel_gap": summarise_samples(
                np.asarray([np.nan if p["chord_rel_gap"] is None else p["chord_rel_gap"]
                            for p in pairs], dtype=np.float64)),
            "magnitude_rel_gap_med": summarise_samples(
                np.asarray([p["magnitude_rel_gap_med"] for p in pairs], dtype=np.float64)),
            "what": "bf16-recomputed chord / state norm against the float32 T1 value of "
                    "the same generation; P0 bounds the chord gap at 1e-3 and the P1 "
                    "dump's chord_rel_med / magnitude_rel_med are the expected sizes — "
                    "both are copied into this file under "
                    "`p1_store_expectations`, so the comparison needs no second file",
        },
        "in_plane_iqr_trigger": {
            "threshold": IN_PLANE_IQR_TRIGGER,
            "iqr_at_state_50": (
                None if landmarks["50"]["share_in_plane"]["p75"] is None
                or landmarks["50"]["share_in_plane"]["p25"] is None
                else landmarks["50"]["share_in_plane"]["p75"]
                - landmarks["50"]["share_in_plane"]["p25"]),
            "note": "plan section 4.2 says a method whose in-plane share IQR exceeds "
                    "0.3 may need 30 pairs. Reported as a number; P7 proposes nothing "
                    "and generates nothing (that decision is P8's / the user's).",
        },
    }


def load_cell_reports(cells_dir: Path, backbone: str) -> list[dict[str, Any]]:
    files = sorted(cells_dir.glob(f"cached_bend_{backbone}__*.json"))
    reports = []
    for f in files:
        rep = read_json_if_readable(f)
        if rep is None:
            raise SystemExit(f"{f} is empty or truncated; re-run that cell with "
                             f"--cells naming it")
        if rep.get("schema") != CACHED_SCHEMA or rep.get("stage") != "cells":
            raise SystemExit(f"{f}: not a P7 per-cell report "
                             f"({rep.get('schema')} / {rep.get('stage')})")
        rep["_file"] = str(f)
        reports.append(rep)
    return reports


def run_cached_merge(args: argparse.Namespace) -> None:
    tables = Path(args.out_tables)
    figs = Path(args.out_figs)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    T = args.backbone
    cells_dir = tables / "cells"
    reports = load_cell_reports(cells_dir, T)
    if not reports:
        raise SystemExit(f"no per-cell reports under {cells_dir}; run `--stage cells` "
                         f"over {CELLS_T3_DIRNAME}/ first")

    # every shard must agree on what was measured; the SELECTOR may differ
    param_sets = {json.dumps(r["run_params"], sort_keys=True) for r in reports}
    if len(param_sets) != 1:
        raise SystemExit("per-cell reports disagree on their run parameters:\n  "
                         + "\n  ".join(sorted(param_sets)))
    params = reports[0]["run_params"]
    dims = {r["d"] for r in reports}
    if len(dims) != 1:
        raise SystemExit(f"per-cell reports disagree on d: {sorted(dims)}")

    datasets = sorted({parse_stream(s)[0] for s in params["streams"]})
    expected = {(m, k, ds) for m in METHODS for k in KS for ds in datasets}
    if len(datasets) == 2 and len(expected) != EXPECTED_CELLS:
        raise SystemExit(
            f"the method / budget vocabulary imported from P6 gives {len(expected)} cells "
            f"over 2 datasets, but the path layer is {EXPECTED_CELLS} groups "
            f"({len(METHODS)} methods x {len(KS)} budgets x 2). One of the two has drifted; "
            f"do not merge against a vocabulary the data was not generated with.")
    # a (method, K, dataset) is now covered by THREE directories, one per seed
    # stream, which the merge joins into that dataset's 25 pairs; only the same
    # (method, K, dataset, base seed) twice is a genuine duplicate
    seen: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    by_seed: dict[tuple[str, int, str, int], list[str]] = {}
    for r in reports:
        seen.setdefault((r["method"], r["K_nominal"], r["dataset"]), []).append(r)
        by_seed.setdefault((r["method"], r["K_nominal"], r["dataset"],
                            r["base_seed"]), []).append(r["cell_dir"])
    duplicates = {str(k): v for k, v in by_seed.items() if len(v) > 1}
    if duplicates:
        raise SystemExit(f"the same (method, K, dataset, base seed) is covered twice: "
                         f"{duplicates}")
    missing = sorted(expected - set(seen))
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"{len(missing)} of {len(expected)} (method, K, dataset) groups are missing "
            f"from {cells_dir}: {missing}\nRun `--stage cells` for their directories, "
            f"then merge again. --allow_incomplete records the gap instead, and every "
            f"table then states a denominator below {PAIRS_PER_GROUP}.")

    refside = read_json_if_readable(refside_path(tables, T))
    if refside is None:
        raise SystemExit(
            f"the reference-side block {refside_path(tables, T)} is missing. The plan's "
            f"P7 acceptance list requires the reproducibility-floor row, the bf16 lower "
            f"bound and the T2-vs-T3 plane check; run\n"
            f"  python analysis/video_trajectory/latent_paths.py cached --stage refside "
            f"--backbone {T} ...\non the cluster that holds both references/ and "
            f"references_t3/ (site_a).")
    # the refside block supplies the floor row and the plane-check row of tables
    # that state the cells' denominator, so it has to describe the same backbone.
    # Neither its stream list nor its index cap is compared: the reproduction
    # floor needs a stream that has BOTH a matrix wave and a re-run stored path,
    # which only the two base streams do, and it is read on their first indices —
    # a property of the references, unrelated to which prompts the sample drew
    ref_params = refside.get("run_params") or {}
    ref_diffs = {k: (ref_params.get(k, "<absent>"), params[k])
                 for k in ("backbone",)
                 if ref_params.get(k, "<absent>") != params[k]}
    if ref_diffs:
        raise SystemExit(
            f"{refside_path(tables, T)} was produced with different run parameters than "
            f"the per-cell reports: "
            + "; ".join(f"{k}: refside {r!r} vs cells {c!r}" for k, (r, c) in
                        sorted(ref_diffs.items()))
            + "\nRe-run `--stage refside` with the parameters the cells used.")

    # the merge overwrites every final artefact, so it guards itself like the
    # other three stages: a partial merge (fewer cells, fewer pairs, or
    # --allow_incomplete) must not silently replace a complete one's tables
    json_path = tables / f"cached_bend_{T}.json"
    merge_params = {**params, "stage": "merge",
                    "allow_incomplete": bool(args.allow_incomplete),
                    "cells_found": len(seen),
                    "dirs_found": len(reports),
                    "pairs_found": sum(r["n_pairs"] for r in reports)}
    merge_outputs = [tables / f"cached_bend_{T}.md",
                     tables / f"cached_shares_{T}.tsv",
                     tables / f"cached_endpoint_{T}.tsv",
                     tables / f"cached_event_D_{T}.tsv",
                     tables / f"cached_floor_{T}.tsv",
                     tables / f"cached_plane_check_{T}.tsv",
                     figs / f"cached_D_profile_{T}.png",
                     figs / f"cached_event_D_{T}.png"]
    if resolve_reuse(json_path, merge_params, caps=("cells_found", "dirs_found",
                                                    "pairs_found"),
                     force=args.force, extra_outputs=merge_outputs) is not None:
        print(f"[skip] {json_path.name} and its tables already cover this merge "
              f"({len(seen)} cells, {merge_params['pairs_found']} pairs)")
        return

    # the sample table's 50 pairs per (method, budget) = 25 per dataset, drawn
    # across that dataset's three seed streams; a partial run states the same
    # denominator and shows the gap in n_pairs_dropped
    per_dataset_expected = int(params.get("pairs_per_dataset")
                               or PAIRS_PER_GROUP // len(datasets))
    pooled_expected = int(params.get("pairs_per_cell")
                          or per_dataset_expected * len(datasets))
    groups: list[dict[str, Any]] = []
    for m in METHODS:
        for k in KS:
            per_dataset: list[dict[str, Any]] = []
            pooled_pairs: list[dict[str, Any]] = []
            pooled_cells: list[str] = []
            mode_raw = ""
            for ds in datasets:
                rs = seen.get((m, k, ds), [])
                pairs = []
                for r in rs:
                    mode_raw = r["mode_raw"]
                    for p in r["pairs"]:
                        p = dict(p)
                        p["_cell"] = r["cell_dir"]
                        pairs.append(p)
                    pooled_cells.append(r["cell_dir"])
                if not rs:
                    continue
                per_dataset.append(summarise_cached_group(
                    pairs, method=m, budget=f"K{k}", dataset=ds, mode_raw=mode_raw,
                    cells=[r["cell_dir"] for r in rs], expected=per_dataset_expected))
                pooled_pairs.extend(pairs)
            if not pooled_pairs:
                continue
            pooled = summarise_cached_group(pooled_pairs, method=m, budget=f"K{k}",
                                            dataset=POOLED, mode_raw=mode_raw,
                                            cells=pooled_cells, expected=pooled_expected)
            # keyed off each group's OWN dataset, never zip(datasets, ...):
            # `per_dataset` skips a dataset with no cells, which under
            # --allow_incomplete would label the surviving half with the
            # missing half's name
            per_stream: dict[str, int] = {}
            for pr in pooled_pairs:
                parts = str(pr["_cell"]).split("_")      # <mode>_<dataset>_<K>_s<seed>
                key = f"{parts[-3]}_{parts[-1]}"
                per_stream[key] = per_stream.get(key, 0) + 1
            pooled["pooling"] = {
                "n_pairs_per_dataset": {g["dataset"]: g["n_pairs"] for g in per_dataset},
                "n_pairs_per_stream": dict(sorted(per_stream.items())),
                "n_directories": len(pooled_cells),
                "weighting": "pair-weighted " + " + ".join(
                    f"{g['n_pairs']} ({g['dataset']})" for g in per_dataset)
                    + f" across {len(per_dataset)} of {len(datasets)} datasets",
                "note": CAV_POOLING,
            }
            groups.append(pooled)
            groups.extend(per_dataset)

    rep: dict[str, Any] = {
        "schema": CACHED_SCHEMA, "stage": "merge", "plan_section": "3.9.3",
        "backbone": T, "d": sorted(dims)[0],
        "run_params": merge_params,
        "cells_expected": len(expected),
        "cells_expected_by_plan": EXPECTED_CELLS,
        "cells_found": len(seen),
        "cells_missing": [list(x) for x in missing],
        "shards": [{"cell_dir": r["cell_dir"], "file": r["_file"],
                    "n_pairs": r["n_pairs"],
                    "n_pairs_dropped": r.get("n_pairs_dropped"),
                    "n_pairs_dropped_reason": r.get("n_pairs_dropped_reason"),
                    **r["shard"]} for r in reports],
        "groups": groups,
        "floor": refside["streams"], "bf16_bound": refside["bf16_bound"],
        "floor_sources": refside["floor_sources"],
        "p1_store_expectations": refside.get("p1_store_expectations"),
        "p1_turn_window": refside.get("p1_turn_window"),
        "refside_file": str(refside_path(tables, T)),
        "refside_run_params": ref_params,
        "caveats": list(CACHED_CAVEATS),
        "completion_criteria": [{"item": a, "artefact": b} for a, b in CACHED_COMPLETION],
        "not_p7": ("P8 (block C) and P9 (the results documents and the CLAUDE.md index "
                   "rows) are not touched here; P7 writes only under "
                   "resources/video_full_trajectory/<T>/ and "
                   "docs/figures/video_full_trajectory/<T>/."),
    }
    atomic_write_json(json_path, rep)
    print(f"wrote {json_path} ({json_path.stat().st_size / 1e6:.2f} MB)")
    write_cached_tables(tables, rep)
    write_cached_markdown(tables / f"cached_bend_{T}.md", rep)
    write_cached_figures(figs, rep, True)


# --------------------------------------------------------------------------
# merge outputs: tables, markdown, figures
# --------------------------------------------------------------------------

SHARE_TSV_COLUMNS = (
    "method", "mode_raw", "K_nominal", "dataset", "state_label", "state",
    "state_range", "n_pairs", "n_pairs_expected", "n_pairs_dropped",
    "n_pairs_dropped_reason", "n_pairs_with_a_t2_frame", "frame_sources",
    "share_chord_med", "share_chord_p25", "share_chord_p75",
    "share_in_plane_med", "share_in_plane_p25", "share_in_plane_p75",
    "share_off_plane_med", "share_off_plane_p25", "share_off_plane_p75",
    "share_in_plane_of_offchord_med",
    "D_over_chord_ref_med", "D_over_norm_ref_med", "in_plane_iqr_at_this_state",
    "frame", "n_finite_share_chord",
)
ENDPOINT_TSV_COLUMNS = (
    "method", "mode_raw", "K_nominal", "dataset", "n_pairs", "n_pairs_expected",
    "n_pairs_dropped", "n_pairs_dropped_reason", "n_pairs_with_a_t2_frame",
    "frame_sources", "n_pairs_z_T_mismatch", "n_pairs_z_T_not_comparable",
    "D50_med", "D50_p25", "D50_p75",
    "D50_over_chord_ref_med", "D50_over_chord_ref_p25", "D50_over_chord_ref_p75",
    "D50_over_norm_ref_med", "D50_over_norm_ref_p25", "D50_over_norm_ref_p75",
    "chord_angle_vs_t2_deg_med", "chord_angle_vs_t2_deg_p25", "chord_angle_vs_t2_deg_p75",
    "chord_angle_vs_t3_deg_med", "ref_chord_t2_vs_t3_deg_med",
    "ref_chord_t2_vs_t3_deg_p75", "chord_angle_t2_minus_t3_deg_med",
    "plane_angle1_deg_med", "plane_angle1_deg_p25", "plane_angle1_deg_p75",
    "plane_angle2_deg_med", "t2_vs_t3_plane_angle_med_reference_side",
)
EVENT_TSV_COLUMNS = (
    "method", "K_nominal", "dataset", "view", "index", "stratum", "quantity",
    "median", "p25", "p75", "mean", "n_samples", "n_finite", "n_pairs",
    "n_samples_before_k0", "n_pairs_in_group", "n_pairs_expected",
    "n_pairs_dropped", "n_pairs_dropped_reason",
)
FLOOR_TSV_COLUMNS = (
    "scope", "dataset", "base_seed", "state", "n_paths",
    "floor_med", "floor_p25", "floor_p75",
    "floor_over_chord_ref_med", "floor_over_norm_ref_med",
    "share_chord_med", "share_in_plane_med", "share_off_plane_med",
    # every reading in this file is a bf16 reading, so the P1 bound travels
    # WITH the floor table instead of only in the json and the markdown
    "bf16_bound_over_chord_ref", "bf16_bound_route", "bf16_bound_applies_after_k0",
)
PLANE_TSV_COLUMNS = (
    "stream", "dataset", "base_seed", "n_paths",
    "first_angle_deg_med", "first_angle_deg_p5", "first_angle_deg_p95",
    "first_angle_deg_max", "second_angle_deg_med",
    # which frame each angle used: `matrix_reference` rows are NOT a clean
    # store-rounding reading (the frame is a different generation's)
    "n_paths_same_run_frame", "n_paths_matrix_reference_frame", "frame_sources",
    "what_it_is",
)


def _stat(entry: dict[str, Any] | None, field: str) -> Any:
    return None if entry is None else entry.get(field)


def write_cached_tables(tables: Path, rep: dict[str, Any]) -> None:
    T = rep["backbone"]
    # the T2-vs-T3 plane check is a property of the REFERENCES, so it is emitted
    # once and quoted next to the endpoint plane angle of every row
    plane_med: dict[str, Any] = {}
    for name, s in rep["floor"].items():
        plane_med[s["dataset"]] = _stat(s["plane_check"]["first_angle_deg"], "median")
    finite_plane = [v for v in plane_med.values() if v is not None]
    plane_med_pooled = float(np.median(finite_plane)) if finite_plane else None

    share_rows, endpoint_rows, event_rows = [], [], []
    for g in rep["groups"]:
        head = {"method": g["method"], "mode_raw": g["mode_raw"],
                "K_nominal": g["K_nominal"], "dataset": g["dataset"]}
        for label in ("k0_plus_1",) + tuple(str(n) for n in CACHED_LANDMARKS):
            lm = g["landmarks"][label]
            iqr = (None if _stat(lm["share_in_plane"], "p75") is None
                   or _stat(lm["share_in_plane"], "p25") is None
                   else lm["share_in_plane"]["p75"] - lm["share_in_plane"]["p25"])
            share_rows.append({
                **head, "state_label": label, "state": lm["state"],
                "state_range": None if lm["state_range"] is None
                else f"{lm['state_range'][0]}-{lm['state_range'][1]}",
                "n_pairs": g["n_pairs"], "n_pairs_expected": g["n_pairs_expected"],
                "n_pairs_dropped": g["n_pairs_dropped"],
                "n_pairs_dropped_reason": g.get("n_pairs_dropped_reason"),
                "n_pairs_with_a_t2_frame": g["n_pairs_with_a_t2_frame"],
                "frame_sources": ";".join(f"{k}={v}" for k, v in
                                          sorted(g["frame_sources"].items())),
                **{f"{f}_{stat}": _stat(lm[f], key)
                   for f in ("share_chord", "share_in_plane", "share_off_plane")
                   for stat, key in (("med", "median"), ("p25", "p25"), ("p75", "p75"))},
                "share_in_plane_of_offchord_med":
                    _stat(lm["share_in_plane_of_offchord"], "median"),
                "D_over_chord_ref_med": _stat(lm["D_over_chord_ref"], "median"),
                "D_over_norm_ref_med": _stat(lm["D_over_norm_ref"], "median"),
                "in_plane_iqr_at_this_state": iqr,
                "frame": "reference T2 (computed on the float32 path, stored float16)",
                "n_finite_share_chord": _stat(lm["share_chord"], "n_finite"),
            })
        ep = g["endpoint"]
        endpoint_rows.append({
            **head, "n_pairs": g["n_pairs"], "n_pairs_expected": g["n_pairs_expected"],
            "n_pairs_dropped": g["n_pairs_dropped"],
            "n_pairs_dropped_reason": g.get("n_pairs_dropped_reason"),
            "n_pairs_with_a_t2_frame": g["n_pairs_with_a_t2_frame"],
            "frame_sources": ";".join(f"{k}={v}" for k, v in
                                      sorted(g["frame_sources"].items())),
            "n_pairs_z_T_mismatch": g.get("n_pairs_z_T_mismatch"),
            "n_pairs_z_T_not_comparable": g.get("n_pairs_z_T_not_comparable"),
            **{f"{f}_{stat}": _stat(ep[f], key)
               for f in ("D50", "D50_over_chord_ref", "D50_over_norm_ref",
                         "chord_angle_vs_t2_deg", "plane_angle1_deg")
               for stat, key in (("med", "median"), ("p25", "p25"), ("p75", "p75"))},
            "chord_angle_vs_t3_deg_med": _stat(ep["chord_angle_vs_t3_deg"], "median"),
            "ref_chord_t2_vs_t3_deg_med": _stat(ep["ref_chord_t2_vs_t3_deg"], "median"),
            "ref_chord_t2_vs_t3_deg_p75": _stat(ep["ref_chord_t2_vs_t3_deg"], "p75"),
            "chord_angle_t2_minus_t3_deg_med":
                _stat(ep["chord_angle_t2_minus_t3_deg"], "median"),
            "plane_angle2_deg_med": _stat(ep["plane_angle2_deg"], "median"),
            "t2_vs_t3_plane_angle_med_reference_side":
                plane_med.get(g["dataset"], plane_med_pooled),
        })
        for row in g["events"]["per_step"] + g["events"]["event"]:
            for quantity in ("dD", "dD_over_chord_ref"):
                s = row[quantity]
                event_rows.append({
                    "method": g["method"], "K_nominal": g["K_nominal"],
                    "dataset": g["dataset"], "view": row["view"],
                    "index": row["index"], "stratum": row["stratum"],
                    "quantity": quantity, "median": s["median"], "p25": s["p25"],
                    "p75": s["p75"], "mean": s["mean"], "n_samples": s["n_samples"],
                    "n_finite": s["n_finite"], "n_pairs": row["n_pairs"],
                    "n_samples_before_k0": row["n_samples_before_k0"],
                    "n_pairs_in_group": g["n_pairs"],
                    "n_pairs_expected": g["n_pairs_expected"],
                    "n_pairs_dropped": g["n_pairs_dropped"],
                    "n_pairs_dropped_reason": g.get("n_pairs_dropped_reason"),
                })
    _write_tsv(tables / f"cached_shares_{T}.tsv", SHARE_TSV_COLUMNS, share_rows)
    _write_tsv(tables / f"cached_endpoint_{T}.tsv", ENDPOINT_TSV_COLUMNS, endpoint_rows)
    _write_tsv(tables / f"cached_event_D_{T}.tsv", EVENT_TSV_COLUMNS, event_rows)

    floor_rows: list[dict[str, Any]] = []
    pooled_pairs: list[dict[str, Any]] = []
    bound = rep["bf16_bound"]
    for name, s in rep["floor"].items():
        pooled_pairs.extend(s["pairs"])
        floor_rows.extend(_floor_rows(name, s["dataset"], s["base_seed"], s["pairs"],
                                      bound))
    floor_rows.extend(_floor_rows(POOLED, POOLED, None, pooled_pairs, bound))
    _write_tsv(tables / f"cached_floor_{T}.tsv", FLOOR_TSV_COLUMNS, floor_rows)

    plane_rows_out = [{
        "stream": name, "dataset": s["dataset"], "base_seed": s["base_seed"],
        "n_paths": s["plane_check"]["n"],
        "first_angle_deg_med": _stat(s["plane_check"]["first_angle_deg"], "median"),
        "first_angle_deg_p5": _stat(s["plane_check"]["first_angle_deg"], "p5"),
        "first_angle_deg_p95": _stat(s["plane_check"]["first_angle_deg"], "p95"),
        "first_angle_deg_max": _stat(s["plane_check"]["first_angle_deg"], "max"),
        "second_angle_deg_med": _stat(s["plane_check"]["second_angle_deg"], "median"),
        "n_paths_same_run_frame": s["plane_check"].get("n_paths_same_run_frame"),
        "n_paths_matrix_reference_frame":
            s["plane_check"].get("n_paths_matrix_reference_frame"),
        "frame_sources": ";".join(f"{k}={v}" for k, v in sorted(
            (s["plane_check"].get("frame_sources") or {}).items())),
        "what_it_is": s["plane_check"]["what_it_is"],
    } for name, s in rep["floor"].items()]
    _write_tsv(tables / f"cached_plane_check_{T}.tsv", PLANE_TSV_COLUMNS, plane_rows_out)


def _floor_rows(scope: str, dataset: str, base_seed: Any,
                pairs: list[dict[str, Any]],
                bound: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not pairs:
        return []
    bound = bound or {}
    chord_med = bound.get("chord_len_median")
    bound_rel = (bound["max_route"] / chord_med
                 if bound.get("max_route") is not None and chord_med
                 else None)
    bound_route = ("deviation" if bound.get("deviation_route", -1)
                   >= bound.get("spacing_route", -1) else "spacing") if bound else None
    D = np.asarray([_from_series(p["D"]) for p in pairs], dtype=np.float64)
    Dc = np.asarray([_from_series(p["D_over_chord_ref"]) for p in pairs], dtype=np.float64)
    Dn = np.asarray([_from_series(p["D_over_norm_ref"]) for p in pairs], dtype=np.float64)
    sh = np.asarray([[[np.nan if v is None else v for v in row] for row in p["shares"]]
                     for p in pairs], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med, p25, p75 = np.nanpercentile(D, [50, 25, 75], axis=0)
        medc = np.nanmedian(Dc, axis=0)
        medn = np.nanmedian(Dn, axis=0)
        meds = np.nanmedian(sh, axis=0)
    rows = []
    for n in range(D.shape[1]):
        rows.append({
            "scope": scope, "dataset": dataset, "base_seed": base_seed,
            "state": n, "n_paths": int(D.shape[0]),
            "floor_med": float(med[n]), "floor_p25": float(p25[n]),
            "floor_p75": float(p75[n]),
            "floor_over_chord_ref_med": float(medc[n]),
            "floor_over_norm_ref_med": float(medn[n]),
            "share_chord_med": float(meds[n, 0]),
            "share_in_plane_med": float(meds[n, 1]),
            "share_off_plane_med": float(meds[n, 2]),
            "bf16_bound_over_chord_ref": bound_rel,
            "bf16_bound_route": bound_route,
            "bf16_bound_applies_after_k0": bound.get("applies"),
        })
    return rows


def floor_profile(rep: dict[str, Any]) -> np.ndarray:
    """Pooled median reproduction floor in chord units, for the figure band."""
    pairs = [p for s in rep["floor"].values() for p in s["pairs"]]
    if not pairs:
        return np.full(N_STATES, np.nan)
    mat = np.asarray([_from_series(p["D_over_chord_ref"]) for p in pairs],
                     dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(mat, axis=0)


def write_cached_markdown(path: Path, rep: dict[str, Any]) -> None:
    T = rep["backbone"]
    pooled = [g for g in rep["groups"] if g["dataset"] == POOLED]
    L: list[str] = [
        f"# P7 — cache bend, T3 layer — {T} (plan section 3.9.3)", "",
        f"{rep['cells_found']}/{rep['cells_expected']} cells; "
        f"{sum(g['n_pairs'] for g in pooled)} pairs over {len(pooled)} (method, budget) "
        f"rows; d = {rep['d']}. Every row states its own denominator.", "",
        "Shards: " + ", ".join(
            f"{s['cell_dir']} ({s.get('cluster')}/{s.get('hostname')}, {s['n_pairs']} pairs)"
            for s in rep["shards"][:4])
        + (f" … {len(rep['shards'])} in total." if len(rep["shards"]) > 4 else ""), "",
        "## 1. D[50] and the endpoint angles (median over the pooled pairs)", "",
        "| method | K nominal | k realized (med) | pairs (used/expected) | "
        "per dataset | D[50]/chord_r | "
        "D[50]/‖Z^r[50]‖ | chord angle vs T2 (deg) | chord angle vs T3 (deg) | "
        "ref chord T2-vs-T3 (deg) | plane angle_1 (deg) | z_T mismatch |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in pooled:
        ep = g["endpoint"]
        split = " + ".join(f"{v} {k}" for k, v in
                           sorted((g.get("pooling") or {}).get(
                               "n_pairs_per_dataset", {}).items())) or "—"
        L.append(f"| {g['method']} | {g['K_nominal']} | "
                 f"{_fmt(_stat(g['n_cached'], 'median'))} | "
                 f"{g['n_pairs']}/{g['n_pairs_expected']} | {split} | "
                 f"{_fmt(_stat(ep['D50_over_chord_ref'], 'median'))} | "
                 f"{_fmt(_stat(ep['D50_over_norm_ref'], 'median'))} | "
                 f"{_fmt(_stat(ep['chord_angle_vs_t2_deg'], 'median'))} | "
                 f"{_fmt(_stat(ep['chord_angle_vs_t3_deg'], 'median'))} | "
                 f"{_fmt(_stat(ep['ref_chord_t2_vs_t3_deg'], 'median'))} | "
                 f"{_fmt(_stat(ep['plane_angle1_deg'], 'median'))} | "
                 f"{g.get('n_pairs_z_T_mismatch', 0)} |")
    # The nominal budget is the label the schedule search TARGETED; a gate that
    # saturates can realize fewer steps, and two nominal budgets can then land on
    # one schedule -- in which case their rows are the same operating point and
    # are identical by construction, not by coincidence.
    _realized = [(g["method"], g["K_nominal"], _stat(g["n_cached"], "median"))
                 for g in pooled]
    _off = [(m, k, v) for m, k, v in _realized
            if v is not None and abs(v - k) >= 1]
    if _off:
        L += ["", "**Realized k differs from the nominal budget in "
              f"{len(_off)} of {len(_realized)} rows**: "
              + "; ".join(f"{m} K{k} realizes {_fmt(v)}" for m, k, v in _off)
              + ". The nominal K is what the threshold search targeted; the "
                "realized k is what the gate actually did on these prompts."]
        _dup: dict[tuple[str, float], list[int]] = {}
        for m, k, v in _realized:
            if v is not None:
                _dup.setdefault((m, round(float(v), 6)), []).append(k)
        _same = sorted((mv, ks) for mv, ks in _dup.items() if len(ks) > 1)
        if _same:
            L += ["", "**Nominal budgets that collapse onto one realized "
                  "schedule**: "
                  + "; ".join(
                      f"{m} K" + "/K".join(str(x) for x in sorted(ks))
                      + f" all realize k = {_fmt(v)}" for (m, v), ks in _same)
                  + ". Those rows are the SAME operating point: identical "
                    "numbers across them are expected, and they must not be "
                    "read as independent budget points."]
    dropped = [g for g in pooled if g["n_pairs_dropped"]]
    if dropped:
        L += ["", "Rows below their expected denominator: " + "; ".join(
            f"{g['method']} K{g['K_nominal']} ({g['n_pairs']}/{g['n_pairs_expected']}, "
            f"{g.get('n_pairs_dropped_reason') or 'reason not recorded'})"
            for g in dropped) + "."]

    L += ["", "## 2. Direction of the difference at state 50 (shares of the total energy)",
          "", "| method | K | pairs | T2 frame (source counts) | along chord | "
          "in bend plane | off plane | in-plane share of the off-chord energy | "
          "in-plane IQR |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for g in pooled:
        lm = g["landmarks"]["50"]
        iqr = g["in_plane_iqr_trigger"]["iqr_at_state_50"]
        src = ", ".join(f"{k}={v}" for k, v in sorted(g["frame_sources"].items())) or "—"
        L.append(f"| {g['method']} | {g['K_nominal']} | {g['n_pairs']} | "
                 f"{g['n_pairs_with_a_t2_frame']} with a frame ({src}) | "
                 f"{_fmt(_stat(lm['share_chord'], 'median'))} | "
                 f"{_fmt(_stat(lm['share_in_plane'], 'median'))} | "
                 f"{_fmt(_stat(lm['share_off_plane'], 'median'))} | "
                 f"{_fmt(_stat(lm['share_in_plane_of_offchord'], 'median'))} | "
                 f"{_fmt(iqr)}{' (>0.3)' if iqr is not None and iqr > IN_PLANE_IQR_TRIGGER else ''} |")
    fallback = sum(g["frame_sources"].get("matrix_reference", 0) for g in pooled)
    no_frame = sum(g["frame_sources"].get("missing", 0) for g in pooled)
    L += ["", f"The full share table at n = k0+1, {', '.join(str(n) for n in CACHED_LANDMARKS)} "
          f"is `cached_shares_{T}.tsv`. Frame provenance over all "
          f"{sum(g['n_pairs'] for g in pooled)} pooled pairs: {fallback} used the "
          f"`matrix_reference` fallback (the T2 frame of the matrix-wave generation of the "
          f"same prompt, NOT of the reference path being differenced) and {no_frame} had "
          f"no frame at all and contribute no share. {CAV_FRAME}", ""]

    L += ["## 3. Prefix identity (states 0..k0)", "",
          "| method | K | pairs with a k0 | exactly zero | not exactly zero | "
          "no cache step | max |D| over pairs |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for g in pooled:
        pf = g["prefix"]
        L.append(f"| {g['method']} | {g['K_nominal']} | "
                 f"{pf.get('n_pairs_with_a_prefix', g['n_pairs'])} | "
                 f"{pf['n_pairs_exactly_zero']} | {pf['n_pairs_not_exactly_zero']} | "
                 f"{pf.get('n_pairs_without_a_cache_step', 0)} | "
                 f"{_fmt(pf['max_abs_D_over_pairs'])} |")
    entries = sorted((e for g in pooled for e in g["prefix"]["pairs_not_exactly_zero"]),
                     key=lambda e: -(e["max_abs_D"] or 0.0))
    named = [f"{e['cell']} idx {e['prompt_idx']} (max |D| = {_fmt(e['max_abs_D'])})"
             for e in entries[:MAX_NONZERO_PREFIX_ROWS]]
    bmax = rep["bf16_bound"].get("max_route")
    if named:
        line = (f"Pairs whose prefix is not exactly zero, worst first: {'; '.join(named)}"
                + (f" — {len(entries) - len(named)} further rows omitted here; the "
                   f"complete list is in `cached_bend_{T}.json` under each group's "
                   f"`prefix.pairs_not_exactly_zero`." if len(entries) > len(named)
                   else ".")
                + (f" Largest of these is {_fmt(entries[0]['max_abs_D'])} against a bf16 "
                   f"store bound of {_fmt(bmax)}." if bmax is not None else ""))
    else:
        line = "Every pair's prefix is exactly zero."
    no_k0 = [f"{e['cell']} idx {e['prompt_idx']}"
             for g in pooled for e in g["prefix"].get("pairs_without_a_cache_step", [])]
    L += ["", line]
    if no_k0:
        L += ["", f"{len(no_k0)} pair(s) cache nothing at all and therefore make no "
                  f"prefix-identity claim (they are NOT counted above): "
                  f"{'; '.join(no_k0[:MAX_NONZERO_PREFIX_ROWS])}."]
    L += ["", CAV_PREFIX, ""]

    L += ["## 4. Reproducibility floor and the bf16 lower bound", "",
          "| stream | dataset | floor pairs | median floor at n=50 (÷ chord_r) | "
          "T2-vs-T3 first principal angle: median / p95 / max (deg) |",
          "|---|---|---:|---:|---|"]
    for name, s in rep["floor"].items():
        fr = _floor_rows(name, s["dataset"], s["base_seed"], s["pairs"])
        last = fr[-1] if fr else {}
        pc = s["plane_check"]["first_angle_deg"]
        L.append(f"| {name} | {s['dataset']} | {s['n_floor_pairs']} | "
                 f"{_fmt(last.get('floor_over_chord_ref_med'))} | "
                 f"{_fmt(_stat(pc, 'median'))} / {_fmt(_stat(pc, 'p95'))} / "
                 f"{_fmt(pc.get('max'))} |")
    b = rep["bf16_bound"]
    L += ["", f"bf16 lower bound (P1 dump `{b.get('quant_dtype')}`, "
          f"{b.get('source')}): deviation route {_fmt(b['deviation_route'])}, spacing route "
          f"{_fmt(b['spacing_route'])}, analytic {_fmt(b['analytic'])} "
          f"(median chord {_fmt(b['chord_len_median'])}); in chord units the binding "
          f"route is {_fmt((b['max_route'] / b['chord_len_median']) if b.get('chord_len_median') else float('nan'))}, "
          f"which is the column `bf16_bound_over_chord_ref` of `cached_floor_{T}.tsv`. "
          f"{b['applies']}", ""]
    exp = rep.get("p1_store_expectations") or {}
    if exp:
        L += [f"Store cross-check expectations from the same P1 dump: chord_rel_med = "
              f"{_fmt(exp.get('chord_rel_med'))}, median magnitude_rel_med = "
              f"{_fmt(exp.get('magnitude_rel_med_median'))} (|max| "
              f"{_fmt(exp.get('magnitude_rel_med_absmax'))}). Each group's measured "
              f"`store_cross_checks` in the json is compared against these.", ""]
    tw = rep.get("p1_turn_window") or {}
    if tw:
        w = tw.get("min_readable_multistep_window_bf16")
        L += [f"P1's minimum readable multi-step turn window in bf16 for this backbone, "
              f"read from {tw.get('source')}: "
              f"{'none (no multi-step window clears the floor)' if w is None else f'w = {w}'}. "
              f"No P7 quantity uses it.", ""]
    L += [CAV_FLOOR, ""]

    L += ["## 5. Event-aligned Delta D", "",
          f"`cached_event_D_{T}.tsv` carries both views with `n_samples` per bucket. "
          f"{CAV_OVERLAP}", "", CAV_NEG_OFFSET, "", CAV_DELTA_NORM, "",
          "## 6. Caveats that travel with every number above", ""]
    L += [f"{i + 1}. {c}" for i, c in enumerate(rep["caveats"])]
    L += ["", "## 7. Completion criteria (plan section 6, P7 row)", "",
          "| item | artefact |", "|---|---|"]
    L += [f"| {c['item']} | `{c['artefact']}` |" for c in rep["completion_criteria"]]
    L += ["", rep["not_p7"], ""]
    atomic_write_text(path, "\n".join(L) + "\n")


def write_cached_figures(fig_dir: Path, rep: dict[str, Any], force: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    T = rep["backbone"]
    pooled = {(g["method"], g["K_nominal"]): g for g in rep["groups"]
              if g["dataset"] == POOLED}
    methods = [m for m in METHODS if any(key[0] == m for key in pooled)]
    if not methods:
        methods = sorted({key[0] for key in pooled})
    floor = floor_profile(rep)
    bound = rep["bf16_bound"]
    chord_med = bound.get("chord_len_median") or float("nan")
    states = np.arange(N_STATES)
    colours = {29: "C0", 37: "C1", 41: "C3"}

    out = fig_dir / f"cached_D_profile_{T}.png"
    if force or not output_complete(out):
        ncol = 3
        nrow = int(math.ceil(len(methods) / ncol)) or 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, m in zip(axes.ravel(), methods):
            for k in KS:
                g = pooled.get((m, k))
                if g is None:
                    continue
                prof = g["profiles"]["D_over_chord_ref"]
                med = _from_series(prof["median"])
                lo = _from_series(prof["p25"])
                hi = _from_series(prof["p75"])
                ax.plot(states, med, color=colours.get(k, "k"), lw=1.5,
                        label=f"K{k} (n={g['n_pairs']})")
                ax.fill_between(states, lo, hi, color=colours.get(k, "k"),
                                alpha=0.16, lw=0)
            ax.plot(states, floor, color="0.35", ls=":", lw=1.1,
                    label="run-to-run floor")
            if math.isfinite(chord_med) and chord_med > 0:
                ax.axhline(bound["max_route"] / chord_med, color="0.6", ls="--", lw=1.0,
                           label="bf16 bound (after k0)")
            ax.set_title(m, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(methods):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("state n (0 = z_T, 50 = result)")
        for row in axes:
            row[0].set_ylabel(r"$\|Z^c[n]-Z^r[n]\|\ /\ \mathrm{chord}_r$")
        axes[0][0].legend(fontsize=7)
        n_by_row = sorted({g["n_pairs"] for g in pooled.values()})
        n_txt = (f"{n_by_row[0]} pairs" if len(n_by_row) == 1
                 else f"{n_by_row[0]}-{n_by_row[-1]} pairs, per-row n in the legend")
        n_floor = sum(len(s_["pairs"]) for s_ in rep["floor"].values())
        fig.suptitle(f"{T} — per-state difference between a cached run and its own "
                     f"reference (median over {n_txt}; band = IQR)\n"
                     f"bf16 T3 store; the floor line is the zero-cache run-to-run "
                     f"difference of the same prompts (n = {n_floor})", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        atomic_savefig(fig, out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")

    out = fig_dir / f"cached_event_D_{T}.png"
    if force or not output_complete(out):
        ncol = 3
        nrow = int(math.ceil(len(methods) / ncol)) or 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow),
                                 sharex=True, squeeze=False)
        for ax, m in zip(axes.ravel(), methods):
            for stratum, colour, marker in (("cache", "C3", "o"), ("full", "C0", "s")):
                xs, ys, ns = [], [], []
                for k in KS:
                    g = pooled.get((m, k))
                    if g is None:
                        continue
                    for row in g["events"]["event"]:
                        if row["stratum"] != stratum:
                            continue
                        xs.append(row["index"] + (0.06 * (k - 37) / 4))
                        ys.append(row["dD_over_chord_ref"]["median"])
                        ns.append(row["dD_over_chord_ref"]["n_samples"])
                if xs:
                    ax.scatter(xs, ys, s=14, color=colour, marker=marker,
                               label=f"{stratum} (n={sum(ns)})")
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.set_title(m, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(methods):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("offset j from a cache step k")
        for row in axes:
            row[0].set_ylabel(r"median $\Delta D[k+j]\ /\ \mathrm{chord}_r$")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{T} — increment of the cached-vs-reference difference around each "
                     f"cache step\n(three budgets overlaid per method; windows overlap, "
                     f"so samples inside a bucket are not independent)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        atomic_savefig(fig, out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


def run_cached(args: argparse.Namespace) -> None:
    """Section 3.9.3 (P7): the T3 layer of the cache bend, in three stages.

    One pair = (cells_t3_rand50 directory, prompt idx): a cached run's whole
    path Z^c against the same-prompt reference path Z^r of `references_t3/`.
    The frozen sample table draws 25 (seed stream, prompt idx) per dataset out
    of that dataset's 3 streams x idx 0-119, the same 50 for every (method,
    budget) and both backbones, so a backbone carries 27 x 50 = 1,350 pairs
    spread over 162 directories of 5-12 prompts each.

    Three stages, all on the filesystem holding the directories:

        # one json per directory (the whole set, or any --only / --cells subset)
        OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \\
        python analysis/video_trajectory/latent_paths.py cached --stage cells \\
            --backbone hunyuan_video --data_root outputs \\
            --streams penguin599_s54 penguin599_s55 penguin599_s56 \\
                      vbench944_s42 vbench944_s43 vbench944_s44 \\
            --cluster site_a \\
            --out_tables resources/video_full_trajectory/hunyuan_video \\
            --out_figs  docs/figures/video_full_trajectory/hunyuan_video

        # once per backbone: reproduction floor + T2-vs-T3 + bf16 bound
        ... cached --stage refside --cluster site_a ...

        # after every per-cell json is in <out_tables>/cells/
        ... cached --stage merge ...

    Quantities, all against the reference's stored T2 frame (float32, computed
    in flight — see CAV_FRAME): `D[n] = ||Z^c[n] - Z^r[n]||` with both
    normalisations, the chord / in-plane / off-plane decomposition, the
    endpoint triple (D[50], chord angle, first principal angle between the two
    bend planes), the event-aligned increments `Delta D[k+1] - Delta D[k]`, the
    run-to-run reproduction floor and the P1 bf16 lower bound.

    Cost per pair: two paths resident (266 MB HYV / 346 MB Wan as float32) plus
    four float64 `[51, d]` arrays inside `plane_frame` for the cached bend
    plane, and `PathCache` evicts only AFTER the next path is materialised, so
    a third path is transiently live at the top of each iteration. Measured
    peak at the real HunyuanVideo d = 1,305,600 is ~2.9 GB, ~3.3 GB with the
    transient third path; Wan2.1 scales by ~1.30, i.e. ~3.8 / ~4.2 GB. Size
    the sbatch `--mem` from THAT (8 GB is a comfortable request), not from the
    two-path figure. 1,350 pairs is ~1-2 h of I/O. An sbatch job, never a login
    node.

    Not P7: no generation, no results document, no CLAUDE.md row (P8/P9); the
    in-plane-share IQR trigger of plan section 4.2 is REPORTED as a number and
    nothing is proposed or produced from it.
    """
    # `common._cap` reads 0 and -1 as "no cap", but `stored_indices` would slice
    # with them (`idx[:0]` = nothing, `idx[:-1]` = all but the last). The two
    # readings disagree, so neither value is accepted for the refside cap.
    if args.refside_idx_limit is not None and args.refside_idx_limit <= 0:
        raise SystemExit(
            f"--refside_idx_limit {args.refside_idx_limit} is not a prompt count. Pass a "
            f"positive number, or omit the flag.")
    if args.stage == "cells":
        run_cached_cells(args)
    elif args.stage == "refside":
        run_cached_refside(args)
    else:
        run_cached_merge(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs", help="clean block-A references: sections 3.10, 3.6, 3.8-2")
    p.add_argument("--backbone", required=True, choices=["hunyuan_video", "wan21"])
    p.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--out_tables", required=True)
    p.add_argument("--out_figs", required=True)
    p.add_argument("--t1_merged", default=None,
                   help="default <data_root>/<T>/matrix/trajectory/t1_merged.jsonl")
    p.add_argument("--p1_floor_dir", default=None,
                   help="directory holding p1_floor_bfloat16.json (default "
                        "docs/figures/video_full_trajectory/<T>/ in the repo, not "
                        "--out_figs)")
    p.add_argument("--p1_floor_json", default=None,
                   help="explicit path, overriding --p1_floor_dir")
    p.add_argument("--streams", nargs="+", default=list(DEFAULT_STREAMS))
    p.add_argument("--limit", type=int, default=None,
                   help="cap stored paths per stream (smoke runs)")
    p.add_argument("--sample", type=int, default=500,
                   help="both-different control pairs (plan section 3.10)")
    p.add_argument("--overlay_paths", type=int, default=30,
                   help="paths per stream for the 3-D overlay and the turn arrays")
    p.add_argument("--windows", type=int, nargs="+", default=list(TURN_WINDOWS))
    p.add_argument("--cache_paths", type=int, default=2,
                   help="paths held in memory at once (a pair needs 2)")
    p.add_argument("--cache_frames", type=int, default=8,
                   help="T2 frames held in memory at once")
    p.add_argument("--g_floor_source", choices=["deviation", "spacing", "max"],
                   default="deviation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="torch intra-op threads")
    p.add_argument("--force", action="store_true",
                   help="recompute outputs that already exist")

    c = sub.add_parser(
        "cached", help="section 3.9.3 (P7): the T3 layer of the cache bend",
        description=run_cached.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("--stage", choices=["cells", "refside", "merge"], default="cells",
                   help=f"cells = one json per {CELLS_T3_DIRNAME} directory on THIS "
                        "filesystem; "
                        "refside = the reproducibility floor + T2-vs-T3 plane check + "
                        "bf16 bound (needs references/ AND references_t3/, i.e. cluster "
                        "site_a, once per backbone); merge = concatenate the per-cell json "
                        "into the final tables and figures")
    c.add_argument("--backbone", required=True, choices=["hunyuan_video", "wan21"])
    c.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    c.add_argument("--out_tables", required=True,
                   help="per-cell json goes to <out_tables>/cells/, the merged tables "
                        "to <out_tables>/")
    c.add_argument("--out_figs", required=True)
    c.add_argument("--streams", nargs="+", default=list(DEFAULT_STREAMS),
                   help="the reference streams the cached runs are paired against")
    c.add_argument("--only", default=None,
                   help="glob matched against the path-layer directory basename "
                        "(e.g. 'seacache_*' or '*_penguin599_K29_s54')")
    c.add_argument("--cells", default=None,
                   help="file with one path-layer directory basename per line")
    c.add_argument("--sample_list", default=str(DEFAULT_SAMPLE_TABLE),
                   help="the frozen sample table naming the prompt indices each "
                        "cells_t3_rand50 directory holds (stage cells)")
    c.add_argument("--refside_streams", nargs="+", default=list(DEFAULT_STREAMS),
                   help="streams the reproduction floor is read on (stage refside); "
                        "they need a matrix-wave copy as well as a re-run stored path")
    c.add_argument("--refside_idx_limit", type=int, default=REFSIDE_IDX_LIMIT,
                   help="reference prompts per stream for the reproduction floor "
                        "(stage refside)")
    c.add_argument("--cache_paths", type=int, default=2,
                   help="paths held in memory at once (a pair needs 2)")
    c.add_argument("--p1_floor_dir", default=None,
                   help="directory holding p1_floor_bfloat16.json (default "
                        "docs/figures/video_full_trajectory/<T>/ in the repo)")
    c.add_argument("--p1_floor_json", default=None,
                   help="explicit path, overriding --p1_floor_dir")
    c.add_argument("--cluster", default=None,
                   help="label recorded with each per-cell json (e.g. site_a)")
    c.add_argument("--allow_incomplete", action="store_true",
                   help="merge with cells missing: the gap is recorded and every table "
                        f"then states a denominator below {PAIRS_PER_GROUP} pairs")
    c.add_argument("--workers", type=int, default=0, help="torch intra-op threads")
    c.add_argument("--force", action="store_true",
                   help="recompute outputs that already exist")

    args = ap.parse_args()
    if getattr(args, "workers", 0):
        try:
            import torch
            torch.set_num_threads(args.workers)
        except Exception:  # torch is only needed to open .pt files
            pass
    if args.cmd == "refs":
        run_refs(args)
    else:
        run_cached(args)


if __name__ == "__main__":
    main()
