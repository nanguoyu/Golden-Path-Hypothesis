#!/usr/bin/env python3
"""Construct golden-path family candidates from measured trajectory geometry.

Zero search, zero GPU: the schedules come out of the full-trajectory tables
(`resources/full_trajectory/tables_jsonl/`) and the 30 stored raw trajectories
(`resources/full_trajectory/latents_flux/`), per
`docs/research_plan_schedule_payload_cross.md` section 8.2.

Recipe
------
1. The cached signal is `s ~ dz/dsigma`; an order-`m` payload extrapolates it
   with a degree-`m` polynomial in sigma, so its local error over a sigma gap
   `g` from the anchor scales as `||z^(m+2)|| * g^(m+1)`. Hence

       payload `reuse`     (m=0) -> rho2 = ||z''||,  exponent 1
       payload `taylor_o1` (m=1) -> rho3 = ||z'''||, exponent 2

   (Section 8.2 item 3 writes the pair as `rho_{m+1} * g^{m+1}`; the derivative
   order of `z` and the gap exponent differ by one, and item 1 is the
   authority. Both are recorded per row in the manifest, so nothing is
   implicit.)

2. Profiles. Every quantity is a population mean over one table's rows, taken
   in the **sigma domain** (the real per-record sigma grid, never the step
   index). Each row is normalized by its own `chord_len` before averaging:
   that is the repo's scale-free convention for this data
   (`docs/full_trajectory_results.md` sections 3.1/5.1 report `dev/chord`), and
   without it the mean is tilted toward the large-norm prompts. Two components,
   from the planar structure of the path (top-2 chord-orthogonal EVR of the
   deviation = 0.95 on FLUX, 0.90-0.92 on Qwen, section 5.2): with `z ~ z0 + xi(sigma) * u + d(sigma) * v`,

       in-plane     |d^2 d_perp/dsigma^2|   (d_perp/chord, snapshot grid)
       along-chord  |d   speed  /dsigma  |   (speed = spacing/|dsigma|, /chord)
       rho2 = hypot(in-plane, along-chord)

   and one more sigma-derivative **per component** for rho3
   (`hypot(|d^3 d_perp|, |d^2 speed|)`). Differentiating the combined magnitude
   instead would return `d||z''||/dsigma`, which is not `||z'''||` and vanishes
   at extrema of `rho2`; the component form is the one that matches item 1.
   The along-chord component lives on the step-midpoint sigma grid and is
   carried to step index `n`; that half-step offset is an order of magnitude
   below the resolution the smoothing window imposes.

3. Smoothing. `docs/full_trajectory_results.md` section 2 (instrument audit):
   single-step turn angles and second differences sit at the bf16 quantization
   floor (measured/floor 0.9-1.3x for junctions 0-42), while the +-5-step
   turn-angle stencil clears it throughout (3.4-25.5x). That measurement is a
   symmetric stencil, not a Savitzky-Golay window, so it motivates rather than
   calibrates the choice below; the >= 5 / >= 7 minima are a deliberately
   conservative reading of it. Derivatives here are local-polynomial (SG)
   estimates on a non-uniform grid over a window of `--rho2_window` (>= 5)
   points for rho2 and `--rho3_window` (>= 7) for the noisier rho3, which is
   flagged low-confidence in the manifest. `--cross_check` re-measures the
   floor from the stored trajectories and reports measured/floor per window.

   Windows are clamped, not truncated, at the ends, so the first and last
   `window // 2` rows share a single fit and the profile is *constant* there: on
   the real table rho2 over steps 0-2 and 48-49, rho3 over steps 0-3 and 47-49.
   Within those bands the cost cannot rank one step against another. That costs
   the construction nothing -- steps 0 and 49 are forced full and the end region
   is the expensive one under any exponent -- but the profile does not resolve
   structure there and should not be read as if it did.

4. Cross-check. rho2 is compared against the direct coarse second difference of
   the 30 raw trajectories (full 262144-dim vectors, no component model), with
   a bf16 floor control built by re-quantizing each trajectory's exact
   straight-line projection (zero curvature by construction).

   What that comparison licenses: the component route reproduces the direct
   measure's *shape*, not its absolute scale. On the Parti table it sits at a
   global ~0.37x of the floor-subtracted direct curvature (manifest column
   `rho2_xcheck_ratio_median`) with Pearson 0.97, while Spearman is only 0.61 --
   the rank slack lives in the flat 20-35 plateau, where the DP is close to
   indifferent anyway. The residual mismatch is NOT a single global factor (the
   per-step ratio varies), so it can in principle move a schedule; what the
   cross-check establishes is agreement in magnitude ordering, not proportional
   equality. Component rho2 is a relative profile and must not be quoted as an
   absolute curvature magnitude.

5. Cost + DP. Gap-aware segment cost between consecutive full steps `a < b`:

       C(a, b) = sum_{n in (a, b)} w_A[n] * rho[n] * (sigma_a - sigma_n)^e

   with `w_A = 1` (the amplification weighting waits for the cluster S/A
   calibration). An exact DP over (last full step, fulls used) with forced full
   steps {0, num_steps-1} and `num_steps - K` fulls in total minimizes the sum
   of segment costs; `j_best_schedules` enumerates the J cheapest schedules by
   Lawler partition (disjoint, exhaustive prefix/branch decomposition of the
   solution set), not by any heuristic re-ranking.

6. Family. J-best plus exponent-perturbation variants (`e -> e +- 0.5`),
   ordered variant-major so the exponent uncertainty is spanned before the
   cost-landscape flatness, deduped by Hamming distance, 3-4 members per
   (model, K, order).

7. Self-check. A constructed family should NOT be strictly nested across K
   (the measured golden paths are not); strict nesting means the cost model has
   degenerated to a per-step ranking (which is exactly what `exponent = 0`
   gives, since the cost then stops depending on the anchor). The builder warns
   loudly in that case.

Outputs: `resources/sp_cross_schedules/<model>_k<K>_gpf_<variant>_<j>.txt`
(50-character bitstring, `1` = cached, `0` = full) plus rows appended to
`resources/sp_cross_schedules/gpf_manifest.tsv`, which records every
constructor parameter behind each file. Re-running replaces the rows of the
files it rewrites and leaves every other row alone.

Example:

    python analysis/build_golden_path_family.py --cross_check
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.build_sp_cross_schedules import check_schedule, write_tsv  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "resources" / "full_trajectory" / "tables_jsonl"
LATENTS_DIR = REPO_ROOT / "resources" / "full_trajectory" / "latents_flux"
OUT_DIR = REPO_ROOT / "resources" / "sp_cross_schedules"
MANIFEST_NAME = "gpf_manifest.tsv"
BUILDER = "analysis/build_golden_path_family.py"

INF = float("inf")


@dataclass(frozen=True)
class PayloadOrder:
    """One payload order = one (derivative order, gap exponent) pair."""

    name: str
    payload: str  # the SPX payload axis name this schedule is matched to
    rho_kind: str  # "rho2" | "rho3"
    exponent: float
    confidence: str


ORDERS = (
    PayloadOrder("reuse", "reuse", "rho2", 1.0, "high"),
    PayloadOrder("o1", "taylor_o1", "rho3", 2.0, "low"),
)
ORDER_BY_NAME = {order.name: order for order in ORDERS}
MANIFEST_FIELDS = (
    "model",
    "target_k",
    "name",
    "order",
    "payload",
    "rho_kind",
    "rho_source",
    "rho_dataset",
    "rho_rows",
    "rho_normalization",
    "rho_confidence",
    "smooth_window",
    "inplane_deriv_order",
    "alongchord_deriv_order",
    "amplification_weight",
    "exponent",
    "j_index",
    "member_rank",
    "cost",
    "num_steps",
    "n_full",
    "forced_full_steps",
    "cache_count",
    "k_vs_target",
    "min_hamming_in_family",
    "max_gap",
    "exponent_delta",
    "j_best",
    "members_per_order",
    "min_hamming_threshold",
    "rho2_xcheck_window",
    "rho2_xcheck_pearson",
    "rho2_xcheck_spearman",
    "rho2_xcheck_ratio_median",
    "full_steps",
    "schedule",
    "builder",
    "path",
)


# ---------------------------------------------------------------------------
# population profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """Population-mean geometry profiles of one full-trajectory table."""

    source: Path
    model: str
    dataset: str
    n_rows: int
    num_steps: int
    sigmas: np.ndarray  # [num_steps + 1] snapshot grid
    sigmas_mid: np.ndarray  # [num_steps] step-midpoint grid
    deviation: np.ndarray  # [num_steps + 1] mean d_perp / chord_len
    speed: np.ndarray  # [num_steps] mean (spacing / |dsigma|) / chord_len


def read_population(path: Path, *, num_steps: int) -> Population:
    """Mean `d_perp/chord` and `speed/chord` profiles of one table (jsonl)."""

    deviation = np.zeros(num_steps + 1, dtype=np.float64)
    speed = np.zeros(num_steps, dtype=np.float64)
    sigmas: np.ndarray | None = None
    d_sigma: np.ndarray | None = None
    model = dataset = ""
    rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row["num_steps"]) != int(num_steps):
                raise SystemExit(
                    f"{path}: row has num_steps={row['num_steps']}, expected {num_steps}"
                )
            row_sigmas = np.asarray(row["sigmas"], dtype=np.float64)
            if sigmas is None:
                sigmas = row_sigmas
                d_sigma = np.abs(np.diff(sigmas))
                model = str(row["model"])
                dataset = str(row["dataset"])
            elif not np.array_equal(row_sigmas, sigmas):
                raise SystemExit(f"{path}: rows disagree on the sigma grid")
            chord = float(row["chord_len"])
            if not chord > 0.0:
                raise SystemExit(f"{path}: non-positive chord_len in row {rows}")
            d_perp = np.asarray(row["d_perp"], dtype=np.float64)
            spacing = np.asarray(row["spacing"], dtype=np.float64)
            if d_perp.shape != (num_steps + 1,) or spacing.shape != (num_steps,):
                raise SystemExit(f"{path}: unexpected profile lengths in row {rows}")
            # Per-row scale normalization BEFORE averaging: see module docstring.
            deviation += d_perp / chord
            speed += spacing / d_sigma / chord
            rows += 1
    if sigmas is None or d_sigma is None or rows == 0:
        raise SystemExit(f"{path}: no rows")
    return Population(
        source=Path(path),
        model=model,
        dataset=dataset,
        n_rows=rows,
        num_steps=int(num_steps),
        sigmas=sigmas,
        sigmas_mid=0.5 * (sigmas[:-1] + sigmas[1:]),
        deviation=deviation / rows,
        speed=speed / rows,
    )


def derivative_operator(
    x: Sequence[float],
    *,
    order: int,
    window: int,
    degree: int | None = None,
) -> np.ndarray:
    """Local-polynomial derivative operator on a (possibly non-uniform) grid.

    Row `i` of the returned `[n, n]` matrix holds the weights of the `order`-th
    derivative at `x[i]`, obtained by least-squares fitting a polynomial of
    `degree` (default `order`, i.e. maximum smoothing) to the `window` samples
    nearest `i`. Windows are clamped, not truncated, at the ends, so every row
    averages over the same number of points. Because it is a linear operator it
    applies unchanged to a scalar profile and to a `[n, d]` trajectory.
    """

    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    order = int(order)
    window = int(window)
    degree = order if degree is None else int(degree)
    if degree < order:
        raise ValueError("degree must be at least the derivative order")
    if window < degree + 1:
        raise ValueError(f"window {window} is too small for degree {degree}")
    if window > n:
        raise ValueError(f"window {window} exceeds the grid length {n}")
    weights = np.zeros((n, n), dtype=np.float64)
    half = window // 2
    for i in range(n):
        lo = max(0, min(i - half, n - window))
        hi = lo + window
        vander = np.vander(x[lo:hi] - x[i], degree + 1, increasing=True)
        weights[i, lo:hi] = np.linalg.pinv(vander)[order] * math.factorial(order)
    return weights


@dataclass(frozen=True)
class RiskProfiles:
    """Per-step risk magnitudes on the step grid (`sigma[n]` = step-`n` input)."""

    rho2: np.ndarray
    rho3: np.ndarray
    inplane2: np.ndarray
    alongchord2: np.ndarray
    inplane3: np.ndarray
    alongchord3: np.ndarray
    rho2_window: int
    rho3_window: int

    def by_kind(self, kind: str) -> np.ndarray:
        return {"rho2": self.rho2, "rho3": self.rho3}[kind]

    def window_of(self, kind: str) -> int:
        return {"rho2": self.rho2_window, "rho3": self.rho3_window}[kind]


def risk_profiles(
    population: Population,
    *,
    rho2_window: int = 5,
    rho3_window: int = 7,
) -> RiskProfiles:
    """rho2 = ||z''||, rho3 = ||z'''|| from the two-component planar model."""

    if rho2_window < 5:
        raise ValueError("rho2_window must be >= 5 (bf16 floor, results doc section 2)")
    if rho3_window < 7:
        raise ValueError("rho3_window must be >= 7 (bf16 floor, results doc section 2)")
    steps = population.num_steps
    node = population.sigmas
    mid = population.sigmas_mid
    inplane2 = np.abs(
        derivative_operator(node, order=2, window=rho2_window) @ population.deviation
    )[:steps]
    along2 = np.abs(
        derivative_operator(mid, order=1, window=rho2_window) @ population.speed
    )
    inplane3 = np.abs(
        derivative_operator(node, order=3, window=rho3_window) @ population.deviation
    )[:steps]
    along3 = np.abs(
        derivative_operator(mid, order=2, window=rho3_window) @ population.speed
    )
    return RiskProfiles(
        rho2=np.hypot(inplane2, along2),
        rho3=np.hypot(inplane3, along3),
        inplane2=inplane2,
        alongchord2=along2,
        inplane3=inplane3,
        alongchord3=along3,
        rho2_window=int(rho2_window),
        rho3_window=int(rho3_window),
    )


# ---------------------------------------------------------------------------
# raw-trajectory cross-check
# ---------------------------------------------------------------------------


def straightened(trajectory: np.ndarray) -> np.ndarray:
    """Exact-zero-curvature control: project each point onto the chord line."""

    chord = trajectory[-1] - trajectory[0]
    progress = ((trajectory - trajectory[0]) @ chord) / float(chord @ chord)
    return trajectory[0] + np.outer(progress, chord)


def direct_curvature_profile(
    latent_paths: Sequence[Path],
    *,
    sigmas: np.ndarray,
    window: int,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean `||z''||/chord` of the stored trajectories and its bf16 floor.

    The measurement is the same coarse-window second derivative used for the
    profiles, applied to the full latent vectors (no component model). The floor
    is the same statistic on each trajectory's straight-line projection cast
    back through bf16: that path has exactly zero curvature, so whatever the
    estimator returns is quantization.
    """

    import torch  # local: the profile path never needs torch

    operator = derivative_operator(sigmas, order=2, window=window)
    measured = np.zeros(num_steps, dtype=np.float64)
    floor = np.zeros(num_steps, dtype=np.float64)
    for path in latent_paths:
        latents = torch.load(path, weights_only=True, map_location="cpu")
        trajectory = latents.float().numpy().astype(np.float64)
        if trajectory.shape[0] != num_steps + 1:
            raise SystemExit(f"{path}: expected {num_steps + 1} snapshots")
        chord_len = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        measured += np.linalg.norm(operator @ (trajectory / chord_len), axis=1)[:num_steps]
        control = torch.from_numpy(straightened(trajectory)).to(torch.bfloat16)
        control = control.float().numpy().astype(np.float64) / chord_len
        floor += np.linalg.norm(operator @ control, axis=1)[:num_steps]
    return measured / len(latent_paths), floor / len(latent_paths)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def cross_check_rho2(
    rho2: np.ndarray,
    latent_paths: Sequence[Path],
    *,
    sigmas: np.ndarray,
    windows: Sequence[int],
    num_steps: int,
    margin: int = 3,
) -> list[dict[str, Any]]:
    """Compare the two-component rho2 with the direct raw-trajectory measure."""

    interior = slice(margin, num_steps - margin)
    reports: list[dict[str, Any]] = []
    for window in windows:
        measured, floor = direct_curvature_profile(
            latent_paths, sigmas=sigmas, window=window, num_steps=num_steps
        )
        ratio = measured / np.maximum(floor, 1e-30)
        # Quadrature floor subtraction: the estimator sees signal + independent
        # quantization noise, so the signal estimate is sqrt(meas^2 - floor^2).
        corrected = np.sqrt(np.maximum(measured**2 - floor**2, 0.0))
        row: dict[str, Any] = {
            "window": int(window),
            "n_trajectories": len(latent_paths),
            "snr_min": float(ratio[interior].min()),
            "snr_median": float(np.median(ratio[interior])),
            "snr_max": float(ratio[interior].max()),
            "direct": measured,
            "floor": floor,
            "corrected": corrected,
        }
        for tag, other in (("raw", measured), ("floor_sub", corrected)):
            left = rho2[interior]
            right = other[interior]
            row[f"pearson_{tag}"] = float(np.corrcoef(left, right)[0, 1])
            row[f"log_pearson_{tag}"] = float(
                np.corrcoef(np.log(np.maximum(left, 1e-30)), np.log(np.maximum(right, 1e-30)))[0, 1]
            )
            row[f"spearman_{tag}"] = float(np.corrcoef(_rank(left), _rank(right))[0, 1])
            row[f"ratio_median_{tag}"] = float(np.median(left / np.maximum(right, 1e-30)))
        reports.append(row)
    return reports


# ---------------------------------------------------------------------------
# segment cost + exact DP + Lawler J-best
# ---------------------------------------------------------------------------


def segment_cost_matrix(
    rho: Sequence[float],
    sigmas: Sequence[float],
    *,
    exponent: float,
    amplification: Sequence[float] | None = None,
    max_gap: int | None = None,
) -> np.ndarray:
    """`C[a, b]` = cost of caching every step strictly between full steps a < b.

    `C[a, b] = sum_{n in (a, b)} w_A[n] * rho[n] * (sigma_a - sigma_n)^exponent`;
    entries with `b <= a` are `inf` (a schedule visits full steps in order).

    `max_gap` forbids segments longer than that many steps by leaving their
    entries at `inf`. Without it the DP is free to buy one very long jump with
    the savings from many short ones, and at the tightest budget it does: the
    K41 member built at exponent 1.5 placed a 19-step gap and its SSIM
    collapsed to 0.478 while every other K41 cell sat at 0.60-0.74. A cap also
    keeps the emitted schedules inside the span that competing cost models can
    even express - MeanCache's multigraph carries no edge longer than 15
    (`flux/meancache_calibrate.py`, `--max_edge_gap`).
    """

    rho = np.asarray(rho, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)[: rho.size]
    if rho.ndim != 1 or sigmas.size != rho.size:
        raise ValueError("rho and sigmas must be 1-D and cover the same steps")
    if np.any(np.diff(sigmas) >= 0.0):
        raise ValueError("sigmas must be strictly decreasing over the step grid")
    weight = (
        np.ones_like(rho)
        if amplification is None
        else np.asarray(amplification, dtype=np.float64)
    )
    steps = rho.size
    if max_gap is not None and int(max_gap) < 1:
        raise ValueError("max_gap must be at least 1 step")
    cost = np.full((steps, steps), INF, dtype=np.float64)
    for a in range(steps):
        gaps = np.zeros(steps, dtype=np.float64)
        gaps[a + 1 :] = (sigmas[a] - sigmas[a + 1 :]) ** float(exponent)
        terms = weight * rho * gaps
        # cumulative[k] = sum(terms[:k]), so C[a, b] = cumulative[b] - cumulative[a+1]
        cumulative = np.concatenate(([0.0], np.cumsum(terms)))
        cost[a, a + 1 :] = cumulative[a + 1 : steps] - cumulative[a + 1]
    # Guards float cancellation in the cumulative sums; inf stays inf.
    cost = np.maximum(cost, 0.0, out=cost)
    if max_gap is not None:
        limit = int(max_gap)
        for a in range(steps):
            cost[a, a + limit + 1 :] = INF
    return cost


def schedule_cost(cost: np.ndarray, full_steps: Sequence[int]) -> float:
    return float(
        sum(cost[a, b] for a, b in zip(full_steps[:-1], full_steps[1:]))
    )


def cost_to_go(cost: np.ndarray, *, n_full: int) -> np.ndarray:
    """`g[p, s]` = cheapest completion when full-step slot `p` sits at step `s`.

    Slot indices are 0-based; slot `n_full - 1` must land on the last step, so
    `g[n_full - 1, s]` is 0 at `s = num_steps - 1` and `inf` everywhere else.
    """

    steps = cost.shape[0]
    if n_full < 2 or n_full > steps:
        raise ValueError(f"n_full={n_full} is out of range for {steps} steps")
    g = np.full((n_full, steps), INF, dtype=np.float64)
    g[n_full - 1, steps - 1] = 0.0
    for slot in range(n_full - 2, -1, -1):
        for step in range(steps - 1):
            tail = cost[step, step + 1 :] + g[slot + 1, step + 1 :]
            if tail.size:
                g[slot, step] = tail.min()
    return g


def _complete(
    cost: np.ndarray, g: np.ndarray, *, slot: int, step: int, n_full: int
) -> list[int]:
    """Follow the argmin chain of `g` from an already-placed slot to the end."""

    steps = cost.shape[0]
    chain = [int(step)]
    for slot_next in range(slot + 1, n_full):
        tail = cost[step, step + 1 :] + g[slot_next, step + 1 :]
        if not tail.size or not np.isfinite(tail.min()):
            raise RuntimeError("cost_to_go promised a completion that does not exist")
        step = int(step + 1 + int(np.argmin(tail)))
        chain.append(step)
    if chain[-1] != steps - 1:
        raise RuntimeError("completion did not end on the forced last full step")
    return chain


def _node_solution(
    cost: np.ndarray,
    g: np.ndarray,
    *,
    prefix: tuple[int, ...],
    banned: frozenset[int],
    n_full: int,
) -> tuple[float, tuple[int, ...]] | None:
    """Cheapest schedule that extends `prefix` without using `banned` next."""

    slot = len(prefix)  # slot index of the element being chosen
    if slot >= n_full:
        return None
    last = prefix[-1]
    steps = cost.shape[0]
    best_value = INF
    best_step = -1
    for step in range(last + 1, steps):
        if step in banned:
            continue
        value = cost[last, step] + g[slot, step]
        if value < best_value:
            best_value = value
            best_step = step
    if best_step < 0 or not np.isfinite(best_value):
        return None
    tail = _complete(cost, g, slot=slot, step=best_step, n_full=n_full)
    solution = tuple(prefix) + tuple(tail)
    return schedule_cost(cost, solution), solution


def j_best_schedules(
    cost: np.ndarray, *, n_full: int, j_best: int
) -> list[tuple[float, tuple[int, ...]]]:
    """The `j_best` cheapest full-step sets, cheapest first (Lawler partition).

    A candidate node is `(prefix, banned)`: the schedules that agree with
    `prefix` on its slots and avoid `banned` in the next one. Popping a node's
    optimum `P` replaces it with the disjoint children "same prefix, also ban
    `P[len(prefix)]`" and, for every later slot `i`, "prefix `P[:i]`, ban
    `P[i]`". Those children plus `P` partition the node exactly, so every
    schedule is emitted once and in non-decreasing cost order.
    """

    if j_best < 1:
        raise ValueError("j_best must be >= 1")
    g = cost_to_go(cost, n_full=n_full)
    root = _node_solution(cost, g, prefix=(0,), banned=frozenset(), n_full=n_full)
    if root is None:
        return []
    heap: list[tuple[float, int, tuple[int, ...], frozenset[int], tuple[int, ...]]] = []
    counter = 0
    heapq.heappush(heap, (root[0], counter, (0,), frozenset(), root[1]))
    counter += 1
    found: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    while heap and len(found) < j_best:
        value, _, prefix, banned, solution = heapq.heappop(heap)
        if solution in seen:  # defensive; the partition is disjoint by construction
            raise RuntimeError("Lawler partition emitted a duplicate schedule")
        seen.add(solution)
        found.append((value, solution))
        slot = len(prefix)
        children = [(prefix, banned | {solution[slot]})]
        children.extend(
            (tuple(solution[:i]), frozenset({solution[i]}))
            for i in range(slot + 1, n_full - 1)
        )
        for child_prefix, child_banned in children:
            child = _node_solution(
                cost, g, prefix=child_prefix, banned=child_banned, n_full=n_full
            )
            if child is not None:
                heapq.heappush(
                    heap, (child[0], counter, child_prefix, child_banned, child[1])
                )
                counter += 1
    return found


def brute_force_schedules(
    cost: np.ndarray, *, n_full: int, j_best: int
) -> list[tuple[float, tuple[int, ...]]]:
    """Reference enumeration over every feasible schedule (tests / tiny grids)."""

    from itertools import combinations

    steps = cost.shape[0]
    solutions = [
        (schedule_cost(cost, (0, *middle, steps - 1)), (0, *middle, steps - 1))
        for middle in combinations(range(1, steps - 1), n_full - 2)
    ]
    solutions.sort(key=lambda item: (item[0], item[1]))
    return solutions[:j_best]


# ---------------------------------------------------------------------------
# family assembly
# ---------------------------------------------------------------------------


def schedule_bits(full_steps: Iterable[int], *, num_steps: int) -> str:
    fulls = set(int(step) for step in full_steps)
    return "".join("0" if step in fulls else "1" for step in range(int(num_steps)))


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("bitstrings must be the same length")
    return sum(a != b for a, b in zip(left, right))


def exponent_tag(exponent: float) -> str:
    scaled = round(float(exponent) * 10.0)
    if abs(scaled - float(exponent) * 10.0) > 1e-9:
        raise ValueError(f"exponent {exponent} is not a multiple of 0.1")
    return f"e{scaled:02d}"


@dataclass(frozen=True)
class Member:
    model: str
    target_k: int
    order: PayloadOrder
    exponent: float
    j_index: int
    rank: int
    cost: float
    full_steps: tuple[int, ...]
    bits: str

    @property
    def name(self) -> str:
        return f"gpf_{self.order.name}_{exponent_tag(self.exponent)}_{self.j_index}"


def candidate_exponents(base: float, delta: float) -> tuple[float, ...]:
    """Base exponent first, then the two perturbations (`m + 1 -> m + 1 +- 0.5`)."""

    return (base, base - delta, base + delta)


def build_family(
    rho: Sequence[float],
    sigmas: Sequence[float],
    *,
    model: str,
    target_k: int,
    order: PayloadOrder,
    num_steps: int,
    exponent_delta: float = 0.5,
    j_best: int = 4,
    members: int = 4,
    min_hamming: int = 4,
    max_gap: int | None = None,
) -> list[Member]:
    """J-best x exponent-perturbation candidates, Hamming-deduped to `members`.

    Candidates are ordered variant-major (every variant's `j = 1` before any
    `j = 2`) so a family of 3-4 spans the exponent uncertainty first and the
    flatness of the cost landscape second.
    """

    n_full = int(num_steps) - int(target_k)
    if n_full < 2:
        raise ValueError(f"K={target_k} leaves fewer than 2 full steps")
    pool: list[tuple[int, int, float, float, tuple[int, ...]]] = []
    for variant_index, exponent in enumerate(
        candidate_exponents(order.exponent, exponent_delta)
    ):
        cost = segment_cost_matrix(rho, sigmas, exponent=exponent, max_gap=max_gap)
        for j_index, (value, full_steps) in enumerate(
            j_best_schedules(cost, n_full=n_full, j_best=j_best), start=1
        ):
            pool.append((j_index, variant_index, exponent, value, full_steps))
    pool.sort(key=lambda item: (item[0], item[1]))

    kept: list[Member] = []
    for j_index, _, exponent, value, full_steps in pool:
        if len(kept) >= members:
            break
        bits = schedule_bits(full_steps, num_steps=num_steps)
        if any(hamming(bits, member.bits) < min_hamming for member in kept):
            continue
        kept.append(
            Member(
                model=model,
                target_k=int(target_k),
                order=order,
                exponent=float(exponent),
                j_index=int(j_index),
                rank=len(kept) + 1,
                cost=float(value),
                full_steps=tuple(int(step) for step in full_steps),
                bits=bits,
            )
        )
    return kept


def nesting_report(members: Sequence[Member]) -> dict[str, Any]:
    """Self-check (plan 8.2 item 6): a family must not be strictly nested in K.

    Chains are matched across K by (order, exponent, j index). A chain is
    strictly nested when every smaller-K member's cached set is a proper subset
    of the next one's -- the signature of a cost model that has collapsed into a
    per-step ranking.
    """

    chains: dict[tuple[str, float, int], list[Member]] = {}
    for member in members:
        chains.setdefault(
            (member.order.name, member.exponent, member.j_index), []
        ).append(member)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(chains.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda member: member.target_k)
        nested = True
        escapes = 0
        for lower, upper in zip(group[:-1], group[1:]):
            low = {i for i, bit in enumerate(lower.bits) if bit == "1"}
            high = {i for i, bit in enumerate(upper.bits) if bit == "1"}
            escapes += len(low - high)
            nested = nested and low < high
        rows.append(
            {
                "order": key[0],
                "exponent": key[1],
                "j_index": key[2],
                "budgets": [member.target_k for member in group],
                "strictly_nested": bool(nested),
                "cached_steps_lost": int(escapes),
            }
        )
    return {
        "chains": rows,
        "n_chains": len(rows),
        "n_nested": sum(1 for row in rows if row["strictly_nested"]),
        "degenerate": bool(rows) and all(row["strictly_nested"] for row in rows),
    }


# ---------------------------------------------------------------------------
# manifest + emission
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def merge_manifest(
    existing: Sequence[Mapping[str, str]], fresh: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Replace the rows of every (model, K, order) group just rebuilt.

    Returns `(rows, dropped)`.

    Two failure modes to avoid, both seen in practice:

    * Keying on (model, name) alone would make a partial rerun
      (`--budgets 29`) delete the other budgets' rows, since member names
      carry no budget and repeat verbatim across them.
    * Keying on (model, K, name) alone leaves orphans behind when a rebuild
      changes the member SET rather than the members: adding the `--max_gap`
      cap replaced `gpf_o1_e20_4` with `gpf_o1_e15_3` at K41 and the old row
      and file survived, which matters because
      `analysis/build_gpf_screen_cells.py` selects members by manifest rank.

    So the unit of replacement is the (model, target_k, order) group: every
    prior row of a group present in `fresh` is dropped, and the caller deletes
    the corresponding files.
    """

    def group(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("model", "")),
            str(row.get("target_k", "")),
            str(row.get("order", "")),
        )

    rebuilt = {group(row) for row in fresh}
    kept, dropped = [], []
    for row in existing:
        (kept if group(row) not in rebuilt else dropped).append(dict(row))
    return kept + [dict(row) for row in fresh], dropped


def emit_member(
    member: Member,
    *,
    out_dir: Path,
    num_steps: int,
    forced_full: Sequence[int],
) -> Path:
    label = f"{member.model}_k{member.target_k}_{member.name}"
    check_schedule(
        member.bits,
        num_steps=num_steps,
        cache_count=member.target_k,
        label=label,
    )
    for step in forced_full:
        if member.bits[int(step)] != "0":
            raise SystemExit(f"{label}: step {step} must be a full step")
    path = Path(out_dir) / f"{label}.txt"
    path.write_text(member.bits + "\n", encoding="utf-8")
    return path


def manifest_row(
    member: Member,
    *,
    population: Population,
    profiles: RiskProfiles,
    path: Path,
    num_steps: int,
    forced_full: Sequence[int],
    family: Sequence[Member],
    cross_check: Mapping[str, Any] | None,
    knobs: Mapping[str, Any],
) -> dict[str, Any]:
    others = [other for other in family if other is not member]
    return {
        "model": member.model,
        "target_k": member.target_k,
        "name": member.name,
        "order": member.order.name,
        "payload": member.order.payload,
        "rho_kind": member.order.rho_kind,
        "rho_source": str(population.source.relative_to(REPO_ROOT))
        if population.source.is_relative_to(REPO_ROOT)
        else str(population.source),
        "rho_dataset": population.dataset,
        "rho_rows": population.n_rows,
        "rho_normalization": "per_row_chord_len",
        "rho_confidence": member.order.confidence,
        "smooth_window": profiles.window_of(member.order.rho_kind),
        "inplane_deriv_order": 2 if member.order.rho_kind == "rho2" else 3,
        "alongchord_deriv_order": 1 if member.order.rho_kind == "rho2" else 2,
        "amplification_weight": "1",
        "exponent": member.exponent,
        "j_index": member.j_index,
        "member_rank": member.rank,
        "cost": f"{member.cost:.10g}",
        "num_steps": int(num_steps),
        "n_full": int(num_steps) - member.target_k,
        "forced_full_steps": ",".join(str(int(step)) for step in forced_full),
        "cache_count": member.bits.count("1"),
        "k_vs_target": member.bits.count("1") - member.target_k,
        "min_hamming_in_family": (
            min(hamming(member.bits, other.bits) for other in others) if others else ""
        ),
        # The construction knobs that decide the bits. Plan section 8.2 promises
        # the manifest records ALL construction parameters, and max_gap is the
        # proven counterexample: it changed the K41 e15 member, so a manifest
        # without it cannot say which knobs produced its own rows.
        "max_gap": knobs["max_gap"] if knobs["max_gap"] is not None else "",
        "exponent_delta": knobs["exponent_delta"],
        "j_best": knobs["j_best"],
        "members_per_order": knobs["members"],
        "min_hamming_threshold": knobs["min_hamming"],
        "rho2_xcheck_window": (cross_check or {}).get("window", ""),
        "rho2_xcheck_pearson": (cross_check or {}).get("pearson_floor_sub", ""),
        "rho2_xcheck_spearman": (cross_check or {}).get("spearman_floor_sub", ""),
        "rho2_xcheck_ratio_median": (cross_check or {}).get("ratio_median_floor_sub", ""),
        "full_steps": ",".join(str(step) for step in member.full_steps),
        "schedule": member.bits,
        "builder": BUILDER,
        "path": str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else str(path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flux")
    parser.add_argument(
        "--table",
        type=Path,
        default=TABLES_DIR / "full_traj_flux_parti_full.jsonl",
        help="population table; default = the dataset the SPX runs on (Parti)",
    )
    parser.add_argument("--latents_dir", type=Path, default=LATENTS_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--manifest_name", default=MANIFEST_NAME)
    parser.add_argument("--budgets", type=int, nargs="+", default=[29, 37, 41])
    parser.add_argument("--orders", nargs="+", default=[order.name for order in ORDERS])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--rho2_window", type=int, default=5)
    parser.add_argument("--rho3_window", type=int, default=7)
    parser.add_argument("--exponent_delta", type=float, default=0.5)
    parser.add_argument("--j_best", type=int, default=4)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--min_hamming", type=int, default=4)
    parser.add_argument(
        "--max_gap",
        type=int,
        default=15,
        help="longest run of consecutive cached steps a segment may cover "
        "(default 15, matching MeanCache's multigraph edge span); 0 disables",
    )
    parser.add_argument(
        "--cross_check",
        action="store_true",
        help="re-measure rho2 against the 30 stored trajectories + bf16 floor",
    )
    parser.add_argument("--cross_check_windows", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--cross_check_limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    num_steps = int(args.num_steps)
    forced_full = (0, num_steps - 1)
    population = read_population(args.table, num_steps=num_steps)
    profiles = risk_profiles(
        population, rho2_window=args.rho2_window, rho3_window=args.rho3_window
    )
    print(
        f"[gpf] rho source {population.source.name} dataset={population.dataset} "
        f"rows={population.n_rows} windows rho2={profiles.rho2_window} "
        f"rho3={profiles.rho3_window}"
    )

    cross_check: dict[str, Any] | None = None
    if args.cross_check:
        paths = sorted(Path(args.latents_dir).glob("latents_*.pt"))
        if args.cross_check_limit:
            paths = paths[: int(args.cross_check_limit)]
        if not paths:
            raise SystemExit(f"no trajectories under {args.latents_dir}")
        reports = cross_check_rho2(
            profiles.rho2,
            paths,
            sigmas=population.sigmas,
            windows=args.cross_check_windows,
            num_steps=num_steps,
        )
        for report in reports:
            print(
                f"[gpf] cross-check w={report['window']:>2} "
                f"n={report['n_trajectories']} "
                f"snr(meas/bf16-floor) min={report['snr_min']:.2f} "
                f"med={report['snr_median']:.2f} max={report['snr_max']:.2f} | "
                f"raw: r={report['pearson_raw']:.3f} rho_s={report['spearman_raw']:.3f} "
                f"ratio={report['ratio_median_raw']:.3f} | "
                f"floor-sub: r={report['pearson_floor_sub']:.3f} "
                f"logr={report['log_pearson_floor_sub']:.3f} "
                f"rho_s={report['spearman_floor_sub']:.3f} "
                f"ratio={report['ratio_median_floor_sub']:.3f}"
            )
        cross_check = min(reports, key=lambda report: abs(report["window"] - args.rho2_window))

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    fresh: list[dict[str, Any]] = []
    everyone: list[Member] = []
    for target_k in args.budgets:
        for order_name in args.orders:
            order = ORDER_BY_NAME[order_name]
            family = build_family(
                profiles.by_kind(order.rho_kind),
                population.sigmas,
                model=args.model,
                target_k=int(target_k),
                order=order,
                num_steps=num_steps,
                exponent_delta=args.exponent_delta,
                j_best=args.j_best,
                max_gap=(int(args.max_gap) if int(args.max_gap) > 0 else None),
                members=args.members,
                min_hamming=args.min_hamming,
            )
            if len(family) < 3:
                print(
                    f"[gpf][WARN] {args.model} K{target_k} {order.name}: only "
                    f"{len(family)} distinct members at min_hamming="
                    f"{args.min_hamming} (plan asks for 3-4)"
                )
            everyone.extend(family)
            for member in family:
                path = out_dir / f"{member.model}_k{member.target_k}_{member.name}.txt"
                if not args.dry_run:
                    path = emit_member(
                        member,
                        out_dir=out_dir,
                        num_steps=num_steps,
                        forced_full=forced_full,
                    )
                fresh.append(
                    manifest_row(
                        member,
                        population=population,
                        profiles=profiles,
                        path=path,
                        num_steps=num_steps,
                        forced_full=forced_full,
                        family=family,
                        cross_check=cross_check,
                        knobs={
                            "max_gap": (int(args.max_gap) if int(args.max_gap) > 0
                                        else None),
                            "exponent_delta": args.exponent_delta,
                            "j_best": args.j_best,
                            "members": args.members,
                            "min_hamming": args.min_hamming,
                        },
                    )
                )
                print(
                    f"{path.name}\tK={member.bits.count('1')}\trank={member.rank}"
                    f"\te={member.exponent}\tj={member.j_index}"
                    f"\tcost={member.cost:.6g}\tfulls={list(member.full_steps)}"
                )

    report = nesting_report(everyone)
    for row in report["chains"]:
        print(
            f"[gpf] nesting {row['order']} e={row['exponent']} j={row['j_index']} "
            f"K{row['budgets']}: strictly_nested={row['strictly_nested']} "
            f"cached_steps_lost={row['cached_steps_lost']}"
        )
    print(
        f"[gpf] non-nesting self-check: {report['n_chains'] - report['n_nested']}"
        f"/{report['n_chains']} chains are NOT strictly nested"
    )
    if report["degenerate"]:
        print(
            "[gpf][WARN] every cross-K chain is strictly nested: the cost model has "
            "degenerated to a per-step ranking (check the exponent and rho profile)"
        )

    if args.dry_run:
        print(f"[gpf] dry run: {len(fresh)} schedules NOT written")
        return 0
    manifest_path = out_dir / args.manifest_name
    rows, dropped = merge_manifest(read_manifest(manifest_path), fresh)
    kept_files = {f"{row['model']}_k{row['target_k']}_{row['name']}.txt" for row in rows}
    for row in dropped:
        name = f"{row['model']}_k{row['target_k']}_{row['name']}.txt"
        stale = out_dir / name
        if name not in kept_files and stale.is_file():
            if not args.dry_run:
                stale.unlink()
            print(f"[gpf] removed superseded member {name}")
    write_tsv(manifest_path, rows, MANIFEST_FIELDS)
    print(f"[gpf] wrote {len(fresh)} schedules; manifest {manifest_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
