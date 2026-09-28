"""Benchmark schedule-search algorithms against the K41 exhaustive ground truth.

The exhaustive K41 run scored every feasible FLUX 50-step / K=41 schedule: the
four forced full steps are {0, 1, 2, 49} and the remaining five full steps are
chosen freely among positions 3..48, so the space is C(46, 5) = 1,370,754
schedules, each with a four-pair mean PSNR.  That table is a complete oracle,
which makes it possible to ask a question no GPU experiment can answer directly:
given a budget of B schedule evaluations, how close to the global optimum does a
search algorithm get, and does a smarter algorithm buy anything over drawing B
schedules at random?

Every oracle call here stands for one real evaluation, which on a GPU costs the
four generations behind one mean-PSNR number.  Repeat visits to an already
evaluated schedule are memoized and do not consume budget, matching what a real
search loop would do.

Schedules are addressed by their rank in the exhaustive table, which is the
lexicographic index of the five chosen positions among 0..45 (position p means
step p + 3).  The script verifies that identity against the table before running
anything.

Usage:

    python analysis/golden_path_search_bench.py \
        --table /data/.../exhaustive_k41/formal_combined/merged/merged.tsv.gz \
        --summary resources/exhaustive_k41/formal_results/summary.json \
        --manifest resources/exhaustive_k41/formal_results/candidate_manifest.tsv \
        --out_json resources/search_bench/results.json \
        --out_fig resources/search_bench/curves.png
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import time
import zlib
from math import comb
from pathlib import Path

import numpy as np

N_POS = 46  # variable positions, step 3 .. step 48
K_PICK = 5  # variable full steps
SPACE_SIZE = comb(N_POS, K_PICK)  # 1_370_754
STEP_OFFSET = 3  # position index p corresponds to timestep p + 3

BUDGETS = (50, 200, 1000, 5000)
THRESHOLDS = (0.05, 0.1, 0.2)


# --------------------------------------------------------------------------
# rank <-> combination
# --------------------------------------------------------------------------


def _rank_tables() -> np.ndarray:
    """Prefix sums used to map a sorted combination to its lexicographic rank."""
    table = np.zeros((K_PICK, N_POS + 1), dtype=np.int64)
    for i in range(K_PICK):
        acc = 0
        for j in range(N_POS):
            table[i, j] = acc
            acc += comb(N_POS - 1 - j, K_PICK - 1 - i) if N_POS - 1 - j >= 0 else 0
        table[i, N_POS] = acc
    return table


_PREFIX = _rank_tables()


def ranks_of(combos: np.ndarray) -> np.ndarray:
    """Lexicographic ranks of an (N, 5) array of ascending position combinations."""
    combos = np.asarray(combos, dtype=np.int64)
    out = np.zeros(combos.shape[0], dtype=np.int64)
    for i in range(K_PICK):
        hi = combos[:, i]
        lo = combos[:, i - 1] + 1 if i > 0 else np.zeros_like(hi)
        out += _PREFIX[i][hi] - _PREFIX[i][lo]
    return out


def build_all_combos() -> np.ndarray:
    """All C(46, 5) combinations, row r being the combination of rank r."""
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(N_POS), K_PICK)),
        dtype=np.int8,
        count=SPACE_SIZE * K_PICK,
    )
    return flat.reshape(SPACE_SIZE, K_PICK)


# --------------------------------------------------------------------------
# table loading and verification
# --------------------------------------------------------------------------


def load_table(path: Path, combos: np.ndarray) -> tuple[np.ndarray, dict]:
    """Read the merged exhaustive table into a rank-indexed score array."""
    scores = np.full(SPACE_SIZE, np.nan, dtype=np.float64)
    seen = np.zeros(SPACE_SIZE, dtype=bool)
    listed = np.zeros((SPACE_SIZE, K_PICK), dtype=np.int8)
    n_rows = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        i_rank = header.index("rank")
        i_steps = header.index("variable_full_steps")
        i_mean = header.index("mean_psnr_db")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            rank = int(fields[i_rank])
            scores[rank] = float(fields[i_mean])
            seen[rank] = True
            n_rows += 1
            listed[rank] = [int(x) - STEP_OFFSET for x in fields[i_steps].split(",")]
    mismatch = int((listed != combos).any(axis=1).sum()) if seen.all() else -1
    info = {
        "n_rows": n_rows,
        "space_size": SPACE_SIZE,
        "ranks_complete": bool(seen.all()),
        "combination_rank_mismatches": mismatch,
        "any_nan": bool(np.isnan(scores).any()),
    }
    return scores, info


def verify(
    scores: np.ndarray,
    info: dict,
    summary: dict,
    manifest_rows: list,
    combos: np.ndarray,
) -> dict:
    """Cross-check the loaded table against the recorded run summary."""
    best_rank = int(np.argmax(scores))
    best_score = float(scores[best_rank])
    summary_best = summary.get("best_mean", {})
    manifest_ok = 0
    manifest_bad = 0
    for row in manifest_rows:
        rank = int(row["rank"])
        if abs(scores[rank] - float(row["mean_psnr_db"])) <= 1e-9:
            manifest_ok += 1
        else:
            manifest_bad += 1
    out = dict(info)
    out.update(
        {
            "argmax_rank": best_rank,
            "argmax_mean_psnr_db": best_score,
            "summary_best_rank": summary_best.get("rank"),
            "summary_best_psnr_db": summary_best.get("psnr_db"),
            "argmax_matches_summary": (
                summary_best.get("rank") == best_rank
                and abs(float(summary_best.get("psnr_db", -1)) - best_score) <= 1e-9
            ),
            "manifest_rows_matching": manifest_ok,
            "manifest_rows_disagreeing": manifest_bad,
            "worst_mean_psnr_db": float(np.min(scores)),
            "median_mean_psnr_db": float(np.median(scores)),
            "best_schedule_full_steps": [0, 1, 2]
            + [int(p) + STEP_OFFSET for p in combos[best_rank]]
            + [49],
        }
    )
    return out


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------


class Oracle:
    """Budgeted access to the exhaustive table, one call per unseen schedule."""

    def __init__(self, scores: np.ndarray, budget: int) -> None:
        self.scores = scores
        self.budget = budget
        self.seen: dict[int, float] = {}
        self.trace = np.empty(budget, dtype=np.float64)
        self.calls = 0
        self.best = -math.inf
        self.best_rank = -1

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def __call__(self, rank: int) -> float:
        rank = int(rank)
        hit = self.seen.get(rank)
        if hit is not None:
            return hit
        if self.exhausted:
            raise BudgetExhausted
        value = float(self.scores[rank])
        self.seen[rank] = value
        if value > self.best:
            self.best = value
            self.best_rank = rank
        self.trace[self.calls] = self.best
        self.calls += 1
        return value


class BudgetExhausted(Exception):
    pass


# --------------------------------------------------------------------------
# neighbourhood
# --------------------------------------------------------------------------


def neighbour_ranks(combo: np.ndarray) -> np.ndarray:
    """Ranks of every schedule reached by moving one full step to a free slot."""
    combo = np.asarray(combo, dtype=np.int64)
    free = np.setdiff1d(np.arange(N_POS, dtype=np.int64), combo, assume_unique=False)
    rows = np.repeat(combo[None, :], K_PICK * free.size, axis=0)
    idx = np.repeat(np.arange(K_PICK), free.size)
    rows[np.arange(rows.shape[0]), idx] = np.tile(free, K_PICK)
    rows.sort(axis=1)
    return ranks_of(rows)


def local_neighbour_ranks(combo: np.ndarray, window: int) -> np.ndarray:
    """BudCache's windowed neighbourhood: destinations within +/- window steps."""
    combo = np.asarray(combo, dtype=np.int64)
    free = np.setdiff1d(np.arange(N_POS, dtype=np.int64), combo)
    rows = []
    for source in free:
        for i, held in enumerate(combo):
            if abs(int(held) - int(source)) > window:
                continue
            row = combo.copy()
            row[i] = source
            rows.append(np.sort(row))
    if not rows:
        return np.empty(0, dtype=np.int64)
    return np.unique(ranks_of(np.array(rows)))


def landscape_check(
    scores: np.ndarray,
    combos: np.ndarray,
    n_starts: int = 2000,
    seed: int = 7,
) -> dict:
    """Free-lookup portrait of the landscape the searches run on.

    Counts how many schedules sit near the global best, then runs steepest
    ascent from random starts with unlimited lookups to enumerate the local
    optima of the one-swap neighbourhood and the share of climbs that each
    one captures.  This is diagnosis, not search: it uses the full table.
    """
    mx = float(scores.max())
    near = {
        str(t): int((scores >= mx - t).sum()) for t in (0.01, 0.05, 0.1, 0.2, 0.5)
    }
    rng = np.random.default_rng(seed)
    ends: dict[int, int] = {}
    for start in rng.integers(scores.size, size=n_starts):
        cur = int(start)
        while True:
            nb = neighbour_ranks(combos[cur])
            j = int(nb[np.argmax(scores[nb])])
            if scores[j] <= scores[cur]:
                break
            cur = j
        ends[cur] = ends.get(cur, 0) + 1
    optima = sorted(ends.items(), key=lambda kv: -kv[1])
    return {
        "n_starts": n_starts,
        "schedules_within_db_of_best": near,
        "n_local_optima_reached": len(ends),
        "local_optima": [
            {
                "rank": int(r),
                "basin_share": c / n_starts,
                "mean_psnr_db": float(scores[r]),
                "gap_db": mx - float(scores[r]),
            }
            for r, c in optima
        ],
    }


# --------------------------------------------------------------------------
# search algorithms
# --------------------------------------------------------------------------


def random_rank(rng: np.random.Generator) -> int:
    return int(rng.integers(SPACE_SIZE))


def run_random(oracle: Oracle, rng: np.random.Generator, combos: np.ndarray, **kw) -> None:
    while not oracle.exhausted:
        oracle(random_rank(rng))


def _hill_from(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    start_rank: int,
    steepest: bool,
) -> None:
    """Climb from one start until a local optimum or the budget runs out."""
    current = start_rank
    current_score = oracle(current)
    while True:
        cand = neighbour_ranks(combos[current])
        if steepest:
            best_rank, best_score = current, current_score
            for r in cand:
                value = oracle(r)
                if value > best_score:
                    best_rank, best_score = int(r), value
            if best_rank == current:
                return
            current, current_score = best_rank, best_score
        else:
            rng.shuffle(cand)
            moved = False
            for r in cand:
                value = oracle(r)
                if value > current_score:
                    current, current_score = int(r), value
                    moved = True
                    break
            if not moved:
                return


def _multistart(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    steepest: bool,
    seeds: list[int] | None,
) -> None:
    queue = list(seeds or [])
    while not oracle.exhausted:
        start = queue.pop(0) if queue else random_rank(rng)
        _hill_from(oracle, rng, combos, start, steepest)


def run_hill_first(oracle, rng, combos, **kw):
    _multistart(oracle, rng, combos, steepest=False, seeds=None)


def run_hill_steepest(oracle, rng, combos, **kw):
    _multistart(oracle, rng, combos, steepest=True, seeds=None)


def run_hill_first_seeded(oracle, rng, combos, seed_ranks=(), **kw):
    order = list(seed_ranks)
    rng.shuffle(order)
    _multistart(oracle, rng, combos, steepest=False, seeds=order)


def _budcache_proposal(
    rng: np.random.Generator,
    combo: np.ndarray,
    window: int = 3,
    local_probability: float = 0.7,
) -> np.ndarray:
    """One BudCache swap: a free slot becomes full, one full step becomes cached."""
    free = np.setdiff1d(np.arange(N_POS, dtype=np.int64), combo)
    source = int(rng.choice(free))
    local = [i for i, held in enumerate(combo) if abs(int(held) - source) <= window]
    if local and rng.random() < local_probability:
        drop = int(rng.choice(local))
    else:
        drop = int(rng.integers(K_PICK))
    row = np.array(combo, dtype=np.int64)
    row[drop] = source
    row.sort()
    return row


def run_budcache_sa(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    sa_iters: int = 200,
    hill_iters: int = 20,
    hill_window: int = 3,
    t_max: float = 0.5,
    t_min: float = 1e-4,
    starts: list[int] | None = None,
    **kw,
) -> None:
    """BudCache's own loop: one restart of 200 annealed proposals plus a local climb.

    Run as an anytime method by starting a fresh chain whenever one finishes, so
    that reading the curve at 200 calls gives the published BudCache scale and
    larger budgets show what the same recipe does with more evaluations.  The
    temperature range is stated in dB because the oracle here is a PSNR mean,
    not the MSE loss the original code anneals over.
    """
    queue = list(starts or [])
    while not oracle.exhausted:
        current = queue.pop(0) if queue else random_rank(rng)
        current_score = oracle(current)
        for it in range(sa_iters):
            progress = it / max(sa_iters, 1)
            temperature = t_max * (t_min / t_max) ** progress
            row = _budcache_proposal(rng, combos[current])
            cand = int(ranks_of(row[None, :])[0])
            value = oracle(cand)
            delta = current_score - value  # positive when the proposal is worse
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                current, current_score = cand, value
        for _ in range(hill_iters):
            cand = local_neighbour_ranks(combos[current], hill_window)
            best_rank, best_score = current, current_score
            for r in cand:
                value = oracle(r)
                if value > best_score:
                    best_rank, best_score = int(r), value
            if best_rank == current:
                break
            current, current_score = best_rank, best_score


def run_learned_additive(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    n0: int = 100,
    batch: int = 50,
    ridge: float = 1e-3,
    **kw,
) -> None:
    """Fit an additive cost over the 46 positions, then probe its own top picks."""
    for _ in range(min(n0, oracle.budget)):
        if oracle.exhausted:
            return
        oracle(random_rank(rng))
    while not oracle.exhausted:
        ranks = np.fromiter(oracle.seen.keys(), dtype=np.int64, count=len(oracle.seen))
        targets = np.array([oracle.seen[int(r)] for r in ranks], dtype=np.float64)
        design = np.zeros((ranks.size, N_POS + 1), dtype=np.float64)
        design[:, N_POS] = 1.0
        rows = combos[ranks].astype(np.int64)
        design[np.repeat(np.arange(ranks.size), K_PICK), rows.ravel()] = 1.0
        gram = design.T @ design + ridge * np.eye(N_POS + 1)
        weights = np.linalg.solve(gram, design.T @ targets)
        predicted = weights[combos.astype(np.int64)].sum(axis=1) + weights[N_POS]
        predicted[ranks] = -np.inf
        take = min(batch, oracle.budget - oracle.calls)
        picks = np.argpartition(-predicted, take - 1)[:take] if take > 1 else [int(np.argmax(predicted))]
        for r in picks:
            if oracle.exhausted:
                return
            oracle(int(r))


# Edge index for the pairwise model: virtual start node -1 and end node 46
# in position space, so an edge is a consecutive pair of full steps.
_EDGE_ID = np.full((N_POS + 1, N_POS + 1), -1, dtype=np.int64)
_n = 0
for _a in range(-1, N_POS):
    for _b in range(_a + 1, N_POS + 1):
        _EDGE_ID[_a + 1, _b] = _n
        _n += 1
N_EDGES = _n


def _edge_ids(combos_subset: np.ndarray) -> np.ndarray:
    """(N, 6) edge ids of the consecutive-gap chain of each schedule."""
    c = np.asarray(combos_subset, dtype=np.int64)
    out = np.empty((c.shape[0], K_PICK + 1), dtype=np.int64)
    out[:, 0] = _EDGE_ID[0, c[:, 0]]
    for i in range(1, K_PICK):
        out[:, i] = _EDGE_ID[c[:, i - 1] + 1, c[:, i]]
    out[:, K_PICK] = _EDGE_ID[c[:, -1] + 1, N_POS]
    return out


def run_learned_pairwise(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    n0: int = 100,
    batch: int = 50,
    ridge: float = 1e-3,
    **kw,
) -> None:
    """Fit one cost per consecutive full-step gap, then probe its own argmax.

    The additive model has one weight per position and cannot express
    interactions; this one has one weight per (previous full step, next
    full step) gap, which is the smallest model that can.  The fitted
    model is scored on every schedule in the space by summing its six
    gap weights, so the probe evaluates the model's exact top picks."""
    all_edges = _edge_ids(combos)
    for _ in range(min(n0, oracle.budget)):
        if oracle.exhausted:
            return
        oracle(random_rank(rng))
    while not oracle.exhausted:
        ranks = np.fromiter(oracle.seen.keys(), dtype=np.int64, count=len(oracle.seen))
        targets = np.array([oracle.seen[int(r)] for r in ranks], dtype=np.float64)
        rows_e = all_edges[ranks]
        design = np.zeros((ranks.size, N_EDGES + 1), dtype=np.float64)
        design[:, N_EDGES] = 1.0
        design[np.repeat(np.arange(ranks.size), K_PICK + 1), rows_e.ravel()] = 1.0
        gram = design.T @ design + ridge * np.eye(N_EDGES + 1)
        weights = np.linalg.solve(gram, design.T @ targets)
        predicted = weights[all_edges].sum(axis=1) + weights[N_EDGES]
        predicted[ranks] = -np.inf
        take = min(batch, oracle.budget - oracle.calls)
        picks = np.argpartition(-predicted, take - 1)[:take] if take > 1 else [int(np.argmax(predicted))]
        for r in picks:
            if oracle.exhausted:
                return
            oracle(int(r))


def run_bayes_opt(
    oracle: Oracle,
    rng: np.random.Generator,
    combos: np.ndarray,
    n0: int = 50,
    batch: int = 25,
    pool_size: int = 5000,
    trees: int = 50,
    **kw,
) -> None:
    """Forest surrogate plus expected improvement on a sampled candidate pool.

    Each schedule is a 46-bit indicator vector.  A random forest fitted on
    everything seen so far gives a mean and a spread (across trees) for each
    candidate in a pool of random schedules plus the one-swap neighbours of
    the best seen; the batch evaluates the top expected-improvement picks.
    Refits every 25 calls up to 1,000 calls, every 200 after, to keep the
    cost of 50 repetitions manageable."""
    from scipy.stats import norm
    from sklearn.ensemble import RandomForestRegressor

    def onehot(ranks: np.ndarray) -> np.ndarray:
        x = np.zeros((ranks.size, N_POS), dtype=np.float32)
        rows = combos[ranks].astype(np.int64)
        x[np.repeat(np.arange(ranks.size), K_PICK), rows.ravel()] = 1.0
        return x

    for _ in range(min(n0, oracle.budget)):
        if oracle.exhausted:
            return
        oracle(random_rank(rng))
    while not oracle.exhausted:
        ranks = np.fromiter(oracle.seen.keys(), dtype=np.int64, count=len(oracle.seen))
        targets = np.array([oracle.seen[int(r)] for r in ranks], dtype=np.float64)
        model = RandomForestRegressor(
            n_estimators=trees,
            random_state=int(rng.integers(2**31 - 1)),
            n_jobs=-1,
        )
        model.fit(onehot(ranks), targets)
        pool = np.unique(
            np.concatenate(
                [
                    rng.integers(SPACE_SIZE, size=pool_size),
                    neighbour_ranks(combos[oracle.best_rank]),
                ]
            )
        )
        pool = pool[~np.isin(pool, ranks)]
        if pool.size == 0:
            oracle(random_rank(rng))
            continue
        x = onehot(pool)
        per_tree = np.stack([t.predict(x) for t in model.estimators_])
        mu = per_tree.mean(axis=0)
        sd = per_tree.std(axis=0) + 1e-9
        z = (mu - oracle.best) / sd
        ei = (mu - oracle.best) * norm.cdf(z) + sd * norm.pdf(z)
        take = min(batch if oracle.calls < 1000 else 200, oracle.budget - oracle.calls)
        take = min(take, pool.size)
        picks = pool[np.argpartition(-ei, take - 1)[:take]] if take > 1 else [pool[int(np.argmax(ei))]]
        for r in picks:
            if oracle.exhausted:
                return
            oracle(int(r))


def run_greedy_coordinate(oracle, rng, combos, **kw):
    """Greedy coordinate ascent: optimise one of the five slots exactly,
    holding the other four fixed, and sweep the slots until one full sweep
    changes nothing.  Restart from a fresh random schedule while budget
    remains.  Repeated lookups are memoized like everywhere else, so a
    converged sweep costs little to confirm."""
    while not oracle.exhausted:
        current = random_rank(rng)
        current_score = oracle(current)
        improved = True
        while improved and not oracle.exhausted:
            improved = False
            for slot in range(K_PICK):
                combo = np.array(combos[current], dtype=np.int64)
                held = np.delete(combo, slot)
                best_rank, best_score = current, current_score
                for pos in range(N_POS):
                    if pos in combo:
                        continue
                    row = np.sort(np.append(held, pos)).astype(np.int64)
                    r = int(ranks_of(row[None, :])[0])
                    value = oracle(r)
                    if value > best_score:
                        best_rank, best_score = r, value
                    if oracle.exhausted:
                        break
                if best_rank != current:
                    current, current_score = best_rank, best_score
                    improved = True
                if oracle.exhausted:
                    break


def run_budcache_sa_seeded(oracle, rng, combos, seed_ranks=(), **kw):
    """The same annealing loop, but the chains start from the in-domain
    baseline schedules (shuffled) before falling back to random starts."""
    order = list(seed_ranks)
    rng.shuffle(order)
    run_budcache_sa(oracle, rng, combos, starts=order, **kw)


METHODS = {
    "random": (run_random, "random sampling"),
    "hill_first": (run_hill_first, "multi-start hill climb, first improvement"),
    "hill_steepest": (run_hill_steepest, "multi-start hill climb, steepest ascent"),
    "hill_first_seeded": (
        run_hill_first_seeded,
        "hill climb started from the in-domain baseline schedules",
    ),
    "budcache_sa": (run_budcache_sa, "BudCache annealing plus local climb"),
    "budcache_sa_seeded": (
        run_budcache_sa_seeded,
        "annealing started from the baseline schedules",
    ),
    "greedy_coordinate": (run_greedy_coordinate,
                          "greedy coordinate ascent, one slot at a time"),
    "learned_additive": (run_learned_additive, "learned additive cost over positions"),
    "learned_pairwise": (
        run_learned_pairwise,
        "learned pairwise cost over gaps, exact argmax",
    ),
    "bayes_opt": (
        run_bayes_opt,
        "Bayesian optimization, forest surrogate",
    ),
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def percentile_of(sorted_scores: np.ndarray, value: float) -> float:
    """Share of the space, in percent, scoring at or below `value`."""
    idx = np.searchsorted(sorted_scores, value, side="right")
    return 100.0 * idx / sorted_scores.size


def summarise(
    traces: np.ndarray,
    sorted_scores: np.ndarray,
    global_max: float,
    budgets: tuple[int, ...],
) -> dict:
    out: dict = {"by_budget": {}, "calls_to_threshold": {}}
    gaps = global_max - traces
    for b in budgets:
        best = traces[:, b - 1]
        gap = global_max - best
        pct = np.array([percentile_of(sorted_scores, v) for v in best])
        out["by_budget"][str(b)] = {
            "gap_db_mean": float(gap.mean()),
            "gap_db_p10": float(np.percentile(gap, 10)),
            "gap_db_p90": float(np.percentile(gap, 90)),
            "percentile_mean": float(pct.mean()),
            "percentile_p10": float(np.percentile(pct, 10)),
            "percentile_p90": float(np.percentile(pct, 90)),
            "found_global_max_share": float((gap <= 0).mean()),
        }
    for t in THRESHOLDS:
        reached = gaps <= t
        first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, -1)
        hit = first > 0
        if hit.mean() < 0.5:
            out["calls_to_threshold"][str(t)] = {
                "median_calls": None,
                "reached_share": float(hit.mean()),
                "note": "not reached in over half the repetitions",
            }
        else:
            vals = first[hit].astype(float)
            out["calls_to_threshold"][str(t)] = {
                "median_calls": float(np.median(vals)),
                "reached_share": float(hit.mean()),
                "p10_calls": float(np.percentile(vals, 10)),
                "p90_calls": float(np.percentile(vals, 90)),
            }
    return out


def curve_grid(max_budget: int, points: int = 60) -> np.ndarray:
    grid = np.unique(
        np.round(np.geomspace(1, max_budget, points)).astype(int)
    )
    return grid


def make_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array(results["curve_calls"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.get_cmap("tab10")
    floor = 0.01
    for i, (name, payload) in enumerate(results["methods"].items()):
        mean = np.maximum(np.array(payload["curve_gap_mean"]), floor)
        lo = np.maximum(np.array(payload["curve_gap_p10"]), floor)
        hi = np.maximum(np.array(payload["curve_gap_p90"]), floor)
        color = colors(i % 10)
        ax.plot(grid, mean, label=payload["label"], color=color, linewidth=1.8)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("oracle calls (schedule evaluations)")
    ax.set_ylabel("PSNR gap to the global best (dB)")
    ax.set_title(
        "Search on the exhaustive K41 table, %d repetitions"
        % results["repetitions"]
    )
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_ylim(floor * 0.8, None)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.text(
        0.5,
        0.005,
        "bands are the 10 to 90 percent range over repetitions; gaps are clipped at %.2f dB"
        % floor,
        ha="center",
        fontsize=7,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", help="merged exhaustive table (.tsv or .tsv.gz)")
    p.add_argument("--summary", help="formal_results/summary.json")
    p.add_argument("--manifest", help="formal_results/candidate_manifest.tsv")
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_fig", required=True)
    p.add_argument(
        "--figure_only",
        action="store_true",
        help="redraw the figure from an existing results.json, no table needed",
    )
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--max_budget", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    if args.figure_only:
        results = json.loads(Path(args.out_json).read_text())
        make_figure(results, Path(args.out_fig))
        print("[INFO] wrote %s" % args.out_fig, flush=True)
        return 0
    for required in ("table", "summary", "manifest"):
        if getattr(args, required) is None:
            raise SystemExit("--%s is required unless --figure_only is set" % required)

    summary = json.loads(Path(args.summary).read_text())
    with open(args.manifest, newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    seed_rows = [r for r in manifest_rows if r["reasons"].startswith("included:")]
    seed_ranks = [int(r["rank"]) for r in seed_rows]

    combos = build_all_combos()
    check = np.random.default_rng(0).integers(SPACE_SIZE, size=100_000)
    if not np.array_equal(ranks_of(combos[check]), check):
        raise SystemExit("rank round-trip check failed")

    print("[INFO] loading %s" % args.table, flush=True)
    scores, info = load_table(Path(args.table), combos)
    ver = verify(scores, info, summary, manifest_rows, combos)
    print("[INFO] verification: %s" % json.dumps(ver, indent=2), flush=True)
    if not ver["ranks_complete"] or ver["combination_rank_mismatches"]:
        raise SystemExit("table failed the rank / combination verification")
    if not ver["argmax_matches_summary"]:
        raise SystemExit("table best row disagrees with summary.json")

    land = landscape_check(scores, combos)
    print("[INFO] landscape: %s" % json.dumps(land), flush=True)

    global_max = float(scores.max())
    sorted_scores = np.sort(scores)
    grid = curve_grid(args.max_budget)
    budgets = tuple(b for b in BUDGETS if b <= args.max_budget)

    results = {
        "schema": "golden_path_search_bench.v1",
        "table": str(args.table),
        "verification": ver,
        "landscape": land,
        "global_max_mean_psnr_db": global_max,
        "space_size": SPACE_SIZE,
        "repetitions": args.reps,
        "budgets": list(budgets),
        "thresholds_db": list(THRESHOLDS),
        "max_budget": args.max_budget,
        "seed": args.seed,
        "seed_schedules": [
            {
                "name": r["reasons"].split(":", 1)[1],
                "rank": int(r["rank"]),
                "variable_full_steps": r["variable_full_steps"],
                "mean_psnr_db": float(r["mean_psnr_db"]),
            }
            for r in seed_rows
        ],
        "curve_calls": grid.tolist(),
        "methods": {},
    }

    for name in args.methods:
        fn, label = METHODS[name]
        t0 = time.perf_counter()
        traces = np.empty((args.reps, args.max_budget), dtype=np.float64)
        for rep in range(args.reps):
            rng = np.random.default_rng(
                [args.seed, rep, int(zlib.crc32(name.encode()))]
            )
            oracle = Oracle(scores, args.max_budget)
            try:
                fn(oracle, rng, combos, seed_ranks=seed_ranks)
            except BudgetExhausted:
                pass
            if oracle.calls < args.max_budget:
                oracle.trace[oracle.calls :] = oracle.best
            traces[rep] = oracle.trace
        stats = summarise(traces, sorted_scores, global_max, budgets)
        gaps = global_max - traces[:, grid - 1]
        stats["label"] = label
        stats["curve_gap_mean"] = gaps.mean(axis=0).round(6).tolist()
        stats["curve_gap_p10"] = np.percentile(gaps, 10, axis=0).round(6).tolist()
        stats["curve_gap_p90"] = np.percentile(gaps, 90, axis=0).round(6).tolist()
        stats["seconds"] = round(time.perf_counter() - t0, 2)
        results["methods"][name] = stats
        print(
            "[INFO] %-18s done in %6.1f s, gap at %d = %.4f dB"
            % (
                name,
                stats["seconds"],
                budgets[-1],
                stats["by_budget"][str(budgets[-1])]["gap_db_mean"],
            ),
            flush=True,
        )

    results["total_seconds"] = round(time.perf_counter() - started, 2)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print("[INFO] wrote %s" % args.out_json, flush=True)
    try:
        make_figure(results, Path(args.out_fig))
    except ImportError as exc:
        print(
            "[WARN] no figure here (%s); redraw with --figure_only where "
            "matplotlib is installed" % exc,
            flush=True,
        )
    else:
        print("[INFO] wrote %s" % args.out_fig, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
