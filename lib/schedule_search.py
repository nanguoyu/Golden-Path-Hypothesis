"""Schedule-search space and the four search algorithms of the SS experiment.

The recipes are the ones selected on the K41 truth table by
`analysis/golden_path_search_bench.py`; this module holds the same proposal
structure, +/- 3 window, 70 % local swap, geometric cooling and warm-start
queues, with the table lookup replaced by an arbitrary evaluation callback so
the same code runs against a real model on FLUX and on Qwen-Image.

A candidate is the sorted tuple of its *free* full steps (absolute step
indices inside `variable_start .. variable_end`).  The forced full steps are
implicit, so every tuple the algorithms produce is a member of the space by
construction and never has to be repaired.  `SearchSpace.bits` writes the
schedule in the SPX convention (`'1'` = cached, `'0'` = full).

Budget and stopping are owned by `SearchControl`: repeat visits are memoized
and do not consume budget, the hard cap raises `BudgetExhausted`, and the
restart-unit rule stops a search once two consecutive units improve the best
mean by less than twice the standard error of the current best's eight-pair
mean.

`PairPool` is the optional intra-node fan-out both model runners share: the
search stays sequential, but the pairs of one evaluation are independent
generations and can be spread over the GPUs of a node.

Every evaluation records the matrix's five metrics per calibration pair;
`ObjectiveEvaluator` maps that vector to the one number the search maximises
(`OBJECTIVES`: `psnr`, `lpips`, `psnr_lpips_z`) and writes the JSONL row.
Everything downstream -- the stop rule, the standard error, `select_best`,
the centered worst-pair guard and the arbitration -- reads those objective
values, so the objective is the only thing that changes between runs.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class BudgetExhausted(Exception):
    """Raised by :class:`SearchControl` when a new evaluation is refused."""


@dataclass(frozen=True)
class SearchSpace:
    """Fixed-budget schedule space shared by FLUX and Qwen-Image.

    `cache_count` is K, the number of cached steps, so the number of full
    steps is `num_steps - cache_count` and the free ones are that minus the
    forced set.  K41 / K37 / K29 give 5 / 9 / 17 free full steps.
    """

    num_steps: int = 50
    cache_count: int = 41
    forced_full_steps: tuple[int, ...] = (0, 1, 2, 49)
    variable_start: int = 3
    variable_end: int = 48

    @property
    def variable_steps(self) -> tuple[int, ...]:
        return tuple(range(self.variable_start, self.variable_end + 1))

    @property
    def full_count(self) -> int:
        return self.num_steps - self.cache_count

    @property
    def free_full_count(self) -> int:
        return self.full_count - len(self.forced_full_steps)

    @property
    def total(self) -> int:
        return math.comb(len(self.variable_steps), self.free_full_count)

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "cache_count": self.cache_count,
            "forced_full_steps": list(self.forced_full_steps),
            "variable_start": self.variable_start,
            "variable_end": self.variable_end,
            "free_full_count": self.free_full_count,
            "space_size": self.total,
        }

    # -- membership -------------------------------------------------------

    def validate(self, combo: Sequence[int]) -> tuple[int, ...]:
        values = tuple(int(step) for step in combo)
        if values != tuple(sorted(set(values))):
            raise ValueError(f"free full steps must be sorted and unique: {values}")
        if len(values) != self.free_full_count:
            raise ValueError(
                f"expected {self.free_full_count} free full steps, got {len(values)}"
            )
        if values and (
            values[0] < self.variable_start or values[-1] > self.variable_end
        ):
            raise ValueError(f"free full steps outside the variable window: {values}")
        return values

    def full_steps(self, combo: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted((*self.forced_full_steps, *self.validate(combo))))

    def cache_steps(self, combo: Sequence[int]) -> tuple[int, ...]:
        full = set(self.full_steps(combo))
        return tuple(step for step in range(self.num_steps) if step not in full)

    def bits(self, combo: Sequence[int]) -> str:
        """SPX bitstring: `'1'` is cached, `'0'` is full."""

        full = set(self.full_steps(combo))
        return "".join(
            "0" if step in full else "1" for step in range(self.num_steps)
        )

    def combo_of_full_steps(self, full_steps: Iterable[int]) -> tuple[int, ...]:
        full = tuple(sorted(set(int(step) for step in full_steps)))
        if len(full) != self.full_count:
            raise ValueError(f"expected {self.full_count} full steps, got {full}")
        if not set(self.forced_full_steps).issubset(full):
            raise ValueError(f"full steps omit the forced set {self.forced_full_steps}")
        return self.validate(
            tuple(step for step in full if step not in self.forced_full_steps)
        )

    def combo_of_bits(self, bits: str) -> tuple[int, ...]:
        text = str(bits).strip()
        if len(text) != self.num_steps or set(text) - {"0", "1"}:
            raise ValueError(
                f"schedule must hold {self.num_steps} characters of 0/1: {text!r}"
            )
        return self.combo_of_full_steps(
            step for step, bit in enumerate(text) if bit == "0"
        )

    # -- moves ------------------------------------------------------------

    def random_combo(self, rng: np.random.Generator) -> tuple[int, ...]:
        picked = rng.choice(
            np.asarray(self.variable_steps, dtype=np.int64),
            size=self.free_full_count,
            replace=False,
        )
        return tuple(int(step) for step in np.sort(picked))

    def _free_slots(self, combo: Sequence[int]) -> list[int]:
        held = set(combo)
        return [step for step in self.variable_steps if step not in held]

    def neighbours(self, combo: Sequence[int]) -> list[tuple[int, ...]]:
        """Every schedule reached by moving one full step to a cached slot."""

        combo = self.validate(combo)
        out: set[tuple[int, ...]] = set()
        for source in self._free_slots(combo):
            for index in range(len(combo)):
                row = list(combo)
                row[index] = source
                out.add(tuple(sorted(row)))
        out.discard(tuple(combo))
        return sorted(out)

    def local_neighbours(
        self, combo: Sequence[int], window: int
    ) -> list[tuple[int, ...]]:
        """The windowed neighbourhood: destinations within +/- `window` steps."""

        combo = self.validate(combo)
        out: set[tuple[int, ...]] = set()
        for source in self._free_slots(combo):
            for index, held in enumerate(combo):
                if abs(int(held) - int(source)) > int(window):
                    continue
                row = list(combo)
                row[index] = source
                out.add(tuple(sorted(row)))
        out.discard(tuple(combo))
        return sorted(out)

    def proposal(
        self,
        rng: np.random.Generator,
        combo: Sequence[int],
        *,
        window: int = 3,
        local_probability: float = 0.7,
    ) -> tuple[int, ...]:
        """One annealing swap: a cached slot becomes full, one full step goes."""

        combo = self.validate(combo)
        free = self._free_slots(combo)
        source = int(rng.choice(np.asarray(free, dtype=np.int64)))
        local = [
            index
            for index, held in enumerate(combo)
            if abs(int(held) - source) <= int(window)
        ]
        if local and rng.random() < float(local_probability):
            drop = int(rng.choice(np.asarray(local, dtype=np.int64)))
        else:
            drop = int(rng.integers(len(combo)))
        row = list(combo)
        row[drop] = source
        return tuple(sorted(row))


def hamming(space: SearchSpace, left: Sequence[int], right: Sequence[int]) -> int:
    """Bitstring Hamming distance between two schedules of the same budget."""

    return 2 * len(set(space.validate(left)) - set(space.validate(right)))


# --------------------------------------------------------------------------
# objectives: the five recorded metrics -> the one number a search maximises
# --------------------------------------------------------------------------
#
# Every evaluation records the matrix's five metrics per calibration pair
# (`lib/search_metrics.py`).  The objective picks the scalar out of that vector
# for each pair; the mean over pairs is what `SearchControl` maximises, and the
# stop rule, the standard error, the selection and the arbitration all read the
# same per-pair values.  LPIPS is a distance, so the objective is its negative
# and larger is better throughout.

#: The metric names an evaluation records, in the matrix's order.
METRIC_NAMES: tuple[str, ...] = ("psnr", "ssim", "lpips", "image_reward", "clip")

#: Search objectives.  `psnr` is the original one and stays the default.
OBJECTIVES: tuple[str, ...] = ("psnr", "lpips", "psnr_lpips_z")

#: The metrics `psnr_lpips_z` standardises, and the sign each enters with.
Z_TERMS: tuple[tuple[str, float], ...] = (("psnr", 1.0), ("lpips", -1.0))


def objective_units(objective: str) -> str:
    return {
        "psnr": "db",
        "lpips": "negative_lpips",
        "psnr_lpips_z": "z",
    }[objective]


def _scale(scales: dict[str, Any] | None, metric: str) -> tuple[float, float]:
    if not scales or metric not in scales:
        raise ValueError(
            f"objective scales for {metric!r} are missing; run the probe for this "
            "setting and backfill search.objective_scales"
        )
    row = scales[metric]
    mean, std = float(row["mean"]), float(row["std"])
    if not std > 0.0:
        raise ValueError(f"objective scale for {metric!r} has a non-positive std")
    return mean, std


def objective_pair_values(
    objective: str,
    metrics: dict[str, Sequence[float]],
    *,
    scales: dict[str, Any] | None = None,
) -> list[float]:
    """One scalar per calibration pair, larger is better.

    `metrics` is the evaluation's `{metric name: per-pair values}`.  For
    `psnr_lpips_z` the PSNR and the negative LPIPS are each standardised by the
    mean and standard deviation the setting's probe measured, and the two
    z-scores are added.
    """

    if objective == "psnr":
        return [float(v) for v in metrics["psnr"]]
    if objective == "lpips":
        return [-float(v) for v in metrics["lpips"]]
    if objective == "psnr_lpips_z":
        columns = []
        for metric, sign in Z_TERMS:
            mean, std = _scale(scales, metric)
            columns.append(
                [(sign * float(v) - sign * mean) / std for v in metrics[metric]]
            )
        return [float(sum(column[i] for column in columns)) for i in range(len(columns[0]))]
    raise ValueError(f"unknown objective {objective!r}; pick one of {OBJECTIVES}")


def objective_value(
    objective: str,
    metric_means: dict[str, float],
    *,
    scales: dict[str, Any] | None = None,
) -> float:
    """The objective of one evaluation from its per-metric means.

    Every objective is affine in the metrics, so the objective of the means is
    the mean of the per-pair objectives; this is the form the probe summary and
    the temperature backfill work in.
    """

    return objective_pair_values(
        objective,
        {name: [float(value)] for name, value in metric_means.items()},
        scales=scales,
    )[0]


def lookup_objective_scales(
    search_cfg: dict[str, Any], objective: str, model: str, k: int
) -> dict[str, Any] | None:
    """`search.objective_scales[<model>][<k>]`, or None when not needed."""

    if objective != "psnr_lpips_z":
        return None
    scales = (
        search_cfg.get("objective_scales", {}).get(model, {}).get(str(k))
    )
    if not scales:
        raise SystemExit(
            f"objective {objective} needs search.objective_scales[{model}][{k}] in "
            "the config: run the probe for this setting and backfill it with "
            "analysis/backfill_schedule_search_temperatures.py"
        )
    return scales


def lookup_temperatures(
    search_cfg: dict[str, Any], objective: str, model: str, k: int
) -> dict[str, Any]:
    """The annealing temperatures of one (objective, model, K).

    The PSNR entries stay where P1 put them, `search.temperatures[<model>][<k>]`;
    a later objective is `search.temperatures[<objective>][<model>][<k>]`.
    """

    temperatures = search_cfg.get("temperatures") or {}
    if objective == "psnr":
        return temperatures.get(model, {}).get(str(k), {}) or {}
    return temperatures.get(objective, {}).get(model, {}).get(str(k), {}) or {}


# --------------------------------------------------------------------------
# budget, best tracking and the stop rule
# --------------------------------------------------------------------------


class SearchControl:
    """Budgeted access to the real evaluation, plus the restart-unit stop rule.

    `evaluate(combo)` returns the per-pair scores of one schedule; the
    objective is their mean.  A restart unit is one annealing chain, one hill
    climb, or one greedy sweep; after each unit the improvement of the best
    mean is compared against `se_factor` times the standard error of the
    current best's per-pair scores, and two consecutive small units stop the
    search.
    """

    def __init__(
        self,
        evaluate: Callable[[tuple[int, ...]], Sequence[float]],
        *,
        max_evals: int,
        se_factor: float = 2.0,
        stop_units: int = 2,
        use_stop_rule: bool = True,
    ) -> None:
        self.evaluate = evaluate
        self.max_evals = int(max_evals)
        self.se_factor = float(se_factor)
        self.stop_units = int(stop_units)
        self.use_stop_rule = bool(use_stop_rule)
        self.calls = 0
        self.seen: dict[tuple[int, ...], float] = {}
        self._preloaded: dict[tuple[int, ...], tuple[float, ...]] = {}
        self.records: list[dict[str, Any]] = []
        self.best = -math.inf
        self.best_combo: tuple[int, ...] | None = None
        self.best_scores: tuple[float, ...] = ()
        self.units = 0
        self.stopped = False
        self.stop_reason: str | None = None
        self._unit_start_best = -math.inf
        self._quiet_units = 0

    @property
    def exhausted(self) -> bool:
        return self.stopped or self.calls >= self.max_evals

    def preload(self, rows: Iterable[tuple[Sequence[int], Sequence[float]]]) -> int:
        """Hold an earlier trace of the same run so a resubmission can replay it.

        A search is a deterministic function of its RNG stream and the
        objective, so a resubmitted job that answers already-scored schedules
        from the trace instead of the GPU walks the identical path and then
        continues past where the previous job was cut off.  Replayed answers
        still count against the cap: they were real evaluations.
        """

        loaded = 0
        for combo, scores in rows:
            key = tuple(int(step) for step in combo)
            if key in self._preloaded:
                continue
            self._preloaded[key] = tuple(float(v) for v in scores)
            loaded += 1
        return loaded

    def __call__(self, combo: Sequence[int]) -> float:
        key = tuple(int(step) for step in combo)
        hit = self.seen.get(key)
        if hit is not None:
            return hit
        if self.calls >= self.max_evals:
            self.stop_reason = self.stop_reason or "cap"
            raise BudgetExhausted
        replayed = self._preloaded.pop(key, None)
        scores = (
            replayed
            if replayed is not None
            else tuple(float(v) for v in self.evaluate(key))
        )
        value = float(sum(scores) / len(scores))
        self.calls += 1
        self.seen[key] = value
        self.records.append(
            {
                "combo": key,
                "mean": value,
                "per_pair": list(scores),
                "replayed": replayed is not None,
            }
        )
        if value > self.best:
            self.best, self.best_combo, self.best_scores = value, key, scores
        return value

    def best_se(self) -> float:
        """Standard error of the current best's per-pair mean."""

        if len(self.best_scores) < 2:
            return 0.0
        return float(
            np.std(np.asarray(self.best_scores, dtype=np.float64), ddof=1)
            / math.sqrt(len(self.best_scores))
        )

    def begin_unit(self) -> None:
        self._unit_start_best = self.best

    def end_unit(self) -> None:
        self.units += 1
        if not self.use_stop_rule:
            return
        if self._unit_start_best == -math.inf:
            # The first unit has no baseline to improve on.
            return
        improvement = self.best - self._unit_start_best
        if improvement < self.se_factor * self.best_se():
            self._quiet_units += 1
        else:
            self._quiet_units = 0
        if self._quiet_units >= self.stop_units:
            self.stopped = True
            self.stop_reason = "stop_rule"


# --------------------------------------------------------------------------
# search algorithms
# --------------------------------------------------------------------------


def _shuffled(
    rng: np.random.Generator, combos: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Warm-start queue in a deterministic random order."""

    rows = [tuple(int(step) for step in combo) for combo in combos]
    return [rows[int(index)] for index in rng.permutation(len(rows))]


def run_random(
    control: SearchControl,
    rng: np.random.Generator,
    space: SearchSpace,
    **_kw: Any,
) -> None:
    """Control arm: draw schedules uniformly until the cap (no stop rule)."""

    while not control.exhausted:
        control(space.random_combo(rng))


def _hill_from(
    control: SearchControl,
    rng: np.random.Generator,
    space: SearchSpace,
    start: tuple[int, ...],
) -> None:
    """First-improvement climb from one start, to a local optimum or the cap."""

    current = start
    current_score = control(current)
    while True:
        candidates = space.neighbours(current)
        moved = False
        for index in rng.permutation(len(candidates)):
            candidate = candidates[int(index)]
            value = control(candidate)
            if value > current_score:
                current, current_score = candidate, value
                moved = True
                break
        if not moved:
            return


def run_hill(
    control: SearchControl,
    rng: np.random.Generator,
    space: SearchSpace,
    warm_starts: Sequence[tuple[int, ...]] = (),
    **_kw: Any,
) -> None:
    """First-improvement hill climb, warm-started from the baseline schedules."""

    queue = _shuffled(rng, warm_starts)
    while not control.exhausted:
        start = queue.pop(0) if queue else space.random_combo(rng)
        control.begin_unit()
        _hill_from(control, rng, space, start)
        control.end_unit()


def run_anneal(
    control: SearchControl,
    rng: np.random.Generator,
    space: SearchSpace,
    warm_starts: Sequence[tuple[int, ...]] = (),
    *,
    sa_iters: int = 200,
    t_max: float = 0.5,
    t_min: float = 1e-4,
    hill_iters: int = 20,
    hill_window: int = 3,
    on_state: Callable[[dict[str, Any]], None] | None = None,
    **_kw: Any,
) -> None:
    """Annealed swaps plus a local climb, warm-started from the baselines.

    One chain is `sa_iters` proposals under geometric cooling from `t_max` to
    `t_min`, followed by up to `hill_iters` steepest moves over the windowed
    neighbourhood.  Chains restart while budget remains; the temperature range
    is in dB because the objective is a PSNR mean.
    """

    queue = _shuffled(rng, warm_starts)
    chain = 0
    while not control.exhausted:
        current = queue.pop(0) if queue else space.random_combo(rng)
        control.begin_unit()
        if on_state is not None:
            on_state({"chain": chain, "stage": "start", "temperature": None})
        current_score = control(current)
        for iteration in range(sa_iters):
            progress = iteration / max(sa_iters, 1)
            temperature = t_max * (t_min / t_max) ** progress
            candidate = space.proposal(rng, current, window=hill_window)
            if on_state is not None:
                on_state(
                    {
                        "chain": chain,
                        "stage": "sa",
                        "iteration": iteration,
                        "temperature": float(temperature),
                    }
                )
            value = control(candidate)
            delta = current_score - value  # positive when the proposal is worse
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                current, current_score = candidate, value
        for iteration in range(hill_iters):
            if on_state is not None:
                on_state(
                    {
                        "chain": chain,
                        "stage": "hill",
                        "iteration": iteration,
                        "temperature": None,
                    }
                )
            best_combo, best_score = current, current_score
            for candidate in space.local_neighbours(current, hill_window):
                value = control(candidate)
                if value > best_score:
                    best_combo, best_score = candidate, value
            if best_combo == current:
                break
            current, current_score = best_combo, best_score
        control.end_unit()
        chain += 1


def run_greedy(
    control: SearchControl,
    rng: np.random.Generator,
    space: SearchSpace,
    **_kw: Any,
) -> None:
    """Greedy coordinate ascent: optimise one slot exactly, sweep, restart.

    A restart unit is one sweep over the free slots; the outer loop restarts
    from a fresh random schedule once a sweep changes nothing.
    """

    while not control.exhausted:
        current = space.random_combo(rng)
        current_score = control(current)
        improved = True
        while improved and not control.exhausted:
            control.begin_unit()
            improved = False
            for slot in range(space.free_full_count):
                held = [step for i, step in enumerate(current) if i != slot]
                best_combo, best_score = current, current_score
                for step in space.variable_steps:
                    if step in current:
                        continue
                    candidate = tuple(sorted((*held, step)))
                    value = control(candidate)
                    if value > best_score:
                        best_combo, best_score = candidate, value
                if best_combo != current:
                    current, current_score = best_combo, best_score
                    improved = True
                if control.exhausted:
                    break
            control.end_unit()


ALGORITHMS: dict[str, tuple[Callable[..., None], str]] = {
    "random": (run_random, "random sampling, control arm"),
    "hill": (run_hill, "first-improvement hill climb, warm-started"),
    "anneal": (run_anneal, "annealing plus local climb, warm-started"),
    "greedy": (run_greedy, "greedy coordinate ascent"),
}


# --------------------------------------------------------------------------
# selection among the evaluated schedules
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# config, logging and the P1 probe, shared by the two model runners
# --------------------------------------------------------------------------


def load_space(config: dict[str, Any], model: str, k: int) -> SearchSpace:
    """Build the (model, K) space declared in `config.v1.json`."""

    spec = config["spaces"][model]
    space = SearchSpace(
        num_steps=int(spec["num_steps"]),
        cache_count=int(k),
        forced_full_steps=tuple(int(s) for s in spec["forced_full_steps"]),
        variable_start=int(spec["variable_start"]),
        variable_end=int(spec["variable_end"]),
    )
    expected = int(spec["free_full_count"][str(k)])
    if space.free_full_count != expected:
        raise ValueError(
            f"config says K{k} has {expected} free full steps, the space gives "
            f"{space.free_full_count}"
        )
    return space


def load_warm_starts(
    config: dict[str, Any], space: SearchSpace, model: str, k: int
) -> list[tuple[int, ...]]:
    """The frozen in-domain baseline schedules the climbers start from."""

    rows = config.get("warm_starts", {}).get(model, {}).get(str(k), [])
    return [space.combo_of_bits(row["bits"]) for row in rows]


def parse_schedule_line(space: SearchSpace, line: str) -> tuple[int, ...]:
    """One schedule from a text file: a bitstring or comma-separated full steps."""

    text = line.strip()
    if "," in text:
        return space.combo_of_full_steps(
            int(piece) for piece in text.split(",") if piece.strip()
        )
    return space.combo_of_bits(text)


class EvalSink:
    """Append one JSONL row per evaluation and carry the algorithm state."""

    def __init__(self, path: Any, *, header: dict[str, Any]) -> None:
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")
        self.header = header
        self.state: dict[str, Any] = {
            "chain": None,
            "stage": None,
            "iteration": None,
            "temperature": None,
        }
        self.count = 0

    def on_state(self, state: dict[str, Any]) -> None:
        self.state.update(
            {key: state.get(key) for key in ("chain", "stage", "iteration", "temperature")}
        )

    def write(self, row: dict[str, Any]) -> None:
        import json

        self.count += 1
        payload = {"eval": self.count, **self.header, **self.state, **row}
        self.handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ObjectiveEvaluator:
    """One schedule -> the objective's per-pair values, with the row logged.

    `score(cache_steps)` is the model side: it returns one
    `{metric name: value}` dict per calibration pair, in pair order.  This
    turns that into the JSONL row -- `psnr_db` as the search has always written
    it, plus `metrics` with all five -- and hands `SearchControl` the per-pair
    values of the chosen objective.  Under `psnr` the values *are* `psnr_db`,
    so those rows carry nothing beyond the extra `metrics` field.
    """

    def __init__(
        self,
        *,
        space: SearchSpace,
        score: Callable[[Sequence[int]], Sequence[dict[str, float]]],
        sink: EvalSink | None,
        objective: str = "psnr",
        scales: dict[str, Any] | None = None,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}; pick one of {OBJECTIVES}")
        self.space = space
        self.score = score
        self.sink = sink
        self.objective = str(objective)
        self.scales = scales
        self.metrics: dict[tuple[int, ...], dict[str, list[float]]] = {}

    # -- metric bookkeeping ----------------------------------------------

    def _remember(
        self, combo: Sequence[int], metrics: dict[str, list[float]]
    ) -> list[float]:
        key = tuple(int(step) for step in combo)
        self.metrics[key] = metrics
        return objective_pair_values(self.objective, metrics, scales=self.scales)

    def metrics_of(self, combo: Sequence[int]) -> dict[str, list[float]]:
        """The per-pair metrics of an already-evaluated schedule."""

        return self.metrics[tuple(int(step) for step in combo)]

    def metric_means(self, combo: Sequence[int]) -> dict[str, float]:
        """The pair means of an already-evaluated schedule, one per metric."""

        return {
            name: float(np.mean(np.asarray(values, dtype=np.float64)))
            for name, values in self.metrics_of(combo).items()
        }

    def preload_trace(
        self, rows: Iterable[tuple[Sequence[int], dict[str, list[float]]]]
    ) -> list[tuple[tuple[int, ...], list[float]]]:
        """An earlier trace, re-scored under this objective, for `preload`."""

        out: list[tuple[tuple[int, ...], list[float]]] = []
        for combo, metrics in rows:
            key = tuple(int(step) for step in combo)
            out.append((key, self._remember(key, metrics)))
        return out

    # -- evaluation -------------------------------------------------------

    def __call__(self, combo: Sequence[int]) -> list[float]:
        started = time.perf_counter()
        rows = list(self.score(self.space.cache_steps(combo)))
        metrics = {
            name: [float(row[name]) for row in rows] for name in METRIC_NAMES
        }
        values = self._remember(combo, metrics)
        if self.sink is not None:
            row: dict[str, Any] = {
                "schedule": ",".join(str(s) for s in self.space.full_steps(combo)),
                "bits": self.space.bits(combo),
                "psnr_db": [round(v, 6) for v in metrics["psnr"]],
                "mean_psnr_db": float(np.mean(metrics["psnr"])),
                "min_psnr_db": float(np.min(metrics["psnr"])),
                "metrics": {
                    name: [round(v, 6) for v in values_]
                    for name, values_ in metrics.items()
                },
            }
            if self.objective != "psnr":
                row.update(
                    {
                        "objective_values": [round(v, 6) for v in values],
                        "mean_objective": float(np.mean(values)),
                        "min_objective": float(np.min(values)),
                    }
                )
            row["wall_s"] = round(time.perf_counter() - started, 3)
            self.sink.write(row)
        return values


def summary_row(
    space: SearchSpace, evaluate: ObjectiveEvaluator, record: dict[str, Any]
) -> dict[str, Any]:
    """One evaluated schedule as `summary.json` writes it.

    `per_pair` and `mean_psnr_db` keep their meaning from the PSNR search --
    under any other objective `mean_psnr_db` is still the PSNR mean, and
    `mean_objective` carries the number the search compared.
    """

    combo = record["combo"]
    means = evaluate.metric_means(combo)
    row = {
        "schedule": ",".join(str(s) for s in space.full_steps(combo)),
        "bits": space.bits(combo),
        "per_pair": record["per_pair"],
        "mean_psnr_db": (
            float(record["mean"])
            if evaluate.objective == "psnr"
            else float(means["psnr"])
        ),
        "metrics": means,
    }
    if evaluate.objective != "psnr":
        row["mean_objective"] = float(record["mean"])
    return row


# --------------------------------------------------------------------------
# intra-node fan-out: one evaluation's pair generations over several GPUs
# --------------------------------------------------------------------------
#
# The search itself is sequential -- every proposal depends on the previous
# decision -- but the pairs inside one evaluation are independent generations,
# so they are what a four-GPU node can run at once.  A worker holds one model
# replica on one device and owns the pairs `i % world_size == rank`, including
# their full-compute references, so no image ever crosses a process boundary
# and each pair is generated exactly as the single-GPU runner generates it.


def _pool_devices(world_size: int) -> list[str]:
    """Rank -> device id, in the convention of `RUN/multi_gpu_sp_cross.sh`.

    The ids are read off the parent's `CUDA_VISIBLE_DEVICES` so a Slurm
    allocation of, say, GPUs 1 and 3 hands its own two devices out rather than
    physical 0 and 1.
    """

    visible = [
        piece.strip()
        for piece in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if piece.strip()
    ]
    if not visible:
        return [str(rank) for rank in range(world_size)]
    if len(visible) < world_size:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES lists {len(visible)} device(s), "
            f"{world_size} workers were asked for"
        )
    return visible[:world_size]


def _pair_worker(
    rank: int,
    world_size: int,
    owned: tuple[int, ...],
    setup: Callable[..., dict[str, Any]],
    evaluate: Callable[..., Sequence[float]],
    blob: dict[str, Any],
    requests: Any,
    results: Any,
) -> None:
    """One GPU: load the model once, then answer one schedule at a time.

    The process is started with `CUDA_VISIBLE_DEVICES` already narrowed to its
    own device, so `setup` and `evaluate` see a single GPU and need no device
    argument.  Any failure is reported once and ends the worker; the parent
    turns that into a nonzero exit.
    """

    import traceback

    try:
        state = setup(rank, world_size, owned, blob)
    except BaseException:  # noqa: BLE001 - reported to the parent, which raises
        results.put(("error", rank, -1, traceback.format_exc()))
        return
    results.put(
        (
            "ready",
            rank,
            -1,
            {
                "device": state.get("device"),
                "model_load_s": state.get("model_load_s"),
            },
        )
    )
    while True:
        message = requests.get()
        if message is None:
            return
        eval_id, cache_steps = message
        try:
            scores = [
                {str(name): float(value) for name, value in row.items()}
                for row in evaluate(state, owned, cache_steps)
            ]
            if len(scores) != len(owned):
                raise RuntimeError(
                    f"worker {rank} scored {len(scores)} of its {len(owned)} pairs"
                )
        except BaseException:  # noqa: BLE001 - same contract as setup
            results.put(("error", rank, eval_id, traceback.format_exc()))
            return
        results.put(("scores", rank, eval_id, list(zip(owned, scores))))


class PairPool:
    """`world_size` resident model replicas, one per GPU, scoring pairs.

    `setup(rank, world_size, owned, blob)` runs once per worker and returns its
    state dict -- the pipeline, the resident metric models, and the references
    of the pairs it owns; the optional `device` and `model_load_s` keys are read
    back for the run summary.  `evaluate(state, owned, cache_steps)` returns one
    `{metric name: value}` dict per owned pair, in `owned` order.  Both must be
    module-level functions and `blob` picklable: the workers are spawned, not
    forked.

    `evaluate(cache_steps)` on the pool sends the schedule to every worker and
    reassembles the answers in pair order, so the caller sees exactly the list
    the single-GPU loop produces.
    """

    def __init__(
        self,
        *,
        world_size: int,
        n_pairs: int,
        setup: Callable[..., dict[str, Any]],
        evaluate: Callable[..., Sequence[float]],
        blob: dict[str, Any],
        tag: str = "pool",
        poll_s: float = 30.0,
    ) -> None:
        import torch.multiprocessing as mp

        if int(world_size) < 1:
            raise ValueError("world_size must be at least 1")
        if int(n_pairs) < 1:
            raise ValueError("a pool needs at least one pair")
        self.n_pairs = int(n_pairs)
        self.world_size = min(int(world_size), self.n_pairs)
        self.poll_s = float(poll_s)
        self.tag = str(tag)
        self._eval_id = 0
        self._closed = False
        devices = _pool_devices(self.world_size)
        self.owned = [
            tuple(
                index
                for index in range(self.n_pairs)
                if index % self.world_size == rank
            )
            for rank in range(self.world_size)
        ]
        context = mp.get_context("spawn")
        self.requests = [context.Queue() for _ in range(self.world_size)]
        self.results = [context.Queue() for _ in range(self.world_size)]
        self.procs: list[Any] = []
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for rank in range(self.world_size):
                # The child inherits the environment as of `start()`, so the
                # narrowing happens before it imports torch: every worker sees
                # exactly one GPU, as the shell sharders arrange it.
                os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
                proc = context.Process(
                    target=_pair_worker,
                    args=(
                        rank,
                        self.world_size,
                        self.owned[rank],
                        setup,
                        evaluate,
                        blob,
                        self.requests[rank],
                        self.results[rank],
                    ),
                    daemon=True,
                )
                proc.start()
                self.procs.append(proc)
        finally:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved

        self.device_name: str | None = None
        self.model_load_s = 0.0
        for rank in range(self.world_size):
            ready = self._receive(rank, "ready")
            if self.device_name is None:
                self.device_name = ready.get("device")
            self.model_load_s = max(
                self.model_load_s, float(ready.get("model_load_s") or 0.0)
            )
            print(
                f"[{self.tag}] worker {rank} on GPU {devices[rank]} "
                f"({ready.get('device')}) owns pairs "
                f"{','.join(str(index) for index in self.owned[rank])}",
                flush=True,
            )

    def _receive(self, rank: int, kind: str) -> Any:
        import queue as _queue

        while True:
            try:
                tag, worker, _eval_id, payload = self.results[rank].get(
                    timeout=self.poll_s
                )
            except _queue.Empty:
                if self.procs[rank].is_alive():
                    continue
                exitcode = self.procs[rank].exitcode
                self.close()
                raise RuntimeError(
                    f"{self.tag}: worker {rank} died with exit code {exitcode}; "
                    "resubmit the job with --resume"
                )
            if tag == "error":
                self.close()
                raise RuntimeError(f"{self.tag}: worker {worker} failed:\n{payload}")
            if tag != kind:
                self.close()
                raise RuntimeError(
                    f"{self.tag}: worker {worker} sent {tag!r}, expected {kind!r}"
                )
            return payload

    def evaluate(self, cache_steps: Sequence[int]) -> list[dict[str, float]]:
        """One schedule -> its per-pair metric dicts, in pair order."""

        if self._closed:
            raise RuntimeError(f"{self.tag}: the pool is closed")
        self._eval_id += 1
        message = (self._eval_id, tuple(int(step) for step in cache_steps))
        for queue in self.requests:
            queue.put(message)
        scores: list[dict[str, float] | None] = [None] * self.n_pairs
        for rank in range(self.world_size):
            for index, value in self._receive(rank, "scores"):
                scores[int(index)] = dict(value)
        missing = [index for index, value in enumerate(scores) if value is None]
        if missing:
            raise RuntimeError(f"{self.tag}: no score came back for pairs {missing}")
        return [dict(value) for value in scores]  # type: ignore[arg-type]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in self.requests:
            try:
                queue.put(None)
            except Exception:  # noqa: BLE001 - a dead worker's queue
                pass
        for proc in self.procs:
            # A worker whose last result nobody read cannot drain its queue and
            # will not exit on its own; after the grace period it is terminated.
            proc.join(timeout=30.0)
            if proc.is_alive():
                proc.terminate()


def run_probe(
    evaluate: Callable[[Sequence[int]], Sequence[float]],
    rng: np.random.Generator,
    space: SearchSpace,
    n_evals: int,
    *,
    hill_window: int = 3,
    local_probability: float = 0.7,
    metrics_of: Callable[[Sequence[int]], dict[str, Sequence[float]]] | None = None,
) -> dict[str, Any]:
    """P1 temperature reading: N/2 (schedule, one-swap neighbour) pairs.

    The median absolute swap difference fixes the order of magnitude of
    `t_max`; the calibration standard error puts a limit under `t_min`.  The
    spread of the N means is the first measured read of the landscape at K37
    and K29, recorded but not acted on.

    `metrics_of(combo)` hands back the five per-pair metrics of the evaluation
    that just ran; their pair means, in evaluation order, and the mean and
    standard deviation of each are what the objective scales and the
    objective-aware temperatures are backfilled from.  Evaluation `2t` and
    `2t + 1` are the two halves of swap pair `t`.
    """

    means: list[float] = []
    deltas: list[float] = []
    ses: list[float] = []
    metric_means: dict[str, list[float]] = {}
    for _ in range(int(n_evals) // 2):
        base = space.random_combo(rng)
        swap = space.proposal(
            rng, base, window=hill_window, local_probability=local_probability
        )
        for combo in (base, swap):
            scores = np.asarray(evaluate(combo), dtype=np.float64)
            means.append(float(scores.mean()))
            ses.append(
                float(scores.std(ddof=1) / math.sqrt(scores.size))
                if scores.size > 1
                else 0.0
            )
            if metrics_of is not None:
                for name, values in metrics_of(combo).items():
                    metric_means.setdefault(name, []).append(
                        float(np.mean(np.asarray(values, dtype=np.float64)))
                    )
        deltas.append(means[-1] - means[-2])
    absolute = np.abs(np.asarray(deltas, dtype=np.float64))
    median_abs = float(np.median(absolute)) if absolute.size else 0.0
    se = float(np.median(ses)) if ses else 0.0
    out = {
        "n_evals": len(means),
        "n_swap_pairs": len(deltas),
        "mean_psnr_db": {
            "min": float(np.min(means)) if means else None,
            "max": float(np.max(means)) if means else None,
            "mean": float(np.mean(means)) if means else None,
            "std": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        },
        "swap_delta_db": {
            "median_abs": median_abs,
            "p10_abs": float(np.percentile(absolute, 10)) if absolute.size else None,
            "p90_abs": float(np.percentile(absolute, 90)) if absolute.size else None,
        },
        "calibration_se_db": se,
        "suggested_t_max": (
            10.0 ** round(math.log10(median_abs)) if median_abs > 0 else 0.0
        ),
        "suggested_t_min": se / 10.0,
    }
    if metric_means:
        out["metric_means"] = metric_means
        out["metric_stats"] = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
            for name, values in metric_means.items()
        }
    return out


def read_eval_trace(
    path: Any,
    space: SearchSpace,
    *,
    algorithm: str,
    k: int,
    objective: str = "psnr",
) -> list[tuple[tuple[int, ...], dict[str, list[float]]]]:
    """`evals.jsonl` rows of one (search, K, algorithm, objective) job.

    Rows written before the objective flag existed carry no `objective` key and
    no `metrics`; they are the PSNR search's, and their `psnr_db` is the whole
    metric vector such a row recorded.
    """

    import json
    from pathlib import Path as _Path

    path = _Path(path)
    if not path.is_file():
        return []
    rows: list[tuple[tuple[int, ...], dict[str, list[float]]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if (
            row.get("mode") != "search"
            or row.get("algorithm") != algorithm
            or int(row.get("k", -1)) != int(k)
            or str(row.get("objective", "psnr")) != str(objective)
        ):
            continue
        metrics = row.get("metrics") or {"psnr": row["psnr_db"]}
        rows.append(
            (
                space.combo_of_bits(row["bits"]),
                {name: [float(v) for v in values] for name, values in metrics.items()},
            )
        )
    return rows


def centered_worst_pair(
    records: Sequence[dict[str, Any]]
) -> dict[tuple[int, ...], float]:
    """Per-candidate worst pair after removing each pair's own difficulty.

    Each pair's score is first reduced by that pair's mean over every
    evaluated candidate, so the hardest pair (text rendering) cannot dominate
    the gate on its own.
    """

    if not records:
        return {}
    matrix = np.asarray([row["per_pair"] for row in records], dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    worst = centered.min(axis=1)
    return {row["combo"]: float(value) for row, value in zip(records, worst)}


def select_best(
    records: Sequence[dict[str, Any]], *, se: float, se_factor: float = 2.0
) -> dict[str, Any] | None:
    """Highest mean; ties inside `se_factor * se` broken on the centered worst pair."""

    if not records:
        return None
    top = max(row["mean"] for row in records)
    tied = [row for row in records if top - row["mean"] <= se_factor * se]
    if len(tied) == 1:
        return tied[0]
    worst = centered_worst_pair(records)
    return max(tied, key=lambda row: (worst[row["combo"]], row["mean"]))


def arbitration_candidates(
    space: SearchSpace,
    records: Sequence[dict[str, Any]],
    *,
    count: int = 3,
    min_hamming: int = 4,
) -> list[dict[str, Any]]:
    """Top `count` schedules that are pairwise at least `min_hamming` apart."""

    ordered = sorted(records, key=lambda row: -row["mean"])
    picked: list[dict[str, Any]] = []
    for row in ordered:
        if len(picked) >= count:
            break
        if all(
            hamming(space, row["combo"], other["combo"]) >= min_hamming
            for other in picked
        ):
            picked.append(row)
    if len(picked) < count:
        # Not enough separated candidates: fill by score, as the plan states.
        for row in ordered:
            if len(picked) >= count:
                break
            if row not in picked:
                picked.append(row)
    return picked


# --------------------------------------------------------------------------
# P3: arbitration on the fifty held-out captions, and the delivery list
# --------------------------------------------------------------------------

#: Algorithm names in the order the plan lists them; the joined delivery name
#: of a schedule several algorithms agree on follows it.
ALGORITHM_ORDER: tuple[str, ...] = tuple(ALGORITHMS)


def bits_hamming(left: str, right: str) -> int:
    """Hamming distance between two schedule bitstrings of the same length."""

    if len(left) != len(right):
        raise ValueError(f"bitstrings differ in length: {len(left)} vs {len(right)}")
    return sum(1 for a, b in zip(left, right) if a != b)


def load_candidate_summaries(
    paths: Sequence[Any], *, model: str, k: int, objective: str = "psnr"
) -> list[dict[str, Any]]:
    """Collect the arbitration candidates of several (model, K) search summaries.

    Identical bitstrings proposed by different algorithms are one candidate:
    the schedule is generated and scored once, and every algorithm that
    proposed it is carried along with the calibration mean it measured.
    """

    import json
    from pathlib import Path as _Path

    merged: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = _Path(raw)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("mode") != "search":
            raise ValueError(
                f"{path}: mode is {summary.get('mode')!r}; arbitration reads "
                "search summaries"
            )
        if summary.get("model") != model or int(summary.get("k", -1)) != int(k):
            raise ValueError(
                f"{path}: summary is {summary.get('model')} K{summary.get('k')}, "
                f"this arbitration is {model} K{k}"
            )
        summary_objective = str(summary.get("objective", "psnr"))
        if summary_objective != str(objective):
            raise ValueError(
                f"{path}: searched for {summary_objective}, this arbitration is "
                f"{objective}"
            )
        algorithm = str(summary["algorithm"])
        rows = summary.get("arbitration_candidates") or []
        if not rows:
            raise ValueError(f"{path}: no arbitration_candidates recorded")
        for rank, row in enumerate(rows):
            entry = merged.setdefault(
                str(row["bits"]),
                {
                    "bits": str(row["bits"]),
                    "schedule": str(row["schedule"]),
                    "proposed_by": [],
                },
            )
            entry["proposed_by"].append(
                {
                    "algorithm": algorithm,
                    "rank": int(rank),
                    "calibration_mean_psnr_db": float(row["mean_psnr_db"]),
                    "calibration_mean_objective": float(
                        row.get("mean_objective", row["mean_psnr_db"])
                    ),
                    "summary_file": str(path),
                }
            )
    return list(merged.values())


def arbitration_winners(scored: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Each algorithm's delivered schedule: its own highest arbitration mean.

    The mean compared is the searched objective's; under `psnr` that is the
    PSNR mean the field has always held.
    """

    winners: dict[str, str] = {}
    best: dict[str, float] = {}
    for row in scored:
        value = float(
            row.get("arbitration_mean_objective", row["arbitration_mean_psnr_db"])
        )
        for proposer in row["proposed_by"]:
            algorithm = str(proposer["algorithm"])
            if algorithm not in best or value > best[algorithm]:
                best[algorithm], winners[algorithm] = value, str(row["bits"])
    return winners


def score_candidates(
    *,
    evaluate: ObjectiveEvaluator,
    space: SearchSpace,
    summaries: Sequence[Any],
    model: str,
    k: int,
    pair_labels: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Re-score the search summaries' candidates on the arbitration prompts.

    One generation pass per distinct bitstring, scored with the same objective
    the search used; the per-prompt objective values, their mean and their
    worst are what the delivery decision reads, and each candidate also carries
    the mean of all five metrics.
    """

    objective = evaluate.objective
    candidates = load_candidate_summaries(
        summaries, model=model, k=k, objective=objective
    )
    scored: list[dict[str, Any]] = []
    for entry in candidates:
        combo = space.combo_of_bits(entry["bits"])
        scores = [float(v) for v in evaluate(combo)]
        metrics = evaluate.metrics_of(combo)
        means = evaluate.metric_means(combo)
        scored.append(
            {
                **entry,
                "per_prompt_objective": [round(v, 6) for v in scores],
                "per_prompt_metrics": {
                    name: [round(float(v), 6) for v in values]
                    for name, values in metrics.items()
                },
                "arbitration_objective": objective,
                "arbitration_mean_objective": float(sum(scores) / len(scores)),
                "arbitration_min_objective": float(min(scores)),
                "arbitration_metrics": means,
                "per_prompt_psnr_db": [round(float(v), 6) for v in metrics["psnr"]],
                "arbitration_mean_psnr_db": float(means["psnr"]),
                "arbitration_min_psnr_db": float(min(metrics["psnr"])),
            }
        )
    return {
        "objective": objective,
        "seed": int(seed),
        "n_prompts": len(pair_labels),
        "prompt_labels": list(pair_labels),
        "summary_files": [str(path) for path in summaries],
        "n_candidates": len(scored),
        "candidates": scored,
        "winners": arbitration_winners(scored),
        "delivery": delivery_list(scored),
    }


def compact_arbitration(record: dict[str, Any]) -> dict[str, Any]:
    """The arbitration record without the per-prompt scores, for `summary.json`."""

    return {
        **{key: value for key, value in record.items() if key != "candidates"},
        "candidates": [
            {
                key: value
                for key, value in row.items()
                if key
                not in ("per_prompt_psnr_db", "per_prompt_objective", "per_prompt_metrics")
            }
            for row in record["candidates"]
        ],
    }


def delivery_list(scored: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The setting's distinct delivered schedules, algorithms that agree merged."""

    winners = arbitration_winners(scored)
    order = [name for name in ALGORITHM_ORDER if name in winners]
    order += sorted(set(winners) - set(ALGORITHM_ORDER))
    by_bits: dict[str, list[str]] = {}
    for algorithm in order:
        by_bits.setdefault(winners[algorithm], []).append(algorithm)
    rows = {str(row["bits"]): row for row in scored}
    out: list[dict[str, Any]] = []
    for bits, algorithms in by_bits.items():
        row = rows[bits]
        objective = str(row.get("arbitration_objective", "psnr"))
        entry = {
            "name": "ss_" + "+".join(algorithms),
            "bits": bits,
            "schedule": row["schedule"],
            "algorithms": list(algorithms),
            "arbitration_mean_psnr_db": float(row["arbitration_mean_psnr_db"]),
            "arbitration_min_psnr_db": float(row["arbitration_min_psnr_db"]),
        }
        if objective != "psnr":
            entry["name"] = f"ss_{objective}_" + "+".join(algorithms)
            entry["objective"] = objective
            entry["arbitration_mean_objective"] = float(
                row["arbitration_mean_objective"]
            )
            entry["arbitration_min_objective"] = float(row["arbitration_min_objective"])
        if row.get("arbitration_metrics"):
            entry["arbitration_metrics"] = dict(row["arbitration_metrics"])
        out.append(entry)
    return out
