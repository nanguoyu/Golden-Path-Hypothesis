"""Reader for the frozen baseline-matrix configuration (plan section 3).

All 162 cells of the main matrix read one file,
`resources/hunyuan_video/baseline_matrix_config.v1.json`: a threshold per
(dynamic method, dataset, budget) and a 50-step schedule per (fixed-schedule
method, budget), with MeanCache's per-edge `jvp_spans` carried alongside its
schedules because the spans are part of the solved path, not a runtime constant.

The dataset axis is on the thresholds and not on the schedules, because it is
there for one reason: a dynamic gate's threshold decides how many steps it
actually caches, and that count moves with the prompt distribution, so a
threshold frozen on one distribution does not hold the budget on another. The
image side hit exactly this and re-calibrated
(`docs/cross_model_diffusiondb_clean10k_baseline_extension_plan_zh.md` section 2,
"为什么需要新的 DiffusionDB 校准集": "prompt 分布变化也可能令 mean actual K 偏离"),
which is why its two evaluation families read two different threshold tables. A fixed schedule
realizes its K exactly on every prompt, so there is nothing about it to
calibrate per distribution and it stays dataset-free.

Four rules the plan states and this module enforces:

  - a cell with no frozen entry fails. There is no default to fall back to, and
    `entry()` raises rather than inventing one. Loading is stricter still: a
    config that does not cover all 9 methods x 2 datasets x 3 budgets is
    rejected outright, because P5 is a gate (24 thresholds, and 9 schedule
    tables filling 15 schedule cells -- BudCache and MeanCache have their own
    per budget, the triplet shares one) and a half-frozen file must not be
    able to start a matrix.
  - a frozen cell is complete. Every knob a method may freeze is frozen for
    every cell, so reading the file tells you every number its gate ran with;
    nothing silently falls back to the runner's argparse default, and no cell
    can inherit a knob calibrated for a different one.
  - TaylorSeer O1 / HiCache O2 / L2P read one and the same table per budget
    (section 3.3). The triplet exists to vary the payload at a fixed schedule,
    so a file that gives them separate tables is not a weaker version of the
    comparison, it is a different experiment, and loading it fails.
  - the file is self-hashed. `sha256` goes into every run record, so a result
    can be traced back to the configuration that produced it; a config edited
    after freezing no longer loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hunyuan_video.config import ROOT
from hunyuan_video.records import load_self_hashed_json


CONFIG_PATH = ROOT / "resources/hunyuan_video/baseline_matrix_config.v1.json"
SCHEMA = "hunyuan_video.baseline_matrix_config.v1"
HASH_FIELD = "config_sha256"

NUM_STEPS = 50
BUDGETS = ("K29", "K37", "K41")
BUDGET_CACHE_COUNTS = {"K29": 29, "K37": 37, "K41": 41}
# The two evaluation sets of plan section 1.2. Each calibrates its own
# thresholds, on its own calibration subset.
DATASETS = ("penguin599", "vbench944")

# Native gates keep their per-prompt decisions; only the threshold is frozen.
DYNAMIC_METHODS = ("seacache", "teacache", "sencache", "dicache")
# Fixed-schedule methods execute exactly K cached steps per prompt. TaylorSeer
# O1 / HiCache O2 / L2P share one table per budget (plan section 3.3).
SCHEDULE_METHODS = ("budcache", "meancache", "taylorseer_o1", "hicache_o2", "l2p")
TRIPLET_METHODS = ("taylorseer_o1", "hicache_o2", "l2p")
METHODS = DYNAMIC_METHODS + SCHEDULE_METHODS


@dataclass(frozen=True)
class ParamSpec:
    """The legal values of one frozen knob, and where that bound comes from.

    A bound here is not belt-and-braces: `method_params` reaches the adapters
    unmodified, so a value outside the adapter's domain either raises after the
    transformer is already resident on a GPU node, or -- worse -- runs and
    quietly produces a cell that is not the cell it is labelled as.
    """

    kind: type
    low: float
    high: float | None = None
    low_open: bool = False
    high_open: bool = False
    # Whether this knob may differ between the two datasets. Only what moves the
    # realized cache count may: that is the one thing the prompt distribution
    # changes and the only reason the dataset axis exists. A knob that does not
    # -- a probe depth, an exponent, a magnitude floor -- differing per dataset
    # would make one column of the results table two methods under one name.
    per_dataset: bool = False
    source: str = ""

    def check(self, where: str, name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where} {name} must be a number")
        # before the int() cast below, which raises OverflowError/ValueError on
        # these and names neither the knob nor this bound; an unbounded float
        # spec would otherwise pass inf through to hash_json, which fails with
        # no idea which cell it came from
        if isinstance(value, float) and (value != value or value in (float("inf"),
                                                                     float("-inf"))):
            raise ValueError(f"{where} {name}={value} is not finite")
        if self.kind is int and int(value) != value:
            raise ValueError(f"{where} {name}={value} must be a whole number ({self.source})")
        number = float(value)
        low_ok = number > self.low if self.low_open else number >= self.low
        high_ok = (
            True if self.high is None
            else (number < self.high if self.high_open else number <= self.high)
        )
        if not (low_ok and high_ok):
            # spelled out rather than nested in one f-string: reusing the outer
            # quote character inside a nested replacement field needs PEP 701
            # (Python 3.12), and the cluster env runs 3.10
            open_bracket = "(" if self.low_open else "["
            if self.high is None:
                close = "inf)"
            else:
                close = f"{self.high}" + (")" if self.high_open else "]")
            span = f"{open_bracket}{self.low}, {close}"
            raise ValueError(f"{where} {name}={value} is outside {span} ({self.source})")


# What each method freezes beyond its threshold, and the legal range of each.
# Anything outside this table is rejected rather than passed through: a frozen
# file naming a knob the runner does not read would be a config the matrix
# silently disobeys. Every knob listed here is mandatory for every cell of that
# method -- see `validate_payload` -- so no cell can fall back to an argparse
# default the frozen file does not record.
METHOD_PARAMS: dict[str, dict[str, ParamSpec]] = {
    "seacache": {
        "first_enhance": ParamSpec(int, 1, NUM_STEPS - 1, per_dataset=True,
                                   source="methods/seacache.py:60-63, warmup and terminal "
                                          "steps run full"),
        "power_exp": ParamSpec(float, 0.0, low_open=True,
                               source="lib/wiener.py:77 apply_sea_from_ab exponent; the "
                                      "code sets no upper bound, so neither does this"),
    },
    "teacache": {
        "first_enhance": ParamSpec(int, 1, NUM_STEPS - 1, per_dataset=True,
                                   source="methods/teacache.py:57-60, warmup and terminal "
                                          "steps run full"),
    },
    "sencache": {
        # the gate's own default and the port's convention; _method_config
        # clamps to 3, so a frozen 1 or 2 would be a number the run disobeys
        "first_enhance": ParamSpec(int, 3, NUM_STEPS - 1, per_dataset=True,
                                   source="methods/sencache.py:48 and backend.py:434 both say 3"),
        # a threshold in its own right -- it is what the first 20% of steps are
        # scored against -- so it is swept and frozen per dataset like one
        "sencache_threshold_start": ParamSpec(float, 0.0, low_open=True, per_dataset=True,
                                              source="_validate requires it > 0"),
    },
    "dicache": {
        "dicache_ret_ratio": ParamSpec(float, 0.0, 1.0, high_open=True, per_dataset=True,
                                       source="dicache.py warmup, image-side gate semantics: "
                                              "steps 0..int(ratio*50) and the terminal step "
                                              "forced full"),
        "dicache_probe_depth": ParamSpec(int, 1,
                                         source="dicache.py:60, 1 <= depth <= len(double_blocks); "
                                                "the upper bound is the loaded stack's"),
    },
    "budcache": {},
    "meancache": {},
    "taylorseer_o1": {},
    "hicache_o2": {
        "sigma": ParamSpec(float, 0.0, 1.0, low_open=True,
                           source="methods/hicache.py:19, sigma must be in (0, 1]"),
        # pinned rather than ranged: FORBIDDEN_CACHE_STEPS below and the table
        # the triplet shares are both built for a 3-step warmup, and moving it
        # invalidates a table TaylorSeer and L2P also read
        "first_enhance": ParamSpec(int, 3, 3,
                                   source="backend.py:362 forbids range(first_enhance); the "
                                          "shared triplet table is built for 3"),
    },
    "l2p": {
        # the upper end is a degeneracy guard, not a fitted domain: lib/l2p.py
        # drops every coefficient with |coef| <= this, and a floor above all of
        # them makes predict_l2p return the last history value unchanged --
        # which is BudCache's payload running under L2P's name. Ridge
        # coefficients on this predictor are order 1, so 1.0 is the point past
        # which the method certainly stops being itself.
        "l2p_min_abs_weight": ParamSpec(float, 0.0, 1.0, high_open=True,
                                        source="methods/l2p.py:23 floor, lib/l2p.py:123 "
                                               "drops |coef| <= it"),
    },
}

# Inputs that are files rather than numbers, and that decide what the method
# does: SenCache scores every step against its sensitivity table, and L2P's
# cached payload IS the predictor in its weights file. A cell that runs against
# a different file than its threshold was calibrated on is off-budget, so the
# config freezes the file's digest and the runner refuses a file that does not
# match. This is the `source_sha256` the schedule tables already carry, applied
# to the two remaining inputs.
METHOD_ASSETS: dict[str, tuple[str, ...]] = {
    "sencache": ("sencache_sensitivity_path",),
    "l2p": ("l2p_weights",),
}


# SenCache reads three more inputs that bound its cache count, and the runner
# has no flag for any of them, so in this lane they are constants: `_method_config`
# never sets them and `backend.py:435-438` resolves them from these defaults.
# They are not free parameters, which is exactly why they belong in the bound
# rather than in an apology for its looseness.
SENCACHE_LANE_MAX_SKIP = 10
SENCACHE_LANE_CUTOFF_STEPS = -1
SENCACHE_LANE_RET_STEPS = 0


def max_cacheable_steps(method: str, params: Mapping[str, float]) -> int:
    """The most steps a dynamic gate could cache under `params`.

    Computed by walking the trajectory and asking each gate's own forced-full
    predicate, with its score answering "cache" every time -- so it is an exact
    upper bound rather than a hand-derived formula that can drift from the gate
    it describes. A gate reaches it only if the threshold is loose enough to
    cache everything it is allowed to, which is what the sweep measures; what
    this catches is the cell no sweep could rescue, because the forced-full
    steps alone already leave fewer than K.

    The predicates:
      seacache / teacache  step < first_enhance, or the terminal step
                           (methods/seacache.py:60-63, methods/teacache.py:57-60)
      sencache             step < first_enhance, step 0, step >= cutoff_step,
                           step < ret_steps, or max_skip consecutive skips
                           (methods/sencache.py:94-99, :135)
      dicache              step <= int(ret_ratio * num_steps), plus the
                           terminal step (image-side gate semantics)
    """
    if method in {"seacache", "teacache"}:
        forced = {step for step in range(NUM_STEPS)
                  if step < int(params["first_enhance"]) or step == NUM_STEPS - 1}
        return NUM_STEPS - len(forced)
    if method == "dicache":
        # step <= int(ratio * N) forced, plus the terminal step
        # (dicache.py, aligned with flux/dicache_native.py:166-173)
        warmup_last = int(float(params["dicache_ret_ratio"]) * NUM_STEPS)
        forced = {step for step in range(NUM_STEPS)
                  if step <= warmup_last or step == NUM_STEPS - 1}
        return NUM_STEPS - len(forced)
    if method == "sencache":
        first_enhance = int(params["first_enhance"])
        cutoff_step = (NUM_STEPS - 1 if SENCACHE_LANE_CUTOFF_STEPS < 0
                       else SENCACHE_LANE_CUTOFF_STEPS)
        cached = consecutive = 0
        for step in range(NUM_STEPS):
            forced = (step < first_enhance or step == 0
                      or step >= max(min(cutoff_step, NUM_STEPS), 0)
                      or step < SENCACHE_LANE_RET_STEPS
                      # the gate stops caching once it has skipped this many in
                      # a row, whatever the score says (methods/sencache.py:135)
                      or consecutive >= SENCACHE_LANE_MAX_SKIP)
            if forced:
                consecutive = 0
            else:
                cached += 1
                consecutive += 1
        return cached
    raise KeyError(f"{method} is not a dynamic gate with a frozen warmup")


# Which runner mode implements each matrix method. BudCache is a searched
# schedule executed with whole-transformer residual reuse (plan section 2.3),
# and `reuse_exact` is that payload; the other eight are one-to-one.

METHOD_RUNNER_MODE: dict[str, str] = {
    "seacache": "seacache",
    "teacache": "teacache",
    "sencache": "sencache",
    "dicache": "dicache",
    "budcache": "reuse_exact",
    "meancache": "meancache_exact",
    "taylorseer_o1": "taylorseer_exact",
    "hicache_o2": "hicache_exact",
    "l2p": "l2p_output_exact",
}
RUNNER_MODE_METHOD: dict[str, str] = {mode: method
                                      for method, mode in METHOD_RUNNER_MODE.items()}

# Steps each method may not cache — each entry is the rule its own producer or
# adapter enforces, not a uniform "first and last stay full": TaylorSeer from
# `hunyuan_video/backend.py:333`, HiCache's warmup from `:362`, L2P from `:384`
# (which forbids step 0 alone, and nothing else — plan section 3.3, "L2P 只禁
# 0"), MeanCache's mandatory `first_full_steps=5` / `last_full_steps=1` (plan
# section 2.4 item 4), and BudCache from the search that produces its tables
# (`analysis/search_budcache_hunyuan.py::forced_full_steps` = {0, 1, 2, 49}), so
# a frozen BudCache table the search could never have emitted is rejected here.
# A table shared by the triplet has to satisfy all three at once, which is the
# {0, 1, 2, 49} of section 3.3.
FORBIDDEN_CACHE_STEPS: dict[str, frozenset[int]] = {
    "budcache": frozenset({0, 1, 2, NUM_STEPS - 1}),
    "meancache": frozenset(range(5)) | frozenset({NUM_STEPS - 1}),
    "taylorseer_o1": frozenset({0, NUM_STEPS - 1}),
    "hicache_o2": frozenset({0, 1, 2, NUM_STEPS - 1}),
    "l2p": frozenset({0}),
}


@dataclass(frozen=True)
class MatrixEntry:
    """One (method, dataset, budget) cell. One of `threshold` / `cache_steps`."""

    method: str
    budget: str
    dataset: str
    cache_count: int
    threshold: float | None = None
    table_id: str | None = None
    cache_steps: tuple[int, ...] = ()
    jvp_spans: Mapping[int, int] | None = None
    # Knobs beyond the threshold that decide the realized cache count, so they
    # have to be frozen with it. DiCache is the reason this is not optional:
    # steps 0..int(ret_ratio * 50) plus the terminal step are forced full, so
    # at the default 0.2 the budget cannot exceed 38 and K41 is only reachable
    # by moving it. The image side's frozen table carries the same two columns
    # (resources/diffusiondb_clean10k_native_calibration_results).
    method_params: Mapping[str, float] = field(default_factory=dict)
    # `runner argument name -> sha256` for this method's file inputs
    assets: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineMatrixConfig:
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def entry(self, method: str, budget: str, dataset: str) -> MatrixEntry:
        """Look up one cell. Raises `KeyError` when the cell is not frozen."""
        if dataset not in DATASETS:
            raise KeyError(f"unknown dataset: {dataset}; the matrix has {list(DATASETS)}")
        if method in DYNAMIC_METHODS:
            threshold = (self.payload["thresholds"].get(method, {})
                         .get(dataset, {}).get(budget))
            if threshold is None:
                raise KeyError(f"no frozen threshold for ({method}, {dataset}, {budget}) "
                               f"in {self.path}")
            return MatrixEntry(
                method=method,
                budget=budget,
                dataset=dataset,
                cache_count=BUDGET_CACHE_COUNTS[budget],
                threshold=float(threshold),
                method_params=self._params(method, budget, dataset),
                assets=self._assets(method),
            )
        if method in SCHEDULE_METHODS:
            # no dataset axis: a fixed schedule realizes its K exactly on every
            # prompt, so there is nothing about it to calibrate per distribution
            table_id = self.payload["schedules"].get(method, {}).get(budget)
            if table_id is None:
                raise KeyError(f"no frozen schedule for ({method}, {budget}) in {self.path}")
            table = self.payload["schedule_tables"][table_id]
            spans = table.get("jvp_spans")
            return MatrixEntry(
                method=method,
                budget=budget,
                dataset=dataset,
                cache_count=BUDGET_CACHE_COUNTS[budget],
                table_id=table_id,
                cache_steps=tuple(int(step) for step in table["cache_steps"]),
                jvp_spans=(
                    {int(step): int(span) for step, span in spans.items()}
                    if spans is not None
                    else None
                ),
                method_params=self._params(method, budget, dataset),
                assets=self._assets(method),
            )
        raise KeyError(f"unknown baseline-matrix method: {method}")

    def _params(self, method: str, budget: str, dataset: str) -> dict[str, float]:
        """The knobs frozen alongside this cell, empty when it freezes none.

        Never partial: `validate_payload` refuses a file where a method freezes
        some of its knobs and not others, so this is either all of them or none.
        """
        per_method = (self.payload.get("method_params") or {}).get(method) or {}
        per_dataset = per_method.get(dataset) or {}
        return {name: float(value) for name, value in (per_dataset.get(budget) or {}).items()}

    def calibration(self, dataset: str) -> dict[str, str]:
        """The prompt file this dataset's thresholds were swept on."""
        row = (self.payload.get("threshold_calibration") or {}).get(dataset) or {}
        return {"file": str(row.get("file", "")), "sha256": str(row.get("sha256", ""))}

    def _assets(self, method: str) -> dict[str, str]:
        """`runner argument name -> sha256` of this method's file inputs."""
        return {name: str(digest) for name, digest
                in ((self.payload.get("method_assets") or {}).get(method) or {}).items()}

    def provenance(self) -> dict[str, str]:
        """The fields every run record carries so a result traces to its config."""
        return {"matrix_config": str(self.path), "matrix_config_sha256": self.sha256}


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Reject anything the matrix must not be started from."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema: {payload.get('schema')!r}")
    if payload.get("num_steps") != NUM_STEPS:
        raise ValueError(f"the baseline matrix is frozen to {NUM_STEPS} steps")
    if payload.get("budgets") != BUDGET_CACHE_COUNTS:
        raise ValueError(f"budgets must be {BUDGET_CACHE_COUNTS}")

    thresholds = payload.get("thresholds") or {}
    if set(thresholds) != set(DYNAMIC_METHODS):
        raise ValueError(f"thresholds must cover exactly {list(DYNAMIC_METHODS)}")
    for method, per_dataset in thresholds.items():
        if set(per_dataset) != set(DATASETS):
            raise ValueError(
                f"{method} thresholds must cover exactly {list(DATASETS)}; a gate's "
                f"threshold decides how many steps it caches and that count moves with "
                f"the prompt distribution, so each dataset carries its own")
        for dataset, per_budget in per_dataset.items():
            if set(per_budget) != set(BUDGETS):
                raise ValueError(
                    f"{method}/{dataset} thresholds must cover exactly {list(BUDGETS)}")
            for budget, value in per_budget.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{method}/{dataset}/{budget} threshold must be a number")
                if not float(value) > 0.0:
                    raise ValueError(f"{method}/{dataset}/{budget} threshold must be positive")

    tables = payload.get("schedule_tables") or {}
    schedules = payload.get("schedules") or {}
    if set(schedules) != set(SCHEDULE_METHODS):
        raise ValueError(f"schedules must cover exactly {list(SCHEDULE_METHODS)}")
    referenced: set[str] = set()
    for method, per_budget in schedules.items():
        if set(per_budget) != set(BUDGETS):
            raise ValueError(f"{method} schedules must cover exactly {list(BUDGETS)}")
        for budget, table_id in per_budget.items():
            if table_id not in tables:
                raise ValueError(f"{method}/{budget} references unknown table {table_id!r}")
            referenced.add(table_id)
            _validate_table(tables[table_id], table_id, method=method, budget=budget)
    for budget in BUDGETS:
        # The triplet's whole point is that the payload differs while the
        # schedule is held fixed (plan section 3.3, "三方法读同一张"); one table
        # per method would silently destroy that comparison.
        shared = {schedules[method][budget] for method in TRIPLET_METHODS}
        if len(shared) != 1:
            raise ValueError(
                f"{budget}: {list(TRIPLET_METHODS)} must read one shared table, got "
                f"{sorted(shared)}"
            )
    _validate_method_params(payload.get("method_params") or {})
    _validate_method_assets(payload.get("method_assets") or {})
    _validate_calibration(payload.get("threshold_calibration") or {})

    unused = sorted(set(tables) - referenced)
    if unused:
        raise ValueError(f"schedule tables no cell reads: {unused}")


def _validate_method_params(params: Mapping[str, Any]) -> None:
    """Every knob of every method, for every cell, inside its declared range."""
    expected = {method for method, spec in METHOD_PARAMS.items() if spec}
    if set(params) != expected:
        missing = sorted(expected - set(params))
        stray = sorted(set(params) - expected)
        raise ValueError(
            f"method_params must cover exactly the methods that freeze knobs "
            f"{sorted(expected)}; missing {missing}, unexpected {stray}")
    for method, per_dataset in params.items():
        if set(per_dataset) != set(DATASETS):
            raise ValueError(f"{method} method_params must cover exactly {list(DATASETS)}")
        for dataset, per_budget in per_dataset.items():
            if set(per_budget) != set(BUDGETS):
                raise ValueError(
                    f"{method}/{dataset} method_params must cover exactly {list(BUDGETS)}")
            for budget, knobs in per_budget.items():
                where = f"{method}/{dataset}/{budget}"
                if set(knobs) != set(METHOD_PARAMS[method]):
                    # a knob the runner does not read is a config the matrix
                    # silently disobeys; a knob left out silently reverts to an
                    # argparse default the frozen file does not record. Both are
                    # the failure freezing exists to stop.
                    raise ValueError(
                        f"{where} must freeze exactly {sorted(METHOD_PARAMS[method])}, "
                        f"got {sorted(knobs)}")
                for name, value in knobs.items():
                    METHOD_PARAMS[method][name].check(where, name, value)
                if method in DYNAMIC_METHODS:
                    reachable = max_cacheable_steps(method, knobs)
                    if reachable < BUDGET_CACHE_COUNTS[budget]:
                        raise ValueError(
                            f"{where} forces enough steps full that at most {reachable} "
                            f"can be cached, below the {BUDGET_CACHE_COUNTS[budget]} this "
                            f"budget needs; no threshold can rescue this cell")
    for method, per_dataset in params.items():
        for name, spec in METHOD_PARAMS[method].items():
            if spec.per_dataset:
                continue
            for budget in BUDGETS:
                values = {_json_key(per_dataset[dataset][budget][name])
                          for dataset in DATASETS}
                if len(values) != 1:
                    raise ValueError(
                        f"{method}/{budget} freezes a different {name} per dataset. Only "
                        f"what moves the realized cache count is calibrated per prompt "
                        f"distribution; {name} is not, so it must match across "
                        f"{list(DATASETS)}")


def _json_key(value: Any) -> str:
    """A stable string for comparing two nested plain-JSON structures."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_calibration(calibration: Mapping[str, Any]) -> None:
    """Which prompt file each dataset's thresholds were swept on.

    Thresholds are the only input the config takes as a bare number -- schedules
    and file inputs both carry a digest -- so without this the dataset axis is
    unfalsifiable: one sweep's 12 numbers pasted under both dataset keys is a
    perfectly well-formed file. Two datasets pointing at the same calibration
    file is refused, which is what that mistake looks like.
    """
    if set(calibration) != set(DATASETS):
        raise ValueError(
            f"threshold_calibration must name exactly {list(DATASETS)}, got "
            f"{sorted(calibration)}; a threshold with no record of what it was swept "
            f"on cannot be checked against anything")
    digests: dict[str, str] = {}
    for dataset, row in calibration.items():
        digest = (row or {}).get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or \
                any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{dataset} threshold_calibration needs a lowercase sha256 "
                             f"hex digest of its prompt file")
        if not (row or {}).get("file"):
            raise ValueError(f"{dataset} threshold_calibration needs the prompt file it "
                             f"was swept on")
        digests[dataset] = digest
    if len(set(digests.values())) != len(DATASETS):
        raise ValueError(
            f"every dataset's thresholds were swept on the same prompt file "
            f"{sorted(set(digests.values()))}; then only one distribution was calibrated "
            f"and the dataset axis records a difference that does not exist")


def _validate_method_assets(assets: Mapping[str, Any]) -> None:
    """The digest of every file input, for exactly the methods that read one."""
    if set(assets) != set(METHOD_ASSETS):
        raise ValueError(
            f"method_assets must cover exactly {sorted(METHOD_ASSETS)}, got {sorted(assets)}")
    for method, per_name in assets.items():
        if set(per_name) != set(METHOD_ASSETS[method]):
            raise ValueError(
                f"{method} method_assets must name exactly {list(METHOD_ASSETS[method])}")
        for name, digest in per_name.items():
            if not isinstance(digest, str) or len(digest) != 64 or \
                    any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{method}/{name} must be a lowercase sha256 hex digest")


def _validate_table(table: Mapping[str, Any], table_id: str, *, method: str, budget: str) -> None:
    steps = table.get("cache_steps")
    if not isinstance(steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in steps
    ):
        raise TypeError(f"table {table_id} cache_steps must be a list of integers")
    if steps != sorted(set(steps)):
        raise ValueError(f"table {table_id} cache_steps must be sorted and unique")
    if len(steps) != BUDGET_CACHE_COUNTS[budget]:
        raise ValueError(
            f"table {table_id} has {len(steps)} cached steps, {budget} needs "
            f"{BUDGET_CACHE_COUNTS[budget]}"
        )
    if steps and (steps[0] < 0 or steps[-1] >= NUM_STEPS):
        raise ValueError(f"table {table_id} cache_steps outside the {NUM_STEPS}-step trajectory")
    forbidden = sorted(FORBIDDEN_CACHE_STEPS[method].intersection(steps))
    if forbidden:
        raise ValueError(f"table {table_id} caches steps {method} must run full: {forbidden}")

    spans = table.get("jvp_spans")
    if method == "meancache":
        if not isinstance(spans, dict):
            raise TypeError(f"table {table_id} must carry MeanCache per-edge jvp_spans")
        keys = sorted(int(step) for step in spans)
        if keys != steps:
            raise ValueError(f"table {table_id} jvp_spans must cover exactly its cached steps")
        for step, span in spans.items():
            if isinstance(span, bool) or not isinstance(span, int) or span < 1:
                raise ValueError(f"table {table_id} jvp_span for step {step} must be a positive int")
    elif spans is not None:
        raise ValueError(f"table {table_id} carries jvp_spans but is read by {method}")


def load_matrix_config(path: Path = CONFIG_PATH) -> BaselineMatrixConfig:
    """Load and fully validate the frozen configuration."""
    path = Path(path)
    payload = load_self_hashed_json(path, HASH_FIELD)
    validate_payload(payload)
    return BaselineMatrixConfig(path=path, sha256=str(payload[HASH_FIELD]), payload=payload)
