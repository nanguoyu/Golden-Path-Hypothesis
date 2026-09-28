"""Reader for the frozen Wan2.1 baseline-matrix configuration (plan section 3).

Twin of `hunyuan_video/matrix_config.py`, for the Wan2.1 t2v-1.3B lane. All 162
cells of the main matrix read one file,
`resources/wan21/baseline_matrix_config.v1.json`: a threshold per (dynamic
method, dataset, budget) and a 50-step schedule per (fixed-schedule method,
budget), with MeanCache's per-edge `jvp_spans` carried alongside its schedules
because the spans are part of the solved path, not a runtime constant.

The dataset axis is on the thresholds and not on the schedules, for the reason
the plan gives in section 3.1: a dynamic gate's threshold decides how many steps
it actually caches, and that count moves with the prompt distribution, so a
threshold frozen on one distribution does not hold the budget on another. A
fixed schedule realizes its K exactly on every prompt, so there is nothing about
it to calibrate per distribution and it stays dataset-free (section 3.2).

Four load rules, the same four the Hunyuan twin enforces:

  - a cell with no frozen entry fails. `entry()` raises rather than inventing a
    default, and loading is stricter still: a config that does not cover all
    9 methods x 2 datasets x 3 budgets is rejected outright, because P5 is a
    gate (24 thresholds, and 9 schedule tables filling 15 schedule cells).
  - a frozen cell is complete. Every knob a method may freeze is frozen for
    every cell, so reading the file tells you every number its gate ran with and
    nothing falls back to the runner's argparse default.
  - TaylorSeer O1 / HiCache O2 / L2P read one and the same table per budget
    (section 3.3). The triplet exists to vary the payload at a fixed schedule,
    so separate tables would be a different experiment, and loading fails.
  - the file is self-hashed, and `config_sha256` goes into every run record.

What is Wan-specific here, beyond the schema string:

  * **Method ids are the runner's modes.** `wan21/backend.py::build_adapter`
    dispatches on the plan's section 3.3 keys directly, so unlike the Hunyuan
    lane -- whose runner modes predate the matrix and need a
    `METHOD_RUNNER_MODE` translation -- there is only one vocabulary.
  * **Knob names carry their method.** The Hunyuan lane shares one
    `--first_enhance` across Sea/Tea/Sen/HiCache and repairs the collision with
    a `max(3, ...)` clamp inside `_method_config`, which means a frozen 1 and a
    running 3 are both defensible readings of the same file. The Wan runner is
    new, so each method gets its own argparse dest and the clamp disappears:
    what the file says is what runs.
  * **DiCache's probe depth is bounded by Wan's stack**, 1..30
    (`wan21/dicache.py:85`), not by a Hunyuan double-block count.
  * **Forbidden steps are imported, not restated.**
    `wan21.methods_glue.forbidden_cache_steps` derives them from each method's
    own warmup constants, so a change to `first_enhance` cannot leave this
    validator behind.
  * **The datasets are the Hunyuan lane's frozen files, referenced by digest.**
    Plan section 1.2 forbids copying them into `resources/wan21/` -- two copies
    of "the same" dataset can fork -- so the config carries their sha256 and the
    runner refuses a manifest that does not match.

`ParamSpec` is imported from the Hunyuan twin rather than copied: it is a
schema-free validation primitive with no Hunyuan constants in it, and the plan's
section 2.3 reuse rule ("additive import, no edits") covers exactly this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hunyuan_video.matrix_config import ParamSpec
from hunyuan_video.records import load_self_hashed_json
from wan21.methods_glue import (
    HICACHE_FIRST_ENHANCE,
    MATRIX_METHODS,
    MEANCACHE_JVP_SPAN,
    SENCACHE_CUTOFF_STEPS,
    SENCACHE_FIRST_ENHANCE,
    SENCACHE_MAX_SKIP,
    SENCACHE_RET_STEPS,
    SENCACHE_SWITCH_RATIO,
    SHARED_SCHEDULE_METHODS,
    TAYLORSEER_FIRST_ENHANCE,
    TEACACHE_RET_STEPS,
    WAN_NUM_LAYERS,
    forbidden_cache_steps,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "resources/wan21/baseline_matrix_config.v1.json"
SCHEMA = "wan21.baseline_matrix_config.v1"
HASH_FIELD = "config_sha256"

#: The generation protocol every cell runs at (plan section 1.1, first named
#: there). `wan21/backend.py::validate_protocol` hard-fails on each constant;
#: freezing the id here is what stops a config swept at one resolution from
#: being read by a run at another.
PROTOCOL_ID = "WAN-CachePaper-480"

NUM_STEPS = 50
BUDGETS = ("K29", "K37", "K41")
BUDGET_CACHE_COUNTS = {"K29": 29, "K37": 37, "K41": 41}

#: The two evaluation sets of plan section 1.2. Each calibrates its own
#: thresholds, on its own 48-prompt calibration subset. Same names as the
#: Hunyuan matrix on purpose: the two backbones' cells pair up one-to-one.
DATASETS = ("penguin599", "vbench944")

#: Frozen prompt manifests, reused verbatim from the Hunyuan lane (plan section
#: 1.2: pure text, model-independent, and copying them would create two
#: forkable copies of one dataset). Structured JSON rather than one prompt per
#: line because two Penguin prompts contain a literal newline.
EVALUATION_MANIFESTS = {
    "penguin599": ROOT / "resources/hunyuan_video/evaluation/penguin599.json",
    "vbench944": ROOT / "resources/hunyuan_video/evaluation/vbench944.json",
}
#: The manifests keep their producer's schema string; they are the Hunyuan
#: lane's frozen artifacts, read here rather than reissued.
EVALUATION_MANIFEST_SCHEMA = "hunyuan_video.evaluation_prompts.v1"

#: Threshold-calibration sets, 48 prompts each, also reused (plan section 3.1).
CALIBRATION_PROMPTS = {
    "penguin599": ROOT / "resources/hunyuan_video/calibration/penguin_b-cal48.txt",
    "vbench944": ROOT / "resources/hunyuan_video/calibration/vbench_cal48.txt",
}

#: Where the three fitted assets are fitted (plan section 3.2b). Not part of the
#: frozen payload -- the assets enter it by digest -- but the binding belongs
#: with the rest of the lane's file identities.
ASSET_FIT_PROMPTS = ROOT / "resources/baseline_exact/hunyuan_calibration50.txt"
ASSET_HOLDOUT_PROMPTS = ROOT / "resources/baseline_exact/hunyuan_holdout10.txt"
ASSET_FIT_BASE_SEED = 20260723

# Native gates keep their per-prompt decisions; only the threshold is frozen.
DYNAMIC_METHODS = ("seacache", "teacache", "sencache", "dicache")
# Fixed-schedule methods execute exactly K cached steps per prompt. TaylorSeer
# O1 / HiCache O2 / L2P share one table per budget (plan section 3.3).
SCHEDULE_METHODS = ("budcache", "meancache", "taylorseer_o1", "hicache_o2", "l2p")
TRIPLET_METHODS = SHARED_SCHEDULE_METHODS
METHODS = DYNAMIC_METHODS + SCHEDULE_METHODS

#: What `--mode` accepts. The uncached reference has no frozen entry.
RUNNER_MODES = ("original",) + METHODS

if set(METHODS) != set(MATRIX_METHODS):
    # One vocabulary, two files: a method the config knows and the backend
    # cannot build (or the reverse) would only surface on a GPU node.
    raise RuntimeError(
        f"matrix methods disagree with wan21.methods_glue: "
        f"{sorted(set(METHODS) ^ set(MATRIX_METHODS))}"
    )


# What each method freezes beyond its threshold, and the legal range of each.
# Anything outside this table is rejected rather than passed through: a frozen
# file naming a knob the runner does not read would be a config the matrix
# silently disobeys. Every knob listed here is mandatory for every cell of that
# method (see `_validate_method_params`), so no cell can fall back to an
# argparse default the frozen file does not record.
#
# Each name is also the runner's argparse dest, which is how
# `baseline_screen_runner._apply_matrix_config` sets it.
METHOD_PARAMS: dict[str, dict[str, ParamSpec]] = {
    "seacache": {
        "seacache_first_enhance": ParamSpec(
            int, 1, NUM_STEPS - 1, per_dataset=True,
            source="wan21/methods_glue.py:332-336, steps below first_enhance and the "
                   "terminal step run full"),
        "seacache_power_exp": ParamSpec(
            float, 0.0, low_open=True,
            source="wan21/seacache.py SEA exponent (3.0 in the validated golden-path "
                   "lane); lib/wiener.py:77 sets no upper bound, so neither does this"),
    },
    "teacache": {
        # Pinned, not ranged: the official use_ret_steps branch fits its quartic
        # with `ret_steps = 5*2` calls, and `lib/teacache_coeffs.py:37` is that
        # fit. Moving the warmup would leave the coefficients describing a
        # different state machine (plan sections 2.2 item 3 / 2.6).
        "teacache_ret_steps": ParamSpec(
            int, TEACACHE_RET_STEPS, TEACACHE_RET_STEPS,
            source="reference/teacache/code/TeaCache4Wan2.1/teacache_generate.py "
                   "use_ret_steps branch: ret_steps = 5 solver steps, and the "
                   "('wan21','1.3b') coefficients are fitted with it"),
    },
    "sencache": {
        # The upstream Wan variant has no warmup at all (`retention_steps = 0`);
        # 3 is this repository's paradigm requirement (plan section 2.2 item 1),
        # recorded as a deviation in section 2.4.
        "sencache_first_enhance": ParamSpec(
            int, SENCACHE_FIRST_ENHANCE, NUM_STEPS - 1, per_dataset=True,
            source="wan21/methods_glue.py:77 and plan section 2.4: this lane freezes a "
                   "3-step warmup where upstream Wan has none"),
        # A threshold in its own right -- it is what the first `switch_ratio` of
        # the trajectory is scored against -- so it is swept and frozen per
        # dataset like one. `flux/sencache.py:284` makes it a required argument.
        "sencache_threshold_start": ParamSpec(
            float, 0.0, low_open=True, per_dataset=True,
            source="flux/sencache.py:284, no default; the image side froze 0.8 on all "
                   "three backbones"),
        # The two knobs that set how many steps the gate can reach at all. The
        # paper ablates `n` (this repo's max_skip) in its section 4, so it is a
        # speed knob and belongs in the frozen file next to the threshold; the
        # strict window it interacts with has to be frozen with it or the
        # ceiling the cell ran under is not recoverable from this file.
        "sencache_max_skip": ParamSpec(
            int, 1, NUM_STEPS - 1, per_dataset=True,
            source="upstream `sencache_K` default 10; raised per budget so the "
                   "structural ceiling clears the target, see the recalibration "
                   "plan section 9.1"),
        "sencache_switch_ratio": ParamSpec(
            float, 0.0, 1.0, per_dataset=True,
            source="upstream `total_calls * 0.2`; shrunk only where max_skip alone "
                   "cannot reach the budget, see the recalibration plan section 9.1"),
    },
    "dicache": {
        "dicache_ret_ratio": ParamSpec(
            float, 0.0, 1.0, high_open=True, per_dataset=True,
            source="wan21/dicache.py warmup: steps 0..int(ratio*50) plus the terminal "
                   "step are forced full (image-side locked semantics, note the <=)"),
        "dicache_probe_depth": ParamSpec(
            int, 1, WAN_NUM_LAYERS,
            source="wan21/dicache.py:85 requires 1 <= probe_depth <= len(model.blocks), "
                   "and Wan2.1 t2v-1.3B has 30 blocks"),
    },
    "budcache": {},
    "meancache": {},
    "taylorseer_o1": {
        # Pinned rather than ranged, and not omitted: O1's order clamp is
        # numerically invisible on a table that keeps steps 0-2 full, but plan
        # section 2.2 item 1 requires it to be implemented, and freezing it is
        # how the file records that it ran.
        "taylorseer_first_enhance": ParamSpec(
            int, TAYLORSEER_FIRST_ENHANCE, TAYLORSEER_FIRST_ENHANCE,
            source="hunyuan_video/methods/taylorseer.py:49-62 warmup order clamp; plan "
                   "section 2.2 item 1 pins it at 3"),
    },
    "hicache_o2": {
        "hicache_sigma": ParamSpec(
            float, 0.0, 1.0, low_open=True,
            source="hunyuan_video/methods/hicache.py:19, sigma must be in (0, 1]"),
        # The shared triplet table is built to avoid {0, 1, 2, 49}; moving this
        # warmup invalidates a table TaylorSeer and L2P also read.
        "hicache_first_enhance": ParamSpec(
            int, HICACHE_FIRST_ENHANCE, HICACHE_FIRST_ENHANCE,
            source="reference/hicache/code/models/hicache_fast_impl.py:130; the shared "
                   "triplet table is built for a 3-step warmup"),
    },
    "l2p": {
        # The upper end is a degeneracy guard, not a fitted domain: lib/l2p.py
        # drops every coefficient with |coef| <= this, and a floor above all of
        # them makes the predictor return the last history value unchanged --
        # which is BudCache's payload running under L2P's name.
        "l2p_min_abs_weight": ParamSpec(
            float, 0.0, 1.0, high_open=True,
            source="hunyuan_video/methods/l2p.py:23 floor, lib/l2p.py:123 drops "
                   "|coef| <= it"),
    },
}

# Inputs that are files rather than numbers, and that decide what the method
# does: SenCache scores every step against its Wan sensitivity table, and L2P's
# cached payload IS the predictor in its weights file. A cell that runs against
# a different file than its threshold was calibrated on is off-budget, so the
# config freezes the digest (not the path -- the files live under $DATA and the
# matrix runs on more than one cluster) and the runner refuses a mismatch.
METHOD_ASSETS: dict[str, tuple[str, ...]] = {
    "sencache": ("sencache_sensitivity_path",),
    "l2p": ("l2p_weights",),
}

# SenCache reads three more inputs that bound its cache count, and the runner
# has no flag for any of them: in this lane they are constants, taken from the
# upstream Wan variant (plan section 2.5). They are not free parameters, which
# is why they belong inside the reachability bound rather than beside it.
SENCACHE_LANE_MAX_SKIP = SENCACHE_MAX_SKIP
SENCACHE_LANE_CUTOFF_STEPS = SENCACHE_CUTOFF_STEPS
SENCACHE_LANE_RET_STEPS = SENCACHE_RET_STEPS
SENCACHE_LANE_SWITCH_RATIO = SENCACHE_SWITCH_RATIO

#: SeaCache's SEA filtering mode and TeaCache's coefficient row are single-valued
#: on this backbone, so they are lane constants carried into `method_config` for
#: provenance rather than freezable knobs.
SEACACHE_LANE_NORM_MODE = "mean"
TEACACHE_LANE_VARIANT = "1.3b"
#: The per-edge spans of a frozen MeanCache table cover every cached step, so
#: this fallback is never consulted; it is recorded so the run record states it.
MEANCACHE_LANE_JVP_SPAN = MEANCACHE_JVP_SPAN


def max_cacheable_steps(method: str, params: Mapping[str, float]) -> int:
    """The most steps a dynamic gate could cache under `params`.

    Computed by walking the 50-step trajectory and asking each gate's own
    forced-full predicate, with its score answering "cache" every time -- an
    exact upper bound rather than a hand-derived formula that can drift from the
    gate it describes. A gate reaches it only if the threshold is loose enough,
    which is what the sweep measures; what this catches is the cell no sweep
    could rescue, because the forced-full steps alone already leave fewer than K.

    The predicates:
      seacache   step < first_enhance, or the terminal step
                 (`wan21/methods_glue.py:332-336`)
      teacache   step < ret_steps, or the terminal step -- the terminal clamp is
                 this lane's recorded deviation from upstream, which never
                 forces it (`wan21/methods_glue.py:409-414`)
      sencache   step < first_enhance, step 0, step >= cutoff_step,
                 step < ret_steps, or max_skip consecutive skips
                 (`hunyuan_video/methods/sencache.py:89-99,135`)
      dicache    step <= int(ret_ratio * num_steps), plus the terminal step
                 (`wan21/dicache.py:143-151`)
    """
    if method == "seacache":
        warmup = int(params["seacache_first_enhance"])
        forced = {step for step in range(NUM_STEPS)
                  if step < warmup or step == NUM_STEPS - 1}
        return NUM_STEPS - len(forced)
    if method == "teacache":
        warmup = int(params["teacache_ret_steps"])
        forced = {step for step in range(NUM_STEPS)
                  if step < warmup or step == NUM_STEPS - 1}
        return NUM_STEPS - len(forced)
    if method == "dicache":
        warmup_last = int(float(params["dicache_ret_ratio"]) * NUM_STEPS)
        forced = {step for step in range(NUM_STEPS)
                  if step <= warmup_last or step == NUM_STEPS - 1}
        return NUM_STEPS - len(forced)
    if method == "sencache":
        first_enhance = int(params["sencache_first_enhance"])
        # per-cell since the recalibration: the budget that a cell can reach at
        # all moves with the run limit it was frozen at
        max_skip = int(params.get("sencache_max_skip", SENCACHE_LANE_MAX_SKIP))
        cutoff_step = (NUM_STEPS - 1 if SENCACHE_LANE_CUTOFF_STEPS < 0
                       else SENCACHE_LANE_CUTOFF_STEPS)
        cached = consecutive = 0
        for step in range(NUM_STEPS):
            forced_here = (step < first_enhance or step == 0
                           or step >= max(min(cutoff_step, NUM_STEPS), 0)
                           or step < SENCACHE_LANE_RET_STEPS
                           # the gate stops caching once it has skipped this many
                           # in a row, whatever the score says
                           or consecutive >= max_skip)
            if forced_here:
                consecutive = 0
            else:
                cached += 1
                consecutive += 1
        return cached
    raise KeyError(f"{method} is not a dynamic gate with a frozen warmup")


# Steps each fixed-schedule method may not cache. Imported rather than restated:
# `wan21.methods_glue.forbidden_cache_steps` derives each set from that method's
# own warmup constants, and `wan21/methods_glue.py::fixed_cache_steps` rejects a
# violating table at construction time using the same function -- so a table
# this validator accepts is a table the adapter will accept. Section 3.3: the
# table the triplet shares has to satisfy all three at once, which is the
# {0, 1, 2, 49} union of TaylorSeer {0, 49}, HiCache {0, 1, 2, 49} and L2P {0}.
FORBIDDEN_CACHE_STEPS: dict[str, frozenset[int]] = {
    method: forbidden_cache_steps(method, NUM_STEPS) for method in SCHEDULE_METHODS
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
    # steps 0..int(ret_ratio * 50) plus the terminal step are forced full, so at
    # the default 0.2 the budget cannot exceed 38 and K41 is only reachable by
    # moving it (plan section 3.1 freezes 0.1 there).
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
        if budget not in BUDGETS:
            raise KeyError(f"unknown budget: {budget}; the matrix has {list(BUDGETS)}")
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

        Never partial: `_validate_method_params` refuses a file where a method
        freezes some of its knobs and not others, so this is all of them or none.
        """
        per_method = (self.payload.get("method_params") or {}).get(method) or {}
        per_dataset = per_method.get(dataset) or {}
        return {name: float(value) for name, value in (per_dataset.get(budget) or {}).items()}

    def _assets(self, method: str) -> dict[str, str]:
        """`runner argument name -> sha256` of this method's file inputs."""
        return {name: str(digest) for name, digest
                in ((self.payload.get("method_assets") or {}).get(method) or {}).items()}

    def calibration(self, dataset: str) -> dict[str, str]:
        """The prompt file this dataset's thresholds were swept on."""
        row = (self.payload.get("threshold_calibration") or {}).get(dataset) or {}
        return {"file": str(row.get("file", "")), "sha256": str(row.get("sha256", ""))}

    def evaluation_dataset(self, dataset: str) -> dict[str, str]:
        """The frozen prompt manifest this dataset's cells generate from.

        Plan section 1.2: the Wan lane does not copy the two manifests into
        `resources/wan21/`, it references them by digest. This is what the runner
        checks `--prompt_manifest` against.
        """
        row = (self.payload.get("evaluation_datasets") or {}).get(dataset) or {}
        return {"file": str(row.get("file", "")), "sha256": str(row.get("sha256", ""))}

    def provenance(self) -> dict[str, str]:
        """The fields every run record carries so a result traces to its config."""
        return {"matrix_config": str(self.path), "matrix_config_sha256": self.sha256}


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Reject anything the matrix must not be started from."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema: {payload.get('schema')!r}")
    if payload.get("protocol_id") != PROTOCOL_ID:
        # The thresholds were swept under one protocol. Another resolution or
        # frame count means every gate's score is taken over a different token
        # grid -- a far larger distribution shift than the dataset axis this file
        # exists for.
        raise ValueError(
            f"protocol_id must be {PROTOCOL_ID!r}, got {payload.get('protocol_id')!r}")
    if payload.get("num_steps") != NUM_STEPS:
        raise ValueError(f"the Wan2.1 baseline matrix is frozen to {NUM_STEPS} steps")
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
        # schedule is held fixed (plan section 3.3); one table per method would
        # silently destroy that comparison.
        shared = {schedules[method][budget] for method in TRIPLET_METHODS}
        if len(shared) != 1:
            raise ValueError(
                f"{budget}: {list(TRIPLET_METHODS)} must read one shared table, got "
                f"{sorted(shared)}"
            )
    _validate_method_params(payload.get("method_params") or {})
    _validate_method_assets(payload.get("method_assets") or {})
    _validate_file_bindings(payload.get("threshold_calibration") or {},
                            name="threshold_calibration",
                            why="a threshold with no record of what it was swept on "
                                "cannot be checked against anything")
    _validate_file_bindings(payload.get("evaluation_datasets") or {},
                            name="evaluation_datasets",
                            why="plan section 1.2 references the two frozen manifests by "
                                "digest instead of copying them into resources/wan21")

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


def _validate_file_bindings(rows: Mapping[str, Any], *, name: str, why: str) -> None:
    """One `{file, sha256}` row per dataset, with the two digests distinct.

    Used for both file-valued dataset bindings: the calibration set a dataset's
    thresholds were swept on, and the evaluation manifest its cells generate
    from. Requiring the two datasets' digests to differ is what makes the
    dataset axis falsifiable -- one sweep's numbers pasted under both dataset
    keys, or one manifest referenced twice, is otherwise a well-formed file.
    """
    if set(rows) != set(DATASETS):
        raise ValueError(
            f"{name} must name exactly {list(DATASETS)}, got {sorted(rows)}; {why}")
    digests: dict[str, str] = {}
    for dataset, row in rows.items():
        digest = (row or {}).get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or \
                any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name}/{dataset} needs a lowercase sha256 hex digest "
                             f"of its file")
        if not (row or {}).get("file"):
            raise ValueError(f"{name}/{dataset} needs the file path it refers to")
        digests[dataset] = digest
    if len(set(digests.values())) != len(DATASETS):
        raise ValueError(
            f"{name}: every dataset points at the same file "
            f"{sorted(set(digests.values()))}; then only one distribution is recorded and "
            f"the dataset axis records a difference that does not exist")


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
                raise ValueError(
                    f"table {table_id} jvp_span for step {step} must be a positive int")
    elif spans is not None:
        # A MeanCache solution satisfies the triplet's forbidden steps too, so
        # without this a searched MeanCache table could be frozen as the shared
        # table and three methods would silently run MeanCache's schedule with
        # half its solution dropped.
        raise ValueError(f"table {table_id} carries jvp_spans but is read by {method}")


def load_matrix_config(path: Path = CONFIG_PATH) -> BaselineMatrixConfig:
    """Load and fully validate the frozen configuration."""
    path = Path(path)
    payload = load_self_hashed_json(path, HASH_FIELD)
    validate_payload(payload)
    return BaselineMatrixConfig(path=path, sha256=str(payload[HASH_FIELD]), payload=payload)
