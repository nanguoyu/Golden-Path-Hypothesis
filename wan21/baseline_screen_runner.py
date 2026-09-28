#!/usr/bin/env python3
"""Wan2.1 t2v-1.3B runner for the nine-method baseline matrix (plan section 4).

One invocation generates one shard of one matrix cell: a (method, dataset,
budget, seed) row, or the uncached reference that row is compared against. It is
the same program the P3 threshold sweep runs, so the decision files a sweep
reads are written by the code the matrix itself will run -- there is no separate
measurement path.

Twin of `hunyuan_video/baseline_screen_runner.py`; four things are Wan's:

  * **The generation loop is the lane's own.** `wan21/runner.py::generate_t2v`
    is imported unchanged: it builds the `FlowUniPCMultistepScheduler`, attaches
    it to `pipe.model` (which is where SeaCache's SEA filter and MeanCache's
    sigmas read it from), runs the cond/uncond pair per step, and decodes. The
    matrix adds no second copy of the sampler.
  * **`--mode` is the method id.** `wan21/backend.py::build_adapter` dispatches
    on the plan's section 3.3 keys, so there is no mode-to-method translation
    table as on the Hunyuan lane.
  * **Every fixed schedule is explicit.** `wan21/methods_glue.py::fixed_cache_steps`
    has no schedule generator: BudCache, MeanCache and the triplet all take
    their table from the frozen config or from `--cache_steps` /
    `--meancache_schedule`. A budget with no table is a failure, not a derived
    schedule.
  * **Block-call accounting is checked per video.** Plan section 2.0 fixes 60
    original block calls on a fully-computed step (30 blocks x 2 CFG forwards)
    and 0 on a cached one, DiCache excepted -- its shallow probe really runs, on
    both branches, so a cached step costs `2 * probe_depth`. Both numbers are
    already in the decision rows, so checking them is arithmetic, not
    instrumentation, and it fails on video 1 of a cell rather than at audit time
    after the cell is spent.

Two transports for prompts, and exactly one may be given. `--prompt_manifest` is
the only one that can carry the two Penguin prompts containing a literal newline
and the only one that carries a prompt_id and a dataset name; `--prompt_file` is
for the 48-prompt threshold-calibration sets, which have neither problem.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hunyuan_video.records import load_self_hashed_json, sha256_file
from hunyuan_video.trajectory_retention import (
    LatentPath,
    retain_trajectory,
    retention_plan,
)
from lib.io_utils import read_prompts, split_shard
from wan21._helpers import decisions_filename, git_sha, seed_for, video_filename
from wan21.backend import (
    WAN_UPSTREAM_ROOT,
    WanProtocol,
    WanRunSpec,
    import_wan,
    load_wan_pipeline,
    maybe_adapter,
    validate_protocol,
)
from wan21.matrix_config import (
    BUDGETS,
    DATASETS,
    DYNAMIC_METHODS,
    EVALUATION_MANIFEST_SCHEMA,
    MEANCACHE_LANE_JVP_SPAN,
    METHOD_PARAMS,
    NUM_STEPS,
    PROTOCOL_ID,
    RUNNER_MODES,
    SCHEDULE_METHODS,
    SEACACHE_LANE_NORM_MODE,
    SENCACHE_LANE_CUTOFF_STEPS,
    SENCACHE_LANE_MAX_SKIP,
    SENCACHE_LANE_RET_STEPS,
    SENCACHE_LANE_SWITCH_RATIO,
    TEACACHE_LANE_VARIANT,
    BaselineMatrixConfig,
    load_matrix_config,
)
from wan21.methods_glue import (
    CondDecidesArbiter,
    RELAXED_FIRST_FULL_STEPS,
    WAN_NUM_LAYERS,
    WAN_ORIGINAL_BLOCK_CALLS_PER_STEP,
    forbidden_cache_steps,
)
from wan21.runner import generate_t2v, save_video_tensor
# Private only by name: `save_video_tensor` calls it anyway, and calling it
# up front turns a missing imageio-ffmpeg into a startup failure instead of one
# that lands after the first ~40-minute generation.
from wan21.runner import _require_video_writer_backend as require_video_writer_backend


DECISION_SCHEMA = "wan21.baseline_screen_decisions.v1"
TIMING_SCHEMA = "wan21.baseline_screen_timing.v1"
CELL_SCHEMA = "wan21.baseline_screen_cell.v1"

#: Every knob any method may freeze, read off the table rather than repeated, so
#: a knob added there cannot be left reachable from the command line here.
FREEZABLE_ARGS = tuple(sorted({name for spec in METHOD_PARAMS.values() for name in spec}))


def _step_list(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item.strip())
    if steps != tuple(sorted(set(steps))):
        raise argparse.ArgumentTypeError("cache steps must be sorted and unique")
    return steps


def _idx_list(value: str) -> tuple[int, ...]:
    indices = tuple(int(item) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("--prompt_indices needs at least one index")
    if indices != tuple(sorted(set(indices))):
        raise argparse.ArgumentTypeError("prompt indices must be sorted and unique")
    if indices[0] < 0:
        raise argparse.ArgumentTypeError("prompt indices must be non-negative")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=RUNNER_MODES, required=True,
                        help="one of the nine matrix methods, or 'original' for the "
                             "uncached reference")
    # exactly one of these
    parser.add_argument("--prompt_file", type=Path,
                        help="one prompt per line; the threshold-calibration transport")
    parser.add_argument("--prompt_manifest", type=Path,
                        help="frozen evaluation manifest (resources/hunyuan_video/"
                             "evaluation/{penguin599,vbench944}.json, reused verbatim -- "
                             "plan section 1.2)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ckpt_dir", type=Path, default=os.environ.get("WAN21_CKPT_DIR"),
                        help="Wan2.1-T2V-1.3B checkpoint directory (or WAN21_CKPT_DIR)")
    parser.add_argument("--wan_repo", type=Path, default=WAN_UPSTREAM_ROOT,
                        help="directory holding the pinned importable `wan` package")
    parser.add_argument("--protocol_id", default=PROTOCOL_ID)

    parser.add_argument("--cache_count", type=int)
    parser.add_argument("--cache_steps", type=_step_list, default=())
    parser.add_argument("--threshold", type=float, default=0.2)

    # One dest per method: the Hunyuan lane shares `--first_enhance` across four
    # methods and repairs the collision with a max(3, ...) clamp, which makes a
    # frozen 1 and a running 3 both defensible readings of one file.
    parser.add_argument("--seacache_first_enhance", type=int, default=1)
    parser.add_argument("--seacache_power_exp", type=float, default=3.0)
    parser.add_argument("--teacache_ret_steps", type=int, default=5)
    parser.add_argument("--sencache_sensitivity_path", type=Path)
    parser.add_argument("--sencache_threshold_start", type=float, default=0.8)
    parser.add_argument("--sencache_first_enhance", type=int, default=3)
    parser.add_argument("--sencache_max_skip", type=int,
                        default=SENCACHE_LANE_MAX_SKIP)
    parser.add_argument("--sencache_switch_ratio", type=float,
                        default=SENCACHE_LANE_SWITCH_RATIO)
    parser.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    parser.add_argument("--dicache_probe_depth", type=int, default=1)
    parser.add_argument("--taylorseer_first_enhance", type=int, default=3)
    parser.add_argument("--hicache_sigma", type=float, default=0.5)
    parser.add_argument("--hicache_first_enhance", type=int, default=3)
    parser.add_argument("--l2p_weights", type=Path)
    parser.add_argument("--l2p_min_abs_weight", type=float, default=0.0)
    parser.add_argument("--meancache_schedule", type=Path,
                        help="searched MeanCache table (cache_steps + per-edge jvp_spans); "
                             "the frozen config supplies both instead")
    parser.add_argument("--meancache_jvp_span", type=int, default=MEANCACHE_LANE_JVP_SPAN,
                        help="global fallback JVP span for cached steps the schedule file "
                             "gives no per-edge span for. The offline search only solved "
                             "per-edge spans for MeanCache's own tables, so every "
                             "transplanted schedule (video SPX) runs on this value "
                             "(hunyuan_video/methods/meancache.py span_for)")
    parser.add_argument("--spx_relax_warmup", action="store_true",
                        help="video SPX only: lower the head-of-trajectory warmup of "
                             "meancache (5 -> 2) and budcache (3 -> 1) to what the payload "
                             "structurally needs, so a transplanted schedule that caches "
                             "step 2 or 3 still has a cell. See "
                             "wan21.methods_glue.RELAXED_FIRST_FULL_STEPS. Refused "
                             "together with --matrix_config: the frozen 162 cells run the "
                             "unrelaxed rule")

    parser.add_argument("--seed", type=int, default=42,
                        help="base seed; every prompt uses base + prompt_idx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt_indices", type=_idx_list, default=None,
                        help="generate exactly these manifest indices (sorted unique "
                             "comma list) instead of the contiguous --limit slice; an "
                             "idx keeps its manifest position, so its seed and its "
                             "prompt_id are unchanged. Not combinable with --limit")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")

    parser.add_argument("--matrix_config", type=Path,
                        help="frozen baseline-matrix config; with --budget and --dataset "
                             "it supplies this method's threshold or schedule and every "
                             "knob frozen with it, instead of the command line")
    parser.add_argument("--budget", choices=BUDGETS,
                        help="which tier to look up in --matrix_config")
    parser.add_argument("--dataset", choices=DATASETS,
                        help="which evaluation set this row belongs to; its thresholds "
                             "were calibrated on that set's own calibration subset")

    parser.add_argument("--retain_trajectory", action="store_true",
                        help="write the plan section 4.2 tiers next to each video")
    parser.add_argument("--t3_seed", type=int,
                        help="the one reference seed whose first --t3_prompt_count "
                             "prompts keep their whole latent path")
    parser.add_argument("--t3_prompt_count", type=int, default=0)
    parser.add_argument("--t3_cached", action="store_true",
                        help="apply the --t3_seed/--t3_prompt_count T3 predicate to a "
                             "cached (non-original) run as well; T2 stays reference-only")
    return parser


# The defaults, read off the parser instead of copied: a duplicated literal that
# drifts makes a config-driven cell refuse a flag nobody passed.
_PARSER_DEFAULTS = {action.dest: action.default for action in build_parser()._actions}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _differs_from_default(args: argparse.Namespace, name: str) -> bool:
    """Whether `name` was actually given, judged against the parser's own
    default rather than a copy of it."""
    value = getattr(args, name, None)
    default = _PARSER_DEFAULTS.get(name)
    if value in (None, ()):
        return False
    if isinstance(value, float) and isinstance(default, (int, float)):
        return abs(float(value) - float(default)) > 1e-12
    return value != default


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------


def _apply_matrix_config(args: argparse.Namespace) -> BaselineMatrixConfig | None:
    """Fill this cell's frozen hyper-parameters in, or return None when the run
    is not driven by a config.

    The plan makes the matrix read one frozen file and forbids falling back to a
    default, so a missing cell stops the run rather than being filled in.
    Everything downstream -- validation, `_method_config`, the adapters -- then
    sees exactly what a command-line run would produce, which is what makes a
    config-driven smoke test evidence about the matrix.
    """
    given = [name for name in ("matrix_config", "budget", "dataset")
             if getattr(args, name) is not None]
    if given and len(given) != 3:
        raise SystemExit("--matrix_config, --budget and --dataset are used together or "
                         "not at all; a threshold is frozen per (method, dataset, budget)")
    if args.matrix_config is None:
        return None
    if args.mode == "original":
        raise SystemExit("original rows are the uncached reference and have no frozen "
                         "entry; run them without --matrix_config")

    supplied = [name for name in ("threshold", "cache_count", "cache_steps",
                                  "meancache_schedule")
                if _differs_from_default(args, name)]
    if args.mode not in DYNAMIC_METHODS:
        supplied = [name for name in supplied if name != "threshold"]
    if supplied:
        raise SystemExit(
            f"{', '.join(sorted(supplied))} came from the command line, but "
            f"--matrix_config supplies them; pass one or the other")

    ########################
    # Knobs beyond the threshold that move the realized cache count come from the
    # frozen entry too, not the command line -- otherwise a cell runs on a number
    # the frozen record does not contain. Which ones each method may freeze is
    # `matrix_config.METHOD_PARAMS`, and FREEZABLE_ARGS is read off it.
    ########################
    unrecordable = [name for name in (*FREEZABLE_ARGS, "meancache_jvp_span",
                                      "spx_relax_warmup")
                    if _differs_from_default(args, name)]
    if unrecordable:
        raise SystemExit(
            f"{', '.join(sorted(unrecordable))} came from the command line; a "
            f"config-driven cell takes them from the frozen entry's method_params, "
            f"so a value here would not survive into the matrix")

    config = load_matrix_config(args.matrix_config)
    frozen_protocol = config.payload.get("protocol_id")
    if args.protocol_id != frozen_protocol:
        raise SystemExit(
            f"--protocol_id {args.protocol_id} is not the {frozen_protocol} this config "
            f"was frozen for; its thresholds were swept at that protocol's resolution "
            f"and frame count")
    # KeyError if the cell is not frozen -- there is no default to fall back to
    entry = config.entry(args.mode, args.budget, args.dataset)
    for name, value in entry.method_params.items():
        # ints stay ints: probe depths and warmups are counts
        current = getattr(args, name)
        setattr(args, name, type(current)(value) if isinstance(current, int) else float(value))
    for name, digest in entry.assets.items():
        path = getattr(args, name)
        if path is None or not Path(path).is_file():
            raise SystemExit(f"{args.mode} reads {name}, which the frozen config pins by "
                             f"digest; pass the file (got {path!r})")
        actual = sha256_file(Path(path))
        if actual != digest:
            raise SystemExit(
                f"{name} {path} hashes to {actual}, but the frozen config was built "
                f"against {digest}; this cell was calibrated on that file")
    if entry.threshold is not None:
        args.threshold = float(entry.threshold)
    else:
        args.cache_count = int(entry.cache_count)
        args.cache_steps = tuple(int(step) for step in entry.cache_steps)
        if args.mode == "meancache":
            if entry.jvp_spans is None:
                raise SystemExit(f"frozen meancache table {entry.table_id} carries no "
                                 f"jvp_spans; the spans are part of the solved path")
            args.meancache_frozen_spans = {int(k): int(v) for k, v in entry.jvp_spans.items()}
    return config


def _resolve_meancache_schedule(args: argparse.Namespace) -> None:
    """Take MeanCache's table and per-edge spans from `--meancache_schedule`.

    Only for runs the frozen config does not drive (P2 walk-throughs, P4 search
    checks). The two halves arrive together or not at all: a table without the
    spans the path search chose is a different schedule wearing its name.
    """
    if args.mode != "meancache" or getattr(args, "meancache_frozen_spans", None) is not None:
        return
    if args.meancache_schedule is None or not Path(args.meancache_schedule).is_file():
        raise SystemExit("meancache requires an existing --meancache_schedule "
                         "(or --matrix_config with --budget and --dataset)")
    if args.cache_steps:
        raise SystemExit("meancache reads its schedule from --meancache_schedule")
    payload = json.loads(Path(args.meancache_schedule).read_text(encoding="utf-8"))
    steps = tuple(int(step) for step in payload["cache_steps"])
    spans = {int(step): int(span) for step, span in (payload.get("jvp_spans") or {}).items()}
    # A search-solved table covers every edge, and a MISSING span there would
    # silently substitute the runtime default for the span the search chose --
    # a different schedule. A transplanted schedule (video SPX) has no per-edge
    # solution at all, so the rule is "all of them or none of them", and the
    # none case runs `--meancache_jvp_span` on every edge through `span_for`.
    stray = sorted(step for step in spans if step not in set(steps))
    if stray:
        raise SystemExit(f"--meancache_schedule jvp_spans name uncached steps: {stray}")
    if spans and len(spans) != len(steps):
        raise SystemExit(
            "--meancache_schedule jvp_spans cover only part of the cached steps; the "
            "search solves every edge of a table or none of them. Drop jvp_spans "
            "entirely to run the global --meancache_jvp_span on all of them.")
    if args.cache_count is not None and int(args.cache_count) != len(steps):
        raise SystemExit(
            f"--cache_count {args.cache_count} differs from the {len(steps)} steps in "
            f"{args.meancache_schedule}")
    args.cache_steps = steps
    args.cache_count = len(steps)
    args.meancache_frozen_spans = spans


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSource:
    prompts: list[str]
    ids: list[str]
    path: Path
    sha256: str
    manifest_sha256: str | None = None
    dataset: str | None = None


def _load_prompts(args: argparse.Namespace,
                  config: BaselineMatrixConfig | None) -> PromptSource:
    """Read the prompts from whichever transport was given.

    A line-per-prompt file cannot carry the two Penguin prompts that contain a
    newline and carries no prompt_id at all; the manifest does both and also
    states which dataset it is, which is the only thing that catches a row
    generating one dataset's prompts under the other's frozen thresholds. When a
    frozen config is driving the run, the manifest's digest is checked against
    the one the config pins (plan section 1.2: the Wan lane references the two
    manifests rather than copying them).
    """
    if (args.prompt_file is None) == (args.prompt_manifest is None):
        raise SystemExit("pass exactly one of --prompt_file / --prompt_manifest")
    if args.prompt_file is not None:
        prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
        # a line index, since a text file carries no identity of its own
        ids = [f"prompt-{index}" for index in range(len(prompts))]
        return PromptSource(prompts=prompts, ids=ids, path=args.prompt_file,
                            sha256=sha256_file(args.prompt_file))

    manifest = Path(args.prompt_manifest)
    payload = load_self_hashed_json(manifest, "manifest_sha256")
    if payload.get("schema") != EVALUATION_MANIFEST_SCHEMA:
        raise SystemExit(f"{manifest} is not an evaluation prompt manifest "
                         f"({EVALUATION_MANIFEST_SCHEMA})")
    if args.dataset is not None and payload.get("dataset") != args.dataset:
        raise SystemExit(
            f"--dataset {args.dataset} but the manifest holds "
            f"{payload.get('dataset')!r}; that row would generate one dataset's prompts "
            f"under the other's frozen thresholds")
    digest = sha256_file(manifest)
    if config is not None:
        frozen = config.evaluation_dataset(args.dataset)
        if digest != frozen["sha256"]:
            raise SystemExit(
                f"{manifest} hashes to {digest}, but the frozen config pins "
                f"{frozen['sha256']} for {args.dataset} ({frozen['file']}); this row "
                f"would run a different prompt list under this cell's frozen numbers")
    items = list(payload["items"])
    if args.limit:
        items = items[: int(args.limit)]
    return PromptSource(
        prompts=[str(item["prompt"]) for item in items],
        ids=[str(item["prompt_id"]) for item in items],
        path=manifest,
        sha256=digest,
        manifest_sha256=str(payload["manifest_sha256"]),
        dataset=str(payload.get("dataset")),
    )


# ---------------------------------------------------------------------------
# Validation and the method config the adapters read
# ---------------------------------------------------------------------------


def dicache_fixed_schedule(args: argparse.Namespace) -> bool:
    """Whether this run is the video SPX `di_two_anchor` payload on a fixed table.

    DiCache is normally a fused gate + payload with no schedule entry at all
    (video SPX plan section 3). Given `--cache_steps` it keeps the payload --
    shallow probe, gamma clamp, two-anchor extrapolation -- and takes the action
    from the table, which is what makes the fifth payload column comparable with
    the other four on a foreign schedule. Twin of the Hunyuan runner's helper.
    """
    return args.mode == "dicache" and bool(args.cache_steps)


def _validate(args: argparse.Namespace) -> None:
    if args.protocol_id != PROTOCOL_ID:
        raise SystemExit(f"the Wan2.1 matrix lane is frozen to protocol {PROTOCOL_ID}")
    if args.ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required, or WAN21_CKPT_DIR must be set")
    if not Path(args.ckpt_dir).is_dir():
        raise SystemExit(f"missing Wan2.1 checkpoint directory: {args.ckpt_dir}")
    if args.t3_prompt_count and not args.retain_trajectory:
        raise SystemExit("--t3_prompt_count needs --retain_trajectory")

    if args.mode == "original":
        if args.cache_count is not None or args.cache_steps:
            raise SystemExit("original mode does not accept cache_count/cache_steps")
        return

    if int(args.meancache_jvp_span) < 1:
        raise SystemExit("--meancache_jvp_span must be at least 1 step")

    if args.mode in DYNAMIC_METHODS and not dicache_fixed_schedule(args):
        if args.cache_count is not None or args.cache_steps:
            raise SystemExit(
                "native dynamic gates do not accept cache_count/cache_steps; calibrate "
                "their threshold against the dataset-level mean K")
        if float(args.threshold) <= 0.0:
            raise SystemExit("native dynamic gates require --threshold > 0")
        if args.mode == "seacache" and not 1 <= int(args.seacache_first_enhance) < NUM_STEPS:
            raise SystemExit("seacache requires 1 <= --seacache_first_enhance < 50")
        if args.mode == "teacache" and not 0 <= int(args.teacache_ret_steps) < NUM_STEPS:
            raise SystemExit("teacache requires 0 <= --teacache_ret_steps < 50")
        if args.mode == "sencache":
            # The gate loads the frozen table in its constructor, so a missing
            # file would only surface after the model is already resident.
            if (args.sencache_sensitivity_path is None
                    or not Path(args.sencache_sensitivity_path).is_file()):
                raise SystemExit("sencache requires an existing --sencache_sensitivity_path")
            if float(args.sencache_threshold_start) <= 0.0:
                raise SystemExit("sencache requires --sencache_threshold_start > 0")
            if not 1 <= int(args.sencache_max_skip) < NUM_STEPS:
                raise SystemExit(
                    f"sencache requires 1 <= --sencache_max_skip < {NUM_STEPS}")
            if not 0.0 <= float(args.sencache_switch_ratio) <= 1.0:
                raise SystemExit(
                    "sencache requires 0 <= --sencache_switch_ratio <= 1")
            if not 3 <= int(args.sencache_first_enhance) < NUM_STEPS:
                raise SystemExit(
                    f"sencache requires 3 <= --sencache_first_enhance < {NUM_STEPS}; "
                    f"upstream Wan has no warmup at all, and this lane's paradigm "
                    f"requirement is 3 (plan section 2.2)")
        if args.mode == "dicache":
            if not 0.0 <= float(args.dicache_ret_ratio) < 1.0:
                raise SystemExit("dicache requires 0 <= --dicache_ret_ratio < 1")
            if not 1 <= int(args.dicache_probe_depth) <= WAN_NUM_LAYERS:
                raise SystemExit(
                    f"dicache requires 1 <= --dicache_probe_depth <= {WAN_NUM_LAYERS}")
        return

    # Fixed schedules. Wan has no schedule generator, so a table is mandatory --
    # searched (BudCache/MeanCache), shared (the triplet), or empty for the
    # forced-all-full conformance check of plan section 2.0 item 1.
    if args.cache_count is None or not 0 <= int(args.cache_count) <= NUM_STEPS - 2:
        raise SystemExit(f"fixed schedule modes require --cache_count in [0, {NUM_STEPS - 2}]")
    if len(args.cache_steps) != int(args.cache_count):
        raise SystemExit(
            f"--cache_steps holds {len(args.cache_steps)} steps, --cache_count says "
            f"{args.cache_count}; on this lane every fixed schedule is explicit")
    outside = [step for step in args.cache_steps if step < 0 or step >= NUM_STEPS]
    if outside:
        raise SystemExit(f"--cache_steps outside the {NUM_STEPS}-step trajectory: {outside}")
    banned = sorted(forbidden_cache_steps(
        args.mode, NUM_STEPS,
        relax_warmup=bool(args.spx_relax_warmup)).intersection(args.cache_steps))
    if banned:
        raise SystemExit(f"{args.mode} may not cache steps {banned}")
    if args.spx_relax_warmup:
        print(f"[baseline-screen] --spx_relax_warmup: {args.mode} warmup lowered to "
              f"{RELAXED_FIRST_FULL_STEPS.get(args.mode, 'unchanged')}", flush=True)
    if args.mode == "hicache_o2" and not 0.0 < float(args.hicache_sigma) <= 1.0:
        raise SystemExit("hicache_o2 requires 0 < --hicache_sigma <= 1")
    if args.mode == "l2p":
        # Hard requirement even for the empty-schedule conformance run: no
        # weights file, no L2P (plan section 2.8).
        if args.l2p_weights is None or not Path(args.l2p_weights).is_file():
            raise SystemExit("l2p requires an existing --l2p_weights file")
    if args.mode == "meancache" and getattr(args, "meancache_frozen_spans", None) is None:
        raise SystemExit("meancache requires its per-edge jvp_spans")


def _method_config(args: argparse.Namespace) -> dict[str, Any]:
    """The dict `wan21.backend.build_adapter` builds this cell's method from.

    Lane constants that are not free parameters (SenCache's max_skip / cutoff /
    ret_steps / switch_ratio, SeaCache's norm mode, TeaCache's coefficient row)
    are written in explicitly rather than left to a default, so the decision
    record states every number the method ran with.
    """
    if args.mode == "original":
        return {}
    if args.mode in DYNAMIC_METHODS:
        config: dict[str, Any] = {"threshold": float(args.threshold)}
        if args.mode == "seacache":
            config.update({
                "first_enhance": int(args.seacache_first_enhance),
                "power_exp": float(args.seacache_power_exp),
                "norm_mode": SEACACHE_LANE_NORM_MODE,
            })
        elif args.mode == "teacache":
            config.update({
                "ret_steps": int(args.teacache_ret_steps),
                "variant": TEACACHE_LANE_VARIANT,
            })
        elif args.mode == "sencache":
            config.update({
                "sensitivity_path": str(args.sencache_sensitivity_path),
                "threshold_start": float(args.sencache_threshold_start),
                "first_enhance": int(args.sencache_first_enhance),
                "max_skip": int(args.sencache_max_skip),
                "switch_ratio": float(args.sencache_switch_ratio),
                "ret_steps": SENCACHE_LANE_RET_STEPS,
                "cutoff_steps": SENCACHE_LANE_CUTOFF_STEPS,
            })
        elif args.mode == "dicache":
            config.update({
                "ret_ratio": float(args.dicache_ret_ratio),
                "probe_depth": int(args.dicache_probe_depth),
            })
            if dicache_fixed_schedule(args):
                # The gate is replaced by the table; the payload is untouched.
                config.update({
                    "cache_count": int(args.cache_count),
                    "cache_steps": [int(step) for step in args.cache_steps],
                })
        if args.spx_relax_warmup:
            config["relax_warmup"] = True
        return config

    config = {
        "cache_count": int(args.cache_count),
        "cache_steps": [int(step) for step in args.cache_steps],
    }
    if args.mode == "taylorseer_o1":
        config["first_enhance"] = int(args.taylorseer_first_enhance)
    elif args.mode == "hicache_o2":
        config.update({
            "sigma": float(args.hicache_sigma),
            "first_enhance": int(args.hicache_first_enhance),
        })
    elif args.mode == "l2p":
        config.update({
            "weights_path": str(args.l2p_weights),
            "min_abs_weight": float(args.l2p_min_abs_weight),
        })
    elif args.mode == "meancache":
        spans = getattr(args, "meancache_frozen_spans", None)
        if spans is None:
            raise SystemExit("meancache has no per-edge jvp_spans; they are part of the "
                             "solved path and arrive with the table")
        config.update({
            # string keys so the record stays plain JSON; methods_glue casts back
            "jvp_spans": {str(step): int(span) for step, span in sorted(spans.items())},
            "jvp_span": int(args.meancache_jvp_span),
        })
    if args.spx_relax_warmup:
        # `methods_glue.fixed_cache_steps` reads it when it rebuilds the
        # forbidden set at adapter-build time. Only written when it is asked
        # for, so a matrix cell's recorded method_config is byte-identical to
        # what it was before this flag.
        config["relax_warmup"] = True
    return config


# ---------------------------------------------------------------------------
# Decision rows
# ---------------------------------------------------------------------------


def _records(adapter: Any | None, *, num_steps: int) -> list[dict[str, Any]]:
    """One row per solver step -- not per model forward.

    The reference lane installs nothing, so its rows are synthesized: 30 blocks
    on each of the two CFG forwards, no gate, hence no branch arbitration to
    report.
    """
    if adapter is None:
        return [
            {
                "step": step,
                "action": "full",
                "reason": "reference_full",
                "original_block_calls": WAN_ORIGINAL_BLOCK_CALLS_PER_STEP,
                "cond_block_calls": WAN_NUM_LAYERS,
                "uncond_block_calls": WAN_NUM_LAYERS,
                "branch_policy": "no_gate",
                "timestep": None,
            }
            for step in range(int(num_steps))
        ]
    return [row.as_dict() for row in adapter.decisions]


def expected_block_calls(mode: str, *, cached: bool, probe_depth: int) -> int:
    """Plan section 2.0 items 2 and 5, as arithmetic.

    A fully-computed step runs all 30 blocks on both CFG forwards: 60. A cached
    step runs none of them -- except DiCache's, whose shallow probe really
    executes `probe_depth` blocks on each branch, which is the one recorded
    exception and is why its latency is counted separately in section 4.4.
    """
    if not cached:
        return WAN_ORIGINAL_BLOCK_CALLS_PER_STEP
    return 2 * int(probe_depth) if mode == "dicache" else 0


def _verify_block_calls(records: list[dict[str, Any]], *, mode: str, probe_depth: int) -> None:
    for row in records:
        cached = row["action"] == "cache"
        expected = expected_block_calls(mode, cached=cached, probe_depth=probe_depth)
        actual = int(row["original_block_calls"])
        if actual != expected:
            raise RuntimeError(
                f"step {row['step']} ({row['action']}, {row['reason']}) made {actual} "
                f"original block calls, expected {expected}")


# ---------------------------------------------------------------------------
# Cell identity, timing
# ---------------------------------------------------------------------------


def _cell_identity(args: argparse.Namespace, *, matrix_sha: str | None,
                   prompt_sha: str) -> dict[str, Any]:
    """What makes this output directory one cell of the matrix.

    Filenames carry only the prompt index, so two cells writing into one
    directory is invisible: --resume then reports the first cell's videos as the
    second cell's work. This is the `sweep_identity.txt` guard the threshold
    sweep wrapper writes, applied inside the runner so it also covers the matrix.
    """
    return {
        "schema": CELL_SCHEMA,
        "mode": args.mode,
        "protocol_id": args.protocol_id,
        "seed": int(args.seed),
        "budget": args.budget,
        "dataset": args.dataset,
        "matrix_config_sha256": matrix_sha,
        "prompt_source_sha256": prompt_sha,
        "shard_count": int(args.shard_count),
    }


def _check_cell_identity(output_dir: Path, identity: dict[str, Any]) -> None:
    path = output_dir / "cell_identity.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        differing = sorted(key for key in identity if previous.get(key) != identity[key])
        if differing:
            raise SystemExit(
                f"{output_dir} was produced by a different cell; {differing} differ.\n"
                f"  on disk: { {k: previous.get(k) for k in differing} }\n"
                f"  now:     { {k: identity[k] for k in differing} }\n"
                f"Use a separate --output_dir; resuming here would report the other "
                f"cell's videos as this one's.")
        return
    path.write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")


def timing_path_for(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json")


def keep_existing_timing(args: argparse.Namespace, generated: int) -> bool:
    """Whether this invocation must leave the shard's timing file alone.

    A resumed shard with nothing left to generate has timed nothing, and the
    write at the end of `main` is unconditional: it would replace the measured
    means with zeros, which is exactly what plan section 4.4 (and the section 5
    T_ref backfill) reads.
    """
    return generated == 0 and bool(args.resume) and timing_path_for(args).is_file()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Trajectory capture (plan section 4.2)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_latent_path(scheduler_cls: Any, scheduler_source: Any) -> Iterator[LatentPath]:
    """Record the latent after every denoise step of the generations run inside.

    Wan twin of `hunyuan_video/trajectory_retention.py::capture_latent_path`,
    which is bound to a `HunyuanVideoSampler` object and so cannot be called
    here; the tiering itself (`retention_plan`, `retain_trajectory`) is backbone
    independent and is imported unchanged. `LatentPath._observe` / `._seal` are
    that module's capture contract, and `hunyuan_video/` is read-only during
    this port.

    The wrapper goes on the scheduler *class*: `generate_t2v` builds a fresh
    `FlowUniPCMultistepScheduler` per generation (`wan21/runner.py:380-385`), so
    an instance patch installed beforehand is thrown away before the first solver
    step. `functools.wraps` keeps the signature intact for anything that inspects
    it, and the transformer is left alone -- which is what lets
    `verify_untouched_transformer` still pass on the reference rows T2 and T3 are
    collected from.
    """
    original = scheduler_cls.step
    if getattr(original, "_latent_path_capture", False):
        raise RuntimeError("scheduler.step is already wrapped; nested capture would "
                           "double-count")
    path = LatentPath()

    @functools.wraps(original)
    def step(scheduler: Any, model_output: Any, timestep: Any, sample: Any,
             *args: Any, **kwargs: Any) -> Any:
        result = original(scheduler, model_output, timestep, sample, *args, **kwargs)
        prev_sample = result[0] if isinstance(result, (tuple, list)) else result.prev_sample
        path._observe(sample, prev_sample)
        return result

    step._latent_path_capture = True
    scheduler_cls.step = step
    try:
        yield path
    finally:
        scheduler_cls.step = original
        scheduler = scheduler_source()
        if scheduler is not None:
            # `set_timesteps` appends the final sigma
            # (`.../wan/utils/fm_solvers_unipc.py:205-209`), so this is exactly
            # num_steps + 1 long -- `retain_trajectory` refuses anything else.
            path._seal(scheduler)


# ---------------------------------------------------------------------------


def _require_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Wan2.1 generation must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1 generation requires CUDA")


def select_work(args: argparse.Namespace,
                prompts: list[str]) -> list[tuple[int, str]]:
    """The `(idx, prompt)` pairs this shard generates, in generation order.

    `idx` is always the prompt's position in the manifest, which is what fixes
    its seed (`seed_for`), its `prompt_id` and its filenames. `--prompt_indices`
    therefore only chooses WHICH positions run, never renumbers them: a scattered
    sample of a stream pairs with the same references as the contiguous run does.
    """
    if args.prompt_indices:
        if args.limit:
            raise SystemExit("--prompt_indices names the positions to generate, so it "
                             "does not take a --limit slice as well")
        past_end = [i for i in args.prompt_indices if i >= len(prompts)]
        if past_end:
            raise SystemExit(f"--prompt_indices {past_end} are past the end of the "
                             f"{len(prompts)}-prompt list")
        start, end = split_shard(len(args.prompt_indices), args.shard_count,
                                 args.shard_idx)
        return [(idx, prompts[idx]) for idx in args.prompt_indices[start:end]]
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    return [(start + offset, prompt)
            for offset, prompt in enumerate(prompts[start:end])]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_compute_node()

    protocol = WanProtocol()
    validate_protocol(protocol)
    config = _apply_matrix_config(args)
    matrix_sha = config.sha256 if config is not None else None
    _resolve_meancache_schedule(args)
    _validate(args)

    source = _load_prompts(args, config)
    work = select_work(args, source.prompts)
    if not work:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _check_cell_identity(args.output_dir,
                         _cell_identity(args, matrix_sha=matrix_sha,
                                        prompt_sha=source.sha256))

    wan, wan_configs, size_configs, attention_backend = import_wan(args.wan_repo)
    video_writer = require_video_writer_backend()
    pipe, load_info = load_wan_pipeline(
        wan,
        wan_configs,
        ckpt_dir=Path(args.ckpt_dir),
        protocol=protocol,
        device_id=0,
        t5_cpu=bool(args.t5_cpu),
    )
    size = size_configs[protocol.size]
    fps = int(wan_configs[protocol.task].sample_fps)

    scheduler_cls = None
    if args.retain_trajectory:
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

        scheduler_cls = FlowUniPCMultistepScheduler

    method_config = _method_config(args)
    run_spec = WanRunSpec(method=args.mode, method_config=method_config)
    per_video: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    for idx, prompt in work:
        video_path = args.output_dir / video_filename(idx)
        decision_path = args.output_dir / decisions_filename(idx)
        if args.resume and video_path.is_file() and decision_path.is_file():
            continue
        seed = seed_for(args.seed, idx)
        plan = retention_plan(
            is_reference=(args.mode == "original"),
            prompt_idx=idx,
            num_steps=protocol.steps,
            base_seed=args.seed,
            t3_seed=args.t3_seed,
            t3_prompt_count=args.t3_prompt_count,
            allow_cached=args.t3_cached,
        ) if args.retain_trajectory else None

        torch.cuda.reset_peak_memory_stats()
        with contextlib.ExitStack() as stack:
            latent_path = (
                stack.enter_context(
                    capture_latent_path(scheduler_cls,
                                        lambda: getattr(pipe.model, "scheduler", None))
                )
                if plan is not None else None
            )
            adapter = stack.enter_context(maybe_adapter(pipe.model, protocol, run_spec))
            video, timing = generate_t2v(
                pipe,
                prompt=prompt,
                size=size,
                frame_num=protocol.frames,
                shift=protocol.sample_shift,
                sample_solver=protocol.sample_solver,
                sampling_steps=protocol.steps,
                guide_scale=protocol.guidance_scale,
                seed=seed,
                offload_model=bool(args.offload_model),
            )
            records = _records(adapter, num_steps=protocol.steps)
            slot_count = getattr(adapter, "slot_count", None)
            latent_numel = getattr(adapter, "latent_numel", None)
        peak_vram = int(torch.cuda.max_memory_allocated())

        save_started = time.perf_counter()
        save_video_tensor(video, video_path, fps=fps)
        save_s = time.perf_counter() - save_started

        if len(records) != protocol.steps:
            raise RuntimeError(
                f"decision count differs from executed steps: {len(records)} != "
                f"{protocol.steps}")
        _verify_block_calls(records, mode=args.mode,
                            probe_depth=int(args.dicache_probe_depth))
        actual_cache_count = sum(row["action"] == "cache" for row in records)
        fixed_schedule = (args.mode in SCHEDULE_METHODS
                          or dicache_fixed_schedule(args))
        if fixed_schedule and actual_cache_count != int(args.cache_count):
            raise RuntimeError(
                f"fixed schedule cache count mismatch: {actual_cache_count} != "
                f"{args.cache_count}")

        if plan is not None and latent_path is not None:
            retain_trajectory(
                latent_path,
                output_dir=args.output_dir,
                stem=f"{idx:05d}",
                num_steps=protocol.steps,
                plan=plan,
                record={
                    # The record's `schema` names its writer, which stays the
                    # read-only Hunyuan module; `model` names the backbone, and
                    # that is what the cross-backbone readers key on
                    # (`analysis/merge_full_traj.py`).
                    "model": "wan21",
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "budget": args.budget,
                    "seed": seed,
                    "prompt_idx": idx,
                    "prompt_id": source.ids[idx],
                    "matrix_config_sha256": matrix_sha,
                },
            )

        decision_payload = {
            "schema": DECISION_SCHEMA,
            "mode": args.mode,
            "protocol_id": args.protocol_id,
            "prompt_idx": idx,
            "prompt_id": source.ids[idx],
            "prompt": prompt,
            "seed": seed,
            "num_steps": protocol.steps,
            "branch_policy": (CondDecidesArbiter.POLICY if args.mode != "original"
                              else "no_gate"),
            "fixed_schedule": fixed_schedule,
            "expected_cache_count": int(args.cache_count) if fixed_schedule else None,
            "actual_cache_count": actual_cache_count,
            # Plan section 2.0 items 3 and 5, and the P0 completion criteria:
            # 90 fine payload slots, 60 original block calls per full step.
            "slot_count": slot_count,
            "original_block_calls_per_full_step": WAN_ORIGINAL_BLOCK_CALLS_PER_STEP,
            # Plan section 2.5: sqrt(d) multiplies SenCache's threshold, so the
            # measured element count is archived with the run that used it.
            "latent_numel": latent_numel,
            "method_config": method_config,
            "matrix_config": (str(args.matrix_config) if args.matrix_config else None),
            "matrix_config_sha256": matrix_sha,
            "budget": args.budget,
            "dataset": args.dataset,
            "prompt_file": str(source.path),
            "prompt_file_sha256": source.sha256,
            "manifest_sha256": source.manifest_sha256,
            "records": records,
        }
        decision_path.write_text(
            json.dumps(decision_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        per_video.append({
            "idx": idx,
            "seed": seed,
            "prompt_id": source.ids[idx],
            "denoise_s": float(timing["denoise_s"]),
            "decode_s": float(timing["decode_s"]),
            "generation_s": float(timing["total_s"]),
            "video_save_s": float(save_s),
            "latency_s": float(timing["total_s"] + save_s),
            "peak_vram_bytes": peak_vram,
            "cache_steps_realized": actual_cache_count,
            "video_file": video_path.name,
            "decisions_file": decision_path.name,
        })
        del video, adapter
        torch.cuda.empty_cache()
        print(
            f"[wan21-baseline] mode={args.mode} idx={idx} seed={seed} "
            f"cache={actual_cache_count} generation={timing['total_s']:.2f}s "
            f"save={save_s:.2f}s",
            flush=True,
        )

    timing_payload = {
        "schema": TIMING_SCHEMA,
        "mode": args.mode,
        "protocol_id": args.protocol_id,
        "method_config": method_config,
        "matrix_config": (str(args.matrix_config) if args.matrix_config else None),
        "matrix_config_sha256": matrix_sha,
        "budget": args.budget,
        "dataset": args.dataset,
        "prompt_file": str(source.path),
        "prompt_file_sha256": source.sha256,
        "base_seed": int(args.seed),
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "device": torch.cuda.get_device_name(0),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        # The frozen config and the upstream commit pin the numbers and the
        # model; this pins the code that read them, which is what the
        # commit/push/pull cluster-sync rule is checked against.
        "git_sha": git_sha(_ROOT),
        "upstream": load_info["identity"]["upstream_source"],
        "attention_backend": attention_backend,
        "video_writer": video_writer,
        "model_load_s": float(load_info["load_seconds"]),
        "wallclock_total_s": (time.perf_counter() - wall_started
                              + float(load_info["load_seconds"])),
        "generation_per_video_s_mean": _mean([row["generation_s"] for row in per_video]),
        "denoise_per_video_s_mean": _mean([row["denoise_s"] for row in per_video]),
        "decode_per_video_s_mean": _mean([row["decode_s"] for row in per_video]),
        "video_save_per_video_s_mean": _mean([row["video_save_s"] for row in per_video]),
        "latency_per_video_s_mean": _mean([row["latency_s"] for row in per_video]),
        "per_video": per_video,
    }
    if keep_existing_timing(args, len(per_video)):
        print(f"[wan21-baseline] resumed with nothing to generate; kept "
              f"{timing_path_for(args).name}")
        return 0
    timing_path_for(args).write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
