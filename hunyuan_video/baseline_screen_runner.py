#!/usr/bin/env python3
"""HunyuanVideo runner for native-gate and fixed-schedule baseline screening."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hunyuan_video.backend import (
    generate,
    inference_step_count,
    load_official_sampler,
    save_video,
)
from hunyuan_video.matrix_config import (
    BUDGETS,
    DATASETS,
    RUNNER_MODE_METHOD,
    load_matrix_config,
)
from hunyuan_video.config import RunSpec, load_protocol
from hunyuan_video.records import load_self_hashed_json, sha256_file
from hunyuan_video.trajectory_retention import (
    capture_latent_path,
    retain_trajectory,
    retention_plan,
)
from lib.io_utils import read_prompts, seed_for, split_shard


MODES = (
    "original",
    "reuse_exact",
    "seacache",
    "teacache",
    "taylorseer_exact",
    "hicache_exact",
    "l2p_output_exact",
    "meancache_exact",
    "dicache",
    "sencache",
)

DYNAMIC_MODES = frozenset({"seacache", "teacache", "dicache", "sencache"})
FIXED_SCHEDULE_MODES = frozenset(
    {
        "reuse_exact",
        "taylorseer_exact",
        "hicache_exact",
        "l2p_output_exact",
        "meancache_exact",
    }
)


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
    parser.add_argument("--mode", choices=MODES, required=True)
    # exactly one of these; --prompt_manifest is the only transport that can
    # carry the two Penguin prompts containing a newline, and the only one that
    # carries a prompt_id rather than a line index
    parser.add_argument("--prompt_file", type=Path)
    parser.add_argument("--prompt_manifest", type=Path,
                        help="evaluation prompt manifest from "
                             "analysis/hunyuan_video/build_evaluation_prompts.py")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_base", type=Path, required=True)
    parser.add_argument("--protocol_id", default="HY-CachePaper-480")
    parser.add_argument("--cache_count", type=int)
    parser.add_argument("--cache_steps", type=_step_list, default=())
    parser.add_argument("--infer_steps", type=int)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--first_enhance", type=int, default=1)
    parser.add_argument("--power_exp", type=float, default=3.0)
    parser.add_argument("--max_order", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--l2p_weights", type=Path)
    parser.add_argument("--l2p_min_abs_weight", type=float, default=0.0)
    parser.add_argument("--meancache_schedule", type=Path)
    parser.add_argument("--meancache_jvp_span", type=int, default=4,
                        help="global fallback JVP span for cached steps the schedule "
                             "file gives no per-edge span for. The offline search only "
                             "solved per-edge spans for MeanCache's own table, so every "
                             "transplanted schedule (video SPX) runs on this value "
                             "(methods/meancache.py span_for)")
    parser.add_argument("--spx_relax_warmup", action="store_true",
                        help="video SPX only: relax MeanCache's runtime forbidden warmup "
                             "from steps {0..4} to {0,1}. The span is already clamped to "
                             "the available history and degrades to velocity reuse at "
                             "history <= 1 (methods/meancache.py:98-102), so this does "
                             "not change MeanCache's behaviour on its own table (which "
                             "caches from step 5 up); it is what lets a foreign schedule "
                             "run under the mean_vel payload. Refused together with "
                             "--matrix_config: the frozen matrix runs the unrelaxed rule")
    parser.add_argument("--dicache_ret_ratio", type=float, default=0.2)
    parser.add_argument("--dicache_probe_depth", type=int, default=1)
    parser.add_argument("--sencache_sensitivity_path", type=Path)
    parser.add_argument("--sencache_threshold_start", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt_indices", type=_idx_list, default=None,
                        help="generate exactly these manifest indices (sorted unique "
                             "comma list) instead of the contiguous --limit slice; an "
                             "idx keeps its manifest position, so its seed and its "
                             "prompt_id are unchanged. Not combinable with --limit")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--matrix_config", type=Path,
        help="frozen baseline-matrix config; with --budget it supplies this method's "
             "threshold or schedule instead of the command line")
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


# the defaults, read off the parser instead of copied: a duplicated literal
# that drifts makes every config-driven cell refuse a flag nobody passed
_PARSER_DEFAULTS = {action.dest: action.default
                    for action in build_parser()._actions}


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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


def _apply_matrix_config(args: argparse.Namespace) -> str | None:
    """Fill the method's frozen hyper-parameters in from the config, or return
    None when the run is not driven by one.

    The plan makes the matrix read one frozen file and forbids it from falling
    back to a default, so a missing cell has to stop the run rather than be
    filled in. Everything downstream -- validation, `_method_config`, the
    adapters -- then sees exactly what a command-line run would produce, which
    is what makes a config-driven smoke test evidence about the matrix.
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
    method = RUNNER_MODE_METHOD.get(args.mode)
    if method is None:
        raise SystemExit(f"mode {args.mode} is not one of the matrix's nine methods")

    # Read the parser's own defaults rather than repeating them: a duplicated
    # literal that drifts would blame the operator for a flag they never passed.
    supplied = [name for name in ("threshold", "cache_count", "cache_steps",
                                  "meancache_schedule")
                if _differs_from_default(args, name)]
    if args.mode not in DYNAMIC_MODES:
        supplied = [name for name in supplied if name != "threshold"]
    if supplied:
        raise SystemExit(
            f"{', '.join(sorted(supplied))} came from the command line, but "
            f"--matrix_config supplies them; pass one or the other")

    ########################
    # Knobs beyond the threshold that move the realized cache count come from
    # the frozen entry too, not the command line -- otherwise a cell runs on a
    # number the frozen record does not contain. Which ones each method may
    # freeze is `matrix_config.METHOD_PARAMS`.
    ########################
    unrecordable = [name for name in ("power_exp", "first_enhance", "sigma",
                                      "l2p_min_abs_weight", "dicache_ret_ratio",
                                      "dicache_probe_depth", "sencache_threshold_start",
                                      "max_order", "meancache_jvp_span",
                                      "spx_relax_warmup")
                    if _differs_from_default(args, name)]
    if unrecordable:
        raise SystemExit(
            f"{', '.join(sorted(unrecordable))} came from the command line; a "
            f"config-driven cell takes them from the frozen entry's method_params, "
            f"so a value here would not survive into the matrix")

    config = load_matrix_config(args.matrix_config)
    # The thresholds were swept under one protocol. Switching to the other one
    # changes the resolution and the frame count, so every gate's score is taken
    # over a different token grid -- a far larger distribution shift than the
    # dataset axis this file exists for, and the only field that records it is
    # already here.
    frozen_protocol = config.payload.get("protocol_id")
    if frozen_protocol is not None and args.protocol_id != frozen_protocol:
        raise SystemExit(
            f"--protocol_id {args.protocol_id} is not the {frozen_protocol} this config "
            f"was frozen for; its thresholds were swept at that protocol's resolution "
            f"and frame count")
    # KeyError if the cell is not frozen
    entry = config.entry(method, args.budget, args.dataset)
    for name, value in entry.method_params.items():
        # ints stay ints: probe_depth and first_enhance are counts
        current = getattr(args, name)
        setattr(args, name, type(current)(value) if isinstance(current, int) else float(value))
    # max_order is not calibrated -- it is what makes TaylorSeer O1 and HiCache
    # O2 the methods they are (`_validate` pins both) -- so the mode supplies it
    # rather than the operator, who would otherwise have to pass --max_order 2
    # on every HiCache cell of a run that is supposed to read one frozen file.
    args.max_order = {"taylorseer_exact": 1, "hicache_exact": 2}.get(args.mode, args.max_order)
    for name, digest in entry.assets.items():
        path = getattr(args, name)
        if path is None or not Path(path).is_file():
            raise SystemExit(f"{method} reads {name}, which the frozen config pins by "
                             f"digest; pass the file (got {path!r})")
        actual = sha256_file(Path(path))
        if actual != digest:
            raise SystemExit(
                f"{name} {path} hashes to {actual}, but the frozen config was built "
                f"against {digest}; this cell's threshold was calibrated on that file")
    if entry.threshold is not None:
        args.threshold = float(entry.threshold)
    else:
        args.cache_count = int(entry.cache_count)
        # the frozen entry keeps cache_steps as a tuple for immutability and
        # every adapter type-checks for a list
        args.cache_steps = tuple(int(step) for step in entry.cache_steps)
        if args.mode == "meancache_exact":
            if entry.jvp_spans is None:
                raise SystemExit(f"frozen meancache table {entry.table_id} carries no "
                                 f"jvp_spans; the spans are part of the solved path")
            args.meancache_frozen_spans = {int(k): int(v) for k, v in entry.jvp_spans.items()}
    return config.sha256


def _load_prompts(args: argparse.Namespace) -> tuple[list[str], list[str], Path, str]:
    """`(prompts, prompt_ids, source, source_sha256)` from whichever transport
    was given.

    A line-per-prompt file cannot carry the two Penguin prompts that contain a
    newline, and cannot carry a prompt_id at all; the manifest does both, and
    also states which dataset it is, which is the only thing that can catch a
    row generating one dataset's prompts under the other's frozen thresholds.
    """
    if (args.prompt_file is None) == (args.prompt_manifest is None):
        raise SystemExit("pass exactly one of --prompt_file / --prompt_manifest")
    if args.prompt_file is not None:
        prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
        # a line index, since a text file carries no identity of its own
        ids = [f"prompt-{index}" for index in range(len(prompts))]
        return prompts, ids, args.prompt_file, sha256_file(args.prompt_file)

    payload = load_self_hashed_json(args.prompt_manifest, "manifest_sha256")
    if payload.get("schema") != "hunyuan_video.evaluation_prompts.v1":
        raise SystemExit(f"{args.prompt_manifest} is not an evaluation prompt manifest")
    if args.dataset is not None and payload.get("dataset") != args.dataset:
        raise SystemExit(
            f"--dataset {args.dataset} but the manifest holds "
            f"{payload.get('dataset')!r}; that row would generate one dataset's "
            f"prompts under the other's frozen thresholds")
    items = list(payload["items"])
    if args.limit:
        items = items[: int(args.limit)]
    return ([str(item["prompt"]) for item in items],
            [str(item["prompt_id"]) for item in items],
            args.prompt_manifest,
            str(payload["manifest_sha256"]))


def timing_path_for(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"timing_shard{args.shard_idx:03d}of{args.shard_count:03d}.json")


def keep_existing_timing(args: argparse.Namespace, generated: int) -> bool:
    """Whether this invocation must leave the shard's timing file alone.

    A resumed shard with nothing left to generate has timed nothing, and the
    write at the end of `main` is unconditional: it would replace the measured
    means with zeros, which is exactly what plan section 4.4 reads.
    """
    return (generated == 0 and bool(args.resume)
            and timing_path_for(args).is_file())


def _cell_identity(args: argparse.Namespace, *, matrix_sha: str | None,
                   prompt_sha: str) -> dict[str, Any]:
    """What makes this output directory one cell of the matrix.

    Filenames carry only the prompt index, so two cells writing to one directory
    is invisible: --resume then reports the first cell's videos as the second
    cell's work. This is the `sweep_identity.txt` guard the threshold sweep
    already uses, applied to the matrix runner.
    """
    return {
        "schema": "hunyuan_video.baseline_screen_cell.v1",
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
        differing = sorted(key for key in identity
                           if previous.get(key) != identity[key])
        if differing:
            raise SystemExit(
                f"{output_dir} was produced by a different cell; {differing} differ.\n"
                f"  on disk: { {k: previous.get(k) for k in differing} }\n"
                f"  now:     { {k: identity[k] for k in differing} }\n"
                f"Use a separate --output_dir; resuming here would report the other "
                f"cell's videos as this one's.")
        return
    path.write_text(json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8")


#: Steps MeanCache keeps full at the head of the trajectory. The path search
#: forced 5 and the matrix runs 5; `--spx_relax_warmup` lowers it to 2, which is
#: the smallest value at which the JVP reference `k - span` still lands on a
#: step that has a velocity (two full steps have run).
MEANCACHE_FIRST_FULL_STEPS = 5
MEANCACHE_FIRST_FULL_STEPS_RELAXED = 2

#: A fixed-schedule DiCache needs two anchors before its first cached step, or
#: `lib/dicache.aligned_residual` has no gamma to estimate and silently degrades
#: to plain reuse.
DICACHE_FIRST_FULL_STEPS = 2


def dicache_fixed_schedule(args: argparse.Namespace) -> bool:
    """Whether this run is the video SPX `di_two_anchor` payload on a fixed table.

    DiCache is normally a fused gate + payload with no schedule entry at all
    (plan section 3). Given `--cache_steps` it keeps the payload -- shallow
    probe, gamma clamp, two-anchor extrapolation -- and takes the action from
    the table instead of from its own threshold, which is what makes the fifth
    payload column comparable with the other four on a foreign schedule.
    """
    return args.mode == "dicache" and bool(args.cache_steps)


def _validate(args: argparse.Namespace, num_steps: int) -> None:
    if num_steps != 50:
        raise SystemExit("the HunyuanVideo baseline-screen lane is frozen to 50 steps")
    if args.mode == "original":
        if args.cache_count is not None or args.cache_steps:
            raise SystemExit("original mode does not accept cache_count/cache_steps")
        return
    if args.infer_steps is not None:
        raise SystemExit("--infer_steps is only valid for original mode")
    if args.mode in DYNAMIC_MODES and not dicache_fixed_schedule(args):
        if args.cache_count is not None or args.cache_steps:
            raise SystemExit(
                "native dynamic gates do not accept cache_count/cache_steps; "
                "calibrate their threshold against the dataset-level mean K"
            )
        if float(args.threshold) <= 0.0:
            raise SystemExit("native dynamic gates require --threshold > 0")
        if args.mode == "sencache":
            # The gate loads the frozen table in its constructor, so a missing
            # file would only surface after the sampler is already resident.
            if (
                args.sencache_sensitivity_path is None
                or not args.sencache_sensitivity_path.is_file()
            ):
                raise SystemExit(
                    "sencache requires an existing --sencache_sensitivity_path"
                )
            if float(args.sencache_threshold_start) <= 0.0:
                raise SystemExit("sencache requires --sencache_threshold_start > 0")
        return
    if args.cache_count is None or not 0 <= args.cache_count <= num_steps - 2:
        raise SystemExit("fixed schedule modes require --cache_count in [0, 48]")
    if args.cache_steps:
        if len(args.cache_steps) != args.cache_count:
            raise SystemExit("--cache_steps length differs from --cache_count")
        if args.cache_steps[0] <= 0 or args.cache_steps[-1] >= num_steps - 1:
            raise SystemExit("explicit cache schedule must preserve first and terminal steps")
    if args.mode in {"taylorseer_exact", "hicache_exact", "l2p_output_exact"}:
        if not args.cache_steps:
            raise SystemExit(f"{args.mode} requires a frozen --cache_steps schedule")
    if dicache_fixed_schedule(args):
        # Refused here rather than inside the adapter, which would only notice
        # after the sampler is resident: with fewer than two full steps behind
        # it the two-anchor payload has no gamma and is not DiCache any more.
        early = sorted(step for step in args.cache_steps
                       if step < DICACHE_FIRST_FULL_STEPS)
        if early:
            raise SystemExit(
                f"a fixed-schedule dicache run needs two full steps before its first "
                f"cached one (lib/dicache.aligned_residual needs two anchors); "
                f"steps {early} are inside the warmup")
    if args.mode == "l2p_output_exact":
        if args.l2p_weights is None or not args.l2p_weights.is_file():
            raise SystemExit("l2p_output_exact requires an existing --l2p_weights file")
    if args.mode == "meancache_exact":
        # The cached steps and the per-edge spans are one solved object, so they
        # arrive together -- from the schedule file, or from the frozen entry,
        # never half from each.
        if getattr(args, "meancache_frozen_spans", None) is None:
            if args.meancache_schedule is None or not args.meancache_schedule.is_file():
                raise SystemExit(
                    "meancache_exact requires an existing --meancache_schedule file "
                    "(or --matrix_config with --budget)")
            if args.cache_steps:
                raise SystemExit("meancache_exact reads its schedule from --meancache_schedule")
        first_full = (MEANCACHE_FIRST_FULL_STEPS_RELAXED if args.spx_relax_warmup
                      else MEANCACHE_FIRST_FULL_STEPS)
        if args.spx_relax_warmup:
            print(f"[baseline-screen] --spx_relax_warmup: MeanCache keeps steps "
                  f"0..{first_full - 1} full instead of 0..{MEANCACHE_FIRST_FULL_STEPS - 1}",
                  flush=True)
    if args.mode == "taylorseer_exact" and args.max_order != 1:
        raise SystemExit("the HunyuanVideo TaylorSeer source lane freezes max_order=1")
    if args.mode == "hicache_exact" and args.max_order != 2:
        raise SystemExit("the HunyuanVideo HiCache reimplementation freezes max_order=2")
    if int(args.meancache_jvp_span) < 1:
        raise SystemExit("--meancache_jvp_span must be at least 1 step")


def _method_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "original":
        return {} if args.infer_steps is None else {"infer_steps": args.infer_steps}
    if args.mode in DYNAMIC_MODES:
        config: dict[str, Any] = {"threshold": float(args.threshold)}
        if args.mode in {"seacache", "teacache"}:
            config["first_enhance"] = int(args.first_enhance)
        if args.mode == "seacache":
            config["power_exp"] = float(args.power_exp)
        elif args.mode == "sencache":
            # `threshold` is SenCache's main threshold; the first-20% window has
            # its own, and the frozen sensitivity table is the third input the
            # gate cannot be constructed without.
            config.update(
                {
                    "sensitivity_path": str(args.sencache_sensitivity_path),
                    "threshold_start": float(args.sencache_threshold_start),
                    # the port, backend.py:434 and plan section 2.2 all say 3;
                    # --first_enhance is shared with Sea/Tea whose default is 1,
                    # so passing it through unclamped runs a 1-step warmup. A
                    # config-driven cell cannot reach this clamp with anything
                    # below 3 -- matrix_config.METHOD_PARAMS rejects it at load
                    # -- so the frozen number and the number that runs agree.
                    "first_enhance": max(3, int(args.first_enhance)),
                }
            )
        elif args.mode == "dicache":
            config.update(
                {
                    "ret_ratio": float(args.dicache_ret_ratio),
                    "probe_depth": int(args.dicache_probe_depth),
                }
            )
            if dicache_fixed_schedule(args):
                # The gate is replaced by the table; the payload is untouched.
                config.update(
                    {
                        "cache_count": int(args.cache_count),
                        "cache_steps": list(args.cache_steps),
                    }
                )
        return config

    config = {"cache_count": int(args.cache_count)}
    if args.cache_steps:
        config["cache_steps"] = list(args.cache_steps)
    if args.mode == "taylorseer_exact":
        config["max_order"] = int(args.max_order)
    elif args.mode == "hicache_exact":
        config.update(
            {
                "max_order": int(args.max_order),
                "sigma": float(args.sigma),
                # same clamp and same guarantee as sencache above; HiCache O2
                # freezes first_enhance at exactly 3 because the table the
                # triplet shares is built for a 3-step warmup
                "first_enhance": max(3, int(args.first_enhance)),
            }
        )
    elif args.mode == "l2p_output_exact":
        config.update(
            {
                "weights_path": str(args.l2p_weights),
                "min_abs_weight": float(args.l2p_min_abs_weight),
            }
        )
    elif args.mode == "meancache_exact":
        frozen = getattr(args, "meancache_frozen_spans", None)
        if frozen is not None:
            cache_steps = [int(step) for step in args.cache_steps]
            spans = {str(step): int(span) for step, span in frozen.items()}
        else:
            payload = json.loads(args.meancache_schedule.read_text(encoding="utf-8"))
            cache_steps = [int(step) for step in payload["cache_steps"]]
            spans = {str(step): int(span)
                     for step, span in (payload.get("jvp_spans") or {}).items()}
        # The search's own table carries one span per cached step, and a MISSING
        # span there would silently substitute the runtime default for the span
        # the search chose -- a different schedule. A transplanted schedule
        # (video SPX) has no per-edge solution at all, so the cover rule is
        # "every span given belongs to a cached step", and the uncovered edges
        # take `--meancache_jvp_span` through `MeanCacheMethod.span_for`.
        stray = sorted(step for step in (int(key) for key in spans)
                       if step not in set(cache_steps))
        if stray:
            raise SystemExit(f"--meancache_schedule jvp_spans name uncached steps: {stray}")
        if spans and len(spans) != len(cache_steps):
            raise SystemExit(
                "--meancache_schedule jvp_spans cover only part of the cached steps; the "
                "search solves every edge of a table or none of them. Drop jvp_spans "
                "entirely to run the global --meancache_jvp_span on all of them.")
        config.update({"cache_steps": cache_steps, "jvp_spans": spans,
                       "jvp_span": int(args.meancache_jvp_span)})
        if args.spx_relax_warmup:
            # only written when it is asked for, so a matrix cell's recorded
            # method_config is byte-identical to what it was before this flag
            config["first_full_steps"] = MEANCACHE_FIRST_FULL_STEPS_RELAXED
    return config


def _records(
    adapter: Any | None,
    *,
    executed_steps: int,
    block_count: int,
) -> list[dict[str, Any]]:
    if adapter is None:
        return [
            {
                "step": step,
                "action": "full",
                "reason": "official_full",
                "original_block_calls": block_count,
            }
            for step in range(executed_steps)
        ]
    return [row.as_dict() for row in adapter.decisions]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


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


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("HunyuanVideo generation must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo generation requires CUDA")

    protocol = load_protocol(args.protocol_id)
    matrix_sha = _apply_matrix_config(args)
    _validate(args, protocol.steps)
    prompts, prompt_ids, prompt_source, prompt_file_sha = _load_prompts(args)
    work = select_work(args, prompts)
    if not work:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _check_cell_identity(args.output_dir,
                         _cell_identity(args, matrix_sha=matrix_sha,
                                        prompt_sha=prompt_file_sha))

    sampler, api, load_info = load_official_sampler(args.model_base, protocol)
    transformer = sampler.pipeline.transformer
    block_count = len(transformer.double_blocks) + len(transformer.single_blocks)
    method_config = _method_config(args)
    per_video: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    for idx, prompt in work:
        video_path = args.output_dir / f"video_{idx:05d}.mp4"
        decision_path = args.output_dir / f"decisions_{idx:05d}.json"
        if args.resume and video_path.is_file() and decision_path.is_file():
            continue
        seed = seed_for(args.seed, idx)
        run = RunSpec(
            phase="baseline_screen",
            task_id=f"{args.mode}-{args.shard_idx}-{idx}",
            protocol_id=protocol.protocol_id,
            mode=args.mode,
            prompt_id=prompt_ids[idx],
            prompt=prompt,
            seed=seed,
            repeat=0,
            method_config=method_config,
        )
        plan = retention_plan(
            is_reference=(args.mode == "original"),
            prompt_idx=idx,
            num_steps=protocol.steps,
            base_seed=args.seed,
            t3_seed=args.t3_seed,
            t3_prompt_count=args.t3_prompt_count,
            allow_cached=args.t3_cached,
        ) if args.retain_trajectory else None
        if plan is None:
            output, adapter, generation_s, peak_vram = generate(sampler, protocol, run)
        else:
            with capture_latent_path(sampler) as latent_path:
                output, adapter, generation_s, peak_vram = generate(sampler, protocol, run)
        samples = output["samples"]
        if len(samples) != 1:
            raise RuntimeError(f"expected one generated sample, got {len(samples)}")
        save_started = time.perf_counter()
        save_video(api, samples[0], video_path, protocol.fps)
        save_s = time.perf_counter() - save_started
        executed_steps = inference_step_count(protocol, run)
        records = _records(
            adapter,
            executed_steps=executed_steps,
            block_count=block_count,
        )
        if len(records) != executed_steps:
            raise RuntimeError(
                f"decision count differs from executed steps: {len(records)} != {executed_steps}"
            )
        if plan is not None:
            # after the decisions are read off the adapter and before the video is
            # written: retain_trajectory drops the latents, and a kill between the
            # two only costs this generation
            retain_trajectory(
                latent_path,
                output_dir=args.output_dir,
                stem=f"{idx:05d}",
                num_steps=protocol.steps,
                plan=plan,
                record={
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "budget": args.budget,
                    "seed": seed,
                    "prompt_idx": idx,
                    "prompt_id": prompt_ids[idx],
                    "matrix_config_sha256": matrix_sha,
                },
            )
        actual_cache_count = sum(row["action"] == "cache" for row in records)
        fixed_schedule = (args.mode in FIXED_SCHEDULE_MODES
                          or dicache_fixed_schedule(args))
        if fixed_schedule and actual_cache_count != args.cache_count:
            raise RuntimeError(
                f"fixed schedule cache count mismatch: "
                f"{actual_cache_count} != {args.cache_count}"
            )
        decision_payload = {
            "schema": "hunyuan_video.baseline_screen_decisions.v1",
            "mode": args.mode,
            "prompt_idx": idx,
            "prompt_id": prompt_ids[idx],
            "prompt": prompt,
            "seed": seed,
            "num_steps": executed_steps,
            "fixed_schedule": fixed_schedule,
            "expected_cache_count": args.cache_count if fixed_schedule else None,
            "actual_cache_count": actual_cache_count,
            "method_config": method_config,
            "matrix_config_sha256": matrix_sha,
            "budget": args.budget,
            "dataset": args.dataset,
            "prompt_file": str(prompt_source),
            "prompt_file_sha256": prompt_file_sha,
            "records": records,
        }
        decision_path.write_text(
            json.dumps(decision_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_video.append(
            {
                "idx": idx,
                "seed": seed,
                "generation_s": float(generation_s),
                "video_save_s": float(save_s),
                "latency_s": float(generation_s + save_s),
                "peak_vram_bytes": int(peak_vram),
                "video_file": video_path.name,
                "decisions_file": decision_path.name,
            }
        )
        del output, adapter, samples
        torch.cuda.empty_cache()
        print(
            f"[hunyuan-baseline] mode={args.mode} idx={idx} "
            f"generation={generation_s:.2f}s save={save_s:.2f}s",
            flush=True,
        )

    timing = {
        "schema": "hunyuan_video.baseline_screen_timing.v1",
        "mode": args.mode,
        "protocol_id": protocol.protocol_id,
        "method_config": method_config,
        "matrix_config_sha256": matrix_sha,
        "budget": args.budget,
        "dataset": args.dataset,
        "prompt_file": str(prompt_source),
        "prompt_file_sha256": prompt_file_sha,
        "base_seed": args.seed,
        "seed_rule": "base_plus_prompt_idx",
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "model_load_s": float(load_info["load_seconds"]),
        "wallclock_total_s": time.perf_counter() - wall_started + float(load_info["load_seconds"]),
        "generation_per_video_s_mean": _mean(
            [row["generation_s"] for row in per_video]
        ),
        "video_save_per_video_s_mean": _mean(
            [row["video_save_s"] for row in per_video]
        ),
        "latency_per_video_s_mean": _mean(
            [row["latency_s"] for row in per_video]
        ),
        "per_video": per_video,
    }
    if keep_existing_timing(args, len(per_video)):
        # every prompt was already on disk, so this invocation timed nothing;
        # writing would replace the measured file with zeroed means, and section
        # 4.4 reads exactly those means
        print(f"[baseline-screen] resumed with nothing to generate; kept "
              f"{timing_path_for(args).name}")
        return 0
    timing_path_for(args).write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
