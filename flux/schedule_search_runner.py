#!/usr/bin/env python3
"""Search a fixed FLUX schedule on the eight frozen calibration pairs.

The model stays resident and every candidate schedule is scored the
way the exhaustive K41 run scored one: generate each calibration pair under the
schedule with the residual-reuse payload, and measure it against a
full-compute reference decoded by the same pipeline.  Every evaluation records
the matrix's five metrics per pair -- psnr, ssim and lpips against that pair's
reference, image_reward and clip on the candidate with that pair's prompt --
and `--objective` picks which of them the search maximises.  The per-schedule
evaluation core is `flux/exhaustive_k41_runner.py`'s, reused rather than
rewritten -- one installed `flux/oracle_runner.py::install_oracle` reuse path
serves both the references (empty cache set) and every candidate, and only
`cache_steps_set` changes between calls.  That is also the engine
`flux/sp_cross_runner.py` runs its `reuse` payload through, so search and the
later SPX evaluation share one payload.

Three modes:

``search``  (default) run one algorithm from `lib/schedule_search.py` until the
            restart-unit stop rule fires or the cap is reached;
``--probe N``
            N evaluations arranged as N/2 (schedule, one-swap neighbour) pairs,
            which is what P1 needs to set the annealing temperature: the median
            absolute swap difference fixes the scale of `t_max` and the
            calibration standard error puts a limit under `t_min`;
``--anchor``
            score a given list of schedules on the *original* exhaustive four
            pairs (Parti indices 5,8,9,15 at seed 42 + index), so the numbers
            can be checked against the K41 truth table;
``--arbitrate``
            P3: re-score the `arbitration_candidates` of the search summaries
            named by `--candidate_summaries` on the fifty held-out COCO
            captions of the config, at the frozen arbitration seed.  Identical
            bitstrings from different algorithms are generated once.

Every evaluation is one JSONL row in `<output_dir>/evals.jsonl`.

`--gpus N` (default 1) spreads the pairs of one evaluation over N GPUs of the
node: N worker processes each hold their own replica, their own metric models
and the pairs `i % N == rank`, references included, and the parent gathers the
metrics back in pair order.  The search is unchanged -- it stays sequential,
because each proposal depends on the previous decision -- and so is every
generated image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io_utils import read_prompts  # noqa: E402
from lib.schedule_search import (  # noqa: E402
    ALGORITHMS,
    METRIC_NAMES,
    OBJECTIVES,
    BudgetExhausted,
    EvalSink,
    ObjectiveEvaluator,
    PairPool,
    SearchControl,
    arbitration_candidates,
    compact_arbitration,
    load_space,
    load_warm_starts,
    lookup_objective_scales,
    lookup_temperatures,
    objective_units,
    parse_schedule_line,
    read_eval_trace,
    run_probe,
    score_candidates,
    select_best,
    summary_row,
)

MODEL = "flux"
#: The exhaustive K41 protocol, reproduced for `--anchor`.
ANCHOR_PROMPT_FILE = Path("resources/prompts/partiprompts_full_eval1632_seed42.txt")
ANCHOR_PROMPT_INDICES = (5, 8, 9, 15)
ANCHOR_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_config",
        type=Path,
        default=Path("resources/schedule_search/config.v1.json"),
    )
    parser.add_argument("--k", type=int, choices=(29, 37, 41), required=True)
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="anneal")
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="psnr",
        help="what the search maximises; all five metrics are recorded either "
        "way (psnr_lpips_z needs search.objective_scales from the probe)",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help="run N probe evaluations (N/2 one-swap pairs) instead of a search",
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="score --anchor_schedules on the original exhaustive four pairs",
    )
    parser.add_argument(
        "--anchor_schedules",
        type=Path,
        default=None,
        help="one schedule per line: a 50-bit string or comma-separated full steps",
    )
    parser.add_argument(
        "--arbitrate",
        action="store_true",
        help="P3: score the candidate schedules on the fifty arbitration captions",
    )
    parser.add_argument(
        "--candidate_summaries",
        type=Path,
        nargs="+",
        default=None,
        help="arbitrate mode: the search summary.json files whose "
        "arbitration_candidates are re-scored",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=0,
        help="override the config cap (default: cap minus the shared probe budget)",
    )
    parser.add_argument("--search_seed", type=int, default=0)
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs of this node to spread one evaluation's pairs over "
        "(1 = one resident model in this process, the original path)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="answer schedules already in evals.jsonl from that trace, so a "
        "resubmitted job replays its own path instead of recomputing it",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=0.0,
        help="annealing start temperature in dB; default from the config (P1)",
    )
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", default="3de623fc")
    parser.add_argument(
        "--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev"
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--conditioning_file",
        type=Path,
        default=None,
        help="anchor mode: the exhaustive run's frozen prompt embeddings",
    )
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_pairs(
    pipe: Any,
    conditioning: dict[int, Any],
    pairs: Sequence[tuple[str, int]],
    indices: Sequence[int],
    cache_steps: Sequence[int],
    args: argparse.Namespace,
    height: int,
    width: int,
) -> list[np.ndarray]:
    """The pairs `indices`, generated under one schedule, as uint8 RGB arrays.

    The schedule is the transformer's `cache_steps_set` and the per-trajectory
    oracle state is cleared before each pair, so a pair's image depends on
    nothing but its own conditioning, its own seed and the schedule -- which is
    what lets the pairs be split across processes.
    """

    from flux.exhaustive_k41_runner import _image_array, _run_conditioned
    from flux.oracle_runner import _decode_to_pil, reset_oracle_state

    pipe.transformer.cache_steps_set = frozenset(int(step) for step in cache_steps)
    out: list[np.ndarray] = []
    for index in indices:
        reset_oracle_state(pipe)
        latent = _run_conditioned(pipe, conditioning[index], pairs[index][1], args)
        out.append(_image_array(_decode_to_pil(pipe, latent, height, width)))
        del latent
    return out


def worker_setup(
    rank: int, world_size: int, owned: Sequence[int], blob: dict[str, Any]
) -> dict[str, Any]:
    """`--gpus > 1`: one FLUX replica on this worker's GPU, plus its references.

    Only the conditioning and the references of the pairs this worker owns are
    built here; text encoding carries no state and the reference of pair `i` is
    only ever compared against candidates of pair `i`, generated in this same
    process.  The five metric models are loaded here too and stay resident
    beside the pipeline for the life of the worker.
    """

    import torch
    from diffusers import DiffusionPipeline

    from flux.exhaustive_k41_runner import _encode_prompt, _load_conditioning_artifact
    from flux.oracle_runner import install_oracle
    from lib.search_metrics import MetricModels

    args = blob["args"]
    pairs = blob["pairs"]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype, revision=args.revision
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start

    if blob["conditioning_file"] is not None:
        loaded = _load_conditioning_artifact(
            blob["conditioning_file"],
            pipe=pipe,
            args=args,
            prompt_indices=blob["prompt_indices"],
            selected_prompts=tuple(prompt for prompt, _seed in pairs),
            resolved_model_commit=blob["model_commit"],
        )
        conditioning = {
            index: loaded[prompt_idx]
            for index, prompt_idx in enumerate(blob["prompt_indices"])
        }
    else:
        conditioning = {
            index: _encode_prompt(pipe, pairs[index][0], args) for index in owned
        }

    install_oracle(
        pipe, cache_steps=(), num_steps=blob["num_steps"], cache_mode="seacache"
    )
    height = (args.height // 16) * 16
    width = (args.width // 16) * 16
    references = generate_pairs(
        pipe, conditioning, pairs, owned, (), args, height, width
    )
    return {
        "pipe": pipe,
        "args": args,
        "pairs": pairs,
        "conditioning": conditioning,
        "references": dict(zip(owned, references)),
        "metrics": MetricModels(device="cuda"),
        "height": height,
        "width": width,
        "model_load_s": model_load_s,
        "device": torch.cuda.get_device_name(0),
    }


def worker_evaluate(
    state: dict[str, Any], indices: Sequence[int], cache_steps: Sequence[int]
) -> list[dict[str, float]]:
    """This worker's pairs under one schedule -> their metrics, in `indices` order."""

    candidates = generate_pairs(
        state["pipe"],
        state["conditioning"],
        state["pairs"],
        indices,
        cache_steps,
        state["args"],
        state["height"],
        state["width"],
    )
    return state["metrics"].score_pairs(
        [state["references"][index] for index in indices],
        candidates,
        [state["pairs"][index][0] for index in indices],
    )


def main() -> int:  # noqa: C901 - one linear driver, as elsewhere in flux/
    args = parse_args()
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    space = load_space(config, MODEL, args.k)
    if args.num_steps != space.num_steps:
        raise SystemExit(f"this space requires --num_steps {space.num_steps}")
    search_cfg = config["search"]

    if args.anchor and args.anchor_schedules is None:
        raise SystemExit("--anchor requires --anchor_schedules")
    if args.arbitrate and not args.candidate_summaries:
        raise SystemExit("--arbitrate requires --candidate_summaries")
    modes = [
        name
        for flag, name in (
            (args.anchor, "anchor"),
            (args.probe, "probe"),
            (args.arbitrate, "arbitrate"),
        )
        if flag
    ]
    if len(modes) > 1:
        raise SystemExit(f"{', '.join(modes)} are different modes; pick one")
    mode = modes[0] if modes else "search"

    # Calibration pairs, the exhaustive four pairs under --anchor, or the
    # fifty held-out arbitration captions at their frozen single seed.
    if mode == "anchor":
        prompts = read_prompts(ANCHOR_PROMPT_FILE)
        pairs = [
            (prompts[idx], ANCHOR_SEED + idx) for idx in ANCHOR_PROMPT_INDICES
        ]
        pair_labels = [f"parti_{idx}" for idx in ANCHOR_PROMPT_INDICES]
    elif mode == "arbitrate":
        arbitration = config["arbitration"]
        seed = int(arbitration["seed"])
        pairs = [(row["prompt"], seed) for row in arbitration["prompts"]]
        pair_labels = [f"coco_{row['caption_id']}" for row in arbitration["prompts"]]
    else:
        slots = config["calibration"]["pairs"]
        pairs = [(row["prompt"], int(row["seed"])) for row in slots]
        pair_labels = [row["slot"] for row in slots]

    seed_source = int(args.search_seed or search_cfg["search_seed"])
    rng = np.random.default_rng(
        [seed_source, int(args.k), int(zlib.crc32(args.algorithm.encode()))]
    )

    caps = search_cfg["caps"]
    probe_evals = int(search_cfg["probe_evals"])
    if args.max_evals:
        max_evals = int(args.max_evals)
    elif mode == "probe":
        max_evals = int(args.probe)
    elif mode in ("anchor", "arbitrate"):
        max_evals = 0
    else:
        max_evals = int(caps[str(args.k)]) - probe_evals

    if int(args.gpus) < 1:
        raise SystemExit("--gpus must be at least 1")

    import torch

    from flux.exhaustive_k41_runner import _prompt_text_sha256
    from flux.sp_cross_runner import resolve_model_commit

    revision = args.revision or None
    args.revision = revision  # `_load_conditioning_artifact` compares this field
    weights = resolve_model_commit(args.model_id, revision)
    conditioning_file = (
        args.conditioning_file
        if (mode == "anchor" and args.conditioning_file is not None)
        else None
    )
    args.conditioning_artifact_sha256 = (
        _file_sha256(conditioning_file) if conditioning_file is not None else None
    )
    scales = lookup_objective_scales(search_cfg, args.objective, MODEL, args.k)
    print(
        f"[ss-flux] mode={mode} K={args.k} algorithm={args.algorithm} "
        f"objective={args.objective} pairs={len(pairs)} max_evals={max_evals} "
        f"gpus={args.gpus} commit={weights['model_commit'] or 'UNRESOLVED'}",
        flush=True,
    )

    header = {
        "mode": mode,
        "model": MODEL,
        "k": int(args.k),
        "algorithm": args.algorithm if mode == "search" else mode,
        "objective": args.objective,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sink = EvalSink(args.output_dir / "evals.jsonl", header=header)

    summary: dict[str, Any] = {}
    arbitration_record: dict[str, Any] | None = None
    close: Callable[[], None] = lambda: None
    try:
        if int(args.gpus) > 1:
            # One model replica per GPU; each worker owns the pairs
            # `i % gpus == rank`, references included.
            pool = PairPool(
                world_size=int(args.gpus),
                n_pairs=len(pairs),
                setup=worker_setup,
                evaluate=worker_evaluate,
                blob={
                    "args": args,
                    "pairs": pairs,
                    "num_steps": space.num_steps,
                    "conditioning_file": conditioning_file,
                    "prompt_indices": ANCHOR_PROMPT_INDICES,
                    "model_commit": weights.get("model_commit"),
                },
                tag="ss-flux",
            )
            close = pool.close
            score = pool.evaluate
            model_load_s = pool.model_load_s
            device_name = pool.device_name or "unknown"
            print(
                f"[ss-flux] {len(pairs)} full-compute references ready "
                f"across {pool.world_size} workers",
                flush=True,
            )
        else:
            from diffusers import DiffusionPipeline

            from flux.exhaustive_k41_runner import (
                _encode_prompt,
                _load_conditioning_artifact,
            )
            from flux.oracle_runner import install_oracle

            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[args.dtype]
            load_start = time.perf_counter()
            pipe = DiffusionPipeline.from_pretrained(
                args.model_id, torch_dtype=dtype, revision=revision
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            torch.cuda.synchronize()
            model_load_s = time.perf_counter() - load_start
            device_name = torch.cuda.get_device_name(0)

            # Text conditioning is immutable across schedules, so encode it once.
            if conditioning_file is not None:
                loaded = _load_conditioning_artifact(
                    conditioning_file,
                    pipe=pipe,
                    args=args,
                    prompt_indices=ANCHOR_PROMPT_INDICES,
                    selected_prompts=tuple(prompt for prompt, _seed in pairs),
                    resolved_model_commit=weights.get("model_commit"),
                )
                conditioning = {
                    index: loaded[prompt_idx]
                    for index, prompt_idx in enumerate(ANCHOR_PROMPT_INDICES)
                }
            else:
                conditioning = {
                    index: _encode_prompt(pipe, prompt, args)
                    for index, (prompt, _seed) in enumerate(pairs)
                }

            close = install_oracle(
                pipe, cache_steps=(), num_steps=space.num_steps, cache_mode="seacache"
            )
            height = (args.height // 16) * 16
            width = (args.width // 16) * 16
            indices = tuple(range(len(pairs)))
            references = generate_pairs(
                pipe, conditioning, pairs, indices, (), args, height, width
            )
            from lib.search_metrics import MetricModels

            metric_models = MetricModels(device="cuda")
            print(
                f"[ss-flux] {len(references)} full-compute references ready",
                flush=True,
            )

            def score(cache_steps: Sequence[int]) -> list[dict[str, float]]:
                candidates = generate_pairs(
                    pipe, conditioning, pairs, indices, cache_steps, args, height, width
                )
                return metric_models.score_pairs(
                    references, candidates, [prompt for prompt, _seed in pairs]
                )

        # `wall_s` is the search itself, model load and references excluded.
        started = time.perf_counter()
        evaluate = ObjectiveEvaluator(
            space=space,
            score=score,
            sink=sink,
            objective=args.objective,
            scales=scales,
        )

        if mode == "anchor":
            rows = [
                line
                for line in args.anchor_schedules.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            scored = []
            for line in rows:
                combo = parse_schedule_line(space, line)
                evaluate(combo)
                metrics = evaluate.metrics_of(combo)
                scored.append(
                    {
                        "schedule": ",".join(str(s) for s in space.full_steps(combo)),
                        "bits": space.bits(combo),
                        "per_pair": list(metrics["psnr"]),
                        "mean_psnr_db": float(np.mean(metrics["psnr"])),
                        "min_psnr_db": float(np.min(metrics["psnr"])),
                        "metrics": evaluate.metric_means(combo),
                    }
                )
            summary = {"anchor_rows": scored}
        elif mode == "arbitrate":
            arbitration_record = score_candidates(
                evaluate=evaluate,
                space=space,
                summaries=args.candidate_summaries,
                model=MODEL,
                k=int(args.k),
                pair_labels=pair_labels,
                seed=pairs[0][1],
            )
            summary = {"arbitration": compact_arbitration(arbitration_record)}
        elif mode == "probe":
            summary = {
                "probe": run_probe(
                    evaluate,
                    rng,
                    space,
                    int(args.probe),
                    hill_window=int(search_cfg["hill_window"]),
                    local_probability=float(search_cfg["local_probability"]),
                    metrics_of=evaluate.metrics_of,
                )
            }
        else:
            warm_starts = load_warm_starts(config, space, MODEL, args.k)
            control = SearchControl(
                evaluate,
                max_evals=max_evals,
                se_factor=float(search_cfg["se_factor"]),
                stop_units=int(search_cfg["stop_units"]),
                use_stop_rule=args.algorithm != "random",
            )
            replayed = 0
            if args.resume:
                replayed = control.preload(
                    evaluate.preload_trace(
                        read_eval_trace(
                            args.output_dir / "evals.jsonl",
                            space,
                            algorithm=args.algorithm,
                            k=args.k,
                            objective=args.objective,
                        )
                    )
                )
                print(f"[ss-flux] resume: {replayed} scored schedules", flush=True)
            temperatures = lookup_temperatures(
                search_cfg, args.objective, MODEL, args.k
            )
            t_max = float(args.t_max or temperatures.get("t_max") or 0.0)
            t_min = float(args.t_min or temperatures.get("t_min") or 0.0)
            if args.algorithm == "anneal" and not (t_max > t_min > 0.0):
                raise SystemExit(
                    f"annealing on the {args.objective} objective needs "
                    "t_max > t_min > 0 from --t_max/--t_min or the config's "
                    f"temperatures for that objective (units: "
                    f"{objective_units(args.objective)})"
                )
            function, label = ALGORITHMS[args.algorithm]
            try:
                function(
                    control,
                    rng,
                    space,
                    warm_starts=warm_starts,
                    sa_iters=int(search_cfg["chain_lengths"][str(args.k)]),
                    t_max=t_max,
                    t_min=t_min,
                    hill_iters=int(search_cfg["hill_iters"]),
                    hill_window=int(search_cfg["hill_window"]),
                    on_state=sink.on_state,
                )
            except BudgetExhausted:
                pass
            se = control.best_se()
            selected = select_best(
                control.records, se=se, se_factor=float(search_cfg["se_factor"])
            )
            candidates = arbitration_candidates(
                space,
                control.records,
                count=int(search_cfg["arbitration_candidates"]),
                min_hamming=int(search_cfg["arbitration_min_hamming"]),
            )
            summary = {
                "label": label,
                "evaluations": control.calls,
                "replayed_evaluations": replayed,
                "restart_units": control.units,
                "stop_reason": control.stop_reason or "loop_end",
                "warm_starts": len(warm_starts),
                "temperature": {"t_max": t_max, "t_min": t_min},
                "sa_iters": int(search_cfg["chain_lengths"][str(args.k)]),
                "best_se_db": se,
                "selected": None
                if selected is None
                else summary_row(space, evaluate, selected),
                "arbitration_candidates": [
                    summary_row(space, evaluate, row) for row in candidates
                ],
            }
    finally:
        close()
        sink.close()

    run = {
        "schema": "flux_schedule_search_run.v1",
        "mode": mode,
        "model": MODEL,
        "k": int(args.k),
        "algorithm": args.algorithm if mode == "search" else mode,
        "objective": args.objective,
        "objective_units": objective_units(args.objective),
        "objective_scales": scales,
        "space": space.identity_payload,
        "config_file": str(args.model_config),
        "config_sha256": _file_sha256(args.model_config),
        "pairs": [
            {"slot": label, "prompt": prompt, "seed": seed}
            for label, (prompt, seed) in zip(pair_labels, pairs)
        ],
        "prompt_text_sha256": _prompt_text_sha256(
            tuple(prompt for prompt, _seed in pairs)
        ),
        "conditioning_artifact_sha256": args.conditioning_artifact_sha256,
        "search_seed": seed_source,
        "max_evals": max_evals,
        "model_id": args.model_id,
        "model_revision": revision,
        "weights": weights,
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "guidance": float(args.guidance),
        "dtype": args.dtype,
        "payload": "residual_reuse",
        "metric": "uint8_rgb_psnr_data_range_255",
        "metrics": list(METRIC_NAMES),
        "model_load_s": float(model_load_s),
        "wall_s": float(time.perf_counter() - started),
        "device": device_name,
        **summary,
    }
    _atomic_write_json(args.output_dir / "summary.json", run)
    if arbitration_record is not None:
        _atomic_write_json(
            args.output_dir / "arbitration.json",
            {**run, "arbitration": arbitration_record},
        )
        for row in arbitration_record["delivery"]:
            print(
                f"[ss-flux] delivery {MODEL} K{args.k} {row['name']} "
                f"{row['bits']} {args.objective} mean="
                f"{row.get('arbitration_mean_objective', row['arbitration_mean_psnr_db']):.4f} "
                f"min={row.get('arbitration_min_objective', row['arbitration_min_psnr_db']):.4f} "
                f"(psnr {row['arbitration_mean_psnr_db']:.3f} dB)",
                flush=True,
            )
    print(
        json.dumps(
            {
                key: value
                for key, value in run.items()
                if key
                not in ("pairs", "anchor_rows", "arbitration_candidates", "arbitration")
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
