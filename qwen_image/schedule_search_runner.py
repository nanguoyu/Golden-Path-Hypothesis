#!/usr/bin/env python3
"""Search a fixed Qwen-Image schedule on the eight frozen calibration pairs.

Mirror of `flux/schedule_search_runner.py` on the Qwen-Image protocol constants
of `qwen_image/runner.py` (1328x1328, 50 steps, true CFG 4.0, bf16).  The space
and the search recipes are shared through `lib/schedule_search.py`; only the
per-schedule evaluation core differs, and that one is
`qwen_image/sp_cross_runner.py::QwenSPCrossResidualAdapter` with the `reuse`
payload -- the same whole-transformer residual reuse the SPX cells run, reused
rather than rewritten.  The model stays resident and the adapter is
re-installed per candidate; both CFG branches of a step follow the same
schedule bit, as they do in the SPX runner.

There is no Qwen counterpart of the FLUX `--anchor` mode: the K41 truth table
exists only for FLUX, so there is nothing on this side to check against.

Modes: a search, `--probe N` (N evaluations as N/2 one-swap pairs) for the P1
temperature reading, or `--arbitrate` (P3), which re-scores the
`arbitration_candidates` of the search summaries named by
`--candidate_summaries` on the fifty held-out COCO captions of the config at
the frozen arbitration seed, generating each distinct bitstring once.  Every
evaluation is one JSONL row in `<output_dir>/evals.jsonl`.

Every evaluation records the matrix's five metrics per calibration pair --
psnr, ssim and lpips against that pair's full-compute reference, image_reward
and clip on the candidate with that pair's prompt -- and `--objective` picks
which of them the search maximises.

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
    read_eval_trace,
    run_probe,
    score_candidates,
    select_best,
    summary_row,
)

MODEL = "qwen"


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
    parser.add_argument("--model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
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


def _image_array(image: Any) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"decoded image has unexpected shape {array.shape}")
    return array


def run_one(pipe: Any, args: argparse.Namespace, index: int, prompt: str, seed: int):
    """One Qwen-Image generation at the protocol constants."""

    import torch

    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=float(args.true_cfg_scale),
        height=int(args.height),
        width=int(args.width),
        num_inference_steps=int(args.num_steps),
        generator=generator,
        return_dict=True,
        output_type="pil",
    )
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError(f"Qwen-Image returned no image for pair {index}")
    torch.cuda.synchronize()
    return _image_array(images[0])


def generate_pairs(
    pipe: Any,
    args: argparse.Namespace,
    pairs: Sequence[tuple[str, int]],
    indices: Sequence[int],
    cache_steps: Sequence[int],
    num_steps: int,
) -> list[np.ndarray]:
    """The pairs `indices`, generated under one schedule, under residual reuse.

    The adapter is installed once for the schedule and reset before each pair,
    so a pair's image depends on nothing but its own prompt, its own seed and
    the schedule -- which is what lets the pairs be split across processes.
    """

    import torch

    from qwen_image.sp_cross_runner import QwenSPCrossResidualAdapter

    adapter = QwenSPCrossResidualAdapter(
        pipe,
        cache_steps=tuple(int(step) for step in cache_steps),
        payload="reuse",
        num_steps=int(num_steps),
        true_cfg=True,
    )
    adapter.install()
    try:
        out: list[np.ndarray] = []
        for index in indices:
            prompt, seed = pairs[index]
            adapter.reset(prompt_idx=index, seed=seed)
            with torch.no_grad():
                out.append(run_one(pipe, args, index, prompt, seed))
        return out
    finally:
        adapter.restore()


def worker_setup(
    rank: int, world_size: int, owned: Sequence[int], blob: dict[str, Any]
) -> dict[str, Any]:
    """`--gpus > 1`: one Qwen-Image replica on this worker's GPU, its references
    and its own resident copy of the five metric models."""

    import torch
    from diffusers import QwenImagePipeline

    from lib.search_metrics import MetricModels

    args = blob["args"]
    pairs = blob["pairs"]
    num_steps = int(blob["num_steps"])
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    load_start = time.perf_counter()
    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start
    references = generate_pairs(pipe, args, pairs, owned, (), num_steps)
    return {
        "pipe": pipe,
        "args": args,
        "pairs": pairs,
        "num_steps": num_steps,
        "references": dict(zip(owned, references)),
        "metrics": MetricModels(device="cuda"),
        "model_load_s": model_load_s,
        "device": torch.cuda.get_device_name(0),
    }


def worker_evaluate(
    state: dict[str, Any], indices: Sequence[int], cache_steps: Sequence[int]
) -> list[dict[str, float]]:
    """This worker's pairs under one schedule -> their metrics, in `indices` order."""

    candidates = generate_pairs(
        state["pipe"],
        state["args"],
        state["pairs"],
        indices,
        cache_steps,
        state["num_steps"],
    )
    return state["metrics"].score_pairs(
        [state["references"][index] for index in indices],
        candidates,
        [state["pairs"][index][0] for index in indices],
    )


def main() -> int:  # noqa: C901 - one linear driver, as elsewhere in qwen_image/
    args = parse_args()
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen-Image protocol requires --true_cfg_scale > 1")
    config = json.loads(args.model_config.read_text(encoding="utf-8"))

    space = load_space(config, MODEL, args.k)
    if args.num_steps != space.num_steps:
        raise SystemExit(f"this space requires --num_steps {space.num_steps}")
    search_cfg = config["search"]
    if args.arbitrate and not args.candidate_summaries:
        raise SystemExit("--arbitrate requires --candidate_summaries")
    if args.probe and args.arbitrate:
        raise SystemExit("--probe and --arbitrate are two different modes; pick one")
    mode = "arbitrate" if args.arbitrate else ("probe" if args.probe else "search")

    # Calibration pairs, or the fifty held-out arbitration captions at their
    # frozen single seed.
    if mode == "arbitrate":
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

    if args.max_evals:
        max_evals = int(args.max_evals)
    elif mode == "probe":
        max_evals = int(args.probe)
    elif mode == "arbitrate":
        max_evals = 0
    else:
        max_evals = int(search_cfg["caps"][str(args.k)]) - int(
            search_cfg["probe_evals"]
        )

    if int(args.gpus) < 1:
        raise SystemExit("--gpus must be at least 1")

    scales = lookup_objective_scales(search_cfg, args.objective, MODEL, args.k)
    print(
        f"[ss-qwen] mode={mode} K={args.k} algorithm={args.algorithm} "
        f"objective={args.objective} pairs={len(pairs)} max_evals={max_evals} "
        f"gpus={args.gpus}",
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
                },
                tag="ss-qwen",
            )
            close = pool.close
            score = pool.evaluate
            model_load_s = pool.model_load_s
            device_name = pool.device_name or "unknown"
            print(
                f"[ss-qwen] {len(pairs)} full-compute references ready "
                f"across {pool.world_size} workers",
                flush=True,
            )
        else:
            import torch

            from diffusers import QwenImagePipeline

            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }[args.dtype]
            load_start = time.perf_counter()
            pipe = QwenImagePipeline.from_pretrained(
                args.model_id, torch_dtype=dtype
            ).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            torch.cuda.synchronize()
            model_load_s = time.perf_counter() - load_start
            device_name = torch.cuda.get_device_name(0)

            indices = tuple(range(len(pairs)))
            references = generate_pairs(
                pipe, args, pairs, indices, (), space.num_steps
            )
            from lib.search_metrics import MetricModels

            metric_models = MetricModels(device="cuda")
            print(
                f"[ss-qwen] {len(references)} full-compute references ready",
                flush=True,
            )

            def score(cache_steps: Sequence[int]) -> list[dict[str, float]]:
                candidates = generate_pairs(
                    pipe, args, pairs, indices, cache_steps, space.num_steps
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

        if mode == "arbitrate":
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
                print(f"[ss-qwen] resume: {replayed} scored schedules", flush=True)
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
        "schema": "qwen_image_schedule_search_run.v1",
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
        "search_seed": seed_source,
        "max_evals": max_evals,
        "model_id": args.model_id,
        "num_steps": int(args.num_steps),
        "width": int(args.width),
        "height": int(args.height),
        "true_cfg_scale": float(args.true_cfg_scale),
        "negative_prompt": args.negative_prompt,
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
                f"[ss-qwen] delivery {MODEL} K{args.k} {row['name']} "
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
                if key not in ("pairs", "arbitration_candidates", "arbitration")
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
