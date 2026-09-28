#!/usr/bin/env python3
"""Search a fixed-budget BudCache schedule on HunyuanVideo terminal latents."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
# Unconditional: Python puts this script's own directory (analysis/) at
# sys.path[0], and analysis/hunyuan_video/ is a namespace package that shadows
# the real hunyuan_video/ at the repo root. The old `if not in sys.path` guard
# was defeated by wrappers exporting PYTHONPATH=<repo root> -- the root was
# "already present", but BEHIND the script dir, so the shadow won and the
# import below died with "No module named 'hunyuan_video.backend'".
sys.path.insert(0, str(_ROOT))

from hunyuan_video.backend import generate, load_official_sampler
from hunyuan_video.config import GenerationProtocol, RunSpec, load_protocol
from lib.fixed_schedule import validate_cache_steps
from lib.io_utils import read_prompts, seed_for


NUM_STEPS = 50
# The deployed runtime is `golden_reuse_schedule`: a fixed schedule driving the
# coarse whole-transformer residual reuse, the same payload the FLUX search
# deploys (`flux/fixed_residual_exact.py:44-49`). Search and deployment agree.
RUNTIME_MODE = "golden_reuse_schedule"
MEMO_FORMAT = "hunyuan-video-budcache-search-memo-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_base", type=Path, required=True)
    parser.add_argument("--protocol_id", default="HY-CachePaper-480")
    parser.add_argument("--cache_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--search_seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--sa_iters", type=int, default=200)
    parser.add_argument("--temperature_max", type=float, default=0.05)
    parser.add_argument("--temperature_min", type=float, default=1e-5)
    parser.add_argument("--hill_iters", type=int, default=20)
    parser.add_argument("--hill_window", type=int, default=3)
    return parser.parse_args()


def forced_full_steps(num_steps: int) -> frozenset[int]:
    return frozenset({0, 1, 2, int(num_steps) - 1})


def _random_schedule(
    rng: random.Random,
    *,
    num_steps: int,
    cache_count: int,
) -> tuple[int, ...]:
    forced = forced_full_steps(num_steps)
    full_count = num_steps - cache_count
    if full_count < len(forced):
        raise ValueError("BudCache cache_count leaves too few mandatory full steps")
    extra = rng.sample(
        [step for step in range(num_steps) if step not in forced],
        full_count - len(forced),
    )
    full = set(forced) | set(extra)
    return tuple(step for step in range(num_steps) if step not in full)


def _swap_candidate(
    rng: random.Random,
    schedule: tuple[int, ...],
    *,
    num_steps: int,
    local_probability: float = 0.7,
) -> tuple[int, ...]:
    cached = set(schedule)
    movable_full = [step for step in range(3, num_steps - 1) if step not in cached]
    targets = [step for step in schedule if 3 <= step < num_steps - 1]
    source = rng.choice(movable_full)
    nearby = [step for step in targets if abs(step - source) <= 3]
    destination = (
        rng.choice(nearby)
        if nearby and rng.random() < local_probability
        else rng.choice(targets)
    )
    cached.remove(destination)
    cached.add(source)
    return tuple(sorted(cached))


def _neighbors(
    schedule: tuple[int, ...],
    *,
    num_steps: int,
    window: int,
) -> list[tuple[int, ...]]:
    cached = set(schedule)
    full = [step for step in range(num_steps) if step not in cached]
    candidates: set[tuple[int, ...]] = set()
    for source in full:
        if source < 3 or source == num_steps - 1:
            continue
        for destination in range(
            max(3, source - int(window)),
            min(num_steps - 2, source + int(window)) + 1,
        ):
            if destination == source or destination not in cached:
                continue
            candidate = set(cached)
            candidate.remove(destination)
            candidate.add(source)
            candidates.add(tuple(sorted(candidate)))
    candidates.discard(schedule)
    return sorted(candidates)


def annealing_temperature(
    iteration: int,
    *,
    sa_iters: int,
    temperature_max: float,
    temperature_min: float,
) -> float:
    progress = iteration / max(int(sa_iters), 1)
    return float(temperature_max) * (
        float(temperature_min) / float(temperature_max)
    ) ** progress


def _read_memo(
    path: Path,
    memo_key: dict[str, Any] | None,
    *,
    num_steps: int,
    cache_count: int,
    forced: frozenset[int],
) -> dict[tuple[int, ...], float]:
    """Reload the schedule -> loss memo an earlier submission left behind."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != MEMO_FORMAT:
        raise ValueError(f"not a BudCache search memo: {path}")
    if payload.get("key") != memo_key:
        raise ValueError(
            f"BudCache search memo {path} was written for a different search "
            "(prompts, seed or tier changed); delete it before resubmitting"
        )
    memo: dict[tuple[int, ...], float] = {}
    for raw, value in payload["losses"].items():
        schedule = validate_cache_steps(
            (int(step) for step in raw.split(",") if step),
            num_steps=num_steps,
            cache_count=cache_count,
            forced_full_steps=forced,
        )
        memo[schedule] = float(value)
    return memo


def _write_memo(
    path: Path,
    memo_key: dict[str, Any] | None,
    memo: dict[tuple[int, ...], float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MEMO_FORMAT,
        "key": memo_key,
        "losses": {
            ",".join(str(step) for step in schedule): value
            for schedule, value in memo.items()
        },
    }
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(payload), encoding="utf-8")
    scratch.replace(path)


@dataclass
class SearchOutcome:
    schedule: tuple[int, ...]
    loss: float
    evaluated: int
    history: list[dict[str, Any]] = field(default_factory=list)
    primed: int = 0


def search_schedule(
    objective: Callable[[tuple[int, ...]], float],
    *,
    num_steps: int,
    cache_count: int,
    search_seed: int,
    restarts: int = 1,
    sa_iters: int = 200,
    temperature_max: float = 0.05,
    temperature_min: float = 1e-5,
    hill_iters: int = 20,
    hill_window: int = 3,
    memo_path: Path | None = None,
    memo_key: dict[str, Any] | None = None,
) -> SearchOutcome:
    """Simulated annealing followed by hill climbing over exactly-K schedules.

    `objective` sees every distinct schedule once; repeats are served from the
    memo keyed by the schedule tuple, which is what keeps the number of video
    generations far below the nominal iteration count.

    With `memo_path` the memo also survives the process. The search is a
    deterministic function of `search_seed`, so a resubmission of the same tier
    walks the same schedules and serves every one it already scored from the
    file instead of regenerating the calibration videos. `memo_key` records
    which search the file belongs to; a mismatch is refused rather than mixed.
    """

    forced = forced_full_steps(num_steps)
    loss_cache: dict[tuple[int, ...], float] = {}
    if memo_path is not None and Path(memo_path).exists():
        loss_cache.update(
            _read_memo(
                memo_path,
                memo_key,
                num_steps=num_steps,
                cache_count=cache_count,
                forced=forced,
            )
        )
    primed = len(loss_cache)

    def evaluate(schedule: tuple[int, ...]) -> float:
        schedule = validate_cache_steps(
            schedule,
            num_steps=num_steps,
            cache_count=cache_count,
            forced_full_steps=forced,
        )
        if schedule in loss_cache:
            return loss_cache[schedule]
        value = float(objective(schedule))
        loss_cache[schedule] = value
        if memo_path is not None:
            _write_memo(memo_path, memo_key, loss_cache)
        return value

    rng = random.Random(search_seed)
    best_schedule: tuple[int, ...] | None = None
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    for restart in range(int(restarts)):
        current = _random_schedule(
            rng,
            num_steps=num_steps,
            cache_count=cache_count,
        )
        current_loss = evaluate(current)
        if current_loss < best_loss:
            best_schedule, best_loss = current, current_loss
        for iteration in range(int(sa_iters)):
            temperature = annealing_temperature(
                iteration,
                sa_iters=sa_iters,
                temperature_max=temperature_max,
                temperature_min=temperature_min,
            )
            candidate = _swap_candidate(rng, current, num_steps=num_steps)
            candidate_loss = evaluate(candidate)
            delta = candidate_loss - current_loss
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                current, current_loss = candidate, candidate_loss
            if current_loss < best_loss:
                best_schedule, best_loss = current, current_loss
            history.append(
                {
                    "restart": restart,
                    "stage": "sa",
                    "iteration": iteration,
                    "current_loss": current_loss,
                    "best_loss": best_loss,
                }
            )
        for iteration in range(int(hill_iters)):
            scored = [
                (evaluate(candidate), candidate)
                for candidate in _neighbors(
                    current,
                    num_steps=num_steps,
                    window=hill_window,
                )
            ]
            candidate_loss, candidate = min(scored, default=(current_loss, current))
            if candidate_loss >= current_loss:
                break
            current, current_loss = candidate, candidate_loss
            if current_loss < best_loss:
                best_schedule, best_loss = current, current_loss
            history.append(
                {
                    "restart": restart,
                    "stage": "hill",
                    "iteration": iteration,
                    "current_loss": current_loss,
                    "best_loss": best_loss,
                }
            )
    if best_schedule is None:
        raise RuntimeError("BudCache search evaluated no schedule")
    return SearchOutcome(
        schedule=best_schedule,
        loss=best_loss,
        evaluated=len(loss_cache),
        history=history,
        primed=primed,
    )


class TerminalLatentCapture:
    """Read the terminal latent off the solver during one HunyuanVideo run.

    `HunyuanVideoSampler.predict` hard-codes `output_type="pil"`, and the
    pipeline clamps whatever it returns into [0, 1]
    (`hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py:1089`), so the
    latent cannot be recovered from the returned samples. The last value the
    scheduler produces is exactly z_50.

    The patch goes on the scheduler *class*, not on the live instance:
    `predict` builds a fresh `FlowMatchDiscreteScheduler` and assigns it to the
    pipeline on every call (`hyvideo/inference.py:611-616`), so an
    instance-level patch installed before the call is discarded before the
    first solver step ever runs.
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.scheduler_class = type(pipeline.scheduler)
        self.latent: torch.Tensor | None = None
        self.steps = 0
        self._original: Any = None

    def __enter__(self) -> "TerminalLatentCapture":
        original = self.scheduler_class.step
        if getattr(original, "_terminal_latent_capture", False):
            raise RuntimeError("terminal latent capture is already installed")

        @functools.wraps(original)
        def step(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(scheduler, *args, **kwargs)
            sample = output[0] if isinstance(output, tuple) else output.prev_sample
            self.latent = sample.detach().to(torch.float32).cpu()
            self.steps += 1
            return output

        step._terminal_latent_capture = True
        self.scheduler_class.step = step
        self._original = original
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.scheduler_class.step = self._original
        self._original = None


def _run_spec(
    protocol: GenerationProtocol,
    *,
    prompt: str,
    prompt_idx: int,
    seed: int,
    mode: str,
    schedule: tuple[int, ...] | None,
) -> RunSpec:
    method_config: dict[str, Any] = {}
    if schedule is not None:
        method_config = {
            "cache_steps": list(schedule),
            "cache_count": len(schedule),
        }
    return RunSpec(
        phase="budcache_search",
        task_id=f"budcache-{mode}-{prompt_idx}",
        protocol_id=protocol.protocol_id,
        mode=mode,
        prompt_id=f"budcache-calib-{prompt_idx}",
        prompt=prompt,
        seed=seed,
        repeat=0,
        method_config=method_config,
    )


def terminal_latent(
    sampler: Any,
    protocol: GenerationProtocol,
    run: RunSpec,
) -> torch.Tensor:
    with TerminalLatentCapture(sampler.pipeline) as capture:
        output, _adapter, _seconds, _peak = generate(sampler, protocol, run)
    del output
    if capture.steps != protocol.steps or capture.latent is None:
        raise RuntimeError(
            f"terminal latent capture saw {capture.steps} solver steps, "
            f"expected {protocol.steps}"
        )
    torch.cuda.empty_cache()
    return capture.latent


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("HunyuanVideo BudCache search must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo BudCache search requires CUDA")
    protocol = load_protocol(args.protocol_id)
    if protocol.steps != NUM_STEPS:
        raise SystemExit(f"BudCache search is frozen to {NUM_STEPS} steps")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    if not prompts:
        raise SystemExit("BudCache search requires at least one calibration prompt")
    seeds = [seed_for(args.seed, index) for index in range(len(prompts))]

    sampler, _api, _load = load_official_sampler(args.model_base, protocol)
    reference = [
        terminal_latent(
            sampler,
            protocol,
            _run_spec(
                protocol,
                prompt=prompt,
                prompt_idx=index,
                seed=seed,
                mode="original",
                schedule=None,
            ),
        )
        for index, (prompt, seed) in enumerate(zip(prompts, seeds))
    ]

    evaluations = 0

    def objective(schedule: tuple[int, ...]) -> float:
        nonlocal evaluations
        losses: list[float] = []
        for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
            student = terminal_latent(
                sampler,
                protocol,
                _run_spec(
                    protocol,
                    prompt=prompt,
                    prompt_idx=index,
                    seed=seed,
                    mode=RUNTIME_MODE,
                    schedule=schedule,
                ),
            )
            losses.append(
                torch.nn.functional.mse_loss(student, reference[index]).item()
            )
        evaluations += 1
        value = float(sum(losses) / len(losses))
        print(
            f"[hunyuan-budcache] eval={evaluations} K={len(schedule)} mse={value:.6e}",
            flush=True,
        )
        return value

    # The memo sits next to --output so it is per tier. A resubmission of the
    # same tier reloads it and only generates the schedules it has not scored.
    memo_path = args.output.with_name(f"{args.output.stem}.memo.json")
    memo_key = {
        "protocol_id": protocol.protocol_id,
        "runtime_mode": RUNTIME_MODE,
        "cache_count": args.cache_count,
        "prompt_file": str(args.prompt_file),
        "prompt_count": len(prompts),
        "base_seed": args.seed,
        "search_seed": args.search_seed,
    }

    started = time.perf_counter()
    outcome = search_schedule(
        objective,
        num_steps=protocol.steps,
        cache_count=args.cache_count,
        search_seed=args.search_seed,
        restarts=args.restarts,
        sa_iters=args.sa_iters,
        temperature_max=args.temperature_max,
        temperature_min=args.temperature_min,
        hill_iters=args.hill_iters,
        hill_window=args.hill_window,
        memo_path=memo_path,
        memo_key=memo_key,
    )

    cached = set(outcome.schedule)
    payload = {
        "format": "hunyuan-video-budcache-stage1-schedule-v1",
        "model": "hunyuan_video",
        "protocol_id": protocol.protocol_id,
        "runtime_mode": RUNTIME_MODE,
        "payload": "latest_whole_transformer_residual",
        "num_steps": protocol.steps,
        "cache_count": args.cache_count,
        "cache_steps": list(outcome.schedule),
        "full_steps": [
            step for step in range(protocol.steps) if step not in cached
        ],
        # F.mse_loss is an element-wise mean, not a squared L2 norm; the argmin
        # is the same but the recorded number is on the mean convention.
        "objective": "terminal_latent_elementwise_mean_squared_error",
        "best_terminal_latent_mse": outcome.loss,
        "calibration_prompt_file": str(args.prompt_file),
        "calibration_prompt_count": len(prompts),
        "base_seed": args.seed,
        "seed_rule": "base_plus_prompt_idx",
        "search_seed": args.search_seed,
        "restarts": args.restarts,
        "sa_iters": args.sa_iters,
        "hill_iters": args.hill_iters,
        "hill_window": args.hill_window,
        "evaluated_schedules": outcome.evaluated,
        "memo_path": str(memo_path),
        "memo_primed_schedules": outcome.primed,
        # This submission only: a resubmission primed from the memo regenerates
        # the reference latents but none of the schedules the memo already has.
        "generated_videos": (len(reference) + evaluations * len(prompts)),
        "search_seconds": time.perf_counter() - started,
        "history": outcome.history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "history"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
