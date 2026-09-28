#!/usr/bin/env python3
"""BudCache Stage-1 schedule search on the repository's FLUX pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.oracle_runner import install_oracle, reset_oracle_state, _run_one_pipe_call
from lib.fixed_schedule import validate_cache_steps
from lib.io_utils import read_prompts, seed_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_name", choices=("flux-dev",), default="flux-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search_seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--sa_iters", type=int, default=200)
    parser.add_argument("--temperature_max", type=float, default=0.05)
    parser.add_argument("--temperature_min", type=float, default=1e-5)
    parser.add_argument("--hill_iters", type=int, default=20)
    parser.add_argument("--hill_window", type=int, default=3)
    return parser.parse_args()


def _random_schedule(
    rng: random.Random,
    *,
    num_steps: int,
    cache_count: int,
) -> tuple[int, ...]:
    forced_full = {0, 1, 2, num_steps - 1}
    full_count = num_steps - cache_count
    extra = rng.sample(
        [step for step in range(num_steps) if step not in forced_full],
        full_count - len(forced_full),
    )
    full = forced_full | set(extra)
    return tuple(step for step in range(num_steps) if step not in full)


def _swap_candidate(
    rng: random.Random,
    schedule: tuple[int, ...],
    *,
    num_steps: int,
    local_probability: float = 0.7,
) -> tuple[int, ...]:
    cached = set(schedule)
    movable_full = [
        step for step in range(3, num_steps - 1) if step not in cached
    ]
    targets = [step for step in schedule if 3 <= step < num_steps - 1]
    source = rng.choice(movable_full)
    local = [step for step in targets if abs(step - source) <= 3]
    destination = (
        rng.choice(local)
        if local and rng.random() < local_probability
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
    out: set[tuple[int, ...]] = set()
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
            out.add(tuple(sorted(candidate)))
    out.discard(schedule)
    return sorted(out)


def main() -> int:
    args = parse_args()
    if args.num_steps != 50:
        raise SystemExit("BudCache screening is frozen to 50 steps")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    if not prompts:
        raise SystemExit("BudCache search requires at least one calibration prompt")

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    teardown = install_oracle(
        pipe,
        cache_steps=(),
        num_steps=args.num_steps,
        cache_mode="seacache",
    )
    teacher: list[torch.Tensor] = []
    try:
        pipe.transformer.cache_steps_set = frozenset()
        for index, prompt in enumerate(prompts):
            reset_oracle_state(pipe)
            teacher.append(
                _run_one_pipe_call(
                    pipe,
                    prompt,
                    seed_for(args.seed, index),
                    args,
                ).detach()
            )

        loss_cache: dict[tuple[int, ...], float] = {}

        @torch.no_grad()
        def objective(schedule: tuple[int, ...]) -> float:
            schedule = validate_cache_steps(
                schedule,
                num_steps=args.num_steps,
                cache_count=args.cache_count,
                forced_full_steps={0, 1, 2, args.num_steps - 1},
            )
            if schedule in loss_cache:
                return loss_cache[schedule]
            pipe.transformer.cache_steps_set = frozenset(schedule)
            losses = []
            for index, prompt in enumerate(prompts):
                reset_oracle_state(pipe)
                student = _run_one_pipe_call(
                    pipe,
                    prompt,
                    seed_for(args.seed, index),
                    args,
                )
                losses.append(
                    torch.nn.functional.mse_loss(
                        student.to(torch.float32),
                        teacher[index].to(torch.float32),
                    ).item()
                )
            value = float(sum(losses) / len(losses))
            loss_cache[schedule] = value
            return value

        rng = random.Random(args.search_seed)
        best_schedule: tuple[int, ...] | None = None
        best_loss = math.inf
        history: list[dict[str, Any]] = []
        started = time.perf_counter()
        for restart in range(args.restarts):
            current = _random_schedule(
                rng,
                num_steps=args.num_steps,
                cache_count=args.cache_count,
            )
            current_loss = objective(current)
            if current_loss < best_loss:
                best_schedule, best_loss = current, current_loss
            for iteration in range(args.sa_iters):
                progress = iteration / max(args.sa_iters, 1)
                temperature = args.temperature_max * (
                    args.temperature_min / args.temperature_max
                ) ** progress
                candidate = _swap_candidate(
                    rng,
                    current,
                    num_steps=args.num_steps,
                )
                candidate_loss = objective(candidate)
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

            for iteration in range(args.hill_iters):
                candidates = _neighbors(
                    current,
                    num_steps=args.num_steps,
                    window=args.hill_window,
                )
                scored = [(objective(candidate), candidate) for candidate in candidates]
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

        assert best_schedule is not None
        payload = {
            "format": "flux-budcache-stage1-schedule-v1",
            "num_steps": args.num_steps,
            "cache_count": args.cache_count,
            "cache_steps": list(best_schedule),
            "full_steps": [
                step for step in range(args.num_steps) if step not in set(best_schedule)
            ],
            "best_terminal_latent_mse": best_loss,
            "calibration_prompt_file": str(args.prompt_file),
            "calibration_prompt_count": len(prompts),
            "base_seed": args.seed,
            "search_seed": args.search_seed,
            "restarts": args.restarts,
            "sa_iters": args.sa_iters,
            "hill_iters": args.hill_iters,
            "evaluated_schedules": len(loss_cache),
            "search_seconds": time.perf_counter() - started,
            "history": history,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in payload.items() if key != "history"}, indent=2))
    finally:
        teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
