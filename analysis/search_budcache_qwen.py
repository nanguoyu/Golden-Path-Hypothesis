#!/usr/bin/env python3
"""Search a fixed-budget BudCache schedule on Qwen-Image terminal latents."""

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

from lib.fixed_schedule import validate_cache_steps
from lib.io_utils import read_prompts, seed_for
from qwen_image.coarse_cache import (
    QwenCoarseConfig,
    install_qwen_coarse_forward,
    reset_qwen_coarse_state,
    restore_qwen_coarse_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cache_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--search_seed", type=int, default=20260723)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int, default=3)
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
    if full_count < len(forced_full):
        raise ValueError("BudCache cache_count leaves too few mandatory full steps")
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
    cache_targets = [
        step for step in schedule if 3 <= step < num_steps - 1
    ]
    source = rng.choice(movable_full)
    nearby = [step for step in cache_targets if abs(step - source) <= 3]
    destination = (
        rng.choice(nearby)
        if nearby and rng.random() < local_probability
        else rng.choice(cache_targets)
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


def _bitstring(schedule: tuple[int, ...], num_steps: int) -> str:
    cached = set(schedule)
    return "".join("1" if step in cached else "0" for step in range(num_steps))


def _run_latent(
    pipe: Any,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        generator=generator,
        output_type="latent",
        return_dict=True,
    )
    latent = getattr(result, "images", None)
    if not isinstance(latent, torch.Tensor):
        raise RuntimeError("Qwen pipeline did not return a terminal latent")
    return latent.detach()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen BudCache search requires CUDA")
    if args.true_cfg_scale <= 1.0:
        raise SystemExit("Qwen BudCache search requires true_cfg_scale > 1")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    if not prompts:
        raise SystemExit("Qwen BudCache search requires calibration prompts")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to("cuda")
    seeds = [seed_for(args.seed, index) for index in range(len(prompts))]
    with torch.no_grad():
        teacher = [
            _run_latent(pipe, prompt, seed, args)
            for prompt, seed in zip(prompts, seeds)
        ]

    rng = random.Random(args.search_seed)
    initial = _random_schedule(
        rng,
        num_steps=args.num_steps,
        cache_count=args.cache_count,
    )
    install_qwen_coarse_forward(
        pipe,
        QwenCoarseConfig(
            mode="BudCache",
            num_steps=args.num_steps,
            first_enhance=3,
            fixed_cache_steps=initial,
            true_cfg=True,
        ),
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
        bits = _bitstring(schedule, args.num_steps)
        losses: list[float] = []
        for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
            reset_qwen_coarse_state(
                pipe,
                prompt_idx=index,
                seed=seed,
                locked_action_bitstring=bits,
            )
            student = _run_latent(pipe, prompt, seed, args)
            losses.append(
                torch.nn.functional.mse_loss(
                    student.to(torch.float32),
                    teacher[index].to(torch.float32),
                ).item()
            )
        value = float(sum(losses) / len(losses))
        loss_cache[schedule] = value
        return value

    best_schedule: tuple[int, ...] | None = None
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for restart in range(args.restarts):
            current = (
                initial
                if restart == 0
                else _random_schedule(
                    rng,
                    num_steps=args.num_steps,
                    cache_count=args.cache_count,
                )
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
                scored = [
                    (objective(candidate), candidate)
                    for candidate in _neighbors(
                        current,
                        num_steps=args.num_steps,
                        window=args.hill_window,
                    )
                ]
                candidate_loss, candidate = min(
                    scored,
                    default=(current_loss, current),
                )
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
    finally:
        restore_qwen_coarse_forward(pipe)

    assert best_schedule is not None
    cached = set(best_schedule)
    payload = {
        "format": "qwen-image-budcache-stage1-schedule-v1",
        "model": "qwen_image",
        "num_steps": args.num_steps,
        "cache_count": args.cache_count,
        "cache_steps": list(best_schedule),
        "full_steps": [
            step for step in range(args.num_steps) if step not in cached
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
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "history"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
