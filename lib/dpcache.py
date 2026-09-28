"""DPCache path selection and non-uniform Taylor feature prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
import torch


DerivativeHistory = dict[int, torch.Tensor]


def derivatives_from_history(
    history: Sequence[tuple[int, torch.Tensor]],
    *,
    order: int = 2,
) -> DerivativeHistory:
    """Reproduce DPCache's recursive non-uniform finite differences."""

    if not history:
        raise ValueError("DPCache history cannot be empty")
    ordered = [(int(step), value) for step, value in history]
    if any(right[0] <= left[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("DPCache history steps must be strictly increasing")

    cache: dict[tuple[int, int], torch.Tensor | None] = {}

    def derivative(index: int, degree: int) -> torch.Tensor | None:
        key = (index, degree)
        if key in cache:
            return cache[key]
        if degree == 0:
            value: torch.Tensor | None = ordered[index][1]
        elif index == 0:
            value = None
        else:
            current = derivative(index, degree - 1)
            previous = derivative(index - 1, degree - 1)
            if current is None or previous is None:
                value = None
            else:
                gap = ordered[index][0] - ordered[index - 1][0]
                value = (current - previous) / float(gap)
        cache[key] = value
        return value

    result: DerivativeHistory = {}
    for degree in range(int(order) + 1):
        value = derivative(len(ordered) - 1, degree)
        if value is None:
            break
        result[degree] = value
    return result


def update_derivatives(
    previous: Mapping[int, torch.Tensor] | None,
    feature: torch.Tensor,
    *,
    step_gap: int,
    order: int = 2,
) -> DerivativeHistory:
    """Update DPCache's divided-difference Taylor state at a full step."""

    gap = int(step_gap)
    if gap <= 0:
        raise ValueError("step_gap must be positive")
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")

    old = {} if previous is None else dict(previous)
    updated: DerivativeHistory = {0: feature.detach()}
    for degree in range(order):
        prior = old.get(degree)
        if prior is None:
            break
        updated[degree + 1] = (updated[degree] - prior) / float(gap)
    return updated


def predict_derivatives(
    history: Mapping[int, torch.Tensor],
    *,
    step_offset: int,
    order: int = 2,
) -> torch.Tensor:
    """Evaluate the Taylor polynomial stored by :func:`update_derivatives`."""

    if 0 not in history:
        raise ValueError("DPCache history has no zeroth-order feature")
    offset = float(step_offset)
    result = history[0].detach().clone()
    for degree in range(1, min(int(order), max(history)) + 1):
        if degree not in history:
            break
        result.add_(history[degree], alpha=(offset**degree) / math.factorial(degree))
    return result


def select_full_steps(
    cost_tensor: np.ndarray,
    *,
    total_steps: int,
    full_count: int,
    first_full_steps: int = 3,
    last_full_steps: int = 1,
    max_jump_fraction: float = 0.3,
) -> tuple[int, ...]:
    """Run DPCache's path-aware dynamic program for an exact full-step count."""

    total_steps = int(total_steps)
    full_count = int(full_count)
    first_full_steps = int(first_full_steps)
    last_full_steps = int(last_full_steps)
    costs = np.asarray(cost_tensor, dtype=np.float64)
    expected_shape = (total_steps, total_steps + 1, total_steps + 1)
    if costs.shape != expected_shape:
        raise ValueError(f"DPCache cost tensor shape {costs.shape}, expected {expected_shape}")

    mandatory = set(range(first_full_steps))
    mandatory.update(range(total_steps - last_full_steps, total_steps))
    mandatory.add(0)
    if full_count < len(mandatory) or full_count > total_steps:
        raise ValueError("full_count is incompatible with mandatory full steps")

    sentinel = total_steps
    nodes = total_steps + 1
    edge_count = full_count + 1
    is_mandatory = np.zeros(nodes, dtype=bool)
    for step in mandatory:
        is_mandatory[step] = True
    is_mandatory[sentinel] = True

    dp = np.full((edge_count, nodes), np.inf, dtype=np.float64)
    parent = np.full((edge_count, nodes), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    max_jump = max(int(total_steps * float(max_jump_fraction)), 1)

    for used in range(1, edge_count):
        for destination in range(used, nodes):
            start = max(used - 1, destination - max_jump)
            for source in range(start, destination):
                if not np.isfinite(dp[used - 1, source]):
                    continue
                if np.any(is_mandatory[source + 1 : destination]):
                    continue
                if source == 0:
                    anchor = 0
                elif is_mandatory[source] and is_mandatory[source - 1]:
                    anchor = source - 1
                else:
                    anchor = int(parent[used - 1, source])
                if anchor < 0 or anchor >= total_steps:
                    continue
                transition = costs[anchor, source, destination]
                if not np.isfinite(transition):
                    continue
                candidate = dp[used - 1, source] + transition
                if candidate < dp[used, destination]:
                    dp[used, destination] = candidate
                    parent[used, destination] = source

    if not np.isfinite(dp[edge_count - 1, sentinel]):
        raise RuntimeError("DPCache dynamic program could not reach the terminal sentinel")

    path = [sentinel]
    current = sentinel
    for used in range(edge_count - 1, 0, -1):
        current = int(parent[used, current])
        if current < 0:
            raise RuntimeError("DPCache dynamic-program backtracking failed")
        path.append(current)
    full_steps = tuple(sorted(path[1:]))
    if len(full_steps) != full_count:
        raise RuntimeError("DPCache dynamic program returned the wrong full-step count")
    if not mandatory.issubset(full_steps):
        raise RuntimeError("DPCache dynamic program skipped a mandatory full step")
    return full_steps


def cache_steps_from_full(
    full_steps: tuple[int, ...] | list[int],
    *,
    total_steps: int,
) -> tuple[int, ...]:
    full = frozenset(int(step) for step in full_steps)
    return tuple(step for step in range(int(total_steps)) if step not in full)


@torch.no_grad()
def calibration_cost_tensor(
    features: Sequence[torch.Tensor],
    *,
    order: int = 2,
    sentinel_alpha: float = 0.8,
    max_anchor_fraction: float = 0.3,
) -> np.ndarray:
    """Construct the official DPCache 3-D accumulated L1 cost tensor."""

    if int(order) != 2:
        raise ValueError("the FLUX DPCache baseline freezes order=2")
    steps = len(features)
    if steps <= order:
        raise ValueError("DPCache calibration needs more features than its order")
    if not 0.0 <= float(sentinel_alpha) <= 1.0:
        raise ValueError("sentinel_alpha must be in [0,1]")

    values = [feature.detach().to(torch.float32) for feature in features]
    tail = [
        (step, values[step])
        for step in range(steps - order - 1, steps)
    ]
    sentinel_prediction = predict_derivatives(
        derivatives_from_history(tail, order=order),
        step_offset=1,
        order=order,
    )
    sentinel_seed = torch.stack(values[-(order + 1) :], dim=0).mean(dim=0)
    sentinel = (
        float(sentinel_alpha) * sentinel_prediction
        + (1.0 - float(sentinel_alpha)) * sentinel_seed
    )

    costs = np.full((steps, steps + 1, steps + 1), np.inf, dtype=np.float64)
    for source in range(order):
        costs[: source + 1, source, source + 1] = 0.0

    anchor_limit = max(int(steps * float(max_anchor_fraction)), 1)
    for source in range(order, steps):
        for anchor in range(max(0, source - anchor_limit), source):
            if anchor < order - 1:
                continue
            history = [
                (index, values[index])
                for index in range(anchor - (order - 1), anchor + 1)
            ]
            history.append((source, values[source]))
            derivatives = derivatives_from_history(history, order=order)
            accumulated = 0.0
            for destination in range(source + 1, steps + 1):
                prediction = predict_derivatives(
                    derivatives,
                    step_offset=destination - source,
                    order=order,
                )
                target = sentinel if destination == steps else values[destination]
                accumulated += float(
                    (prediction / 2.0 - target / 2.0).abs().mean().item()
                )
                costs[anchor, source, destination] = accumulated
    return costs
