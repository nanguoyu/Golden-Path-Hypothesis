"""Small helpers for exact-count fixed cache schedules."""

from __future__ import annotations

from collections.abc import Iterable


def validate_cache_steps(
    cache_steps: Iterable[int],
    *,
    num_steps: int,
    cache_count: int | None = None,
    forced_full_steps: Iterable[int] = (),
) -> tuple[int, ...]:
    steps = tuple(int(step) for step in cache_steps)
    if steps != tuple(sorted(set(steps))):
        raise ValueError("cache steps must be sorted and unique")
    if any(step < 0 or step >= int(num_steps) for step in steps):
        raise ValueError("cache steps contain an out-of-range index")
    if cache_count is not None and len(steps) != int(cache_count):
        raise ValueError(
            f"cache schedule has {len(steps)} steps, expected {int(cache_count)}"
        )
    overlap = set(steps).intersection(int(step) for step in forced_full_steps)
    if overlap:
        raise ValueError(f"cache schedule overlaps forced-full steps: {sorted(overlap)}")
    return steps


def evenly_spaced_cache_steps(
    *,
    num_steps: int,
    cache_count: int,
    first_full_steps: int = 3,
    last_full_steps: int = 1,
) -> tuple[int, ...]:
    """Return an exact-count schedule with approximately even full evaluations.

    Fixed-interval predictors cannot realize every integer cache count. This
    helper preserves their regular-refresh principle while making the number of
    full evaluations exact.
    """

    num_steps = int(num_steps)
    cache_count = int(cache_count)
    first_full_steps = int(first_full_steps)
    last_full_steps = int(last_full_steps)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if first_full_steps < 1 or last_full_steps < 0:
        raise ValueError("invalid full-step boundary counts")

    full_count = num_steps - cache_count
    forced = set(range(first_full_steps))
    forced.update(range(num_steps - last_full_steps, num_steps))
    if full_count < len(forced) or full_count > num_steps:
        raise ValueError("cache_count is incompatible with forced full steps")

    eligible = [step for step in range(num_steps) if step not in forced]
    need = full_count - len(forced)
    selected: set[int] = set()
    if need == 1:
        selected.add(eligible[len(eligible) // 2])
    elif need > 1:
        for index in range(need):
            ideal = index * (len(eligible) - 1) / (need - 1)
            candidate = eligible[round(ideal)]
            if candidate in selected:
                candidate = min(
                    (step for step in eligible if step not in selected),
                    key=lambda step: (abs(eligible.index(step) - ideal), step),
                )
            selected.add(candidate)

    full_steps = forced | selected
    cache_steps = tuple(step for step in range(num_steps) if step not in full_steps)
    return validate_cache_steps(
        cache_steps,
        num_steps=num_steps,
        cache_count=cache_count,
        forced_full_steps=forced,
    )
