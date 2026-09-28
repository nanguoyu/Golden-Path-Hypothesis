from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodDecision:
    full: bool
    reason: str
    gate_scalar: float | None = None
    accumulated: float | None = None
    threshold: float | None = None


@dataclass
class FullAction:
    num_steps: int
    step: int = 0

    def reset(self) -> None:
        self.step = 0

    def decide_full(self) -> tuple[bool, str]:
        self.step += 1
        return True, "forced_all_full"


@dataclass
class IntervalAction:
    num_steps: int
    interval: int
    first_enhance: int = 1
    force_last: bool = False
    step: int = 0
    last_full: int | None = None

    def __post_init__(self) -> None:
        if self.interval < 1 or self.first_enhance < 1:
            raise ValueError("interval and first_enhance must be positive")

    def reset(self) -> None:
        self.step = 0
        self.last_full = None

    def decide_full(self) -> tuple[bool, str]:
        current = self.step
        if current < self.first_enhance:
            full, reason = True, "warmup"
        elif self.force_last and current == self.num_steps - 1:
            full, reason = True, "terminal"
        elif self.last_full is None or current - self.last_full >= self.interval:
            full, reason = True, "interval"
        else:
            full, reason = False, "interval_cache"
        if full:
            self.last_full = current
        self.step += 1
        return full, reason


@dataclass
class FixedScheduleAction:
    num_steps: int
    cache_steps: frozenset[int]
    step: int = 0

    def __post_init__(self) -> None:
        invalid = sorted(step for step in self.cache_steps if step < 0 or step >= self.num_steps)
        if invalid:
            raise ValueError(f"cache steps outside trajectory: {invalid}")

    def reset(self) -> None:
        self.step = 0

    def decide_full(self) -> tuple[bool, str]:
        current = self.step
        self.step += 1
        return (False, "fixed_cache") if current in self.cache_steps else (True, "fixed_full")


def evenly_spaced_cache_steps(num_steps: int, cache_count: int) -> frozenset[int]:
    eligible = list(range(1, num_steps - 1))
    if cache_count < 0 or cache_count > len(eligible):
        raise ValueError("cache_count exceeds non-terminal eligible steps")
    if cache_count == 0:
        return frozenset()
    indices = [((2 * i + 1) * len(eligible)) // (2 * cache_count) for i in range(cache_count)]
    selected = {eligible[min(index, len(eligible) - 1)] for index in indices}
    if len(selected) != cache_count:
        for step in eligible:
            if len(selected) == cache_count:
                break
            selected.add(step)
    return frozenset(selected)
