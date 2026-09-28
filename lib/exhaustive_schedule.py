"""Finite K=41 schedule universe used by the exhaustive path experiment.

The universe is deliberately narrower than all 50-bit strings: steps 0, 1, 2,
and 49 are full, and exactly five additional full steps are selected from
3..48.  Every member therefore has nine full and 41 cached steps.  Paths are
ranked by the lexicographic order of the five variable full-step positions.

This module is pure Python so the enumeration, sharding, merge, and tests share
one definition without importing a model runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class ExhaustiveScheduleSpace:
    num_steps: int = 50
    cache_count: int = 41
    forced_full_steps: tuple[int, ...] = (0, 1, 2, 49)
    variable_start: int = 3
    variable_end: int = 48
    variable_full_count: int = 5
    payload: str = "reuse"

    @property
    def variable_steps(self) -> tuple[int, ...]:
        return tuple(range(self.variable_start, self.variable_end + 1))

    @property
    def total(self) -> int:
        return math.comb(len(self.variable_steps), self.variable_full_count)

    @property
    def full_count(self) -> int:
        return len(self.forced_full_steps) + self.variable_full_count

    @property
    def identity_payload(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "schema": "exhaustive_schedule_space.v1",
                "variable_steps": list(self.variable_steps),
                "total": self.total,
                "full_count": self.full_count,
            }
        )
        return payload

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.identity_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


K41_SPACE = ExhaustiveScheduleSpace()


def _validate_n_k(n: int, k: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid combination shape n={n}, k={k}")


def unrank_lex(rank: int, n: int, k: int) -> tuple[int, ...]:
    """Return the `rank`-th k-combination of range(n) in lexicographic order."""

    _validate_n_k(n, k)
    total = math.comb(n, k)
    rank = int(rank)
    if not (0 <= rank < total):
        raise ValueError(f"rank {rank} outside [0, {total})")
    out: list[int] = []
    lower = 0
    remaining_rank = rank
    for pos in range(k):
        remaining = k - pos - 1
        for candidate in range(lower, n - remaining):
            count = math.comb(n - candidate - 1, remaining)
            if remaining_rank < count:
                out.append(candidate)
                lower = candidate + 1
                break
            remaining_rank -= count
        else:  # pragma: no cover - guarded by rank bounds
            raise AssertionError("combination unranking exhausted candidates")
    return tuple(out)


def rank_lex(combination: Sequence[int], n: int, k: int) -> int:
    """Inverse of :func:`unrank_lex` for a sorted unique combination."""

    _validate_n_k(n, k)
    values = tuple(int(v) for v in combination)
    if len(values) != k or tuple(sorted(set(values))) != values:
        raise ValueError(f"combination must contain {k} sorted unique values")
    if values and (values[0] < 0 or values[-1] >= n):
        raise ValueError(f"combination {values} outside range({n})")
    rank = 0
    lower = 0
    for pos, value in enumerate(values):
        remaining = k - pos - 1
        for candidate in range(lower, value):
            rank += math.comb(n - candidate - 1, remaining)
        lower = value + 1
    return rank


def _next_combination(values: list[int], n: int) -> bool:
    """Advance a sorted combination in place; return False at the final one."""

    k = len(values)
    for pos in range(k - 1, -1, -1):
        limit = n - k + pos
        if values[pos] < limit:
            values[pos] += 1
            for tail in range(pos + 1, k):
                values[tail] = values[tail - 1] + 1
            return True
    return False


def iter_lex_range(start: int, end: int, n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Yield combinations whose lexicographic ranks are in [start, end)."""

    _validate_n_k(n, k)
    total = math.comb(n, k)
    start, end = int(start), int(end)
    if not (0 <= start <= end <= total):
        raise ValueError(f"invalid rank interval [{start}, {end}) for total {total}")
    if start == end:
        return
    values = list(unrank_lex(start, n, k))
    for rank in range(start, end):
        yield tuple(values)
        if rank + 1 < end and not _next_combination(values, n):
            raise AssertionError("combination iterator ended before requested rank")


def split_interval(total: int, shard_count: int, shard_idx: int) -> tuple[int, int]:
    """Balanced contiguous rank interval, matching the project's prompt shards."""

    total, shard_count, shard_idx = int(total), int(shard_count), int(shard_idx)
    if total < 0:
        raise ValueError("total must be non-negative")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not (0 <= shard_idx < shard_count):
        raise ValueError(f"shard_idx {shard_idx} outside [0, {shard_count})")
    base, remainder = divmod(total, shard_count)
    start = shard_idx * base + min(shard_idx, remainder)
    size = base + (1 if shard_idx < remainder else 0)
    return start, start + size


def split_rank_interval(
    start: int, end: int, shard_count: int, shard_idx: int
) -> tuple[int, int]:
    """Balanced shard of an arbitrary half-open rank interval."""

    start, end = int(start), int(end)
    if start < 0 or end < start:
        raise ValueError(f"invalid rank interval [{start}, {end})")
    local_start, local_end = split_interval(end - start, shard_count, shard_idx)
    return start + local_start, start + local_end


def variable_full_steps(
    rank: int, space: ExhaustiveScheduleSpace = K41_SPACE
) -> tuple[int, ...]:
    indices = unrank_lex(rank, len(space.variable_steps), space.variable_full_count)
    return tuple(space.variable_steps[index] for index in indices)


def schedule_for_rank(
    rank: int, space: ExhaustiveScheduleSpace = K41_SPACE
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Return `(full_steps, cache_steps, bits)` for one universe rank.

    The bit convention matches SPX: ``1`` is cached and ``0`` is full.
    """

    variable = variable_full_steps(rank, space)
    full = tuple(sorted((*space.forced_full_steps, *variable)))
    full_set = frozenset(full)
    cache = tuple(step for step in range(space.num_steps) if step not in full_set)
    bits = "".join("1" if step in cache else "0" for step in range(space.num_steps))
    if len(cache) != space.cache_count or len(full) != space.full_count:
        raise AssertionError("schedule construction violated the frozen budget")
    return full, cache, bits


def rank_for_full_steps(
    full_steps: Iterable[int], space: ExhaustiveScheduleSpace = K41_SPACE
) -> int:
    full = tuple(sorted(set(int(v) for v in full_steps)))
    if len(full) != space.full_count:
        raise ValueError(f"expected {space.full_count} full steps, got {full}")
    if not set(space.forced_full_steps).issubset(full):
        raise ValueError(f"full steps omit forced set {space.forced_full_steps}")
    variable = tuple(step for step in full if step not in space.forced_full_steps)
    positions = tuple(space.variable_steps.index(step) for step in variable)
    return rank_lex(positions, len(space.variable_steps), space.variable_full_count)


def mask_hex(bits: str) -> str:
    if len(bits) != K41_SPACE.num_steps or set(bits) - {"0", "1"}:
        raise ValueError("bits must be one 50-character binary schedule")
    return f"{int(bits, 2):013x}"
