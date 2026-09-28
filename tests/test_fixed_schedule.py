from __future__ import annotations

import pytest

from lib.fixed_schedule import evenly_spaced_cache_steps, validate_cache_steps


def test_evenly_spaced_schedule_has_exact_budget_and_boundaries() -> None:
    cache = evenly_spaced_cache_steps(
        num_steps=50,
        cache_count=29,
        first_full_steps=3,
        last_full_steps=1,
    )
    assert len(cache) == 29
    assert tuple(sorted(set(cache))) == cache
    assert not {0, 1, 2, 49}.intersection(cache)


def test_validate_cache_steps_rejects_forced_full_overlap() -> None:
    with pytest.raises(ValueError, match="forced-full"):
        validate_cache_steps(
            (1, 3),
            num_steps=5,
            cache_count=2,
            forced_full_steps={0, 1, 4},
        )
