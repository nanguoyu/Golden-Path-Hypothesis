"""The SenCache structural ceiling, checked two ways and against the plan.

Plan: docs/sencache_recalibration_plan_zh.md sections 9.1 and 9.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.sencache_frontier import (  # noqa: E402
    FIRST_ENHANCE,
    ceiling_knobs,
    run_limit_is_inert,
    run_limit_refusals,
    strict_steps,
    structural_ceiling,
    structural_ceiling_dp,
    unprotected_ceiling,
)


@pytest.mark.parametrize("switch_ratio", [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.06])
@pytest.mark.parametrize("max_skip", [1, 2, 3, 5, 7, 10, 13, 14, 20, 38, 39, 43, 60])
def test_closed_form_matches_step_walk(switch_ratio: float, max_skip: int) -> None:
    assert structural_ceiling(switch_ratio, max_skip) == structural_ceiling_dp(
        switch_ratio, max_skip
    )


def test_upstream_lane_ceiling() -> None:
    """The three rows of the plan's table that hold."""
    assert structural_ceiling(0.2, 10) == 36
    assert structural_ceiling(0.2, 13) == 37
    assert structural_ceiling(0.2, 10**6) == 39


def test_shrunk_strict_region_needs_a_matching_max_skip() -> None:
    """The row section 9.5 corrects: 43 is the window, not the ceiling at n=14."""
    assert strict_steps(0.12) == 6
    assert (50 - 1) - strict_steps(0.12) == 43  # cacheable window
    assert structural_ceiling(0.12, 14) == 41  # not 43
    assert structural_ceiling(0.12, 43) == 43


def test_ceiling_is_monotone_in_max_skip() -> None:
    previous = 0
    for max_skip in range(1, 50):
        current = structural_ceiling(0.2, max_skip)
        assert current >= previous
        previous = current


@pytest.mark.parametrize(
    "target,switch_ratio,max_skip,ceiling",
    [(29, 0.2, 10, 36), (37, 0.2, 39, 39), (41, 0.12, 43, 43)],
)
def test_selected_knobs(target: int, switch_ratio: float, max_skip: int,
                        ceiling: int) -> None:
    picked = ceiling_knobs(target)
    assert picked["switch_ratio"] == switch_ratio
    assert picked["max_skip"] == max_skip
    assert picked["ceiling"] == ceiling
    assert picked["ceiling"] >= target + 2


def test_k29_leaves_upstream_knobs_alone() -> None:
    picked = ceiling_knobs(29)
    assert picked["max_skip_is_upstream"]
    assert picked["switch_ratio_is_upstream"]


def test_no_more_conservative_combination_exists() -> None:
    """Every knob that is moved is moved because nothing smaller clears the bar."""
    for target in (37, 41):
        picked = ceiling_knobs(target)
        need = target + 2
        if picked["max_skip"] > 10:
            assert structural_ceiling(picked["switch_ratio"],
                                      picked["max_skip"] - 1) < need
        if picked["switch_ratio"] != 0.2:
            larger = [r for r in (0.2, 0.18, 0.16, 0.14) if r > picked["switch_ratio"]]
            for switch_ratio in larger:
                assert structural_ceiling(switch_ratio, max_skip=50) < need


# --- the n ladder of the plan's section 9.1.1 -------------------------------


def test_the_ablation_ladder_runs_at_k29() -> None:
    """Every rung's ceiling has to clear 29 or it cannot hold the budget."""
    for max_skip, ceiling in ((3, 30), (10, 36), (20, 38), (39, 39)):
        assert structural_ceiling(0.2, max_skip) == ceiling
        assert ceiling >= 29


def test_the_top_rung_covers_both_main_cell_values() -> None:
    """At K29 the run limit is inert from 39 up, so n=39 and n=43 are one gate.

    Not merely an equal ceiling: neither value refuses a cache at any reachable
    decision point, so the two produce identical decisions and the ladder needs
    one rung rather than two.
    """
    assert run_limit_refusals(0.2, 39) == []
    assert run_limit_refusals(0.2, 43) == []
    assert run_limit_is_inert(0.2, 39) and run_limit_is_inert(0.2, 43)


def test_the_lower_rungs_are_genuinely_different_gates() -> None:
    for max_skip in (3, 10, 20):
        assert run_limit_refusals(0.2, max_skip), max_skip


def test_k41_still_needs_43_on_its_own_lane() -> None:
    """The equivalence is a property of K29's strict window, not of n itself."""
    assert structural_ceiling(0.12, 39) == 42
    assert structural_ceiling(0.12, 43) == 43
    assert run_limit_refusals(0.12, 39)
    assert run_limit_refusals(0.12, 43) == []


# --- which ceiling bounds a swept row ---------------------------------------


def test_a_loose_start_is_bounded_by_the_unprotected_ceiling() -> None:
    """The bug this guards: rows swept at a loose `threshold_start` were read
    against the protected ceiling, which assumes the strict window never caches.
    They reported counts that looked impossible and were simply the wrong
    comparison -- 85 of them on the first Wan collect."""
    assert structural_ceiling(0.2, 10) == 36          # strict window protected
    assert unprotected_ceiling(10) == 42              # only the warmup is
    # the largest count actually observed at start=0.4 on that lane
    assert 41 > structural_ceiling(0.2, 10)
    assert 41 <= unprotected_ceiling(10)


def test_the_unprotected_ceiling_ignores_switch_ratio() -> None:
    for max_skip in (3, 10, 20, 39, 43):
        assert unprotected_ceiling(max_skip) == structural_ceiling(0.0, max_skip)
        assert unprotected_ceiling(max_skip) >= structural_ceiling(0.2, max_skip)


def test_the_two_ceilings_agree_when_the_strict_window_is_only_the_warmup() -> None:
    assert strict_steps(0.06) == FIRST_ENHANCE
    for max_skip in (3, 10, 43):
        assert structural_ceiling(0.06, max_skip) == unprotected_ceiling(max_skip)


def test_the_warmup_is_a_lane_property_not_a_constant() -> None:
    """The image lane runs first_enhance=1, Wan freezes 3. Reading image rows
    against Wan's warmup flagged 13 rows that were inside their real bound."""
    from analysis.sencache_frontier import LANE_FIRST_ENHANCE
    assert LANE_FIRST_ENHANCE == {"image": 1, "wan21": 3}
    for max_skip, observed in ((10, 43), (39, 46), (43, 46)):
        assert observed > unprotected_ceiling(max_skip, first_enhance=3)
        assert observed <= unprotected_ceiling(max_skip, first_enhance=1)


def test_the_knob_choice_is_unaffected_by_the_warmup() -> None:
    """Every budget's strict window exceeds either warmup, so the protected
    ceilings -- the ones ceiling_knobs selects on -- are identical."""
    for switch_ratio in (0.2, 0.12):
        for max_skip in (10, 39, 43):
            assert (structural_ceiling(switch_ratio, max_skip, first_enhance=1)
                    == structural_ceiling(switch_ratio, max_skip, first_enhance=3))
