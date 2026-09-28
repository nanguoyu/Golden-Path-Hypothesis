from __future__ import annotations

import numpy as np

from analysis.four_factor_profile_ablation import _tie_expected_recall


def test_tie_expected_recall_handles_boundary_group() -> None:
    # Top 50% selects the score-3 row and one of the two tied score-2 rows.
    # One certain hit plus an expected half hit gives recall 1.5 / 2.
    prediction = np.array([3.0, 2.0, 2.0, 1.0])
    target = np.array([4.0, 3.0, 1.0, 2.0])
    assert _tie_expected_recall(prediction, target, 50.0) == 0.75


def test_tie_expected_recall_matches_exact_selection_without_ties() -> None:
    prediction = np.array([4.0, 3.0, 2.0, 1.0])
    target = np.array([4.0, 3.0, 1.0, 2.0])
    assert _tie_expected_recall(prediction, target, 50.0) == 1.0
