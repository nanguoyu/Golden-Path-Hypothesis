from __future__ import annotations

import torch

from lib.l2p import (
    accumulate_l2p_gram,
    l2p_teacher_forced_errors,
    solve_l2p_weights,
)


def test_l2p_gram_solver_recovers_linear_trajectory() -> None:
    history = {
        0: torch.tensor([1.0, 2.0]),
        1: torch.tensor([2.0, 4.0]),
        2: torch.tensor([4.0, 8.0]),
    }
    gram = torch.zeros((3, 3), dtype=torch.float64)
    accumulate_l2p_gram(gram, history, num_steps=3)
    weights = solve_l2p_weights(gram, ridge=0.0)

    assert weights.shape == (3, 3)
    assert torch.allclose(weights[1, :1], torch.tensor([2.0]))
    prediction = weights[2, :2] @ torch.stack([history[0], history[1]])
    assert torch.allclose(prediction, history[2], atol=1e-5)


def test_teacher_forced_errors_are_zero_for_exact_linear_fit() -> None:
    features = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    gram = features @ features.t()
    weights = torch.zeros((3, 3), dtype=torch.float32)
    weights[1, 0] = 2.0
    weights[2, 0] = 3.0

    errors = l2p_teacher_forced_errors(gram, weights)

    assert errors[0]["relative_mse"] == 0.0
    assert errors[1]["relative_mse"] == 0.0
