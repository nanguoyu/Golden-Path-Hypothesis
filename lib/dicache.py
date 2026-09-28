"""Small tensor primitives shared by the DiCache backbone adapters."""

from __future__ import annotations

import torch


def relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
    return float((numerator / denominator).item())


def aligned_residual(
    current_probe_residual: torch.Tensor,
    residual_history: list[torch.Tensor],
    probe_history: list[torch.Tensor],
) -> tuple[torch.Tensor, float | None]:
    """Return DiCache's two-anchor trajectory-aligned residual."""

    if len(residual_history) < 2 or len(probe_history) < 2:
        if not residual_history:
            raise RuntimeError("DiCache cache action has no residual history")
        return residual_history[-1], None

    probe_delta = probe_history[-1] - probe_history[-2]
    denominator = probe_delta.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
    gamma = (
        (current_probe_residual - probe_history[-2]).abs().mean() / denominator
    ).clamp(1.0, 1.5)
    payload = residual_history[-2] + gamma * (
        residual_history[-1] - residual_history[-2]
    )
    return payload, float(gamma.item())


def append_anchor(history: list[torch.Tensor], value: torch.Tensor) -> None:
    history.append(value.detach())
    del history[:-2]
