"""Learnable Linear Predictor helpers for FLUX cache experiments.

L2P uses a timestep-indexed scalar weight matrix to reconstruct a feature at
step ``t`` from features stored at previous full-refresh steps:

    F_hat[t] = sum_j W[t, j] * F[j]

This module intentionally keeps the math backbone-independent.  The new
cross-model baseline uses it for the final projected transformer output; the
older final-hidden and fine-slot experiments remain separate callers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


History = Dict[int, torch.Tensor]


def interval_full_steps(*, num_steps: int, interval: int, first_enhance: int) -> list[int]:
    """Return the full-refresh steps produced by ``IntervalGate`` semantics."""
    last_activated = -(10**9)
    full_steps: list[int] = []
    for step in range(int(num_steps)):
        if step < int(first_enhance):
            should_skip = False
        elif step >= int(num_steps) - 1:
            should_skip = False
        elif (step - last_activated) >= int(interval):
            should_skip = False
        else:
            should_skip = True
        if not should_skip:
            last_activated = step
            full_steps.append(step)
    return full_steps


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a checkpoint or manifest file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_l2p_weight_file(path: str | Path, *, num_steps: Optional[int] = None) -> dict[str, Any]:
    """Load an L2P weight file.

    Accepted formats:
      - a raw tensor with shape ``[T, T]``;
      - a dict containing ``weights`` or ``W``.

    Metadata is preserved in the returned dict. We keep weights on CPU in
    float32; prediction moves only the current row scalars to Python floats.
    """
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, torch.Tensor):
        weights = raw
        meta: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        if "weights" in raw:
            weights = raw["weights"]
        elif "W" in raw:
            weights = raw["W"]
        else:
            raise ValueError(f"L2P weight file {path} missing `weights` or `W`")
        meta = dict(raw)
    else:
        raise TypeError(f"unsupported L2P weight file type: {type(raw).__name__}")

    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights)
    weights = weights.detach().to(dtype=torch.float32, device="cpu")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"L2P weights must be square [T,T], got shape {tuple(weights.shape)}")
    if num_steps is not None and int(num_steps) > int(weights.shape[0]):
        raise ValueError(
            f"L2P weights have only {weights.shape[0]} steps but run requests {num_steps}"
        )
    meta["weights"] = weights
    meta["path"] = str(path)
    meta["sha256"] = file_sha256(path)
    meta.setdefault("num_steps", int(weights.shape[0]))
    meta.setdefault("format", "l2p-raw-tensor" if isinstance(raw, torch.Tensor) else "l2p-v1")
    return meta


def latest_history_step(history: Mapping[int, torch.Tensor]) -> Optional[int]:
    if not history:
        return None
    return max(int(k) for k in history.keys())


def append_history(history: Optional[History], step: int, feature: torch.Tensor) -> History:
    """Return a new history with ``feature`` stored for an absolute step."""
    out: History = {} if history is None else dict(history)
    out[int(step)] = feature.detach()
    return out


def _iter_weighted_history(
    history: Mapping[int, torch.Tensor],
    weights_row: torch.Tensor,
    *,
    current_step: int,
    min_abs_weight: float,
) -> Iterable[tuple[int, float, torch.Tensor]]:
    for step in sorted(int(k) for k in history.keys()):
        if step >= int(current_step) or step >= int(weights_row.shape[0]):
            continue
        coef = float(weights_row[step].item())
        if abs(coef) <= float(min_abs_weight):
            continue
        yield step, coef, history[step]


def predict_l2p(
    history: Mapping[int, torch.Tensor],
    weights: torch.Tensor,
    *,
    current_step: int,
    min_abs_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predict a feature tensor with the L2P row for ``current_step``.

    Returns ``(prediction, fields)`` where fields are cheap scalar diagnostics
    suitable for per-step decision logs.
    """
    if int(current_step) >= int(weights.shape[0]):
        raise ValueError(f"current_step={current_step} outside L2P weight shape {tuple(weights.shape)}")
    if not history:
        raise ValueError("empty L2P history")

    row = weights[int(current_step)]
    pred: Optional[torch.Tensor] = None
    used_steps: list[int] = []
    used_coeffs: list[float] = []
    ref_dtype: Optional[torch.dtype] = None
    ref_device: Optional[torch.device] = None
    for step, coef, value in _iter_weighted_history(
        history, row, current_step=int(current_step), min_abs_weight=float(min_abs_weight)
    ):
        if pred is None:
            ref_dtype = value.dtype
            ref_device = value.device
            pred = value.detach().to(torch.float32).mul(coef)
        else:
            pred.add_(value.detach().to(torch.float32), alpha=coef)
        used_steps.append(int(step))
        used_coeffs.append(float(coef))

    fallback = False
    if pred is None:
        last_step = latest_history_step(history)
        if last_step is None:
            raise ValueError("empty L2P history")
        value = history[int(last_step)]
        pred = value.detach().clone()
        ref_dtype = value.dtype
        ref_device = value.device
        used_steps = [int(last_step)]
        used_coeffs = [1.0]
        fallback = True

    assert ref_dtype is not None and ref_device is not None
    pred = pred.to(dtype=ref_dtype, device=ref_device)
    coeff_tensor = torch.tensor(used_coeffs, dtype=torch.float32)
    fields = {
        "l2p_current_step": int(current_step),
        "l2p_history_steps": used_steps,
        "l2p_weights_used": int(len(used_steps)),
        "l2p_weight_l1": float(coeff_tensor.abs().sum().item()) if used_coeffs else 0.0,
        "l2p_weight_l2": float(coeff_tensor.norm().item()) if used_coeffs else 0.0,
        "l2p_weight_sum": float(coeff_tensor.sum().item()) if used_coeffs else 0.0,
        "l2p_fallback_latest": bool(fallback),
    }
    return pred, fields


def accumulate_l2p_gram(
    gram: torch.Tensor,
    history: Mapping[int, torch.Tensor],
    *,
    num_steps: int,
) -> None:
    """Accumulate one full trajectory into a timestep Gram matrix."""
    expected = list(range(int(num_steps)))
    observed = sorted(int(step) for step in history)
    if observed != expected:
        raise ValueError(
            f"L2P history steps mismatch: expected 0..{num_steps - 1}, got {observed}"
        )
    features = torch.stack(
        [
            history[step].detach().to(torch.float32).reshape(-1)
            for step in expected
        ],
        dim=0,
    )
    contribution = features @ features.t()
    gram.add_(contribution.detach().to(device=gram.device, dtype=gram.dtype))


def solve_l2p_weights(gram: torch.Tensor, *, ridge: float = 1e-5) -> torch.Tensor:
    """Solve the causal linear least-squares predictor from a Gram matrix."""
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"L2P Gram matrix must be square, got {tuple(gram.shape)}")
    if float(ridge) < 0.0:
        raise ValueError("L2P ridge must be non-negative")
    num_steps = int(gram.shape[0])
    source = gram.detach().to(dtype=torch.float64, device="cpu")
    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for step in range(1, num_steps):
        weights[step, step - 1] = 1.0
        lhs = source[:step, :step].clone()
        rhs = source[:step, step].clone()
        scale = float(lhs.diag().mean().item())
        lhs.add_(
            torch.eye(step, dtype=torch.float64),
            alpha=float(ridge) * max(scale, 1.0),
        )
        try:
            coefficients = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            coefficients = torch.linalg.pinv(lhs) @ rhs
        if torch.isfinite(coefficients).all():
            weights[step, :step] = coefficients.to(torch.float32)
    return weights


def l2p_teacher_forced_errors(
    gram: torch.Tensor,
    weights: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Compute per-step fit errors directly from trajectory Gram statistics."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"L2P Gram matrix must be square, got {tuple(gram.shape)}")
    if weights.shape != gram.shape:
        raise ValueError(
            f"L2P weights/Gram shape mismatch: {tuple(weights.shape)} vs {tuple(gram.shape)}"
        )
    source = gram.detach().to(dtype=torch.float64, device="cpu")
    fitted = weights.detach().to(dtype=torch.float64, device="cpu")
    rows: list[dict[str, float | int]] = []
    for step in range(1, int(source.shape[0])):
        coefficients = fitted[step, :step]
        target_energy = float(source[step, step].item())
        cross = source[:step, step]
        history = source[:step, :step]
        squared_error = float(
            (
                source[step, step]
                - 2.0 * coefficients.dot(cross)
                + coefficients.dot(history @ coefficients)
            ).item()
        )
        squared_error = max(squared_error, 0.0)
        relative_mse = squared_error / max(
            target_energy,
            torch.finfo(torch.float64).eps,
        )
        rows.append(
            {
                "step": step,
                "squared_error": squared_error,
                "relative_mse": relative_mse,
                "relative_l2": relative_mse**0.5,
            }
        )
    return rows
