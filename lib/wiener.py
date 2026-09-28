"""N-D separable Wiener filter (SeaCache).

Adapted (no algorithmic change) from the SeaCache reference implementation at
`reference/seacache/code/util_seacache.py`. The filter gain on each axis is

    H1(f) = (a * Sx0) / (a^2 * Sx0 + b^2 + eps),   Sx0 = power_const / (|f|^p + eps)

with separable axes combined by product, then optionally normalized so the full
filter has unit mean (or unit peak). `apply_sea_with_scheduler` is the typical
entry point: it pulls `(a_t, b_t)` from a diffusers scheduler.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def _rfft_full_mean_weights_1d(n_last: int, device, dtype) -> torch.Tensor:
    """1D weights so that a weighted sum over an rFFT half-spectrum recovers the
    full-spectrum mean. Even N: [1, 2, ..., 2, 1] (DC and Nyquist counted once).
    Odd N:  [1, 2, ..., 2] (DC counted once, no Nyquist).
    """
    Lh = n_last // 2 + 1
    w = torch.ones(Lh, device=device, dtype=dtype)
    if n_last % 2 == 0:
        if Lh > 2:
            w[1:-1] *= 2.0
    else:
        if Lh > 1:
            w[1:] *= 2.0
    return w


def apply_sea_from_ab(
    x: torch.Tensor,
    a: float,
    b: float,
    power_exp: float = 2.0,
    power_const: float = 1.0,
    dims: Optional[Sequence[int]] = None,
    eps: float = 1e-16,
    norm_mode: str = "mean",
    *,
    real: bool = False,
) -> torch.Tensor:
    """Apply an N-D separable Wiener filter using mixing coefficients (a, b).

    Args:
        x:            input tensor.
        a, b:         scheduler-derived mixing scalars; see `ab_from_scheduler`.
        power_exp:    exponent p in Sx0 = power_const / (|f|^p + eps).
        power_const:  numerator constant in Sx0.
        dims:         axes to filter over. Default: last two for >=3-D inputs.
        eps:          numerical guard.
        norm_mode:    "mean" or "peak"; how the assembled filter is normalized.
        real:         if True, use rfftn/irfftn for real input speed.
    """
    orig_dtype = x.dtype
    x32 = x.contiguous().to(torch.float32)

    if dims is None:
        dims = tuple(range(x32.ndim)) if x32.ndim <= 2 else tuple(range(-2, -x32.ndim, -1))

    X = torch.fft.rfftn(x32, dim=dims) if real else torch.fft.fftn(x32, dim=dims)

    H: Optional[torch.Tensor] = None
    for i, ax in enumerate(dims):
        N = x32.shape[ax]
        if real and (i == len(dims) - 1):
            f = torch.fft.rfftfreq(N, device=x32.device, dtype=torch.float32)
        else:
            f = torch.fft.fftfreq(N, device=x32.device, dtype=torch.float32)
        rad = torch.abs(f)
        Sx0 = power_const / ((rad**power_exp) + eps)
        H1 = (a * Sx0) / (a * a * Sx0 + (b * b) + eps)
        shape_i = [1] * x32.ndim
        shape_i[ax] = H1.shape[0]
        H1 = H1.reshape(shape_i)
        H = H1 if H is None else (H * H1)

    nm = norm_mode.lower()
    if nm == "peak":
        maxv = torch.amax(H)
        if torch.isfinite(maxv) and maxv > 0:
            H = H / maxv
    elif nm == "mean":
        if real:
            N_last = int(x32.shape[dims[-1]])
            w_last = _rfft_full_mean_weights_1d(N_last, device=x32.device, dtype=torch.float32)
            wshape = [1] * x32.ndim
            wshape[dims[-1]] = w_last.numel()
            W = w_last.view(*wshape)
            denom = torch.sum(W) * float(
                torch.prod(torch.tensor([x32.shape[d] for d in dims[:-1]]))
            )
            meanv = torch.sum(H * W) / denom
        else:
            meanv = torch.mean(H)
        if torch.isfinite(meanv) and meanv > 0:
            H = H / meanv

    Y = X * H
    if real:
        s = [x32.shape[d] for d in dims]
        y = torch.fft.irfftn(Y, s=s, dim=dims)
    else:
        y = torch.fft.ifftn(Y, dim=dims).real
    return y.to(orig_dtype)


def ab_from_scheduler(scheduler, idx: int, mode: str = "flow") -> Tuple[float, float]:
    """Extract (a_t, b_t) from a diffusers scheduler at discrete index `idx`.

    `mode="flow"` (rectified-flow, e.g. FlowMatchEulerDiscreteScheduler) uses
    a = 1 - sigma, b = sigma. `mode="vp"` (VP/DDPM-style) uses alphas_cumprod.
    Falls back to sigma-derived estimates if neither attribute is present.
    """
    def _clamp01(v: float) -> float:
        return max(1e-6, min(1.0 - 1e-6, float(v)))

    if isinstance(idx, torch.Tensor):
        idx = int(idx.detach().cpu().item())

    if mode == "flow":
        sigma = (
            float(scheduler.sigmas[idx])
            if hasattr(scheduler, "sigmas")
            else 1.0 - (idx + 1) / float(getattr(scheduler, "num_inference_steps", idx + 1))
        )
        sigma = _clamp01(sigma)
        return 1.0 - sigma, sigma

    ac = getattr(scheduler, "alphas_cumprod", None)
    if isinstance(ac, torch.Tensor) and ac.numel() > 0:
        a2 = _clamp01(ac[max(0, min(idx, ac.numel() - 1))])
        return math.sqrt(a2), math.sqrt(max(1e-16, 1.0 - a2))

    if hasattr(scheduler, "sigmas"):
        sigma = float(scheduler.sigmas[idx])
        a2 = _clamp01(1.0 / (1.0 + sigma * sigma))
        return math.sqrt(a2), math.sqrt(max(1e-16, 1.0 - a2))

    n = getattr(scheduler, "num_inference_steps", None) or (idx + 1)
    a2 = _clamp01((idx + 1) / float(n))
    return math.sqrt(a2), math.sqrt(max(1e-12, 1.0 - a2))


def apply_sea_with_scheduler(
    x: torch.Tensor,
    scheduler,
    idx: int,
    power_exp: float = 2.0,
    dims: Optional[Sequence[int]] = None,
    mode: str = "flow",
    norm_mode: str = "mean",
    *,
    real: bool = False,
) -> torch.Tensor:
    """Convenience: pull (a_t, b_t) from `scheduler` and filter `x`."""
    a, b = ab_from_scheduler(scheduler, idx, mode=mode)
    return apply_sea_from_ab(
        x, a, b, power_exp=power_exp, dims=dims, norm_mode=norm_mode, real=real
    )
