"""Tiny helpers shared by `flux/seacache.py`, `flux/hicache.py`, and `flux/runner.py`.

Anything backbone-specific that doesn't depend on the chosen cache method.
"""

from __future__ import annotations

from typing import Tuple


def enforce_image_size(width: int, height: int, multiple: int = 16) -> Tuple[int, int]:
    """FLUX requires H, W divisible by `multiple` (16 by default for FLUX.1-dev)."""
    return (int(width) // multiple) * multiple, (int(height) // multiple) * multiple


def get_transformer(pipe):
    """Return the diffusers FluxTransformer2DModel underneath `pipe`."""
    return pipe.transformer


def get_scheduler(pipe):
    """Return the scheduler that decides timesteps (FlowMatchEulerDiscreteScheduler
    for FLUX.1-dev/-schnell)."""
    return pipe.scheduler


def default_guidance_scale(model_name: str, requested: float) -> float:
    """flux-schnell ignores CFG (distilled); flux-dev uses the requested value."""
    return 0.0 if model_name == "flux-schnell" else float(requested)


def default_max_sequence_length(model_name: str) -> int:
    """T5 token cap: 256 for schnell, 512 for dev."""
    return 256 if model_name == "flux-schnell" else 512
