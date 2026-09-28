"""Retained latent trajectories for a replayed SPX cell.

`docs/image_cached_trajectory_plan_zh.md` section 7.1. An SPX cell stored only
its images and `decisions_*.json`; the 51 states the sampler walked through were
never written, and cannot be recovered afterwards. The cache-bend layer needs
them, so the selected (cell, prompt) pairs are **replayed** with retention on:
the same weights, the same seed, the same schedule and payload, producing the
same image plus the path it took to get there.

Nothing here touches the transformer hot path. `prepare_latents` runs once per
generation, before the denoise loop, and `callback_on_step_end` is the
pipeline's own per-step hook, which reads `latents` and returns them unchanged.

The two runners that use this (`flux/sp_cross_runner.py`,
`qwen_image/sp_cross_runner.py`) keep their default behaviour byte for byte
when `--retain_trajectory` is absent: the capture wrapper is not installed and
no callback is passed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

N_EXTRA_STATES = 1  # z_T, in front of the num_steps post-step states


def trajectory_filename(global_idx: int) -> str:
    """Canonical per-generation path file, next to `img_<idx>.png`."""
    return f"latents_{int(global_idx):05d}.pt"


def install_z_t_capture(pipe: Any) -> dict[str, Any]:
    """Wrap `pipe.prepare_latents` so each generation's z_T is kept.

    Same construction as `flux/full_trajectory_probe.py::_install_z_t_capture`:
    `prepare_latents` is called once per `pipe(...)`, outside the denoise loop,
    so this is not a per-step hook. The wrapper only clones; it returns the
    pipeline's own object unchanged.
    """
    holder: dict[str, Any] = {}
    original = pipe.prepare_latents

    def wrapped(*args: Any, **kwargs: Any):
        out = original(*args, **kwargs)
        latents = out[0] if isinstance(out, tuple) else out
        holder["z_T"] = latents.detach().clone()
        return out

    pipe.prepare_latents = wrapped
    holder["_restore"] = lambda: setattr(pipe, "prepare_latents", original)
    return holder


def make_step_collector() -> tuple[list, Callable[..., dict]]:
    """`(trace, callback)` for `callback_on_step_end`.

    The callback appends the post-step packed latent as float32 on CPU and
    hands `callback_kwargs` back untouched, so the sampler sees exactly what it
    would have seen without it.
    """
    import torch

    trace: list = []

    def _cb(_pipe: Any, _step_index: int, _timestep: Any, callback_kwargs: dict) -> dict:
        trace.append(callback_kwargs["latents"].detach().to("cpu", torch.float32))
        return callback_kwargs

    return trace, _cb


def stack_trajectory(z_T: Any, trace: list, *, num_steps: int) -> Any:
    """`[num_steps + 1, d]` float32, rows flattened, z_T first."""
    import torch

    if z_T is None:
        raise RuntimeError("prepare_latents was never called: no z_T to retain")
    if len(trace) != int(num_steps):
        raise RuntimeError(
            f"captured {len(trace)} post-step states, expected {int(num_steps)}"
        )
    rows = [z_T.to("cpu", torch.float32)] + list(trace)
    return torch.stack([row.reshape(-1) for row in rows])


def scheduler_sigmas(pipe: Any, *, num_steps: int) -> list[float]:
    """The sigma grid the run walked, `num_steps + 1` long."""
    sigmas = [float(v) for v in pipe.scheduler.sigmas.detach().cpu().tolist()]
    if len(sigmas) != int(num_steps) + 1:
        raise RuntimeError(
            f"scheduler has {len(sigmas)} sigmas, expected {int(num_steps) + 1}"
        )
    return sigmas


def save_trajectory(
    path: Path,
    *,
    Z: Any,
    latent_shape: list[int],
    sigmas: list[float],
    prompt_idx: int,
    seed: int,
    num_steps: int,
    store_dtype: str = "bf16",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write one retained path and return its manifest fields.

    `store_dtype` is `bf16` for the production wave and `fp32` for the small
    dual-store subset that measures what the store does to a direction
    (plan section 7.3, last bullet).
    """
    import torch

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[str(store_dtype)]
    z_T = Z[0].to(torch.float32).numpy()
    payload: dict[str, Any] = {
        "schema": "image_retained_trajectory.v1",
        "path": Z.to(dtype),
        "path_dtype": str(store_dtype),
        "sigmas": list(sigmas),
        "dims": [int(Z.shape[0]), int(Z.shape[1])],
        "latent_shape": [int(v) for v in latent_shape],
        "num_steps": int(num_steps),
        "prompt_idx": int(prompt_idx),
        "seed": int(seed),
        "z_T_sha256": hashlib.sha256(z_T.tobytes()).hexdigest(),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)
    return {k: v for k, v in payload.items() if k != "path"}


def parse_prompt_indices(spec: str, *, total: int) -> list[int]:
    """`--prompt_indices` -> sorted unique global prompt indices.

    Accepts a comma-separated list of indices and `a-b` inclusive ranges. The
    indices are GLOBAL positions in the prompt file and are never renumbered:
    the seed of a generation is `base_seed + global_idx`, so renumbering would
    change the noise and break the pairing with the stored cell.
    """
    out: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token[1:]:
            lo_s, _, hi_s = token.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"--prompt_indices range {token!r} runs backwards")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    if not out:
        raise ValueError("--prompt_indices selected nothing")
    bad = sorted(i for i in out if i < 0 or i >= int(total))
    if bad:
        raise ValueError(
            f"--prompt_indices out of range for {int(total)} prompts: {bad[:8]}"
        )
    return sorted(out)


def select_pairs(
    prompts: list[str], *, indices: list[int] | None
) -> list[tuple[int, str]]:
    """`[(global_idx, prompt)]`, either the whole list or the selected indices."""
    if indices is None:
        return list(enumerate(prompts))
    return [(int(i), prompts[int(i)]) for i in indices]
