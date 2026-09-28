#!/usr/bin/env python3
"""Collect Wan2.1 SenCache sensitivity calibration rows.

This is compute-heavy and must run on compute nodes.  It estimates the two
timestep-wise directional sensitivity norms the online SenCache gate scores
with, from full Wan2.1 trajectories:

  J_x(t_i) ~= ||f(x_ref, t_i) - f(x_i, t_i)|| / ||x_ref - x_i||
  J_t(t_i) ~= ||f(x_i, t_ref) - f(x_i, t_i)|| / |t_ref - t_i|

``ref`` is the adjacent full-trajectory state/timestep, so every step costs two
extra transformer forwards on top of the trajectory itself (plan section 2.5,
"每步 2 次额外前向").  ``x`` is the latent ``WanModel.forward`` receives before
``patch_embedding`` and ``t`` its native timestep -- the same two quantities
``wan21/adapter.py::SenCacheAdapter`` hands to the gate.

Upstream comparison (plan sections 2.4 / 2.5).  The authority is
``knowledge/SenCache/code/Wan2.1/sensitivity_calculation.py``:

* its ``calculate_solver_step_sensitivity`` is
  ``||f(x_next, t) - f(x, t)|| / ||x_next - x||`` where ``x_next`` is one solver
  step ahead -- the same directional quotient computed here, with the difference
  that upstream re-noises an encoded *video* latent per timestep and takes the
  step from that synthetic state, while this runner reads the states off a real
  generation.  On a real trajectory ``x_{i+1}`` *is* the solver step output, so
  the estimator is the same quantity measured on the distribution the gate
  actually sees;
* its ``calculate_jacobian_norm_T`` is ``||f(x, t_next) - f(x, t)||`` divided by
  ``t_next - t``, with ``t_next`` from ``np.roll(timesteps, -1)`` -- i.e. the
  next scheduler timestep, which is the reference used here as well;
* upstream applies **no scaling** to ``t``: it feeds the forward's own timestep
  to both the finite difference and the nearest-neighbour table lookup.  Wan's
  timesteps come out of ``FlowUniPCMultistepScheduler.set_timesteps`` as
  ``sigmas * num_train_timesteps``, i.e. native range(0, 1000), and every row
  written here carries the measured value in the ``timestep`` column so the unit
  is archived rather than assumed (plan section 2.5, P0 item).

``analysis/merge_sencache_sensitivity.py --backbone wan21 --aggregation q90``
freezes the per-step q90 aggregate of these rows into the npz table the gate
reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io_utils import read_prompts, seed_for, split_shard  # noqa: E402
from wan21.backend import (  # noqa: E402
    WanProtocol,
    import_wan,
    load_wan_pipeline,
    verify_untouched_transformer,
)
from wan21.methods_glue import WAN_LATENT_NUMEL, CondDecidesArbiter  # noqa: E402
from wan21.runner import generate_t2v  # noqa: E402


#: Plan section 1.1 names the Wan2.1 matrix generation protocol; the constants
#: themselves are `wan21/backend.py::WanProtocol` and are validated there.
PROTOCOL_ID = "WAN-CachePaper-480"

#: Plan section 3.2b: the three fitted assets share one seed base.
CALIBRATION_SEED_BASE = 20260723


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=os.environ.get("WAN21_CKPT_DIR"),
        help="Wan2.1-T2V-1.3B checkpoint directory (or set WAN21_CKPT_DIR).",
    )
    parser.add_argument(
        "--wan_repo",
        type=Path,
        default=None,
        help="Directory holding the pinned upstream `wan` package; defaults to "
        "wan21/backend.py::WAN_UPSTREAM_ROOT.",
    )
    parser.add_argument("--protocol_id", default=PROTOCOL_ID)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED_BASE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_commit() -> tuple[str | None, bool | None]:
    """Return optional provenance; experiment execution never requires Git."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(_ROOT), "status", "--short"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _norm(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).norm().item())


def _scalar(t: torch.Tensor) -> float:
    return float(t.detach().to(torch.float32).reshape(-1)[0].item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class WanTrajectoryRecorder:
    """Record the cond-branch transformer inputs, outputs and conditioning.

    Wan runs two model forwards per solver step, cond then uncond
    (`wan21/runner.py:407-408`), and both receive the *same* latent, so the
    trajectory is recorded on the cond branch only -- the branch the plan's
    section 2.1 ruling puts every gate feature on, and the branch the upstream
    Wan SenCache variant observes.
    """

    def __init__(self, model: Any, *, num_steps: int, autocast_dtype: torch.dtype):
        self.model = model
        self.num_steps = int(num_steps)
        self.autocast_dtype = autocast_dtype
        self.inputs: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.conditioning: dict[str, Any] | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.forward_index = 0
        self._is_cond = True
        self._handles: list[Any] = []

    def __enter__(self) -> "WanTrajectoryRecorder":
        self._handles.append(
            self.model.register_forward_pre_hook(self._pre, with_kwargs=True)
        )
        self._handles.append(
            self.model.register_forward_hook(self._post, with_kwargs=True)
        )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _latent(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        x = args[0] if args else kwargs["x"]
        if isinstance(x, (list, tuple)):
            if len(x) != 1:
                raise ValueError(f"Wan2.1 calibration expects batch size 1, got {len(x)}")
            return x[0]
        return x

    @staticmethod
    def _timestep(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        t = kwargs.get("t")
        if t is None and len(args) > 1:
            t = args[1]
        if t is None:
            raise RuntimeError("WanModel.forward was called without a timestep")
        return t

    def _pre(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.forward_index >= 2 * self.num_steps:
            raise RuntimeError(
                f"Wan2.1 model forward {self.forward_index} exceeds the "
                f"{self.num_steps}-step x 2-branch protocol"
            )
        self._is_cond = (self.forward_index % 2) == 0
        if not self._is_cond:
            return
        x = self._latent(args, kwargs)
        t = self._timestep(args, kwargs)
        if self.conditioning is None:
            self.conditioning = {
                key: value for key, value in kwargs.items() if key not in ("x", "t")
            }
            self.device = x.device
            self.dtype = x.dtype
        self.inputs.append(x.detach().to("cpu", torch.float32))
        self.timesteps.append(t.detach().clone())

    def _post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self._is_cond:
            value = output[0] if isinstance(output, (list, tuple)) else output
            self.outputs.append(value.detach().to("cpu", torch.float32))
        self.forward_index += 1
        return output

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Re-run the recorded cond conditioning on a substituted (x, t) pair."""

        if self.conditioning is None or self.device is None or self.dtype is None:
            raise RuntimeError("no recorded Wan2.1 transformer call to replay")
        model_x = [x.to(device=self.device, dtype=self.dtype)]
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            with torch.no_grad():
                output = self.model(model_x, t=t.to(self.device), **self.conditioning)
        value = output[0] if isinstance(output, (list, tuple)) else output
        return value.detach().to("cpu", torch.float32)


def _rows_for_prompt(
    recorder: WanTrajectoryRecorder,
    *,
    protocol: WanProtocol,
    prompt_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    steps = len(recorder.inputs)
    if steps != protocol.steps or len(recorder.outputs) != steps:
        raise RuntimeError(
            f"recorded {steps} cond inputs / {len(recorder.outputs)} cond outputs "
            f"for {protocol.steps} steps"
        )
    rows: list[dict[str, Any]] = []
    for i in range(steps):
        if steps <= 1:
            ref = i
        elif i < steps - 1:
            ref = i + 1
        else:
            ref = i - 1
        x_i = recorder.inputs[i]
        x_ref = recorder.inputs[ref]
        o_i = recorder.outputs[i]
        t_i = recorder.timesteps[i]
        t_ref = recorder.timesteps[ref]
        # sqrt(d) multiplies the SenCache threshold, so d is protocol-defining:
        # 16 x 17 x 60 x 104 = 1,697,280 at 832x480x65 (plan section 2.5).  The
        # sensitivity table and the deployed gate must agree on it, so a wrong
        # shape fails here rather than producing a table nobody can use.
        if int(x_i.numel()) != WAN_LATENT_NUMEL:
            raise RuntimeError(
                f"Wan2.1 latent has {int(x_i.numel())} elements, "
                f"protocol expects {WAN_LATENT_NUMEL}"
            )
        o_x = recorder.forward(x_ref, t_i)
        o_t = recorder.forward(x_i, t_ref)
        dx = _norm(x_ref - x_i)
        dt = abs(_scalar(t_ref) - _scalar(t_i))
        rows.append(
            {
                "prompt_id": int(prompt_idx),
                "seed": int(seed),
                "step_index": int(i),
                "timestep": _scalar(t_i),
                "reference_step_index": int(ref),
                "reference_timestep": _scalar(t_ref),
                "latent_shape": json.dumps(list(x_i.shape)),
                "latent_numel": int(x_i.numel()),
                "delta_latent_norm": float(dx),
                "delta_t_abs": float(dt),
                "J_x_directional": (_norm(o_x - o_i) / dx) if dx > 0.0 else None,
                "J_t_directional": (_norm(o_t - o_i) / dt) if dt > 0.0 else None,
                "protocol_id": PROTOCOL_ID,
                "branch": "cond",
                "branch_policy": CondDecidesArbiter.POLICY,
                # Native Wan units: `set_timesteps` writes
                # `sigmas * num_train_timesteps`, and the upstream Wan SenCache
                # variant applies no conversion (plan section 2.5).
                "timestep_units": "wan_native_0_1000",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if args.protocol_id != PROTOCOL_ID:
        raise SystemExit(
            f"Wan2.1 SenCache calibration runs protocol {PROTOCOL_ID}, not {args.protocol_id}"
        )
    if args.ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required or WAN21_CKPT_DIR must be set")
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Wan2.1 sensitivity calibration must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1 sensitivity calibration requires CUDA")

    protocol = WanProtocol()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_csv = args.output_dir / (
        f"sencache_sensitivity_rows_shard{args.shard_idx}of{args.shard_count}.csv"
    )
    if args.resume and shard_csv.is_file():
        print(f"[wan21-sencache-calib] {shard_csv} exists, skip", flush=True)
        return 0

    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    wan, wan_configs, size_configs, _attention = import_wan(args.wan_repo)
    torch.cuda.set_device(0)
    pipe, _info = load_wan_pipeline(
        wan,
        wan_configs,
        ckpt_dir=args.ckpt_dir,
        protocol=protocol,
        device_id=0,
        t5_cpu=bool(args.t5_cpu),
    )
    size = size_configs[protocol.size]
    autocast_dtype = pipe.param_dtype
    git_sha, git_dirty = _git_commit()

    rows: list[dict[str, Any]] = []
    for local_idx, prompt in enumerate(selected):
        idx = start + local_idx
        seed = seed_for(args.seed, idx)
        started = time.perf_counter()
        recorder = WanTrajectoryRecorder(
            pipe.model, num_steps=protocol.steps, autocast_dtype=autocast_dtype
        )
        with recorder:
            video, _timing = generate_t2v(
                pipe,
                prompt=prompt,
                size=size,
                frame_num=protocol.frames,
                shift=protocol.sample_shift,
                sample_solver=protocol.sample_solver,
                sampling_steps=protocol.steps,
                guide_scale=protocol.guidance_scale,
                seed=seed,
                offload_model=False,
            )
        del video
        rows.extend(
            _rows_for_prompt(recorder, protocol=protocol, prompt_idx=idx, seed=seed)
        )
        del recorder
        torch.cuda.empty_cache()
        print(
            f"[wan21-sencache-calib] shard={args.shard_idx} idx={idx} "
            f"steps={protocol.steps} dt={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    verify_untouched_transformer(pipe.model)
    for row in rows:
        row["git_commit"] = git_sha
        row["git_dirty"] = git_dirty
    _write_csv(shard_csv, rows)
    print(f"[wan21-sencache-calib] wrote {shard_csv} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
