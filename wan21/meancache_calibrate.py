#!/usr/bin/env python3
"""Collect MeanCache edge costs from full Wan2.1 trajectories.

MeanCache picks its anchor steps by a shortest path over a multigraph whose edge
``(source, destination)`` cost is the error of replacing the velocity across that
interval with the average-velocity extrapolation.  This runner scores every such
edge on real trajectories; ``analysis/build_meancache_schedule.py --model wan21``
turns the summed costs into one exact-NFE schedule per cache tier, together with
the per-edge ``jvp_span`` that is part of the solution.

Three Wan-specific points (plan section 2.7):

* **The solver is UniPC, not Euler.**  The Hunyuan collector reconstructs its
  terminal latent ``z_N`` with one Euler step because its scheduler's update is
  exactly that; Wan's ``FlowUniPCMultistepScheduler`` is multistep and no
  hand-written mirror of it would be right.  So nothing is reconstructed here:
  ``FlowUniPCMultistepScheduler.step`` is wrapped and every ``prev_sample`` it
  produces is recorded, which makes ``z_1 .. z_50`` the solver's own states and
  ``z_0`` the first latent the transformer received.  The patch goes on the
  *class* because ``wan21/runner.py:380-385`` builds a fresh scheduler inside
  every generation.
* **The sigma guard.**  ``hunyuan_video/methods/meancache.py:71`` requires
  exactly ``num_steps + 1`` inference sigmas and refuses anything else, and the
  plan lists the Wan exposure of that vector as 待验证 -- to be verified, and
  asserted at P0.  ``require_inference_sigmas`` below is that assertion, run
  against the live scheduler the generation left on ``pipe.model``, and the
  observed length and endpoints are written into the shard's sidecar JSON as the
  archived evidence.  (``set_timesteps`` concatenates ``sigma_last`` at
  ``.../wan/utils/fm_solvers_unipc.py:205-209``, so 51 is expected; the guard is
  what makes it a fact rather than an expectation.)
* **Two CFG branches, one cost array.**  Wan produces a cond and an uncond
  velocity per solver step and the deployed method keeps a velocity history per
  branch, so both branches are scored and both are summed into the same
  ``cost_sums`` / ``cost_counts``.  The latent trajectory is shared -- the two
  forwards receive the same ``z_k`` -- which is exactly what the deployed
  predictor sees: ``MeanCacheMethod._predict`` builds its JVP proxy from the
  shared latent and the branch's own velocity history
  (``hunyuan_video/methods/meancache.py:110-115``).  Scoring the same mixture
  keeps the offline cost model on the quantity that actually deploys.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from wan21.runner import generate_t2v  # noqa: E402


#: Plan section 1.1 names the Wan2.1 matrix generation protocol; the constants
#: themselves are `wan21/backend.py::WanProtocol` and are validated there.
PROTOCOL_ID = "WAN-CachePaper-480"

#: Plan section 3.2b: the three fitted assets share one seed base.
CALIBRATION_SEED_BASE = 20260723

#: `analysis/build_meancache_schedule.py` cannot tell backbones apart from the
#: arrays alone -- they are shape-identical -- so this tag is the only marker.
MODEL_TAG = "wan21"

BRANCHES = ("cond", "uncond")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    parser.add_argument("--max_edge_gap", type=int, default=15)
    parser.add_argument("--jvp_spans", default="2,3,4,5")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def require_inference_sigmas(scheduler: Any, *, num_steps: int) -> torch.Tensor:
    """The plan's section 2.7 sigma guard, asserted on the live scheduler.

    `MeanCacheMethod._scheduler_sigmas` refuses anything but exactly
    `num_steps + 1` inference sigmas, because a scheduler still carrying its
    construction-time grid would hand out sigma_t = 1.0 and a solver step an
    order of magnitude too small.  The calibration must be scored on the same
    vector the deployed method will read, so the same length condition is
    checked here rather than assumed.
    """

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) != int(num_steps) + 1:
        raise RuntimeError(
            f"MeanCache calibration requires the {int(num_steps)}-step inference "
            f"sigmas, got {'none' if sigmas is None else len(sigmas)}"
        )
    return torch.as_tensor(sigmas, dtype=torch.float32)


class UniPCTrajectoryCapture:
    """Record every latent the UniPC solver produces.

    Patching `FlowUniPCMultistepScheduler.step` on the class is required, not a
    convenience: `wan21/runner.py:380-385` constructs a new scheduler per
    generation, so anything installed on an instance beforehand is thrown away
    before the first solver step runs.
    """

    def __init__(self, scheduler_class: Any) -> None:
        self.scheduler_class = scheduler_class
        self.latents: list[torch.Tensor] = []
        self._original: Any = None

    def __enter__(self) -> "UniPCTrajectoryCapture":
        original = self.scheduler_class.step
        if getattr(original, "_unipc_trajectory_capture", False):
            raise RuntimeError("UniPC trajectory capture is already installed")

        @functools.wraps(original)
        def step(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
            output = original(scheduler, *args, **kwargs)
            sample = output[0] if isinstance(output, tuple) else output.prev_sample
            self.latents.append(sample.detach().squeeze(0))
            return output

        step._unipc_trajectory_capture = True
        self.scheduler_class.step = step
        self._original = original
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.scheduler_class.step = self._original
        self._original = None

    def clear(self) -> None:
        self.latents.clear()


def _velocity(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        return output[0]
    if isinstance(output, dict):
        return output["x"]
    return output


def _trajectory_costs(
    latents: list[torch.Tensor],
    velocities: list[torch.Tensor],
    sigmas: torch.Tensor,
    spans: tuple[int, ...],
    *,
    max_edge_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute error of the MeanCache payload on every (source, destination) edge.

    Mirrors `hunyuan_video/meancache_calibrate.py:50-94`, itself a mirror of
    `flux/meancache_calibrate.py:50-89`. The cost is the mean absolute error, not
    an L2 norm, and the edge is only scored when the sliding lookback
    `source - span` exists on the trajectory.
    """

    steps = len(velocities)
    sums = np.zeros((len(spans), steps + 1, steps + 1), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for source in range(steps):
        sums[:, source, source + 1] = 0.0
        counts[:, source, source + 1] = 1
        for span_index, span in enumerate(spans):
            reference = source - span
            if reference < 0:
                continue
            sigma_r = sigmas[reference].to(torch.float32)
            sigma_t = sigmas[source].to(torch.float32)
            rt = sigma_t - sigma_r
            if float(rt.abs().item()) < 1e-12:
                continue
            z_r = latents[reference].to(torch.float32)
            z_t = latents[source].to(torch.float32)
            v_r = velocities[reference].to(torch.float32)
            jvp = (z_t - z_r - rt * v_r) / (rt * rt)
            for destination in range(
                source + 1,
                min(steps, source + int(max_edge_gap)) + 1,
            ):
                ts = sigmas[destination].to(torch.float32) - sigma_t
                true_average = (latents[destination].to(torch.float32) - z_t) / ts
                predicted_average = velocities[source].to(torch.float32) + ts * jvp
                cost = (true_average - predicted_average).abs().mean().item()
                sums[span_index, source, destination] += float(cost)
                counts[span_index, source, destination] += 1
    return sums, counts


def main() -> int:
    args = parse_args()
    if args.protocol_id != PROTOCOL_ID:
        raise SystemExit(
            f"Wan2.1 MeanCache calibration runs protocol {PROTOCOL_ID}, "
            f"not {args.protocol_id}"
        )
    if args.ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required or WAN21_CKPT_DIR must be set")
    if args.resume and args.out.is_file():
        print(f"[wan21-meancache-calibration] preserve {args.out}", flush=True)
        return 0
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Wan2.1 MeanCache calibration must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1 MeanCache calibration requires CUDA")
    spans = tuple(int(value) for value in args.jvp_spans.split(",") if value)
    if not spans or min(spans) < 2:
        raise SystemExit("MeanCache calibration expects JVP spans >= 2")

    protocol = WanProtocol()
    prompts = read_prompts(args.prompt_file, limit=(args.limit or None))
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    if not selected:
        return 0

    wan, wan_configs, size_configs, _attention = import_wan(args.wan_repo)
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

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

    forward_index = 0
    first_latent: list[torch.Tensor] = []
    branch_velocities: dict[str, list[torch.Tensor]] = {name: [] for name in BRANCHES}

    def collect(
        _module: Any,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        nonlocal forward_index
        branch = BRANCHES[forward_index % 2]
        if forward_index == 0:
            x = inputs[0] if inputs else kwargs["x"]
            latent = x[0] if isinstance(x, (list, tuple)) else x
            first_latent.append(latent.detach())
        branch_velocities[branch].append(_velocity(output).detach())
        forward_index += 1

    handle = pipe.model.register_forward_hook(collect, with_kwargs=True)
    total_sums = np.zeros((len(spans), protocol.steps + 1, protocol.steps + 1))
    total_counts = np.zeros_like(total_sums, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    sigma_evidence: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        with UniPCTrajectoryCapture(FlowUniPCMultistepScheduler) as capture:
            for local_index, prompt in enumerate(selected):
                global_index = start + local_index
                seed = seed_for(args.seed, global_index)
                forward_index = 0
                first_latent.clear()
                for name in BRANCHES:
                    branch_velocities[name].clear()
                capture.clear()

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

                if forward_index != 2 * protocol.steps:
                    raise RuntimeError(
                        f"MeanCache calibration saw {forward_index} model forwards, "
                        f"expected {2 * protocol.steps} (cond + uncond per step)"
                    )
                for name in BRANCHES:
                    if len(branch_velocities[name]) != protocol.steps:
                        raise RuntimeError(
                            f"MeanCache calibration collected "
                            f"{len(branch_velocities[name])} {name} velocities, "
                            f"expected {protocol.steps}"
                        )
                if len(capture.latents) != protocol.steps or not first_latent:
                    raise RuntimeError(
                        f"UniPC produced {len(capture.latents)} solver states, "
                        f"expected {protocol.steps}"
                    )
                sigmas = require_inference_sigmas(
                    pipe.model.scheduler, num_steps=protocol.steps
                ).to(first_latent[0].device)
                if not sigma_evidence:
                    sigma_evidence = {
                        "scheduler_class": type(pipe.model.scheduler).__name__,
                        "sigma_count": int(len(sigmas)),
                        "expected_sigma_count": int(protocol.steps) + 1,
                        "sigma_first": float(sigmas[0].item()),
                        "sigma_last": float(sigmas[-1].item()),
                        "terminal_latent_source": "unipc_step_prev_sample",
                    }
                # z_0 is the latent the first transformer call received; z_1..z_50
                # are the solver's own outputs, so the trajectory is UniPC's, not
                # a reconstruction of it.
                trajectory = [first_latent[0], *capture.latents]

                for name in BRANCHES:
                    sums, counts = _trajectory_costs(
                        trajectory,
                        branch_velocities[name],
                        sigmas,
                        spans,
                        max_edge_gap=args.max_edge_gap,
                    )
                    total_sums += sums
                    total_counts += counts
                rows.append(
                    {
                        "prompt_idx": global_index,
                        "seed": seed,
                        "branches_scored": list(BRANCHES),
                    }
                )
                del trajectory
                capture.clear()
                first_latent.clear()
                for name in BRANCHES:
                    branch_velocities[name].clear()
                torch.cuda.empty_cache()
                print(
                    f"[wan21-meancache-calibration] prompt={global_index}",
                    flush=True,
                )
    finally:
        handle.remove()
    verify_untouched_transformer(pipe.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cost_sums=total_sums,
        cost_counts=total_counts,
        jvp_spans=np.asarray(spans, dtype=np.int64),
        num_steps=np.asarray(protocol.steps, dtype=np.int64),
        max_edge_gap=np.asarray(args.max_edge_gap, dtype=np.int64),
        # FLUX, Qwen, HunyuanVideo and Wan cost arrays are shape-identical, so the
        # schedule builder cannot tell them apart from the arrays alone; the tag
        # is what makes its `--model` more than free text.
        model=np.asarray(MODEL_TAG),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "wan21-meancache-cost-shard-v1",
                "model": MODEL_TAG,
                "protocol_id": PROTOCOL_ID,
                "task": protocol.task,
                "size": protocol.size,
                "frames": protocol.frames,
                "num_steps": protocol.steps,
                "sample_solver": protocol.sample_solver,
                "sample_shift": protocol.sample_shift,
                "guidance_scale": protocol.guidance_scale,
                "prompt_file": str(args.prompt_file),
                "base_seed": int(args.seed),
                "seed_rule": "base_plus_prompt_idx",
                "jvp_spans": list(spans),
                "max_edge_gap": int(args.max_edge_gap),
                "branch_policy": "cond_decides",
                "branch_cost_aggregation": "cond_and_uncond_into_one_array",
                "injection_point": "wan_model_head",
                # Plan section 2.7 lists the UniPC sigma exposure as a P0 item to
                # verify; this is the measurement, not a restatement of it.
                "sigma_guard": sigma_evidence,
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "prompts": rows,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[wan21-meancache-calibration] wrote {args.out} prompts={len(rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
