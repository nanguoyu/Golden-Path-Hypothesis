#!/usr/bin/env python3
"""Full-trajectory regularity probe on FLUX (docs/research_plan_full_trajectory.md).

One generation = one `original`-mode (no cache) FLUX sample, run with exactly
the protocol of `flux/runner.py --mode original` so each trajectory pairs 1:1
with a matrix baseline generation. The pipeline's own `callback_on_step_end`
collects the packed latent after every solver step, and the initial z_T is
taken from `prepare_latents` (called once per generation, outside the denoise
loop). That gives Z = [z_T, z_1, ..., z_N] with exactly `num_steps + 1` rows.

Nothing is decoded (`output_type="latent"`) and no latents are persisted except
for the `--save_latents_first` figure sample: the regularity metrics are
computed online by `analysis/trajectory_math.py` and one compact JSON record is
written per (prompt_idx, seed). Existing records are skipped, so a failed shard
is re-run by resubmitting the same command.

Example (single shard):

    PYTHONPATH=$PWD python flux/full_trajectory_probe.py \\
        --prompts resources/prompts/prompt.txt --dataset drawbench_full \\
        --seed 41 --output_dir ~/full_traj_results/flux/drawbench_s41 \\
        --shard 0/4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.trajectory_math import (  # noqa: E402
    parse_segments,
    plane_frame,
    segment_tag,
    trajectory_metrics,
)
from lib.io_utils import (  # noqa: E402
    read_prompts,
    seed_for,
    split_shard,
    write_plane_frame,
)

# v2 adds the update-subspace fields (update_chord_share,
# update_in_position_plane, update_own_evr); v1 rows lack them, so the
# version is what distinguishes "field absent" from "field degenerate".
# v3 adds the coarse-window curvature profile (turn_angle_w5_deg), the
# `device_name` the run happened on, and the optional `--save_frame` sidecar.
# v4 adds turn_angle_w7_deg: w=5 does not clear the bf16 floor in the middle
# of a Qwen trajectory, and the profile cannot be recomputed without latents.
# v5 replaces the single `frame_file` with a `frame_files` map, one frame per
# requested row range, so a pairwise plane comparison can be run on a segment
# of the path instead of only the whole of it.
SCHEMA = "full_trajectory.v5"
MODEL = "flux"


def _shard(value: str) -> tuple[int, int]:
    """Parse `--shard i/n`."""
    idx_s, _, count_s = value.partition("/")
    if not count_s:
        raise argparse.ArgumentTypeError("--shard must look like 'i/n'")
    idx, count = int(idx_s), int(count_s)
    if count <= 0 or not (0 <= idx < count):
        raise argparse.ArgumentTypeError(f"--shard {value} out of range")
    return idx, count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True,
                   help="Prompt list, one per line (registry file for --dataset).")
    p.add_argument("--dataset", required=True,
                   help="Dataset label recorded in every record, e.g. drawbench_full.")
    p.add_argument("--seed", type=int, required=True,
                   help="Base seed; per-prompt seed = seed + global_idx.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--shard", type=_shard, default=(0, 1),
                   help="'i/n' inter-node shard, split via lib.io_utils.split_shard.")
    p.add_argument("--save_latents_first", type=int, default=0,
                   help="Persist full Z as bf16 .pt for prompt indices < M "
                        "(figure sample only; 0 = never).")
    p.add_argument("--save_frame", action="store_true",
                   help="Persist the [chord, PC1, PC2] orthonormal frame as a "
                        "float16 .npy per generation. Needed for any pairwise "
                        "'do two trajectories bend in the same plane' analysis, "
                        "which no per-trajectory scalar can answer. ~1.6 MB per "
                        "frame, against the whole latent path at 17x that.")
    p.add_argument("--frame_segments", default=None,
                   help="Row ranges to store a frame for, e.g. '0:51,38:51'. "
                        "Implies --save_frame. Storing a frame per segment is "
                        "what lets a later analysis ask whether two trajectories "
                        "still share a plane over the LATE part of the path; that "
                        "cannot be recovered afterwards, because the latents "
                        "themselves are not kept. Default: the whole path only.")

    # protocol: same defaults as flux/runner.py --mode original
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Atomic write: skip-if-exists must never see a half-written record."""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _install_z_t_capture(pipe) -> dict[str, Any]:
    """Record the pipeline's own initial latents. `prepare_latents` runs once
    per generation, before the denoise loop, so this is not a hot-path hook."""
    holder: dict[str, Any] = {}
    original = pipe.prepare_latents

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        latents = out[0] if isinstance(out, tuple) else out
        holder["z_T"] = latents.detach().clone()
        return out

    pipe.prepare_latents = wrapped
    return holder


def main() -> int:
    args = parse_args()
    shard_idx, shard_count = args.shard

    # Resolve and validate BEFORE the model loads: a malformed spec should cost
    # seconds, not a model load plus a GPU hour.
    N_ROWS = int(args.num_steps) + 1
    if args.frame_segments:
        segments = parse_segments(args.frame_segments, N_ROWS)
    elif args.save_frame:
        segments = [(0, N_ROWS)]
    else:
        segments = []

    # A run dir already holding records is resumed by skip-if-exists, which keys
    # on the record alone. Resuming into a dir whose records were written for a
    # DIFFERENT segment set would skip every generation and write no frames at
    # all, while reporting success — so refuse instead.
    if segments:
        _existing = next(iter(sorted(args.output_dir.glob("traj_*.json"))), None)
        if _existing is not None:
            _prev = json.loads(_existing.read_text(encoding="utf-8"))
            _want = [[a, b] for a, b in segments]
            if _prev.get("frame_segments") != _want:
                raise SystemExit(
                    f"{args.output_dir} holds records written for segments "
                    f"{_prev.get('frame_segments')}, not {_want}. Resume would skip "
                    f"every generation and store no frames. Use a fresh --output_dir."
                )

    import torch

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompts, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), shard_count, shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {shard_idx}/{shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} "
          f"dtype={args.dtype} dataset={args.dataset} seed={args.seed}", flush=True)
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in "
          f"{time.perf_counter() - load_start:.1f}s; shard {shard_idx}/{shard_count} has "
          f"{len(shard_prompts)} prompts (global idx {start}..{end - 1})", flush=True)

    z_t_holder = _install_z_t_capture(pipe)
    N = int(args.num_steps)
    written = skipped = 0

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        out_path = args.output_dir / f"traj_{global_idx:05d}_s{args.seed}.json"
        if out_path.is_file():
            skipped += 1
            continue

        wall_start = time.perf_counter()
        prompt_seed = seed_for(args.seed, global_idx)
        generator = torch.Generator(device=device).manual_seed(int(prompt_seed))

        trace: list[torch.Tensor] = []

        def _cb(_pipe, _step_index, _timestep, callback_kwargs):
            trace.append(callback_kwargs["latents"].detach().to("cpu", torch.float32))
            return callback_kwargs

        z_t_holder.pop("z_T", None)
        denoise_start = time.perf_counter()
        pipe(
            prompt=prompt,
            num_inference_steps=N,
            guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
            height=(args.height // 16) * 16,
            width=(args.width // 16) * 16,
            max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
            num_images_per_prompt=1,
            generator=generator,
            output_type="latent",
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        denoise_s = time.perf_counter() - denoise_start

        if "z_T" not in z_t_holder:
            raise RuntimeError(f"prompt {global_idx}: prepare_latents was never called")
        z_T = z_t_holder.pop("z_T").to("cpu", torch.float32)
        latents = [z_T] + trace
        if len(latents) != N + 1:
            raise RuntimeError(
                f"prompt {global_idx}: captured {len(latents)} latents, expected {N + 1}"
            )

        sigmas = [float(v) for v in pipe.scheduler.sigmas.detach().cpu().tolist()]
        if len(sigmas) != N + 1:
            raise RuntimeError(
                f"prompt {global_idx}: scheduler has {len(sigmas)} sigmas, expected {N + 1}"
            )

        latent_shape = list(z_T.shape)
        Z = torch.stack([t.reshape(-1) for t in latents])
        metrics_start = time.perf_counter()
        metrics = trajectory_metrics(Z.numpy(), sigmas)
        metrics_s = time.perf_counter() - metrics_start

        if global_idx < int(args.save_latents_first):
            torch.save(
                Z.to(torch.bfloat16),
                args.output_dir / f"latents_{global_idx:05d}_s{args.seed}.pt",
            )

        frame_files: dict[str, str] = {}
        if segments:
            Z_np = Z.numpy()
            for a, b in segments:
                frame = plane_frame(Z_np[a:b])
                if frame is None:
                    raise RuntimeError(
                        f"prompt {global_idx}: rows {a}:{b} have no chord-orthogonal plane"
                    )
                tag = segment_tag(a, b)
                name = f"frame_{global_idx:05d}_s{args.seed}_{tag}.npy"
                write_plane_frame(args.output_dir / name, frame)
                frame_files[tag] = name

        record: dict[str, Any] = {
            "schema": SCHEMA,
            "model": MODEL,
            "model_id": args.model_id,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "prompt_file": str(args.prompts),
            "prompt_idx": int(global_idx),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "seed": int(args.seed),
            "prompt_seed": int(prompt_seed),
            "num_steps": N,
            "width": (args.width // 16) * 16,
            "height": (args.height // 16) * 16,
            "guidance": float(args.guidance),
            "dtype": args.dtype,
            "latent_shape": latent_shape,
            "d": int(Z.shape[1]),
            "sigmas": sigmas,
            "z_T_sha256": hashlib.sha256(z_T.numpy().tobytes()).hexdigest(),
            "git_sha": git_sha,
            # same seed gives a different z_T on a different GPU architecture
            # (the CUDA Philox mapping depends on SM count), so any paired /
            # same-noise analysis must not mix device_name values
            "device_name": device_name,
            "frame_files": frame_files,
            "frame_segments": [[a, b] for a, b in segments],
            "denoise_s": denoise_s,
            "metrics_s": metrics_s,
            "wall_s": time.perf_counter() - wall_start,
        }
        record.update(metrics)
        _write_record(out_path, record)
        written += 1
        print(f"[shard {shard_idx}] idx={global_idx} seed={prompt_seed} "
              f"max_dev_ratio={metrics['max_dev_ratio']:.5f} "
              f"straightness={metrics['straightness']:.5f} "
              f"denoise={denoise_s:.1f}s", flush=True)

    print(f"[shard {shard_idx}/{shard_count}] wrote {written}, skipped {skipped} "
          f"(existing) into {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
