#!/usr/bin/env python3
"""Collect a Wan2.1 head-output Gram shard for L2P fitting.

L2P predicts the transformer's final output feature from the features stored at
earlier steps, ``F_hat_k = sum_{j<k} W_kj F_j``, and ``W`` is solved offline from
the timestep Gram of full trajectories.  On Wan the feature is the output of
``model.head`` -- the final projection before ``unpatchify`` and the tensor
``wan21/adapter.py::HeadOutputAdapter`` substitutes at deployment.

The collection run installs that same adapter with an **empty cache schedule**,
so every step is full and every stored feature is the real head output; the
weights file it is nevertheless required to load (plan section 2.8 makes
``weights_path`` a hard argument) is a throwaway identity matrix that is never
consulted, because ``L2POutputMethod.final_output`` only predicts on cache steps.

**CFG decision (plan section 2.8).**  Wan runs a cond and an uncond forward per
solver step and each keeps its own prediction history, but the fitted asset has
no branch axis: both branches' Grams are summed into one matrix and one shared
50x50 ``W`` is solved from it.  A branch is not a dataset, and per-branch weights
would break the "one asset per backbone" discipline the plan holds the fitted
assets to (section 3.2b).  The shard therefore contributes ``2 * len(prompts)``
trajectories, recorded as ``prompt_count``, with ``prompt_pairs`` carrying the
prompt count itself -- the field ``analysis/solve_l2p_grams.py:56`` already reads
for the Qwen true-CFG collector.

``analysis/solve_l2p_grams.py --ridge 1e-5`` merges the shards and reports the
holdout error; the same script is run a second time on
``resources/baseline_exact/hunyuan_holdout10.txt`` to produce the holdout Gram.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.io_utils import read_prompts, seed_for, split_shard  # noqa: E402
from lib.l2p import accumulate_l2p_gram  # noqa: E402
from wan21.backend import (  # noqa: E402
    WanProtocol,
    WanRunSpec,
    build_adapter,
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

#: The tag `analysis/solve_l2p_grams.py` cross-checks across shards and writes
#: into the solved checkpoint, and `hunyuan_video/methods/l2p.py` never reads --
#: it is provenance, and mixing a Wan Gram into a Hunyuan fit is what it stops.
MODEL_TAG = "wan21"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--gram_out", type=Path, required=True)
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


def _identity_weights(path: Path, num_steps: int) -> None:
    """Write the placeholder L2P weight file the collector is required to load.

    `L2POutputMethod` refuses to build without a readable weights file, and the
    collector runs with no cache steps, so this matrix is loaded, validated and
    never evaluated.  It is deleted in the `finally` block.
    """

    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for step in range(1, num_steps):
        weights[step, step - 1] = 1.0
    torch.save(
        {
            "format": "l2p-v1",
            "target": "final_output",
            "model": MODEL_TAG,
            "num_steps": num_steps,
            "weights": weights,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.protocol_id != PROTOCOL_ID:
        raise SystemExit(
            f"Wan2.1 L2P collection runs protocol {PROTOCOL_ID}, not {args.protocol_id}"
        )
    if args.ckpt_dir is None:
        raise SystemExit("--ckpt_dir is required or WAN21_CKPT_DIR must be set")
    if args.resume and args.gram_out.is_file():
        print(f"[wan21-l2p-collect] preserve {args.gram_out}", flush=True)
        return 0
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Wan2.1 L2P collection must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.1 L2P collection requires CUDA")

    protocol = WanProtocol()
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

    args.gram_out.parent.mkdir(parents=True, exist_ok=True)
    identity = args.gram_out.with_name(
        f".{args.gram_out.name}.identity.{os.getpid()}.pt"
    )
    _identity_weights(identity, protocol.steps)
    gram = torch.zeros((protocol.steps, protocol.steps), dtype=torch.float64)
    trajectories = 0
    try:
        adapter = build_adapter(
            pipe.model,
            protocol,
            WanRunSpec(
                method="l2p",
                method_config={
                    "cache_steps": [],
                    "cache_count": 0,
                    "weights_path": str(identity),
                },
            ),
        )
        with adapter:
            for local_idx, prompt in enumerate(selected):
                global_idx = start + local_idx
                seed = seed_for(args.seed, global_idx)
                adapter.reset()
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
                cached = [
                    row for row in adapter.decisions if row.action != "full"
                ]
                if cached:
                    raise RuntimeError(
                        "Wan2.1 L2P collection must run every step full, "
                        f"saw {len(cached)} cached steps on prompt {global_idx}"
                    )
                for branch in ("cond", "uncond"):
                    accumulate_l2p_gram(
                        gram,
                        adapter.methods[branch].history,
                        num_steps=protocol.steps,
                    )
                    trajectories += 1
                adapter.reset()
                torch.cuda.empty_cache()
                print(
                    f"[wan21-l2p-collect] shard={args.shard_idx} "
                    f"{local_idx + 1}/{len(selected)}",
                    flush=True,
                )
    finally:
        identity.unlink(missing_ok=True)
    verify_untouched_transformer(pipe.model)

    payload: dict[str, Any] = {
        "format": "l2p-gram-v1",
        "model": MODEL_TAG,
        "target": "final_output",
        "num_steps": protocol.steps,
        "prompt_file": str(args.prompt_file),
        # `prompt_count` is the trajectory count the solver sums; `prompt_pairs`
        # is the prompt count, which is half of it because Wan contributes one
        # cond and one uncond trajectory per prompt.
        "prompt_count": trajectories,
        "prompt_pairs": len(selected),
        "protocol_id": PROTOCOL_ID,
        "branch_policy": "cond_decides",
        "injection_point": "wan_model_head",
        "shard_idx": args.shard_idx,
        "shard_count": args.shard_count,
        "gram": gram,
    }
    torch.save(payload, args.gram_out)
    print(
        f"[wan21-l2p-collect] wrote {args.gram_out} "
        f"trajectories={trajectories} prompts={len(selected)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
