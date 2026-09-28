#!/usr/bin/env python3
"""Collect a Tencent HunyuanVideo final-output Gram shard for L2P fitting."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hunyuan_video.backend import generate, load_official_sampler
from hunyuan_video.config import RunSpec, load_protocol
from lib.io_utils import read_prompts, seed_for, split_shard
from lib.l2p import accumulate_l2p_gram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--gram_out", type=Path, required=True)
    parser.add_argument("--model_base", type=Path, required=True)
    parser.add_argument("--protocol_id", default="HY-CachePaper-480")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    return parser.parse_args()


def _identity_weights(path: Path, num_steps: int) -> None:
    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for step in range(1, num_steps):
        weights[step, step - 1] = 1.0
    torch.save(
        {
            "format": "l2p-v1",
            "target": "final_output",
            "model": "hunyuan_video",
            "num_steps": num_steps,
            "weights": weights,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Hunyuan L2P collection must run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan L2P collection requires CUDA")
    protocol = load_protocol(args.protocol_id)
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]

    sampler, _api, _load = load_official_sampler(args.model_base, protocol)
    args.gram_out.parent.mkdir(parents=True, exist_ok=True)
    identity = args.gram_out.with_name(
        f".{args.gram_out.name}.identity.{os.getpid()}.pt"
    )
    _identity_weights(identity, protocol.steps)
    gram = torch.zeros((protocol.steps, protocol.steps), dtype=torch.float64)
    try:
        for local_idx, prompt in enumerate(selected):
            global_idx = start + local_idx
            seed = seed_for(args.seed, global_idx)
            run = RunSpec(
                phase="l2p_fit",
                task_id=f"l2p-fit-{args.shard_idx}-{global_idx}",
                protocol_id=protocol.protocol_id,
                mode="l2p_output_exact",
                prompt_id=f"l2p-fit-{global_idx}",
                prompt=prompt,
                seed=seed,
                repeat=0,
                method_config={
                    "cache_count": 0,
                    "cache_steps": [],
                    "weights_path": str(identity),
                },
            )
            output, adapter, _seconds, _peak = generate(sampler, protocol, run)
            if adapter is None:
                raise RuntimeError("Hunyuan L2P collector did not receive its adapter")
            accumulate_l2p_gram(
                gram,
                adapter.method.history,
                num_steps=protocol.steps,
            )
            del output, adapter
            torch.cuda.empty_cache()
            print(
                f"[hunyuan-l2p-collect] shard={args.shard_idx} "
                f"{local_idx + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        identity.unlink(missing_ok=True)

    torch.save(
        {
            "format": "l2p-gram-v1",
            "model": "hunyuan_video",
            "target": "final_output",
            "num_steps": protocol.steps,
            "prompt_file": str(args.prompt_file),
            "prompt_count": len(selected),
            "protocol_id": protocol.protocol_id,
            "shard_idx": args.shard_idx,
            "shard_count": args.shard_count,
            "gram": gram,
        },
        args.gram_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
