#!/usr/bin/env python3
"""Collect a FLUX final-output Gram shard for L2P fitting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.l2p_output import FluxL2POutputAdapter
from lib.io_utils import read_prompts, seed_for, split_shard
from lib.l2p import accumulate_l2p_gram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--gram_out", type=Path, required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def _identity_weights(path: Path, num_steps: int) -> None:
    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for step in range(1, num_steps):
        weights[step, step - 1] = 1.0
    torch.save(
        {
            "format": "l2p-v1",
            "target": "final_output",
            "model": "flux",
            "num_steps": num_steps,
            "weights": weights,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX L2P collection requires CUDA")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to("cuda")
    args.gram_out.parent.mkdir(parents=True, exist_ok=True)
    identity = args.gram_out.with_name(
        f".{args.gram_out.name}.identity.{os.getpid()}.pt"
    )
    _identity_weights(identity, args.num_steps)
    adapter = FluxL2POutputAdapter(
        pipe,
        weights_path=identity,
        cache_steps=(),
        num_steps=args.num_steps,
    )
    gram = torch.zeros((args.num_steps, args.num_steps), dtype=torch.float64)
    adapter.install()
    try:
        for local_idx, prompt in enumerate(selected):
            global_idx = start + local_idx
            seed = seed_for(args.seed, global_idx)
            adapter.reset(prompt_idx=global_idx, seed=seed)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            pipe(
                prompt=prompt,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance,
                height=(args.height // 16) * 16,
                width=(args.width // 16) * 16,
                max_sequence_length=512,
                generator=generator,
                output_type="latent",
            )
            torch.cuda.synchronize()
            accumulate_l2p_gram(gram, adapter.history, num_steps=args.num_steps)
            print(
                f"[flux-l2p-collect] shard={args.shard_idx} "
                f"{local_idx + 1}/{len(selected)}",
                flush=True,
            )
    finally:
        adapter.restore()
        identity.unlink(missing_ok=True)

    torch.save(
        {
            "format": "l2p-gram-v1",
            "model": "flux",
            "target": "final_output",
            "num_steps": args.num_steps,
            "prompt_file": str(args.prompt_file),
            "prompt_count": len(selected),
            "shard_idx": args.shard_idx,
            "shard_count": args.shard_count,
            "gram": gram,
        },
        args.gram_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
