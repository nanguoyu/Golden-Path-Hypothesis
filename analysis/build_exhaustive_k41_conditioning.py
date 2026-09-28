#!/usr/bin/env python3
"""Build the frozen canonical conditioning artifact for exhaustive K41."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flux.exhaustive_k41_runner import (  # noqa: E402
    CONDITIONING_SCHEMA,
    DEFAULT_PROMPT_INDICES,
    _encode_prompt,
    _file_sha256,
    _prompt_text_sha256,
)
from flux.sp_cross_runner import resolve_model_commit  # noqa: E402
from lib.io_utils import read_prompts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=Path("resources/prompts/partiprompts_full_eval1632_seed42.txt"),
    )
    parser.add_argument(
        "--prompt_indices",
        default=",".join(str(idx) for idx in DEFAULT_PROMPT_INDICES),
    )
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", default="3de623fc")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    indices = tuple(int(piece.strip()) for piece in args.prompt_indices.split(","))
    if indices != DEFAULT_PROMPT_INDICES:
        raise SystemExit(
            f"canonical artifact requires prompt indices {DEFAULT_PROMPT_INDICES}"
        )
    selected_prompts = tuple(prompts[idx] for idx in indices)
    run_args = SimpleNamespace(model_name="flux-dev")

    from diffusers import DiffusionPipeline

    weights = resolve_model_commit(args.model_id, args.revision)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, revision=args.revision
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    if weights.get("model_commit") is None:
        commit = getattr(getattr(pipe, "config", None), "_commit_hash", None)
        if isinstance(commit, str) and len(commit) == 40:
            weights["model_commit"] = commit
            weights["model_commit_source"] = "loaded_pipeline_config"
    if weights.get("model_commit") is None:
        raise RuntimeError("could not resolve the loaded model commit")

    conditioning = {}
    for prompt_idx, prompt in zip(indices, selected_prompts):
        encoded = _encode_prompt(pipe, prompt, run_args)
        conditioning[str(prompt_idx)] = {
            "prompt_embeds": encoded["prompt_embeds"].detach().cpu().contiguous(),
            "pooled_prompt_embeds": encoded["pooled_prompt_embeds"]
            .detach()
            .cpu()
            .contiguous(),
        }
    artifact = {
        "schema": CONDITIONING_SCHEMA,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_commit": weights["model_commit"],
        "prompt_indices": list(indices),
        "prompt_text_sha256": _prompt_text_sha256(selected_prompts),
        "max_sequence_length": 512,
        "software_versions": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "torch",
                    "diffusers",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "pillow",
                )
            },
        },
        "conditioning": conditioning,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    torch.save(artifact, tmp)
    tmp.replace(args.output)
    result = {
        "schema": CONDITIONING_SCHEMA,
        "output": str(args.output),
        "sha256": _file_sha256(args.output),
        "model_commit": weights["model_commit"],
        "prompt_indices": list(indices),
        "prompt_text_sha256": artifact["prompt_text_sha256"],
        "software_versions": artifact["software_versions"],
        "bytes": args.output.stat().st_size,
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
