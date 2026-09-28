#!/usr/bin/env python3
"""Collect Qwen-Image input/output relative-L1 pairs for TeaCache fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.gates import rel_l1
from lib.io_utils import read_prompts, seed_for, split_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen-Image")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class QwenTeaFitCollector:
    def __init__(self, transformer: Any):
        self.transformer = transformer
        self.rows: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self.reset()

    def reset(self) -> None:
        self.call_idx = 0
        self.current_input: torch.Tensor | None = None
        self.previous = {
            "cond": {"input": None, "output": None},
            "uncond": {"input": None, "output": None},
        }

    def install(self) -> None:
        self._handles.append(
            self.transformer.transformer_blocks[0].register_forward_pre_hook(
                self._first_block_pre,
                with_kwargs=True,
            )
        )
        self._handles.append(
            self.transformer.register_forward_hook(
                self._transformer_post,
                with_kwargs=True,
            )
        )

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def _first_block_pre(
        self,
        block: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        temb = kwargs.get("temb", args[2] if len(args) > 2 else None)
        if hidden is None or temb is None:
            raise RuntimeError("TeaCache collector could not capture first-block inputs")
        img_mod1, _img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
        normed = block.img_norm1(hidden)
        self.current_input = block._modulate(normed, img_mod1)[0].detach()

    def _transformer_post(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        branch = "cond" if self.call_idx % 2 == 0 else "uncond"
        step = self.call_idx // 2
        value = getattr(
            output,
            "sample",
            output[0] if isinstance(output, tuple) else output,
        )
        if self.current_input is None or not isinstance(value, torch.Tensor):
            raise RuntimeError("TeaCache collector did not capture a tensor trajectory")
        previous = self.previous[branch]
        if previous["input"] is not None:
            self.rows.append(
                {
                    "step": int(step),
                    "branch": branch,
                    "input_rel_l1": float(rel_l1(self.current_input, previous["input"])),
                    "output_rel_l1": float(rel_l1(value, previous["output"])),
                }
            )
        previous["input"] = self.current_input
        previous["output"] = value.detach()
        self.current_input = None
        self.call_idx += 1
        return output


def main() -> int:
    args = parse_args()
    if args.resume and args.out.is_file():
        print(f"[qwen-tea-fit] preserve {args.out}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen TeaCache fitting requires CUDA")
    prompts = read_prompts(args.prompt_file, limit=args.limit)
    start, end = split_shard(len(prompts), args.shard_count, args.shard_idx)
    selected = prompts[start:end]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype).to("cuda")
    collector = QwenTeaFitCollector(pipe.transformer)
    collector.install()
    all_rows: list[dict[str, Any]] = []
    try:
        for local_idx, prompt in enumerate(selected):
            idx = start + local_idx
            collector.reset()
            generator = torch.Generator(device="cuda").manual_seed(seed_for(args.seed, idx))
            pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                true_cfg_scale=args.true_cfg_scale,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                generator=generator,
                output_type="latent",
                return_dict=True,
            )
            torch.cuda.synchronize()
            all_rows.extend({"prompt_idx": idx, **row} for row in collector.rows)
            print(f"[qwen-tea-fit] {local_idx + 1}/{len(selected)}", flush=True)
    finally:
        collector.restore()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "format": "qwen_teacache_pairs.v1",
                "prompt_file": str(args.prompt_file),
                "prompt_count": len(selected),
                "shard_idx": args.shard_idx,
                "shard_count": args.shard_count,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
