#!/usr/bin/env python3
"""Fit shared L2P timestep weights from FLUX full trajectories.

The script runs full 50-step trajectories on calibration prompts and accumulates
feature Gram matrices, then solves the least-squares linear predictor

    F_t ~= sum_{j<t} W[t,j] F_j

without storing the raw feature bank on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.flux_fine_scaffold import CacheHistory, install_fine_cache, reset_per_image_state_fine  # noqa: E402
from lib.io_utils import read_prompts  # noqa: E402
from lib.l2p import append_history, latest_history_step  # noqa: E402


class FineCollectorPredictor:
    """Fine scaffold predictor that only records full-step slot histories."""

    max_order = 0

    def update(
        self,
        prev_history: Optional[CacheHistory],
        feature: torch.Tensor,
        step_gap: int,
        effective_max_order: int,
    ) -> CacheHistory:
        last_step = latest_history_step(prev_history or {})
        current_step = 0 if last_step is None else int(last_step) + int(step_gap)
        # Fine collection has 114 slots; keeping all histories on GPU OOMs even
        # on H100. Skip paths are disabled during fitting, so CPU histories are
        # safe and are moved back only slot-by-slot for Gram accumulation.
        return append_history(prev_history, current_step, feature.detach().to("cpu"))

    def predict(self, history: CacheHistory, step_offset: int) -> torch.Tensor:
        raise RuntimeError("FineCollectorPredictor should never be asked to predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit FLUX L2P weights from calibration prompts")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output checkpoint path")
    p.add_argument("--granularity", choices=("final_hidden", "fine_114"), required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=("flux-dev", "flux-schnell"), default="flux-dev")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--ridge", type=float, default=1e-5)
    p.add_argument("--output_type", default="latent", choices=("latent", "pil"))
    return p.parse_args()


def _identity_checkpoint(path: Path, num_steps: int) -> None:
    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for t in range(1, num_steps):
        weights[t, t - 1] = 1.0
    torch.save({
        "format": "l2p-v1",
        "target": "final_hidden",
        "granularity": "final_hidden",
        "num_steps": int(num_steps),
        "weights": weights,
    }, path)


def _history_matrix(history: dict[int, torch.Tensor], num_steps: int) -> torch.Tensor:
    steps = sorted(int(k) for k in history.keys())
    expected = list(range(int(num_steps)))
    if steps != expected:
        raise RuntimeError(f"history steps mismatch: expected {expected[:3]}..{expected[-3:]}, got {steps}")
    return torch.stack(
        [history[step].detach().to(torch.float32).reshape(-1) for step in expected],
        dim=0,
    )


def _accumulate_history_gram(
    gram: torch.Tensor,
    history: dict[int, torch.Tensor],
    *,
    num_steps: int,
) -> None:
    mat = _history_matrix(history, num_steps)
    if torch.cuda.is_available() and not mat.is_cuda:
        mat = mat.to("cuda")
    gram.add_((mat @ mat.t()).detach().cpu().to(torch.float64))
    del mat


def _solve_weights(gram: torch.Tensor, *, ridge: float) -> torch.Tensor:
    num_steps = int(gram.shape[0])
    weights = torch.zeros((num_steps, num_steps), dtype=torch.float32)
    for t in range(1, num_steps):
        weights[t, t - 1] = 1.0
        lhs = gram[:t, :t].clone()
        rhs = gram[:t, t].clone()
        scale = float(lhs.diag().mean().item()) if t > 0 else 1.0
        lhs = lhs + torch.eye(t, dtype=torch.float64) * (float(ridge) * max(scale, 1.0))
        try:
            coef = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            coef = torch.linalg.pinv(lhs) @ rhs
        if torch.isfinite(coef).all():
            weights[t, :t] = coef.to(torch.float32)
    return weights


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    if not prompts:
        raise SystemExit("no prompts")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    teardown = None
    identity_path = args.out.with_suffix(".identity_tmp.pt")
    if args.granularity == "final_hidden":
        from flux import l2p
        _identity_checkpoint(identity_path, args.num_steps)
        teardown = l2p.install(
            pipe,
            weights_path=identity_path,
            interval=1,
            first_enhance=0,
            num_steps=args.num_steps,
        )
        reset = l2p.reset_per_image_state
    else:
        teardown = install_fine_cache(
            pipe,
            predictor=FineCollectorPredictor(),
            interval=1,
            first_enhance=0,
            num_steps=args.num_steps,
            method_tag="l2p_fine_collect_full",
        )
        reset = reset_per_image_state_fine

    gram = torch.zeros((args.num_steps, args.num_steps), dtype=torch.float64)
    start = time.perf_counter()
    try:
        for idx, prompt in enumerate(prompts):
            reset(pipe)
            generator = torch.Generator(device=device).manual_seed(int(args.seed) + idx)
            pipe(
                prompt=prompt,
                num_inference_steps=int(args.num_steps),
                guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
                height=(args.height // 16) * 16,
                width=(args.width // 16) * 16,
                max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
                num_images_per_prompt=1,
                generator=generator,
                output_type=args.output_type,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if args.granularity == "final_hidden":
                _accumulate_history_gram(
                    gram,
                    pipe.transformer._l2p_final_hidden_history,
                    num_steps=args.num_steps,
                )
            else:
                state = pipe.transformer._fine_state
                if len(state.cache_dic) != 114:
                    raise RuntimeError(f"expected 114 fine slots, got {len(state.cache_dic)}")
                for history in state.cache_dic.values():
                    _accumulate_history_gram(gram, history, num_steps=args.num_steps)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[l2p-fit] {idx + 1}/{len(prompts)} prompts accumulated", flush=True)
    finally:
        if teardown is not None:
            teardown()
        if identity_path.exists():
            try:
                identity_path.unlink()
            except OSError:
                pass

    weights = _solve_weights(gram, ridge=float(args.ridge))
    target = "final_hidden" if args.granularity == "final_hidden" else "fine_pregate_shared"
    checkpoint = {
        "format": "l2p-v1",
        "target": target,
        "granularity": args.granularity,
        "num_steps": int(args.num_steps),
        "weights": weights,
        "ridge": float(args.ridge),
        "fit_method": "gram_least_squares",
        "train_prompt_file": str(args.prompt_file),
        "train_prompt_count": int(len(prompts)),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "model_name": str(args.model_name),
        "width": int(args.width),
        "height": int(args.height),
        "guidance": float(args.guidance),
        "dtype": str(args.dtype),
        "elapsed_s": float(time.perf_counter() - start),
    }
    torch.save(checkpoint, args.out)
    manifest = {k: v for k, v in checkpoint.items() if k != "weights"}
    manifest["weights_shape"] = list(weights.shape)
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[l2p-fit] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
