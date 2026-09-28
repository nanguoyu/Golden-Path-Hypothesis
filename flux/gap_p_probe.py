#!/usr/bin/env python3
"""Step 1 of the gap-aware extension (docs/research_plan_extension.md §8).

Gap-aware prediction-error probe:

    P_{a->k} = || F_hat_{a->k} - F_k ||

Runs ONE full-computation trajectory per prompt, records the coarse
whole-transformer residual F_k at every solver step, then OFFLINE computes
the method-specific gap-aware prediction error for every (a, k) pair with
0 <= a < k <= N-1:

  - seacache  : F_hat_{a->k} = F_a                       (zero-order reuse)
  - hicache   : F_hat_{a->k} = hicache_predict(hist_a, g=k-a, sigma, O)
  - taylorseer: F_hat_{a->k} = taylor_predict(hist_a,  g=k-a, O)

where hist_a is the divided-difference history built from F_0..F_a on a
uniform step grid -- exactly what `flux/oracle_runner.py` builds online via
`hermite_update`. The single-step Phase 1 proxy r_k is the g=1 diagonal:
P_{k-1->k}.

No cache event is ever injected; this is pure offline tensor arithmetic on
one saved trajectory. It tests H5 (P_{a->k} grows with gap g=k-a) and
produces the (a, g) heatmap input.

Output per prompt: <output_dir>/prompt_XXXXX/p_probe.json  (flat per-pair
list + per-step residual norms). Cost ~1x baseline per prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.hermite import hermite_update, hicache_predict  # noqa: E402
from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.taylor import taylor_predict  # noqa: E402
from flux.oracle_runner import _decode_to_pil, _run_one_pipe_call  # noqa: E402

logger = logging.get_logger(__name__)


# ----------------------------------------------------------------------------
# Probe forward: full computation at every step, records the coarse residual.
# Identical block-compute path to flux/oracle_runner.py::_oracle_forward, but
# no gating -- it never caches, it only dumps `actual_residual` per step.
# ----------------------------------------------------------------------------
def _probe_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """Replacement for FluxTransformer2DModel.forward that records F_k."""
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale") is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    if txt_ids is not None and txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # ---- Full block compute (no caching) -----------------------------------
    ori_hidden_states = hidden_states
    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
            )
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    for index_block, block in enumerate(self.single_transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if controlnet_single_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples))
            )
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

    # ---- Record the coarse whole-transformer residual F_k ------------------
    # Same definition as oracle_runner: F_k is what the SeaCache/HiCache/
    # TaylorSeer predictors reuse/extrapolate. Kept in model dtype (bf16) so
    # the offline divided differences match what the online predictor sees.
    # `actual_residual` is a fresh subtraction result (not a view), so
    # `.detach()` alone is enough -- matches oracle_runner's history update.
    actual_residual = hidden_states - ori_hidden_states
    self.residual_dump.append(actual_residual.detach())
    self.cnt += 1
    if self.cnt == int(self.num_steps):
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----------------------------------------------------------------------------
# install / teardown
# ----------------------------------------------------------------------------
def install_probe(pipe, *, num_steps: int) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward to dump per-step residuals."""
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _probe_forward

    tr = pipe.transformer
    tr.enable_probe = True
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.residual_dump = []

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("enable_probe", "num_steps", "cnt", "residual_dump"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_probe_state(pipe) -> None:
    """Clear per-trajectory state before each new pipe(...) call."""
    tr = pipe.transformer
    tr.cnt = 0
    tr.residual_dump = []


# ----------------------------------------------------------------------------
# Offline P_{a->k} matrix
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_p_matrix(
    residuals: List[torch.Tensor],
    *,
    hicache_max_order: int,
    hicache_sigma: float,
    taylorseer_max_order: int,
) -> tuple[List[float], List[Dict[str, Any]]]:
    """Compute P_{a->k} for every 0 <= a < k <= N-1 and all three predictors.

    `residuals[j]` is the coarse residual F_j (model dtype, on device).
    Returns (f_norm, per_pair) where f_norm[k] = ||F_k|| and per_pair is a
    flat list of dicts {a, k, g, p_seacache, p_hicache, p_taylorseer}.
    """
    N = len(residuals)
    history_max_order = max(int(hicache_max_order), int(taylorseer_max_order))

    # snapshots[a] = divided-difference history as of full step a, built
    # incrementally on a uniform grid (step_gap=1) -- matches oracle_runner.
    # hermite_update returns a fresh dict each call and never mutates the
    # input, so keeping each returned dict is safe.
    snapshots: List[Dict[int, torch.Tensor]] = []
    hist: Dict[int, torch.Tensor] = {}
    for j in range(N):
        hist = hermite_update(hist, residuals[j], step_gap=1,
                              max_order=history_max_order)
        snapshots.append(hist)

    res_f = [r.float() for r in residuals]
    f_norm = [float(res_f[k].norm().item()) for k in range(N)]

    per_pair: List[Dict[str, Any]] = []
    for a in range(N - 1):
        snap = snapshots[a]
        f0_f = snap[0].float()
        for k in range(a + 1, N):
            g = k - a
            target = res_f[k]
            # seacache zero-order: F_hat = F_a
            p_sea = float((f0_f - target).norm().item())
            # hicache Hermite extrapolation
            pred_hi = hicache_predict(snap, g, hicache_sigma, hicache_max_order)
            p_hi = float((pred_hi.float() - target).norm().item())
            # taylorseer Taylor extrapolation
            pred_ts = taylor_predict(snap, g, taylorseer_max_order)
            p_ts = float((pred_ts.float() - target).norm().item())
            per_pair.append({
                "a": a, "k": k, "g": g,
                "p_seacache": p_sea,
                "p_hicache": p_hi,
                "p_taylorseer": p_ts,
            })
    return f_norm, per_pair


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gap-aware prediction-error probe (P_{a->k}) on FLUX."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after read, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose p_probe.json marks complete.")
    p.add_argument("--save_baseline_png", action="store_true",
                   help="Also decode + save the full-trajectory image (sanity).")
    p.add_argument("--hicache_max_order", type=int, default=2,
                   help="HiCache Hermite truncation order O (paper Tab 1: 2).")
    p.add_argument("--hicache_sigma", type=float, default=0.5,
                   help="HiCache dual-scaling factor (paper Tab 1: 0.5).")
    p.add_argument("--taylorseer_max_order", type=int, default=1,
                   help="TaylorSeer expansion order (project default: 1).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded in {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts "
          f"(global {start}..{end - 1})", flush=True)

    N = int(args.num_steps)
    H = (args.height // 16) * 16
    W = (args.width // 16) * 16

    teardown = install_probe(pipe, num_steps=N)
    per_prompt_records = []
    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
            json_path = prompt_dir / "p_probe.json"

            if args.resume and json_path.is_file():
                try:
                    if json.loads(json_path.read_text()).get("complete", False):
                        print(f"[shard {args.shard_idx}] prompt {global_idx} complete, skip",
                              flush=True)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass

            prompt_dir.mkdir(parents=True, exist_ok=True)
            per_image_seed = args.seed + global_idx
            t_p = time.perf_counter()

            # ---- one full trajectory; probe forward dumps F_0..F_{N-1} -----
            reset_probe_state(pipe)
            z_base = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
            residuals = list(pipe.transformer.residual_dump)
            if len(residuals) != N:
                raise RuntimeError(
                    f"prompt {global_idx}: dumped {len(residuals)} residuals, expected {N}"
                )
            if args.save_baseline_png:
                _decode_to_pil(pipe, z_base, H, W).save(prompt_dir / "baseline.png")

            # ---- offline P_{a->k} matrix -----------------------------------
            f_norm, per_pair = compute_p_matrix(
                residuals,
                hicache_max_order=int(args.hicache_max_order),
                hicache_sigma=float(args.hicache_sigma),
                taylorseer_max_order=int(args.taylorseer_max_order),
            )
            # free trajectory tensors before next prompt
            residuals.clear()
            reset_probe_state(pipe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            wall = time.perf_counter() - t_p
            payload = {
                "experiment": "gap_p_probe",
                "prompt_idx": global_idx,
                "prompt": prompt,
                "seed": int(per_image_seed),
                "num_steps": N,
                "hicache_max_order": int(args.hicache_max_order),
                "hicache_sigma": float(args.hicache_sigma),
                "taylorseer_max_order": int(args.taylorseer_max_order),
                "f_norm": f_norm,
                "per_pair": per_pair,
                "wall_seconds": wall,
                "complete": True,
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False))
            per_prompt_records.append({"global_idx": global_idx, "wall_seconds": wall})
            print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
                  f"({len(per_pair)} pairs) ({local_idx + 1}/{len(shard_prompts)})",
                  flush=True)
    finally:
        teardown()

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing_path.write_text(json.dumps({
        "experiment": "gap_p_probe",
        "num_steps": N,
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "base_seed": int(args.seed),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": device_str,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts),
        "per_prompt": per_prompt_records,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path} "
          f"(total wall {process_end - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
