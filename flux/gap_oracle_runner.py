#!/usr/bin/env python3
"""Step 2 of the gap-aware extension (docs/research_plan_extension.md §8).

Gap single-step oracle (Oracle A):

    R_{a->k}^oracle = || z_N^{cache only k, history frozen at a} - z_N^full ||

For each (a, k) pair the trajectory runs full computation everywhere EXCEPT
step k, where one cache event is injected using the chosen predictor with
cache history FROZEN at the last refresh step a (gap g = k - a). This
isolates the effect of the refresh gap on a single cached step.

Mechanism (the key difference from flux/oracle_runner.py's Phase 1 oracle,
whose history is always the immediately-preceding full step, g=1):
  - steps 0..a   : full computation, history updates (uniform grid);
  - steps a+1..k-1 : full computation, history NOT updated (frozen at a);
  - step k       : cache event, predictor extrapolates over gap g = k - a;
  - steps k+1..N-1 : full computation (history updates are dead state,
                     never read again -- only one cache event per run).

`R_k` of Phase 1 is the g=1 special case (a = k-1).

Used by H6 (does P_{a->k}*S*Q*A predict R_{a->k}^oracle?) and, run densely
along an interval, supplies the per-step Oracle-A vectors whose sum is the
clean-reference interval error E_clean -> the I_drift residual for H7.

Cost: naive -- each pair = one full trajectory with one cache event. A
prefix-reuse / partial-propagate optimisation (run the reference once, then
resume each pair from step k) would cut this ~2x; it is left as future
work to keep this runner a thin reuse of the pipe infrastructure.

Output dir layout per shard:
    output_dir/prompt_XXXXX/
      baseline.pt / baseline.png            z_N^full (once per prompt)
      gap_aAA_kKK.pt / gap_aAA_kKK.png       z_N for pair (a=AA, k=KK)
      manifest.json                          per-pair a,k,g,R_latent_l2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from flux.oracle_runner import (  # noqa: E402
    CACHE_MODES,
    _decode_to_pil,
    _run_one_pipe_call,
    _save_latent,
)

logger = logging.get_logger(__name__)


# ----------------------------------------------------------------------------
# Gap-oracle forward: full everywhere except one cache event at `cache_step`,
# whose predictor reads cache history frozen at `freeze_step` (= a).
# ----------------------------------------------------------------------------
def _gap_oracle_forward(
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
    """Replacement for FluxTransformer2DModel.forward with gap-oracle gating."""
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

    # ---- Gap-oracle gating: cache fires only at cache_step -----------------
    cur_step = -1
    should_calc = True
    if getattr(self, "enable_gap_oracle", False):
        cur_step = int(self.cnt)
        if cur_step == int(self.cache_step) and self.residual_history:
            should_calc = False
        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for next trajectory

    # ---- Cache event (single step k) --------------------------------------
    if (
        getattr(self, "enable_gap_oracle", False)
        and not should_calc
        and self.residual_history
    ):
        # History was last updated at the freeze step a, so step_offset is
        # exactly the refresh gap g = k - a.
        step_offset = cur_step - int(self.last_activation_step)
        if self.cache_mode == "seacache":
            predicted_residual = self.residual_history[0]
        elif self.cache_mode == "hicache":
            predicted_residual = hicache_predict(
                self.residual_history,
                step_offset=int(step_offset),
                sigma=float(self.hicache_sigma),
                max_order=int(self.hicache_max_order),
            )
        elif self.cache_mode == "taylorseer":
            predicted_residual = taylor_predict(
                self.residual_history,
                step_offset=int(step_offset),
                max_order=int(self.taylorseer_max_order),
            )
        else:
            raise ValueError(f"unknown cache_mode: {self.cache_mode!r}")
        hidden_states = hidden_states + predicted_residual
    else:
        # ---- Full block compute --------------------------------------------
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

        # Update the residual history ONLY up to and including the freeze
        # step a. Steps a+1..N-1 run full but leave the history frozen at a,
        # so the cache event at k extrapolates over the true gap g = k - a.
        if getattr(self, "enable_gap_oracle", False) and cur_step <= int(self.freeze_step):
            actual_residual = hidden_states - ori_hidden_states
            if self.last_activation_step < 0:
                step_gap = 1
            else:
                step_gap = cur_step - int(self.last_activation_step)
            self.residual_history = hermite_update(
                self.residual_history,
                actual_residual.detach(),
                step_gap=int(step_gap),
                max_order=int(self.history_max_order),
            )
            self.last_activation_step = cur_step

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ----------------------------------------------------------------------------
# install / teardown / reset
# ----------------------------------------------------------------------------
def install_gap_oracle(
    pipe,
    *,
    freeze_step: int,
    cache_step: int,
    num_steps: int,
    cache_mode: str = "seacache",
    hicache_max_order: int = 2,
    hicache_sigma: float = 0.5,
    taylorseer_max_order: int = 1,
) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward for one gap-oracle run (a, k).

    Args:
        freeze_step: a -- last refresh step; history is built on steps 0..a
            and then frozen.
        cache_step: k -- the single step where the cache event is injected.
            Requires 0 <= a < k <= num_steps-1.
        num_steps: total sampling steps.
        cache_mode / hicache_* / taylorseer_*: predictor settings, same as
            flux/oracle_runner.py.
    """
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    a, k = int(freeze_step), int(cache_step)
    if not (0 <= a < k <= int(num_steps) - 1):
        raise ValueError(
            f"install_gap_oracle: need 0<=a<k<=N-1, got a={a}, k={k}, N={num_steps}"
        )

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _gap_oracle_forward

    tr = pipe.transformer
    tr.enable_gap_oracle = True
    tr.freeze_step = a
    tr.cache_step = k
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr.cache_mode = str(cache_mode)
    tr.hicache_max_order = int(hicache_max_order)
    tr.hicache_sigma = float(hicache_sigma)
    tr.taylorseer_max_order = int(taylorseer_max_order)
    tr.history_max_order = max(int(hicache_max_order), int(taylorseer_max_order))
    tr.residual_history = {}
    tr.last_activation_step = -1

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_gap_oracle", "freeze_step", "cache_step", "num_steps",
            "cnt", "cache_mode", "hicache_max_order", "hicache_sigma",
            "taylorseer_max_order", "history_max_order",
            "residual_history", "last_activation_step",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_gap_oracle_state(pipe) -> None:
    """Reset per-trajectory state before each new pipe(...) call."""
    tr = pipe.transformer
    if hasattr(tr, "cnt"):
        tr.cnt = 0
    if hasattr(tr, "residual_history"):
        tr.residual_history = {}
    if hasattr(tr, "last_activation_step"):
        tr.last_activation_step = -1


# ----------------------------------------------------------------------------
# Pair parsing
# ----------------------------------------------------------------------------
def parse_pairs(spec: str, num_steps: int) -> List[Tuple[int, int]]:
    """Parse '--pairs a:k,a:k,...' into a sorted, de-duplicated list.

    Each 'a:k' is a gap-oracle pair: freeze history at a, cache step k.
    Requires 0 <= a < k <= num_steps-1.
    """
    out: List[Tuple[int, int]] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"bad pair token {tok!r}, expected 'a:k'")
        a_s, k_s = tok.split(":", 1)
        a, k = int(a_s), int(k_s)
        if not (0 <= a < k <= int(num_steps) - 1):
            raise ValueError(
                f"pair (a={a}, k={k}) invalid for num_steps={num_steps} "
                f"(need 0<=a<k<=N-1)"
            )
        out.append((a, k))
    if not out:
        raise ValueError("no pairs parsed from --pairs")
    return sorted(set(out))


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gap single-step oracle (Oracle A): R_{a->k}^oracle on FLUX."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--pairs", required=True,
                   help="comma-separated 'a:k' gap-oracle pairs (freeze "
                        "history at a, cache step k). e.g. '3:7,8:24,18:34'.")
    p.add_argument("--cache_mode", choices=list(CACHE_MODES), default="seacache",
                   help="predictor that defines the cache event at step k.")
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
                   help="Skip prompts whose manifest.json marks complete.")
    p.add_argument("--hicache_max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--taylorseer_max_order", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    N = int(args.num_steps)
    pairs = parse_pairs(args.pairs, N)

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
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded in {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts "
          f"(global {start}..{end - 1}); cache_mode={args.cache_mode}; "
          f"{len(pairs)} pairs", flush=True)

    H = (args.height // 16) * 16
    W = (args.width // 16) * 16
    per_prompt_records = []

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"

        if args.resume and manifest_path.is_file():
            try:
                if json.loads(manifest_path.read_text()).get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} complete, skip",
                          flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        prompt_dir.mkdir(parents=True, exist_ok=True)
        per_image_seed = args.seed + global_idx
        t_p = time.perf_counter()

        # ---- baseline: z_N^full --------------------------------------------
        z_base = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
        _save_latent(z_base, prompt_dir / "baseline.pt")
        _decode_to_pil(pipe, z_base, H, W).save(prompt_dir / "baseline.png")
        z_base_f = z_base.float()

        # ---- gap-oracle sweep over (a,k) pairs -----------------------------
        per_pair: List[Dict[str, Any]] = []
        for (a, k) in pairs:
            teardown = install_gap_oracle(
                pipe,
                freeze_step=a,
                cache_step=k,
                num_steps=N,
                cache_mode=str(args.cache_mode),
                hicache_max_order=int(args.hicache_max_order),
                hicache_sigma=float(args.hicache_sigma),
                taylorseer_max_order=int(args.taylorseer_max_order),
            )
            try:
                reset_gap_oracle_state(pipe)
                z_ak = _run_one_pipe_call(pipe, prompt, per_image_seed, args)
            finally:
                teardown()
            r_l2 = float((z_ak.float() - z_base_f).norm().item())
            tag = f"gap_a{a:02d}_k{k:02d}"
            _save_latent(z_ak, prompt_dir / f"{tag}.pt")
            _decode_to_pil(pipe, z_ak, H, W).save(prompt_dir / f"{tag}.png")
            per_pair.append({
                "a": a, "k": k, "g": k - a,
                "R_latent_l2": r_l2,
                "latent_file": f"{tag}.pt",
                "image_file": f"{tag}.png",
            })

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "gap_oracle",
            "prompt_idx": global_idx,
            "prompt": prompt,
            "seed": int(per_image_seed),
            "num_steps": N,
            "cache_mode": str(args.cache_mode),
            "hicache_max_order": int(args.hicache_max_order),
            "hicache_sigma": float(args.hicache_sigma),
            "taylorseer_max_order": int(args.taylorseer_max_order),
            "n_pairs": len(per_pair),
            "per_pair": per_pair,
            "files": {"baseline_latent": "baseline.pt",
                      "baseline_image": "baseline.png"},
            "wall_seconds": wall,
            "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt_records.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)})", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing_path.write_text(json.dumps({
        "experiment": "gap_oracle",
        "cache_mode": str(args.cache_mode),
        "num_steps": N,
        "n_pairs": len(pairs),
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
