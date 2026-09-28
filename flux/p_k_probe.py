#!/usr/bin/env python3
"""H2 probe: per-(prompt, k) local prediction error P_k for 4 methods on FLUX.

Hypothesis H2 in docs/my_research_plan.md: P_k = |r_k - r_k_pred| (local
feature prediction error) is insufficient to predict R_k^oracle. To test this
we need P_k for each of the 4 baseline methods at every (prompt, k), so the
post-hoc correlation step can compare P_k against R_k^oracle (from
flux/oracle_runner.py output) under multiple correlation metrics.

Approach: one full baseline trajectory per prompt; at every step k compute
  - actual_residual_k = hidden_states_post_blocks - hidden_states_pre_blocks
  - prediction from each method using past actual residuals as history:
      * SeaCache         (zero-order):  pred = r_{k-1}
      * HiCache          (Hermite O=2): pred = hicache_predict(history, x=1, sigma=0.5, O=2)
      * TaylorSeer       (Taylor O=1):  pred = taylor_predict(history, x=1, O=1)
  - P_k_method = ||actual - pred||_2  and the relative form  P_k / ||actual||_2

Also records TeaCache's gate signal (poly-rescaled rel_L1 between modulated
inputs at k and k-1). TeaCache's effective predictor is zero-order so its
"P_k" in the prediction-error sense equals SeaCache's; the gate signal is
included as the alternative scalar TeaCache itself uses to decide caching.

Cost: 1 full forward sweep per prompt -> ~10s on H100. 100 prompts on
4 GPUs: ~4-5 minutes (vs oracle's ~3.5h).

## Design choices vs paper-fidelity install (deliberate)

  1. **No warmup-suppression on Delta^k history.** flux/hicache.py and
     flux/taylorseer.py install code uses `effective_max_order = 0 if cnt <
     first_enhance else max_order` during update; my probe always uses full
     max_order from step 0. Reason: the warmup guard is an engineering
     heuristic, not a property of the predictor math. We want to give each
     predictor the BEST-CASE history so H2 is testing the math, not the
     init policy. Early k's (history too short) automatically degenerate to
     the lower-order term via `min(max_order, order_avail)`.

  2. **TaylorSeer default O=1 here, not flux/taylorseer.py install's O=2.**
     O=1 matches CLAUDE.md project convention (SeaCache paper FLUX
     comparison). Pass `--taylorseer_max_order 2` to test TaylorSeer's own
     paper default. The install signature's O=2 default is independent.

  3. **TeaCache "p_teacache_gate" is NOT a residual prediction error.**
     TeaCache's effective skip-step predictor is zero-order (reuse previous
     residual), identical to SeaCache, so its residual-prediction-error
     P_k_TeaCache = P_k_SeaCache. The unique TeaCache signal recorded here
     is its GATE input: poly_flux(rel_L1(modulated_inp_k, modulated_inp_{k-1})),
     i.e., the per-step contribution to the accumulator that drives the
     skip decision. For H2's "is P_k a sufficient predictor" question this
     is the right scalar for TeaCache (it's the scalar TeaCache itself uses
     to decide caching at step k), but readers should not confuse it with
     ‖actual_r - pred_r‖ — it has different units and dimensionless scale.

Output per prompt:
    output_dir/prompt_XXXXX/p_k_metrics.json
    {
      "prompt_idx": int, "prompt": str, "seed": int, "num_steps": int,
      "per_k": [
        {"k": 0, "r_actual_norm": float,
         "p_seacache_abs": null, "p_seacache_rel": null,
         "p_hicache_abs":  null, "p_hicache_rel":  null,
         "p_taylorseer_abs": null, "p_taylorseer_rel": null,
         "p_teacache_gate": null},
        {"k": 1, ...},
        ...
      ],
      "config": {hicache_max_order, hicache_sigma, taylorseer_max_order,
                 teacache_backbone},
      "wall_seconds": float, "complete": true
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

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

from lib.gates import rel_l1  # noqa: E402
from lib.hermite import hermite_update, hicache_predict  # noqa: E402
from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.taylor import taylor_predict  # noqa: E402
from lib.teacache_coeffs import get_coeffs  # noqa: E402

logger = logging.get_logger(__name__)


# ----------------------------------------------------------------------------
# Probe forward: runs full computation at every step + computes per-step P_k
# for the 3 distinct predictors (zero-order, Hermite-O2, Taylor-O1) and the
# TeaCache poly-rescaled gate signal. State stored on `self` (the transformer).
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
    """Replacement for FluxTransformer2DModel.forward that runs full at every
    step AND records per-step prediction errors for 3 cache methods."""
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

    # ---- TeaCache gate signal (computed BEFORE block compute) --------------
    # Needs first_block.norm1(input, emb=temb) to obtain modulated input;
    # same as flux/teacache.py upstream.
    teacache_gate_val: Optional[float] = None
    if getattr(self, "enable_probe", False):
        first_block = self.transformer_blocks[0]
        modulated_inp, _, _, _, _ = first_block.norm1(hidden_states, emb=temb)
        prev_mod = self._probe_prev_modulated_input
        if prev_mod is not None:
            d = rel_l1(modulated_inp, prev_mod)
            teacache_gate_val = float(self._probe_teacache_rescale(d))
        # else: leave as None (k=0)
        # Update prev for next step
        self._probe_prev_modulated_input = modulated_inp.detach()

    # ---- Snapshot predictor state from past steps (BEFORE running this step
    #      and BEFORE updating history). All 3 predictors share one dict.
    history: Dict[int, torch.Tensor] = self._probe_history
    pred_sea: Optional[torch.Tensor] = history.get(0) if history else None
    pred_hi: Optional[torch.Tensor] = (
        hicache_predict(history, step_offset=1, sigma=float(self._probe_hicache_sigma),
                        max_order=int(self._probe_hicache_max_order))
        if history else None
    )
    pred_ts: Optional[torch.Tensor] = (
        taylor_predict(history, step_offset=1,
                       max_order=int(self._probe_taylorseer_max_order))
        if history else None
    )

    # ---- Full forward through all transformer blocks -----------------------
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

    # ---- Compute actual residual + per-step metrics ------------------------
    if getattr(self, "enable_probe", False):
        actual_residual = hidden_states - ori_hidden_states
        # Norms in fp32 for numerical stability of small residuals
        r_actual_norm = float(actual_residual.detach().to(torch.float32).norm().item())

        def _err(pred: Optional[torch.Tensor]) -> tuple[Optional[float], Optional[float]]:
            if pred is None:
                return None, None
            diff = (actual_residual - pred).detach().to(torch.float32)
            abs_norm = float(diff.norm().item())
            rel_norm = abs_norm / r_actual_norm if r_actual_norm > 0 else float("nan")
            return abs_norm, rel_norm

        p_sea_abs, p_sea_rel = _err(pred_sea)
        p_hi_abs, p_hi_rel = _err(pred_hi)
        p_ts_abs, p_ts_rel = _err(pred_ts)

        self._probe_per_k.append({
            "k": int(self.cnt),
            "r_actual_norm": r_actual_norm,
            "p_seacache_abs": p_sea_abs,
            "p_seacache_rel": p_sea_rel,
            "p_hicache_abs": p_hi_abs,
            "p_hicache_rel": p_hi_rel,
            "p_taylorseer_abs": p_ts_abs,
            "p_taylorseer_rel": p_ts_rel,
            "p_teacache_gate": teacache_gate_val,
        })

        # Update Delta^k history with this step's actual residual.
        # step_gap = 1 always (baseline runs full at every step).
        self._probe_history = hermite_update(
            history, actual_residual.detach(), step_gap=1,
            max_order=int(self._probe_hicache_max_order),
        )

        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for next trajectory

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
def install_probe(
    pipe,
    *,
    num_steps: int,
    hicache_max_order: int = 2,
    hicache_sigma: float = 0.5,
    taylorseer_max_order: int = 1,
    teacache_backbone: str = "flux",
) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward to record per-step P_k metrics."""
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _probe_forward

    rescale_fn = np.poly1d(list(get_coeffs(teacache_backbone)))

    tr = pipe.transformer
    tr.enable_probe = True
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr._probe_history = {}
    tr._probe_prev_modulated_input = None
    tr._probe_teacache_rescale = rescale_fn
    tr._probe_hicache_max_order = int(hicache_max_order)
    tr._probe_hicache_sigma = float(hicache_sigma)
    tr._probe_taylorseer_max_order = int(taylorseer_max_order)
    tr._probe_per_k = []

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_probe", "num_steps", "cnt",
            "_probe_history", "_probe_prev_modulated_input",
            "_probe_teacache_rescale",
            "_probe_hicache_max_order", "_probe_hicache_sigma",
            "_probe_taylorseer_max_order",
            "_probe_per_k",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_probe_state(pipe) -> None:
    """Reset per-trajectory state before each new pipe(...) call."""
    tr = pipe.transformer
    tr.cnt = 0
    tr._probe_history = {}
    tr._probe_prev_modulated_input = None
    tr._probe_per_k = []


# ----------------------------------------------------------------------------
# Pipe call helper (no decode — we don't need images for H2)
# ----------------------------------------------------------------------------
def _run_one_pipe_call(pipe, prompt: str, seed: int, args) -> None:
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    _ = pipe(
        prompt=prompt,
        num_inference_steps=int(args.num_steps),
        guidance_scale=(0.0 if args.model_name == "flux-schnell" else float(args.guidance)),
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        max_sequence_length=(256 if args.model_name == "flux-schnell" else 512),
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="H2 P_k probe: per-(prompt, k) local prediction error for "
                    "4 cache methods on FLUX."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
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
                   help="Skip prompts whose p_k_metrics.json marks complete.")
    p.add_argument("--hicache_max_order", type=int, default=2,
                   help="HiCache Hermite truncation order O (paper default 2).")
    p.add_argument("--hicache_sigma", type=float, default=0.5,
                   help="HiCache dual-scaling factor (paper default 0.5).")
    p.add_argument("--taylorseer_max_order", type=int, default=1,
                   help="TaylorSeer expansion order (project default 1; "
                        "TaylorSeer paper itself uses 2).")
    p.add_argument("--teacache_backbone", default="flux",
                   help="TeaCache polynomial-coeffs key in lib/teacache_coeffs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    prompts_all = read_prompts(args.prompt_file, limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} dtype={args.dtype}", flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in {model_load_end - process_start:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
          f"(global idx {start}..{end - 1})", flush=True)

    # Install probe once for the whole shard — state reset per prompt.
    teardown = install_probe(
        pipe,
        num_steps=int(args.num_steps),
        hicache_max_order=int(args.hicache_max_order),
        hicache_sigma=float(args.hicache_sigma),
        taylorseer_max_order=int(args.taylorseer_max_order),
        teacache_backbone=str(args.teacache_backbone),
    )

    per_prompt_records = []
    config_payload = {
        "hicache_max_order": int(args.hicache_max_order),
        "hicache_sigma": float(args.hicache_sigma),
        "taylorseer_max_order": int(args.taylorseer_max_order),
        "teacache_backbone": str(args.teacache_backbone),
    }

    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
            metrics_path = prompt_dir / "p_k_metrics.json"

            if args.resume and metrics_path.is_file():
                try:
                    m = json.loads(metrics_path.read_text())
                    if m.get("complete", False):
                        print(f"[shard {args.shard_idx}] prompt {global_idx} already complete, skip",
                              flush=True)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass

            prompt_dir.mkdir(parents=True, exist_ok=True)
            per_image_seed = args.seed + global_idx
            t_prompt = time.perf_counter()

            reset_probe_state(pipe)
            _run_one_pipe_call(pipe, prompt, per_image_seed, args)

            per_k = list(pipe.transformer._probe_per_k)
            t_done = time.perf_counter()
            prompt_seconds = float(t_done - t_prompt)

            metrics_data = {
                "prompt_idx": global_idx,
                "prompt": prompt,
                "seed": int(per_image_seed),
                "num_steps": int(args.num_steps),
                "per_k": per_k,
                "config": config_payload,
                "model_id": args.model_id,
                "model_name": args.model_name,
                "dtype": args.dtype,
                "guidance": float(args.guidance),
                "width": (args.width // 16) * 16,
                "height": (args.height // 16) * 16,
                "wall_seconds": prompt_seconds,
                "complete": True,
            }
            metrics_path.write_text(json.dumps(metrics_data, indent=2, ensure_ascii=False))

            per_prompt_records.append({
                "global_idx": global_idx,
                "wall_seconds": prompt_seconds,
            })
            print(f"[shard {args.shard_idx}] prompt {global_idx} done {prompt_seconds:.1f}s "
                  f"({local_idx + 1}/{len(shard_prompts)})", flush=True)
    finally:
        teardown()

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    timing_data = {
        "experiment": "p_k_probe",
        "num_steps": int(args.num_steps),
        "shard_idx": int(args.shard_idx),
        "shard_count": int(args.shard_count),
        "base_seed": int(args.seed),
        "model_id": args.model_id,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "device": device_str,
        "config": config_payload,
        "model_load_s": float(model_load_end - process_start),
        "wallclock_total_s": float(process_end - process_start),
        "n_prompts_in_shard": len(shard_prompts),
        "per_prompt": per_prompt_records,
    }
    timing_path.write_text(json.dumps(timing_data, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}  "
          f"(total wall {process_end - process_start:.1f}s)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
