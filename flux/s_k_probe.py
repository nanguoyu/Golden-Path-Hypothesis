#!/usr/bin/env python3
"""H3 prerequisite: per-(prompt, k, method) directional output sensitivity S_k^dir on FLUX.

Per research plan §9.2:
    S_{l,k}^dir = || G_{l,k}(F_{l,k} + r_{l,k}) - G_{l,k}(F_{l,k}) ||_2
                  -----------------------------------------------------
                            || r_{l,k} ||_2 + epsilon

with the project's coarse-grain choice of cache layer (l = whole-transformer
residual):
  - F_k = hidden_states after all transformer / single_transformer blocks at
          step k = ori_hidden_states + actual_residual_k
  - G_k = norm_out + proj_out  (the operator FLUX uses to map post-block
          features to the output velocity that the scheduler integrates)
  - r_k = method's CACHE PREDICTION ERROR vector at step k (plan §9.1)
        = method_predicted_residual_k - actual_residual_k
        (the feature-space offset the cache injects: F_cached = F + r_k.
         With this convention G(F + r_k) = G(F_cached) = v_cached, so
         delta_v = G(F + r_k) - G(F) is the real cache's velocity error.)

S_k^dir measures: along the true cache-error direction r_k, how much does
G_k amplify the feature error into a velocity error? It is required to build
Score_3 = P_k * S_k^dir * Q_k for H3 in the score_compare analysis.

This probe is a strict SUPERSET of flux/p_k_probe.py: it runs the same full
baseline trajectory once per prompt and additionally computes 3 extra G_k
forward calls per step (one per method) to produce v_perturbed. P_k values
are emitted as a byproduct so downstream analysis can use s_k_metrics.json
alone if it has been generated.

Extra cost per step: ~3 lightweight (norm_out + proj_out) calls
(<= 15% step time on H100 bf16). n=100 wall: ~5-6 min on 4 H100.

Output per prompt:
    output_dir/prompt_XXXXX/s_k_metrics.json
    {
      "prompt_idx": int, "prompt": str, "seed": int, "num_steps": int,
      "per_k": [
        {"k": 0,
         "r_actual_norm": float,
         "p_seacache_abs": null, "p_seacache_rel": null,
         "p_hicache_abs":  null, "p_hicache_rel":  null,
         "p_taylorseer_abs": null, "p_taylorseer_rel": null,
         "p_teacache_gate": null,
         "s_k_seacache_dir": null, "s_k_hicache_dir": null,
         "s_k_taylorseer_dir": null,
         "v_baseline_norm": float},
        ...
      ],
      "config": {hicache_max_order, hicache_sigma, taylorseer_max_order,
                 teacache_backbone},
      "complete": true
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

EPS = 1e-12


# ----------------------------------------------------------------------------
# Probe forward: full baseline + compute v_baseline + 3 perturbed v's per step.
# Sets state on `self` (the FluxTransformer2DModel instance).
# ----------------------------------------------------------------------------
def _s_k_probe_forward(
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

    # ---- TeaCache gate signal (per-step, before block compute) -------------
    teacache_gate_val: Optional[float] = None
    if getattr(self, "enable_sk_probe", False):
        first_block = self.transformer_blocks[0]
        modulated_inp, _, _, _, _ = first_block.norm1(hidden_states, emb=temb)
        prev_mod = self._sk_prev_modulated_input
        if prev_mod is not None:
            d = rel_l1(modulated_inp, prev_mod)
            teacache_gate_val = float(self._sk_teacache_rescale(d))
        self._sk_prev_modulated_input = modulated_inp.detach()

    # ---- Snapshot predictor history from PAST steps (before this step's update)
    history: Dict[int, torch.Tensor] = self._sk_history
    pred_sea: Optional[torch.Tensor] = history.get(0) if history else None
    pred_hi: Optional[torch.Tensor] = (
        hicache_predict(history, step_offset=1,
                        sigma=float(self._sk_hicache_sigma),
                        max_order=int(self._sk_hicache_max_order))
        if history else None
    )
    pred_ts: Optional[torch.Tensor] = (
        taylor_predict(history, step_offset=1,
                       max_order=int(self._sk_taylorseer_max_order))
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

    # ---- F_k (post-blocks) = ori_hidden_states + actual_residual -----------
    F_k = hidden_states
    actual_residual = F_k - ori_hidden_states  # = r_actual_k

    # ---- v_baseline = G_k(F_k) = proj_out(norm_out(F_k, temb)) -------------
    # This is the value the original forward returns; we compute it once and
    # reuse it both as the trajectory output AND as the S_k reference.
    h_normed_base = self.norm_out(F_k, temb)
    v_baseline = self.proj_out(h_normed_base)

    # ---- Per-step record: P_k + S_k_dir for 3 methods ----------------------
    if getattr(self, "enable_sk_probe", False):
        r_actual_norm = float(actual_residual.detach().to(torch.float32).norm().item())
        v_baseline_norm = float(v_baseline.detach().to(torch.float32).norm().item())

        record: Dict[str, Any] = {
            "k": int(self.cnt),
            "r_actual_norm": r_actual_norm,
            "v_baseline_norm": v_baseline_norm,
            "p_teacache_gate": teacache_gate_val,
        }

        # All 3 predictor variants share the same finite-difference history
        # built at max_order=hicache_max_order; taylor_predict and SeaCache
        # downsample via min(max_order, order_avail).
        method_preds = [
            ("seacache", pred_sea),
            ("hicache", pred_hi),
            ("taylorseer", pred_ts),
        ]
        for method_name, pred in method_preds:
            if pred is None:
                record[f"p_{method_name}_abs"] = None
                record[f"p_{method_name}_rel"] = None
                record[f"s_k_{method_name}_dir"] = None
                continue
            # Cache prediction error VECTOR, plan §9.1 convention:
            # r_k = F_hat - F = pred - actual (the offset the method INJECTS
            # into the feature). Then F + r_k = ori + actual + (pred - actual)
            # = ori + pred = F_cached, the real cached feature. With the
            # opposite sign convention the probe would evaluate G at a
            # "doubly-perturbed" point F + (actual - pred) instead of at the
            # cached feature; norms agree to first order but exact alignment
            # with the real cache event requires this convention.
            r_k = pred - actual_residual
            r_k_norm = float(r_k.detach().to(torch.float32).norm().item())
            record[f"p_{method_name}_abs"] = r_k_norm
            record[f"p_{method_name}_rel"] = (
                r_k_norm / r_actual_norm if r_actual_norm > 0 else float("nan")
            )

            # S_k^dir: forward-difference probe of G_k along direction r_k.
            # F_k + r_k now equals the cached feature F_hat = ori + pred,
            # so v_perturbed = G(F_cached) = v_cached, and
            # delta_v = v_cached - v_baseline is the real cache velocity error.
            #   S_k^dir = ||delta_v|| / (||r_k|| + eps)
            F_k_perturbed = F_k + r_k
            h_normed_pert = self.norm_out(F_k_perturbed, temb)
            v_perturbed = self.proj_out(h_normed_pert)
            delta_v_norm = float((v_perturbed - v_baseline).detach()
                                 .to(torch.float32).norm().item())
            record[f"s_k_{method_name}_dir"] = delta_v_norm / (r_k_norm + EPS)

        self._sk_per_k.append(record)

        # Update Delta^k history with this step's actual residual (step_gap=1
        # because baseline runs full at every step). See p_k_probe docstring
        # for the deliberate omission of HiCache/TaylorSeer warmup suppression.
        self._sk_history = hermite_update(
            history, actual_residual.detach(), step_gap=1,
            max_order=int(self._sk_hicache_max_order),
        )

        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for the next trajectory

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (v_baseline,)
    return Transformer2DModelOutput(sample=v_baseline)


# ----------------------------------------------------------------------------
# install / teardown / reset
# ----------------------------------------------------------------------------
def install_sk_probe(
    pipe,
    *,
    num_steps: int,
    hicache_max_order: int = 2,
    hicache_sigma: float = 0.5,
    taylorseer_max_order: int = 1,
    teacache_backbone: str = "flux",
) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward to compute per-step P_k + S_k^dir."""
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _s_k_probe_forward

    rescale_fn = np.poly1d(list(get_coeffs(teacache_backbone)))

    tr = pipe.transformer
    tr.enable_sk_probe = True
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr._sk_history = {}
    tr._sk_prev_modulated_input = None
    tr._sk_teacache_rescale = rescale_fn
    tr._sk_hicache_max_order = int(hicache_max_order)
    tr._sk_hicache_sigma = float(hicache_sigma)
    tr._sk_taylorseer_max_order = int(taylorseer_max_order)
    tr._sk_per_k = []

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_sk_probe", "num_steps", "cnt",
            "_sk_history", "_sk_prev_modulated_input",
            "_sk_teacache_rescale",
            "_sk_hicache_max_order", "_sk_hicache_sigma",
            "_sk_taylorseer_max_order", "_sk_per_k",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_sk_probe_state(pipe) -> None:
    """Reset per-trajectory state before each new pipe(...) call."""
    tr = pipe.transformer
    tr.cnt = 0
    tr._sk_history = {}
    tr._sk_prev_modulated_input = None
    tr._sk_per_k = []


# ----------------------------------------------------------------------------
# Pipe call helper (no decode — we don't need images for S_k)
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
        description="H3 prerequisite S_k probe: per-(prompt, k, method) "
                    "directional output sensitivity on FLUX. Also dumps P_k."
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
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hicache_max_order", type=int, default=2)
    p.add_argument("--hicache_sigma", type=float, default=0.5)
    p.add_argument("--taylorseer_max_order", type=int, default=1)
    p.add_argument("--teacache_backbone", default="flux")
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

    teardown = install_sk_probe(
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
            metrics_path = prompt_dir / "s_k_metrics.json"

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

            reset_sk_probe_state(pipe)
            _run_one_pipe_call(pipe, prompt, per_image_seed, args)

            per_k = list(pipe.transformer._sk_per_k)
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
        "experiment": "s_k_probe",
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
