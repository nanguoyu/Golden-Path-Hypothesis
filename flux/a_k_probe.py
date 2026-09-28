#!/usr/bin/env python3
"""H3 prerequisite: per-(prompt, k, method) future amplification A_k^dir on FLUX.

Per research plan §9.4:
    A_k^dir = || z_N' - z_N ||_2 / (|| eta_k ||_2 + epsilon)

where  eta_k = H_k * delta_v_k  is the latent-space perturbation that a
single-step cache event at step k would inject into z_{k+1}:
  - delta_v_k = G_k(F_k + r_k) - G_k(F_k)   (velocity error, in S_k probe)
  - H_k = sigmas[k+1] - sigmas[k]            (signed solver step)
  - r_k = method_predicted_residual_k - actual_residual_k  (plan §9.1 sign:
          F + r_k = ori + actual + (pred - actual) = ori + pred = F_cached,
          so delta_v = v_cached - v_baseline matches the real cache event)

We compute a SHORT-HORIZON proxy per §9.4 too:
    A_k^(m) = || z_{k+1+m}' - z_{k+1+m} || / (|| z_{k+1}' - z_{k+1} || + eps)

(Identical to A_k^dir when m = N-1-k, i.e. propagating to the end.)

This probe is a strict superset of flux/s_k_probe.py: it instruments the
baseline pass to collect F_k, P_k, S_k AND delta_v tensors, then runs
partial denoise trajectories from z_{k+1} + eta_k for m steps per
(k, method) combination.

Cost per prompt for default --a_k_horizon 5:
  - Pass 1 (baseline): N forwards
  - Pass 2 (propagation): 3 methods x sum_{k=1..N-1} min(m, N-1-k) forwards
    ~ 3 x 49 x 5 = 735 forwards
  - Total: 50 + 735 = 785 forwards per prompt
  - 4 H100 sharded n=100: ~1h wall (vs oracle ~3h, ~3.2x cheaper)

For --a_k_horizon 0 ("full"), each propagation runs to z_N:
  - 3 x sum_{k=1..N-1} (N-1-k) = 3 x sum_{j=0..48} j = 3 x 1176 = 3528
  - Total: 50 + 3528 = 3578 per prompt; ~5h on 4 H100.

Output per prompt:
    output_dir/prompt_XXXXX/a_k_metrics.json
    {
      "prompt_idx": int, "prompt": str, "seed": int, "num_steps": int,
      "horizon_m": int,                              # 0 = "full" (propagate to z_N)
      "per_k": [
        {"k": 0..N-1,
         "r_actual_norm": float,                     # baseline residual
         "z_at_kp1_norm": float,                     # |z_{k+1}|
         "p_seacache_abs": ..., "p_hicache_abs": ..., "p_taylorseer_abs": ...,
         "s_k_seacache_dir": ..., "s_k_hicache_dir": ..., "s_k_taylorseer_dir": ...,
         "delta_v_seacache_norm": ..., ...,
         "eta_k_seacache_norm": ..., ...,
         "horizon_used_m": int,                      # actual m used (clipped at trajectory end)
         "a_k_seacache_dir": ..., "a_k_hicache_dir": ..., "a_k_taylorseer_dir": ...
        }, ...
      ],
      "config": {...},
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

from lib.gates import rel_l1  # noqa: E402
from lib.hermite import hermite_update, hicache_predict  # noqa: E402
from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.taylor import taylor_predict  # noqa: E402
from lib.teacache_coeffs import get_coeffs  # noqa: E402

logger = logging.get_logger(__name__)

EPS = 1e-12
METHODS = ("seacache", "hicache", "taylorseer")


# ----------------------------------------------------------------------------
# Instrumented forward (Pass 1 only): records F_k, P_k, S_k, delta_v per
# method as tensors. When self.in_pass_1 = False, runs vanilla forward.
# ----------------------------------------------------------------------------
def _a_k_probe_forward(
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

    instrumenting = bool(getattr(self, "in_pass_1", False) and getattr(self, "enable_a_k_probe", False))

    # TeaCache gate (Pass 1 only, for completeness with prior probes)
    teacache_gate_val: Optional[float] = None
    if instrumenting:
        first_block = self.transformer_blocks[0]
        modulated_inp, _, _, _, _ = first_block.norm1(hidden_states, emb=temb)
        prev_mod = self._ak_prev_modulated_input
        if prev_mod is not None:
            d = rel_l1(modulated_inp, prev_mod)
            teacache_gate_val = float(self._ak_teacache_rescale(d))
        self._ak_prev_modulated_input = modulated_inp.detach()

    # Snapshot predictor state BEFORE running blocks (uses past activations)
    history: Dict[int, torch.Tensor] = self._ak_history if instrumenting else {}
    pred_sea: Optional[torch.Tensor] = history.get(0) if history else None
    pred_hi: Optional[torch.Tensor] = (
        hicache_predict(history, step_offset=1,
                        sigma=float(self._ak_hicache_sigma),
                        max_order=int(self._ak_hicache_max_order))
        if (instrumenting and history) else None
    )
    pred_ts: Optional[torch.Tensor] = (
        taylor_predict(history, step_offset=1,
                       max_order=int(self._ak_taylorseer_max_order))
        if (instrumenting and history) else None
    )

    # Full forward through all blocks
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

    F_k = hidden_states
    actual_residual = F_k - ori_hidden_states

    # v_baseline = G_k(F_k); same expression as the original forward's tail.
    h_normed_base = self.norm_out(F_k, temb)
    v_baseline = self.proj_out(h_normed_base)

    if instrumenting:
        r_actual_norm = float(actual_residual.detach().to(torch.float32).norm().item())

        # delta_v per method: G_k(F_k + r_k) - G_k(F_k), kept as a TENSOR
        # (CPU bf16) for use in Pass 2 propagation. Norms recorded as scalars
        # for the JSON.
        record: Dict[str, Any] = {
            "k": int(self.cnt),
            "r_actual_norm": r_actual_norm,
            "p_teacache_gate": teacache_gate_val,
        }
        delta_v_for_pass2: Dict[str, Optional[torch.Tensor]] = {}

        for method_name, pred in (("seacache", pred_sea),
                                  ("hicache", pred_hi),
                                  ("taylorseer", pred_ts)):
            if pred is None:
                record[f"p_{method_name}_abs"] = None
                record[f"p_{method_name}_rel"] = None
                record[f"s_k_{method_name}_dir"] = None
                record[f"delta_v_{method_name}_norm"] = None
                delta_v_for_pass2[method_name] = None
                continue

            # Cache prediction error VECTOR (plan §9.1 convention):
            #   r_k = F_hat - F = pred - actual
            # so F + r_k = ori + actual + (pred - actual) = ori + pred = F_cached
            # and delta_v = v_cached - v_baseline (real cache velocity error).
            # eta_k = H_k * delta_v injected at z_{k+1} then exactly
            # reproduces the cache event's latent perturbation.
            r_k = pred - actual_residual
            r_k_norm = float(r_k.detach().to(torch.float32).norm().item())
            record[f"p_{method_name}_abs"] = r_k_norm
            record[f"p_{method_name}_rel"] = (
                r_k_norm / r_actual_norm if r_actual_norm > 0 else float("nan")
            )

            F_k_perturbed = F_k + r_k  # = F_cached
            h_normed_pert = self.norm_out(F_k_perturbed, temb)
            v_perturbed = self.proj_out(h_normed_pert)
            delta_v = v_perturbed - v_baseline  # = v_cached - v_baseline
            delta_v_norm = float(delta_v.detach().to(torch.float32).norm().item())
            record[f"s_k_{method_name}_dir"] = delta_v_norm / (r_k_norm + EPS)
            record[f"delta_v_{method_name}_norm"] = delta_v_norm
            # Stash delta_v tensor on CPU in bf16 for Pass 2 (real cache direction).
            delta_v_for_pass2[method_name] = delta_v.detach().to("cpu", dtype=torch.bfloat16)

        self._ak_per_k.append(record)
        self._ak_delta_v.append(delta_v_for_pass2)

        # Update Hermite history for next step's predictors
        self._ak_history = hermite_update(
            history, actual_residual.detach(), step_gap=1,
            max_order=int(self._ak_hicache_max_order),
        )

        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0  # ready for next trajectory

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (v_baseline,)
    return Transformer2DModelOutput(sample=v_baseline)


# ----------------------------------------------------------------------------
# install / teardown / reset
# ----------------------------------------------------------------------------
def install_a_k_probe(
    pipe,
    *,
    num_steps: int,
    hicache_max_order: int = 2,
    hicache_sigma: float = 0.5,
    taylorseer_max_order: int = 1,
    teacache_backbone: str = "flux",
) -> Callable[[], None]:
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _a_k_probe_forward

    rescale_fn = np.poly1d(list(get_coeffs(teacache_backbone)))

    tr = pipe.transformer
    tr.enable_a_k_probe = True
    tr.in_pass_1 = False  # explicit; switched on/off around baseline pass
    tr.num_steps = int(num_steps)
    tr.cnt = 0
    tr._ak_history = {}
    tr._ak_prev_modulated_input = None
    tr._ak_teacache_rescale = rescale_fn
    tr._ak_hicache_max_order = int(hicache_max_order)
    tr._ak_hicache_sigma = float(hicache_sigma)
    tr._ak_taylorseer_max_order = int(taylorseer_max_order)
    tr._ak_per_k = []
    tr._ak_delta_v = []  # list of dicts {method: tensor or None}

    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_a_k_probe", "in_pass_1", "num_steps", "cnt",
            "_ak_history", "_ak_prev_modulated_input",
            "_ak_teacache_rescale",
            "_ak_hicache_max_order", "_ak_hicache_sigma",
            "_ak_taylorseer_max_order",
            "_ak_per_k", "_ak_delta_v",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def reset_a_k_probe_state(pipe) -> None:
    tr = pipe.transformer
    tr.cnt = 0
    tr._ak_history = {}
    tr._ak_prev_modulated_input = None
    tr._ak_per_k = []
    tr._ak_delta_v = []


# ----------------------------------------------------------------------------
# Custom partial-denoise helpers (replicates FluxPipeline post-encoder loop).
# We hand-roll the loop so we can (a) capture intermediate latents z_{k+1},
# (b) start from an arbitrary (z_init, scheduler_step_idx) in Pass 2.
# ----------------------------------------------------------------------------
def _calculate_shift(image_seq_len: int,
                     base_seq_len: int, max_seq_len: int,
                     base_shift: float, max_shift: float) -> float:
    """Mirrors diffusers FluxPipeline.calculate_shift.

    All shift parameters are MANDATORY — pass the real values from
    pipe.scheduler.config (NOT hardcoded). Older diffusers used
    max_shift=1.16 as a hardcoded fallback; FLUX.1-dev's scheduler config
    actually carries max_shift=1.15, and the ~1% mu difference compounds
    over 50 sampler steps into a ~19% z_N drift vs pipe(). Caught by
    Check 1 v2 diagnostic (job 50175).
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _shift_args_from_scheduler(scheduler) -> Dict[str, float]:
    """Read FLUX time-shift parameters from scheduler.config with fallbacks
    matching diffusers FluxPipeline."""
    cfg = scheduler.config
    return {
        "base_seq_len": cfg.get("base_image_seq_len", 256),
        "max_seq_len": cfg.get("max_image_seq_len", 4096),
        "base_shift": cfg.get("base_shift", 0.5),
        "max_shift": cfg.get("max_shift", 1.16),
    }


def _encode_and_prepare(pipe, prompt: str, seed: int, args) -> Dict[str, Any]:
    """Encode prompt + prepare initial latents + set scheduler timesteps.

    `pipe.encode_prompt` runs the T5 + CLIP text encoders. They have grads
    disabled internally only by virtue of FluxPipeline.__call__ being
    wrapped in @torch.no_grad — we replicate that here so the encoder
    forward doesn't accumulate a graph. `set_timesteps` resets
    `scheduler._step_index` to None as a side effect, so the first call
    to `scheduler.step` after this auto-inits the index.
    """
    device = pipe.device
    H = (args.height // 16) * 16
    W = (args.width // 16) * 16
    is_dev = args.model_name == "flux-dev"

    with torch.no_grad():
        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=(512 if is_dev else 256),
        )

        generator = torch.Generator(device=device).manual_seed(int(seed))
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=H,
            width=W,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

    sigmas_init = np.linspace(1.0, 1.0 / args.num_steps, args.num_steps).tolist()
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(image_seq_len, **_shift_args_from_scheduler(pipe.scheduler))
    pipe.scheduler.set_timesteps(sigmas=sigmas_init, mu=mu, device=device)

    if is_dev:
        guidance = torch.full([1], float(args.guidance), device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    return {
        "device": device,
        "latents_init": latents,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
        "latent_image_ids": latent_image_ids,
        "guidance": guidance,
        "timesteps": pipe.scheduler.timesteps,
        "sigmas": pipe.scheduler.sigmas.detach().cpu().numpy().tolist(),  # length N+1
        "H": H, "W": W,
    }


def _denoise_step(transformer, scheduler, latents: torch.Tensor,
                  timestep: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
    """Single (transformer + scheduler) step. Mirrors FluxPipeline.__call__:
    transformer expects timestep as a 1-d (batch_size,) tensor cast to the
    latent dtype (the AdaLayerNorm time embedding's `get_timestep_embedding`
    asserts `len(timesteps.shape) == 1`); scheduler.step takes the original
    scalar.
    """
    t_expanded = timestep.expand(latents.shape[0]).to(latents.dtype)
    noise_pred = transformer(
        hidden_states=latents,
        timestep=t_expanded / 1000,
        guidance=ctx["guidance"],
        pooled_projections=ctx["pooled_prompt_embeds"],
        encoder_hidden_states=ctx["prompt_embeds"],
        txt_ids=ctx["text_ids"],
        img_ids=ctx["latent_image_ids"],
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    return scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]


def _baseline_pass(pipe, ctx: Dict[str, Any]) -> List[torch.Tensor]:
    """Run full N-step denoise with instrumented forward (Pass 1).

    Returns a list of length N: latents[i] = z_{i+1} (the latent AFTER
    completing scheduler step i, which corresponds to z at the boundary
    between step i and step i+1).

    Wrapped in torch.no_grad() — FluxPipeline.__call__ has the same wrap;
    without it the inference loop accumulates a ~1.25 GB forward graph.
    Scheduler._step_index is None at this point (set_timesteps in
    _encode_and_prepare resets it); the first step() auto-inits it.
    """
    latents = ctx["latents_init"]
    z_after_step: List[torch.Tensor] = []
    with torch.no_grad():
        for i, t in enumerate(ctx["timesteps"]):
            latents = _denoise_step(pipe.transformer, pipe.scheduler, latents, t, ctx)
            z_after_step.append(latents.detach())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return z_after_step


def _partial_denoise(pipe, ctx: Dict[str, Any],
                     z_start: torch.Tensor,
                     start_idx: int, end_idx: int) -> torch.Tensor:
    """Run scheduler steps [start_idx, end_idx) with vanilla forward.

    z_start is the latent BEFORE step start_idx (i.e., the input to step
    start_idx's transformer call). After this function returns, the
    output latent corresponds to z_{end_idx} = "the latent AFTER step
    (end_idx - 1) completes".

    IMPORTANT: FlowMatchEulerDiscreteScheduler.step() depends on
    `scheduler._step_index` to pick (sigmas[idx], sigmas[idx+1]) for the
    Euler update. After Pass 1 baseline this index sits at N, so the
    first step() of Pass 2 would otherwise reach beyond sigmas[N]. We
    set `_step_index = None` here so the first step() call's
    `_init_step_index(timestep)` re-resolves the index by matching
    `timesteps[start_idx]` against the schedule. Subsequent steps in
    this loop advance _step_index normally.
    """
    timesteps = ctx["timesteps"]
    end_idx = min(end_idx, len(timesteps))
    if start_idx >= end_idx:
        return z_start
    pipe.scheduler._step_index = None
    latents = z_start
    with torch.no_grad():
        for i in range(start_idx, end_idx):
            latents = _denoise_step(pipe.transformer, pipe.scheduler, latents, timesteps[i], ctx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return latents


# ----------------------------------------------------------------------------
# Per-prompt A_k probe
# ----------------------------------------------------------------------------
def probe_one_prompt(pipe, prompt: str, seed: int, args) -> Dict[str, Any]:
    """Run Pass 1 (instrumented baseline) + Pass 2 (propagation) for one prompt.

    Returns the per-prompt metrics dict (per_k records + config + meta).

    Both passes are inference-only; wrapping in torch.no_grad() at the
    outermost level avoids any accidental grad tracking (e.g. through
    `.norm()` on tensors that inherited requires_grad). The instrumented
    forward also tries to `.detach()` delta_v before stashing, but
    no_grad here is a belt-and-braces guard.
    """
    with torch.no_grad():
        ctx = _encode_and_prepare(pipe, prompt, seed, args)
        N = int(args.num_steps)
        device = ctx["device"]
        horizon_m = int(args.a_k_horizon)  # 0 => "to end"

        # ---- Pass 1: instrumented baseline --------------------------------
        reset_a_k_probe_state(pipe)
        pipe.transformer.in_pass_1 = True
        try:
            z_after = _baseline_pass(pipe, ctx)  # length N; z_after[i] = z_{i+1}
        finally:
            pipe.transformer.in_pass_1 = False

        per_k_records: List[Dict[str, Any]] = list(pipe.transformer._ak_per_k)
        delta_v_per_step: List[Dict[str, Optional[torch.Tensor]]] = list(pipe.transformer._ak_delta_v)
        if len(per_k_records) != N or len(delta_v_per_step) != N:
            raise RuntimeError(
                f"Pass 1 collected {len(per_k_records)} per_k records and "
                f"{len(delta_v_per_step)} delta_v entries, expected {N} of each."
            )

        # ---- Pass 2: per (k, method) short-horizon propagation ------------
        sigmas = ctx["sigmas"]  # length N+1

        for k in range(N):
            rec = per_k_records[k]
            # z_{k+1} is the latent AFTER step k completes  =  z_after[k]
            z_kp1 = z_after[k].to(device)
            rec["z_at_kp1_norm"] = float(z_kp1.to(torch.float32).norm().item())

            # m used = clipped to remaining trajectory after step k. If
            # horizon_m==0 we propagate all the way to step N-1 (z_N).
            max_m = (N - 1) - k                  # available scheduler steps after step k
            if horizon_m <= 0:
                m_used = max_m
            else:
                m_used = min(horizon_m, max_m)
            rec["horizon_used_m"] = int(m_used)

            if m_used < 1:
                # No future to propagate (k = N-1).
                for M in METHODS:
                    rec[f"eta_k_{M}_norm"] = None
                    rec[f"a_k_{M}_dir"] = None
                continue

            H_k = float(sigmas[k + 1] - sigmas[k])  # signed step size
            # z_baseline at the end of the propagation: z_after[k + m_used]
            z_end_base = z_after[k + m_used].to(device)

            for M in METHODS:
                dv_cpu = delta_v_per_step[k].get(M)
                if dv_cpu is None:
                    rec[f"eta_k_{M}_norm"] = None
                    rec[f"a_k_{M}_dir"] = None
                    continue
                dv = dv_cpu.to(device=device, dtype=z_kp1.dtype, non_blocking=True)
                eta_k = H_k * dv  # signed latent-space perturbation
                eta_k_norm = float(eta_k.to(torch.float32).norm().item())
                rec[f"eta_k_{M}_norm"] = eta_k_norm

                z_perturbed = z_kp1 + eta_k
                # Run scheduler steps [k+1, k+1+m_used)
                z_end_pert = _partial_denoise(pipe, ctx, z_perturbed, k + 1, k + 1 + m_used)
                diff_norm = float((z_end_pert - z_end_base).to(torch.float32).norm().item())
                rec[f"a_k_{M}_dir"] = diff_norm / (eta_k_norm + EPS)

        # Free per-prompt GPU + CPU buffers explicitly before next prompt
        del z_after, delta_v_per_step
        pipe.transformer._ak_delta_v = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "per_k": per_k_records,
            "n_steps": N,
            "horizon_m": horizon_m,
        }


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="H3 prerequisite A_k probe: per-(prompt, k, method) future "
                    "amplification on FLUX. Also dumps P_k and S_k."
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
    p.add_argument("--a_k_horizon", type=int, default=5,
                   help="m for A_k^(m) short-horizon proxy. 0 => propagate to z_N "
                        "(full A_k^dir per plan §9.4). Default 5.")
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
    print(f"[shard {args.shard_idx}] a_k_horizon={args.a_k_horizon} "
          f"({'propagate to z_N' if args.a_k_horizon <= 0 else f'short-horizon m={args.a_k_horizon}'})",
          flush=True)

    teardown = install_a_k_probe(
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
        "a_k_horizon": int(args.a_k_horizon),
    }

    try:
        for local_idx, prompt in enumerate(shard_prompts):
            global_idx = start + local_idx
            prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
            metrics_path = prompt_dir / "a_k_metrics.json"

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

            result = probe_one_prompt(pipe, prompt, per_image_seed, args)

            t_done = time.perf_counter()
            prompt_seconds = float(t_done - t_prompt)

            metrics_data = {
                "prompt_idx": global_idx,
                "prompt": prompt,
                "seed": int(per_image_seed),
                "num_steps": int(args.num_steps),
                "horizon_m": result["horizon_m"],
                "per_k": result["per_k"],
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
        "experiment": "a_k_probe",
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
