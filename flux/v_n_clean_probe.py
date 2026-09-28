#!/usr/bin/env python3
"""E0-clean-RMS V_n probe for the method-native SQA experiment plan §6 E0-clean-RMS.

Method-AGNOSTIC calibration of the per-step sensitivity (S_n) and future
amplification (A_n) factors that compose V_n = S_n · Q_n · A_n. Unlike
flux/s_k_probe.py and flux/a_k_probe.py (which perturb along a method's
cache-error direction, plan §6 E0-method-directional), this probe perturbs
along K random fixed-Frobenius-norm directions and reports the RMS
directional gain. Q_n is analytical (lib analysis/extract_q_k.py).

Two production modes + one diagnostic:

  --mode s
      For each (prompt, step n), perturb the whole-transformer residual
      R_n with K_S random directions and measure the resulting velocity
      change. Per plan §6 E0-clean-RMS:
        target_n = R_n = h_out - h_in_post_xembed          (= flux/seacache.py
                                                              previous_residual)
        F_perturbed = F_n + ε · d_k                        (= h_in + R_n + ε·d
                                                              = (h_in + R_n) + ε·d)
        v_perturbed = G_n(F_perturbed) = proj_out(norm_out(F_perturbed, temb))
        v_baseline  = G_n(F_n)
        response_k  = ||v_perturbed - v_baseline||_F / ε
      where ε = ε_rel · ||R_n||_F and d_k has unit Frobenius norm. The
      reported S_n is the RMS over K_S directions:
        S_n = sqrt( mean_k response_k^2 )
      Plan §6 explicitly warns NOT to call this Frobenius/Hutchinson
      unless using the correct unnormalized convention — we call it the
      RMS directional gain.

  --mode a
      For each (prompt, step n), perturb the POST-Euler latent z_{n+1}
      with K_A random directions and run the remaining sampler to z_N.
      Per plan §6 E0-clean-RMS:
        target_n = z_{n+1}
        z_perturbed = z_{n+1} + ε · d_k
        z_N^perturbed = remaining_sampler(z_perturbed, start_idx = n+1)
        response_k = ||z_N^perturbed - z_N||_F / ε
        A_n = sqrt( mean_k response_k^2 )
      The injection point is AFTER step n's Euler update completes, so
      the propagated operator is Φ_{N, n+1}.

  --mode linearity_check
      Before either production mode, sweep ε ∈ {1e-4, 1e-3, 1e-2, 1e-1}
      on 3 prompts × 5 steps × 1 direction (per plan §6 E0-clean-RMS
      §2.1 of the deleted e0_impl spec, the only pinned numeric in this
      plan for ε). Asserts the bottom three ε values agree within ±20 %
      so that ε_rel = 1e-2 is in the linear regime. Mode = s applies
      since the same forward path is used.

  --mode linearity_check_a
      Same ε sweep, but for the A_n post-Euler latent perturbation and suffix
      propagation path. This is separate because S-linearity does not imply
      A-linearity.

Direction generation (deterministic, byte-reproducible):
  Per (prompt_idx, step n, k) we draw a random Gaussian with the shape
  of the perturbation target, then normalize to ||d||_F = 1. Production
  modes use an effective per-prompt seed
  `direction_seed + 100003 * prompt_idx` (S default 0, A default 1), so
  directions are independent across prompts while remaining shard-stable.
  dtype: fp32 for the Gaussian draw + the normalization, cast to bf16 just
  before the injection.

dtype contract (per plan §6 E0-clean-RMS):
  - Model forward: bf16 (locked baseline default).
  - Perturbation construction (direction × ε_rel × ρ): fp32, cast to
    bf16 just before injection.
  - Response measurement: cast both branches' outputs to fp32 before
    subtraction + ||·||_F, so we don't lose ~3 sig figs at the bf16 tail
    of small δv.

Output per prompt (under output_dir/prompt_NNNNN/):
  v_n_s_metrics.json (mode s)            v_n_a_metrics.json (mode a)
  {
    "prompt_idx": int, "prompt": str, "seed": int, "num_steps": int,
    "mode": "s" | "a",
    "K": int, "eps_rel": float, "direction_seed": int,
    "per_k": [
      {"k": int,                     # step index n
       "target_norm": float,         # ||R_n||_F (s) or ||z_{n+1}||_F (a)
       "responses": [K floats],      # per-direction responses
       "rms_response": float,        # the reported S_n or A_n at step n
       "min_response": float,
       "max_response": float},
      ...
    ],
    "config": { ... }
  }

After both runs, build the unified V_n calibration with
analysis/build_sa_calib.py --calibration_mode clean-rms.

Sharding: prompt-sharded the same way as flux/oracle_runner.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.io_utils import read_prompts, split_shard  # noqa: E402

logger = logging.get_logger(__name__)


EPS = 1e-12


# =============================================================================
# Direction sampling — deterministic Gaussian, unit Frobenius norm
# =============================================================================

def _draw_unit_frob_direction(
    shape: Tuple[int, ...],
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """Draw a fp32 Gaussian, normalize to ||d||_F = 1, return on CPU.

    The Gaussian is drawn with the supplied generator (CPU). Normalization
    is done in fp32 so the unit-norm invariant is exact within fp32
    precision (rather than depending on bf16 quantization)."""
    d = torch.randn(*shape, generator=generator, dtype=dtype, device=device)
    norm = d.norm()
    if not torch.isfinite(norm) or float(norm) < EPS:
        raise RuntimeError(
            "direction draw produced degenerate norm "
            f"(norm={float(norm)}, shape={shape})"
        )
    return d / norm


# =============================================================================
# Mode S: forward replacement that probes G_n along K_S random directions
# =============================================================================

def _s_clean_forward(
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
    """Forward that (1) runs the full block stack to produce v_baseline,
    (2) probes K_S random Frobenius-unit perturbations of R_n, records the
    RMS response, (3) returns v_baseline so the trajectory continues
    unaltered.

    Per-step record appended to `self._s_per_k`.
    """
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

    # ---- full block stack (verbatim from flux/s_k_probe.py / flux/seacache.py)
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

    # F_n = hidden_states (post-block-stack), R_n = F_n - ori_hidden_states
    F_n = hidden_states
    R_n = F_n - ori_hidden_states  # whole-transformer residual (= seacache.py previous_residual on full step)

    # v_baseline = G_n(F_n) = proj_out(norm_out(F_n, temb))
    h_normed_base = self.norm_out(F_n, temb)
    v_baseline = self.proj_out(h_normed_base)

    # ---- S-probe: perturb R_n with K_S random fixed-Frobenius-norm directions
    if getattr(self, "enable_s_clean", False):
        step_idx = int(self.cnt)
        target_norm = float(R_n.detach().to(torch.float32).norm().item())
        eps_abs = float(self.eps_rel) * target_norm

        responses: List[float] = []
        v_baseline_fp32 = v_baseline.detach().to(torch.float32)
        if eps_abs <= 0:
            # Degenerate; record NaNs and continue
            responses = [float("nan")] * int(self.K)
        else:
            for k in range(int(self.K)):
                d = _draw_unit_frob_direction(
                    tuple(R_n.shape),
                    generator=self._s_dir_generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                d = d.to(R_n.device, dtype=R_n.dtype)
                F_perturbed = F_n + eps_abs * d
                h_normed_pert = self.norm_out(F_perturbed, temb)
                v_perturbed = self.proj_out(h_normed_pert).detach().to(torch.float32)
                delta_v_norm = float((v_perturbed - v_baseline_fp32).norm().item())
                responses.append(delta_v_norm / eps_abs)

        arr = np.asarray(responses, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")
        record = {
            "k": step_idx,
            "target_norm": target_norm,
            "eps_abs": eps_abs,
            "responses": responses,
            "rms_response": rms,
            "min_response": float(np.min(arr)) if arr.size else float("nan"),
            "max_response": float(np.max(arr)) if arr.size else float("nan"),
        }
        self._s_per_k.append(record)

        self.cnt += 1
        if self.cnt == int(self.num_steps):
            self.cnt = 0

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (v_baseline,)
    return Transformer2DModelOutput(sample=v_baseline)


def install_s_clean_probe(
    pipe,
    *,
    num_steps: int,
    K: int,
    eps_rel: float,
    direction_seed: int,
) -> Callable[[], None]:
    """Patch FluxTransformer2DModel.forward + attach per-instance S-probe state.

    Returns a teardown callable. Idempotent.
    """
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _s_clean_forward

    tr = pipe.transformer
    tr.enable_s_clean = True
    tr.num_steps = int(num_steps)
    tr.K = int(K)
    tr.eps_rel = float(eps_rel)
    tr.direction_seed = int(direction_seed)
    tr.cnt = 0
    tr._s_per_k = []
    tr._s_dir_generator = torch.Generator(device="cpu").manual_seed(int(direction_seed))

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "enable_s_clean", "num_steps", "K", "eps_rel", "direction_seed",
            "cnt", "_s_per_k", "_s_dir_generator",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_s_clean_state(pipe) -> None:
    tr = pipe.transformer
    tr.cnt = 0
    tr._s_per_k = []
    tr._s_dir_generator = torch.Generator(device="cpu").manual_seed(int(tr.direction_seed))


# =============================================================================
# Mode A: manual denoising loop with post-Euler perturbation + remaining sampler
# =============================================================================

def _calculate_shift(image_seq_len: int,
                     base_seq_len: int, max_seq_len: int,
                     base_shift: float, max_shift: float) -> float:
    # Same as flux/a_k_probe.py — taken from diffusers FluxPipeline source.
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _shift_args_from_scheduler(scheduler) -> Dict[str, float]:
    cfg = scheduler.config
    return {
        "base_seq_len": int(getattr(cfg, "base_image_seq_len", 256)),
        "max_seq_len": int(getattr(cfg, "max_image_seq_len", 4096)),
        "base_shift": float(getattr(cfg, "base_shift", 0.5)),
        "max_shift": float(getattr(cfg, "max_shift", 1.15)),
    }


def _encode_and_prepare(pipe, prompt: str, seed: int, args) -> Dict[str, Any]:
    """Encode prompt + initial latents + scheduler timesteps. Mirrors
    flux/a_k_probe.py: _encode_and_prepare."""
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
        "sigmas": pipe.scheduler.sigmas.detach().cpu().numpy().tolist(),
        "H": H, "W": W,
    }


def _denoise_step(transformer, scheduler, latents: torch.Tensor,
                  timestep: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
    """One (transformer + scheduler) step. Mirrors flux/a_k_probe.py:_denoise_step."""
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


def _baseline_pass_a(pipe, ctx: Dict[str, Any]) -> List[torch.Tensor]:
    """Pass 1: full denoise with vanilla forward. Returns z_after_step[i] = z_{i+1}.
    Mirrors flux/a_k_probe.py:_baseline_pass."""
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
    Mirrors flux/a_k_probe.py:_partial_denoise — must reset
    scheduler._step_index = None before the first step so set_step_index
    re-resolves from the timestep."""
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


def probe_one_prompt_a(
    pipe,
    prompt: str,
    seed: int,
    args,
    direction_generator: torch.Generator,
) -> Dict[str, Any]:
    """A-mode probe for one prompt.

    Pass 1: baseline denoise → z_after_step[i] for i ∈ [0, N-1] (z_after_step[i] = z_{i+1}).
    Pass 2: for each step n ∈ [0, N-2] (n = N-1 has no remaining suffix):
              z_{n+1} = z_after_step[n]
              For k in range(K_A):
                d = unit Frobenius direction
                z_perturbed_{n+1} = z_{n+1} + (eps_rel · ||z_{n+1}||_F) · d
                z_perturbed_N = partial_denoise(z_perturbed_{n+1},
                                                start_idx=n+1, end_idx=N)
                response_k = ||z_perturbed_N - z_after_step[N-1]||_F / eps_abs
    """
    with torch.no_grad():
        ctx = _encode_and_prepare(pipe, prompt, seed, args)
        N = int(args.num_steps)
        device = ctx["device"]

        z_after = _baseline_pass_a(pipe, ctx)
        assert len(z_after) == N
        z_N = z_after[N - 1].to(torch.float32)

        per_k: List[Dict[str, Any]] = []
        for n in range(N):
            if n == N - 1:
                # No remaining sampler to propagate; record empty.
                per_k.append({
                    "k": n,
                    "target_norm": float(z_after[n].to(torch.float32).norm().item()),
                    "eps_abs": 0.0,
                    "responses": [],
                    "rms_response": float("nan"),
                    "min_response": float("nan"),
                    "max_response": float("nan"),
                })
                continue

            z_np1 = z_after[n]  # this is z_{n+1}
            target_norm = float(z_np1.to(torch.float32).norm().item())
            eps_abs = float(args.eps_rel) * target_norm

            responses: List[float] = []
            if eps_abs <= 0:
                responses = [float("nan")] * int(args.K)
            else:
                for k in range(int(args.K)):
                    d = _draw_unit_frob_direction(
                        tuple(z_np1.shape),
                        generator=direction_generator,
                        dtype=torch.float32,
                        device="cpu",
                    )
                    d = d.to(device, dtype=z_np1.dtype)
                    z_perturbed_np1 = z_np1 + eps_abs * d
                    z_perturbed_N = _partial_denoise(
                        pipe, ctx, z_perturbed_np1,
                        start_idx=n + 1, end_idx=N,
                    ).to(torch.float32)
                    delta = (z_perturbed_N - z_N).norm().item()
                    responses.append(float(delta) / eps_abs)

            arr = np.asarray(responses, dtype=np.float64)
            rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")
            per_k.append({
                "k": n,
                "target_norm": target_norm,
                "eps_abs": eps_abs,
                "responses": responses,
                "rms_response": rms,
                "min_response": float(np.min(arr)) if arr.size else float("nan"),
                "max_response": float(np.max(arr)) if arr.size else float("nan"),
            })

        return {
            "per_k": per_k,
            "num_steps": N,
        }


# =============================================================================
# Mode: linearity check (S-style, sweep ε)
# =============================================================================

def _linearity_check_one_prompt(
    pipe,
    prompt: str,
    seed: int,
    args,
    eps_values: List[float],
    step_subset: List[int],
    direction_seed: int,
) -> List[Dict[str, Any]]:
    """For one prompt, run a baseline + at each step in step_subset apply
    ONE direction (per step, NEW seed) at each ε in eps_values; report the
    response at each ε.

    Implementation: install a SPECIAL forward that records responses per
    step but only fires at step_subset, and uses a fresh ε / direction
    pair for each (step, eps). Simpler: run len(eps_values) separate
    pipe(...) calls with different eps_rel settings, using the same
    direction_seed. Each call uses --K 1 internally.
    """
    out_records: List[Dict[str, Any]] = []
    for eps_rel in eps_values:
        teardown = install_s_clean_probe(
            pipe,
            num_steps=int(args.num_steps),
            K=1,
            eps_rel=eps_rel,
            direction_seed=direction_seed,
        )
        try:
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
            per_k = list(pipe.transformer._s_per_k)
        finally:
            teardown()
        for rec in per_k:
            if rec["k"] not in step_subset:
                continue
            out_records.append({
                "step_k": int(rec["k"]),
                "eps_rel": float(eps_rel),
                "response": float(rec["responses"][0]) if rec["responses"] else float("nan"),
                "target_norm": float(rec["target_norm"]),
            })
    return out_records


def run_linearity_check(pipe, args, prompts: List[str], global_idxs: List[int]) -> Dict[str, Any]:
    """Plan §6 E0-clean-RMS verification (ε ∈ {1e-4, 1e-3, 1e-2, 1e-1}
    on 3 prompts × 5 steps × 1 direction). Returns the full sweep + a
    pass/fail summary based on ±20 % agreement across the bottom three
    ε values.
    """
    if len(prompts) < 3 or len(global_idxs) < 3:
        raise SystemExit(
            f"linearity check requires at least 3 prompts; got {len(prompts)} "
            "(supply --limit 3 or a shard with at least 3 prompts)."
        )

    eps_values = [1e-4, 1e-3, 1e-2, 1e-1]
    step_subset = [5, 15, 25, 35, 45]
    per_prompt_records: List[Dict[str, Any]] = []

    for local_i, (prompt, global_idx) in enumerate(zip(prompts[:3], global_idxs[:3])):
        per_prompt_seed = int(args.seed) + int(global_idx)
        records = _linearity_check_one_prompt(
            pipe,
            prompt=prompt,
            seed=per_prompt_seed,
            args=args,
            eps_values=eps_values,
            step_subset=step_subset,
            direction_seed=int(args.direction_seed) + int(local_i),
        )
        per_prompt_records.append({
            "prompt_idx": int(global_idx),
            "seed": int(per_prompt_seed),
            "records": records,
        })

    # Pass/fail: for each (prompt, step), the responses at ε ∈ {1e-4, 1e-3, 1e-2}
    # should agree within ±20 %. ε = 1e-1 is allowed to differ (likely outside
    # the linear regime).
    failures: List[Dict[str, Any]] = []
    for pp in per_prompt_records:
        per_step: Dict[int, Dict[float, float]] = {}
        for rec in pp["records"]:
            per_step.setdefault(rec["step_k"], {})[rec["eps_rel"]] = rec["response"]
        for k, by_eps in per_step.items():
            small_eps = [1e-4, 1e-3, 1e-2]
            values = [by_eps.get(e) for e in small_eps]
            if any(v is None for v in values):
                continue
            mean_v = float(np.mean(values))
            if mean_v <= 0:
                continue
            rel_spread = max(abs(v - mean_v) / mean_v for v in values)
            if rel_spread > 0.20:
                failures.append({
                    "prompt_idx": pp["prompt_idx"],
                    "step_k": k,
                    "values_at_eps_1e-4_1e-3_1e-2": values,
                    "rel_spread": float(rel_spread),
                })

    return {
        "mode": "linearity_check",
        "eps_values": eps_values,
        "step_subset": step_subset,
        "per_prompt": per_prompt_records,
        "failures": failures,
        "pass": (len(failures) == 0),
    }


def _a_linearity_check_one_prompt(
    pipe,
    prompt: str,
    seed: int,
    args,
    eps_values: List[float],
    step_subset: List[int],
    direction_seed: int,
) -> List[Dict[str, Any]]:
    """A-mode linearity check for one prompt.

    For each checked step, use one fixed unit direction across all eps values
    so the spread measures local linearity rather than direction variability.
    """
    ctx = _encode_and_prepare(pipe, prompt, seed, args)
    z_after = _baseline_pass_a(pipe, ctx)
    z_N = z_after[int(args.num_steps) - 1].to(torch.float32)
    records: List[Dict[str, Any]] = []
    for step_k in step_subset:
        if step_k >= int(args.num_steps) - 1:
            continue
        z_np1 = z_after[step_k]
        target_norm = float(z_np1.to(torch.float32).norm().item())
        gen = torch.Generator(device="cpu").manual_seed(
            int(direction_seed) + 100003 * int(step_k)
        )
        direction = _draw_unit_frob_direction(
            tuple(z_np1.shape),
            generator=gen,
            dtype=torch.float32,
            device="cpu",
        ).to(ctx["device"], dtype=z_np1.dtype)
        for eps_rel in eps_values:
            eps_abs = float(eps_rel) * target_norm
            if eps_abs <= 0:
                response = float("nan")
            else:
                z_perturbed = z_np1 + eps_abs * direction
                z_perturbed_N = _partial_denoise(
                    pipe,
                    ctx,
                    z_perturbed,
                    start_idx=step_k + 1,
                    end_idx=int(args.num_steps),
                ).to(torch.float32)
                response = float((z_perturbed_N - z_N).norm().item()) / eps_abs
            records.append({
                "step_k": int(step_k),
                "eps_rel": float(eps_rel),
                "response": float(response),
                "target_norm": float(target_norm),
            })
    return records


def _linearity_failures(
    per_prompt_records: List[Dict[str, Any]],
    *,
    small_eps: Sequence[float] = (1e-4, 1e-3, 1e-2),
    tolerance: float = 0.20,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for pp in per_prompt_records:
        per_step: Dict[int, Dict[float, float]] = {}
        for rec in pp["records"]:
            per_step.setdefault(rec["step_k"], {})[rec["eps_rel"]] = rec["response"]
        for k, by_eps in per_step.items():
            values = [by_eps.get(e) for e in small_eps]
            if any(v is None or not np.isfinite(v) for v in values):
                continue
            mean_v = float(np.mean(values))
            if mean_v <= 0:
                continue
            rel_spread = max(abs(float(v) - mean_v) / mean_v for v in values)
            if rel_spread > tolerance:
                failures.append({
                    "prompt_idx": pp["prompt_idx"],
                    "step_k": int(k),
                    "values_at_eps_1e-4_1e-3_1e-2": [float(v) for v in values],
                    "rel_spread": float(rel_spread),
                })
    return failures


def run_linearity_check_a(pipe, args, prompts: List[str], global_idxs: List[int]) -> Dict[str, Any]:
    if len(prompts) < 3 or len(global_idxs) < 3:
        raise SystemExit(
            f"A-linearity check requires at least 3 prompts; got {len(prompts)} "
            "(supply --limit 3 or a shard with at least 3 prompts)."
        )

    eps_values = [1e-4, 1e-3, 1e-2, 1e-1]
    step_subset = [5, 15, 25, 35, 45]
    per_prompt_records: List[Dict[str, Any]] = []
    for local_i, (prompt, global_idx) in enumerate(zip(prompts[:3], global_idxs[:3])):
        per_prompt_seed = int(args.seed) + int(global_idx)
        direction_seed = int(args.direction_seed) + 1000003 * int(global_idx) + int(local_i)
        records = _a_linearity_check_one_prompt(
            pipe,
            prompt=prompt,
            seed=per_prompt_seed,
            args=args,
            eps_values=eps_values,
            step_subset=step_subset,
            direction_seed=direction_seed,
        )
        per_prompt_records.append({
            "prompt_idx": int(global_idx),
            "seed": int(per_prompt_seed),
            "records": records,
        })

    failures = _linearity_failures(per_prompt_records)
    return {
        "mode": "linearity_check_a",
        "eps_values": eps_values,
        "step_subset": step_subset,
        "per_prompt": per_prompt_records,
        "failures": failures,
        "pass": (len(failures) == 0),
    }


# =============================================================================
# Per-prompt runner (mode s + mode a)
# =============================================================================

def _run_one_pipe_call_for_s_probe(pipe, prompt: str, seed: int, args) -> None:
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


def _process_prompt_s(pipe, *, global_idx: int, prompt: str, args) -> Dict[str, Any]:
    seed = int(args.seed) + int(global_idx)
    direction_seed_effective = int(args.direction_seed) + 100003 * int(global_idx)
    teardown = install_s_clean_probe(
        pipe,
        num_steps=int(args.num_steps),
        K=int(args.K),
        eps_rel=float(args.eps_rel),
        direction_seed=direction_seed_effective,
    )
    try:
        t0 = time.perf_counter()
        _run_one_pipe_call_for_s_probe(pipe, prompt, seed, args)
        per_k = list(pipe.transformer._s_per_k)
        wall = time.perf_counter() - t0
    finally:
        teardown()

    if len(per_k) != int(args.num_steps):
        raise RuntimeError(
            f"S-probe recorded {len(per_k)} steps, expected {args.num_steps}"
        )

    return {
        "prompt_idx": int(global_idx),
        "prompt": prompt,
        "seed": int(seed),
        "num_steps": int(args.num_steps),
        "mode": "s",
        "K": int(args.K),
        "eps_rel": float(args.eps_rel),
        "direction_seed": int(args.direction_seed),
        "direction_seed_effective": int(direction_seed_effective),
        "per_k": per_k,
        "wall_seconds": float(wall),
        "config": {
            "model_id": args.model_id,
            "model_name": args.model_name,
            "dtype": args.dtype,
            "width": int(args.width),
            "height": int(args.height),
        },
    }


def _process_prompt_a(pipe, *, global_idx: int, prompt: str, args) -> Dict[str, Any]:
    seed = int(args.seed) + int(global_idx)
    # Per-prompt direction generator (deterministic), seeded by
    # direction_seed + global_idx so different prompts get independent
    # streams even when sharded across GPUs.
    gen = torch.Generator(device="cpu").manual_seed(
        int(args.direction_seed) + 100003 * int(global_idx)
    )
    direction_seed_effective = int(args.direction_seed) + 100003 * int(global_idx)
    t0 = time.perf_counter()
    result = probe_one_prompt_a(pipe, prompt, seed, args, direction_generator=gen)
    wall = time.perf_counter() - t0

    return {
        "prompt_idx": int(global_idx),
        "prompt": prompt,
        "seed": int(seed),
        "num_steps": int(args.num_steps),
        "mode": "a",
        "K": int(args.K),
        "eps_rel": float(args.eps_rel),
        "direction_seed": int(args.direction_seed),
        "direction_seed_effective": int(direction_seed_effective),
        "per_k": result["per_k"],
        "wall_seconds": float(wall),
        "config": {
            "model_id": args.model_id,
            "model_name": args.model_name,
            "dtype": args.dtype,
            "width": int(args.width),
            "height": int(args.height),
        },
    }


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0-clean-RMS V_n probe (plan §6 E0-clean-RMS) — random "
                    "fixed-Frobenius-norm directions, K-direction RMS gain."
    )
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--mode", choices=["s", "a", "linearity_check", "linearity_check_a"], required=True,
                   help="`s` = R_n perturbation + G_n response; "
                        "`a` = z_{n+1} perturbation + remaining sampler response; "
                        "`linearity_check` = sweep ε on 3 prompts × 5 steps to "
                        "verify S ε_rel=1e-2 is in linear regime; "
                        "`linearity_check_a` = same for A suffix propagation.")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed for the FLUX sampler; per-prompt seed = "
                        "seed + global_idx. Matches Phase-1 oracle convention.")
    p.add_argument("--K", type=int, default=8,
                   help="Number of random directions per (prompt, step). "
                        "Plan default 8.")
    p.add_argument("--eps_rel", type=float, default=1e-2,
                   help="Relative perturbation magnitude: "
                        "eps_abs = eps_rel · ||target||_F. Plan default 1e-2 "
                        "(after linearity_check pass).")
    p.add_argument("--direction_seed", type=int, default=None,
                   help="RNG seed for direction draws. Default: 0 for mode=s, "
                        "1 for mode=a. Pin different seeds for S and A so "
                        "the two probes are independent.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--model_name", choices=["flux-dev", "flux-schnell"], default="flux-dev")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--prompt_offset", type=int, default=0,
                   help="First global prompt index to use before applying --limit. "
                        "The output prompt_idx and seed remain in the original "
                        "prompt-file coordinate system.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts after --prompt_offset, before sharding (0 = all).")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts whose v_n_<mode>_metrics.json already exists.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.direction_seed is None:
        args.direction_seed = 1 if args.mode in ("a", "linearity_check_a") else 0

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if int(args.prompt_offset) < 0:
        raise SystemExit(f"--prompt_offset must be non-negative, got {args.prompt_offset}")
    prompts_full = read_prompts(args.prompt_file, limit=None)
    if int(args.prompt_offset) >= len(prompts_full):
        raise SystemExit(
            f"--prompt_offset={args.prompt_offset} is outside prompt file with "
            f"{len(prompts_full)} prompts"
        )
    slice_end = (
        len(prompts_full)
        if int(args.limit) <= 0
        else min(len(prompts_full), int(args.prompt_offset) + int(args.limit))
    )
    prompts_all = prompts_full[int(args.prompt_offset):slice_end]
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}/{args.shard_count}] empty slice, exiting.", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {args.model_id} "
          f"dtype={args.dtype}", flush=True)
    process_start = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Loaded in "
        f"{model_load_end - process_start:.1f}s; shard "
        f"{args.shard_idx}/{args.shard_count} has {len(shard_prompts)} prompts "
        f"(global idx {int(args.prompt_offset) + start}..{int(args.prompt_offset) + end - 1}); "
        f"mode={args.mode} K={args.K} "
        f"eps_rel={args.eps_rel} direction_seed={args.direction_seed}",
        flush=True,
    )

    # === Linearity-check fast path ===
    if args.mode in ("linearity_check", "linearity_check_a"):
        global_idxs = [int(args.prompt_offset) + i for i in range(start, end)]
        if args.mode == "linearity_check":
            result = run_linearity_check(pipe, args, shard_prompts, global_idxs)
        else:
            result = run_linearity_check_a(pipe, args, shard_prompts, global_idxs)
        out_path = args.output_dir / f"{args.mode}_shard{args.shard_idx}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n[linearity_check] wrote {out_path}", flush=True)
        print(f"[linearity_check] PASS={result['pass']} "
              f"failures={len(result['failures'])}", flush=True)
        if not result["pass"]:
            print(f"[linearity_check] first failure: {result['failures'][0]}", flush=True)
            return 3
        return 0

    # === Production: mode s / mode a, prompt-sharded ===
    completed = 0
    skipped = 0
    failed = 0

    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = int(args.prompt_offset) + start + local_idx
        out_dir = args.output_dir / f"prompt_{global_idx:05d}"
        out_path = out_dir / f"v_n_{args.mode}_metrics.json"
        if args.resume and out_path.is_file():
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        t_p = time.perf_counter()
        try:
            if args.mode == "s":
                result = _process_prompt_s(pipe, global_idx=global_idx, prompt=prompt, args=args)
            else:  # a
                result = _process_prompt_a(pipe, global_idx=global_idx, prompt=prompt, args=args)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            completed += 1
            wall = time.perf_counter() - t_p
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] prompt {global_idx} "
                f"DONE in {wall:.1f}s (mode={args.mode})",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            failed += 1
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] prompt {global_idx} "
                f"FAILED ({type(e).__name__}: {e})",
                flush=True,
            )

    total_s = time.perf_counter() - process_start
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] shard "
        f"{args.shard_idx}/{args.shard_count} done. "
        f"completed={completed} skipped={skipped} failed={failed} "
        f"total={total_s:.1f}s",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
