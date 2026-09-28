#!/usr/bin/env python3
"""Experiment SM-B: marginal cache-vs-full labels.

docs/research_plan_stateful_marginal.md §5 SM-B. The deployable gate's real
target is the marginal harm of caching the current step. SM-B directly
measures it and pairs it with the cheap online features a gate would see.

Per (prompt, policy):
  1. run the prefix policy once through an instrumented forward, recording
     per step the gate-feature drift psi_drift_n and the cache/full
     decision u_n  (interval = fixed schedule; seacache = threshold gate);
  2. replay (psi_drift, u) through `MarginalFeatureTracker` to get the
     online features (q_n, gap, C^stale/traj/mem) entering each step;
  3. at stratified fork points n, measure the full-after marginal

         Delta_n^FA = || z_N^{cache n, full after} - z_N^{full n, full after} ||

     both branches sharing the prefix's realized cache decisions for k < n
     and running full from n+1 (plan §3 / §5 SM-B Target A).

Output per prompt: prompt_XXXXX/marginal_rows.json
  [{prompt_idx, policy, step, gap, last_refresh, q, p_acc, c_stale,
    c_traj, c_mem, psi_drift, native_gate_signal, n_cache_before,
    delta_latent_l2_full_after}, ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.io_utils import read_prompts, split_shard  # noqa: E402
from lib.gates import rel_l1  # noqa: E402
from lib.wiener import apply_sea_with_scheduler  # noqa: E402
from lib.marginal_features import MarginalFeatureTracker  # noqa: E402
from flux.oracle_runner import _run_one_pipe_call  # noqa: E402
from flux.marginal_fork_probe import _run_cache_set  # noqa: E402

logger = logging.get_logger(__name__)

EPS = 1e-6  # matches analysis/marginal_estimator and state_gate_runner.


def _est2_predict(q: float, gap: int, step: int, c_stale: float,
                  c_traj: float, c_mem: float, beta: List[float]) -> float:
    """Δ̂ = exp(β @ design). Layout must match marginal_estimator._design.

    Duplicated from state_gate_runner._est2_predict so the on-policy probe
    uses the byte-identical design row when prefix=smd_est2.
    """
    design = [1.0, math.log(q + EPS), float(gap), float(step),
              float(step) * float(gap),
              math.log1p(c_stale), math.log1p(c_traj), math.log1p(c_mem)]
    return math.exp(sum(b * d for b, d in zip(beta, design)))


# ---------------------------------------------------------------------------
# Δ^policy fork branch: forced decisions for step ≤ fork_n; est2-gate for k > n.
# Used by --fork_tail gate_est2 to test §4.10.1 target mis-specification.
# ---------------------------------------------------------------------------
def _policy_branch_forward(
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
):
    """Hybrid forward for Δ^policy fork branches.

    - Step cnt in self.pb_forced (a dict {step: u}): u is overridden.
    - Step cnt NOT in self.pb_forced (i.e. cnt > fork_n): the est2-gate
      decides via predict<τ AND gap<g_max, using the live tracker state
      built up by replaying the prefix forced decisions.

    Same compute path as state_gate_runner._gate_forward — zero-order
    whole-transformer residual reuse on cache, full block stack on full.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

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

    cnt = int(self.pb_cnt)
    N = int(self.pb_num_steps)
    ori_hidden_states = hidden_states

    # ---- ψ feature (byte-identical to state_gate_runner) ---------------
    modulated_inp, *_ = self.transformer_blocks[0].norm1(ori_hidden_states, emb=temb)
    psi = modulated_inp.reshape(
        modulated_inp.shape[0],
        int(img_ids[:, 1].max().item() + 1),
        int(img_ids[:, 2].max().item() + 1),
        modulated_inp.shape[-1],
    )
    psi = apply_sea_with_scheduler(psi, self.scheduler, cnt,
                                   power_exp=2.0, dims=(-2, -3), norm_mode="mean")
    psi = psi.reshape(psi.shape[0], -1, psi.shape[-1])
    psi_drift = (rel_l1(psi, self.pb_prev_psi)
                 if self.pb_prev_psi is not None else 0.0)
    self.pb_prev_psi = psi.detach()

    feats = self.pb_tracker.observe(cnt, float(psi_drift))

    # ---- decision: forced (prefix or fork) OR gate (tail) --------------
    if cnt in self.pb_forced:
        u_n = int(self.pb_forced[cnt])
        # Safety: forced cache impossible at force_full window (no residual to
        # reuse). Caller should only force cache at steps where the prefix
        # policy itself cached, which by construction is past force_full.
        force_full_window = (cnt == 0 or cnt == N - 1
                             or cnt < int(self.pb_first_enhance)
                             or self.pb_previous_residual is None)
        if u_n == 1 and force_full_window:
            raise RuntimeError(
                f"pb_forced[{cnt}]=1 but step is in force_full window "
                f"(N={N}, first_enhance={self.pb_first_enhance}, "
                f"prev_residual_is_None={self.pb_previous_residual is None}).")
    else:
        force_full = (cnt == 0 or cnt == N - 1
                      or cnt < int(self.pb_first_enhance)
                      or self.pb_previous_residual is None)
        if force_full:
            u_n = 0
        elif self.pb_gate_kind == "q":
            predict = float(feats["q"])
            u_n = 1 if (predict < float(self.pb_threshold)
                        and feats["gap"] < int(self.pb_g_max)) else 0
        elif self.pb_gate_kind == "est2":
            predict = _est2_predict(feats["q"], feats["gap"], cnt,
                                    feats["c_stale"], feats["c_traj"],
                                    feats["c_mem"], self.pb_beta)
            u_n = 1 if (predict < float(self.pb_threshold)
                        and feats["gap"] < int(self.pb_g_max)) else 0
        else:
            raise RuntimeError(
                f"pb_gate_kind={self.pb_gate_kind!r} but step {cnt} is not in "
                f"pb_forced — caller must either pre-force every step "
                f"(schedule_locked) or set a gate_kind ∈ {{'q', 'est2'}}.")

    self.pb_tracker.commit(cnt, u_n)

    # ---- cache or full block compute (zero-order residual reuse) -------
    if u_n == 1 and self.pb_previous_residual is not None:
        hidden_states = hidden_states + self.pb_previous_residual
    else:
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb, image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                ic = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // ic]
        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb, image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                ic = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // ic]
        self.pb_previous_residual = hidden_states - ori_hidden_states

    self.pb_cnt += 1
    if self.pb_cnt == N:
        self.pb_cnt = 0

    self.pb_decisions.append({"step": cnt, "u": int(u_n)})

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install_policy_branch(pipe, *, forced: Dict[int, int],
                          gate_kind: Optional[str] = None,
                          beta: Optional[List[float]] = None,
                          threshold: float = 0.0, g_max: int = 8,
                          num_steps: int,
                          q_abs, s_cal, a_cal, mem_variant: int = 1,
                          first_enhance: int = 1):
    """Patch forward with the hybrid forced/gate controller.

    Three modes (selected by `gate_kind`):
      - `gate_kind=None`: schedule-locked. Every step must appear in `forced`.
        The forward never invokes any gate logic; β / threshold are ignored.
      - `gate_kind='q'`: q-gate decides any step not in `forced`. β not needed;
        threshold required.
      - `gate_kind='est2'`: est2-gate decides any step not in `forced`. β and
        threshold both required.
    """
    if gate_kind not in (None, "q", "est2"):
        raise ValueError(f"gate_kind must be None / 'q' / 'est2', got {gate_kind!r}")
    if gate_kind == "est2" and (beta is None or len(beta) != 8):
        raise ValueError("gate_kind='est2' requires an 8-element beta")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _policy_branch_forward
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.pb_forced = dict(forced)
    tr.pb_gate_kind = gate_kind
    tr.pb_beta = list(beta) if beta is not None else None
    tr.pb_threshold = float(threshold)
    tr.pb_g_max = int(g_max)
    tr.pb_num_steps = int(num_steps)
    tr.pb_first_enhance = int(first_enhance)
    tr.pb_tracker = MarginalFeatureTracker(q_abs, s_cal, a_cal,
                                           mem_variant=int(mem_variant))
    tr.pb_cnt = 0
    tr.pb_prev_psi = None
    tr.pb_previous_residual = None
    tr.pb_decisions = []
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("scheduler", "pb_forced", "pb_gate_kind", "pb_beta",
                     "pb_threshold", "pb_g_max", "pb_num_steps",
                     "pb_first_enhance", "pb_tracker", "pb_cnt", "pb_prev_psi",
                     "pb_previous_residual", "pb_decisions"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True
    return teardown


def _run_policy_branch(pipe, prompt, seed, args, *, forced: Dict[int, int],
                       gate_kind: Optional[str] = None,
                       beta: Optional[List[float]] = None,
                       threshold: float = 0.0, g_max: int = 8,
                       q_abs, s_cal, a_cal, mem_variant: int):
    """One full N-step pipe call with the policy-branch controller; return z_N."""
    teardown = install_policy_branch(
        pipe, forced=forced, gate_kind=gate_kind, beta=beta,
        threshold=threshold, g_max=g_max,
        num_steps=int(args.num_steps), q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
        mem_variant=int(mem_variant), first_enhance=int(args.first_enhance))
    try:
        z = _run_one_pipe_call(pipe, prompt, seed, args)
    finally:
        teardown()
    return z


# ---------------------------------------------------------------------------
# Instrumented forward: gate feature + pluggable policy + per-step recording.
# Zero-order whole-transformer residual reuse (matches oracle_runner seacache
# and flux/seacache.py), so the realized cache set replays identically when
# the fork branches re-run it through install_oracle.
# ---------------------------------------------------------------------------
def _mr_forward(
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
):
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

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

    # ---- gate feature psi: SEA-filtered modulated first-block input --------
    cnt = int(self.mr_cnt)
    N = int(self.mr_num_steps)
    first_block = self.transformer_blocks[0]
    modulated_inp, *_ = first_block.norm1(hidden_states, emb=temb)
    psi = modulated_inp.reshape(
        modulated_inp.shape[0],
        int(img_ids[:, 1].max().item() + 1),
        int(img_ids[:, 2].max().item() + 1),
        modulated_inp.shape[-1],
    )
    psi = apply_sea_with_scheduler(psi, self.scheduler, cnt,
                                   power_exp=2.0, dims=(-2, -3), norm_mode="mean")
    psi = psi.reshape(psi.shape[0], -1, psi.shape[-1])

    psi_drift = (rel_l1(psi, self.mr_prev_psi)
                 if self.mr_prev_psi is not None else 0.0)

    # ---- AV-C direction sketch (plan §7) ---------------------------------
    # Optional fixed separable random projection R(ψ_n) ∈ R^m. Lazily build
    # R1[m,D], R2[m,S] on first call, then reuse. Single forward FLOP cost
    # m·S·D ≈ 2·10^8 for FLUX 1024² m=16 — negligible vs the transformer.
    psi_proj_list: Optional[List[float]] = None
    if int(getattr(self, "mr_direction_m", 0) or 0) > 0:
        B, S, D = int(psi.shape[0]), int(psi.shape[1]), int(psi.shape[2])
        if self.mr_proj_R1 is None:
            m = int(self.mr_direction_m)
            gen = torch.Generator(device="cpu").manual_seed(int(self.mr_direction_seed))
            # Achlioptas-style normalisation: divide by sqrt(D) and sqrt(S)
            # so the separable product roughly preserves ψ-norm scale.
            R1_cpu = torch.randn((m, D), generator=gen, dtype=torch.float32) / math.sqrt(D)
            R2_cpu = torch.randn((m, S), generator=gen, dtype=torch.float32) / math.sqrt(S)
            self.mr_proj_R1 = R1_cpu.to(device=psi.device)
            self.mr_proj_R2 = R2_cpu.to(device=psi.device)
            self.mr_proj_shape = (B, S, D)
        elif self.mr_proj_shape != (B, S, D):
            raise RuntimeError(
                f"AV-C: ψ shape changed across steps: "
                f"{self.mr_proj_shape} → ({B}, {S}, {D}). "
                f"Probe assumes fixed resolution per run.")
        # einsum: (m,S) × (B,S,D) → (B,m,D); then (m,D) · (B,m,D) → (B,m).
        psi_f32 = psi.to(torch.float32)
        tmp = torch.einsum("ms,bsd->bmd", self.mr_proj_R2, psi_f32)
        proj = torch.einsum("md,bmd->bm", self.mr_proj_R1, tmp)
        psi_proj_list = [float(x) for x in proj[0].detach().cpu().tolist()]

    # ---- policy decides u_n (0 full / 1 cache) ----------------------------
    force_full = (cnt == 0 or cnt == N - 1 or cnt < int(self.mr_first_enhance)
                  or self.mr_prev_psi is None)
    if self.mr_policy in ("smd_q", "smd_est2"):
        feats = self.mr_tracker.observe(cnt, float(psi_drift))
        # native_gate_signal field still meaningful (= seacache acc); track too.
        self.mr_acc += float(psi_drift)
        predict = 0.0
        if force_full:
            u_n = 0
        elif self.mr_policy == "smd_q":
            predict = float(feats["q"])
            u_n = 1 if (predict < float(self.mr_threshold)
                        and feats["gap"] < int(self.mr_g_max)) else 0
        else:  # smd_est2
            predict = _est2_predict(feats["q"], feats["gap"], cnt,
                                    feats["c_stale"], feats["c_traj"],
                                    feats["c_mem"], self.mr_beta)
            u_n = 1 if (predict < float(self.mr_threshold)
                        and feats["gap"] < int(self.mr_g_max)) else 0
        self.mr_tracker.commit(cnt, u_n)
        if u_n == 0:
            self.mr_acc = 0.0
        rec_entry = {"step": cnt, "u": int(u_n),
                     "psi_drift": float(psi_drift),
                     "gate_acc": float(self.mr_acc),
                     "predict": float(predict)}
        if psi_proj_list is not None:
            rec_entry["psi_proj"] = psi_proj_list
        self.mr_rec.append(rec_entry)
    elif force_full:
        self.mr_acc = 0.0
        u_n = 0
        rec_entry = {"step": cnt, "u": int(u_n),
                     "psi_drift": float(psi_drift),
                     "gate_acc": float(self.mr_acc)}
        if psi_proj_list is not None:
            rec_entry["psi_proj"] = psi_proj_list
        self.mr_rec.append(rec_entry)
    elif self.mr_policy == "interval":
        self.mr_acc += psi_drift
        u_n = 1 if (cnt % int(self.mr_interval) != 0) else 0
        rec_entry = {"step": cnt, "u": int(u_n),
                     "psi_drift": float(psi_drift),
                     "gate_acc": float(self.mr_acc)}
        if psi_proj_list is not None:
            rec_entry["psi_proj"] = psi_proj_list
        self.mr_rec.append(rec_entry)
    elif self.mr_policy == "seacache":
        self.mr_acc += psi_drift
        if self.mr_acc < float(self.mr_thresh):
            u_n = 1
        else:
            u_n = 0
            self.mr_acc = 0.0
        rec_entry = {"step": cnt, "u": int(u_n),
                     "psi_drift": float(psi_drift),
                     "gate_acc": float(self.mr_acc)}
        if psi_proj_list is not None:
            rec_entry["psi_proj"] = psi_proj_list
        self.mr_rec.append(rec_entry)
    else:
        raise ValueError(f"unknown mr_policy: {self.mr_policy!r}")
    self.mr_prev_psi = psi.detach()
    self.mr_cnt += 1
    if self.mr_cnt == N:
        self.mr_cnt = 0

    # ---- cache (whole-transformer residual reuse) or full block compute ---
    if u_n == 1 and self.mr_previous_residual is not None:
        hidden_states = hidden_states + self.mr_previous_residual
    else:
        ori_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb, image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                ic = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // ic]
        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb, image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                ic = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // ic]
        self.mr_previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install_mr(pipe, *, policy: str, num_steps: int, interval: int = 2,
               thresh: float = 0.3, first_enhance: int = 1,
               threshold: float = 0.0, g_max: int = 8,
               q_abs: Optional[List[float]] = None,
               s_cal: Optional[List[float]] = None,
               a_cal: Optional[List[float]] = None,
               beta: Optional[List[float]] = None,
               mem_variant: int = 1,
               direction_m: int = 0,
               direction_seed: int = 0):
    """Patch FluxTransformer2DModel.forward with the SM-B instrumented gate.

    For prefix policies in {interval, seacache} the original code path runs.
    For prefix policies in {smd_q, smd_est2} a live MarginalFeatureTracker
    is installed so the gate makes decisions on its own self-played C-state
    (matches `state_gate_runner._gate_forward` exactly). `threshold`,
    `g_max`, calibration arrays, and (for est2) `beta` must be supplied.
    """
    if policy in ("smd_q", "smd_est2"):
        if q_abs is None or s_cal is None or a_cal is None:
            raise ValueError(f"prefix={policy!r} requires q_abs / s_cal / a_cal")
        if threshold <= 0:
            raise ValueError(f"prefix={policy!r} requires threshold > 0")
        if policy == "smd_est2" and (beta is None or len(beta) != 8):
            raise ValueError("prefix=smd_est2 requires an 8-element beta")

    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _mr_forward
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.mr_policy = str(policy)
    tr.mr_interval = int(interval)
    tr.mr_thresh = float(thresh)
    tr.mr_first_enhance = int(first_enhance)
    tr.mr_num_steps = int(num_steps)
    tr.mr_threshold = float(threshold)
    tr.mr_g_max = int(g_max)
    tr.mr_beta = list(beta) if beta is not None else None
    if policy in ("smd_q", "smd_est2"):
        tr.mr_tracker = MarginalFeatureTracker(q_abs, s_cal, a_cal,
                                               mem_variant=int(mem_variant))
    else:
        tr.mr_tracker = None
    tr.mr_cnt = 0
    tr.mr_acc = 0.0
    tr.mr_prev_psi = None
    tr.mr_previous_residual = None
    tr.mr_rec = []
    # AV-C direction sketch: lazily-built fixed projection matrices.
    tr.mr_direction_m = int(direction_m)
    tr.mr_direction_seed = int(direction_seed)
    tr.mr_proj_R1 = None        # [m, D] on ψ.device, fp32
    tr.mr_proj_R2 = None        # [m, S] on ψ.device, fp32
    tr.mr_proj_shape = None     # (B, S, D) sentinel for shape consistency
    _done = {"v": False}

    def teardown() -> None:
        if _done["v"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in ("scheduler", "mr_policy", "mr_interval", "mr_thresh",
                     "mr_first_enhance", "mr_num_steps", "mr_cnt", "mr_acc",
                     "mr_prev_psi", "mr_previous_residual", "mr_rec",
                     "mr_threshold", "mr_g_max", "mr_beta", "mr_tracker",
                     "mr_direction_m", "mr_direction_seed",
                     "mr_proj_R1", "mr_proj_R2", "mr_proj_shape"):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _done["v"] = True

    return teardown


def _run_prefix(pipe, prompt, seed, args, policy: str, interval: int,
                thresh: float, *, threshold: float = 0.0, g_max: int = 8,
                q_abs: Optional[List[float]] = None,
                s_cal: Optional[List[float]] = None,
                a_cal: Optional[List[float]] = None,
                beta: Optional[List[float]] = None,
                mem_variant: int = 1,
                direction_m: int = 0,
                direction_seed: int = 0) -> List[Dict]:
    """Run one prefix policy; return the per-step record list.

    For policy ∈ {smd_q, smd_est2} the extra kwargs (threshold, g_max,
    calibration arrays, beta for est2) are required and forwarded to
    `install_mr` so the prefix runs with the same gate as state_gate_runner.
    """
    teardown = install_mr(pipe, policy=policy, num_steps=int(args.num_steps),
                          interval=interval, thresh=thresh,
                          first_enhance=int(args.first_enhance),
                          threshold=threshold, g_max=g_max,
                          q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                          beta=beta, mem_variant=mem_variant,
                          direction_m=direction_m,
                          direction_seed=direction_seed)
    try:
        _run_one_pipe_call(pipe, prompt, seed, args)
        rec = sorted(pipe.transformer.mr_rec, key=lambda r: r["step"])
    finally:
        teardown()
    return rec


def _sample_forks(num_steps: int, n_forks: int) -> List[int]:
    """Position-stratified fork steps in 1..N-2 (one per equal bucket).

    v1 simplification: gap / q / refresh-adjacent strata (plan §5 SM-B
    'Sampling fork points') are left to a later version.
    """
    lo, hi = 1, num_steps - 2
    span = hi - lo + 1
    n = min(n_forks, span)
    out = []
    for i in range(n):
        step = lo + int(round((i + 0.5) * span / n)) - 1
        out.append(min(hi, max(lo, step)))
    return sorted(set(out))


def _features_by_step(rec: List[Dict], q_abs, s_cal, a_cal,
                      mem_variant: int) -> Dict[int, Dict[str, float]]:
    """Replay (psi_drift, u) through the tracker; snapshot features per step."""
    tr = MarginalFeatureTracker(q_abs, s_cal, a_cal, mem_variant=mem_variant)
    out: Dict[int, Dict[str, float]] = {}
    for r in rec:
        out[r["step"]] = dict(tr.observe(r["step"], r["psi_drift"]))
        tr.commit(r["step"], r["u"])
    return out


def _load_calib(q_k_path: Path, sa_calib_path: Path, num_steps: int
                ) -> Tuple[List[float], List[float], List[float]]:
    """Per-step |h_n| (Q_k), S_hat_n, A_hat_n arrays."""
    qd = json.loads(Path(q_k_path).read_text())
    q_abs = [abs(float(x)) for x in qd["Q_k"]]
    sd = json.loads(Path(sa_calib_path).read_text())
    s_cal = [float(x) for x in sd["S_k"]]
    a_cal = [float(x) for x in sd["A_k"]]
    for name, arr in (("Q_k", q_abs), ("S_k", s_cal), ("A_k", a_cal)):
        if len(arr) != num_steps:
            raise SystemExit(f"{name} length {len(arr)} != num_steps {num_steps}")
    return q_abs, s_cal, a_cal


def _parse_policies(spec: str) -> List[Tuple[str, str, int, float]]:
    """Parse comma-separated policy spec.

    Tokens:
      interval_<n>            → (name, "interval", n, 0.0)
      seacache_<delta>        → (name, "seacache", 0, delta)
      smd_q_t<tau>            → (name, "smd_q", 0, tau)
      smd_est2_t<tau>         → (name, "smd_est2", 0, tau)

    smd_q / smd_est2 policies require --threshold-driven runtime args; the
    parsed tau here is parsed out of the *token* and is used as the gate's
    threshold during the prefix run (no separate --threshold CLI knob —
    each policy token carries its own).
    """
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("interval_"):
            out.append((tok, "interval", int(tok.split("_")[1]), 0.0))
        elif tok.startswith("seacache_"):
            out.append((tok, "seacache", 0, float(tok.split("_", 1)[1])))
        elif tok.startswith("smd_q_t"):
            out.append((tok, "smd_q", 0, float(tok.split("_t", 1)[1])))
        elif tok.startswith("smd_est2_t"):
            out.append((tok, "smd_est2", 0, float(tok.split("_t", 1)[1])))
        else:
            raise ValueError(f"unknown policy token: {tok!r}")
    if not out:
        raise ValueError("no policies parsed")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp SM-B: marginal cache-vs-full labels.")
    p.add_argument("--prompt_file", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--q_k", type=Path, required=True)
    p.add_argument("--sa_calib", type=Path, required=True)
    p.add_argument("--policies", default="interval_2,interval_4,seacache_0.3,seacache_0.6",
                   help=("Comma-separated. Tokens: interval_<n>, seacache_<delta>, "
                         "smd_q_t<tau>, smd_est2_t<tau>. smd_q / smd_est2 require "
                         "--q_k / --sa_calib (and --estimator_summary for smd_est2)."))
    p.add_argument("--g_max", type=int, default=8,
                   help="hard cap on consecutive cached steps for smd_q/smd_est2 prefixes.")
    p.add_argument("--estimator_summary", type=Path, default=None,
                   help=("SM-C marginal_estimator_summary.json (β source for "
                         "smd_est2 prefix policies AND fork_tail=gate_est2)."))
    p.add_argument("--fork_tail",
                   choices=["full", "schedule_locked", "gate_q", "gate_est2"],
                   default="full",
                   help=(
                       "Fork tail mode (selects which Δ label is computed):\n"
                       "  full            → Δ^FA: full forward for k > n.\n"
                       "  schedule_locked → Δ^{SL|π_ref}: both branches play "
                           "the prefix policy's recorded u_k for k > n. "
                           "Anchor π_ref = the (single) prefix policy used "
                           "in --policies (recommended for AV-A).\n"
                       "  gate_q          → Δ^{RE|q}: branches run q-gate at "
                           "--fork_tail_threshold for k > n.\n"
                       "  gate_est2       → Δ^{RE|est2}: branches run est2-"
                           "gate at --fork_tail_threshold for k > n "
                           "(requires --estimator_summary)."))
    p.add_argument("--fork_tail_threshold", type=float, default=0.0,
                   help="Gate threshold τ for fork_tail=gate_q/gate_est2.")
    p.add_argument("--n_forks", type=int, default=8)
    p.add_argument("--mem_variant", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--first_enhance", type=int, default=1)
    p.add_argument("--cache_mode", default="seacache")
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
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    # AV-C direction sketch (plan §7). If --direction_m > 0, the probe
    # additionally records per-step R(ψ_n) where R is a fixed separable
    # random projection ψ → R^m. Adds "psi_proj" (list of m floats) to
    # each per_step_trace entry. Default 0 → off (back-compat).
    p.add_argument("--direction_m", type=int, default=0,
                   help=("AV-C: project ψ_n to m dims via fixed separable "
                         "Gaussian R1[m,D] · R2[m,S]; 0 disables (default)."))
    p.add_argument("--direction_seed", type=int, default=0,
                   help="Seed for the fixed projection matrices (default 0).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    N = int(args.num_steps)
    policies = _parse_policies(args.policies)
    q_abs, s_cal, a_cal = _load_calib(args.q_k, args.sa_calib, N)
    forks = _sample_forks(N, int(args.n_forks))

    needs_beta = any(p[1] == "smd_est2" for p in policies) or args.fork_tail == "gate_est2"
    est2_beta: Optional[List[float]] = None
    if needs_beta:
        if args.estimator_summary is None:
            raise SystemExit("smd_est2 prefix or fork_tail=gate_est2 need --estimator_summary")
        summary = json.loads(Path(args.estimator_summary).read_text())
        coef = summary["estimators"]["est2"]["coef"]
        if coef is None:
            raise SystemExit("no est2 coef in --estimator_summary")
        est2_beta = [float(x) for x in coef]
        if len(est2_beta) != 8:
            raise SystemExit(f"est2 beta length {len(est2_beta)} != 8")
    if args.fork_tail in ("gate_q", "gate_est2") and float(args.fork_tail_threshold) <= 0:
        raise SystemExit(
            f"fork_tail={args.fork_tail} requires --fork_tail_threshold > 0")
    if args.fork_tail == "schedule_locked" and len(policies) != 1:
        raise SystemExit(
            "fork_tail=schedule_locked uses the (single) prefix policy as π_ref; "
            "specify exactly one policy via --policies.")

    prompts_all = read_prompts(args.prompt_file,
                               limit=(args.limit if args.limit > 0 else None))
    start, end = split_shard(len(prompts_all), args.shard_count, args.shard_idx)
    shard_prompts = prompts_all[start:end]
    if not shard_prompts:
        print(f"[shard {args.shard_idx}] empty slice, exiting.", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading {args.model_id}", flush=True)
    t0 = time.perf_counter()
    from diffusers import DiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype_map[args.dtype]).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_end = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] Loaded {model_load_end - t0:.1f}s; "
          f"shard {args.shard_idx}/{args.shard_count} = {len(shard_prompts)} prompts; "
          f"{len(policies)} policies; forks {forks}", flush=True)

    per_prompt = []
    for local_idx, prompt in enumerate(shard_prompts):
        global_idx = start + local_idx
        prompt_dir = args.output_dir / f"prompt_{global_idx:05d}"
        manifest_path = prompt_dir / "manifest.json"
        if args.resume and manifest_path.is_file():
            try:
                if json.loads(manifest_path.read_text()).get("complete", False):
                    print(f"[shard {args.shard_idx}] prompt {global_idx} done, skip", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        prompt_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + global_idx
        t_p = time.perf_counter()

        rows: List[Dict] = []
        policy_recs: Dict[str, List[Dict]] = {}
        for (pname, kind, interval, thresh) in policies:
            if kind in ("smd_q", "smd_est2"):
                rec = _run_prefix(pipe, prompt, seed, args, kind, interval, 0.0,
                                  threshold=thresh, g_max=int(args.g_max),
                                  q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                                  beta=est2_beta if kind == "smd_est2" else None,
                                  mem_variant=int(args.mem_variant),
                                  direction_m=int(args.direction_m),
                                  direction_seed=int(args.direction_seed))
            else:
                rec = _run_prefix(pipe, prompt, seed, args, kind, interval, thresh,
                                  direction_m=int(args.direction_m),
                                  direction_seed=int(args.direction_seed))
            policy_recs[pname] = rec
            cache_set = [r["step"] for r in rec if r["u"] == 1]
            feats = _features_by_step(rec, q_abs, s_cal, a_cal, int(args.mem_variant))

            for n in forks:
                cache_F = [k for k in cache_set if k < n]
                cache_C = cache_F + [n]
                if args.fork_tail == "full":
                    # Δ^FA: tail = full forward (no gate logic).
                    z_F = _run_cache_set(pipe, prompt, seed, args, cache_F)
                    z_C = _run_cache_set(pipe, prompt, seed, args, cache_C)
                elif args.fork_tail == "schedule_locked":
                    # Δ^{SL|π_ref}: both branches forced to the prefix policy's
                    # u_ref for ALL k, with the fork step flipped per branch.
                    u_ref = {r["step"]: int(r["u"]) for r in rec}
                    forced_F = dict(u_ref); forced_F[n] = 0
                    forced_C = dict(u_ref); forced_C[n] = 1
                    z_F = _run_policy_branch(
                        pipe, prompt, seed, args, forced=forced_F,
                        gate_kind=None, beta=None,
                        q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                        mem_variant=int(args.mem_variant))
                    z_C = _run_policy_branch(
                        pipe, prompt, seed, args, forced=forced_C,
                        gate_kind=None, beta=None,
                        q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                        mem_variant=int(args.mem_variant))
                elif args.fork_tail in ("gate_q", "gate_est2"):
                    # Δ^{RE|π}: prefix decisions forced for k ≤ n, gate decides
                    # for k > n.
                    forced_F = {k: (1 if k in cache_F else 0) for k in range(n + 1)}
                    forced_C = {k: (1 if k in cache_C else 0) for k in range(n + 1)}
                    gate_kind = "q" if args.fork_tail == "gate_q" else "est2"
                    z_F = _run_policy_branch(
                        pipe, prompt, seed, args, forced=forced_F,
                        gate_kind=gate_kind,
                        beta=est2_beta if gate_kind == "est2" else None,
                        threshold=float(args.fork_tail_threshold),
                        g_max=int(args.g_max),
                        q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                        mem_variant=int(args.mem_variant))
                    z_C = _run_policy_branch(
                        pipe, prompt, seed, args, forced=forced_C,
                        gate_kind=gate_kind,
                        beta=est2_beta if gate_kind == "est2" else None,
                        threshold=float(args.fork_tail_threshold),
                        g_max=int(args.g_max),
                        q_abs=q_abs, s_cal=s_cal, a_cal=a_cal,
                        mem_variant=int(args.mem_variant))
                else:
                    raise ValueError(f"unknown fork_tail: {args.fork_tail!r}")
                delta = float((z_C.float() - z_F.float()).norm())
                f = feats[n]
                rows.append({
                    "prompt_idx": global_idx, "policy": pname, "step": n,
                    "gap": f["gap"], "last_refresh": n - f["gap"],
                    "q": f["q"], "p_acc": f["p_acc"],
                    "c_stale": f["c_stale"], "c_traj": f["c_traj"],
                    "c_mem": f["c_mem"], "psi_drift": f["psi_drift"],
                    "native_gate_signal": f["p_acc"],
                    "n_cache_before": len(cache_F),
                    "delta_latent_l2_full_after": delta,
                })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (prompt_dir / "marginal_rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False))

        # Per-step trace (one entry per policy per step) for offline
        # C-state reconstruction in AV-B and follow-ups. The marginal_rows
        # file only has fork-point snapshots; this trace lets analysis
        # re-derive C-state at any (γ, η, ρ, accumulate_unit) without
        # re-running the GPU probe. Recorded fields are minimal: step, u,
        # psi_drift — the per-step features the tracker consumed.
        traces = {}
        # Re-run the prefix runs once more was already done above; the
        # `rec` lists for each policy are still in scope in the policy
        # loop. We need to capture them as we go. Easier: save them in
        # `policy_recs` (built above) and dump now.
        if policy_recs:
            def _trace_row(r: Dict) -> Dict:
                row = {"step": int(r["step"]),
                       "u": int(r["u"]),
                       "psi_drift": float(r["psi_drift"])}
                if "psi_proj" in r:
                    row["psi_proj"] = [float(x) for x in r["psi_proj"]]
                return row
            traces = {p: [_trace_row(r) for r in rec_list]
                      for p, rec_list in policy_recs.items()}
            (prompt_dir / "per_step_trace.json").write_text(
                json.dumps(traces, indent=2, ensure_ascii=False))

        wall = time.perf_counter() - t_p
        manifest_path.write_text(json.dumps({
            "experiment": "marginal_risk_probe", "prompt_idx": global_idx,
            "prompt": prompt, "seed": int(seed), "num_steps": N,
            "policies": [p[0] for p in policies], "forks": forks,
            "n_rows": len(rows),
            "trace_policies": list(traces.keys()) if traces else [],
            "wall_seconds": wall, "complete": True,
        }, indent=2, ensure_ascii=False))
        per_prompt.append({"global_idx": global_idx, "wall_seconds": wall})
        print(f"[shard {args.shard_idx}] prompt {global_idx} done {wall:.1f}s "
              f"({local_idx + 1}/{len(shard_prompts)}) rows={len(rows)}", flush=True)

    process_end = time.perf_counter()
    timing_path = args.output_dir / f"timing_shard{args.shard_idx}of{args.shard_count}.json"
    timing_path.write_text(json.dumps({
        "experiment": "marginal_risk_probe", "num_steps": N,
        "policies": [p[0] for p in policies], "forks": forks,
        "shard_idx": int(args.shard_idx), "shard_count": int(args.shard_count),
        "base_seed": int(args.seed), "model_id": args.model_id,
        "model_load_s": float(model_load_end - t0),
        "wallclock_total_s": float(process_end - t0),
        "n_prompts_in_shard": len(shard_prompts), "per_prompt": per_prompt,
    }, indent=2, ensure_ascii=False))
    print(f"[shard {args.shard_idx}] wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
