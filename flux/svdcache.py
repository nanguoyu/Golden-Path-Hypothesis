"""Native coarse SVD-Cache on diffusers `FluxTransformer2DModel`.

This is a coarse whole-transformer-residual analogue of SVD-Cache
(arXiv:2601.07396).  It uses SVD-Cache's own interval schedule and cached-step
formula instead of SeaCache/TeaCache threshold gates or fixed payload replay.

At full steps, the transformer residual R_t = h_out - h_in is decomposed into
principal and residual subspaces using a right-basis V_k:

    R_P = R_t V_k V_k^T
    R_R = R_t - R_P

The principal residual is smoothed with EMA, and cached steps use:

    R_hat = EMA(R_P) + R_R

This file intentionally does not modify locked baseline implementations.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

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

from lib.gates import IntervalGate

logger = logging.get_logger(__name__)

EPS = 1e-12


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor.reshape(1, -1)
    return tensor.reshape(-1, int(tensor.shape[-1]))


def _norm(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def _seed_for_shape(shape: Tuple[int, ...], max_rank: int, energy: float) -> int:
    payload = f"{','.join(str(int(x)) for x in shape)}|{int(max_rank)}|{float(energy):.8f}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _basis_from_residual(
    residual: torch.Tensor,
    *,
    energy_threshold: float,
    max_rank: int,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    start = time.perf_counter()
    fields: Dict[str, Any] = {
        "svdcache_basis_created": False,
        "svdcache_rank": None,
        "svdcache_energy_keep_rel": None,
        "svdcache_singular_values": None,
        "svdcache_basis_shape": None,
        "svdcache_basis_elapsed_ms": None,
        "svdcache_fallback_reason": None,
    }
    mat = _matrix_view(residual.detach()).to(torch.float32)
    if mat.numel() == 0 or mat.shape[0] <= 0 or mat.shape[1] <= 0:
        fields["svdcache_fallback_reason"] = "empty_matrix"
        return None, fields
    q = min(int(max_rank), int(mat.shape[0]), int(mat.shape[1]))
    if q <= 0:
        fields["svdcache_fallback_reason"] = "rank_cap_zero"
        return None, fields
    total_energy = float(torch.sum(mat * mat).item())
    if total_energy <= 0.0:
        fields["svdcache_fallback_reason"] = "zero_energy"
        return None, fields

    devices = []
    if mat.is_cuda:
        device_index = mat.device.index
        devices = [int(device_index) if device_index is not None else torch.cuda.current_device()]
    seed = _seed_for_shape(tuple(residual.shape), int(max_rank), float(energy_threshold))
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if mat.is_cuda:
                torch.cuda.manual_seed_all(seed)
            _u, s, v = torch.pca_lowrank(mat, q=q, center=False, niter=2)
    except RuntimeError as exc:
        fields["svdcache_fallback_reason"] = f"pca_lowrank_failed:{type(exc).__name__}"
        return None, fields
    if s.numel() == 0 or v.numel() == 0:
        fields["svdcache_fallback_reason"] = "empty_svd"
        return None, fields
    energy = torch.cumsum(s * s, dim=0) / (total_energy + EPS)
    above = torch.nonzero(energy >= float(energy_threshold), as_tuple=False)
    rank = int(above[0].item() + 1) if above.numel() else int(min(q, s.numel()))
    rank = max(1, min(rank, int(v.shape[1])))
    basis = v[:, :rank].contiguous()
    if not _finite(basis):
        fields["svdcache_fallback_reason"] = "basis_not_finite"
        return None, fields

    fields.update({
        "svdcache_basis_created": True,
        "svdcache_rank": int(rank),
        "svdcache_energy_keep_rel": float(energy[rank - 1].item()),
        "svdcache_singular_values": ",".join(
            f"{float(x):.6g}" for x in s[: min(8, s.numel())].detach().cpu().tolist()
        ),
        "svdcache_basis_shape": f"{int(basis.shape[0])}x{int(basis.shape[1])}",
        "svdcache_basis_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
        "svdcache_fallback_reason": None,
    })
    return basis.to(device=residual.device), fields


def _project(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    mat = _matrix_view(tensor.detach()).to(torch.float32)
    basis_f = basis.detach().to(device=tensor.device, dtype=torch.float32)
    projected = (mat @ basis_f) @ basis_f.T
    return projected.reshape_as(tensor).to(device=tensor.device, dtype=tensor.dtype)


def _empty_step_fields() -> Dict[str, Any]:
    return {
        "svdcache_enabled": True,
        "svdcache_available": False,
        "svdcache_fallback": False,
        "svdcache_fallback_reason": None,
        "svdcache_rank": None,
        "svdcache_energy_keep_rel": None,
        "svdcache_basis_shape": None,
        "svdcache_basis_policy": None,
        "svdcache_basis_created": False,
        "svdcache_basis_elapsed_ms": None,
        "svdcache_beta": None,
        "svdcache_principal_norm": None,
        "svdcache_residual_norm": None,
        "svdcache_ema_norm": None,
        "svdcache_payload_norm": None,
        "svdcache_payload_delta_from_reuse_norm": None,
    }


def _update_full_state(
    self,
    *,
    residual: torch.Tensor,
    step: int,
) -> Dict[str, Any]:
    fields = _empty_step_fields()
    fields.update({
        "svdcache_basis_policy": str(getattr(self, "_svdcache_basis_policy", "first_full")),
        "svdcache_beta": float(getattr(self, "_svdcache_beta", 0.9)),
    })
    basis = getattr(self, "_svdcache_basis", None)
    if basis is None:
        basis, basis_fields = _basis_from_residual(
            residual,
            energy_threshold=float(getattr(self, "_svdcache_energy", 0.85)),
            max_rank=int(getattr(self, "_svdcache_max_rank", 32)),
        )
        fields.update(basis_fields)
        if basis is None:
            fields["svdcache_fallback"] = True
            return fields
        self._svdcache_basis = basis.detach()
        self._svdcache_rank = fields["svdcache_rank"]
        self._svdcache_energy_keep_rel = fields["svdcache_energy_keep_rel"]
        self._svdcache_singular_values = fields["svdcache_singular_values"]
        self._svdcache_basis_shape = fields["svdcache_basis_shape"]
    else:
        fields.update({
            "svdcache_rank": getattr(self, "_svdcache_rank", None),
            "svdcache_energy_keep_rel": getattr(self, "_svdcache_energy_keep_rel", None),
            "svdcache_singular_values": getattr(self, "_svdcache_singular_values", None),
            "svdcache_basis_shape": getattr(self, "_svdcache_basis_shape", None),
        })

    principal = _project(residual, basis)
    residual_part = (residual.detach() - principal.detach()).to(dtype=residual.dtype, device=residual.device)
    prev_ema = getattr(self, "_svdcache_principal_ema", None)
    beta = float(getattr(self, "_svdcache_beta", 0.9))
    if prev_ema is None or tuple(prev_ema.shape) != tuple(principal.shape):
        ema = principal.detach()
    else:
        ema = (
            beta * prev_ema.to(device=principal.device, dtype=principal.dtype)
            + (1.0 - beta) * principal
        ).detach()

    self._svdcache_principal_ema = ema
    self._svdcache_residual_anchor = residual_part.detach()
    self._svdcache_reuse_residual = residual.detach()
    self._svdcache_anchor_step = int(step)
    self._svdcache_updates = int(getattr(self, "_svdcache_updates", 0)) + 1

    fields.update({
        "svdcache_available": True,
        "svdcache_principal_norm": _norm(principal),
        "svdcache_residual_norm": _norm(residual_part),
        "svdcache_ema_norm": _norm(ema),
        "svdcache_payload_norm": _norm(ema + residual_part),
        "svdcache_payload_delta_from_reuse_norm": _norm(
            (ema + residual_part).detach().to(torch.float32) - residual.detach().to(torch.float32)
        ),
    })
    return fields


def _cached_payload(self) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    fields = _empty_step_fields()
    fields.update({
        "svdcache_basis_policy": str(getattr(self, "_svdcache_basis_policy", "first_full")),
        "svdcache_beta": float(getattr(self, "_svdcache_beta", 0.9)),
        "svdcache_rank": getattr(self, "_svdcache_rank", None),
        "svdcache_energy_keep_rel": getattr(self, "_svdcache_energy_keep_rel", None),
        "svdcache_basis_shape": getattr(self, "_svdcache_basis_shape", None),
    })
    ema = getattr(self, "_svdcache_principal_ema", None)
    residual_anchor = getattr(self, "_svdcache_residual_anchor", None)
    reuse_residual = getattr(self, "_svdcache_reuse_residual", None)
    if ema is None or residual_anchor is None or reuse_residual is None:
        fields.update({
            "svdcache_available": False,
            "svdcache_fallback": True,
            "svdcache_fallback_reason": "state_not_ready",
        })
        return None, fields
    if tuple(ema.shape) != tuple(residual_anchor.shape) or tuple(ema.shape) != tuple(reuse_residual.shape):
        fields.update({
            "svdcache_available": False,
            "svdcache_fallback": True,
            "svdcache_fallback_reason": "state_shape_mismatch",
        })
        return None, fields
    payload = ema.to(dtype=reuse_residual.dtype, device=reuse_residual.device) + residual_anchor.to(
        dtype=reuse_residual.dtype,
        device=reuse_residual.device,
    )
    if not _finite(payload):
        fields.update({
            "svdcache_available": False,
            "svdcache_fallback": True,
            "svdcache_fallback_reason": "payload_not_finite",
        })
        return None, fields
    fields.update({
        "svdcache_available": True,
        "svdcache_principal_norm": _norm(ema),
        "svdcache_residual_norm": _norm(residual_anchor),
        "svdcache_ema_norm": _norm(ema),
        "svdcache_payload_norm": _norm(payload),
        "svdcache_payload_delta_from_reuse_norm": _norm(
            payload.detach().to(torch.float32) - reuse_residual.detach().to(torch.float32)
        ),
    })
    return payload, fields


def _svdcache_forward(
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
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
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

    gate: IntervalGate = self._svdcache_gate
    step = int(gate.cnt)
    should_skip = gate.decide()
    force_full_reason = None
    if not should_skip:
        if step < int(gate.first_enhance):
            force_full_reason = "first_enhance"
        elif gate.num_steps is not None and step >= int(gate.num_steps) - 1:
            force_full_reason = "last_step"
        elif step == gate.last_activated:
            force_full_reason = "interval"
    fields = _empty_step_fields()

    if should_skip:
        payload, fields = _cached_payload(self)
        if payload is None:
            should_skip = False
            force_full_reason = fields.get("svdcache_fallback_reason") or "fallback"
        else:
            hidden_states = hidden_states + payload.to(dtype=hidden_states.dtype, device=hidden_states.device)

    if not should_skip:
        ori_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                def _ckpt2(module):
                    def _fwd(hs, ehs, temb_, ire):
                        return module(
                            hidden_states=hs,
                            encoder_hidden_states=ehs,
                            temb=temb_,
                            image_rotary_emb=ire,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return _fwd

                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt2(block), hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb, **ckpt_kwargs,
                )
            else:
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

        residual = hidden_states - ori_hidden_states
        fields = _update_full_state(self, residual=residual, step=step)

    if hasattr(self, "svdcache_decisions"):
        self.svdcache_decisions.append({
            "step": int(step),
            "u": int(should_skip),
            "interval": int(getattr(self, "_svdcache_interval", 0)),
            "first_enhance": int(gate.first_enhance),
            "num_steps": int(gate.num_steps) if gate.num_steps is not None else None,
            "force_full_reason": force_full_reason,
            "energy_threshold": float(getattr(self, "_svdcache_energy", 0.85)),
            "max_rank": int(getattr(self, "_svdcache_max_rank", 32)),
            **fields,
        })

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def install(
    pipe,
    *,
    interval: int = 7,
    first_enhance: int = 3,
    num_steps: int,
    energy: float = 0.85,
    max_rank: int = 32,
    beta: float = 0.9,
    basis_policy: str = "first_full",
) -> Callable[[], None]:
    if str(basis_policy) != "first_full":
        raise ValueError("native coarse SVDCache currently supports only basis_policy='first_full'")
    orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = _svdcache_forward

    tr = pipe.transformer
    tr._svdcache_gate = IntervalGate(
        interval=int(interval),
        first_enhance=int(first_enhance),
        num_steps=int(num_steps),
    )
    tr._svdcache_interval = int(interval)
    tr._svdcache_energy = float(energy)
    tr._svdcache_max_rank = int(max_rank)
    tr._svdcache_beta = float(beta)
    tr._svdcache_basis_policy = str(basis_policy)
    tr._svdcache_basis = None
    tr._svdcache_rank = None
    tr._svdcache_energy_keep_rel = None
    tr._svdcache_singular_values = None
    tr._svdcache_basis_shape = None
    tr._svdcache_principal_ema = None
    tr._svdcache_residual_anchor = None
    tr._svdcache_reuse_residual = None
    tr._svdcache_anchor_step = None
    tr._svdcache_updates = 0
    tr.svdcache_decisions = []

    _torn_down = {"done": False}

    def teardown() -> None:
        if _torn_down["done"]:
            return
        FluxTransformer2DModel.forward = orig_forward
        for attr in (
            "_svdcache_gate",
            "_svdcache_interval",
            "_svdcache_energy",
            "_svdcache_max_rank",
            "_svdcache_beta",
            "_svdcache_basis_policy",
            "_svdcache_basis",
            "_svdcache_rank",
            "_svdcache_energy_keep_rel",
            "_svdcache_singular_values",
            "_svdcache_basis_shape",
            "_svdcache_principal_ema",
            "_svdcache_residual_anchor",
            "_svdcache_reuse_residual",
            "_svdcache_anchor_step",
            "_svdcache_updates",
            "svdcache_decisions",
        ):
            if hasattr(tr, attr):
                try:
                    delattr(tr, attr)
                except AttributeError:
                    pass
        _torn_down["done"] = True

    return teardown


def reset_per_image_state(pipe) -> None:
    tr = pipe.transformer
    tr._svdcache_gate.reset()
    tr._svdcache_basis = None
    tr._svdcache_rank = None
    tr._svdcache_energy_keep_rel = None
    tr._svdcache_singular_values = None
    tr._svdcache_basis_shape = None
    tr._svdcache_principal_ema = None
    tr._svdcache_residual_anchor = None
    tr._svdcache_reuse_residual = None
    tr._svdcache_anchor_step = None
    tr._svdcache_updates = 0
    tr.svdcache_decisions = []
