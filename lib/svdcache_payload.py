"""SVD-Cache-style payload helpers for research runners.

This module implements a coarse residual analogue of SVD-Cache:
forecast/update the principal channel subspace and stabilize the residual
subspace by reuse.  It is intentionally separate from locked baseline code.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import torch

EPS = 1e-12

SVD_CACHE_PAYLOAD_SPECS: Dict[str, Dict[str, Any]] = {
    "svdcache_taylor_o1_e85_r16": {
        "raw_mode": "taylor_o1",
        "control": "svdcache_delta",
        "energy": 0.85,
        "max_rank": 16,
        "beta": None,
    },
    "svdcache_taylor_o1_e90_r16": {
        "raw_mode": "taylor_o1",
        "control": "svdcache_delta",
        "energy": 0.90,
        "max_rank": 16,
        "beta": None,
    },
    "svdcache_taylor_o1_e95_r16": {
        "raw_mode": "taylor_o1",
        "control": "svdcache_delta",
        "energy": 0.95,
        "max_rank": 16,
        "beta": None,
    },
    "svdcache_taylor_o1_e95_r32": {
        "raw_mode": "taylor_o1",
        "control": "svdcache_delta",
        "energy": 0.95,
        "max_rank": 32,
        "beta": None,
    },
    "svdcache_ensemble_e85_r16": {
        "raw_mode": "ensemble_mean",
        "control": "svdcache_delta",
        "energy": 0.85,
        "max_rank": 16,
        "beta": None,
    },
    "svdcache_ensemble_e95_r16": {
        "raw_mode": "ensemble_mean",
        "control": "svdcache_delta",
        "energy": 0.95,
        "max_rank": 16,
        "beta": None,
    },
    "svdcache_ema_e85_b0.9_r16": {
        "raw_mode": "reuse",
        "control": "svdcache_ema",
        "energy": 0.85,
        "max_rank": 16,
        "beta": 0.9,
    },
    "svdcache_ema_e95_b0.9_r16": {
        "raw_mode": "reuse",
        "control": "svdcache_ema",
        "energy": 0.95,
        "max_rank": 16,
        "beta": 0.9,
    },
}

SVD_CACHE_PAYLOAD_MODES = tuple(SVD_CACHE_PAYLOAD_SPECS.keys())
SVD_CACHE_CONTROLS = {"svdcache_delta", "svdcache_ema"}


def init_state() -> Dict[str, Any]:
    return {
        "basis": None,
        "rank": None,
        "energy_keep": None,
        "singular_values": None,
        "principal_ema": None,
        "residual_anchor": None,
        "anchor_step": None,
        "updates": 0,
        "basis_shape": None,
        "basis_source": None,
    }


def empty_fields() -> Dict[str, Any]:
    return {
        "payload_svd_enabled": False,
        "payload_svd_raw_mode": None,
        "payload_svd_control": None,
        "payload_svd_energy_threshold": None,
        "payload_svd_max_rank": None,
        "payload_svd_beta": None,
        "payload_svd_basis_ready_pre": None,
        "payload_svd_update_count_pre": None,
        "payload_svd_anchor_step_pre": None,
        "payload_svd_rank_used": None,
        "payload_svd_energy_keep_rel": None,
        "payload_svd_singular_values": None,
        "payload_svd_basis_shape": None,
        "payload_svd_basis_source": None,
        "payload_svd_projected_delta_norm": None,
        "payload_svd_residual_delta_norm": None,
        "payload_svd_delta_energy_keep_rel": None,
        "payload_svd_principal_norm": None,
        "payload_svd_residual_norm": None,
        "payload_svd_ema_norm": None,
        "payload_svd_output_delta_from_reuse_norm": None,
        "payload_svd_basis_created": False,
        "payload_svd_update_applied": False,
        "payload_svd_elapsed_ms": None,
        "payload_svd_fallback_reason": None,
    }


def spec_for_mode(mode: str) -> Optional[Dict[str, Any]]:
    spec = SVD_CACHE_PAYLOAD_SPECS.get(str(mode))
    return None if spec is None else dict(spec)


def is_mode(mode: str) -> bool:
    return str(mode) in SVD_CACHE_PAYLOAD_SPECS


def payload_spec(mode: str) -> Tuple[str, str]:
    spec = spec_for_mode(mode)
    if spec is None:
        raise KeyError(mode)
    return str(spec["raw_mode"]), str(spec["control"])


def _norm(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().to(torch.float32).norm().item())


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor.reshape(1, -1)
    return tensor.reshape(-1, int(tensor.shape[-1]))


def _same_shape(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    return a is not None and b is not None and tuple(a.shape) == tuple(b.shape)


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def _seed_for_shape(shape: Tuple[int, ...], max_rank: int, energy: float) -> int:
    payload = f"{','.join(str(int(x)) for x in shape)}|{int(max_rank)}|{float(energy):.8f}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def _basis_from_feature(
    residual: torch.Tensor,
    *,
    energy_threshold: float,
    max_rank: int,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    fields = empty_fields()
    mat = _matrix_view(residual.detach()).to(torch.float32)
    if mat.numel() == 0 or mat.shape[0] <= 0 or mat.shape[1] <= 0:
        fields["payload_svd_fallback_reason"] = "empty_matrix"
        return None, fields
    q = min(int(max_rank), int(mat.shape[0]), int(mat.shape[1]))
    if q <= 0:
        fields["payload_svd_fallback_reason"] = "rank_cap_zero"
        return None, fields
    total_energy = float(torch.sum(mat * mat).item())
    if total_energy <= 0.0:
        fields["payload_svd_fallback_reason"] = "zero_energy"
        return None, fields

    if mat.is_cuda:
        device_index = mat.device.index
        devices = [int(device_index) if device_index is not None else torch.cuda.current_device()]
    else:
        devices = []
    seed = _seed_for_shape(tuple(residual.shape), int(max_rank), float(energy_threshold))
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if mat.is_cuda:
                torch.cuda.manual_seed_all(seed)
            _u, s, v = torch.pca_lowrank(mat, q=q, center=False, niter=2)
    except RuntimeError as exc:
        fields["payload_svd_fallback_reason"] = f"pca_lowrank_failed:{type(exc).__name__}"
        return None, fields

    if s.numel() == 0 or v.numel() == 0:
        fields["payload_svd_fallback_reason"] = "empty_svd"
        return None, fields
    energy = torch.cumsum(s * s, dim=0) / (total_energy + EPS)
    above = torch.nonzero(energy >= float(energy_threshold), as_tuple=False)
    rank = int(above[0].item() + 1) if above.numel() else int(min(q, s.numel()))
    rank = max(1, min(rank, int(v.shape[1])))
    basis = v[:, :rank].contiguous()
    if not _finite_tensor(basis):
        fields["payload_svd_fallback_reason"] = "basis_not_finite"
        return None, fields

    fields.update({
        "payload_svd_rank_used": int(rank),
        "payload_svd_energy_keep_rel": float(energy[rank - 1].item()),
        "payload_svd_singular_values": ",".join(
            f"{float(x):.6g}" for x in s[: min(8, s.numel())].detach().cpu().tolist()
        ),
        "payload_svd_basis_shape": f"{int(basis.shape[0])}x{int(basis.shape[1])}",
        "payload_svd_fallback_reason": None,
    })
    return basis.to(device=residual.device), fields


def _project_with_basis(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    mat = _matrix_view(tensor.detach()).to(torch.float32)
    basis_f = basis.detach().to(device=tensor.device, dtype=torch.float32)
    projected = (mat @ basis_f) @ basis_f.T
    return projected.reshape_as(tensor).to(device=tensor.device, dtype=tensor.dtype)


def update_on_full(
    state: Optional[Dict[str, Any]],
    *,
    residual: torch.Tensor,
    step: int,
    mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    spec = spec_for_mode(mode)
    st = init_state() if state is None else dict(state)
    fields = empty_fields()
    if spec is None:
        return st, fields

    start = time.perf_counter()
    fields.update({
        "payload_svd_enabled": True,
        "payload_svd_raw_mode": str(spec["raw_mode"]),
        "payload_svd_control": str(spec["control"]),
        "payload_svd_energy_threshold": float(spec["energy"]),
        "payload_svd_max_rank": int(spec["max_rank"]),
        "payload_svd_beta": spec["beta"],
        "payload_svd_basis_ready_pre": st.get("basis") is not None,
        "payload_svd_update_count_pre": int(st.get("updates") or 0),
        "payload_svd_anchor_step_pre": st.get("anchor_step"),
    })

    basis = st.get("basis")
    basis_created = False
    if basis is None:
        basis, basis_fields = _basis_from_feature(
            residual,
            energy_threshold=float(spec["energy"]),
            max_rank=int(spec["max_rank"]),
        )
        fields.update(basis_fields)
        if basis is None:
            fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
            return st, fields
        basis_created = True
        st["basis"] = basis.detach()
        st["rank"] = fields["payload_svd_rank_used"]
        st["energy_keep"] = fields["payload_svd_energy_keep_rel"]
        st["singular_values"] = fields["payload_svd_singular_values"]
        st["basis_shape"] = fields["payload_svd_basis_shape"]
        st["basis_source"] = "first_full_residual"
    else:
        fields.update({
            "payload_svd_rank_used": st.get("rank"),
            "payload_svd_energy_keep_rel": st.get("energy_keep"),
            "payload_svd_singular_values": st.get("singular_values"),
            "payload_svd_basis_shape": st.get("basis_shape"),
            "payload_svd_fallback_reason": None,
        })

    principal = _project_with_basis(residual, basis)
    residual_part = (residual.detach() - principal.detach()).to(dtype=residual.dtype, device=residual.device)
    prev_ema = st.get("principal_ema")
    beta = 0.0 if spec["beta"] is None else float(spec["beta"])
    if prev_ema is None or not _same_shape(prev_ema, principal):
        ema = principal.detach()
    else:
        ema = (float(beta) * prev_ema.to(device=principal.device, dtype=principal.dtype)
               + (1.0 - float(beta)) * principal).detach()

    st["principal_ema"] = ema
    st["residual_anchor"] = residual_part.detach()
    st["anchor_step"] = int(step)
    st["updates"] = int(st.get("updates") or 0) + 1

    fields.update({
        "payload_svd_basis_created": bool(basis_created),
        "payload_svd_update_applied": True,
        "payload_svd_basis_source": st.get("basis_source"),
        "payload_svd_principal_norm": _norm(principal),
        "payload_svd_residual_norm": _norm(residual_part),
        "payload_svd_ema_norm": _norm(ema),
        "payload_svd_elapsed_ms": float((time.perf_counter() - start) * 1000.0),
    })
    return st, fields


def payload(
    state: Optional[Dict[str, Any]],
    *,
    mode: str,
    reuse: torch.Tensor,
    forecast: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    spec = spec_for_mode(mode)
    fields = empty_fields()
    if spec is None:
        return None, fields

    start = time.perf_counter()
    st = init_state() if state is None else state
    basis = st.get("basis")
    fields.update({
        "payload_svd_enabled": True,
        "payload_svd_raw_mode": str(spec["raw_mode"]),
        "payload_svd_control": str(spec["control"]),
        "payload_svd_energy_threshold": float(spec["energy"]),
        "payload_svd_max_rank": int(spec["max_rank"]),
        "payload_svd_beta": spec["beta"],
        "payload_svd_basis_ready_pre": basis is not None,
        "payload_svd_update_count_pre": int(st.get("updates") or 0),
        "payload_svd_anchor_step_pre": st.get("anchor_step"),
        "payload_svd_rank_used": st.get("rank"),
        "payload_svd_energy_keep_rel": st.get("energy_keep"),
        "payload_svd_singular_values": st.get("singular_values"),
        "payload_svd_basis_shape": st.get("basis_shape"),
        "payload_svd_basis_source": st.get("basis_source"),
    })
    if basis is None:
        fields["payload_svd_fallback_reason"] = "basis_not_ready"
        fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return None, fields

    control = str(spec["control"])
    if control == "svdcache_delta":
        if forecast is None or tuple(forecast.shape) != tuple(reuse.shape):
            fields["payload_svd_fallback_reason"] = "forecast_unavailable"
            fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
            return None, fields
        delta = forecast.detach().to(dtype=reuse.dtype, device=reuse.device) - reuse.detach()
        projected_delta = _project_with_basis(delta, basis)
        residual_delta = delta - projected_delta
        chosen = reuse + projected_delta.to(dtype=reuse.dtype, device=reuse.device)
        delta_norm = float(delta.detach().to(torch.float32).norm().item())
        projected_norm = float(projected_delta.detach().to(torch.float32).norm().item())
        fields.update({
            "payload_svd_projected_delta_norm": projected_norm,
            "payload_svd_residual_delta_norm": _norm(residual_delta),
            "payload_svd_delta_energy_keep_rel": float((projected_norm * projected_norm) / (delta_norm * delta_norm + EPS)),
            "payload_svd_output_delta_from_reuse_norm": _norm(chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)),
            "payload_svd_fallback_reason": None,
        })
    elif control == "svdcache_ema":
        ema = st.get("principal_ema")
        residual_anchor = st.get("residual_anchor")
        if not _same_shape(ema, reuse) or not _same_shape(residual_anchor, reuse):
            fields["payload_svd_fallback_reason"] = "ema_state_unavailable"
            fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
            return None, fields
        chosen = (
            ema.to(dtype=reuse.dtype, device=reuse.device)
            + residual_anchor.to(dtype=reuse.dtype, device=reuse.device)
        )
        fields.update({
            "payload_svd_ema_norm": _norm(ema),
            "payload_svd_residual_norm": _norm(residual_anchor),
            "payload_svd_output_delta_from_reuse_norm": _norm(chosen.detach().to(torch.float32) - reuse.detach().to(torch.float32)),
            "payload_svd_fallback_reason": None,
        })
    else:
        fields["payload_svd_fallback_reason"] = f"unknown_control:{control}"
        fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return None, fields

    if not _finite_tensor(chosen):
        fields["payload_svd_fallback_reason"] = "chosen_not_finite"
        fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return None, fields

    fields["payload_svd_elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
    return chosen.to(dtype=reuse.dtype, device=reuse.device), fields
