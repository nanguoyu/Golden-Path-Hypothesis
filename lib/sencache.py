"""Shared utilities for SenCache-style sensitivity scores.

SenCache's decision statistic is a first-order local output-change bound:

    Lambda = J_x(anchor_t) * ||x_t - x_anchor||_2
           + J_t(anchor_t) * |t - t_anchor|.

The sensitivity table is deliberately loaded from a frozen calibration artifact.
Online collectors and deployment gates must read this table; they must not
estimate Jacobians from current fork labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SenCacheLookup:
    index: int
    timestep: float
    j_x_norm: float
    j_t_norm: float
    abs_timestep_error: float


@dataclass(frozen=True)
class SenCacheSensitivityTable:
    path: str
    sha256: str
    timesteps: np.ndarray
    j_x_norm: np.ndarray
    j_t_norm: np.ndarray
    metadata: Dict[str, Any]

    def lookup(self, timestep: float) -> SenCacheLookup:
        if self.timesteps.size == 0:
            raise ValueError("empty SenCache sensitivity table")
        t = float(timestep)
        idx = int(np.argmin(np.abs(self.timesteps.astype(float) - t)))
        return SenCacheLookup(
            index=idx,
            timestep=float(self.timesteps[idx]),
            j_x_norm=float(self.j_x_norm[idx]),
            j_t_norm=float(self.j_t_norm[idx]),
            abs_timestep_error=float(abs(float(self.timesteps[idx]) - t)),
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_metadata(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            obj = obj.item()
        else:
            obj = obj.tolist()
    if isinstance(obj, bytes):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": obj}
    if isinstance(obj, dict):
        return dict(obj)
    return {"value": obj}


def load_sensitivity_table(path: str | Path) -> SenCacheSensitivityTable:
    """Load a frozen, backbone-specific SenCache sensitivity table.

    Accepted keys:
      - ``timesteps``: scheduler timesteps in the same units used by the
        consuming backbone adapter. FLUX tables use the transformer's
        ``* 1000`` units; Qwen-Image tables use its normalized forward input.
      - ``J_x_norm`` or ``J_z_norm``: latent/input sensitivity norm.
      - ``J_t_norm``: timestep sensitivity norm.

    Extra metadata is preserved when the npz contains ``metadata_json`` or
    ``metadata``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SenCache sensitivity table not found: {p}")
    data = np.load(p, allow_pickle=True)
    if "timesteps" not in data:
        raise ValueError(f"SenCache sensitivity table missing 'timesteps': {p}")
    x_key = "J_x_norm" if "J_x_norm" in data else "J_z_norm" if "J_z_norm" in data else None
    if x_key is None:
        raise ValueError(f"SenCache sensitivity table missing 'J_x_norm'/'J_z_norm': {p}")
    if "J_t_norm" not in data:
        raise ValueError(f"SenCache sensitivity table missing 'J_t_norm': {p}")
    timesteps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
    j_x = np.asarray(data[x_key], dtype=float).reshape(-1)
    j_t = np.asarray(data["J_t_norm"], dtype=float).reshape(-1)
    if not (timesteps.shape == j_x.shape == j_t.shape):
        raise ValueError(
            f"SenCache sensitivity table shape mismatch: timesteps={timesteps.shape} "
            f"{x_key}={j_x.shape} J_t_norm={j_t.shape}"
        )
    for name, arr in (("timesteps", timesteps), (x_key, j_x), ("J_t_norm", j_t)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"SenCache sensitivity table has non-finite {name}: {p}")
    metadata: Dict[str, Any] = {}
    if "metadata_json" in data:
        metadata.update(_read_metadata(data["metadata_json"]))
    if "metadata" in data:
        metadata.update(_read_metadata(data["metadata"]))
    metadata.setdefault("latent_sensitivity_key", x_key)
    return SenCacheSensitivityTable(
        path=str(p),
        sha256=_sha256(p),
        timesteps=timesteps,
        j_x_norm=j_x,
        j_t_norm=j_t,
        metadata=metadata,
    )


def threshold_scale_from_latent(latent: torch.Tensor, value: str | float | int | None) -> float:
    if value is None or str(value).strip().lower() == "auto":
        return float(math.sqrt(max(int(latent.numel()), 1)))
    return float(value)


def online_fields(
    *,
    table: Optional[SenCacheSensitivityTable],
    current_latent: torch.Tensor,
    current_timestep: float,
    anchor_latent: Optional[torch.Tensor],
    anchor_timestep: Optional[float],
    anchor_step: Optional[int],
    prefix: str = "online_sencache",
) -> Dict[str, Any]:
    """Compute decision-time SenCache fields for E0/deployment rows."""
    out: Dict[str, Any] = {
        f"{prefix}_anchor_present_pre": bool(anchor_latent is not None and anchor_timestep is not None),
        f"{prefix}_anchor_step_pre": None if anchor_step is None else int(anchor_step),
        f"{prefix}_anchor_timestep_pre": None if anchor_timestep is None else float(anchor_timestep),
        f"{prefix}_table_index_pre": None,
        f"{prefix}_table_timestep_pre": None,
        f"{prefix}_table_abs_timestep_error_pre": None,
        f"{prefix}_j_x_norm_pre": None,
        f"{prefix}_j_t_norm_pre": None,
        f"{prefix}_delta_latent_norm_pre": None,
        f"{prefix}_delta_t_abs_pre": None,
        f"{prefix}_latent_term_pre": None,
        f"{prefix}_timestep_term_pre": None,
        f"{prefix}_score_pre": None,
        f"{prefix}_score_log1p_pre": None,
    }
    if table is None or anchor_latent is None or anchor_timestep is None:
        return out
    lookup = table.lookup(float(anchor_timestep))
    cur = current_latent.detach().to(torch.float32)
    anc = anchor_latent.detach().to(device=cur.device, dtype=torch.float32)
    if cur.shape != anc.shape:
        return out
    delta_x = float(torch.linalg.vector_norm(cur - anc).item())
    delta_t = float(abs(float(current_timestep) - float(anchor_timestep)))
    latent_term = float(lookup.j_x_norm * delta_x)
    timestep_term = float(lookup.j_t_norm * delta_t)
    score = float(latent_term + timestep_term)
    out.update({
        f"{prefix}_table_index_pre": int(lookup.index),
        f"{prefix}_table_timestep_pre": float(lookup.timestep),
        f"{prefix}_table_abs_timestep_error_pre": float(lookup.abs_timestep_error),
        f"{prefix}_j_x_norm_pre": float(lookup.j_x_norm),
        f"{prefix}_j_t_norm_pre": float(lookup.j_t_norm),
        f"{prefix}_delta_latent_norm_pre": delta_x,
        f"{prefix}_delta_t_abs_pre": delta_t,
        f"{prefix}_latent_term_pre": latent_term,
        f"{prefix}_timestep_term_pre": timestep_term,
        f"{prefix}_score_pre": score,
        f"{prefix}_score_log1p_pre": float(math.log1p(max(score, 0.0))),
    })
    return out
